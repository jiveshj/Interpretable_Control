#!/usr/bin/env python
"""
manifold_probe_ltfv6.py

Tests whether dist_nearest_front sits on a smooth low-dimensional manifold
(a "string" in activation space) that a *linear* probe can't see, following
the method in Goodfire's "Manifold Steering Reveals the Shared Geometry of
Neural Network Representation and Behavior" (arXiv:2605.05115):
  1. reduce activations to 64 PCA components (their number, not ours --
     using only 2-3 components for the FIT badly undersells any curve
     structure and was NOT a fair comparison against a full-dim linear
     probe; 3 components are still used, separately, only for the plots),
  2. bin samples by distance, take per-bin centroids (their Mountain-Car
     recipe for a continuous variable: "partition the position range into
     bins and fit a smooth spline through the means"),
  3. fit a cubic spline through the centroids (parameterized by log-distance),
  4. decode held-out points by nearest-point-on-curve lookup (their
     "Pullback Recovery" -- orthogonal closest-point distance to the curve),
  5. compare curve-decode R^2 against a linear Ridge R^2 fit in the SAME
     64-dim PCA subspace (fair, apples-to-apples -- this is what their
     paper's "Pullback vs Linear R^2" table does), as well as against the
     original full-dim Ridge baseline for continuity with
     probe_results_ltfv6_grouped_cv.json.

Only run on the 4 layers that showed any real linear signal under grouped CV
(backbone_transformers_2, backbone_image_encoder_layer3/4,
backbone_lidar_encoder_layer4) -- see probe_results_ltfv6_grouped_cv.json.

Also reports a "string tightness" residual: mean distance from held-out
points to the nearest point on the curve, relative to the overall spread of
the point cloud. Low ratio = points really do hug a 1D curve. High ratio =
no such structure (scattered cloud), which is itself a real finding.

Usage:
    python manifold_probe_ltfv6.py
    python manifold_probe_ltfv6.py --layers backbone_transformers_2
"""

import argparse
import base64
import io
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT   = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
OUT_JSON     = "/jet/home/jjain2/Interpretable_Control/manifold_probe_ltfv6.json"
OUT_HTML     = "/jet/home/jjain2/Interpretable_Control/manifold_probe_ltfv6.html"

SIGNAL_LAYERS = [
    "backbone_transformers_2",
    "backbone_image_encoder_layer3",
    "backbone_image_encoder_layer4",
    "backbone_lidar_encoder_layer4",
]

N_COMPONENTS_FIT = 64  # PCA dims for curve fitting + decoding (matches the paper)
N_COMPONENTS_VIZ = 3   # PCA dims for plotting only
N_CURVE_BINS = 30      # per-bin centroids the spline is fit through -- the
                       # reference mc_world_model/fit_manifold.py uses 100 bins,
                       # but on ~18.7k activations (~187/bin); our train folds
                       # are ~1400 samples, so 30 bins (~46/bin) keeps centroids
                       # from being noise-dominated at our much smaller scale
N_CURVE_GRID = 2000    # dense re-sampling of the fitted spline for decoding
N_SPLITS     = 5
ALPHAS       = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
BIN_EDGES    = [3.0, 5.0, 10.0, 20.0, 40.0]


def log_id_from_path(p):
    p = p.decode() if isinstance(p, bytes) else p
    return p.split("/")[-3]


def load_groups(h5):
    return np.array([log_id_from_path(p) for p in h5["sample_path"][:]])


def fit_curve(pcs_train, t_train, n_bins=N_CURVE_BINS):
    """Fit a smooth spline PC(t) through per-bin centroids, t sorted ascending.

    Returns (spline, t_min, t_max, centroid_pcs, centroid_t, corrs).

    NOTE: an earlier version of this module also computed an "arc-length
    profile" (geodesic arc-length along this spline vs. metric distance,
    plus a chord-monotonicity/closure check) and used it to argue a real
    curved manifold exists with some particular resolution shape. That
    metric was removed: cumulative arc-length is a running sum of positive
    step-lengths, so it is monotonically non-decreasing in t by construction
    -- and since dist_anchor = exp(t) is also monotonic in t, arc-length vs.
    metric distance is close to guaranteed to correlate highly (~0.97)
    regardless of whether the underlying activations encode anything real
    about distance. It's the classic spurious-correlation-from-shared-trend
    trap (two monotonic sequences indexed by the same ordering), not
    evidence of manifold structure. The valid, non-circular metrics are
    still here: residual_ratio and r2_*_curve (both computed out-of-sample
    in evaluate_layer) are what actually tell you whether the curve fit
    generalizes.
    """
    from scipy.interpolate import CubicSpline

    order = np.argsort(t_train)
    chunks = np.array_split(order, n_bins)
    centroid_t  = np.array([t_train[c].mean() for c in chunks if len(c) > 0])
    centroid_pc = np.array([pcs_train[c].mean(axis=0) for c in chunks if len(c) > 0])

    # de-dup identical t values (CubicSpline needs strictly increasing x)
    keep = np.concatenate([[True], np.diff(centroid_t) > 1e-9])
    centroid_t, centroid_pc = centroid_t[keep], centroid_pc[keep]

    corrs = [float(np.corrcoef(centroid_pc[:, i], centroid_t)[0, 1])
             for i in range(min(3, centroid_pc.shape[1]))]

    spline = CubicSpline(centroid_t, centroid_pc, axis=0, bc_type="natural")
    return spline, centroid_t.min(), centroid_t.max(), centroid_pc, centroid_t, corrs


def decode_via_curve(spline, t_min, t_max, pcs_query, n_grid=N_CURVE_GRID):
    """Nearest-point-on-curve decoding: predicted t = t of closest curve sample."""
    grid_t  = np.linspace(t_min, t_max, n_grid)
    grid_pc = spline(grid_t)
    tree = cKDTree(grid_pc)
    dist, idx = tree.query(pcs_query)
    return grid_t[idx], dist


def evaluate_layer(X, y_dist, y_log, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = {
        "r2_log_curve": [], "r2_dist_curve": [],
        "r2_log_linear_pca64": [], "r2_dist_linear_pca64": [],
        "r2_log_linear_fulldim": [], "r2_dist_linear_fulldim": [],
        "residual_ratio": [], "pca_var_explained": [],
    }

    for train_idx, test_idx in gkf.split(X, groups=groups):
        Xtr, Xte = X[train_idx], X[test_idx]
        ylog_tr, ylog_te = y_log[train_idx], y_log[test_idx]
        ydist_te = y_dist[test_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        n_comp = min(N_COMPONENTS_FIT, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
        pca = PCA(n_components=n_comp).fit(Xtr_s)
        pcs_tr_raw, pcs_te_raw = pca.transform(Xtr_s), pca.transform(Xte_s)
        fold_metrics["pca_var_explained"].append(float(pca.explained_variance_ratio_.sum()))

        # causalab's "StandardizeFeaturizer" step: z-score each PC axis using
        # train statistics, so the later Euclidean nearest-point search isn't
        # dominated by PC1 just because it has the most raw variance. This
        # was missing from the first pass and matters a lot for a fair
        # curve-vs-linear comparison in the same subspace.
        pc_scaler = StandardScaler().fit(pcs_tr_raw)
        pcs_tr, pcs_te = pc_scaler.transform(pcs_tr_raw), pc_scaler.transform(pcs_te_raw)

        # -- nonlinear curve decode (parameterized by log-distance), fit in the
        # full 64-dim PCA subspace, not just the 3 dims used for plotting --
        spline, t_min, t_max, _, _, corrs = fit_curve(pcs_tr, ylog_tr)
        pred_log, curve_dist = decode_via_curve(spline, t_min, t_max, pcs_te)
        pred_dist = np.exp(pred_log)

        fold_metrics["r2_log_curve"].append(float(r2_score(ylog_te, pred_log)))
        fold_metrics["r2_dist_curve"].append(float(r2_score(ydist_te, pred_dist)))
        fold_metrics.setdefault("centroid_pc_corr", []).append(corrs)

        # "string tightness": residual perpendicular distance vs overall spread
        spread = float(np.linalg.norm(pcs_te - pcs_te.mean(axis=0), axis=1).mean())
        fold_metrics["residual_ratio"].append(float(curve_dist.mean() / spread) if spread > 0 else float("nan"))

        # -- linear Ridge in the SAME (standardized) 64-dim PCA subspace as the
        # curve: the fair, apples-to-apples comparison ("Pullback vs Linear R^2"
        # in the paper) --
        ridge_pca = RidgeCV(alphas=ALPHAS).fit(pcs_tr, ylog_tr)
        pred_log_pca = ridge_pca.predict(pcs_te)
        fold_metrics["r2_log_linear_pca64"].append(float(r2_score(ylog_te, pred_log_pca)))
        fold_metrics["r2_dist_linear_pca64"].append(float(r2_score(ydist_te, np.exp(pred_log_pca))))

        # -- original full-dimensional linear Ridge baseline, for continuity
        # with probe_results_ltfv6_grouped_cv.json --
        ridge = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
        ridge.fit(Xtr, ylog_tr)
        pred_log_lin = ridge.predict(Xte)
        fold_metrics["r2_log_linear_fulldim"].append(float(r2_score(ylog_te, pred_log_lin)))
        fold_metrics["r2_dist_linear_fulldim"].append(float(r2_score(ydist_te, np.exp(pred_log_lin))))

    centroid_pc_corr = fold_metrics.pop("centroid_pc_corr")

    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for k, v in fold_metrics.items()}
    out["centroid_pc_corr"] = np.abs(np.array(centroid_pc_corr)).mean(axis=0).tolist()
    return out


def make_viz(X, y_dist, y_log, layer_name):
    """In-sample fit (no held-out split) purely to render a clean picture.
    Uses only 3 PCA components (for plotting) -- NOT the same subspace as the
    64-dim quantitative fit in evaluate_layer, so the picture is illustrative
    only and can look tighter/looser than the real held-out numbers."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=N_COMPONENTS_VIZ).fit(Xs)
    pcs_raw = pca.transform(Xs)
    pcs = StandardScaler().fit_transform(pcs_raw)
    spline, t_min, t_max, centroid_pc, centroid_t, _ = fit_curve(pcs, y_log)
    grid_t = np.linspace(t_min, t_max, N_CURVE_GRID)
    grid_pc = spline(grid_t)

    figs_b64 = []
    for azim in (35, 125):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=y_dist, cmap="viridis", s=6, alpha=0.5)
        ax.plot(grid_pc[:, 0], grid_pc[:, 1], grid_pc[:, 2], color="red", linewidth=2)
        ax.scatter(centroid_pc[:, 0], centroid_pc[:, 1], centroid_pc[:, 2], color="black", s=20)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{layer_name}\n(in-sample fit, colored by dist_nearest_front)")
        ax.view_init(elev=20, azim=azim)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="distance (m)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        figs_b64.append(base64.b64encode(buf.getvalue()).decode())

    # 2D PC1-vs-PC2 view as well (easier to read spline shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=y_dist, cmap="viridis", s=6, alpha=0.5)
    ax.plot(grid_pc[:, 0], grid_pc[:, 1], color="red", linewidth=2)
    ax.scatter(centroid_pc[:, 0], centroid_pc[:, 1], color="black", s=20)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{layer_name} (PC1 vs PC2)")
    fig.colorbar(sc, ax=ax, label="distance (m)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    fig_2d_b64 = base64.b64encode(buf.getvalue()).decode()

    return figs_b64[0], figs_b64[1], fig_2d_b64


def build_html(results, images):
    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LTFv6 Manifold Probe</title>
<style>
body { font-family: -apple-system, sans-serif; background:#0f1117; color:#e6e6e6; margin:2rem; }
h1,h2 { color:#fff; }
table { border-collapse: collapse; margin-bottom: 1.5rem; }
td, th { border: 1px solid #333; padding: 6px 10px; text-align: right; }
th { background:#1a1d27; }
.layer { margin-bottom: 3rem; padding-bottom: 2rem; border-bottom: 1px solid #333; }
img { border-radius: 8px; margin: 4px; background:#fff; }
.note { color:#9aa; font-size: 0.9em; max-width: 900px; }
</style></head><body>
<h1>LTFv6 &mdash; Manifold Probe (nonlinear curve decode vs linear Ridge)</h1>
<p class="note">Following Goodfire's Manifold Steering paper (arXiv:2605.05115): per layer, PCA to
64 components (fit on train only, log-grouped CV), a cubic spline fit through 30 per-bin centroids
parameterized by log(dist_nearest_front), and held-out points decoded by nearest-point-on-curve
lookup ("pullback"). r2_*_curve is compared against two linear baselines: Ridge fit in the SAME
64-dim PCA subspace (fair comparison, matches the paper's Pullback-vs-Linear table) and the
original full-dimensional Ridge (matches probe_results_ltfv6_grouped_cv.json). residual_ratio is
mean distance from held-out points to the nearest curve point, divided by the mean spread
of held-out points around their centroid &mdash; near 0 means points hug a tight 1D string,
near/above 1 means no such structure (scattered cloud, same information content as noise).
The images below use a separate 3-component PCA fit purely for plotting -- illustrative only,
not the same subspace the numbers above are computed in.</p>
<p class="note"><b>Honest conclusion:</b> residual_ratio &asymp; 1.0 and R&sup2;(curve) &lt;
R&sup2;(linear, same subspace) on every layer &mdash; per this script's own criteria, the data does
not lie on a 1D manifold; the linear probe is near the ceiling for what's recoverable here. (An
earlier version of this report also computed an arc-length-based "manifold exists, resolution is
non-uniform" story from the fitted curve. That was removed: cumulative arc-length is a monotonic
running sum by construction, so correlating it against metric distance is close to guaranteed to
read ~0.97 regardless of whether the activations encode anything real &mdash; a spurious
correlation from two monotonic series sharing an index, not evidence of manifold structure.)</p>
"""]
    for name, r in results.items():
        parts.append(f'<div class="layer"><h2>{name}</h2>')
        parts.append("<table><tr><th>metric</th><th>curve (nonlinear)</th><th>linear, same 64-dim PCA</th><th>linear, full-dim</th></tr>")
        parts.append(f"<tr><td>R²(log dist)</td><td>{r['r2_log_curve']['mean']:.3f} ± {r['r2_log_curve']['std']:.3f}</td>"
                      f"<td>{r['r2_log_linear_pca64']['mean']:.3f} ± {r['r2_log_linear_pca64']['std']:.3f}</td>"
                      f"<td>{r['r2_log_linear_fulldim']['mean']:.3f} ± {r['r2_log_linear_fulldim']['std']:.3f}</td></tr>")
        parts.append(f"<tr><td>R²(dist)</td><td>{r['r2_dist_curve']['mean']:.3f} ± {r['r2_dist_curve']['std']:.3f}</td>"
                      f"<td>{r['r2_dist_linear_pca64']['mean']:.3f} ± {r['r2_dist_linear_pca64']['std']:.3f}</td>"
                      f"<td>{r['r2_dist_linear_fulldim']['mean']:.3f} ± {r['r2_dist_linear_fulldim']['std']:.3f}</td></tr>")
        parts.append("</table>")
        parts.append(f"<p>PCA variance explained (64 PCs used for the fit, mean across folds): "
                      f"<b>{r['pca_var_explained']['mean']*100:.1f}%</b> &nbsp;|&nbsp; "
                      f"residual_ratio (string tightness, lower=tighter): "
                      f"<b>{r['residual_ratio']['mean']:.3f} ± {r['residual_ratio']['std']:.3f}</b> &nbsp;|&nbsp; "
                      f"|corr(centroid PC<sub>1-3</sub>, log-dist)|: "
                      f"<b>{', '.join(f'{c:.3f}' for c in r['centroid_pc_corr'])}</b></p>")
        im3d_a, im3d_b, im2d = images[name]
        parts.append(f'<img src="data:image/png;base64,{im3d_a}" width="420">')
        parts.append(f'<img src="data:image/png;base64,{im3d_b}" width="420">')
        parts.append(f'<img src="data:image/png;base64,{im2d}" width="420">')
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--layers", nargs="*", default=SIGNAL_LAYERS)
    ap.add_argument("--out_json", default=OUT_JSON)
    ap.add_argument("--out_html", default=OUT_HTML)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as h5:
        y_raw = h5["dist_nearest_front"][:]
        groups_all = load_groups(h5)
        valid = np.isfinite(y_raw) & (y_raw >= BIN_EDGES[0])
        y = y_raw[valid].astype(np.float32)
        y_log = np.log(y)
        groups = groups_all[valid]

        print(f"n_samples={valid.sum()}  n_logs={len(np.unique(groups))}")

        results, images = {}, {}
        for layer in args.layers:
            X = h5[layer][:][valid].astype(np.float32)
            print(f"\n== {layer} (dim={X.shape[1]}) ==")
            r = evaluate_layer(X, y, y_log, groups)
            for k, v in r.items():
                if k == "centroid_pc_corr":
                    print(f"  {k:24s} " + ", ".join(f"{c:.3f}" for c in v))
                else:
                    print(f"  {k:24s} {v['mean']:+.3f} ± {v['std']:.3f}")
            results[layer] = r
            images[layer] = make_viz(X, y, y_log, layer)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "label": "dist_nearest_front",
            "n_samples": int(valid.sum()),
            "n_groups": int(len(np.unique(groups))),
            "cv_strategy": "GroupKFold by log",
            "n_components_fit": N_COMPONENTS_FIT,
            "n_components_viz": N_COMPONENTS_VIZ,
            "n_curve_bins": N_CURVE_BINS,
            "layers": results,
        }, f, indent=2)
    print(f"\nSaved {args.out_json}")

    with open(args.out_html, "w") as f:
        f.write(build_html(results, images))
    print(f"Saved {args.out_html}")


if __name__ == "__main__":
    main()
