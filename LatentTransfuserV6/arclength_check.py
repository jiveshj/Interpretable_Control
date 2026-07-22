#!/usr/bin/env python
"""
arclength_check.py

Robustness + null control for the arc-length-vs-distance concavity.

Concavity = area between the normalized cumulative-arc-length curve and the
straight diagonal. >0 concave (fine-near/coarse-far), ~0 linear, <0 convex.

Controls:
  --shuffle : permute the distance labels before binning. Holds the layer's
              activations and PCA fixed, breaks ONLY the activation<->distance
              correspondence. If concavity collapses to ~0 under shuffle, the
              real concavity is driven by the genuine activation-distance
              relationship, not by the PCA geometry of the layer. This is the
              airtight control (stronger than log-vs-raw parameterization,
              which threads a spline through the SAME centroids and is nearly
              invariant by construction).

Also compares log- vs raw-distance parameterization (a weak consistency
check: arc-length is robust to how the spline grid is clocked).

Defaults to the halfplane label (4622 samples).

Usage:
    python arclength_check.py --layer backbone_transformers_2
    python arclength_check.py --layer backbone_transformers_2 --shuffle
"""
import argparse, os
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

H5 = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
N_COMPONENTS = 64
N_BINS = 30
N_GRID = 2000
BIN_MIN = 3.0


def centroids(pcs, t, n_bins):
    order = np.argsort(t)
    chunks = np.array_split(order, n_bins)
    ct = np.array([t[c].mean() for c in chunks if len(c) > 0])
    cp = np.array([pcs[c].mean(axis=0) for c in chunks if len(c) > 0])
    keep = np.concatenate([[True], np.diff(ct) > 1e-9])
    return ct[keep], cp[keep]


def arclength_curve(pcs, t_param, dist_for_centroid, n_bins=N_BINS, n_grid=N_GRID):
    ct, cp = centroids(pcs, t_param, n_bins)
    spline = CubicSpline(ct, cp, axis=0, bc_type="natural")
    grid_t = np.linspace(ct.min(), ct.max(), n_grid)
    grid_pc = spline(grid_t)
    seg = np.linalg.norm(np.diff(grid_pc, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cal = np.interp(ct, grid_t, cum)
    cd = dist_for_centroid(ct)
    return cd, cal


def concavity(d, L):
    dd = (d - d.min()) / (d.max() - d.min())
    span = L.max() - L.min()
    if span <= 1e-12:
        return float("nan")   # degenerate: no manifold (e.g. null layer)
    Ln = (L - L.min()) / span
    integ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integ(Ln - dd, dd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="backbone_transformers_2")
    ap.add_argument("--label", default="dist_nearest_front_halfplane")
    ap.add_argument("--shuffle", action="store_true",
                    help="permute distance labels (null control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with h5py.File(H5) as h:
        y_raw = h[a.label][:]
        X = h[a.layer][:]
    v = np.isfinite(y_raw) & (y_raw >= BIN_MIN)
    y = y_raw[v].astype(np.float64)
    X = X[v].astype(np.float64)

    if a.shuffle:
        rng = np.random.default_rng(a.seed)
        y = rng.permutation(y)
        tag = " [SHUFFLED LABELS - null control]"
    else:
        tag = ""
    print(f"{a.layer} / {a.label}  n={len(y)}{tag}")

    Xs = StandardScaler().fit_transform(X)
    # guard against zero-variance / degenerate layers (e.g. fabricated lidar grid)
    var = Xs.var(axis=0)
    if np.all(var < 1e-12):
        print("  DEGENERATE layer: near-zero variance, no manifold. concavity = nan")
        return
    n_comp = min(N_COMPONENTS, Xs.shape[0]-1, Xs.shape[1])
    pcs = PCA(n_components=n_comp).fit_transform(Xs)
    pcs = StandardScaler().fit_transform(pcs)

    d_log, L_log = arclength_curve(pcs, np.log(y), np.exp)
    d_raw, L_raw = arclength_curve(pcs, y, lambda t: t)

    c_log = concavity(d_log, L_log)
    c_raw = concavity(d_raw, L_raw)

    def norm(A):
        s = A.max() - A.min()
        return (A - A.min()) / s if s > 1e-12 else A*0.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d_log, norm(L_log), "o-", color="#c0392b", label="log-distance param")
    ax.plot(d_raw, norm(L_raw), "s-", color="#2c7fb8", label="raw-distance param")
    ax.plot([d_raw.min(), d_raw.max()], [0, 1], "k--", lw=1, alpha=0.6,
            label="linear reference")
    ax.set_xlabel("distance (m)"); ax.set_ylabel("normalized cumulative arc length")
    ttl = f"{a.layer}{tag}\nconcavity: log={c_log:+.3f}  raw={c_raw:+.3f}"
    ax.set_title(ttl); ax.legend(fontsize=9); ax.grid(alpha=0.2)

    suffix = "_shuffled" if a.shuffle else ""
    out = a.out or f"/jet/home/jjain2/Interpretable_Control/arclength_check_{a.layer}{suffix}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved {out}")
    print(f"  concavity(log-param) = {c_log:+.4f}")
    print(f"  concavity(raw-param) = {c_raw:+.4f}")
    print("  (>0 concave = fine-near/coarse-far;  ~0 linear;  <0 convex;  nan = degenerate)")


if __name__ == "__main__":
    main()


def null_test():
    """Run N shuffles, report where the real concavity sits in the null distribution."""
    import sys
    layer = "backbone_transformers_2"
    label = "dist_nearest_front_halfplane"
    N = 200
    with h5py.File(H5) as h:
        y_raw = h[label][:]; X = h[layer][:]
    v = np.isfinite(y_raw) & (y_raw >= BIN_MIN)
    y = y_raw[v].astype(np.float64); X = X[v].astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(N_COMPONENTS, Xs.shape[0]-1, Xs.shape[1])
    pcs = StandardScaler().fit_transform(PCA(n_components=n_comp).fit_transform(Xs))

    def conc(yy):
        d, L = arclength_curve(pcs, np.log(yy), np.exp)
        return concavity(d, L)

    real = conc(y)
    rng = np.random.default_rng(0)
    null = np.array([conc(rng.permutation(y)) for _ in range(N)])
    p = (np.sum(null >= real) + 1) / (N + 1)
    print(f"\nreal concavity      = {real:+.4f}")
    print(f"null mean +- std    = {null.mean():+.4f} +- {null.std():.4f}")
    print(f"null 95th pct       = {np.percentile(null,95):+.4f}")
    print(f"real z-score vs null= {(real-null.mean())/null.std():+.2f}")
    print(f"permutation p-value = {p:.4f}   (real >= null)")
    print("  p<0.05 AND z>2 => concavity is distance-specific, NOT an artifact")
    print("  p~0.5, z~0    => concavity is a method artifact (shuffle reproduces it)")

if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "nulltest":
    null_test()
