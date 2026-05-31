"""
per_bin_analysis.py
-------------------
Per-bin probe performance: which distance ranges does the model encode well?

For each layer, fits a Ridge regression on the train split, then on the
test split computes per bin:
  - MAE         : mean absolute error in meters
  - Rel. MAE    : MAE / bin_width  (0=perfect, 1=error as wide as the bin)
  - R²          : explained variance within the bin
                  (baseline = that bin's mean, not the global mean)

Usage:
  python per_bin_analysis.py --data /path/to/latents.h5
  python per_bin_analysis.py --data /path/to/latents.h5 --layer transformer4
"""

import argparse
import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ── Bin definitions (must match collect_data.py) ──────────────────────────────
DISTANCE_BINS = [
    (0.5,  3.0),    # bin 0: very close
    (3.0,  5.0),    # bin 1: close following
    (5.0,  10.0),   # bin 2: normal following
    (10.0, 20.0),   # bin 3: medium range
    (20.0, 40.0),   # bin 4: far
]
BIN_LABELS = [f"bin{i} [{lo:.0f}–{hi:.0f}m)" for i, (lo, hi) in enumerate(DISTANCE_BINS)]

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
TEST_SIZE    = 0.2
RANDOM_SEED  = 42


def assign_bins(distances: np.ndarray) -> np.ndarray:
    ids = np.full(len(distances), -1, dtype=int)
    for i, (lo, hi) in enumerate(DISTANCE_BINS):
        ids[(distances >= lo) & (distances < hi)] = i
    return ids


def fit_best_ridge(X_tr, y_tr, X_te):
    """Ridge with alpha grid search on a val holdout; refit on full train."""
    rng   = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(0.2 * len(X_tr)))
    idx   = rng.permutation(len(X_tr))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    scaler = StandardScaler().fit(X_tr[tr_idx])
    Xtr_s  = scaler.transform(X_tr[tr_idx])
    Xva_s  = scaler.transform(X_tr[val_idx])
    Xte_s  = scaler.transform(X_te)

    best_alpha, best_r2 = RIDGE_ALPHAS[0], -np.inf
    for alpha in RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(Xtr_s, y_tr[tr_idx])
        r = r2_score(y_tr[val_idx], m.predict(Xva_s))
        if r > best_r2:
            best_r2, best_alpha = r, alpha

    final = Ridge(alpha=best_alpha).fit(scaler.transform(X_tr), y_tr)
    return final.predict(Xte_s)


def analyse_layer(layer_name, X, y, bin_ids):
    rng  = np.random.default_rng(RANDOM_SEED)
    idx  = rng.permutation(len(y))
    n_te = int(TEST_SIZE * len(y))
    te_idx, tr_idx = idx[:n_te], idx[n_te:]

    y_pred = fit_best_ridge(X[tr_idx], y[tr_idx], X[te_idx])
    y_te   = y[te_idx]
    b_te   = bin_ids[te_idx]

    print(f"\n{'═'*65}")
    print(f"  Layer: {layer_name}  (dim={X.shape[1]}, n_test={n_te})")
    print(f"{'─'*65}")
    print(f"  {'Bin':25s}  {'N':>5}  {'MAE(m)':>7}  {'RelMAE':>7}  {'R²':>7}")
    print(f"  {'─'*25}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}")

    for b, label in enumerate(BIN_LABELS):
        mask = b_te == b
        n    = mask.sum()
        if n < 5:
            print(f"  {label:25s}  {n:>5}  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}")
            continue

        yt, yp        = y_te[mask], y_pred[mask]
        mae           = mean_absolute_error(yt, yp)
        bin_width     = DISTANCE_BINS[b][1] - DISTANCE_BINS[b][0]
        rel_mae       = mae / bin_width
        # r2_score on filtered samples uses yt.mean() as baseline = bin mean
        r2            = r2_score(yt, yp)

        print(f"  {label:25s}  {n:>5}  {mae:>7.2f}  {rel_mae:>7.2f}  {r2:>7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True)
    parser.add_argument("--layer", default=None,
                        help="Single layer name (default: all layers)")
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        gt_dist = f["gt_distance"][:]
        grp     = f["latents"]
        layers  = [args.layer] if args.layer else list(grp.keys())
        latents = {name: grp[name][:] for name in layers}

    bin_ids = assign_bins(gt_dist)
    valid   = bin_ids >= 0
    if not valid.all():
        print(f"[warn] Dropping {(~valid).sum()} samples outside bin ranges.")
        gt_dist = gt_dist[valid]
        bin_ids = bin_ids[valid]
        latents = {k: v[valid] for k, v in latents.items()}

    print(f"\nDataset: {len(gt_dist)} samples, "
          f"distance [{gt_dist.min():.1f}, {gt_dist.max():.1f}] m")
    for b, label in enumerate(BIN_LABELS):
        print(f"  {label}: {(bin_ids == b).sum()} samples")

    for name, X in latents.items():
        analyse_layer(name, X, gt_dist, bin_ids)


if __name__ == "__main__":
    main()