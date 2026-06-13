#!/usr/bin/env python
"""
per_layer_probe_ltfv6.py

One GENERAL probe per layer (18 probes total), error reported per bin.

For each layer, a single Ridge probe is trained on samples from ALL distance
bins. Out-of-fold predictions (5-fold CV) are then grouped by the true bin of
each test sample, giving an 18 x 4 table:

  cell (layer, bin) = MAE of layer's general probe on held-out samples
                      whose true distance falls in that bin.

This answers: does one distance readout per layer work equally well at all
ranges, or does it degrade up close / far away?

Compare against:
  - the specialized per-(layer,bin) probes in run_probe_sweep_ltfv6.py
    (the gap = what bin-specialization buys)
  - the naive baseline printed per bin (global-mean predictor, split by bin)

Hyperparameters are identical to run_probe_sweep_ltfv6.py for fair
comparison: StandardScaler + RidgeCV(alphas), KFold(5, shuffle, seed 42),
same bin edges, same validity filter.

Usage:
    python per_layer_probe_ltfv6.py
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT  = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
OUT_DEFAULT = "/jet/home/jjain2/Interpretable_Control/general_probe_per_layer_results_ltfv6.json"

HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]

# Same hyperparameters as run_probe_sweep_ltfv6.py for fair comparison.
ALPHAS     = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS   = 5
BIN_EDGES  = [3.0, 5.0, 10.0, 20.0, 40.0]
BIN_NAMES  = ["3-5m", "5-10m", "10-20m", "20-40m+"]
BIN_WIDTHS = [e2 - e1 for e1, e2 in zip(BIN_EDGES, BIN_EDGES[1:])]  # [2, 5, 10, 20]
LAYER_W    = 56
COL_W      = 18


def reg_pipeline():
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def general_probe_per_bin(X, y, y_bins, kf, n_bins):
    """One probe trained on all bins; per-fold test MAE grouped by true bin.

    For each fold: fit on the train split (all bins mixed), predict the test
    split, compute MAE separately for each bin's test samples. Reported as
    mean +/- std across the 5 folds, matching run_probe_sweep_ltfv6.py.
    """
    fold_mae = np.full((N_SPLITS, n_bins), np.nan)
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        pipe = reg_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        errs = np.abs(pipe.predict(X[test_idx]) - y[test_idx])
        tb = y_bins[test_idx]
        for b in range(n_bins):
            mask = tb == b
            if mask.any():
                fold_mae[k, b] = errs[mask].mean()
    out = {}
    for b in range(n_bins):
        m, s = float(np.nanmean(fold_mae[:, b])), float(np.nanstd(fold_mae[:, b]))
        out[BIN_NAMES[b]] = {
            "mae_mean":     m,
            "mae_std":      s,
            "relative_mae": m / BIN_WIDTHS[b],
            "n":            int((y_bins == b).sum()),
        }
    out["overall_mae"] = float(np.nanmean(fold_mae))
    return out


def print_table(title, rows, fmt_fn):
    sep = "  " + "-" * (LAYER_W + COL_W * len(BIN_NAMES))
    hdr = f"  {'LAYER':<{LAYER_W}}" + "".join(f"{n:^{COL_W}}" for n in BIN_NAMES)
    print(f"\n{title}\n")
    print(hdr)
    print(sep)
    for key, row in rows:
        cols = "".join(f"{fmt_fn(row, n):^{COL_W}}" for n in BIN_NAMES)
        print(f"  {key:<{LAYER_W}}{cols}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",    default=H5_DEFAULT)
    p.add_argument("--label", default="dist_nearest_front",
                   choices=["dist_nearest_any", "dist_nearest_front"])
    p.add_argument("--out",   default=OUT_DEFAULT)
    args = p.parse_args()

    n_bins = len(BIN_NAMES)

    with h5py.File(args.h5, "r") as h5:
        y_raw  = h5[args.label][:]
        valid  = np.isfinite(y_raw) & (y_raw >= BIN_EDGES[0])
        y      = y_raw[valid].astype(np.float32)
        y_bins = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, n_bins - 1).astype(int)
        counts = np.bincount(y_bins, minlength=n_bins)

        # Naive baseline: predict the GLOBAL mean, error split by bin.
        global_mean = float(y.mean())
        naive = {BIN_NAMES[b]: float(np.abs(y[y_bins == b] - global_mean).mean())
                 for b in range(n_bins)}

        print(f"Label         : {args.label}")
        print(f"Valid samples : {valid.sum()} / {len(y_raw)}")
        print(f"Global mean   : {global_mean:.2f} m")
        print("Bin counts    : " + "  ".join(f"{n}(n={c})" for n, c in zip(BIN_NAMES, counts)))
        print("Naive MAE (predict global mean), per bin: " +
              "  ".join(f"{n}={naive[n]:.2f}m" for n in BIN_NAMES))

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        rows, results = [], {}
        for key in HOOK_KEYS:
            if key not in h5:
                rows.append((key, None))
                continue
            X = h5[key][:][valid].astype(np.float32)
            r = general_probe_per_bin(X, y, y_bins, kf, n_bins)
            results[key] = r
            rows.append((key, r))

    def fmt_mae(row, name):
        if row is None:
            return "MISSING"
        d = row[name]
        return f"{d['mae_mean']:.3f}+-{d['mae_std']:.3f}m"

    def fmt_rel(row, name):
        return "MISSING" if row is None else f"{row[name]['relative_mae']:.3f}"

    print_table("General-probe MAE per bin  [one probe per layer, trained on ALL bins]",
                rows, fmt_mae)
    print_table("General-probe relative MAE (MAE / bin width)  [comparable across bins]",
                rows, fmt_rel)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "label": args.label,
            "n_samples": int(valid.sum()),
            "bin_edges": BIN_EDGES, "bin_names": BIN_NAMES, "bin_widths": BIN_WIDTHS,
            "bin_counts": counts.tolist(),
            "global_mean": global_mean,
            "naive_global_mean_mae": naive,
            "alphas": ALPHAS, "cv_splits": N_SPLITS,
            "layers": results,
        }, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()