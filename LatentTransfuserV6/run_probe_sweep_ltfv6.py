#!/usr/bin/env python
"""
run_probe_sweep_ltfv6.py

Per-layer linear probes on LTFv6 activations, 5-fold CV:
  1. Ridge regression on raw distance        -> R²(dist)
  2. Ridge regression on log(distance)       -> R²(log)
  3. Ridge regression on bin index (0-3)     -> R²(bins)  <- coarse
  4. Ridge classifier on fixed bins          -> Acc(bins)

R²(bins) > R²(dist) means the layer encodes coarse distance better than
precise metric distance -- the core hypothesis.

Fixed bins (matching TransFuser work):
  bin 0:  3 -  5 m
  bin 1:  5 - 10 m
  bin 2: 10 - 20 m
  bin 3: 20 - 40 m+  (samples >= 40m kept in this bin)

Usage:
    python run_probe_sweep_ltfv6.py
    python run_probe_sweep_ltfv6.py --label dist_nearest_any
    python run_probe_sweep_ltfv6.py --h5 /path/to/latents.h5
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT  = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
OUT_DEFAULT = "/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.json"

HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]

ALPHAS     = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS   = 5
BIN_EDGES  = [3.0, 5.0, 10.0, 20.0, 40.0]
BIN_NAMES  = ["3-5m", "5-10m", "10-20m", "20-40m+"]
BIN_WIDTHS = [e2 - e1 for e1, e2 in zip(BIN_EDGES, BIN_EDGES[1:])]  # [2, 5, 10, 20]


def probe_per_bin(X, y_dist, y_bins):
    """Train a separate Ridge probe within each distance bin.

    For each bin, only the samples whose ground-truth distance falls in that
    bin are used for training and cross-validation.  This answers: given that
    a vehicle is in the 5-10 m range, how precisely does the layer encode its
    exact distance?

    Returns a dict keyed by BIN_NAMES.  Each entry contains:
      mae        : cross-val MAE (meters) mean ± std
      r2         : cross-val R²          mean ± std
      naive_mae  : predict-the-mean baseline MAE for this bin
      relative_mae : mae_mean / bin_width  (comparable across bins)
      n          : number of samples in this bin
    """
    results = {}
    for b, (name, width) in enumerate(zip(BIN_NAMES, BIN_WIDTHS)):
        mask = y_bins == b
        n_bin = int(mask.sum())
        if n_bin < 2:
            results[name] = {"n": n_bin, "note": "too few samples"}
            continue
        X_b, y_b = X[mask], y_dist[mask]
        n_folds   = min(N_SPLITS, n_bin)
        kf_b      = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        pipe      = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
        mae = -cross_val_score(pipe, X_b, y_b, cv=kf_b, scoring="neg_mean_absolute_error")
        r2  =  cross_val_score(pipe, X_b, y_b, cv=kf_b, scoring="r2")
        naive = float(np.abs(y_b - y_b.mean()).mean())
        results[name] = {
            "mae":          {"mean": float(mae.mean()), "std": float(mae.std())},
            "r2":           {"mean": float(r2.mean()),  "std": float(r2.std())},
            "naive_mae":    naive,
            "relative_mae": float(mae.mean() / width),
            "n":            n_bin,
        }
    return results


def probe_layer(X, y_dist, y_log, y_bins, kf):
    def reg():
        return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
    def clf():
        return Pipeline([("scale", StandardScaler()), ("ridge", RidgeClassifierCV(alphas=ALPHAS))])

    r2_dist = cross_val_score(reg(), X, y_dist,              cv=kf, scoring="r2")
    r2_log  = cross_val_score(reg(), X, y_log,               cv=kf, scoring="r2")
    r2_bins = cross_val_score(reg(), X, y_bins.astype(float), cv=kf, scoring="r2")
    acc     = cross_val_score(clf(), X, y_bins,              cv=kf, scoring="accuracy")
    return {
        "r2_dist": {"mean": float(r2_dist.mean()), "std": float(r2_dist.std())},
        "r2_log":  {"mean": float(r2_log.mean()),  "std": float(r2_log.std())},
        "r2_bins": {"mean": float(r2_bins.mean()), "std": float(r2_bins.std())},
        "acc":     {"mean": float(acc.mean()),     "std": float(acc.std())},
    }


def _print_per_bin_tables(per_bin_rows, y_dist, y_bins):
    """Print per-bin MAE and relative-MAE tables after the main sweep."""
    LAYER_W = 56
    COL_W   = 18

    # Naive baselines (predict bin mean)
    naive_maes = {}
    for b, name in enumerate(BIN_NAMES):
        mask = y_bins == b
        naive_maes[name] = float(np.abs(y_dist[mask] - y_dist[mask].mean()).mean()) if mask.any() else float("nan")

    header = f"  {'LAYER':<{LAYER_W}}" + "".join(f"{n:^{COL_W}}" for n in BIN_NAMES)
    sep    = "  " + "-" * (LAYER_W + COL_W * len(BIN_NAMES))

    # Table 1: absolute MAE
    print(f"\n\nPer-bin MAE (meters)  [lower = more precisely encoded within that range]")
    print(f"  sample counts: " +
          "  ".join(f"{n}(n={(y_bins==b).sum()})" for b, n in enumerate(BIN_NAMES)))
    print(f"  naive MAE baselines: " +
          "  ".join(f"{n}={naive_maes[n]:.3f}m" for n in BIN_NAMES))
    print(header)
    print(sep)
    for key, pb in per_bin_rows:
        if pb is None:
            print(f"  {key:<{LAYER_W}}" + "".join(f"{'MISSING':^{COL_W}}" for _ in BIN_NAMES))
            continue
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            if "note" in d:
                cols.append(f"{'<2 samples':^{COL_W}}")
            else:
                cols.append(f"{d['mae']['mean']:.3f}+-{d['mae']['std']:.3f}m".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))

    # Table 2: relative MAE = MAE / bin_width  (comparable across bins)
    print(f"\n\nPer-bin relative MAE (MAE / bin_width)  [comparable across bins]")
    print(f"  bin widths: " + "  ".join(f"{n}={w:.0f}m" for n, w in zip(BIN_NAMES, BIN_WIDTHS)))
    print(header)
    print(sep)
    for key, pb in per_bin_rows:
        if pb is None:
            print(f"  {key:<{LAYER_W}}" + "".join(f"{'MISSING':^{COL_W}}" for _ in BIN_NAMES))
            continue
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            if "note" in d:
                cols.append(f"{'<2 samples':^{COL_W}}")
            else:
                cols.append(f"{d['relative_mae']:.3f}".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))

    # Table 3: R² within each bin
    print(f"\n\nPer-bin R²  [higher = activations encode fine-grained distance within that range]")
    print(header)
    print(sep)
    for key, pb in per_bin_rows:
        if pb is None:
            print(f"  {key:<{LAYER_W}}" + "".join(f"{'MISSING':^{COL_W}}" for _ in BIN_NAMES))
            continue
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            if "note" in d:
                cols.append(f"{'<2 samples':^{COL_W}}")
            else:
                cols.append(f"{d['r2']['mean']:+.3f}+-{d['r2']['std']:.3f}".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",    default=H5_DEFAULT)
    p.add_argument("--label", default="dist_nearest_front",
                   choices=["dist_nearest_any", "dist_nearest_front", "dist_nearest_front_halfplane"])
    p.add_argument("--out",   default=OUT_DEFAULT)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as h5:
        y_raw = h5[args.label][:]

    # Drop NaNs and samples below minimum distance; keep everything above 40m in last bin
    valid  = np.isfinite(y_raw) & (y_raw >= BIN_EDGES[0])
    y      = y_raw[valid].astype(np.float32)
    y_log  = np.log(y)
    y_bins = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, len(BIN_NAMES) - 1).astype(int)

    # -- Bin distribution -----------------------------------------------------
    counts           = np.bincount(y_bins, minlength=len(BIN_NAMES))
    majority_baseline = counts.max() / valid.sum()

    print(f"Label         : {args.label}")
    print(f"Valid samples : {valid.sum()} / {len(y_raw)}")
    print(f"Distance range: {y.min():.2f} - {y.max():.2f} m  (median {np.median(y):.2f})")
    print()
    print(f"  {'BIN':<12} {'N':>6}  {'%':>6}   distribution")
    print("  " + "-" * 55)
    for name, c in zip(BIN_NAMES, counts):
        bar = "#" * int(30 * c / counts.max())
        print(f"  {name:<12} {c:>6}  {100*c/valid.sum():>5.1f}%   {bar}")
    print(f"\n  majority-class baseline : {majority_baseline:.3f}")
    print(f"  uniform-chance baseline : {1/len(BIN_NAMES):.3f}")

    # -- Per-layer probe sweep ------------------------------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    print(f"\n{'LAYER':<58} {'R2(dist)':>12} {'R2(log)':>12} {'R2(bins)':>12} {'Acc(bins)':>12}")
    print("-" * 108)

    results      = {}
    per_bin_rows = []   # list of (key, per_bin_dict) for table printing
    with h5py.File(args.h5, "r") as h5:
        for key in HOOK_KEYS:
            if key not in h5:
                print(f"  {key:<56}  MISSING")
                per_bin_rows.append((key, None))
                continue
            X = h5[key][:][valid].astype(np.float32)
            r = probe_layer(X, y, y_log, y_bins, kf)
            results[key] = r
            print(f"  {key:<56} "
                  f"{r['r2_dist']['mean']:>+6.3f}+-{r['r2_dist']['std']:>4.3f} "
                  f"{r['r2_log']['mean']:>+6.3f}+-{r['r2_log']['std']:>4.3f} "
                  f"{r['r2_bins']['mean']:>+6.3f}+-{r['r2_bins']['std']:>4.3f} "
                  f"{r['acc']['mean']:>+6.3f}+-{r['acc']['std']:>4.3f}")

            pb = probe_per_bin(X, y, y_bins)
            results[key]["per_bin"] = pb
            per_bin_rows.append((key, pb))

    print(f"\nBaselines -- majority: {majority_baseline:.3f}  uniform: {1/len(BIN_NAMES):.3f}")

    # -- Per-bin tables -------------------------------------------------------
    _print_per_bin_tables(per_bin_rows, y, y_bins)

    # -- Save -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "label": args.label,
            "n_samples": int(valid.sum()),
            "bin_edges": BIN_EDGES,
            "bin_names": BIN_NAMES,
            "bin_widths": BIN_WIDTHS,
            "majority_baseline": majority_baseline,
            "cv_splits": N_SPLITS,
            "alphas": ALPHAS,
            "layers": results,
        }, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()