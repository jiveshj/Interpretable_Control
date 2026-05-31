#!/usr/bin/env python
"""
run_probe_sweep_ltfv6.py

Train per-layer linear probes on LTFv6 activations to test what each layer
encodes about distance to the nearest agent.

For each of 18 hook layers, trains 3 probes via 5-fold cross-validation with
inner alpha grid search:
  1. Ridge regression on distance              -> reports R²
  2. Ridge regression on log(distance)         -> reports R²
  3. Ridge classifier on quartile buckets      -> reports accuracy

Reports mean ± std across folds. Negative R² is meaningful (probe is worse
than predicting the mean -> the layer encodes nothing useful for distance).

Usage:
    python probe_sweep_ltfv6.py
    python probe_sweep_ltfv6.py --label dist_nearest_any
    python probe_sweep_ltfv6.py --n_buckets 5
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

# HDF5 dataset keys (matches the safe_name = name.replace(".", "_") in collect_data_ltfv6.py)
HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS = 5
RANDOM_STATE = 42


def quantile_buckets(y, n_buckets):
    """Bucket distances by quantile so each bucket has the same number of samples."""
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bins = np.quantile(y, quantiles)
    return np.digitize(y, bins).astype(int)


def probe_layer(X, y_dist, y_log, y_buck):
    """Run all 3 probes via k-fold CV. Returns dict with mean/std for each."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def reg_pipeline():
        return Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ])

    def cls_pipeline():
        return Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeClassifierCV(alphas=ALPHAS)),
        ])

    r2_dist = cross_val_score(reg_pipeline(), X, y_dist, cv=kf, scoring="r2", n_jobs=-1)
    r2_log = cross_val_score(reg_pipeline(), X, y_log, cv=kf, scoring="r2", n_jobs=-1)
    acc = cross_val_score(cls_pipeline(), X, y_buck, cv=kf, scoring="accuracy", n_jobs=-1)

    return {
        "r2_dist": {"mean": float(r2_dist.mean()), "std": float(r2_dist.std()),
                    "folds": r2_dist.tolist()},
        "r2_log":  {"mean": float(r2_log.mean()),  "std": float(r2_log.std()),
                    "folds": r2_log.tolist()},
        "acc":     {"mean": float(acc.mean()),     "std": float(acc.std()),
                    "folds": acc.tolist()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/ocean/projects/cis250201p/jjain2/ltfv6/latents_ltfv6.h5")
    p.add_argument("--label", default="dist_nearest_front",
                   choices=["dist_nearest_any", "dist_nearest_front"])
    p.add_argument("--n_buckets", type=int, default=4)
    p.add_argument("--out", default="/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.json")
    args = p.parse_args()

    with h5py.File(args.h5, "r") as h5:
        # Labels
        y_raw = h5[args.label][:]
        valid = np.isfinite(y_raw)
        y = y_raw[valid].astype(np.float32)
        y_log = np.log(y)
        y_buck = quantile_buckets(y, args.n_buckets)

        chance = 1.0 / args.n_buckets
        bucket_counts = np.bincount(y_buck, minlength=args.n_buckets)

        print(f"Label              : {args.label}")
        print(f"Valid samples      : {valid.sum()} / {len(valid)}")
        print(f"Distance range     : min={y.min():.2f}  median={np.median(y):.2f}  max={y.max():.2f}")
        print(f"Buckets (n={args.n_buckets:d})       : {bucket_counts.tolist()}  (chance acc = {chance:.2f})")
        print(f"CV                 : {N_SPLITS}-fold, alphas = {ALPHAS}")
        print()
        print(f"{'LAYER':<58} {'R²(dist)':>14} {'R²(log)':>14} {'Acc(buck)':>14}")
        print("-" * 102)

        results = {}
        for key in HOOK_KEYS:
            if key not in h5:
                print(f"  {key:<56}  MISSING IN HDF5")
                continue
            X = h5[key][:][valid].astype(np.float32)
            r = probe_layer(X, y, y_log, y_buck)
            results[key] = r
            print(f"  {key:<56} "
                  f"{r['r2_dist']['mean']:>+6.3f}±{r['r2_dist']['std']:>4.3f} "
                  f"{r['r2_log']['mean']:>+6.3f}±{r['r2_log']['std']:>4.3f} "
                  f"{r['acc']['mean']:>+6.3f}±{r['acc']['std']:>4.3f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "label": args.label,
            "n_samples": int(valid.sum()),
            "n_buckets": args.n_buckets,
            "chance_accuracy": chance,
            "cv_splits": N_SPLITS,
            "alphas": ALPHAS,
            "layers": results,
        }, f, indent=2)
    print(f"\nSaved results to {args.out}")

    # Quick interpretation hint
    print("\nReading the table:")
    print(f"  R² > 0   : layer encodes distance better than predicting the mean")
    print(f"  R² ≈ 0   : no useful signal")
    print(f"  R² < 0   : probe overfits / no signal (with 92 samples this is common)")
    print(f"  Acc > {chance:.2f} : layer encodes coarse-distance bucket better than chance")


if __name__ == "__main__":
    main()