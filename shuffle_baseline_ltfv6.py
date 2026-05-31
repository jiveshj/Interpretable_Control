#!/usr/bin/env python
"""
shuffle_baseline_ltfv6.py

Null baseline for the probe sweep. For each layer, shuffles the distance
labels N times and re-runs the ridge probe. If shuffled R² is flat near 0
across layers while real R² has clear shape, the real pattern reflects
encoded information rather than feature-dimensionality artifacts.

Compares to the real R²(log) values from a previous `probe_sweep_ltfv6.py`
run (loads from probe_results_ltfv6.json).

Output column meanings:
    REAL       : R²(log) from probe_sweep_ltfv6.py (real labels)
    SHUFFLED   : mean ± std of R²(log) over N permutations of the labels
    GAP        : REAL - SHUFFLED. If much > 0, layer encodes real signal.
    Z          : GAP / SHUFFLED_std. Rough "how many shuffled stds above null".
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS = 5


def probe_r2(X, y, kf_seed=42):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=kf_seed)
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return float(scores.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/ocean/projects/cis250201p/jjain2/ltfv6/latents_ltfv6.h5")
    p.add_argument("--label", default="dist_nearest_front")
    p.add_argument("--real_json", default="/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.json")
    p.add_argument("--n_shuffles", type=int, default=20)
    p.add_argument("--out", default="/jet/home/jjain2/Interpretable_Control/probe_shuffle_baseline_ltfv6.json")
    args = p.parse_args()

    # Load the previous run's real R²(log) values for comparison
    with open(args.real_json) as f:
        real = json.load(f)

    with h5py.File(args.h5, "r") as h5:
        y_raw = h5[args.label][:]
        valid = np.isfinite(y_raw)
        y_log = np.log(y_raw[valid].astype(np.float32))

        print(f"Label              : {args.label}")
        print(f"Valid samples      : {int(valid.sum())}")
        print(f"Shuffles per layer : {args.n_shuffles}")
        print(f"CV                 : {N_SPLITS}-fold (fixed seed across shuffles)")
        print()
        print(f"{'LAYER':<58} {'REAL':>9} {'SHUFFLED':>18} {'GAP':>9} {'Z':>7}")
        print("-" * 108)

        rng = np.random.RandomState(42)
        results = {}

        for key in HOOK_KEYS:
            if key not in h5:
                print(f"  {key:<56}  MISSING IN HDF5")
                continue

            X = h5[key][:][valid].astype(np.float32)
            shuf_r2s = []
            for _ in range(args.n_shuffles):
                y_perm = rng.permutation(y_log)
                shuf_r2s.append(probe_r2(X, y_perm))
            shuf_r2s = np.array(shuf_r2s)
            shuf_mean = float(shuf_r2s.mean())
            shuf_std = float(shuf_r2s.std())

            real_r2 = real["layers"][key]["r2_log"]["mean"]
            gap = real_r2 - shuf_mean
            z = gap / (shuf_std + 1e-9)

            results[key] = {
                "real_r2_log": real_r2,
                "shuffled_r2_log_mean": shuf_mean,
                "shuffled_r2_log_std": shuf_std,
                "shuffled_r2_log_runs": shuf_r2s.tolist(),
                "gap": gap,
                "z": z,
            }
            print(f"  {key:<56} {real_r2:>+8.3f}  {shuf_mean:>+8.3f}±{shuf_std:>4.3f}  {gap:>+8.3f}  {z:>+6.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "label": args.label,
            "n_samples": int(valid.sum()),
            "n_shuffles": args.n_shuffles,
            "alphas": ALPHAS,
            "cv_splits": N_SPLITS,
            "layers": results,
        }, f, indent=2)
    print(f"\nSaved to {args.out}")

    print("\nInterpretation:")
    print("  SHUFFLED near 0 everywhere  -> baseline behaves correctly")
    print("  SHUFFLED rises with depth   -> dimensionality artifact, subtract from real to isolate signal")
    print("  GAP > 0 and Z > 2           -> real signal exceeds null baseline")


if __name__ == "__main__":
    main()