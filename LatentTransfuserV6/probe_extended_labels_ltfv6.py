#!/usr/bin/env python
"""
probe_extended_labels_ltfv6.py

Linear probe sweep over the extended labels produced by
extract_extended_labels_ltfv6.py.  Same 5-fold CV Ridge setup as
run_probe_sweep_ltfv6.py.

Distance regression labels (Ridge, global R²/MAE + per-bin probe):
  dist_2nd_nearest_any, dist_2nd_nearest_front

Other regression label (Ridge, global R²/MAE only — not a distance):
  nearest_front_heading

Binary classification labels (RidgeClassifier, accuracy + balanced accuracy):
  has_vehicle_front, same_lane_binary, opposing_lane_binary

Usage:
    python probe_extended_labels_ltfv6.py
    python probe_extended_labels_ltfv6.py --h5 /path/to/latents.h5
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT  = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
NPZ_DEFAULT = "/jet/home/jjain2/Interpretable_Control/extended_labels_ltfv6.npz"
OUT_DEFAULT = "/jet/home/jjain2/Interpretable_Control/probe_extended_results_ltfv6.json"

HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]

ALPHAS   = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS = 5

BIN_EDGES  = [3.0, 5.0, 10.0, 20.0, 40.0]
BIN_NAMES  = ["3-5m", "5-10m", "10-20m", "20-40m+"]
BIN_WIDTHS = [e2 - e1 for e1, e2 in zip(BIN_EDGES, BIN_EDGES[1:])]  # [2, 5, 10, 20]

# Distance labels get both global + per-bin probing
DIST_REG_LABELS  = ["dist_2nd_nearest_any", "dist_2nd_nearest_front"]
# Heading label gets global probing only (not a distance, binning doesn't apply)
OTHER_REG_LABELS = ["nearest_front_heading"]
CLF_LABELS       = ["has_vehicle_front", "same_lane_binary", "opposing_lane_binary"]


def reg_pipe():
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])

def clf_pipe():
    return Pipeline([("scale", StandardScaler()), ("clf", RidgeClassifierCV(alphas=ALPHAS))])


def probe_regression(X, y, kf):
    r2  = cross_val_score(reg_pipe(), X, y, cv=kf, scoring="r2")
    mae = -cross_val_score(reg_pipe(), X, y, cv=kf, scoring="neg_mean_absolute_error")
    return {
        "r2":  {"mean": float(r2.mean()),  "std": float(r2.std())},
        "mae": {"mean": float(mae.mean()), "std": float(mae.std())},
        "n":   len(y),
    }


def probe_classification(X, y, skf):
    acc  = cross_val_score(clf_pipe(), X, y, cv=skf, scoring="accuracy")
    bacc = cross_val_score(clf_pipe(), X, y, cv=skf, scoring="balanced_accuracy")
    counts   = np.bincount(y.astype(int), minlength=2)
    majority = float(counts.max() / len(y))
    return {
        "acc":               {"mean": float(acc.mean()),  "std": float(acc.std())},
        "balanced_acc":      {"mean": float(bacc.mean()), "std": float(bacc.std())},
        "majority_baseline": majority,
        "class_counts":      counts.tolist(),
        "n":                 len(y),
    }


def probe_per_bin(X, y_dist, y_bins):
    """Train a separate Ridge probe within each distance bin (same as run_probe_sweep)."""
    results = {}
    for b, (name, width) in enumerate(zip(BIN_NAMES, BIN_WIDTHS)):
        mask  = y_bins == b
        n_bin = int(mask.sum())
        if n_bin < 2:
            results[name] = {"n": n_bin, "note": "too few samples"}
            continue
        X_b, y_b = X[mask], y_dist[mask]
        n_folds  = min(N_SPLITS, n_bin)
        kf_b     = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        pipe     = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
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


def _print_per_bin_tables(label_name, per_bin_rows, y_dist, y_bins):
    LAYER_W = 56
    COL_W   = 18

    naive_maes = {}
    for b, name in enumerate(BIN_NAMES):
        mask = y_bins == b
        naive_maes[name] = float(np.abs(y_dist[mask] - y_dist[mask].mean()).mean()) if mask.any() else float("nan")

    header = f"  {'LAYER':<{LAYER_W}}" + "".join(f"{n:^{COL_W}}" for n in BIN_NAMES)
    sep    = "  " + "-" * (LAYER_W + COL_W * len(BIN_NAMES))

    print(f"\n\n[{label_name}] Per-bin MAE (meters)")
    print(f"  sample counts: " + "  ".join(f"{n}(n={(y_bins==b).sum()})" for b, n in enumerate(BIN_NAMES)))
    print(f"  naive MAE baselines: " + "  ".join(f"{n}={naive_maes[n]:.3f}m" for n in BIN_NAMES))
    print(header); print(sep)
    for key, pb in per_bin_rows:
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            cols.append(f"{'<2 samples':^{COL_W}}" if "note" in d
                        else f"{d['mae']['mean']:.3f}+-{d['mae']['std']:.3f}m".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))

    print(f"\n\n[{label_name}] Per-bin relative MAE (MAE / bin_width)")
    print(f"  bin widths: " + "  ".join(f"{n}={w:.0f}m" for n, w in zip(BIN_NAMES, BIN_WIDTHS)))
    print(header); print(sep)
    for key, pb in per_bin_rows:
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            cols.append(f"{'<2 samples':^{COL_W}}" if "note" in d
                        else f"{d['relative_mae']:.3f}".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))

    print(f"\n\n[{label_name}] Per-bin R²")
    print(header); print(sep)
    for key, pb in per_bin_rows:
        cols = []
        for name in BIN_NAMES:
            d = pb.get(name, {})
            cols.append(f"{'<2 samples':^{COL_W}}" if "note" in d
                        else f"{d['r2']['mean']:+.3f}+-{d['r2']['std']:.3f}".center(COL_W))
        print(f"  {key:<{LAYER_W}}" + "".join(cols))


def run_dist_label(label_name, y_raw, h5):
    """Global probe + per-bin probe for distance regression labels."""
    valid = np.isfinite(y_raw) & (y_raw >= BIN_EDGES[0])
    y     = y_raw[valid].astype(np.float32)
    y_bins = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, len(BIN_NAMES) - 1).astype(int)
    n_valid = int(valid.sum())

    print(f"\n{'='*65}")
    print(f"Label: {label_name}  (n={n_valid})")
    counts = np.bincount(y_bins, minlength=len(BIN_NAMES))
    print(f"  Range: {y.min():.3f} – {y.max():.3f}  median={float(np.median(y)):.3f}")
    print(f"  Bin counts: " + "  ".join(f"{n}={c}" for n, c in zip(BIN_NAMES, counts)))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    print(f"\n  {'LAYER':<58} {'R²':>10}  {'MAE':>10}")
    print("  " + "-" * 84)

    layer_results = {}
    per_bin_rows  = []
    for key in HOOK_KEYS:
        if key not in h5:
            print(f"  {key:<58} MISSING")
            continue
        X = h5[key][:][valid].astype(np.float32)

        r  = probe_regression(X, y, kf)
        pb = probe_per_bin(X, y, y_bins)
        layer_results[key] = {**r, "per_bin": pb}
        per_bin_rows.append((key, pb))

        print(f"  {key:<58} "
              f"R²={r['r2']['mean']:+.3f}±{r['r2']['std']:.3f}  "
              f"MAE={r['mae']['mean']:.3f}±{r['mae']['std']:.3f}")

    _print_per_bin_tables(label_name, per_bin_rows, y, y_bins)

    return {"type": "dist_reg", "n_valid": n_valid, "bin_counts": counts.tolist(), "layers": layer_results}


def run_label(label_name, y_raw, h5, is_clf):
    """Global probe for heading (reg) and binary (clf) labels."""
    valid   = np.isfinite(y_raw)
    y       = y_raw[valid]
    n_valid = int(valid.sum())

    print(f"\n{'='*65}")
    print(f"Label: {label_name}  (n={n_valid})")

    if is_clf:
        y = y.astype(int)
        counts = np.bincount(y, minlength=2)
        print(f"  Classes: {counts.tolist()}  majority={counts.max()/n_valid:.3f}")
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    else:
        print(f"  Range: {y.min():.3f} – {y.max():.3f}  median={float(np.median(y)):.3f}")
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    print(f"\n  {'LAYER':<58} {'RESULT'}")
    print("  " + "-" * 88)

    layer_results = {}
    for key in HOOK_KEYS:
        if key not in h5:
            print(f"  {key:<58} MISSING")
            continue
        X = h5[key][:][valid].astype(np.float32)

        if is_clf:
            r    = probe_classification(X, y, cv)
            line = (f"acc={r['acc']['mean']:+.3f}±{r['acc']['std']:.3f}  "
                    f"bacc={r['balanced_acc']['mean']:+.3f}±{r['balanced_acc']['std']:.3f}  "
                    f"(majority={r['majority_baseline']:.3f})")
        else:
            r    = probe_regression(X, y, cv)
            line = (f"R²={r['r2']['mean']:+.3f}±{r['r2']['std']:.3f}  "
                    f"MAE={r['mae']['mean']:.3f}±{r['mae']['std']:.3f}")

        layer_results[key] = r
        print(f"  {key:<58} {line}")

    return {"type": "clf" if is_clf else "reg", "n_valid": n_valid, "layers": layer_results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",  default=H5_DEFAULT)
    p.add_argument("--npz", default=NPZ_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    args = p.parse_args()

    labels_npz = np.load(args.npz)
    print(f"Loaded labels from {args.npz}: {list(labels_npz.keys())}")

    all_results = {}
    with h5py.File(args.h5, "r") as h5:
        for label_name in DIST_REG_LABELS:
            if label_name not in labels_npz:
                print(f"[SKIP] {label_name} not in npz")
                continue
            all_results[label_name] = run_dist_label(label_name, labels_npz[label_name], h5)

        for label_name in OTHER_REG_LABELS:
            if label_name not in labels_npz:
                print(f"[SKIP] {label_name} not in npz")
                continue
            all_results[label_name] = run_label(label_name, labels_npz[label_name], h5, is_clf=False)

        for label_name in CLF_LABELS:
            if label_name not in labels_npz:
                print(f"[SKIP] {label_name} not in npz")
                continue
            all_results[label_name] = run_label(label_name, labels_npz[label_name], h5, is_clf=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
