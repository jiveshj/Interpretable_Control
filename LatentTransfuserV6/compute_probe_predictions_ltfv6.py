#!/usr/bin/env python
"""
compute_probe_predictions_ltfv6.py

Computes out-of-fold probe predictions for every valid sample under both
leaky KFold and grouped-by-log GroupKFold, then saves a JSON used by
make_probe_viz_html_ltfv6.py to build the qualitative HTML.

Intended to run as a cluster job (takes ~5-10 min for 3 layers).

Usage:
    python compute_probe_predictions_ltfv6.py
    python compute_probe_predictions_ltfv6.py --layers backbone_transformers_2 backbone_image_encoder_layer1
    python compute_probe_predictions_ltfv6.py --out /path/probe_predictions.json
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT  = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
NPZ_DEFAULT = "/jet/home/jjain2/Interpretable_Control/extended_labels_ltfv6.npz"
OUT_DEFAULT = "/jet/home/jjain2/Interpretable_Control/probe_predictions_ltfv6.json"

BIN_EDGES = [3.0, 5.0, 10.0, 20.0, 40.0]
BIN_NAMES = ["3-5m", "5-10m", "10-20m", "20-40m+"]
ALPHAS    = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

DEFAULT_LAYERS = [
    "backbone_image_encoder_layer1",
    "backbone_transformers_2",
    "planning_decoder_transformer_decoder_layers_5",
]


def log_id_from_path(p):
    p = p.decode() if isinstance(p, bytes) else p
    return p.split("/")[-3]


def make_pipe():
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",     default=H5_DEFAULT)
    p.add_argument("--npz",    default=NPZ_DEFAULT)
    p.add_argument("--label",  default="dist_nearest_front")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--out",    default=OUT_DEFAULT)
    args = p.parse_args()

    print(f"Loading labels and paths from {args.h5}")
    with h5py.File(args.h5, "r") as h5:
        y_raw  = h5[args.label][:]
        paths  = [x.decode() if isinstance(x, bytes) else x for x in h5["sample_path"][:]]
        groups_all = np.array([log_id_from_path(x) for x in paths])

    valid  = np.isfinite(y_raw) & (y_raw >= BIN_EDGES[0])
    y      = y_raw[valid].astype(np.float32)
    y_bins = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, 3).astype(int)
    grps   = groups_all[valid]
    vpaths = [paths[i] for i in np.where(valid)[0]]
    n      = int(valid.sum())
    print(f"  Valid samples: {n}  |  Distinct logs: {len(np.unique(grps))}")

    # 2nd-nearest distance for traffic labelling
    dist_2nd = None
    if os.path.exists(args.npz):
        npz = np.load(args.npz)
        if "dist_2nd_nearest_any" in npz:
            dist_2nd = npz["dist_2nd_nearest_any"][valid]
            print(f"  Loaded dist_2nd_nearest_any")

    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    predictions = {}   # layer -> {leaky: [...], grouped: [...]}

    with h5py.File(args.h5, "r") as h5:
        for layer in args.layers:
            if layer not in h5:
                print(f"  [SKIP] {layer} not in H5")
                continue
            t0 = time.time()
            X = h5[layer][:][valid].astype(np.float32)
            print(f"  [{layer}] shape={X.shape}  running cross_val_predict...", flush=True)

            pred_l = cross_val_predict(make_pipe(), X, y, cv=kf)
            pred_g = cross_val_predict(make_pipe(), X, y, cv=gkf, groups=grps)
            predictions[layer] = {
                "leaky":   pred_l.tolist(),
                "grouped": pred_g.tolist(),
            }
            print(f"    done in {time.time()-t0:.1f}s")

    # Assemble per-sample records
    samples = []
    for i in range(n):
        rec = {
            "idx":       i,
            "path":      vpaths[i],
            "true_dist": float(y[i]),
            "bin":       BIN_NAMES[int(y_bins[i])],
            "log":       grps[i],
        }
        if dist_2nd is not None:
            rec["dist_2nd"] = float(dist_2nd[i]) if np.isfinite(dist_2nd[i]) else None
        for layer, preds in predictions.items():
            rec[f"pred_leaky_{layer}"]   = round(float(preds["leaky"][i]),   4)
            rec[f"pred_grouped_{layer}"] = round(float(preds["grouped"][i]), 4)
        samples.append(rec)

    out = {
        "label":   args.label,
        "n":       n,
        "layers":  args.layers,
        "bin_names": BIN_NAMES,
        "samples": samples,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {n} sample predictions to {args.out}")


if __name__ == "__main__":
    main()
