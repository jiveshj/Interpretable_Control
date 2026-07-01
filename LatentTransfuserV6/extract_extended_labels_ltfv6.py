#!/usr/bin/env python
"""
extract_extended_labels_ltfv6.py

Reads the NAVSIM target files (transfuser_target.gz) pointed to by the
existing HDF5 and computes additional ground-truth labels for probing.
Saves a .npz file aligned row-for-row with the HDF5.

New labels (all in ego-centric frame, NaN where undefined):
  dist_2nd_nearest_any    : dist to 2nd closest valid agent, any direction
  dist_2nd_nearest_front  : dist to 2nd closest valid agent with X > 0
  has_vehicle_front       : 1.0 if any valid agent with X > 0, else 0.0
  nearest_front_heading   : heading (rad) of closest front agent; NaN if none
  same_lane_binary        : 1.0 if nearest front agent |heading| < pi/6 (~30 deg)
  opposing_lane_binary    : 1.0 if nearest front agent |heading| > 5*pi/6 (~150 deg)

Heading convention: NAVSIM stores agent bounding boxes in the ego-centric
frame. Heading = 0 means the agent faces the same direction as the ego.
Heading = ±pi means the agent faces opposite (oncoming traffic).
If this differs from your dataset, adjust SAME_LANE_DEG / OPP_LANE_DEG.

Usage:
    python extract_extended_labels_ltfv6.py
    python extract_extended_labels_ltfv6.py --h5 /path/to/latents.h5
"""

import argparse
import gzip
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm

H5_DEFAULT  = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
OUT_DEFAULT = "/jet/home/jjain2/Interpretable_Control/extended_labels_ltfv6.npz"

SAME_LANE_DEG = 30.0   # heading within ±30° of 0 → same-direction traffic
OPP_LANE_DEG  = 150.0  # heading beyond ±150° from 0 → oncoming traffic


def compute_extended_labels(target):
    """Compute all extended labels for one NAVSIM sample."""
    agent_states = target["agent_states"]
    agent_labels = target["agent_labels"]
    if hasattr(agent_states, "numpy"):
        agent_states = agent_states.numpy()
    if hasattr(agent_labels, "numpy"):
        agent_labels = agent_labels.numpy()

    valid = agent_labels.astype(bool)
    out = {
        "dist_2nd_nearest_any":   np.nan,
        "dist_2nd_nearest_front": np.nan,
        "has_vehicle_front":      0.0,
        "nearest_front_heading":  np.nan,
        "same_lane_binary":       np.nan,
        "opposing_lane_binary":   np.nan,
    }

    if not valid.any():
        return out

    # agent_states columns: [X, Y, heading, length, width]
    # X = forward, Y = left in ego frame
    xy      = agent_states[valid, :2]   # (N, 2)
    heading = agent_states[valid, 2]    # (N,) in radians, ego frame
    dists   = np.linalg.norm(xy, axis=1)

    # 2nd nearest any direction
    if len(dists) >= 2:
        sorted_idx = np.argsort(dists)
        out["dist_2nd_nearest_any"] = float(dists[sorted_idx[1]])

    # Front-half agents (X > 0)
    front_mask = xy[:, 0] > 0.0
    out["has_vehicle_front"] = float(front_mask.any())

    if front_mask.any():
        front_dists   = dists[front_mask]
        front_heading = heading[front_mask]

        nearest_idx = int(np.argmin(front_dists))
        h = float(front_heading[nearest_idx])
        out["nearest_front_heading"] = h

        thresh_same = np.deg2rad(SAME_LANE_DEG)
        thresh_opp  = np.deg2rad(OPP_LANE_DEG)
        out["same_lane_binary"]     = float(abs(h) < thresh_same)
        out["opposing_lane_binary"] = float(abs(h) > thresh_opp)

        if len(front_dists) >= 2:
            sorted_front = np.argsort(front_dists)
            out["dist_2nd_nearest_front"] = float(front_dists[sorted_front[1]])

    return out


LABEL_KEYS = [
    "dist_2nd_nearest_any",
    "dist_2nd_nearest_front",
    "has_vehicle_front",
    "nearest_front_heading",
    "same_lane_binary",
    "opposing_lane_binary",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",  default=H5_DEFAULT,
                   help="Path to existing latents HDF5")
    p.add_argument("--out", default=OUT_DEFAULT,
                   help="Output .npz path")
    args = p.parse_args()

    with h5py.File(args.h5, "r") as h5:
        paths = [
            (v.decode() if isinstance(v, bytes) else v)
            for v in h5["sample_path"][:]
        ]

    n = len(paths)
    print(f"Extracting extended labels for {n} samples from {args.h5}")
    arrays = {k: np.full(n, np.nan, dtype=np.float32) for k in LABEL_KEYS}

    missing = 0
    for i, feat_path in enumerate(tqdm(paths, desc="extracting")):
        target_path = feat_path.replace("transfuser_feature.gz", "transfuser_target.gz")
        try:
            with gzip.open(target_path, "rb") as f:
                target = pickle.load(f)
        except FileNotFoundError:
            missing += 1
            continue

        row = compute_extended_labels(target)
        for k in LABEL_KEYS:
            arrays[k][i] = row[k]

    print(f"\nMissing target files : {missing} / {n}")
    print(f"\n{'LABEL':<30} {'N_valid':>8}  {'mean':>10}  {'NaN%':>7}")
    print("-" * 60)
    for k in LABEL_KEYS:
        a = arrays[k]
        finite = a[np.isfinite(a)]
        nan_pct = 100.0 * np.isnan(a).sum() / n
        mean_str = f"{finite.mean():.4f}" if len(finite) else "N/A"
        print(f"  {k:<28} {len(finite):>8}  {mean_str:>10}  {nan_pct:>6.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
