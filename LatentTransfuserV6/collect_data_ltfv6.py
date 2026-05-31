#!/usr/bin/env python
"""
collect_data_ltfv6.py

Captures activations from Latent TransFuser v6 at 18 hook points spanning
the image encoder, lidar encoder, cross-modal fusion transformers, and
planning decoder. Saves per-layer global-average-pooled activations along
with ground-truth distance labels to HDF5 for downstream linear probing.

Usage:
    python collect_data_ltfv6.py \\
        --data_root /ocean/projects/cis250201p/jjain2/ltfv6/data \\
        --out       /ocean/projects/cis250201p/jjain2/data/latents_ltfv6.h5

Optional:
    --max_samples N   limit to first N samples (for quick tests)

Activation pooling:
    - 4D conv feature maps (image/lidar encoders): global-average-pooled over H,W -> (C,)
    - tuple outputs from cross-modal transformers: pool both branches, concatenate -> (C_img + C_lidar,)
    - 3D token sequences (planning decoder layers): mean over tokens -> (D,)
"""

import argparse
import gzip
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

LTFV6_DIR = "/ocean/projects/cis250201p/jjain2/ltfv6"
sys.path.insert(0, LTFV6_DIR)

from ltfv6 import load_tf, NavsimData, TrainingConfig  # noqa: E402

# Force fp32 for inference. The config's torch_float_type was meant for training
# (under autocast); without autocast we'd get a weight/activation dtype mismatch.
TrainingConfig.torch_float_type = property(lambda self: torch.float32)


# 18 hook points
HOOK_LAYERS = [
    *[f"backbone.image_encoder.layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone.lidar_encoder.layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone.transformers.{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder.transformer_decoder.layers.{i}" for i in range(6)],
]


def make_hook(name, store):
    """Forward hook: pools activation, stores it in `store[name]` as numpy float32."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            # backbone.transformers.* returns (image_features, lidar_features), each (B, C, H, W)
            img_out, lidar_out = output
            img_pooled = img_out.mean(dim=[2, 3]).detach().cpu().float().numpy()
            lidar_pooled = lidar_out.mean(dim=[2, 3]).detach().cpu().float().numpy()
            store[name] = np.concatenate([img_pooled, lidar_pooled], axis=1)
        elif output.dim() == 4:
            # Conv feature map (B, C, H, W) -> GAP -> (B, C)
            store[name] = output.mean(dim=[2, 3]).detach().cpu().float().numpy()
        elif output.dim() == 3:
            # Token sequence (B, N, D) -> mean over tokens -> (B, D)
            store[name] = output.mean(dim=1).detach().cpu().float().numpy()
        else:
            raise RuntimeError(f"Unexpected output shape at {name}: {tuple(output.shape)}")
    return hook


def make_batch(sample):
    """Wrap a NavsimData sample into a batch-of-1 with proper tensor types.

    NavsimData returns:
        rgb           : numpy uint8 (3, H, W)
        command       : torch.Tensor (4,) -- from status_feature[:4]
        speed         : numpy scalar      -- ||(v_x, v_y)||
        acceleration  : numpy scalar      -- ||(a_x, a_y)||
    Model wants device transfer + correct dtype; both handled inside the
    backbone/decoder via `.to(device, dtype=torch_float_type)`.
    """
    return {
        "rgb":          torch.as_tensor(sample["rgb"]).unsqueeze(0).float(),
        "command":      torch.as_tensor(sample["command"]).unsqueeze(0).float(),
        "speed":        torch.as_tensor(sample["speed"]).reshape(1).float(),
        "acceleration": torch.as_tensor(sample["acceleration"]).reshape(1).float(),
    }


def compute_distance_labels(target):
    """Compute GT distance from agent_states + agent_labels.

    agent_states columns (NavSimBoundingBoxIndex): [X, Y, heading, length, width]
    X = forward, Y = left (ego frame). 30 slots, padded; agent_labels masks valid ones.

    Returns:
        nearest_any   : Euclidean dist to closest valid agent (any direction); NaN if none.
        nearest_front : same, restricted to agents with X > 0; NaN if none.
    """
    agent_states = target["agent_states"]
    agent_labels = target["agent_labels"]
    if isinstance(agent_states, torch.Tensor):
        agent_states = agent_states.numpy()
    if isinstance(agent_labels, torch.Tensor):
        agent_labels = agent_labels.numpy()

    valid = agent_labels.astype(bool)
    if not valid.any():
        return np.nan, np.nan

    xy = agent_states[valid, :2]  # (N_valid, 2)
    dists = np.linalg.norm(xy, axis=1)
    nearest_any = float(dists.min())

    front_mask = xy[:, 0] > 0.0
    if front_mask.any():
        nearest_front = float(np.linalg.norm(xy[front_mask], axis=1).min())
    else:
        nearest_front = np.nan
    return nearest_any, nearest_front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=f"{LTFV6_DIR}/data",
                        help="Directory tree containing transfuser_feature.gz / _target.gz")
    parser.add_argument("--out", default=f"{LTFV6_DIR}/latents_ltfv6.h5",
                        help="Output HDF5 path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on samples processed (for testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    print(f"[setup] loading model from {LTFV6_DIR}/model_0060.pth")
    model = load_tf(f"{LTFV6_DIR}/model_0060.pth", device)
    model.eval()

    # CRITICAL: planning decoder is gated by config.use_planning_decoder, default False.
    # Without this, planning_decoder.* hooks never fire.
    model.config.use_planning_decoder = True
    print("[setup] model.config.use_planning_decoder = True")

    # Build dataset with the same config the model was built with
    with open(f"{LTFV6_DIR}/config.json") as f:
        cfg_dict = json.load(f)
    cfg = TrainingConfig(cfg_dict)
    dataset = NavsimData(args.data_root, cfg)
    n_total = len(dataset)
    n = min(args.max_samples, n_total) if args.max_samples else n_total
    print(f"[setup] {n_total} samples in {args.data_root}; processing {n}")

    # Register hooks
    activations = {}
    module_dict = dict(model.named_modules())
    for name in HOOK_LAYERS:
        if name not in module_dict:
            raise RuntimeError(f"Module not found: {name}")
        module_dict[name].register_forward_hook(make_hook(name, activations))
    print(f"[setup] registered {len(HOOK_LAYERS)} hooks")

    # Warmup forward to discover activation widths per layer
    print("[setup] warmup forward to discover activation widths")
    batch0 = make_batch(dataset[0])
    with torch.no_grad():
        _ = model(batch0)

    widths = {}
    missing = []
    for name in HOOK_LAYERS:
        if name in activations:
            widths[name] = activations[name].shape[1]
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"Hooks did not fire for: {missing}\n"
            "If these are planning_decoder layers, double-check that "
            "model.config.use_planning_decoder is True."
        )

    print(f"{'LAYER':<60} {'WIDTH':>6}")
    for name, w in widths.items():
        print(f"  {name:<58} {w:>6}")

    # Allocate HDF5 file
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.out, "w") as h5:
        # one dataset per hook
        h5_acts = {}
        for name, w in widths.items():
            safe = name.replace(".", "_")
            h5_acts[name] = h5.create_dataset(
                safe,
                shape=(n, w),
                dtype=np.float32,
                chunks=(min(64, n), w),
                compression="gzip",
                compression_opts=4,
            )

        # labels
        h5_dist_any = h5.create_dataset("dist_nearest_any", shape=(n,), dtype=np.float32)
        h5_dist_front = h5.create_dataset("dist_nearest_front", shape=(n,), dtype=np.float32)
        h5_paths = h5.create_dataset("sample_path", shape=(n,), dtype=h5py.string_dtype())

        # main loop
        for i in tqdm(range(n), desc="collecting"):
            sample = dataset[i]
            batch = make_batch(sample)
            with torch.no_grad():
                _ = model(batch)

            for name in HOOK_LAYERS:
                h5_acts[name][i] = activations[name][0]

            feat_path = dataset.feature[i]
            target_path = feat_path.replace("transfuser_feature.gz", "transfuser_target.gz")
            try:
                with gzip.open(target_path, "rb") as f:
                    target = pickle.load(f)
                d_any, d_front = compute_distance_labels(target)
            except FileNotFoundError:
                d_any, d_front = np.nan, np.nan

            h5_dist_any[i] = d_any
            h5_dist_front[i] = d_front
            h5_paths[i] = feat_path

        # metadata
        h5.attrs["model_path"] = f"{LTFV6_DIR}/model_0060.pth"
        h5.attrs["data_root"] = args.data_root
        h5.attrs["n_samples"] = n
        h5.attrs["hook_layers"] = "\n".join(HOOK_LAYERS)

    # quick summary
    with h5py.File(args.out, "r") as h5:
        d_any = h5["dist_nearest_any"][:]
        d_front = h5["dist_nearest_front"][:]
        finite_any = d_any[np.isfinite(d_any)]
        finite_front = d_front[np.isfinite(d_front)]
        print(f"\n=== Distance label summary ===")
        if len(finite_any):
            print(f"nearest_any   n={len(finite_any):5d}  "
                  f"min={finite_any.min():6.2f}  median={np.median(finite_any):6.2f}  "
                  f"max={finite_any.max():6.2f}")
        if len(finite_front):
            print(f"nearest_front n={len(finite_front):5d}  "
                  f"min={finite_front.min():6.2f}  median={np.median(finite_front):6.2f}  "
                  f"max={finite_front.max():6.2f}")
    print(f"\n[done] wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()