#!/usr/bin/env python
"""
collect_data_ltfv6.py

Captures activations from Latent TransFuser v6 at 18 hook points spanning
the image encoder, lidar encoder, cross-modal fusion transformers, and
planning decoder. Saves per-layer pooled activations along with ground-truth
distance labels (and the model's own predicted trajectory) to HDF5 for
downstream linear probing.

Usage:
    python collect_data_ltfv6.py \\
        --data_root /ocean/projects/cis250201p/jjain2/ltfv6/data \\
        --out       /ocean/projects/cis250201p/jjain2/data/latents_ltfv6.h5 \\
        --grid      1

Optional:
    --max_samples N    limit to first N samples (for quick tests)
    --grid {1,2,4}      spatial pooling grid for 4D conv activations (default 1 =
                         plain GAP, the original behavior; 2/4 preserve coarse
                         spatial structure via adaptive_avg_pool2d -- see make_hook)

Activation pooling:
    - 4D conv feature maps (image/lidar encoders): pooled over H,W to a grid x grid
      grid (grid=1 -> plain GAP, i.e. (C,); grid>1 -> (C*grid*grid,))
    - tuple outputs from cross-modal transformers: pool both branches, concatenate
    - 3D token sequences (planning decoder layers): mean over tokens -> (D,),
      unaffected by --grid (no spatial H,W to pool)

Distance labels (see compute_distance_labels):
    - dist_nearest_any             : nearest valid agent, any direction
    - dist_nearest_front           : nearest agent whose bounding box intersects the
                                      ego's GT future-trajectory corridor; arc-length
                                      to the intersection point
    - dist_nearest_front_halfplane : the OLD X>0 half-plane label, kept for comparison

Also caches the model's own predicted trajectory (pred_future_waypoints,
pred_headings) -- free from the existing forward pass, enables a
behavior-vs-representation comparison. Coordinate-flipped from CARLA's
left-handed frame to ISO 8855 to match the GT labels above (see model card
ln2697/ltfv6-navsim: "predicts in CARLA's left-handed frame").
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

LTFV6_DIR = "/ocean/projects/cis250201p/jjain2/ltfv6"
sys.path.insert(0, LTFV6_DIR)

from ltfv6 import load_tf, NavsimData, TrainingConfig  # noqa: E402

# Force fp32 for inference. The config's torch_float_type was meant for training
# (under autocast); without autocast we'd get a weight/activation dtype mismatch.
TrainingConfig.torch_float_type = property(lambda self: torch.float32)

class RobustNavsimData(NavsimData):
    """NavsimData that handles camera_feature stored as either bytes or uint8 tensor."""
    def __getitem__(self, index):
        import gzip, pickle
        feature_path = self.feature[index]
        with gzip.open(feature_path, "rb") as f:
            feature = pickle.load(f)

        cam = feature["camera_feature"]
        if isinstance(cam, torch.Tensor):
            cam = cam.numpy()          # uint8 tensor → numpy array, cv2 can decode directly
        rgb = cv2.imdecode(cam, cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))

        return {
            "rgb": rgb,
            "command": feature["status_feature"][:4],
            "speed": np.linalg.norm(feature["status_feature"][4:6]),
            "acceleration": np.linalg.norm(feature["status_feature"][6:8]),
        }

# 18 hook points
HOOK_LAYERS = [
    *[f"backbone.image_encoder.layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone.lidar_encoder.layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone.transformers.{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder.transformer_decoder.layers.{i}" for i in range(6)],
]


def _pool_4d(x, grid):
    """(B,C,H,W) -> (B, C*grid*grid). grid=1 reduces to plain GAP, i.e. the
    original behavior. grid>1 preserves coarse spatial structure -- distance
    to a lead vehicle is an inherently spatial cue (apparent size + image
    position), which GAP (grid=1) destroys."""
    if grid == 1:
        return x.mean(dim=[2, 3]).detach().cpu().float().numpy()
    pooled = F.adaptive_avg_pool2d(x, (grid, grid))  # (B, C, grid, grid)
    return pooled.flatten(1).detach().cpu().float().numpy()  # (B, C*grid*grid)


def make_hook(name, store, grid=1):
    """Forward hook: pools activation, stores it in `store[name]` as numpy float32.

    grid: spatial pooling grid size for 4D (B,C,H,W) conv outputs (see
    _pool_4d). 3D token-sequence outputs (planning decoder) have no H,W and
    are always mean-pooled over tokens regardless of `grid`.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            # backbone.transformers.* returns (image_features, lidar_features), each (B, C, H, W)
            img_out, lidar_out = output
            img_pooled = _pool_4d(img_out, grid)
            lidar_pooled = _pool_4d(lidar_out, grid)
            store[name] = np.concatenate([img_pooled, lidar_pooled], axis=1)
        elif output.dim() == 4:
            # Conv feature map (B, C, H, W) -> pool -> (B, C) or (B, C*grid*grid)
            store[name] = _pool_4d(output, grid)
        elif output.dim() == 3:
            # Token sequence (B, N, D) -> mean over tokens -> (B, D); no spatial
            # grid concept here, unaffected by --grid
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


# nuPlan's standard ego vehicle (Chrysler Pacifica, get_pacifica_parameters()
# in nuplan.common.actor_state.vehicle_parameters -- not importable here since
# the nuplan-devkit package isn't installed in this env, so the published
# constants are used directly). A physical vehicle dimension, not a tuned
# hyperparameter: this is "would the ego's own footprint, swept along its GT
# path, touch this agent's box" -- not an arbitrary lateral-distance cutoff.
EGO_WIDTH = 2.297
EGO_HALF_WIDTH = EGO_WIDTH / 2.0


def _rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return v @ R.T


def _box_corners(x, y, heading, length, width):
    """4 corners of an oriented box, ego frame, CCW/CW order (order doesn't
    matter for the edge-walk used below)."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return _rotate(local, heading) + np.array([x, y])


def _point_in_convex_polygon(pt, poly):
    n = len(poly)
    sign = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        s = np.sign(cross)
        if s == 0:
            continue
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _segment_box_distance(p1, p2, box):
    """Min distance from segment p1-p2 to an oriented box (4 corners). 0 if
    either endpoint is inside the box or the segment crosses any edge."""
    if _point_in_convex_polygon(p1, box) or _point_in_convex_polygon(p2, box):
        return 0.0
    min_d = np.inf
    for i in range(4):
        b1, b2 = box[i], box[(i + 1) % 4]
        if _segments_intersect(p1, p2, b1, b2):
            return 0.0
        min_d = min(min_d,
                    _point_segment_dist(b1, p1, p2), _point_segment_dist(b2, p1, p2),
                    _point_segment_dist(p1, b1, b2), _point_segment_dist(p2, b1, b2))
    return float(min_d)


def compute_distance_labels(target):
    """Compute GT distance from agent_states + agent_labels + trajectory.

    agent_states columns (NavSimBoundingBoxIndex): [X, Y, heading, length, width]
    X = forward, Y = left (ego frame). 30 slots, padded (vehicles only,
    nearest-30 by raw distance -- see transfuser_features.py); agent_labels
    masks valid ones. `trajectory` is the ego's GT future path, (8,3) =
    [x, y, heading] per waypoint, same ego frame, already present in
    transfuser_target.gz.

    Returns:
        nearest_any             : Euclidean dist to closest valid agent (any
                                   direction); NaN if none.
        nearest_front           : nearest agent whose bounding box intersects
                                   the ego's GT future-trajectory corridor
                                   (buffered by EGO_HALF_WIDTH -- would the
                                   ego's own footprint touch this box).
                                   Distance = arc-length along the trajectory
                                   to the first intersection point. NaN if no
                                   agent's box intersects the corridor.
                                   Replaces the old X>0 half-plane mask, which
                                   conflates any forward vehicle -- a car in a
                                   cross street, one in an adjacent lane far
                                   to the side -- with the true lead vehicle.
        nearest_front_halfplane : the OLD label (X>0 half-plane), kept
                                   separately so the two can be compared.
    """
    agent_states = target["agent_states"]
    agent_labels = target["agent_labels"]
    trajectory = target["trajectory"]
    if isinstance(agent_states, torch.Tensor):
        agent_states = agent_states.numpy()
    if isinstance(agent_labels, torch.Tensor):
        agent_labels = agent_labels.numpy()
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()

    valid = agent_labels.astype(bool)
    if not valid.any():
        return np.nan, np.nan, np.nan

    xy = agent_states[valid, :2]  # (N_valid, 2)
    nearest_any = float(np.linalg.norm(xy, axis=1).min())

    front_mask = xy[:, 0] > 0.0
    nearest_front_halfplane = (float(np.linalg.norm(xy[front_mask], axis=1).min())
                                if front_mask.any() else np.nan)

    # ego path: origin (current pose) + 8 GT future waypoints, ego frame
    path = np.concatenate([np.zeros((1, 2)), trajectory[:, :2].astype(np.float64)], axis=0)
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    best_arc = np.inf
    for x, y, heading, length, width in agent_states[valid]:
        box = _box_corners(float(x), float(y), float(heading), float(length), float(width))
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            if _segment_box_distance(p1, p2, box) > EGO_HALF_WIDTH:
                continue
            # sub-sample within the segment for finer arc-length resolution
            # than the raw ~4-5m waypoint spacing
            sub_t = np.linspace(0.0, 1.0, 10)
            hit_t = 1.0
            for t in sub_t:
                pt = p1 + t * (p2 - p1)
                inside = _point_in_convex_polygon(pt, box)
                edge_d = min(_point_segment_dist(pt, box[j], box[(j + 1) % 4]) for j in range(4))
                if inside or edge_d <= EGO_HALF_WIDTH:
                    hit_t = t
                    break
            arc = cum_arc[i] + hit_t * seg_lens[i]
            best_arc = min(best_arc, arc)
            break  # first (nearest-arc) hit along the path is enough for this agent

    nearest_front = float(best_arc) if np.isfinite(best_arc) else np.nan
    return nearest_any, nearest_front, nearest_front_halfplane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=f"/ocean/projects/cis250201p/jjain2/lead/data/navsim_training_cache/navtest",
                        help="Directory tree containing transfuser_feature.gz / _target.gz")
    parser.add_argument("--out", default=f"/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5",
                        help="Output HDF5 path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on samples processed (for testing)")
    parser.add_argument("--grid", type=int, default=1, choices=[1, 2, 4],
                        help="Spatial pooling grid for 4D conv activations: "
                             "1 = plain GAP (original behavior), 2/4 = "
                             "adaptive_avg_pool2d(grid,grid) then flatten, "
                             "preserving coarse spatial structure. Does not "
                             "affect 3D token-sequence (planning decoder) outputs.")
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
    dataset = RobustNavsimData(args.data_root, cfg)
    n_total = len(dataset)
    n = min(args.max_samples, n_total) if args.max_samples else n_total
    print(f"[setup] {n_total} samples in {args.data_root}; processing {n}")

    # Register hooks
    activations = {}
    module_dict = dict(model.named_modules())
    for name in HOOK_LAYERS:
        if name not in module_dict:
            raise RuntimeError(f"Module not found: {name}")
        module_dict[name].register_forward_hook(make_hook(name, activations, grid=args.grid))
    print(f"[setup] registered {len(HOOK_LAYERS)} hooks (grid={args.grid})")

    # Warmup forward to discover activation widths per layer (also gives us
    # `out` to discover the predicted-trajectory width for HDF5 allocation)
    print("[setup] warmup forward to discover activation widths")
    batch0 = make_batch(dataset[0])
    with torch.no_grad():
        out = model(batch0)

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
        h5_dist_front_hp = h5.create_dataset("dist_nearest_front_halfplane", shape=(n,), dtype=np.float32)
        h5_paths = h5.create_dataset("sample_path", shape=(n,), dtype=h5py.string_dtype())

        # cached model behavior (free from the existing forward pass) --
        # widths discovered from the warmup forward above
        n_wp = int(out.pred_future_waypoints.shape[1]) if out.pred_future_waypoints is not None else 8
        h5_pred_wp = h5.create_dataset("pred_future_waypoints", shape=(n, n_wp, 2), dtype=np.float32)
        h5_pred_hd = h5.create_dataset("pred_headings", shape=(n, n_wp), dtype=np.float32)

        # main loop
        for i in tqdm(range(n), desc="collecting"):
            sample = dataset[i]
            batch = make_batch(sample)
            with torch.no_grad():
                out = model(batch)

            for name in HOOK_LAYERS:
                h5_acts[name][i] = activations[name][0]

            feat_path = dataset.feature[i]
            target_path = feat_path.replace("transfuser_feature.gz", "transfuser_target.gz")
            try:
                with gzip.open(target_path, "rb") as f:
                    target = pickle.load(f)
                d_any, d_front, d_front_hp = compute_distance_labels(target)
            except FileNotFoundError:
                d_any, d_front, d_front_hp = np.nan, np.nan, np.nan

            h5_dist_any[i] = d_any
            h5_dist_front[i] = d_front
            h5_dist_front_hp[i] = d_front_hp
            h5_paths[i] = feat_path

            # CRITICAL coordinate flip: the model predicts in CARLA's
            # left-handed frame (x-forward, y-right), but the GT labels above
            # (agent_states/trajectory, from transfuser_target.gz) are in
            # ISO 8855 (y-left). Without this flip, pred_future_waypoints and
            # the distance labels would silently disagree about which way is
            # left -- per the ln2697/ltfv6-navsim model card.
            wp = out.pred_future_waypoints[0].detach().cpu().float().numpy().copy()
            hd = out.pred_headings[0].detach().cpu().float().numpy().copy()
            wp[:, 1] *= -1.0
            hd *= -1.0
            h5_pred_wp[i] = wp
            h5_pred_hd[i] = hd

        # metadata
        h5.attrs["model_path"] = f"{LTFV6_DIR}/model_0060.pth"
        h5.attrs["data_root"] = args.data_root
        h5.attrs["n_samples"] = n
        h5.attrs["hook_layers"] = "\n".join(HOOK_LAYERS)
        h5.attrs["split"] = "navtest"
        h5.attrs["pool_grid"] = args.grid
        h5.attrs["pred_trajectory_coord_flip"] = "y *= -1, heading *= -1 (CARLA left-handed -> ISO 8855)"


    # quick summary
    with h5py.File(args.out, "r") as h5:
        d_any = h5["dist_nearest_any"][:]
        d_front = h5["dist_nearest_front"][:]
        d_front_hp = h5["dist_nearest_front_halfplane"][:]
        finite_any = d_any[np.isfinite(d_any)]
        finite_front = d_front[np.isfinite(d_front)]
        finite_front_hp = d_front_hp[np.isfinite(d_front_hp)]
        print(f"\n=== Distance label summary ===")
        if len(finite_any):
            print(f"nearest_any             n={len(finite_any):5d}  "
                  f"min={finite_any.min():6.2f}  median={np.median(finite_any):6.2f}  "
                  f"max={finite_any.max():6.2f}")
        if len(finite_front):
            print(f"nearest_front (corridor) n={len(finite_front):5d}  "
                  f"min={finite_front.min():6.2f}  median={np.median(finite_front):6.2f}  "
                  f"max={finite_front.max():6.2f}")
        if len(finite_front_hp):
            print(f"nearest_front (halfplane) n={len(finite_front_hp):5d}  "
                  f"min={finite_front_hp.min():6.2f}  median={np.median(finite_front_hp):6.2f}  "
                  f"max={finite_front_hp.max():6.2f}")
        # how often do the two labels actually disagree on which agent is "the lead"?
        both = np.isfinite(d_front) & np.isfinite(d_front_hp)
        if both.any():
            diff = np.abs(d_front[both] - d_front_hp[both])
            agree = (diff < 0.5).sum()
            print(f"corridor vs halfplane: agree (within 0.5m) on {agree}/{both.sum()} "
                  f"samples where both are defined; mean |diff|={diff.mean():.2f}m")
        pred_wp = h5["pred_future_waypoints"][:]
        print(f"\npred_future_waypoints: shape={pred_wp.shape}  "
              f"nan={np.isnan(pred_wp).any()}  "
              f"x range=[{pred_wp[...,0].min():.2f}, {pred_wp[...,0].max():.2f}]  "
              f"y range=[{pred_wp[...,1].min():.2f}, {pred_wp[...,1].max():.2f}]")
    print(f"\n[done] wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()