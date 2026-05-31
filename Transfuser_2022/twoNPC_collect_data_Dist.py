"""
collect_data_two_npc.py
-----------------------
Collects TransFuser activations with TWO controlled NPCs at different distances.

NPC1: placed directly ahead in ego's lane        → nearest vehicle
NPC2: placed ahead in adjacent lane (right)      → second nearest vehicle

Labels saved per sample:
  gt_distance_1  : distance to NPC1 (nearest)
  gt_distance_2  : distance to NPC2 (second nearest)
  bin_label_1    : distance bin for NPC1
  bin_label_2    : distance bin for NPC2

Design:
  - NPC1 sampled from NEAR_BINS  (0.5–10m)  — always the closer one
  - NPC2 sampled from FAR_BINS   (10–50m)   — always the farther one
  - Sample only accepted if:
      (a) gt_distance_1 falls within its intended bin
      (b) gt_distance_2 falls within its intended bin
      (c) gt_distance_1 < gt_distance_2  (ordering guarantee)
  - NPC2 placed at LANE_WIDTH lateral offset so both are visible to camera

Usage:
    ./CarlaUE4.sh -opengl -RenderOffScreen &
    sleep 15
    python collect_data_two_npc.py \
        --model-ckpt /path/to/transfuser \
        --output     /path/to/latents_two_npc.h5 \
        --n-steps    5000
"""

import argparse
import sys
import os
import time
import numpy as np
import h5py
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2
from PIL import Image
from copy import deepcopy

try:
    import carla
except ImportError:
    raise ImportError(
        "CARLA Python API not found. Make sure PYTHONPATH includes:\n"
        "  $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg\n"
        "  $CARLA_ROOT/PythonAPI/carla"
    )

TRANSFUSER_ROOT = Path(os.environ.get(
    "TRANSFUSER_ROOT",
    "/ocean/projects/cis250201p/jjain2/transfuser"
))
sys.path.insert(0, str(TRANSFUSER_ROOT / "team_code_transfuser"))

from model import LidarCenterNet
from config import GlobalConfig
from data import lidar_to_histogram_features

# ── Distance bins ─────────────────────────────────────────────────────────────
# NPC1 (nearest) — sampled from these bins
NEAR_BINS = [
    (0.5,  3.0),   # bin 0: very close
    (3.0,  5.0),   # bin 1: close following
    (5.0,  10.0),  # bin 2: normal following
]

# NPC2 (second nearest) — sampled from these bins
FAR_BINS = [
    (10.0, 20.0),  # bin 3: medium range
    (20.0, 50.0),  # bin 4: far
]

# Lateral offset for NPC2 (one lane width to the right)
# Decrease if both NPCs need to be closer together laterally
# Increase if occlusion is still an issue
LANE_WIDTH = 3.5  # meters

# How many steps to collect at each NPC placement before teleporting again
TELEPORT_EVERY = 5

# Save to HDF5 every N steps (protects against crashes)
SAVE_EVERY = 50

# Save debug frames every N steps (set to None to disable)
DEBUG_EVERY = 100


# ── Preprocessing (taken directly from TransFuser repo) ───────────────────────

def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    (width, height) = (image.width // scale, image.height // scale)
    if scale != 1:
        image = image.resize((width, height))
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height
    image = np.asarray(image)
    return image[start_y:start_y+crop_y, start_x:start_x+crop_x]


def shift_x_scale_crop(image, scale, crop, crop_shift=0):
    crop_h, crop_w = crop
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.array(im_resized)
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    start_x += int(crop_shift // scale)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return np.transpose(cropped_image, (2, 0, 1))


def preprocess_rgb(left_raw, front_raw, right_raw, config) -> torch.Tensor:
    rgb = []
    for img_raw in [left_raw, front_raw, right_raw]:
        img_pil = Image.fromarray(cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2RGB))
        cropped = scale_crop(img_pil, scale=config.scale,
                             crop_x=config.img_width, crop_y=config.img_resolution[0])
        rgb.append(cropped)
    rgb = np.concatenate(rgb, axis=1)
    image = Image.fromarray(rgb)
    tensor_np = shift_x_scale_crop(image, scale=config.scale,
                                   crop=config.img_resolution, crop_shift=0)
    return torch.from_numpy(tensor_np.copy()).unsqueeze(0).to('cuda', dtype=torch.float32)


def preprocess_lidar(lidar_carla) -> torch.Tensor:
    points = np.frombuffer(lidar_carla.raw_data, dtype=np.float32).reshape(-1, 4)
    lidar = deepcopy(points[:, :3])
    lidar[:, 1] *= -1
    bev_tensor = lidar_to_histogram_features(lidar)
    return torch.from_numpy(bev_tensor).unsqueeze(0).to('cuda', dtype=torch.float32)


def carla_img_to_numpy(image_carla):
    array = np.frombuffer(image_carla.raw_data, dtype=np.uint8)
    return array.reshape((image_carla.height, image_carla.width, 4))[:, :, :3]


# ── CARLA utilities ───────────────────────────────────────────────────────────

def is_alive(actor) -> bool:
    try:
        return actor is not None and actor.is_alive
    except RuntimeError:
        return False


def setup_carla(host: str = "localhost", port: int = 2000):
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    print(f"Connected to CARLA {client.get_server_version()}")
    world = client.get_world()
    return client, world


def spawn_ego_vehicle(world):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = np.random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned ego vehicle at {spawn_point.location}")
    return vehicle


def spawn_npc(world):
    """Spawn a single NPC vehicle at a random spawn point."""
    bp_lib = world.get_blueprint_library()
    npc_bp = np.random.choice(bp_lib.filter("vehicle.*"))
    spawn_points = world.get_map().get_spawn_points()
    sp = np.random.choice(spawn_points)
    return world.try_spawn_actor(npc_bp, sp)


def spawn_cameras(world, vehicle, config):
    bp_lib = world.get_blueprint_library()
    cameras = {}
    for cam_id, rot in [('rgb_left',  config.camera_rot_1),
                         ('rgb_front', config.camera_rot_0),
                         ('rgb_right', config.camera_rot_2)]:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(config.camera_width))
        cam_bp.set_attribute("image_size_y", str(config.camera_height))
        cam_bp.set_attribute("fov", str(config.camera_fov))
        transform = carla.Transform(
            carla.Location(x=config.camera_pos[0],
                           y=config.camera_pos[1],
                           z=config.camera_pos[2]),
            carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]))
        cameras[cam_id] = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    return cameras


def spawn_lidar(world, vehicle, config):
    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0],
                       y=config.lidar_pos[1],
                       z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0],
                       pitch=config.lidar_rot[1],
                       yaw=config.lidar_rot[2]))
    return world.spawn_actor(lidar_bp, transform, attach_to=vehicle)


def teleport_npc1_ahead(npc, ego, target_distance: float) -> bool:
    """
    Teleport NPC1 directly ahead of ego in the same lane.
    """
    if not is_alive(npc) or not is_alive(ego):
        return False

    ego_tf = ego.get_transform()
    yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

    fwd_x = np.cos(yaw_rad)
    fwd_y = np.sin(yaw_rad)

    new_loc = carla.Location(
        x=ego_tf.location.x + fwd_x * target_distance,
        y=ego_tf.location.y + fwd_y * target_distance,
        z=ego_tf.location.z + 0.5,
    )
    new_rot = carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0)
    npc.set_transform(carla.Transform(new_loc, new_rot))
    npc.set_target_velocity(carla.Vector3D(0, 0, 0))
    return True


def teleport_npc2_adjacent(npc, ego, target_distance: float,
                           lane_width: float = LANE_WIDTH) -> bool:
    """
    Teleport NPC2 ahead of ego but offset one lane width to the right.
    This ensures NPC2 is visible to the camera without being occluded by NPC1.
    """
    if not is_alive(npc) or not is_alive(ego):
        return False

    ego_tf = ego.get_transform()
    yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

    # Forward direction
    fwd_x = np.cos(yaw_rad)
    fwd_y = np.sin(yaw_rad)

    # Right direction (90 degrees clockwise from forward)
    right_x = np.cos(yaw_rad - np.pi / 2)
    right_y = np.sin(yaw_rad - np.pi / 2)

    new_loc = carla.Location(
        x=ego_tf.location.x + fwd_x * target_distance + right_x * lane_width,
        y=ego_tf.location.y + fwd_y * target_distance + right_y * lane_width,
        z=ego_tf.location.z + 0.5,
    )
    new_rot = carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0)
    npc.set_transform(carla.Transform(new_loc, new_rot))
    npc.set_target_velocity(carla.Vector3D(0, 0, 0))
    return True


def get_distance(actor_a, actor_b) -> float:
    return actor_a.get_location().distance(actor_b.get_location())


def get_bin_index(dist: float, bins: List[Tuple[float, float]]) -> int:
    """Return bin index if dist falls within any bin, else -1."""
    for i, (lo, hi) in enumerate(bins):
        if lo <= dist < hi:
            return i
    return -1


# ── Hook registration ─────────────────────────────────────────────────────────

def register_hooks(model: torch.nn.Module) -> Tuple[Dict, List]:
    buffer  = defaultdict(list)
    handles = []

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            if out.dim() == 4:
                out = out.mean(dim=[2, 3])
            elif out.dim() == 3:
                out = out.mean(dim=1)
            buffer[name].append(out.detach().cpu().float().numpy())
        return hook

    backbone = model._model

    for i, s in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            h = getattr(backbone.image_encoder.features, s).register_forward_hook(
                make_hook(f"img_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: img_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: img_enc.{s} not found")

    for i, s in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            h = getattr(backbone.lidar_encoder._model, s).register_forward_hook(
                make_hook(f"lid_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: lid_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: lid_enc.{s} not found")

    for i in range(1, 5):
        try:
            h = getattr(backbone, f"transformer{i}").register_forward_hook(
                make_hook(f"transformer{i}"))
            handles.append(h)
            print(f"  Hooked: transformer{i}")
        except AttributeError:
            print(f"  Warning: transformer{i} not found")

    return buffer, handles


# ── Main collection loop ──────────────────────────────────────────────────────

def collect(args):
    # ── 1. Load TransFuser ───────────────────────────────────────────────────
    print("\n── Loading TransFuser ──")
    config = GlobalConfig(setting='eval')
    model  = LidarCenterNet(config, "cuda", backbone="transFuser",
                            image_architecture="regnety_032",
                            lidar_architecture="regnety_032",
                            use_velocity=0)

    ckpt_dir   = Path(args.model_ckpt)
    ckpt_files = list(ckpt_dir.glob("*.pth"))
    assert len(ckpt_files) > 0, f"No .pth files found in {ckpt_dir}"
    ckpt = torch.load(ckpt_files[0], map_location="cuda")
    model.load_state_dict(ckpt, strict=False)
    model.eval().cuda()
    print(f"Loaded weights from {ckpt_files[0]}")

    # ── 2. Register hooks ────────────────────────────────────────────────────
    print("\n── Registering hooks ──")
    activation_buffer, hook_handles = register_hooks(model)

    # ── 3. Connect to CARLA ──────────────────────────────────────────────────
    print("\n── Connecting to CARLA ──")
    client, world = setup_carla(host=args.host, port=args.port)

    ego = spawn_ego_vehicle(world)
    ego.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
    print("Warming up ego (3s)...")
    time.sleep(3.0)

    # Spawn two controlled NPCs
    npc1 = spawn_npc(world)
    npc2 = spawn_npc(world)
    if npc1 is None or npc2 is None:
        raise RuntimeError("Failed to spawn one or both NPC vehicles.")
    print("Spawned NPC1 (nearest) and NPC2 (second nearest)")

    # Spawn sensors
    cameras = spawn_cameras(world, ego, config)
    lidar   = spawn_lidar(world, ego, config)

    rgb_queues  = {cam_id: [] for cam_id in cameras}
    lidar_queue = []
    for cam_id, cam in cameras.items():
        cam.listen(lambda data, cid=cam_id: rgb_queues[cid].append(data))
    lidar.listen(lambda data: lidar_queue.append(data))

    # Debug output directory
    if DEBUG_EVERY is not None:
        debug_dir = Path("debug_frames_two_npc")
        debug_dir.mkdir(exist_ok=True)

    time.sleep(2.0)

    # ── 4. Collection loop ───────────────────────────────────────────────────
    print(f"\n── Collecting {args.n_steps} steps ──\n")

    gt_distances_1  = []
    gt_distances_2  = []
    bin_labels_1    = []
    bin_labels_2    = []
    step            = 0
    total_ticks     = 0
    near_bin_idx    = 0   # cycles through NEAR_BINS
    far_bin_idx     = 0   # cycles through FAR_BINS
    steps_this_placement = 0

    # Track ego position for stuck/tilt detection
    _init_loc    = ego.get_transform().location
    prev_ego_x   = _init_loc.x
    prev_ego_y   = _init_loc.y
    prev_ego_z   = _init_loc.z

    # ── Open HDF5 file for incremental saving ────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf = h5py.File(output_path, "w")
    ds_d1  = hf.create_dataset("gt_distance_1", shape=(0,),   maxshape=(None,),   dtype=np.float32)
    ds_d2  = hf.create_dataset("gt_distance_2", shape=(0,),   maxshape=(None,),   dtype=np.float32)
    ds_b1  = hf.create_dataset("bin_label_1",   shape=(0,),   maxshape=(None,),   dtype=np.int32)
    ds_b2  = hf.create_dataset("bin_label_2",   shape=(0,),   maxshape=(None,),   dtype=np.int32)
    grp    = hf.create_group("latents")
    lat_ds = {}  # will be created on first flush
    print(f"  HDF5 opened for incremental saving → {output_path}")

    # Initial teleport
    dist1_target = np.random.uniform(*NEAR_BINS[near_bin_idx])
    dist2_target = np.random.uniform(*FAR_BINS[far_bin_idx])
    teleport_npc1_ahead(npc1, ego, dist1_target)
    teleport_npc2_adjacent(npc2, ego, dist2_target)
    time.sleep(0.3)

    try:
        while step < args.n_steps:
            time.sleep(0.1)
            total_ticks += 1

            if not is_alive(ego):
                print("Ego destroyed — stopping.")
                break

            ego.apply_control(carla.VehicleControl(throttle=0.8, steer=0.0, brake=0.0))

            # ── Teleport both NPCs if it's time ──────────────────────────────
            if steps_this_placement >= TELEPORT_EVERY:
                # Advance both bin indices
                near_bin_idx = (near_bin_idx + 1) % len(NEAR_BINS)
                far_bin_idx  = (far_bin_idx  + 1) % len(FAR_BINS)

                dist1_target = np.random.uniform(*NEAR_BINS[near_bin_idx])
                dist2_target = np.random.uniform(*FAR_BINS[far_bin_idx])

                # Respawn NPCs if they died
                if not is_alive(npc1):
                    print("  NPC1 died — respawning")
                    npc1 = spawn_npc(world)

                if not is_alive(npc2):
                    print("  NPC2 died — respawning")
                    npc2 = spawn_npc(world)

                if npc1 is not None:
                    teleport_npc1_ahead(npc1, ego, dist1_target)
                if npc2 is not None:
                    teleport_npc2_adjacent(npc2, ego, dist2_target)

                steps_this_placement = 0
                time.sleep(0.3)
                for q in rgb_queues.values():
                    q.clear()
                lidar_queue.clear()
                continue

            # ── Wait for sensor data ─────────────────────────────────────────
            if not all(rgb_queues[c] for c in rgb_queues) or not lidar_queue:
                continue

            # ── Grab latest frames ───────────────────────────────────────────
            left_raw  = carla_img_to_numpy(rgb_queues['rgb_left'][-1]);  rgb_queues['rgb_left'].clear()
            front_raw = carla_img_to_numpy(rgb_queues['rgb_front'][-1]); rgb_queues['rgb_front'].clear()
            right_raw = carla_img_to_numpy(rgb_queues['rgb_right'][-1]); rgb_queues['rgb_right'].clear()
            lidar_data = lidar_queue[-1]; lidar_queue.clear()

            # ── Ground truth distances ────────────────────────────────────────
            if not is_alive(npc1) or not is_alive(npc2):
                steps_this_placement += 1
                continue

            # ── Skip if ego is tilted or stuck ───────────────────────────────
            ego_tf    = ego.get_transform()
            ego_pitch = ego_tf.rotation.pitch
            ego_roll  = ego_tf.rotation.roll

            if abs(ego_pitch) > 10.0 or abs(ego_roll) > 10.0:
                # Ego flipped — teleport to new spawn point immediately
                print(f"  Ego tilted (pitch={ego_pitch:.1f}, roll={ego_roll:.1f}) — respawning")
                spawn_points = world.get_map().get_spawn_points()
                new_sp = np.random.choice(spawn_points)
                ego.set_transform(new_sp)
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                time.sleep(1.0)
                for q in rgb_queues.values():
                    q.clear()
                lidar_queue.clear()
                steps_this_placement = 0
                continue

            ego_loc = ego_tf.location
            if total_ticks > 50:  # only check after warmup
                dist_moved = ego_loc.distance(carla.Location(
                    x=prev_ego_x, y=prev_ego_y, z=prev_ego_z))
                if dist_moved < 0.05:
                    # Ego hasn't moved — teleport to new spawn point
                    if total_ticks % 100 == 0:
                        print("  Ego stuck — teleporting to new spawn point")
                        spawn_points = world.get_map().get_spawn_points()
                        new_sp = np.random.choice(spawn_points)
                        ego.set_transform(new_sp)
                        ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                        time.sleep(1.0)
                        for q in rgb_queues.values():
                            q.clear()
                        lidar_queue.clear()
                        steps_this_placement = 0
                    continue

            prev_ego_x = ego_loc.x
            prev_ego_y = ego_loc.y
            prev_ego_z = ego_loc.z

            gt_dist1 = get_distance(ego, npc1)
            gt_dist2 = get_distance(ego, npc2)

            gt_dist1 = min(gt_dist1, 100.0)
            gt_dist2 = min(gt_dist2, 100.0)

            # ── Validate both distances fall within intended bins ─────────────
            bin1 = get_bin_index(gt_dist1, NEAR_BINS)
            bin2 = get_bin_index(gt_dist2, FAR_BINS)

            if bin1 == -1 or bin2 == -1:
                # One or both distances drifted out of range — skip
                steps_this_placement += 1
                continue

            if bin1 != near_bin_idx or bin2 != far_bin_idx:
                # Drifted to wrong bin — skip
                steps_this_placement += 1
                continue

            if gt_dist1 >= gt_dist2:
                # Ordering violated — skip
                steps_this_placement += 1
                continue

            # ── Preprocess ───────────────────────────────────────────────────
            rgb_tensor   = preprocess_rgb(left_raw, front_raw, right_raw, config)
            lidar_tensor = preprocess_lidar(lidar_data)

            # ── Forward pass — hooks fire here ───────────────────────────────
            with torch.no_grad():
                v = ego.get_velocity()
                speed = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))
                ego_vel = torch.tensor([[speed]], dtype=torch.float32).to("cuda")
                _ = model._model(rgb_tensor, lidar_tensor, ego_vel)

            gt_distances_1.append(gt_dist1)
            gt_distances_2.append(gt_dist2)
            bin_labels_1.append(near_bin_idx)
            bin_labels_2.append(far_bin_idx)
            step += 1
            steps_this_placement += 1

            # ── Diagnostic print (ego + NPC positions) ───────────────────────
            if step % 100 == 0:
                ego_loc  = ego.get_location()
                npc1_loc = npc1.get_location()
                npc2_loc = npc2.get_location()
                print(f"    EGO : ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
                print(f"    NPC1: ({npc1_loc.x:.1f}, {npc1_loc.y:.1f}, {npc1_loc.z:.1f})  dist1={gt_dist1:.1f}m")
                print(f"    NPC2: ({npc2_loc.x:.1f}, {npc2_loc.y:.1f}, {npc2_loc.z:.1f})  dist2={gt_dist2:.1f}m")

            # ── Debug frames ─────────────────────────────────────────────────
            if DEBUG_EVERY is not None and step % DEBUG_EVERY == 0:
                front_debug = front_raw.copy()
                cv2.putText(front_debug,
                            f"dist1={gt_dist1:.1f}m  dist2={gt_dist2:.1f}m",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(debug_dir / f"front_{step:05d}.png"), front_debug)
                bev_np  = lidar_tensor[0, 1].cpu().numpy()
                bev_vis = (bev_np / (bev_np.max() + 1e-6) * 255).astype(np.uint8)
                cv2.imwrite(str(debug_dir / f"bev_{step:05d}.png"), bev_vis)

            if step % 100 == 0:
                n1_counts = [bin_labels_1.count(i) for i in range(len(NEAR_BINS))]
                n2_counts = [bin_labels_2.count(i) for i in range(len(FAR_BINS))]
                print(f"  Step {step}/{args.n_steps}  "
                      f"dist1={gt_dist1:.1f}m  dist2={gt_dist2:.1f}m  "
                      f"near_bins={n1_counts}  far_bins={n2_counts}  "
                      f"tick_eff={step/total_ticks:.1%}")

            # ── Incremental HDF5 flush every SAVE_EVERY steps ────────────────
            if step % SAVE_EVERY == 0:
                n = len(gt_distances_1)
                saved = ds_d1.shape[0]
                new_d1  = np.array(gt_distances_1[saved:],  dtype=np.float32)
                new_d2  = np.array(gt_distances_2[saved:],  dtype=np.float32)
                new_b1  = np.array(bin_labels_1[saved:],    dtype=np.int32)
                new_b2  = np.array(bin_labels_2[saved:],    dtype=np.int32)
                if len(new_d1) > 0:
                    for ds, arr in [(ds_d1, new_d1), (ds_d2, new_d2),
                                    (ds_b1, new_b1), (ds_b2, new_b2)]:
                        ds.resize(ds.shape[0] + len(arr), axis=0)
                        ds[-len(arr):] = arr
                    # Flush activations
                    for layer_name, acts in activation_buffer.items():
                        new_acts = np.concatenate(acts[saved:], axis=0) if len(acts) > saved else np.array([])
                        if len(new_acts) == 0:
                            continue
                        if layer_name not in lat_ds:
                            lat_ds[layer_name] = grp.create_dataset(
                                layer_name, shape=(0, acts[0].shape[-1]),
                                maxshape=(None, acts[0].shape[-1]),
                                dtype=np.float32, compression="gzip")
                        lat_ds[layer_name].resize(lat_ds[layer_name].shape[0] + len(new_acts), axis=0)
                        lat_ds[layer_name][-len(new_acts):] = new_acts
                    hf.flush()
                    print(f"  Flushed {n} samples to disk")

    finally:
        # ── 5. Clean up ──────────────────────────────────────────────────────
        for h in hook_handles:
            h.remove()
        lidar.stop();  lidar.destroy()
        for cam in cameras.values():
            cam.stop();  cam.destroy()
        if is_alive(ego):
            ego.destroy()
        if is_alive(npc1):
            npc1.destroy()
        if is_alive(npc2):
            npc2.destroy()

        # Final flush of any remaining samples
        n = len(gt_distances_1)
        saved = ds_d1.shape[0]
        new_d1 = np.array(gt_distances_1[saved:], dtype=np.float32)
        new_d2 = np.array(gt_distances_2[saved:], dtype=np.float32)
        new_b1 = np.array(bin_labels_1[saved:],   dtype=np.int32)
        new_b2 = np.array(bin_labels_2[saved:],   dtype=np.int32)
        if len(new_d1) > 0:
            for ds, arr in [(ds_d1, new_d1), (ds_d2, new_d2),
                            (ds_b1, new_b1), (ds_b2, new_b2)]:
                ds.resize(ds.shape[0] + len(arr), axis=0)
                ds[-len(arr):] = arr
            for layer_name, acts in activation_buffer.items():
                new_acts = np.concatenate(acts[saved:], axis=0) if len(acts) > saved else np.array([])
                if len(new_acts) == 0:
                    continue
                if layer_name not in lat_ds:
                    lat_ds[layer_name] = grp.create_dataset(
                        layer_name, shape=(0, acts[0].shape[-1]),
                        maxshape=(None, acts[0].shape[-1]),
                        dtype=np.float32, compression="gzip")
                lat_ds[layer_name].resize(lat_ds[layer_name].shape[0] + len(new_acts), axis=0)
                lat_ds[layer_name][-len(new_acts):] = new_acts
        hf.flush()
        hf.close()
        print("\nCARLA cleaned up.")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    n_collected = len(gt_distances_1)
    if n_collected == 0:
        print("No samples collected.")
        return

    gt_distances_1 = np.array(gt_distances_1, dtype=np.float32)
    gt_distances_2 = np.array(gt_distances_2, dtype=np.float32)
    bin_labels_1   = np.array(bin_labels_1,   dtype=np.int32)
    bin_labels_2   = np.array(bin_labels_2,   dtype=np.int32)

    print(f"\nDataset saved → {output_path}")
    print(f"  Total samples  : {n_collected}")
    print(f"  NPC1 dist range: [{gt_distances_1.min():.1f}, {gt_distances_1.max():.1f}] m")
    print(f"  NPC2 dist range: [{gt_distances_2.min():.1f}, {gt_distances_2.max():.1f}] m")
    print(f"\n  NPC1 bin distribution (near):")
    for i, (lo, hi) in enumerate(NEAR_BINS):
        count = int((bin_labels_1 == i).sum())
        print(f"    bin{i} [{lo:4.1f}–{hi:4.1f}m]: {count} samples ({100*count/max(n_collected,1):.1f}%)")
    print(f"\n  NPC2 bin distribution (far):")
    for i, (lo, hi) in enumerate(FAR_BINS):
        count = int((bin_labels_2 == i).sum())
        print(f"    bin{i} [{lo:4.1f}–{hi:4.1f}m]: {count} samples ({100*count/max(n_collected,1):.1f}%)")
    print(f"\nRun probe sweep with:")
    print(f"  python run_probe_sweep.py --data {output_path} --save-figures")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect TransFuser activations with two controlled NPCs")
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--output",     default="data/latents_two_npc.h5")
    p.add_argument("--n-steps",    type=int, default=5000)
    p.add_argument("--town",       default="Town01")
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int, default=2000)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    collect(args)