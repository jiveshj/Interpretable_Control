"""
oneNPC_collect_data.py
----------------------
Runs TransFuser in CARLA, captures per-layer activations via PyTorch forward
hooks, and saves them alongside ground-truth distances to an HDF5 file.

v4 changes over v3:
- NPC now drives autonomously along the road (waypoint steering, same as ego)
- NPC throttle is randomized every NPC_THROTTLE_CHANGE_EVERY steps to produce
  natural distance variation (no more deterministic bin teleportation)
- Initial NPC placement still uses waypoint teleport
- Safety teleport triggers only if ego↔NPC distance exceeds DIST_RECOVERY_MAX
- Bin labels are assigned from observed distance, not forced

v3 changes over v2:
- Ego uses waypoint-following steering (was steer=0 → got stuck on curves)
- NPC is teleported along the ROAD network via waypoints (was raw forward
  vector → could land in buildings/medians)
- Warmup loop after ego spawn to handle async-mode physics delay
- Spawn points filtered to z<1.0 to avoid bridge/elevated spawns

Usage (on a GPU node, after launching CARLA in background):
    ./CarlaUE4.sh -opengl -RenderOffScreen &
    sleep 10
    python oneNPC_collect_data.py \
        --model-ckpt /path/to/model_ckpt/transfuser \
        --output     /path/to/latents.h5 \
        --n-steps    5000 \
        --town       Town01

Read with your existing probe pipeline:
    python run_probe_sweep.py --data /path/to/latents.h5 --save-figures
"""

import argparse
import sys
import os
import time
import math
import random
import numpy as np
import h5py
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
from copy import deepcopy

# ── CARLA + TransFuser imports ───────────────────────────────────────────────
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
DISTANCE_BINS = [
    (3.0,  5.0),   # bin 0: close following
    (5.0,  10.0),  # bin 1: normal following
    (10.0, 20.0),  # bin 2: medium range
    (20.0, 40.0),  # bin 3: far
]

# Simulation mode
SYNC_MODE             = True
FIXED_DELTA_SECONDS   = 0.05        # 20Hz

# Ego driving config
EGO_THROTTLE          = 0.3
WAYPOINT_LOOKAHEAD    = 5.0
STEER_GAIN            = 1.5
STUCK_SPEED_THRESHOLD = 0.5
STUCK_CONSECUTIVE     = 40

# NPC throttle randomization
NPC_THROTTLE_OPTIONS      = [0.0, 0.3, 0.5, 0.7, 0.9]
NPC_THROTTLE_CHANGE_EVERY = 30
INITIAL_NPC_DIST_RANGE    = (5.0, 15.0)
DIST_RECOVERY_MAX         = 50.0
MAX_VALID_DISTANCE        = DISTANCE_BINS[-1][1]   # 40m

# Forced bin-balancing teleport
FORCED_TELEPORT_EVERY     = 40
EGO_BRAKE_TICKS_AFTER_CLOSE_TELEPORT = 10  # brake ticks before resuming after bin-0 teleport

# Logging and frame saving
LOG_EVERY                 = 20
SAVE_FRAMES               = True
FRAME_EVERY               = 100
FRAMES_DIR                = "frames_collect"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    (width, height) = (image.width // scale, image.height // scale)
    if scale != 1:
        image = image.resize((width, height))
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height
    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
    return cropped_image


def shift_x_scale_crop(image, scale, crop, crop_shift=0):
    crop_h, crop_w = crop
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.array(im_resized)
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    start_x += int(crop_shift // scale)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


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

    if SYNC_MODE:
        settings = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        print(f"Synchronous mode enabled ({1/FIXED_DELTA_SECONDS:.0f}Hz)")

    return client, world


def compute_waypoint_steer(vehicle, world_map) -> float:
    veh_tf  = vehicle.get_transform()
    veh_loc = veh_tf.location
    wp = world_map.get_waypoint(veh_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0
    next_wps = wp.next(WAYPOINT_LOOKAHEAD)
    if not next_wps:
        return 0.0
    target_loc = next_wps[0].transform.location
    dx = target_loc.x - veh_loc.x
    dy = target_loc.y - veh_loc.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 0.001:
        return 0.0
    fwd   = veh_tf.get_forward_vector()
    cross = fwd.x * dy - fwd.y * dx
    steer = STEER_GAIN * cross / dist
    return max(-1.0, min(1.0, steer))


def get_waypoint_ahead_transform(ego, world_map, dist: float) -> carla.Transform:
    ego_loc = ego.get_location()
    wp = world_map.get_waypoint(ego_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        ego_tf  = ego.get_transform()
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        loc = carla.Location(
            x=ego_loc.x + dist * math.cos(yaw_rad),
            y=ego_loc.y + dist * math.sin(yaw_rad),
            z=ego_loc.z + 0.5,
        )
        return carla.Transform(loc, ego_tf.rotation)
    next_wps = wp.next(dist)
    if not next_wps:
        return wp.transform
    tf = next_wps[0].transform
    tf.location.z += 0.5
    return tf


def spawn_ego_vehicle(world):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = [sp for sp in world.get_map().get_spawn_points()
                    if sp.location.z < 1.0]
    if not spawn_points:
        spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
    print(f"Spawned ego vehicle at {vehicle.get_location()}")
    return vehicle


def warmup_ego(ego, world, world_map):
    print("Warming up ego vehicle...")
    for warmup_step in range(80):
        steer = compute_waypoint_steer(ego, world_map)
        ego.apply_control(carla.VehicleControl(
            throttle=EGO_THROTTLE, steer=steer, brake=0.0))
        if SYNC_MODE:
            world.tick()
        else:
            time.sleep(0.05)
        v   = ego.get_velocity()
        spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        if spd > 0.5:
            print(f"  Ego moving at {spd:.2f} m/s after {warmup_step+1} ticks")
            return True
    print("  WARNING: ego still not moving after warmup")
    return False


def spawn_npc(world):
    bp_lib = world.get_blueprint_library()
    npc_bp = random.choice(bp_lib.filter("vehicle.*"))
    spawn_points = [sp for sp in world.get_map().get_spawn_points()
                    if sp.location.z < 1.0]
    if not spawn_points:
        spawn_points = world.get_map().get_spawn_points()
    return world.try_spawn_actor(npc_bp, random.choice(spawn_points))


def spawn_cameras(world, vehicle, config):
    bp_lib   = world.get_blueprint_library()
    cameras  = {}
    for cam_id, rot in [('rgb_left',  config.camera_rot_1),
                         ('rgb_front', config.camera_rot_0),
                         ('rgb_right', config.camera_rot_2)]:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(config.camera_width))
        cam_bp.set_attribute("image_size_y", str(config.camera_height))
        cam_bp.set_attribute("fov",          str(config.camera_fov))
        transform = carla.Transform(
            carla.Location(x=config.camera_pos[0],
                           y=config.camera_pos[1],
                           z=config.camera_pos[2]),
            carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]))
        cameras[cam_id] = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    return cameras


def spawn_lidar(world, vehicle, config):
    bp_lib   = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0],
                       y=config.lidar_pos[1],
                       z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0],
                       pitch=config.lidar_rot[1],
                       yaw=config.lidar_rot[2]))
    return world.spawn_actor(lidar_bp, transform, attach_to=vehicle)


def teleport_npc_ahead(npc, ego, world_map, target_distance: float) -> bool:
    if not is_alive(npc) or not is_alive(ego):
        return False
    target_tf = get_waypoint_ahead_transform(ego, world_map, target_distance)
    npc.set_transform(target_tf)
    npc.set_target_velocity(carla.Vector3D(0, 0, 0))
    return True


def get_distance(actor_a, actor_b) -> float:
    return actor_a.get_location().distance(actor_b.get_location())


def flush_queues(rgb_queues, lidar_queue):
    for q in rgb_queues.values():
        q.clear()
    lidar_queue.clear()


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
    world_map     = world.get_map()

    ego = spawn_ego_vehicle(world)
    warmup_ego(ego, world, world_map)

    ego.set_target_velocity(carla.Vector3D(0, 0, 0))
    ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

    npc = spawn_npc(world)
    if npc is None:
        raise RuntimeError("Failed to spawn NPC vehicle.")
    print("Spawned NPC vehicle")

    cameras = spawn_cameras(world, ego, config)
    lidar   = spawn_lidar(world, ego, config)

    rgb_queues  = {cam_id: [] for cam_id in cameras}
    lidar_queue = []
    for cam_id, cam in cameras.items():
        cam.listen(lambda data, cid=cam_id: rgb_queues[cid].append(data))
    lidar.listen(lambda data: lidar_queue.append(data))

    time.sleep(1.0)

    initial_dist = np.random.uniform(*INITIAL_NPC_DIST_RANGE)
    teleport_npc_ahead(npc, ego, world_map, initial_dist)
    if SYNC_MODE:
        world.tick(); world.tick()
    else:
        time.sleep(0.3)

    # ── 4. Collection loop ───────────────────────────────────────────────────
    print(f"\n── Collecting {args.n_steps} steps ──\n")
    if SAVE_FRAMES:
        os.makedirs(FRAMES_DIR, exist_ok=True)

    gt_distances       = []
    bin_labels         = []
    step               = 0
    total_ticks        = 0
    stuck_count        = 0
    npc_stuck_count    = 0
    npc_throttle       = random.choice(NPC_THROTTLE_OPTIONS)
    npc_throttle_ticks = 0
    recovery_teleports = 0

    def bin_for_distance(d: float) -> Optional[int]:
        for i, (lo, hi) in enumerate(DISTANCE_BINS):
            if lo <= d < hi:
                return i
        return None

    try:
        while step < args.n_steps:
            if SYNC_MODE:
                world.tick()
            else:
                time.sleep(FIXED_DELTA_SECONDS)
            total_ticks += 1

            if not is_alive(ego):
                print("Ego destroyed — stopping.")
                break

            ego_steer = compute_waypoint_steer(ego, world_map)
            ego.apply_control(carla.VehicleControl(
                throttle=EGO_THROTTLE, steer=ego_steer, brake=0.0))

            ego_tf    = ego.get_transform()
            ego_loc   = ego_tf.location
            v         = ego.get_velocity()
            ego_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            if is_alive(npc):
                if npc_throttle_ticks >= NPC_THROTTLE_CHANGE_EVERY:
                    npc_throttle       = random.choice(NPC_THROTTLE_OPTIONS)
                    npc_throttle_ticks = 0
                npc_throttle_ticks += 1

                npc_steer = compute_waypoint_steer(npc, world_map)
                npc.apply_control(carla.VehicleControl(
                    throttle=npc_throttle, steer=npc_steer, brake=0.0))

                npc_loc   = npc.get_transform().location
                npc_v     = npc.get_velocity()
                npc_speed = math.sqrt(npc_v.x**2 + npc_v.y**2 + npc_v.z**2)
            else:
                print("  NPC died — respawning")
                npc = spawn_npc(world)
                if npc is not None:
                    teleport_npc_ahead(npc, ego, world_map,
                                       np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                    time.sleep(0.3)
                continue

            # ── Ego stuck detection ───────────────────────────────────────────
            if ego_speed < STUCK_SPEED_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= STUCK_CONSECUTIVE:
                print(f"  Ego stuck (speed={ego_speed:.2f}m/s) — respawning")
                spawn_points = [sp for sp in world_map.get_spawn_points()
                                if sp.location.z < 1.0]
                ego.set_transform(random.choice(spawn_points))
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                if SYNC_MODE:
                    for _ in range(10):
                        world.tick()
                else:
                    time.sleep(1.0)
                stuck_count = 0
                teleport_npc_ahead(npc, ego, world_map,
                                   np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                if SYNC_MODE:
                    world.tick(); world.tick()
                else:
                    time.sleep(0.3)
                flush_queues(rgb_queues, lidar_queue)
                continue

            # ── NPC stuck detection ───────────────────────────────────────────
            if npc_speed < STUCK_SPEED_THRESHOLD:
                npc_stuck_count += 1
            else:
                npc_stuck_count = 0

            if npc_stuck_count >= STUCK_CONSECUTIVE:
                print(f"  NPC stuck — re-teleporting")
                teleport_npc_ahead(npc, ego, world_map,
                                   np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                npc_stuck_count = 0
                if SYNC_MODE:
                    world.tick(); world.tick()
                else:
                    time.sleep(0.3)
                flush_queues(rgb_queues, lidar_queue)
                continue

            # ── Distance + safety re-teleport ─────────────────────────────────
            gt_dist = get_distance(ego, npc)

            if gt_dist > DIST_RECOVERY_MAX:
                print(f"  NPC drifted to {gt_dist:.1f}m — re-teleporting "
                      f"(recovery #{recovery_teleports+1})")
                teleport_npc_ahead(npc, ego, world_map,
                                   np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                recovery_teleports += 1
                if SYNC_MODE:
                    world.tick(); world.tick()
                else:
                    time.sleep(0.3)
                flush_queues(rgb_queues, lidar_queue)
                continue

            # ── Wait for sensor data ──────────────────────────────────────────
            if not all(rgb_queues[c] for c in rgb_queues) or not lidar_queue:
                continue

            # ── Grab latest frames ────────────────────────────────────────────
            left_raw   = carla_img_to_numpy(rgb_queues['rgb_left'][-1]);  rgb_queues['rgb_left'].clear()
            front_raw  = carla_img_to_numpy(rgb_queues['rgb_front'][-1]); rgb_queues['rgb_front'].clear()
            right_raw  = carla_img_to_numpy(rgb_queues['rgb_right'][-1]); rgb_queues['rgb_right'].clear()
            lidar_data = lidar_queue[-1]; lidar_queue.clear()

            # ── Preprocess ────────────────────────────────────────────────────
            rgb_tensor   = preprocess_rgb(left_raw, front_raw, right_raw, config)
            lidar_tensor = preprocess_lidar(lidar_data)

            # ── Bin assignment ────────────────────────────────────────────────
            bin_idx = bin_for_distance(gt_dist)
            if bin_idx is None:
                continue

            # ── Forward pass — hooks fire here ────────────────────────────────
            with torch.no_grad():
                ego_vel = torch.tensor([[ego_speed]], dtype=torch.float32).to("cuda")
                _ = model._model(rgb_tensor, lidar_tensor, ego_vel)

            gt_distances.append(gt_dist)
            bin_labels.append(bin_idx)
            step += 1

            # ── Forced bin-balancing teleport ─────────────────────────────────
            if step % FORCED_TELEPORT_EVERY == 0:
                counts     = [bin_labels.count(i) for i in range(len(DISTANCE_BINS))]
                target_bin = counts.index(min(counts))
                lo, hi     = DISTANCE_BINS[target_bin]
                target_dist = np.random.uniform(lo, hi)
                teleport_npc_ahead(npc, ego, world_map, target_dist)

                if target_bin == 0:
                    # Bin 0 is close (3-5m) — brake ego first so it doesn't
                    # ram the freshly-teleported NPC. In sync mode each tick
                    # is deterministic so sensor data remains consistent.
                    npc_throttle       = 0.3
                    npc_throttle_ticks = 0
                    for _ in range(EGO_BRAKE_TICKS_AFTER_CLOSE_TELEPORT):
                        ego.apply_control(carla.VehicleControl(
                            throttle=0.0, steer=0.0, brake=1.0))
                        world.tick()

                if SYNC_MODE:
                    world.tick(); world.tick()
                else:
                    time.sleep(0.3)

                flush_queues(rgb_queues, lidar_queue)
                stuck_count = 0

            # ── Save verification frame ───────────────────────────────────────
            if SAVE_FRAMES and step % FRAME_EVERY == 0:
                try:
                    fname = os.path.join(FRAMES_DIR,
                                         f"step{step:05d}_dist{gt_dist:.1f}m.jpg")
                    cv2.imwrite(fname, cv2.cvtColor(front_raw, cv2.COLOR_RGB2BGR))
                except Exception:
                    pass

            # ── Periodic state log ────────────────────────────────────────────
            if step % LOG_EVERY == 0:
                counts = [bin_labels.count(i) for i in range(len(DISTANCE_BINS))]
                print(f"  step={step}/{args.n_steps}"
                      f"  ego=({ego_loc.x:.1f},{ego_loc.y:.1f})  spd={ego_speed:.1f}m/s"
                      f"  npc=({npc_loc.x:.1f},{npc_loc.y:.1f})  npc_spd={npc_speed:.1f}m/s"
                      f"  dist={gt_dist:.1f}m  bin={bin_idx}"
                      f"  npc_thr={npc_throttle:.2f}  bins={counts}"
                      f"  eff={step/total_ticks:.1%}")

    finally:
        for h in hook_handles:
            h.remove()
        lidar.stop();  lidar.destroy()
        for cam in cameras.values():
            cam.stop();  cam.destroy()
        if is_alive(ego):
            ego.destroy()
        if is_alive(npc):
            npc.destroy()
        if SYNC_MODE:
            settings = world.get_settings()
            settings.synchronous_mode    = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("Synchronous mode disabled.")
        print("\nCARLA cleaned up.")

    # ── 5. Save to HDF5 ──────────────────────────────────────────────────────
    gt_distances = np.array(gt_distances, dtype=np.float32)
    bin_labels   = np.array(bin_labels,   dtype=np.int32)
    n_collected  = len(gt_distances)
    print(f"\n── Saving {n_collected} samples ──")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("gt_distance", data=gt_distances)
        f.create_dataset("bin_label",   data=bin_labels)
        grp = f.create_group("latents")
        for layer_name, acts in activation_buffer.items():
            arr = np.concatenate(acts, axis=0)
            if arr.shape[0] != n_collected:
                print(f"  Warning: {layer_name} has {arr.shape[0]} samples "
                      f"(expected {n_collected}) — skipping")
                continue
            grp.create_dataset(layer_name, data=arr, compression="gzip")
            print(f"  Saved {layer_name}: shape={arr.shape}")

    print(f"\nDataset saved → {output_path}")
    print(f"  Total samples      : {n_collected}")
    print(f"  Distance range     : [{gt_distances.min():.1f}, {gt_distances.max():.1f}] m")
    print(f"  Distance mean      : {gt_distances.mean():.1f} m  std={gt_distances.std():.1f} m")
    print(f"  Recovery teleports : {recovery_teleports}")
    print(f"\n  Bin distribution:")
    for i, (lo, hi) in enumerate(DISTANCE_BINS):
        count = int((bin_labels == i).sum())
        pct   = 100*count/n_collected if n_collected > 0 else 0
        print(f"    bin{i} [{lo:4.1f}–{hi:4.1f}m]: {count} samples ({pct:.1f}%)")

    print(f"\nRun probe sweep with:")
    print(f"  python run_probe_sweep.py --data {output_path} --save-figures")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--output",     default="data/latents.h5")
    p.add_argument("--n-steps",    type=int, default=5000)
    p.add_argument("--town",       default="Town01")
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int, default=2000)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    collect(args)