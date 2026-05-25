# """
# collect_data.py
# ---------------
# Runs TransFuser in CARLA, captures per-layer activations via PyTorch forward
# hooks, and saves them alongside ground-truth distances to an HDF5 file.

# Key improvement over v1: uses a single controlled NPC that is teleported to
# a sampled distance in front of the ego vehicle every TELEPORT_EVERY steps.
# Distance bins are cycled in order to ensure balanced coverage across the
# full distance range.

# Usage (on a GPU node, after launching CARLA in background):
#     ./CarlaUE4.sh -opengl -RenderOffScreen &
#     sleep 10
#     python collect_data.py \
#         --model-ckpt /path/to/model_ckpt/transfuser \
#         --output     /path/to/latents.h5 \
#         --n-steps    5000 \
#         --town       Town01

# The output HDF5 file is directly readable by your existing probe pipeline:
#     python run_probe_sweep.py --data /path/to/latents.h5 --save-figures
# """

# import argparse
# import sys
# import os
# import time
# import numpy as np
# import h5py
# import torch
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, List, Tuple, Optional
# import cv2
# from PIL import Image
# from copy import deepcopy

# # ── CARLA + TransFuser imports ───────────────────────────────────────────────
# try:
#     import carla
# except ImportError:
#     raise ImportError(
#         "CARLA Python API not found. Make sure PYTHONPATH includes:\n"
#         "  $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg\n"
#         "  $CARLA_ROOT/PythonAPI/carla"
#     )

# TRANSFUSER_ROOT = Path(os.environ.get(
#     "TRANSFUSER_ROOT",
#     "/ocean/projects/cis250201p/jjain2/transfuser"
# ))
# sys.path.insert(0, str(TRANSFUSER_ROOT / "team_code_transfuser"))

# from model import LidarCenterNet
# from config import GlobalConfig
# from data import lidar_to_histogram_features

# # ── Distance bins ─────────────────────────────────────────────────────────────
# # Each bin is (low, high) in meters. The NPC will be teleported to a distance
# # sampled uniformly from within the selected bin.
# DISTANCE_BINS = [
#     (0.5,  3.0),   # bin 0: very close
#     (3.0,  5.0),   # bin 1: close following
#     (5.0,  10.0),  # bin 2: normal following
#     (10.0, 20.0),  # bin 3: medium range
#     (20.0, 50.0),  # bin 4: far
# ]

# # How many steps to collect at each NPC placement before teleporting again.
# # Lower = more diverse but more teleport overhead. 5-10 is a good balance.
# TELEPORT_EVERY = 5


# # ── Preprocessing (taken directly from TransFuser repo) ───────────────────────

# def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
#     (width, height) = (image.width // scale, image.height // scale)
#     if scale != 1:
#         image = image.resize((width, height))
#     if crop_x is None:
#         crop_x = width
#     if crop_y is None:
#         crop_y = height
#     image = np.asarray(image)
#     cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
#     return cropped_image


# def shift_x_scale_crop(image, scale, crop, crop_shift=0):
#     crop_h, crop_w = crop
#     (width, height) = (int(image.width // scale), int(image.height // scale))
#     im_resized = image.resize((width, height))
#     image = np.array(im_resized)
#     start_y = height//2 - crop_h//2
#     start_x = width//2 - crop_w//2
#     start_x += int(crop_shift // scale)
#     cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
#     cropped_image = np.transpose(cropped_image, (2, 0, 1))
#     return cropped_image


# def preprocess_rgb(left_raw, front_raw, right_raw, config) -> torch.Tensor:
#     rgb = []
#     for img_raw in [left_raw, front_raw, right_raw]:
#         img_pil = Image.fromarray(cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2RGB))
#         cropped = scale_crop(img_pil, scale=config.scale,
#                              crop_x=config.img_width, crop_y=config.img_resolution[0])
#         rgb.append(cropped)
#     rgb = np.concatenate(rgb, axis=1)
#     image = Image.fromarray(rgb)
#     tensor_np = shift_x_scale_crop(image, scale=config.scale,
#                                    crop=config.img_resolution, crop_shift=0)
#     return torch.from_numpy(tensor_np.copy()).unsqueeze(0).to('cuda', dtype=torch.float32)


# def preprocess_lidar(lidar_carla) -> torch.Tensor:
#     points = np.frombuffer(lidar_carla.raw_data, dtype=np.float32).reshape(-1, 4)
#     lidar = deepcopy(points[:, :3])
#     lidar[:, 1] *= -1  # CARLA left-hand → right-hand coordinate system
#     bev_tensor = lidar_to_histogram_features(lidar)
#     return torch.from_numpy(bev_tensor).unsqueeze(0).to('cuda', dtype=torch.float32)


# def carla_img_to_numpy(image_carla):
#     array = np.frombuffer(image_carla.raw_data, dtype=np.uint8)
#     return array.reshape((image_carla.height, image_carla.width, 4))[:, :, :3]


# # ── CARLA utilities ───────────────────────────────────────────────────────────

# def is_alive(actor) -> bool:
#     try:
#         return actor is not None and actor.is_alive
#     except RuntimeError:
#         return False


# def setup_carla(host: str = "localhost", port: int = 2000):
#     client = carla.Client(host, port)
#     client.set_timeout(30.0)
#     print(f"Connected to CARLA {client.get_server_version()}")
#     world = client.get_world()
#     return client, world


# def spawn_ego_vehicle(world):
#     bp_lib = world.get_blueprint_library()
#     vehicle_bp = bp_lib.find("vehicle.tesla.model3")
#     spawn_points = world.get_map().get_spawn_points()
#     spawn_point = np.random.choice(spawn_points)
#     vehicle = world.spawn_actor(vehicle_bp, spawn_point)
#     print(f"Spawned ego vehicle at {spawn_point.location}")
#     return vehicle


# def spawn_npc(world):
#     """Spawn a single NPC vehicle at an arbitrary spawn point."""
#     bp_lib = world.get_blueprint_library()
#     npc_bp = np.random.choice(bp_lib.filter("vehicle.*"))
#     spawn_points = world.get_map().get_spawn_points()
#     # Use a random spawn point — we'll teleport it immediately anyway
#     sp = np.random.choice(spawn_points)
#     npc = world.try_spawn_actor(npc_bp, sp)
#     return npc


# def spawn_cameras(world, vehicle, config):
#     bp_lib = world.get_blueprint_library()
#     cameras = {}
#     for cam_id, rot in [('rgb_left',  config.camera_rot_1),
#                          ('rgb_front', config.camera_rot_0),
#                          ('rgb_right', config.camera_rot_2)]:
#         cam_bp = bp_lib.find("sensor.camera.rgb")
#         cam_bp.set_attribute("image_size_x", str(config.img_width))
#         cam_bp.set_attribute("image_size_y", str(config.img_resolution[0]))
#         cam_bp.set_attribute("fov", str(config.camera_fov))
#         transform = carla.Transform(
#             carla.Location(x=config.camera_pos[0],
#                            y=config.camera_pos[1],
#                            z=config.camera_pos[2]),
#             carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]))
#         cameras[cam_id] = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
#     return cameras


# def spawn_lidar(world, vehicle, config):
#     bp_lib = world.get_blueprint_library()
#     lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
#     transform = carla.Transform(
#         carla.Location(x=config.lidar_pos[0],
#                        y=config.lidar_pos[1],
#                        z=config.lidar_pos[2]),
#         carla.Rotation(roll=config.lidar_rot[0],
#                        pitch=config.lidar_rot[1],
#                        yaw=config.lidar_rot[2]))
#     return world.spawn_actor(lidar_bp, transform, attach_to=vehicle)


# def teleport_npc_ahead(npc, ego, target_distance: float) -> bool:
#     """
#     Teleport the NPC to exactly target_distance metres directly ahead of the
#     ego vehicle, matching the ego's yaw so it faces the same direction.

#     Returns True on success, False if the actor is no longer alive.
#     """
#     if not is_alive(npc) or not is_alive(ego):
#         return False

#     ego_tf = ego.get_transform()
#     yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

#     # Unit vector pointing in ego's forward direction
#     fwd_x = np.cos(yaw_rad)
#     fwd_y = np.sin(yaw_rad)

#     new_loc = carla.Location(
#         x=ego_tf.location.x + fwd_x * target_distance,
#         y=ego_tf.location.y + fwd_y * target_distance,
#         z=ego_tf.location.z + 0.5,   # slight z offset to avoid ground collision
#     )
#     new_rot = carla.Rotation(
#         pitch=0.0,
#         yaw=ego_tf.rotation.yaw,     # face same direction as ego
#         roll=0.0,
#     )
#     npc.set_transform(carla.Transform(new_loc, new_rot))
#     npc.set_target_velocity(carla.Vector3D(0, 0, 0))  # stop it after teleport
#     return True


# def get_distance(actor_a, actor_b) -> float:
#     """Euclidean distance between two CARLA actors."""
#     return actor_a.get_location().distance(actor_b.get_location())


# # ── Hook registration ─────────────────────────────────────────────────────────

# def register_hooks(model: torch.nn.Module) -> Tuple[Dict, List]:
#     buffer  = defaultdict(list)
#     handles = []

#     def make_hook(name: str):
#         def hook(module, input, output):
#             if isinstance(output, (tuple, list)):
#                 out = output[0]
#             else:
#                 out = output
#             if out.dim() == 4:       # (B, C, H, W) → global avg pool
#                 out = out.mean(dim=[2, 3])
#             elif out.dim() == 3:     # (B, seq, C) → mean over sequence
#                 out = out.mean(dim=1)
#             buffer[name].append(out.detach().cpu().float().numpy())
#         return hook

#     backbone = model._model

#     for i, s in enumerate(["s1", "s2", "s3", "s4"]):
#         try:
#             h = getattr(backbone.image_encoder.features, s).register_forward_hook(
#                 make_hook(f"img_enc.stage{i+1}"))
#             handles.append(h)
#             print(f"  Hooked: img_enc.stage{i+1}")
#         except AttributeError:
#             print(f"  Warning: img_enc.{s} not found")

#     for i, s in enumerate(["s1", "s2", "s3", "s4"]):
#         try:
#             h = getattr(backbone.lidar_encoder._model, s).register_forward_hook(
#                 make_hook(f"lid_enc.stage{i+1}"))
#             handles.append(h)
#             print(f"  Hooked: lid_enc.stage{i+1}")
#         except AttributeError:
#             print(f"  Warning: lid_enc.{s} not found")

#     for i in range(1, 5):
#         try:
#             h = getattr(backbone, f"transformer{i}").register_forward_hook(
#                 make_hook(f"transformer{i}"))
#             handles.append(h)
#             print(f"  Hooked: transformer{i}")
#         except AttributeError:
#             print(f"  Warning: transformer{i} not found")

#     return buffer, handles


# # ── Main collection loop ──────────────────────────────────────────────────────

# def collect(args):
#     # ── 1. Load TransFuser ───────────────────────────────────────────────────
#     print("\n── Loading TransFuser ──")
#     config = GlobalConfig(setting='eval')
#     model  = LidarCenterNet(config, "cuda", backbone="transFuser",
#                             image_architecture="regnety_032",
#                             lidar_architecture="regnety_032",
#                             use_velocity=0)

#     ckpt_dir   = Path(args.model_ckpt)
#     ckpt_files = list(ckpt_dir.glob("*.pth"))
#     assert len(ckpt_files) > 0, f"No .pth files found in {ckpt_dir}"
#     ckpt = torch.load(ckpt_files[0], map_location="cuda")
#     model.load_state_dict(ckpt, strict=False)
#     model.eval().cuda()
#     print(f"Loaded weights from {ckpt_files[0]}")

#     # ── 2. Register hooks ────────────────────────────────────────────────────
#     print("\n── Registering hooks ──")
#     activation_buffer, hook_handles = register_hooks(model)

#     # ── 3. Connect to CARLA ──────────────────────────────────────────────────
#     print("\n── Connecting to CARLA ──")
#     client, world = setup_carla(host=args.host, port=args.port)

#     ego = spawn_ego_vehicle(world)
#     ego.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
#     print("Warming up ego (3s)...")
#     time.sleep(3.0)

#     # Single controlled NPC
#     npc = spawn_npc(world)
#     if npc is None:
#         raise RuntimeError("Failed to spawn NPC vehicle — no free spawn points.")
#     print("Spawned NPC vehicle")

#     # Spawn sensors
#     cameras = spawn_cameras(world, ego, config)
#     lidar   = spawn_lidar(world, ego, config)

#     rgb_queues  = {cam_id: [] for cam_id in cameras}
#     lidar_queue = []
#     for cam_id, cam in cameras.items():
#         cam.listen(lambda data, cid=cam_id: rgb_queues[cid].append(data))
#     lidar.listen(lambda data: lidar_queue.append(data))

#     # Give sensors time to start producing data
#     time.sleep(2.0)

#     # ── 4. Collection loop ───────────────────────────────────────────────────
#     print(f"\n── Collecting {args.n_steps} steps ──\n")

#     gt_distances   = []
#     bin_labels     = []       # which bin each sample came from
#     step           = 0
#     total_ticks    = 0
#     bin_idx        = 0        # cycles through DISTANCE_BINS in order
#     steps_this_placement = 0  # how many steps since last teleport

#     # Do an initial teleport before collecting anything
#     target_dist = np.random.uniform(*DISTANCE_BINS[bin_idx])
#     teleport_npc_ahead(npc, ego, target_dist)
#     time.sleep(0.3)  # let physics settle after teleport

#     try:
#         while step < args.n_steps:
#             time.sleep(0.1)
#             total_ticks += 1

#             # ── Ego keepalive ────────────────────────────────────────────────
#             if not is_alive(ego):
#                 print("Ego destroyed — stopping.")
#                 break

#             ego.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))

#             # ── Teleport NPC if it's time ────────────────────────────────────
#             if steps_this_placement >= TELEPORT_EVERY:
#                 # Advance to next bin (cycle)
#                 bin_idx = (bin_idx + 1) % len(DISTANCE_BINS)
#                 target_dist = np.random.uniform(*DISTANCE_BINS[bin_idx])

#                 if is_alive(npc):
#                     success = teleport_npc_ahead(npc, ego, target_dist)
#                     if not success:
#                         print("  Warning: teleport failed, skipping")
#                 else:
#                     # NPC died somehow — respawn it
#                     print("  NPC died — respawning")
#                     npc = spawn_npc(world)
#                     if npc is not None:
#                         teleport_npc_ahead(npc, ego, target_dist)
#                     else:
#                         print("  Warning: NPC respawn failed")

#                 steps_this_placement = 0
#                 time.sleep(0.3)  # let physics settle after teleport
#                 # Flush stale sensor data accumulated during sleep
#                 for q in rgb_queues.values():
#                     q.clear()
#                 lidar_queue.clear()
#                 continue  # skip this tick, collect from next one

#             # ── Wait for sensor data ─────────────────────────────────────────
#             if not all(rgb_queues[c] for c in rgb_queues) or not lidar_queue:
#                 continue

#             # ── Grab latest frames ───────────────────────────────────────────
#             left_raw  = carla_img_to_numpy(rgb_queues['rgb_left'][-1]);  rgb_queues['rgb_left'].clear()
#             front_raw = carla_img_to_numpy(rgb_queues['rgb_front'][-1]); rgb_queues['rgb_front'].clear()
#             right_raw = carla_img_to_numpy(rgb_queues['rgb_right'][-1]); rgb_queues['rgb_right'].clear()
#             lidar_data = lidar_queue[-1]; lidar_queue.clear()

#             # ── Preprocess ───────────────────────────────────────────────────
#             rgb_tensor   = preprocess_rgb(left_raw, front_raw, right_raw, config)
#             lidar_tensor = preprocess_lidar(lidar_data)

#             # ── Ground truth distance ────────────────────────────────────────
#             if is_alive(npc):
#                 gt_dist = get_distance(ego, npc)
#             else:
#                 # NPC not alive — skip this sample
#                 continue

#             gt_dist = min(gt_dist, 100.0)  # safety cap

#             # ── Validate sample falls within intended bin ─────────────────────
#             # The ego moves between teleport and measurement, so the actual
#             # distance may have drifted outside the intended bin. Only accept
#             # samples where gt_dist is genuinely within the target bin range,
#             # ensuring bin_label always matches gt_distance exactly.
#             lo, hi = DISTANCE_BINS[bin_idx]
#             if not (lo <= gt_dist < hi):
#                 steps_this_placement += 1  # still count toward teleport cycle
#                 continue

#             # ── Forward pass — hooks fire here ───────────────────────────────
#             with torch.no_grad():
#                 v = ego.get_velocity()
#                 speed = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))
#                 ego_vel = torch.tensor([[speed]], dtype=torch.float32).to("cuda")
#                 _ = model._model(rgb_tensor, lidar_tensor, ego_vel)

#             gt_distances.append(gt_dist)
#             bin_labels.append(bin_idx)
#             step += 1
#             steps_this_placement += 1

#             if step % 100 == 0:
#                 counts = [bin_labels.count(i) for i in range(len(DISTANCE_BINS))]
#                 print(f"  Step {step}/{args.n_steps}  "
#                       f"gt_dist={gt_dist:.1f}m  "
#                       f"bin={bin_idx}  "
#                       f"bin_counts={counts}  "
#                       f"tick_eff={step/total_ticks:.1%}")

#     finally:
#         # ── 5. Clean up ──────────────────────────────────────────────────────
#         for h in hook_handles:
#             h.remove()
#         lidar.stop();  lidar.destroy()
#         for cam in cameras.values():
#             cam.stop();  cam.destroy()
#         if is_alive(ego):
#             ego.destroy()
#         if is_alive(npc):
#             npc.destroy()
#         print("\nCARLA cleaned up.")

#     # ── 6. Save to HDF5 ──────────────────────────────────────────────────────
#     gt_distances = np.array(gt_distances, dtype=np.float32)
#     bin_labels   = np.array(bin_labels,   dtype=np.int32)
#     n_collected  = len(gt_distances)
#     print(f"\n── Saving {n_collected} samples ──")

#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     with h5py.File(output_path, "w") as f:
#         f.create_dataset("gt_distance", data=gt_distances)
#         f.create_dataset("bin_label",   data=bin_labels)

#         grp = f.create_group("latents")
#         for layer_name, acts in activation_buffer.items():
#             arr = np.concatenate(acts, axis=0)
#             if arr.shape[0] != n_collected:
#                 print(f"  Warning: {layer_name} has {arr.shape[0]} samples "
#                       f"(expected {n_collected}) — skipping")
#                 continue
#             grp.create_dataset(layer_name, data=arr, compression="gzip")
#             print(f"  Saved {layer_name}: shape={arr.shape}")

#     # Print distribution summary
#     print(f"\nDataset saved → {output_path}")
#     print(f"  Total samples : {n_collected}")
#     print(f"  Distance range: [{gt_distances.min():.1f}, {gt_distances.max():.1f}] m")
#     print(f"  Distance mean : {gt_distances.mean():.1f} m  std={gt_distances.std():.1f} m")
#     print(f"\n  Bin distribution:")
#     for i, (lo, hi) in enumerate(DISTANCE_BINS):
#         count = int((bin_labels == i).sum())
#         print(f"    bin{i} [{lo:4.1f}–{hi:4.1f}m]: {count} samples ({100*count/n_collected:.1f}%)")

#     print(f"\nRun probe sweep with:")
#     print(f"  python run_probe_sweep.py --data {output_path} --save-figures")


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser(description="Collect TransFuser activations in CARLA")
#     p.add_argument("--model-ckpt", required=True,
#                    help="Path to folder containing TransFuser .pth weights")
#     p.add_argument("--output",     default="data/latents.h5")
#     p.add_argument("--n-steps",    type=int, default=5000)
#     p.add_argument("--town",       default="Town01")
#     p.add_argument("--host",       default="localhost")
#     p.add_argument("--port",       type=int, default=2000)
#     p.add_argument("--seed",       type=int, default=42)
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     collect(args)















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
    (0.5,  3.0),   # bin 0: very close
    (3.0,  5.0),   # bin 1: close following
    (5.0,  10.0),  # bin 2: normal following
    (10.0, 20.0),  # bin 3: medium range
    (20.0, 50.0),  # bin 4: far
]

TELEPORT_EVERY = 5  # kept for backwards-compat, not used in v4

# Ego driving config
EGO_THROTTLE          = 0.3
WAYPOINT_LOOKAHEAD    = 5.0    # metres ahead for steering target
STEER_GAIN            = 1.5
STUCK_SPEED_THRESHOLD = 0.5    # m/s
STUCK_CONSECUTIVE     = 40     # consecutive slow steps before respawn

# NPC throttle randomization (v4)
NPC_THROTTLE_OPTIONS      = [0.0, 0.15, 0.25, 0.35, 0.45]
NPC_THROTTLE_CHANGE_EVERY = 30      # ticks (~3s at 0.1s/tick)
INITIAL_NPC_DIST_RANGE    = (5.0, 15.0)   # initial placement (metres ahead)
DIST_RECOVERY_MAX         = 80.0    # re-teleport NPC if distance exceeds this
MAX_VALID_DISTANCE        = DISTANCE_BINS[-1][1]   # 50m, beyond = skip sample


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
    return client, world


# ── NEW: Waypoint-following steering ─────────────────────────────────────────

def compute_waypoint_steer(vehicle, world_map) -> float:
    """
    Compute steering in [-1, 1] to follow the road via CARLA's waypoint API.
    Uses the cross product between vehicle's forward vector and the direction
    to the next waypoint WAYPOINT_LOOKAHEAD metres ahead along the lane.
    Works for any vehicle (ego or NPC).
    """
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
    """
    Return a Transform `dist` metres ahead of ego along the ROAD NETWORK.
    Used for NPC placement so it always lands on the road, not in a building.
    Falls back to raw forward vector if no road waypoint is available.
    """
    ego_loc = ego.get_location()
    wp = world_map.get_waypoint(ego_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        ego_tf = ego.get_transform()
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
    tf.location.z += 0.5   # lift to avoid spawn-in-ground
    return tf


def spawn_ego_vehicle(world):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    # Filter spawn points to flat ground only (no bridges)
    spawn_points = [sp for sp in world.get_map().get_spawn_points()
                    if sp.location.z < 1.0]
    if not spawn_points:
        spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned ego vehicle at {spawn_point.location}")
    return vehicle


def warmup_ego(ego, world_map):
    """
    Apply throttle until ego starts moving. In async mode CARLA needs
    ~30 physics ticks after spawn before the vehicle responds to controls.
    """
    print("Warming up ego vehicle...")
    for warmup_step in range(60):  # up to 3s at 20Hz
        steer = compute_waypoint_steer(ego, world_map)
        ego.apply_control(carla.VehicleControl(
            throttle=EGO_THROTTLE, steer=steer, brake=0.0))
        time.sleep(0.05)
        v = ego.get_velocity()
        spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        if spd > 0.5:
            print(f"  Ego moving at {spd:.2f} m/s after {warmup_step+1} ticks")
            return True
    print("  WARNING: ego still not moving after warmup")
    return False


def spawn_npc(world):
    """Spawn a single NPC vehicle at an arbitrary spawn point."""
    bp_lib = world.get_blueprint_library()
    npc_bp = random.choice(bp_lib.filter("vehicle.*"))
    spawn_points = [sp for sp in world.get_map().get_spawn_points()
                    if sp.location.z < 1.0]
    if not spawn_points:
        spawn_points = world.get_map().get_spawn_points()
    sp = random.choice(spawn_points)
    npc = world.try_spawn_actor(npc_bp, sp)
    return npc


def spawn_cameras(world, vehicle, config):
    bp_lib = world.get_blueprint_library()
    cameras = {}
    for cam_id, rot in [('rgb_left',  config.camera_rot_1),
                         ('rgb_front', config.camera_rot_0),
                         ('rgb_right', config.camera_rot_2)]:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(config.img_width))
        cam_bp.set_attribute("image_size_y", str(config.img_resolution[0]))
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


def teleport_npc_ahead(npc, ego, world_map, target_distance: float) -> bool:
    """
    Teleport the NPC `target_distance` metres ahead of the ego along the
    ROAD NETWORK (uses waypoints, not raw forward vector). This ensures
    the NPC always lands on a drivable road, even if the ego is mid-turn.
    """
    if not is_alive(npc) or not is_alive(ego):
        return False

    target_tf = get_waypoint_ahead_transform(ego, world_map, target_distance)
    npc.set_transform(target_tf)
    npc.set_target_velocity(carla.Vector3D(0, 0, 0))
    return True


def get_distance(actor_a, actor_b) -> float:
    return actor_a.get_location().distance(actor_b.get_location())


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
    world_map = world.get_map()

    ego = spawn_ego_vehicle(world)

    # Warmup so ego is moving before main loop starts
    warmup_ego(ego, world_map)

    # Single controlled NPC
    npc = spawn_npc(world)
    if npc is None:
        raise RuntimeError("Failed to spawn NPC vehicle — no free spawn points.")
    print("Spawned NPC vehicle")

    # Spawn sensors
    cameras = spawn_cameras(world, ego, config)
    lidar   = spawn_lidar(world, ego, config)

    rgb_queues  = {cam_id: [] for cam_id in cameras}
    lidar_queue = []
    for cam_id, cam in cameras.items():
        cam.listen(lambda data, cid=cam_id: rgb_queues[cid].append(data))
    lidar.listen(lambda data: lidar_queue.append(data))

    time.sleep(2.0)  # give sensors time to start producing data

    # ── 4. Collection loop ───────────────────────────────────────────────────
    print(f"\n── Collecting {args.n_steps} steps ──\n")

    gt_distances   = []
    bin_labels     = []
    step           = 0
    total_ticks    = 0
    stuck_count    = 0

    # NPC throttle state — changes every NPC_THROTTLE_CHANGE_EVERY ticks
    npc_throttle   = random.choice(NPC_THROTTLE_OPTIONS)
    npc_throttle_ticks = 0
    recovery_teleports = 0   # count of distance-overflow re-teleports

    # ── Initial NPC placement ────────────────────────────────────────────────
    initial_dist = np.random.uniform(*INITIAL_NPC_DIST_RANGE)
    teleport_npc_ahead(npc, ego, world_map, initial_dist)
    time.sleep(0.3)

    def bin_for_distance(d: float) -> Optional[int]:
        """Return the bin index for a given distance, or None if out of range."""
        for i, (lo, hi) in enumerate(DISTANCE_BINS):
            if lo <= d < hi:
                return i
        return None

    try:
        while step < args.n_steps:
            time.sleep(0.1)
            total_ticks += 1

            # ── Ego keepalive with waypoint steering ─────────────────────────
            if not is_alive(ego):
                print("Ego destroyed — stopping.")
                break

            ego_steer = compute_waypoint_steer(ego, world_map)
            ego.apply_control(carla.VehicleControl(
                throttle=EGO_THROTTLE, steer=ego_steer, brake=0.0))

            # ── NPC autonomous driving with random throttle ──────────────────
            if is_alive(npc):
                # Pick new throttle periodically
                if npc_throttle_ticks >= NPC_THROTTLE_CHANGE_EVERY:
                    npc_throttle = random.choice(NPC_THROTTLE_OPTIONS)
                    npc_throttle_ticks = 0
                npc_throttle_ticks += 1

                npc_steer = compute_waypoint_steer(npc, world_map)
                npc.apply_control(carla.VehicleControl(
                    throttle=npc_throttle, steer=npc_steer, brake=0.0))
            else:
                # NPC died — respawn and reset
                print("  NPC died — respawning")
                npc = spawn_npc(world)
                if npc is not None:
                    teleport_npc_ahead(npc, ego, world_map,
                                       np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                    time.sleep(0.3)
                continue

            # ── Stuck detection (ego) ────────────────────────────────────────
            v = ego.get_velocity()
            ego_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            if ego_speed < STUCK_SPEED_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= STUCK_CONSECUTIVE:
                print(f"  Ego stuck (speed={ego_speed:.2f}m/s for "
                      f"{stuck_count} ticks) — respawning")
                spawn_points = [sp for sp in world_map.get_spawn_points()
                                if sp.location.z < 1.0]
                ego.set_transform(random.choice(spawn_points))
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                time.sleep(1.0)
                stuck_count = 0
                # Re-teleport NPC ahead of the new ego position
                teleport_npc_ahead(npc, ego, world_map,
                                   np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                time.sleep(0.3)
                for q in rgb_queues.values():
                    q.clear()
                lidar_queue.clear()
                continue

            # ── Compute current distance ─────────────────────────────────────
            gt_dist = get_distance(ego, npc)

            # ── Safety re-teleport if NPC drifts too far ─────────────────────
            # (NPC took different turn, or drove off into the distance)
            if gt_dist > DIST_RECOVERY_MAX:
                print(f"  NPC drifted to {gt_dist:.1f}m — re-teleporting "
                      f"(recovery #{recovery_teleports+1})")
                teleport_npc_ahead(npc, ego, world_map,
                                   np.random.uniform(*INITIAL_NPC_DIST_RANGE))
                recovery_teleports += 1
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

            # ── Preprocess ───────────────────────────────────────────────────
            rgb_tensor   = preprocess_rgb(left_raw, front_raw, right_raw, config)
            lidar_tensor = preprocess_lidar(lidar_data)

            # ── Bin assignment from observed distance ────────────────────────
            bin_idx = bin_for_distance(gt_dist)
            if bin_idx is None:
                # Distance beyond max bin (>50m but < recovery threshold) — skip
                continue

            # ── Forward pass — hooks fire here ───────────────────────────────
            with torch.no_grad():
                ego_vel = torch.tensor([[ego_speed]], dtype=torch.float32).to("cuda")
                _ = model._model(rgb_tensor, lidar_tensor, ego_vel)

            gt_distances.append(gt_dist)
            bin_labels.append(bin_idx)
            step += 1

            if step % 100 == 0:
                counts = [bin_labels.count(i) for i in range(len(DISTANCE_BINS))]
                print(f"  Step {step}/{args.n_steps}  "
                      f"gt_dist={gt_dist:.1f}m  "
                      f"bin={bin_idx}  "
                      f"ego_spd={ego_speed:.1f}m/s  "
                      f"npc_thr={npc_throttle:.2f}  "
                      f"bin_counts={counts}  "
                      f"tick_eff={step/total_ticks:.1%}")

    finally:
        # ── 5. Clean up ──────────────────────────────────────────────────────
        for h in hook_handles:
            h.remove()
        lidar.stop();  lidar.destroy()
        for cam in cameras.values():
            cam.stop();  cam.destroy()
        if is_alive(ego):
            ego.destroy()
        if is_alive(npc):
            npc.destroy()
        print("\nCARLA cleaned up.")

    # ── 6. Save to HDF5 ──────────────────────────────────────────────────────
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
    print(f"  Recovery teleports : {recovery_teleports} (NPC drifted beyond {DIST_RECOVERY_MAX}m)")
    print(f"\n  Bin distribution:")
    for i, (lo, hi) in enumerate(DISTANCE_BINS):
        count = int((bin_labels == i).sum())
        pct = 100*count/n_collected if n_collected > 0 else 0
        print(f"    bin{i} [{lo:4.1f}–{hi:4.1f}m]: {count} samples ({pct:.1f}%)")

    print(f"\nRun probe sweep with:")
    print(f"  python run_probe_sweep.py --data {output_path} --save-figures")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect TransFuser activations in CARLA")
    p.add_argument("--model-ckpt", required=True,
                   help="Path to folder containing TransFuser .pth weights")
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