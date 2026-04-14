"""
collect_data.py
---------------
Runs TransFuser in CARLA, captures per-layer activations via PyTorch forward
hooks, and saves them alongside ground-truth distances to an HDF5 file.

Usage (on a GPU node, after launching CARLA in background):
    ./CarlaUE4.sh -opengl -RenderOffScreen &
    sleep 10
    python collect_data.py \
        --model-ckpt /path/to/model_ckpt/transfuser \
        --output     /path/to/latents.h5 \
        --n-steps    5000 \
        --town       Town01

The output HDF5 file is directly readable by your existing probe pipeline:
    python run_probe_sweep.py --data /path/to/latents.h5 --save-figures
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
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
# ── CARLA + TransFuser imports ───────────────────────────────────────────────
try:
    import carla
except ImportError:
    raise ImportError(
        "CARLA Python API not found. Make sure PYTHONPATH includes:\n"
        "  $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg\n"
        "  $CARLA_ROOT/PythonAPI/carla"
    )

# TransFuser model — path depends on where you cloned the repo
TRANSFUSER_ROOT = Path(os.environ.get(
    "TRANSFUSER_ROOT",
    "/ocean/projects/cis250201p/jjain2/transfuser"
))
sys.path.insert(0, str(TRANSFUSER_ROOT / "team_code_transfuser"))

from model import LidarCenterNet          # TransFuser model class
from config import GlobalConfig           # TransFuser config
from data import scale_image_cv2, crop_image_cv2, lidar_to_histogram_features
from copy import deepcopy

#This is directly taken from the transfuser repo. 
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

#This is also directly taken from the transfuser repo. 
def shift_x_scale_crop(image, scale, crop, crop_shift=0):
    crop_h, crop_w = crop
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.array(im_resized)
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    
    # only shift in x direction
    start_x += int(crop_shift // scale)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def is_alive(actor) -> bool:
    try:
        return actor is not None and actor.is_alive
    except RuntimeError:
        return False


# ── Distance bucketing (mirrors data_utils.py) ───────────────────────────────
DISTANCE_BINS = [0.0, 5.0, 15.0, 40.0, np.inf]

def bucket_distance(d: float) -> int:
    for i, threshold in enumerate(DISTANCE_BINS[1:-1]):
        if d < threshold:
            return i
    return len(DISTANCE_BINS) - 2


# ── Hook registration ────────────────────────────────────────────────────────

def register_hooks(model: torch.nn.Module) -> Tuple[Dict, List]:
    """
    Register forward hooks on TransFuser's key layers.

    Returns
    -------
    buffer : dict  {layer_name: list of activation arrays}
    handles: list of hook handles (call handle.remove() to clean up)
    """
    buffer  = defaultdict(list)
    handles = []

    def make_hook(name: str):
        def hook(module, input, output):
            # output may be a tensor or tuple — take first tensor
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            # Global average pool spatial dims → (batch, C)
            if out.dim() == 4:   # (B, C, H, W)
                out = out.mean(dim=[2, 3])
            elif out.dim() == 3: # (B, seq_len, C) — transformer output
                out = out.mean(dim=1)
            buffer[name].append(out.detach().cpu().float().numpy())
        return hook
    
    backbone = model._model  # TransfuserBackbone

    # ── Image encoder (RegNet stages) ────────────────────────────────────────
    for i, stage_name in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            layer = getattr(backbone.image_encoder.features, stage_name)
            h = layer.register_forward_hook(make_hook(f"img_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: img_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: img_enc.{stage_name} not found, skipping")

    # ── Lidar encoder (same RegNet structure) ─────────────────────────────────
    for i, stage_name in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            layer = getattr(backbone.lidar_encoder._model, stage_name)
            h = layer.register_forward_hook(make_hook(f"lid_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: lid_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: lid_enc.{stage_name} not found, skipping")

    # ── Transformer fusion blocks (4 separate GPT blocks, one per scale) ─────────────────────────────────────────────
    for i in range(1,5):
        try:
            transformer = getattr(backbone, f"transformer{i}")
            h = transformer.register_forward_hook(make_hook(f"transformer{i}"))
            handles.append(h)
            print(f"  Hooked: transformer{i}")
        except AttributeError:
            print(f"  Warning: transformer{i} not found, skipping")

    return buffer, handles


def print_model_structure(model: torch.nn.Module, max_depth: int = 3):
    """
    Utility: print named modules up to max_depth.
    Run this first to verify layer names if hooks aren't firing.
    """
    print("\n── Model structure (first 3 levels) ──")
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth <= max_depth and name:
            indent = "  " * depth
            print(f"{indent}{name}: {type(module).__name__}")
    print()


# ── CARLA utilities ───────────────────────────────────────────────────────────

def setup_carla(host: str = "localhost", port: int = 2000, town: str = "Town01"):
    """Connect to CARLA and set up a simple driving world."""
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    print(f"Connected to CARLA {client.get_server_version()}")
    world = client.get_world()  # get world reference after loading new map
    # settings = world.get_settings()
    # settings.synchronous_mode = False   # Not sure about this
    # settings.fixed_delta_seconds = 0.0  
    # world.apply_settings(settings)

    # traffic_manager = client.get_trafficmanager(8000)
    # traffic_manager.set_synchronous_mode(False)
    # traffic_manager.set_global_distance_to_leading_vehicle(2.0)  # NPCs follow closely for more distance variety
    # traffic_manager.global_percentage_speed_difference(20.0)  # slow NPCs down
    print("Set up CARLA")

    return client, world


def spawn_ego_vehicle(world):
    """Spawn a Tesla Model 3 at a random spawn point."""
    bp_lib   = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")

    spawn_points = world.get_map().get_spawn_points()
    spawn_point  = np.random.choice(spawn_points)

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned ego vehicle at {spawn_point.location}")
    return vehicle


def spawn_cameras(world,vehicle,config):
    bp_lib = world.get_blueprint_library()
    cameras = {}
    for cam_id, rot in [('rgb_left', config.camera_rot_1), ('rgb_front', config.camera_rot_0), ('rgb_right', config.camera_rot_2)]:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(config.img_width))
        cam_bp.set_attribute("image_size_y", str(config.img_resolution[0]))
        cam_bp.set_attribute("fov", str(config.camera_fov))
        transform = carla.Transform(carla.Location(x=config.camera_pos[0], y = config.camera_pos[1], z = config.camera_pos[2]),carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]))
        cameras[cam_id] = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    return cameras

def spawn_lidar(world, vehicle,config):
    """Attach a LiDAR sensor to the ego vehicle."""
    bp_lib   = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    transform = carla.Transform(carla.Location(x=config.lidar_pos[0], y = config.lidar_pos[1], z = config.lidar_pos[2]),carla.Rotation(roll=config.lidar_rot[0], pitch=config.lidar_rot[1], yaw=config.lidar_rot[2]))
    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    return lidar


def get_nearest_vehicle_distance(npc_vehicles, ego_vehicle) -> float:
    """
    Ground truth: Euclidean distance from ego to nearest other vehicle.
    Uses CARLA's scene graph — exact, no sensor noise.
    Returns 999.0 if no other vehicles are present.
    """
    ego_loc = ego_vehicle.get_location()
    min_dist = 999.0
    for v in npc_vehicles:
        if not v.is_alive:
            continue
        loc = v.get_location()
        dist = ego_loc.distance(loc)
        if dist < min_dist:
            min_dist = dist
    return min_dist



def preprocess_rgb(left_raw, front_raw, right_raw, config) -> torch.Tensor:
    rgb = []
    for img_raw in [left_raw, front_raw, right_raw]:
        img_pil = Image.fromarray(cv2.cvtColor(img_raw[:,:,:3], cv2.COLOR_BGR2RGB))
        cropped = scale_crop(img_pil, scale=config.scale, crop_x=config.img_width, crop_y=config.img_resolution[0])
        rgb.append(cropped)
    rgb = np.concatenate(rgb, axis=1)  # matches tick() concatenation
    image = Image.fromarray(rgb)
    tensor_np = shift_x_scale_crop(image, scale=config.scale, crop=config.img_resolution, crop_shift=0)
    return torch.from_numpy(tensor_np.copy()).unsqueeze(0).to('cuda', dtype=torch.float32)

def carla_img_to_numpy(image_carla):
    array = np.frombuffer(image_carla.raw_data, dtype=np.uint8)
    return array.reshape((image_carla.height, image_carla.width, 4))[:, :, :3]  # drop alpha


def preprocess_lidar(lidar_carla) -> torch.Tensor:
    # """
    # Convert CARLA LiDAR point cloud to a BEV (bird's eye view) image tensor.
    # This is a simplified version — TransFuser uses a more elaborate BEV encoding.
    # """
    # points = np.frombuffer(lidar_carla.raw_data, dtype=np.float32)
    # points = points.reshape((-1, 4))[:, :3]   # (N, 3) — drop intensity

    # # Project to 2D BEV grid (256x256, ±50m range)
    # bev_size = 256
    # bev_range = 50.0

    # bev = np.zeros((bev_size, bev_size), dtype=np.float32)
    # x_idx = ((points[:, 0] + bev_range) / (2 * bev_range) * bev_size).astype(int)
    # y_idx = ((points[:, 1] + bev_range) / (2 * bev_range) * bev_size).astype(int)

    # mask  = (x_idx >= 0) & (x_idx < bev_size) & (y_idx >= 0) & (y_idx < bev_size)
    # bev[y_idx[mask], x_idx[mask]] = 1.0

    # # TransFuser expects 2-channel BEV; duplicate for simplicity
    # bev_tensor = torch.from_numpy(
    #     np.stack([bev, bev], axis=0)
    # ).unsqueeze(0).cuda()   # (1, 2, 256, 256)
    points = np.frombuffer(lidar_carla.raw_data, dtype=np.float32).reshape(-1,4)
    lidar = deepcopy(points[:,:3])
    lidar[:,1] *= -1
    bev_tensor = lidar_to_histogram_features(lidar)
    return torch.from_numpy(bev_tensor).unsqueeze(0).to('cuda', dtype=torch.float32)   # (1, 8, 256, 256)



# ── Main collection loop ──────────────────────────────────────────────────────

def collect(args):
    # ── 1. Load TransFuser ───────────────────────────────────────────────────
    print("\n── Loading TransFuser ──")
    config = GlobalConfig(setting = 'eval')
    model  = LidarCenterNet(config, "cuda", backbone="transFuser",
                            image_architecture="regnety_032",
                            lidar_architecture="regnety_032",
                            use_velocity=0)

    # Load pretrained weights
    ckpt_dir = Path(args.model_ckpt)
    ckpt_files = list(ckpt_dir.glob("*.pth"))
    assert len(ckpt_files) > 0, f"No .pth files found in {ckpt_dir}"
    ckpt = torch.load(ckpt_files[0], map_location="cuda")
    model.load_state_dict(ckpt, strict=False)
    model.eval().cuda()
    print(f"Loaded weights from {ckpt_files[0]}")

    # Optionally print model structure for debugging
    if args.print_model:
        print_model_structure(model)

    # ── 2. Register hooks ────────────────────────────────────────────────────
    print("\n── Registering hooks ──")
    activation_buffer, hook_handles = register_hooks(model)

    # ── 3. Connect to CARLA ──────────────────────────────────────────────────
    print("\n── Connecting to CARLA ──")
    client, world = setup_carla(
        host=args.host, port=args.port, town=args.town)
    print("in main loop after setup carla")
    # Spawn some NPC vehicles so there are objects to measure distance to
    npc_vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    print("after spawn points")
    np.random.shuffle(spawn_points)
    for sp in spawn_points[:args.n_npc]:
        npc_bp = np.random.choice(
            world.get_blueprint_library().filter("vehicle.*"))
        print("after npc_bp")
        npc = world.try_spawn_actor(npc_bp, sp)
        print("after npc")
        if npc:
            npc.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
            npc_vehicles.append(npc)

    print(f"Spawned {len(npc_vehicles)} NPC vehicles")

    # Spawn ego vehicle + sensors
    ego    = spawn_ego_vehicle(world)
    ego.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
    print("Warming up simulation (3 seconds)...")
    time.sleep(3.0)

    cameras = spawn_cameras(world, ego, config)
    lidar  = spawn_lidar(world, ego, config)

    # Sensor data queues
    rgb_queues = {cam_id: [] for cam_id in cameras}
    lidar_queue = []
    for cam_id, cam in cameras.items():
        cam.listen(lambda data, cid=cam_id: rgb_queues[cid].append(data))
    lidar.listen(lambda data: lidar_queue.append(data))


    # ── 4. Collection loop ───────────────────────────────────────────────────
    print(f"\n── Collecting {args.n_steps} steps in {args.town} ──\n")

    # Let world run for a bit before collecting
    time.sleep(3.0)


    gt_distances = []
    step = 0
    total_ticks = 0
    try:
        while step < args.n_steps:
            time.sleep(0.1) #~10Hz collection rate
            total_ticks += 1
            if not is_alive(ego):
                print("Ego vehicle destroyed, ending collection")
                break
            
            # Prune dead NPCs to keep the distance query fast and safe
            npc_vehicles = [v for v in npc_vehicles if v.is_alive]
            # Keep vehicles moving
            if step % 20 == 0:
                if is_alive(ego):
                    ego.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
                for v in npc_vehicles:
                    v.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))


            if not all(rgb_queues[c] for c in rgb_queues) or not lidar_queue:
                continue
            
            left = rgb_queues['rgb_left'][-1]; rgb_queues['rgb_left'].clear()
            front = rgb_queues['rgb_front'][-1]; rgb_queues['rgb_front'].clear()
            right = rgb_queues['rgb_right'][-1]; rgb_queues['rgb_right'].clear()
            left_raw = carla_img_to_numpy(left)
            front_raw = carla_img_to_numpy(front)
            right_raw = carla_img_to_numpy(right)
            rgb_tensor = preprocess_rgb(left_raw, front_raw, right_raw,config)
            lidar_data = lidar_queue[-1]; lidar_queue.clear()
            lidar_tensor = preprocess_lidar(lidar_data)
            # Ground truth distance from scene graph
            gt_dist = get_nearest_vehicle_distance(npc_vehicles,ego)
            # if gt_dist > 80.0:
            #     # Skip timesteps with no nearby vehicles — not useful for probing
            #     continue
            gt_dist = min(gt_dist, 100.0)  # cap instead of filter

            # Forward pass (hooks fire here)
            with torch.no_grad():
                v = ego.get_velocity()
                speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
                ego_vel = torch.tensor([[speed]], dtype=torch.float32).to("cuda")
                _ = model._model(rgb_tensor, lidar_tensor, ego_vel)


            gt_distances.append(gt_dist)
            step += 1

            if step % 100 == 0:
                print(f"  Step {step}/{args.n_steps}  "
                  f"gt_dist={gt_dist:.1f}m  "
                  f"tick_efficiency={step/total_ticks:.1%}")
    finally:
        # ── 5. Clean up ──────────────────────────────────────────────────────
        for h in hook_handles:
            h.remove()
        lidar.stop();  lidar.destroy()
        for cam in cameras.values():
            cam.stop();  cam.destroy()
        if is_alive(ego):
            ego.destroy()
        for npc in npc_vehicles:
            if is_alive(npc):
                npc.destroy()

        # Restore async mode
        # settings = world.get_settings()
        # settings.synchronous_mode = False
        # world.apply_settings(settings)
        print("\nCARLA cleaned up.")

    # ── 6. Save to HDF5 ──────────────────────────────────────────────────────
    gt_distances = np.array(gt_distances, dtype=np.float32)
    n_collected  = len(gt_distances)
    print(f"\n── Saving {n_collected} samples ──")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("gt_distance", data=gt_distances)

        grp = f.create_group("latents")
        for layer_name, acts in activation_buffer.items():
            arr = np.concatenate(acts, axis=0)  # (N, D)
            if arr.shape[0] != n_collected:
                print(f"  Warning: {layer_name} has {arr.shape[0]} samples, "
                      f"expected {n_collected} — skipping")
                continue
            grp.create_dataset(layer_name, data=arr, compression="gzip")
            print(f"  Saved {layer_name}: shape={arr.shape}")

    print(f"\nDataset saved → {output_path}")
    print(f"  Samples:       {n_collected}")
    print(f"  Layers saved:  {len(activation_buffer)}")
    print(f"  Distance range: [{gt_distances.min():.1f}, {gt_distances.max():.1f}] m")
    print(f"\nRun probe sweep with:")
    print(f"  python run_probe_sweep.py --data {output_path} --save-figures")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect TransFuser activations in CARLA")
    p.add_argument("--model-ckpt",   required=True,
                   help="Path to folder containing TransFuser .pth weights")
    p.add_argument("--output",       default="data/latents.h5",
                   help="Output HDF5 path (default: data/latents.h5)")
    p.add_argument("--n-steps",      type=int, default=5000,
                   help="Number of timesteps to collect (default: 5000)")
    p.add_argument("--town",         default="Town01",
                   help="CARLA town to use (default: Town01)")
    p.add_argument("--n-npc",        type=int, default=20,
                   help="Number of NPC vehicles to spawn (default: 20)")
    p.add_argument("--host",         default="localhost")
    p.add_argument("--port",         type=int, default=2000)
    p.add_argument("--print-model",  action="store_true",
                   help="Print model layer names before collecting (useful for debugging hooks)")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    collect(args)