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

# ── CARLA + TransFuser imports ───────────────────────────────────────────────
# These are available after activating tfuse env and setting PYTHONPATH
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

    Layer naming convention matches the mock data in data_utils.py:
        image_encoder.layer{1,2,3,4}   ← RegNet image encoder stages
        lidar_encoder.layer{1,2,3,4}   ← RegNet lidar encoder stages
        transformer.block{0,1,2,3}     ← Transformer fusion blocks
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

    # ── Image encoder (RegNet stages) ────────────────────────────────────────
    # RegNet in timm has s1, s2, s3, s4 stages
    for i, stage_name in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            layer = getattr(model.image_encoder, stage_name)
            h = layer.register_forward_hook(make_hook(f"img_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: img_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: img_enc.{stage_name} not found, skipping")

    # ── Lidar encoder (same RegNet structure) ─────────────────────────────────
    for i, stage_name in enumerate(["s1", "s2", "s3", "s4"]):
        try:
            layer = getattr(model.lidar_encoder, stage_name)
            h = layer.register_forward_hook(make_hook(f"lid_enc.stage{i+1}"))
            handles.append(h)
            print(f"  Hooked: lid_enc.stage{i+1}")
        except AttributeError:
            print(f"  Warning: lid_enc.{stage_name} not found, skipping")

    # ── Transformer fusion blocks ─────────────────────────────────────────────
    # TransFuser has n_layer=4 transformer blocks fusing image+lidar features
    try:
        for i, block in enumerate(model.transformer.layers):
            h = block.register_forward_hook(make_hook(f"transformer.block{i}"))
            handles.append(h)
            print(f"  Hooked: transformer.block{i}")
    except AttributeError:
        print("  Warning: model.transformer.layers not found")

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

    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode    = True   # Python controls the tick
    settings.fixed_delta_seconds = 0.05   # 20 Hz
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    return client, world, traffic_manager


def spawn_ego_vehicle(world):
    """Spawn a Tesla Model 3 at a random spawn point."""
    bp_lib   = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")

    spawn_points = world.get_map().get_spawn_points()
    spawn_point  = np.random.choice(spawn_points)

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned ego vehicle at {spawn_point.location}")
    return vehicle


def spawn_rgb_camera(world, vehicle):
    """Attach a front-facing RGB camera to the ego vehicle."""
    bp_lib    = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "900")
    camera_bp.set_attribute("image_size_y", "256")
    camera_bp.set_attribute("fov",          "100")

    transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera    = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera


def spawn_lidar(world, vehicle):
    """Attach a LiDAR sensor to the ego vehicle."""
    bp_lib   = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range",             "50")
    lidar_bp.set_attribute("rotation_frequency","10")
    lidar_bp.set_attribute("channels",          "32")
    lidar_bp.set_attribute("points_per_second", "300000")

    transform = carla.Transform(carla.Location(x=0.0, z=2.5))
    lidar     = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    return lidar


def get_nearest_vehicle_distance(world, ego_vehicle) -> float:
    """
    Ground truth: Euclidean distance from ego to nearest other vehicle.
    Uses CARLA's scene graph — exact, no sensor noise.
    Returns 999.0 if no other vehicles are present.
    """
    ego_loc  = ego_vehicle.get_location()
    vehicles = world.get_actors().filter("vehicle.*")

    min_dist = 999.0
    for v in vehicles:
        if v.id == ego_vehicle.id:
            continue
        dist = ego_loc.distance(v.get_location())
        if dist < min_dist:
            min_dist = dist

    return min_dist


def preprocess_rgb(image_carla) -> torch.Tensor:
    """Convert CARLA RGB image to a normalised (1, 3, H, W) tensor."""
    import cv2
    array = np.frombuffer(image_carla.raw_data, dtype=np.uint8)
    array = array.reshape((image_carla.height, image_carla.width, 4))
    rgb   = array[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB

    # Resize to 256x256 (TransFuser input resolution)
    rgb   = cv2.resize(rgb, (256, 256))
    rgb   = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.cuda()


def preprocess_lidar(lidar_carla) -> torch.Tensor:
    """
    Convert CARLA LiDAR point cloud to a BEV (bird's eye view) image tensor.
    This is a simplified version — TransFuser uses a more elaborate BEV encoding.
    """
    points = np.frombuffer(lidar_carla.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))[:, :3]   # (N, 3) — drop intensity

    # Project to 2D BEV grid (256x256, ±50m range)
    bev_size = 256
    bev_range = 50.0

    bev = np.zeros((bev_size, bev_size), dtype=np.float32)
    x_idx = ((points[:, 0] + bev_range) / (2 * bev_range) * bev_size).astype(int)
    y_idx = ((points[:, 1] + bev_range) / (2 * bev_range) * bev_size).astype(int)

    mask  = (x_idx >= 0) & (x_idx < bev_size) & (y_idx >= 0) & (y_idx < bev_size)
    bev[y_idx[mask], x_idx[mask]] = 1.0

    # TransFuser expects 2-channel BEV; duplicate for simplicity
    bev_tensor = torch.from_numpy(
        np.stack([bev, bev], axis=0)
    ).unsqueeze(0).cuda()   # (1, 2, 256, 256)

    return bev_tensor


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
    client, world, traffic_manager = setup_carla(
        host=args.host, port=args.port, town=args.town)

    # Spawn some NPC vehicles so there are objects to measure distance to
    npc_vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    np.random.shuffle(spawn_points)
    for sp in spawn_points[:args.n_npc]:
        npc_bp = np.random.choice(
            world.get_blueprint_library().filter("vehicle.*"))
        npc = world.try_spawn_actor(npc_bp, sp)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            npc_vehicles.append(npc)
    print(f"Spawned {len(npc_vehicles)} NPC vehicles")

    # Spawn ego vehicle + sensors
    ego    = spawn_ego_vehicle(world)
    camera = spawn_rgb_camera(world, ego)
    lidar  = spawn_lidar(world, ego)
    ego.set_autopilot(True, traffic_manager.get_port())

    # Sensor data queues
    rgb_queue   = []
    lidar_queue = []
    camera.listen(lambda data: rgb_queue.append(data))
    lidar.listen(lambda data: lidar_queue.append(data))

    # ── 4. Collection loop ───────────────────────────────────────────────────
    print(f"\n── Collecting {args.n_steps} steps in {args.town} ──\n")

    gt_distances = []
    step = 0

    try:
        while step < args.n_steps:
            world.tick()

            if not rgb_queue or not lidar_queue:
                continue

            rgb_data   = rgb_queue.pop(0)
            lidar_data = lidar_queue.pop(0)

            # Ground truth distance from scene graph
            gt_dist = get_nearest_vehicle_distance(world, ego)
            if gt_dist > 80.0:
                # Skip timesteps with no nearby vehicles — not useful for probing
                continue

            # Forward pass (hooks fire here)
            with torch.no_grad():
                rgb_tensor   = preprocess_rgb(rgb_data)
                lidar_tensor = preprocess_lidar(lidar_data)
                _ = model(rgb_tensor, lidar_tensor)

            gt_distances.append(gt_dist)
            step += 1

            if step % 100 == 0:
                print(f"  Step {step}/{args.n_steps}  "
                      f"gt_dist={gt_dist:.1f}m  "
                      f"layers={list(activation_buffer.keys())[:3]}...")

    finally:
        # ── 5. Clean up ──────────────────────────────────────────────────────
        for h in hook_handles:
            h.remove()
        camera.stop(); camera.destroy()
        lidar.stop();  lidar.destroy()
        ego.destroy()
        for npc in npc_vehicles:
            npc.destroy()

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
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
    p.add_argument("--n-npc",        type=int, default=50,
                   help="Number of NPC vehicles to spawn (default: 50)")
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