"""
data_utils.py
-------------
Defines the expected data format and provides utilities for loading/saving.

Expected HDF5 schema (output of data collection script):
    /latents/{layer_name}   -> float32 array (N, D_layer)
    /gt_distance            -> float32 array (N,)          # metric distance in meters
    /observations           -> uint8 array   (N, H, W, C)  # optional, for sanity checks

Example layer names for TransFuser:
    "encoder.layer0", "encoder.layer1", ..., "transformer.block0", ...
"""

import h5py
import numpy as np
from pathlib import Path


# ── Distance bucketing ──────────────────────────────────────────────────────
# These thresholds define "near / mid / far" — adjust to your CARLA scene.
DISTANCE_BINS   = [0.0, 5.0, 15.0, 40.0, np.inf]   # meters
DISTANCE_LABELS = ["near (<5m)", "mid (5–15m)", "far (15–40m)", "very far (>40m)"]
N_DISTANCE_BINS = len(DISTANCE_LABELS)


def bucket_distances(distances: np.ndarray) -> np.ndarray:
    """Convert continuous distances (m) to discrete bin indices."""
    return np.digitize(distances, DISTANCE_BINS[1:-1])   # returns 0-indexed bin


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> dict:
    """
    Load a probe dataset from an HDF5 file.

    Returns
    -------
    dict with keys:
        'latents'      : {layer_name: np.ndarray shape (N, D)}
        'gt_distance'  : np.ndarray shape (N,)
        'gt_bins'      : np.ndarray shape (N,)   # bucketed version
        'observations' : np.ndarray or None
    """
    path = Path(path)
    assert path.exists(), f"Dataset not found: {path}"

    data = {"latents": {}, "observations": None}

    with h5py.File(path, "r") as f:
        data["gt_distance"] = f["gt_distance"][:].astype(np.float32)
        data["gt_bins"]     = bucket_distances(data["gt_distance"])

        for layer_name in f["latents"].keys():
            data["latents"][layer_name] = f["latents"][layer_name][:].astype(np.float32)

        if "observations" in f:
            data["observations"] = f["observations"][:]

    print(f"Loaded {len(data['gt_distance'])} samples, "
          f"{len(data['latents'])} layers from {path.name}")
    _print_layer_summary(data["latents"], data["gt_distance"])
    return data


def save_dataset(path: str | Path, latents: dict, gt_distance: np.ndarray,
                 observations: np.ndarray | None = None):
    """Save a probe dataset to HDF5. Called by the data collection script."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("gt_distance", data=gt_distance.astype(np.float32))
        grp = f.create_group("latents")
        for name, arr in latents.items():
            grp.create_dataset(name, data=arr.astype(np.float32), compression="gzip")
        if observations is not None:
            f.create_dataset("observations", data=observations, compression="gzip")

    print(f"Saved dataset → {path}  ({gt_distance.shape[0]} samples)")


# ── Train / val / test split ─────────────────────────────────────────────────

def split_indices(n: int, train: float = 0.7, val: float = 0.15, seed: int = 42):
    """Reproducible stratification-free split."""
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(n)
    t_end = int(n * train)
    v_end = int(n * (train + val))
    return idx[:t_end], idx[t_end:v_end], idx[v_end:]


# ── Mock data generator (for development / unit tests) ──────────────────────

def make_mock_dataset(
    n_samples: int = 3000,
    layer_dims: dict | None = None,
    distance_range: tuple = (1.0, 60.0),
    seed: int = 0,
    save_path: str | Path | None = None,
) -> dict:
    """
    Generate synthetic data that mimics a TransFuser-like architecture.

    The mock is designed so that:
      - Early layers encode distance POORLY  (low R²)
      - Middle layers encode distance WELL   (high R²)
      - Late layers encode distance ROUGHLY  (medium R²)

    This lets you verify the pipeline produces the expected inverted-U curve
    before plugging in real CARLA data.
    """
    if layer_dims is None:
        # Rough TransFuser layer structure:
        # ResNet encoder blocks → transformer blocks → task head
        layer_dims = {
            "encoder.block0":   256,   # early CNN
            "encoder.block1":   512,
            "encoder.block2":   1024,
            "encoder.block3":   2048,  # deepest CNN features
            "transformer.blk0": 512,   # early transformer
            "transformer.blk1": 512,
            "transformer.blk2": 512,
            "transformer.blk3": 512,   # richest representation
            "transformer.blk4": 512,
            "head.fc0":         256,   # task-specific head
            "head.fc1":         128,
        }

    rng = np.random.default_rng(seed)
    gt_distance = rng.uniform(*distance_range, size=n_samples).astype(np.float32)

    # Signal-to-noise ratios per layer (higher = better encoding of distance)
    snr_profile = np.array([0.05, 0.15, 0.40, 0.60,   # encoder
                             0.70, 0.85, 0.95, 1.00,   # transformer (peak)
                             0.80, 0.60, 0.45])         # head

    latents = {}
    for i, (name, dim) in enumerate(layer_dims.items()):
        snr    = snr_profile[i]
        proj   = rng.standard_normal((1, dim)).astype(np.float32)
        signal = (gt_distance[:, None] / distance_range[1]) @ proj  # (N, dim)
        noise  = rng.standard_normal((n_samples, dim)).astype(np.float32)
        latents[name] = snr * signal + (1 - snr) * noise * 0.3

    dataset = {
        "latents":      latents,
        "gt_distance":  gt_distance,
        "gt_bins":      bucket_distances(gt_distance),
        "observations": None,
    }

    if save_path is not None:
        save_dataset(save_path, latents, gt_distance)

    print(f"[Mock] Generated {n_samples} samples across {len(layer_dims)} layers")
    _print_layer_summary(latents, gt_distance)
    return dataset


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_layer_summary(latents: dict, gt_distance: np.ndarray):
    print(f"\n  {'Layer':<25} {'Dim':>6}    Distance range: "
          f"[{gt_distance.min():.1f}, {gt_distance.max():.1f}] m")
    for name, arr in latents.items():
        print(f"  {name:<25} {arr.shape[1]:>6}")
    print()
