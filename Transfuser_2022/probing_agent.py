"""
probing_agent.py
----------------
Subclass of TransFuser's submission agent (HybridAgent) that:
  1. Registers forward hooks on the model's image/lidar/transformer layers
  2. Records ground-truth distance to the nearest vehicle in front per step
  3. Saves activations + distances to HDF5 periodically and at shutdown

This relies on TransFuser's existing driving stack — sensor setup,
preprocessing, route planning, control — to handle naturalistic driving.
We only ADD probing instrumentation on top.

Launch via the standard CARLA leaderboard evaluator:
    bash launch_probing.sh

────────────────────────────────────────────────────────────────────────────
THINGS TO VERIFY against your actual submission_agent.py:
  1. The class name imported below (HybridAgent). In some TransFuser branches
     it's `MultiTaskAgent` or `AutoPilot`. Confirm with:
         grep -n "class.*AutonomousAgent" submission_agent.py
  2. The attribute holding the model (assumed `self.net`). Confirm with:
         grep -n "self\.\(net\|model\)" submission_agent.py
  3. The inner backbone module (assumed `self.net._model` like LidarCenterNet).
     This is where we hook image_encoder / lidar_encoder / transformerN.
  4. The `get_entry_point()` convention (returns class name as string).
────────────────────────────────────────────────────────────────────────────
"""

import os
import math
import h5py
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

# ── Import TransFuser's submission agent ──────────────────────────────────────
# This pulls in their full driving stack (sensors, preprocessing, control)
import sys
TEAM_CODE_ROOT = os.environ.get(
    "TEAM_CODE_ROOT",
    "/ocean/projects/cis250201p/jjain2/transfuser/team_code_transfuser"
)
sys.path.insert(0, TEAM_CODE_ROOT)

from submission_agent import HybridAgent   # ← VERIFY this class name

# ── CARLA + leaderboard ───────────────────────────────────────────────────────
import carla
from leaderboard.utils.route_indexer import RouteIndexer  # noqa
# CarlaDataProvider is how we access world state from inside an agent
try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
except ImportError:
    # Fallback path depending on leaderboard version
    from scenario_runner.srunner.scenariomanager.carla_data_provider \
        import CarlaDataProvider


# ── Probing config ────────────────────────────────────────────────────────────
DISTANCE_BINS = [
    (3.0,  5.0),   # bin 0: close following
    (5.0,  10.0),  # bin 1: normal following
    (10.0, 20.0),  # bin 2: medium range
    (20.0, 40.0),  # bin 3: far
]
MAX_DISTANCE        = DISTANCE_BINS[-1][1]   # samples beyond this dropped
FORWARD_CONE_HALF_W = 5.0                    # m, lateral tolerance for "in front"

# Where to save activations. Configurable via env var.
OUTPUT_HDF5 = os.environ.get(
    "PROBING_OUTPUT",
    "/ocean/projects/cis250201p/jjain2/data/latents_leaderboard.h5",
)
# Save partial file every N collected samples (in case agent crashes)
SAVE_EVERY = 500


def get_entry_point():
    """Required by CARLA leaderboard — returns the agent class name."""
    return "ProbingAgent"


class ProbingAgent(HybridAgent):
    """
    Extends TransFuser's HybridAgent with forward-hook instrumentation.
    Inherits all sensor setup, preprocessing, model loading, route planning,
    and control logic from the parent.
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self, path_to_conf_file):
        """
        Called once at the start of each route. Parent does the heavy lifting
        (sensor spawn, model load); we add hook registration on top.
        """
        super().setup(path_to_conf_file)

        # State for probing
        self._activation_buffer = defaultdict(list)
        self._gt_distances      = []
        self._bin_labels        = []
        self._hook_handles      = []
        self._n_collected       = 0
        self._n_ticks           = 0   # total run_step calls
        self._n_skipped_no_npc  = 0
        self._n_skipped_too_far = 0

        self._register_hooks()
        print(f"[ProbingAgent] Hooks registered, output → {OUTPUT_HDF5}")

    def destroy(self):
        """Called when the route finishes or the agent is killed."""
        # Remove hooks and save BEFORE super().destroy(), which does
        # `del self.nets` — we don't access nets here but being explicit
        # about ordering avoids any future surprises.
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._save_hdf5(final=True)
        super().destroy()

    # ── Hook registration ────────────────────────────────────────────────────

    def _register_hooks(self):
        """
        Hook the same 12 layers as oneNPC_collect_data.py:
          image_encoder.features.s1..s4
          lidar_encoder._model.s1..s4
          transformer1..4
        """
        # self.nets is a list (ensemble support). We hook nets[0] — the first
        # model in the ensemble. The _model attribute is the inner backbone
        # (same as in oneNPC_collect_data.py).
        backbone = self.nets[0]._model

        def make_hook(name):
            def hook(module, input, output):
                out = output[0] if isinstance(output, (tuple, list)) else output
                # Reduce to (B, C) so the probe sees a flat feature vector
                if out.dim() == 4:        # (B, C, H, W)
                    out = out.mean(dim=[2, 3])
                elif out.dim() == 3:      # (B, seq, C)
                    out = out.mean(dim=1)
                self._activation_buffer[name].append(
                    out.detach().cpu().float().numpy())
            return hook

        for i, s in enumerate(["s1", "s2", "s3", "s4"]):
            try:
                h = getattr(backbone.image_encoder.features, s).register_forward_hook(
                    make_hook(f"img_enc.stage{i+1}"))
                self._hook_handles.append(h)
            except AttributeError:
                print(f"[ProbingAgent] WARN: img_enc.{s} not found")

        for i, s in enumerate(["s1", "s2", "s3", "s4"]):
            try:
                h = getattr(backbone.lidar_encoder._model, s).register_forward_hook(
                    make_hook(f"lid_enc.stage{i+1}"))
                self._hook_handles.append(h)
            except AttributeError:
                print(f"[ProbingAgent] WARN: lid_enc.{s} not found")

        for i in range(1, 5):
            try:
                h = getattr(backbone, f"transformer{i}").register_forward_hook(
                    make_hook(f"transformer{i}"))
                self._hook_handles.append(h)
            except AttributeError:
                print(f"[ProbingAgent] WARN: transformer{i} not found")

    # ── Per-step instrumentation ─────────────────────────────────────────────

    def run_step(self, input_data, timestamp):
        """
        Compute GT distance BEFORE the forward pass (so we have a label
        ready), then let the parent run the normal step (which triggers
        the hooks). After it returns, decide whether to keep this sample.
        """
        self._n_ticks += 1

        # Heartbeat every 100 ticks so we know the agent is alive
        if self._n_ticks % 100 == 0:
            print(f"[ProbingAgent] tick={self._n_ticks}  "
                  f"collected={self._n_collected}  "
                  f"skipped_no_npc={self._n_skipped_no_npc}  "
                  f"skipped_too_far={self._n_skipped_too_far}")

        # 1. Ground truth: nearest vehicle in the ego's forward cone
        gt_dist = self._nearest_vehicle_in_front()

        # Track total forward-pass count to align with hook buffer length
        n_before = len(self._activation_buffer.get("img_enc.stage1", []))

        # 2. Run TransFuser's normal step (hooks fire here)
        control = super().run_step(input_data, timestamp)

        n_after = len(self._activation_buffer.get("img_enc.stage1", []))

        # 3. If no forward pass happened this tick, nothing to record
        if n_after == n_before:
            return control

        # 4. Decide whether to keep the sample
        keep = True
        bin_idx = None
        if gt_dist is None:
            self._n_skipped_no_npc += 1
            keep = False
        elif gt_dist >= MAX_DISTANCE:
            self._n_skipped_too_far += 1
            keep = False
        else:
            bin_idx = self._bin_for_distance(gt_dist)
            if bin_idx is None:
                keep = False

        if not keep:
            # Discard the most recent hook entry across all layers so the
            # buffer stays aligned with kept samples.
            for name in self._activation_buffer:
                if self._activation_buffer[name]:
                    self._activation_buffer[name].pop()
            return control

        # 5. Record GT alongside the just-appended activations
        self._gt_distances.append(float(gt_dist))
        self._bin_labels.append(int(bin_idx))
        self._n_collected += 1

        # 6. Periodic save in case the agent crashes mid-route
        if self._n_collected > 0 and self._n_collected % SAVE_EVERY == 0:
            print(f"[ProbingAgent] n={self._n_collected}  "
                  f"skipped: no_npc={self._n_skipped_no_npc} "
                  f"too_far={self._n_skipped_too_far}")
            self._save_hdf5(final=False)

        return control

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _nearest_vehicle_in_front(self):
        """
        Return distance (m) to the nearest other vehicle whose centre lies
        within a forward cone of the ego (≤ FORWARD_CONE_HALF_W laterally,
        positive longitudinal component), or None if no such vehicle exists.
        """
        ego = CarlaDataProvider.get_hero_actor()
        if ego is None:
            return None

        ego_tf  = ego.get_transform()
        ego_loc = ego_tf.location
        fwd     = ego_tf.get_forward_vector()
        # Right-hand perpendicular for lateral component
        right_x, right_y = fwd.y, -fwd.x

        world = CarlaDataProvider.get_world()
        if world is None:
            return None

        min_dist = None
        for v in world.get_actors().filter("vehicle.*"):
            if v.id == ego.id:
                continue
            try:
                v_loc = v.get_location()
            except RuntimeError:
                continue  # actor died mid-query
            dx = v_loc.x - ego_loc.x
            dy = v_loc.y - ego_loc.y
            longi = dx * fwd.x + dy * fwd.y
            if longi <= 0:
                continue  # behind ego
            lat = abs(dx * right_x + dy * right_y)
            if lat > FORWARD_CONE_HALF_W:
                continue  # outside lateral cone
            d = math.sqrt(dx * dx + dy * dy)
            if min_dist is None or d < min_dist:
                min_dist = d
        return min_dist

    @staticmethod
    def _bin_for_distance(d):
        for i, (lo, hi) in enumerate(DISTANCE_BINS):
            if lo <= d < hi:
                return i
        return None

    # ── HDF5 saving ──────────────────────────────────────────────────────────

    def _save_hdf5(self, final=False):
        """
        Concatenate per-layer activation lists and write to HDF5.
        Overwrites previous file each save — partial files are
        intentionally replaceable so the final save is canonical.
        """
        n = len(self._gt_distances)
        if n == 0:
            return

        out_path = Path(OUTPUT_HDF5)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gt   = np.asarray(self._gt_distances, dtype=np.float32)
        bins = np.asarray(self._bin_labels,   dtype=np.int32)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("gt_distance", data=gt)
            f.create_dataset("bin_label",   data=bins)
            f.attrs["n_skipped_no_npc"]  = self._n_skipped_no_npc
            f.attrs["n_skipped_too_far"] = self._n_skipped_too_far
            f.attrs["final_save"]        = bool(final)

            grp = f.create_group("latents")
            for layer_name, acts in self._activation_buffer.items():
                arr = np.concatenate(acts, axis=0)
                if arr.shape[0] != n:
                    print(f"[ProbingAgent] WARN: {layer_name} has "
                          f"{arr.shape[0]} samples (expected {n}) — skipping")
                    continue
                grp.create_dataset(layer_name, data=arr, compression="gzip")

        tag = "FINAL" if final else "partial"
        print(f"[ProbingAgent] {tag} save: n={n} → {out_path}")
        if final:
            print("Bin distribution:")
            for i, (lo, hi) in enumerate(DISTANCE_BINS):
                c = int((bins == i).sum())
                pct = 100 * c / n
                print(f"  bin{i} [{lo:4.1f}–{hi:4.1f}m]: {c} ({pct:.1f}%)")