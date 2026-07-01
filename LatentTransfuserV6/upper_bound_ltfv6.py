#!/usr/bin/env python
"""
upper_bound_ltfv6.py

Upper and lower bound experiments to calibrate what probe R²/MAE means.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — Nonlinear upper bound (MLP vs Ridge per layer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains a 2-hidden-layer MLP on each layer's activations and compares
to the Ridge result.  Gap = MLP − Ridge answers:

  "Is there more distance information in the activations than a linear
   probe can see?"

  Gap ≈ 0   → distance is linearly encoded; linear probes are sufficient.
  Gap >> 0  → distance is nonlinearly encoded; the true encoding capacity
               is higher than linear R² suggests.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — Noise-calibration curve (synthetic features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Creates 1-D synthetic features:

    x_synth = y + σ · ε,   ε ~ N(0,1)

where y = log(distance) and σ is swept over a range.  A Ridge probe
trained on x_synth gets the same R² as a layer that encodes distance
with that noise level.

This maps every real R²(log) value back to an *effective SNR*:

    SNR = Var(y) / σ²

Interpretation guide printed at the end.  The lower bound (σ → ∞,
R² → 0) mirrors the shuffle baseline; the upper bound (σ → 0, R² → 1)
is a perfect oracle.

Usage:
    python upper_bound_ltfv6.py
    python upper_bound_ltfv6.py --skip_mlp          # calibration only (fast)
    python upper_bound_ltfv6.py --label dist_nearest_any
"""

import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_DEFAULT        = "/ocean/projects/cis250201p/jjain2/data/latents_ltfv6_navtest.h5"
REAL_JSON_DEFAULT = "/jet/home/jjain2/Interpretable_Control/probe_results_ltfv6.json"
OUT_DEFAULT       = "/jet/home/jjain2/Interpretable_Control/upper_bound_ltfv6.json"

HOOK_KEYS = [
    *[f"backbone_image_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_lidar_encoder_layer{i}" for i in [1, 2, 3, 4]],
    *[f"backbone_transformers_{i}" for i in [0, 1, 2, 3]],
    *[f"planning_decoder_transformer_decoder_layers_{i}" for i in range(6)],
]
ALPHAS   = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_SPLITS = 5


def ridge_pipe():
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def mlp_pipe():
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            n_iter_no_change=20,
        )),
    ])


def r2_cv(pipe, X, y, kf):
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ── Part 1: nonlinear upper bound ─────────────────────────────────────────────

def run_mlp_vs_ridge(h5, y_log, valid, real_results):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    print(f"\n{'LAYER':<58} {'Ridge R²':>12} {'MLP R²':>12} {'Gap':>8}  "
          f"{'real R² (json)':>16}")
    print("-" * 112)

    layer_results = {}
    for key in HOOK_KEYS:
        if key not in h5:
            print(f"  {key:<56}  MISSING")
            continue
        X = h5[key][:][valid].astype(np.float32)

        ridge_mean, ridge_std = r2_cv(ridge_pipe(), X, y_log, kf)
        mlp_mean,   mlp_std   = r2_cv(mlp_pipe(),   X, y_log, kf)
        gap = mlp_mean - ridge_mean

        real_r2 = (real_results.get("layers", {})
                               .get(key, {})
                               .get("r2_log", {})
                               .get("mean", float("nan")))

        layer_results[key] = {
            "ridge_r2_log":     ridge_mean, "ridge_r2_log_std": ridge_std,
            "mlp_r2_log":       mlp_mean,   "mlp_r2_log_std":   mlp_std,
            "gap":              gap,
            "real_r2_log_json": real_r2,
        }
        print(f"  {key:<56} {ridge_mean:>+6.3f}±{ridge_std:.3f}  "
              f"{mlp_mean:>+6.3f}±{mlp_std:.3f}  {gap:>+7.3f}  "
              f"{real_r2:>+14.3f}")
    return layer_results


# ── Part 2: noise calibration curve ──────────────────────────────────────────

def run_noise_calibration(y_log, n_levels, seed=42):
    """Sweep σ and record R² of Ridge on (y_log + σ·ε) → y_log."""
    rng   = np.random.RandomState(seed)
    kf    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y_var = float(np.var(y_log))
    y_std = float(np.std(y_log))

    # σ from 0.01·std (almost no noise) to 10·std (signal buried in noise)
    sigmas = np.logspace(np.log10(0.01 * y_std), np.log10(10.0 * y_std), n_levels)

    print(f"\n{'sigma':>10}  {'SNR':>8}  {'SNR(dB)':>9}  {'R²':>8}")
    print("-" * 42)

    calibration = []
    for sigma in sigmas:
        noise   = rng.randn(len(y_log)).astype(np.float32) * float(sigma)
        X_synth = (y_log + noise).reshape(-1, 1)
        r2_mean, r2_std = r2_cv(ridge_pipe(), X_synth, y_log, kf)
        snr    = y_var / (float(sigma) ** 2 + 1e-12)
        snr_db = 10.0 * np.log10(snr + 1e-12)
        calibration.append({
            "sigma":   float(sigma),
            "snr":     float(snr),
            "snr_db":  float(snr_db),
            "r2_mean": r2_mean,
            "r2_std":  r2_std,
        })
        print(f"  {sigma:>8.4f}  {snr:>8.2f}  {snr_db:>8.1f} dB  {r2_mean:>+7.3f}")

    return calibration


# ── Interpretation helper ─────────────────────────────────────────────────────

def interpret(real_results, calibration):
    """For each layer's real R²(log), find the closest matching SNR."""
    print("\n── Interpretation: real R²(log) → effective SNR ─────────────────")
    print(f"  {'LAYER':<58} {'R²(log)':>9}  {'eff. SNR':>10}  {'SNR(dB)':>9}")
    print("  " + "-" * 92)

    calib_r2  = np.array([c["r2_mean"] for c in calibration])
    calib_snr = np.array([c["snr"]     for c in calibration])
    calib_db  = np.array([c["snr_db"]  for c in calibration])

    for key, vals in real_results.get("layers", {}).items():
        r2 = vals.get("r2_log", {}).get("mean", float("nan"))
        if np.isnan(r2):
            continue
        # Nearest calibration point
        idx = int(np.argmin(np.abs(calib_r2 - r2)))
        snr = calib_snr[idx]
        db  = calib_db[idx]
        print(f"  {key:<58} {r2:>+8.3f}  {snr:>10.2f}  {db:>8.1f} dB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",        default=H5_DEFAULT)
    p.add_argument("--label",     default="dist_nearest_front",
                   choices=["dist_nearest_any", "dist_nearest_front"])
    p.add_argument("--real_json", default=REAL_JSON_DEFAULT,
                   help="probe_results JSON for cross-referencing real R² values")
    p.add_argument("--out",       default=OUT_DEFAULT)
    p.add_argument("--n_noise",   type=int, default=25,
                   help="Number of σ levels for calibration curve (default 25)")
    p.add_argument("--skip_mlp",  action="store_true",
                   help="Skip MLP experiment (fast: calibration only)")
    args = p.parse_args()

    with h5py.File(args.h5, "r") as h5:
        y_raw = h5[args.label][:]

    # Keep same filter as main probe sweep
    valid = np.isfinite(y_raw) & (y_raw >= 3.0)
    y     = y_raw[valid].astype(np.float32)
    y_log = np.log(y)
    print(f"Label   : {args.label}")
    print(f"N valid : {int(valid.sum())}  range [{y.min():.2f}, {y.max():.2f}]")

    with open(args.real_json) as f:
        real_results = json.load(f)

    # ── Part 2: noise calibration (fast, run first) ─────────────────────────
    print(f"\n[Part 2] Noise calibration curve  (n_levels={args.n_noise})")
    calib = run_noise_calibration(y_log, n_levels=args.n_noise)

    # ── Part 1: MLP vs Ridge ─────────────────────────────────────────────────
    layer_results = {}
    if not args.skip_mlp:
        print(f"\n[Part 1] Nonlinear (MLP) vs linear (Ridge) per layer")
        with h5py.File(args.h5, "r") as h5:
            layer_results = run_mlp_vs_ridge(h5, y_log, valid, real_results)

    # ── Interpretation table ─────────────────────────────────────────────────
    interpret(real_results, calib)

    # ── Guidance printout ────────────────────────────────────────────────────
    print("\n── How to read the noise calibration ───────────────────────────")
    print("  The calibration maps R²(log) to an 'effective SNR'.")
    print("  SNR = Var(log_dist) / σ_noise²  for a single noisy encoding.")
    print()
    print("  Rough thresholds:")
    print("    R² < 0.05  → essentially no distance signal (noise floor)")
    print("    R² ~ 0.5   → SNR ≈ 1 (signal power ≈ noise power)")
    print("    R² ~ 0.75  → SNR ≈ 3 (~5 dB; coarse but meaningful)")
    print("    R² ~ 0.90  → SNR ≈ 9 (~10 dB; fairly precise encoding)")
    print("    R² ~ 0.95  → SNR ≈ 19 (~13 dB; high-fidelity encoding)")
    print()
    print("  MLP gap interpretation:")
    print("    Gap ≈ 0   → distance is linearly decodable (linear probe sufficient)")
    print("    Gap >> 0  → there is additional nonlinearly-encoded distance info")

    # ── Save ────────────────────────────────────────────────────────────────
    out = {
        "label":               args.label,
        "n_samples":           int(valid.sum()),
        "noise_calibration":   calib,
        "layer_mlp_vs_ridge":  layer_results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
