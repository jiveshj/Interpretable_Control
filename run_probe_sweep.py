"""
run_probe_sweep.py
------------------
Main entry point for the linear probe sweep.

Usage:

  # Run on mock data (no CARLA needed):
  python run_probe_sweep.py --mock

  # Run on real data:
  python run_probe_sweep.py --data path/to/latents.h5

  # Run on real data + save figures:
  python run_probe_sweep.py --data path/to/latents.h5 --save-figures

Output:
  - Printed summary table (R² / MAE / gap per layer)
  - PNG figures in ./figures/       (if --save-figures)
  - results/probe_results.npz
  - results/probe_results.json
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import make_mock_dataset, load_dataset
from probe_trainer import LinearProbeTrainer
from visualize import save_all_figures


def parse_args():
    p = argparse.ArgumentParser(description="Run linear probe sweep over policy latents")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true",
                       help="Generate mock data and run (no CARLA needed)")
    group.add_argument("--data", type=str,
                       help="Path to HDF5 file with saved latents")

    p.add_argument("--n-mock-samples", type=int,   default=3000,
                   help="Number of mock samples to generate (default: 3000)")
    p.add_argument("--save-figures",   action="store_true",
                   help="Save figures to ./figures/")
    p.add_argument("--output-dir",     type=str,   default="results",
                   help="Directory to save numeric results (default: results/)")
    p.add_argument("--train-frac",     type=float, default=0.70)
    p.add_argument("--val-frac",       type=float, default=0.15)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no-log",         action="store_true",
                   help="Skip log-distance probe")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load or generate data ─────────────────────────────────────────────
    if args.mock:
        print("=" * 60)
        print("  Running on MOCK data (TransFuser-like architecture)")
        print("=" * 60)
        dataset = make_mock_dataset(n_samples=args.n_mock_samples)
    else:
        print("=" * 60)
        print(f"  Loading dataset from {args.data}")
        print("=" * 60)
        dataset = load_dataset(args.data)

    latents     = dataset["latents"]
    gt_distance = dataset["gt_distance"]
    gt_bins     = dataset["gt_bins"]

    # ── 2. Run probe sweep ───────────────────────────────────────────────────
    print("\n── Running probe sweep ──\n")
    trainer = LinearProbeTrainer(
        train_frac       = args.train_frac,
        val_frac         = args.val_frac,
        use_log_distance = not args.no_log,
        seed             = args.seed,
    )
    results = trainer.run(latents, gt_distance, gt_bins, verbose=True)

    # ── 3. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(results.summary_table())

    # ── 4. Interpret the key hypothesis ──────────────────────────────────────
    best_layer_idx  = int(np.argmax(results.r2_exact))
    best_layer_name = results.layer_names[best_layer_idx]
    best_r2_exact   = results.r2_exact[best_layer_idx]
    best_r2_bins    = results.r2_bins[best_layer_idx]
    mean_gap        = float(np.mean(results.coarse_vs_exact_gap))

    print("\n── Hypothesis Test ──")
    print(f"  Best layer for exact distance:  {best_layer_name}  (R²={best_r2_exact:.3f})")
    print(f"  Coarse R² at same layer:        {best_r2_bins:.3f}")
    print(f"  Mean gap (coarse - exact):      {mean_gap:+.3f}")

    if mean_gap > 0.05:
        print("\n  SUPPORTS hypothesis: coarse distance is easier to recover")
        print("     than exact metric distance across most layers.")
    elif mean_gap > -0.05:
        print("\n  INCONCLUSIVE: no strong difference between exact and coarse.")
    else:
        print("\n  CONTRADICTS hypothesis: exact distance is encoded as well as coarse.")

    # ── 5. Save numeric results ──────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "probe_results.npz",
        layer_names = np.array(results.layer_names),
        r2_exact    = results.r2_exact,
        r2_log      = results.r2_log,
        r2_bins     = results.r2_bins,
        acc_bins    = results.acc_bins,
        gap         = results.coarse_vs_exact_gap,
        mae         = np.array([r.mae_exact for r in results.layer_results]),
    )

    summary = {
        "n_samples":     int(len(gt_distance)),
        "n_layers":      len(results.layer_names),
        "best_layer":    best_layer_name,
        "best_r2_exact": round(float(best_r2_exact), 4),
        "best_r2_bins":  round(float(best_r2_bins),  4),
        "mean_gap":      round(float(mean_gap), 4),
        "layers": [
            {
                "name":     r.layer_name,
                "dim":      r.layer_dim,
                "r2_exact": round(r.r2_exact,  4),
                "r2_log":   round(r.r2_log,    4),
                "r2_bins":  round(r.r2_bins,   4),
                "acc_bins": round(r.acc_bins,  4),
                "gap":      round(r.coarse_vs_exact_gap, 4),
                "mae":      round(r.mae_exact, 3),
            }
            for r in results.layer_results
        ],
    }
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Numeric results saved → {out_dir}/")

    # ── 6. Figures ───────────────────────────────────────────────────────────
    if args.save_figures:
        save_all_figures(results, gt_distance, gt_bins, output_dir="figures")
    else:
        print("\n  (Pass --save-figures to generate PNG plots)")


if __name__ == "__main__":
    main()
