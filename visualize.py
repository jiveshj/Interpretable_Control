"""
visualize.py
------------
Produces all key plots for the probe sweep results.

Main figures:
  1. R²-vs-depth          (the headline result)
  2. Coarse vs. exact     (tests the "rough encoding" hypothesis)
  3. MAE per layer
  4. Distance distribution (sanity check)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from probe_trainer import ProbeSweepResults
from data_utils import DISTANCE_LABELS


# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "exact": "#2196F3",   # blue
    "log":   "#03A9F4",   # light blue
    "bins":  "#FF5722",   # orange-red
    "acc":   "#9C27B0",   # purple
    "gap":   "#4CAF50",   # green
}


def plot_r2_vs_depth(
    results: ProbeSweepResults,
    title: str = "Linear Probe R² per Layer",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Headline figure: R² of linear probes across layers.

    Shows:
      - R²(exact distance)  — how well latents encode precise metric distance
      - R²(log distance)    — same with log transform
      - R²(coarse bins)     — how well latents encode rough near/mid/far

    The gap between orange and blue is the key finding.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    x     = np.arange(len(results.layer_names))
    names = results.layer_names

    ax.plot(x, results.r2_exact, "o-",  color=COLORS["exact"], lw=2,
            ms=7, label="R² (exact distance, m)")
    ax.plot(x, results.r2_log,   "s--", color=COLORS["log"],   lw=1.5,
            ms=6, label="R² (log distance)")
    ax.plot(x, results.r2_bins,  "^-",  color=COLORS["bins"],  lw=2,
            ms=7, label="R² (coarse bins: near/mid/far)")

    # Shade the gap between coarse and exact
    ax.fill_between(x, results.r2_exact, results.r2_bins,
                    where=(results.r2_bins > results.r2_exact),
                    alpha=0.12, color=COLORS["gap"],
                    label="Coarse > Exact gap")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("R² (linear probe, test set)", fontsize=11)
    ax.set_xlabel("Layer (shallow → deep)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.legend(loc="upper left", fontsize=9)

    # Annotate peak layer
    peak_idx = int(np.argmax(results.r2_exact))
    ax.annotate(
        f"Peak: {results.r2_exact[peak_idx]:.2f}\n({names[peak_idx]})",
        xy=(peak_idx, results.r2_exact[peak_idx]),
        xytext=(peak_idx + 0.5, results.r2_exact[peak_idx] + 0.08),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="gray",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


def plot_coarse_vs_exact(
    results: ProbeSweepResults,
    title: str = "Coarse vs. Exact Distance Encoding",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Side-by-side bar chart comparing R²(exact) vs R²(bins) per layer.

    Tall orange bar, short blue bar → model encodes ROUGH distance only.
    Equal bars → model encodes precise metric distance.
    This is the visual test of the core hypothesis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x     = np.arange(len(results.layer_names))
    width = 0.4

    # Left: absolute R² values side by side
    ax = axes[0]
    ax.bar(x - width/2, results.r2_exact, width, color=COLORS["exact"],
           label="R² exact (m)", alpha=0.85)
    ax.bar(x + width/2, results.r2_bins,  width, color=COLORS["bins"],
           label="R² coarse bins", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results.layer_names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Absolute R² comparison")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Right: gap = R²(bins) - R²(exact) → positive means "rough is easier"
    ax2 = axes[1]
    gap  = results.coarse_vs_exact_gap
    cols = [COLORS["gap"] if g > 0 else "#F44336" for g in gap]
    ax2.bar(x, gap, color=cols, alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results.layer_names, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("R²(coarse) − R²(exact)")
    ax2.set_title("Gap: positive = coarse encoding easier than exact")

    pos_patch = mpatches.Patch(color=COLORS["gap"], label="Rough > Exact (supports hypothesis)")
    neg_patch = mpatches.Patch(color="#F44336",      label="Exact ≥ Rough")
    ax2.legend(handles=[pos_patch, neg_patch], fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


def plot_mae_vs_depth(
    results: ProbeSweepResults,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """MAE (meters) per layer — complementary to R² for interpretability."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x   = np.arange(len(results.layer_names))
    mae = np.array([r.mae_exact for r in results.layer_results])

    ax.bar(x, mae, color=COLORS["exact"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(results.layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Absolute Error (meters)")
    ax.set_title("Distance Prediction MAE per Layer", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


def plot_distance_distribution(
    gt_distance: np.ndarray,
    gt_bins: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Sanity check: show the distribution of distances in the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(gt_distance, bins=50, color=COLORS["exact"], alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Ground-truth distance distribution")

    bin_counts = np.bincount(gt_bins, minlength=len(DISTANCE_LABELS))
    axes[1].bar(range(len(DISTANCE_LABELS)), bin_counts,
                color=[COLORS["exact"], COLORS["log"], COLORS["bins"], COLORS["acc"]],
                alpha=0.85)
    axes[1].set_xticks(range(len(DISTANCE_LABELS)))
    axes[1].set_xticklabels(DISTANCE_LABELS, rotation=20, ha="right")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distance bin distribution (near / mid / far)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


def save_all_figures(
    results: ProbeSweepResults,
    gt_distance: np.ndarray,
    gt_bins: np.ndarray,
    output_dir: str | Path = "figures",
):
    """Convenience: save all four figures to a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_r2_vs_depth(results,
                     save_path=output_dir / "r2_vs_depth.png", show=False)
    plot_coarse_vs_exact(results,
                         save_path=output_dir / "coarse_vs_exact.png", show=False)
    plot_mae_vs_depth(results,
                      save_path=output_dir / "mae_vs_depth.png", show=False)
    plot_distance_distribution(gt_distance, gt_bins,
                               save_path=output_dir / "distance_dist.png", show=False)
    print(f"\nAll figures saved to {output_dir}/")
