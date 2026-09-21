#!/usr/bin/env python3
"""Figure 2 - code-assembled preview montages."""
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                    # noqa: E402

PANELS = HERE.parent / "output" / "panels"
OUT = HERE.parent / "output"

MAIN = [
    ("kamath_1x2_family_all", "HMoE vs ground-truth family"),
    ("kamath_2x3_all", "Subtype identity + Anxa1 markers"),
    ("kamath_family_confusion", "Family confusion matrix"),
    ("benchmark_roc_auc", "Cross-species benchmark (accuracy)"),
    ("height_5", "Height=5 partition UMAP"),
    ("kamath_groundtruth_subtype_sankey", "Ground truth -> HMoE subtype"),
]
SUPP = [
    ("kamath_subtype_heatmap", "Within-species subtype marker heatmap"),
    ("kamath_umap_scanvi_vs_hmoe", "scANVI label transfer"),
    ("siletti_umap", "Siletti transfer vs ground truth"),
    ("kamath_hmoe_confidence", "HMoE per-cell confidence"),
]


def show(ax, stem, caption):
    ax.imshow(mpimg.imread(PANELS / f"{stem}.png"))
    ax.set_title(caption, fontsize=9)
    ax.axis("off")


def main():
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.10, wspace=0.05)
    for k, (stem, cap) in enumerate(MAIN):
        show(fig.add_subplot(gs[k // 3, k % 3]), stem, cap)
    figio.save_panel(fig, OUT / "fig2_main", pad_inches=0.1)

    fig = plt.figure(figsize=(18, 17))
    gs = GridSpec(3, 2, figure=fig, hspace=0.12, wspace=0.05, height_ratios=[1.0, 1.0, 1.05])
    for k, (stem, cap) in enumerate(SUPP):
        show(fig.add_subplot(gs[k // 2, k % 2]), stem, cap)
    show(fig.add_subplot(gs[2, :]), "kamath_gad2_exclusion",
         "Gad2-routed nuclei are low-quality / degraded (off-manifold, DA program collapsed) -> excluded")
    figio.save_panel(fig, OUT / "fig2_supp", pad_inches=0.1)


if __name__ == "__main__":
    main()
