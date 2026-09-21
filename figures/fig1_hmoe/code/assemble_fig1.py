#!/usr/bin/env python3
"""Figure 1 - code-assembled preview montages."""
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                   # noqa: E402

PANELS = HERE.parent / "output" / "panels"
OUT = HERE.parent / "output"

MAIN = [
    ("HMoE_cell_count_colored_gates", "HMoE architecture (cell count)"),
    ("persubtype_metric_bars", "Per-subtype 5-fold CV performance"),
    ("salmani_panel_A_territory", "Salmani territory (ground truth)"),
    ("salmani_panel_B_neighborhood", "Salmani neighborhood (ground truth)"),
    ("salmani_panel_C_hmoe_subtype", "HMoE predicted subtype"),
    ("salmani_panel_D_vulnerability_focus", "Vulnerability focus"),
    ("salmani_panel_F_anxa1_expr", "Anxa1 expression"),
]
SUPP = [
    ("salmani_panel_G_territory_family_confusion", "Territory x HMoE family concordance"),
    ("salmani_panel_H_neighborhood_subtype_sankey", "Neighborhood -> HMoE subtype"),
]


def show(ax, stem, caption):
    ax.imshow(mpimg.imread(PANELS / f"{stem}.png"))
    ax.set_title(caption, fontsize=9)
    ax.axis("off")


def main():
    fig = plt.figure(figsize=(15, 13))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[2.4, 1.4, 2.2], hspace=0.12, wspace=0.04)
    show(fig.add_subplot(gs[0, :]), *MAIN[0])
    show(fig.add_subplot(gs[1, :]), *MAIN[1])
    for c, (stem, cap) in enumerate(MAIN[2:]):
        show(fig.add_subplot(gs[2, c]), stem, cap)
    figio.save_panel(fig, OUT / "fig1_main", pad_inches=0.1)

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6], wspace=0.05)
    for c, (stem, cap) in enumerate(SUPP):
        show(fig.add_subplot(gs[0, c]), stem, cap)
    figio.save_panel(fig, OUT / "fig1_supp", pad_inches=0.1)


if __name__ == "__main__":
    main()
