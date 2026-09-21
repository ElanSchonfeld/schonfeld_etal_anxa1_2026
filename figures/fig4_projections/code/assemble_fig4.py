#!/usr/bin/env python3
"""Figure 4 - code-assembled preview montages."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                            # noqa: E402

PANELS = HERE.parent / "output" / "panels"
OUT = HERE.parent / "output"


def load(stem, max_w=1600):
    im = Image.open(PANELS / f"{stem}.png").convert("RGB")
    if im.size[0] > max_w:
        im = im.resize((max_w, int(max_w * im.size[1] / im.size[0])))
    return np.asarray(im)


def show(ax, stem, caption):
    ax.imshow(load(stem)); ax.set_title(caption, fontsize=9); ax.axis("off")


def main():
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.12, wspace=0.06,
                  height_ratios=[1.0, 1.15])
    show(fig.add_subplot(gs[0, 0]), "panelB_mouse_projection", "B  Mouse projection (heatmap)")
    show(fig.add_subplot(gs[0, 1]), "mouse_enrichment_series", "Mouse enrichment series (MERFISH)")
    show(fig.add_subplot(gs[0, 2]), "panelA_human_projection", "A  Human projection (heatmap)")
    show(fig.add_subplot(gs[1, 0]), "mouse_tracing_validation", "Mouse tracing validation")
    show(fig.add_subplot(gs[1, 1:]), "extended_arm2_horizontal",
         "Human extended atlas - MERSCOPE + Kraft (arm2)")
    figio.save_panel(fig, OUT / "fig4_main", pad_inches=0.1)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.16, wspace=0.06, height_ratios=[1.0, 1.05, 0.72])
    show(fig.add_subplot(gs[0, 0]), "panelD_salmani_projection", "Mouse Salmani (heatmap)")
    show(fig.add_subplot(gs[0, 1]), "panelF_kraft_projection", "Human Kraft (heatmap)")
    show(fig.add_subplot(gs[0, 2]), "panelG_macaque_projection", "Macaque rhesus refit (heatmap)")
    show(fig.add_subplot(gs[1, :]), "extended_arm1_horizontal",
         "Human extended atlas - MERSCOPE only (arm1)")
    show(fig.add_subplot(gs[2, :]), "lr_drivers",
         "LR-pair drivers (composite: decomposition x ligand x receptor expression), 5 datasets")
    figio.save_panel(fig, OUT / "fig4_supp", pad_inches=0.1)


if __name__ == "__main__":
    main()
