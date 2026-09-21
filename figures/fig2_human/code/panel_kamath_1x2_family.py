#!/usr/bin/env python3
"""Figure 2 - Kamath HMoE family concordance (two stacked UMAPs)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import FAMILY_COLORS, infer_family   # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"


def fam_order(famvec, rng):
    """Other first, Sox6 background, Calb1 foreground."""
    s_i = np.where(famvec == "Sox6")[0]
    c_i = np.where(famvec == "Calb1")[0]
    o_i = np.where(~np.isin(famvec, ["Sox6", "Calb1"]))[0]
    return np.concatenate([rng.permutation(o_i), rng.permutation(s_i), rng.permutation(c_i)])


def scatter_cat(ax, xy, idx_order, colors):
    ax.scatter(xy[idx_order, 0], xy[idx_order, 1], c=colors, s=8, alpha=0.7,
               rasterized=True, edgecolors="none")


def clean(ax):
    ax.set_box_aspect(0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def roman_label(ax, txt):
    ax.text(-0.04, 1.06, txt, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def fam_legend(ax, famvec):
    cnt = Counter(famvec)
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[f],
                markersize=9, label=f"{f} ({cnt.get(f, 0):,})") for f in ["Sox6", "Calb1"]]
    ax.legend(handles=h, fontsize=10, loc="upper right", frameon=False)


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    df = df[~df["gad2_excluded"].values.astype(bool)]
    xy = df[["umap_x", "umap_y"]].to_numpy()
    fam = df["pred_family"].to_numpy()
    gt_fam = np.array([infer_family(c) for c in df["Cell_Type"].to_numpy()])
    n = len(df)
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(2, 1, figsize=(5.7, 11.2))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.02, hspace=0.14)

    ax = axes[0]
    o = fam_order(fam, rng)
    scatter_cat(ax, xy, o, [FAMILY_COLORS.get(fam[i], "#bbb") for i in o])
    fam_legend(ax, fam)
    ax.set_title("HMoE predicted family", fontweight="bold", fontsize=13)
    roman_label(ax, "i.")

    ax = axes[1]
    o = fam_order(gt_fam, rng)
    scatter_cat(ax, xy, o, [FAMILY_COLORS.get(gt_fam[i], "#cfcfcf") for i in o])
    fam_legend(ax, gt_fam)
    ax.set_title("Kamath ground-truth family", fontweight="bold", fontsize=13)
    roman_label(ax, "ii.")

    for a in axes:
        clean(a)

    fig.suptitle(f"Kamath 2022 – HMoE family concordance\n(all DA cells, n={n:,})",
                 fontweight="bold", fontsize=11.5, y=0.985)
    figio.save_panel(fig, OUT / "kamath_1x2_family_all")
    print(f"  -> kamath_1x2_family_all  ({n:,} cells)")


if __name__ == "__main__":
    main()
