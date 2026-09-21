#!/usr/bin/env python3
"""Figure 2 (supp) - Kamath scANVI label transfer (family + subtype), 1x2."""
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
from kamath_common import FAMILY_COLORS, SUBTYPE_COLORS, infer_family   # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
ANX = ["Sox6:Tafa1", "Sox6:Vcan"]


def fam_order(famvec, rng):
    s_i = np.where(famvec == "Sox6")[0]
    c_i = np.where(famvec == "Calb1")[0]
    o_i = np.where(~np.isin(famvec, ["Sox6", "Calb1"]))[0]
    return np.concatenate([rng.permutation(o_i), rng.permutation(s_i), rng.permutation(c_i)])


def sub_order(subvec, rng):
    """Non-Anxa1 first (background), Anxa1 subtypes last (foreground)."""
    anx_i = np.where(np.isin(subvec, ANX))[0]
    oth_i = np.where(~np.isin(subvec, ANX))[0]
    return np.concatenate([rng.permutation(oth_i), rng.permutation(anx_i)])


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


def sub_legend(ax, subvec):
    cnt = Counter(subvec)
    subs = sorted(set(subvec), key=lambda s: (infer_family(s), s))
    h = []
    for s in subs:
        nn = cnt.get(s, 0)
        if nn == 0:
            continue
        h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=SUBTYPE_COLORS.get(s, "#bbb"),
                        markersize=6, label=f"{s.split(':')[-1]} ({nn:,})"))
    ax.legend(handles=h, fontsize=6.5, ncol=2, loc="upper right",
              handletextpad=0.1, columnspacing=0.5, frameon=False)


def tafa1_note(ax, subvec):
    n = int((subvec == "Sox6:Tafa1").sum())
    ax.text(0.02, 0.02, f"Sox6:Tafa1 (Anxa1) = {n:,} cells", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#08306b", va="bottom", ha="left")


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    df = df[~df["gad2_excluded"].values.astype(bool)]
    xy = df[["umap_x", "umap_y"]].to_numpy()
    obs_names = df["obs_name"].to_numpy()
    n = len(df)

    sv = pd.read_csv(FROZEN / "scanvi_subtype_predictions.csv").set_index("obs_name")["scanvi_pred_subtype"]
    scanvi_sub = sv.reindex(obs_names).astype(str).to_numpy()
    scanvi_fam = np.array([s.split(":", 1)[0] for s in scanvi_sub])

    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.1))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.03, wspace=0.08)

    ax = axes[0]
    o = fam_order(scanvi_fam, rng)
    scatter_cat(ax, xy, o, [FAMILY_COLORS.get(scanvi_fam[i], "#bbb") for i in o])
    fam_legend(ax, scanvi_fam)
    ax.set_title("scANVI predicted family", fontweight="bold", fontsize=13)
    roman_label(ax, "i.")

    ax = axes[1]
    o = sub_order(scanvi_sub, rng)
    scatter_cat(ax, xy, o, [SUBTYPE_COLORS.get(scanvi_sub[i], "#bbb") for i in o])
    sub_legend(ax, scanvi_sub)
    tafa1_note(ax, scanvi_sub)
    ax.set_title("scANVI predicted subtype", fontweight="bold", fontsize=13)
    roman_label(ax, "ii.")

    for a in axes:
        clean(a)

    fig.suptitle(f"Kamath DA neurons: scANVI label transfer (all DA cells, n={n:,})",
                 fontweight="bold", fontsize=14.5, y=0.97)
    figio.save_panel(fig, OUT / "kamath_umap_scanvi_vs_hmoe")
    print(f"  -> kamath_umap_scanvi_vs_hmoe  ({n:,} cells)")


if __name__ == "__main__":
    main()
