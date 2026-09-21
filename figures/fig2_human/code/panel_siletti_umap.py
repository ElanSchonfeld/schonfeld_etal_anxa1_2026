#!/usr/bin/env python3
"""Figure 2 (supp) - Siletti human DA neurons: HMoE transfer vs ground truth (2x2 UMAP)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import FAMILY_COLORS, SUBTYPE_COLORS, infer_family   # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
GREY = "#cfcfcf"


def clean(ax):
    ax.set_box_aspect(0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def roman_label(ax, txt):
    ax.text(-0.04, 1.06, txt, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def scatter_cat(ax, xy, order, colors):
    ax.scatter(xy[order, 0], xy[order, 1], c=colors, s=22, alpha=0.85,
               rasterized=True, edgecolors="none")


def cat_legend(ax, labels, color_map, fs=7.5, ncol=2):
    cnt = Counter(labels)
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[k], markersize=6,
                label=f"{k.split(':')[-1]} ({cnt[k]:,})")
         for k in color_map if cnt.get(k, 0) > 0]
    ax.legend(handles=h, fontsize=fs, ncol=ncol, loc="upper right",
              handletextpad=0.1, columnspacing=0.5, frameon=False)


def blend_expression(s_v, c_v, g_v):
    s_v, c_v, g_v = (np.asarray(x, dtype=float) for x in (s_v, c_v, g_v))

    def norm95(v):
        pos = v > 0
        m = np.percentile(v[pos], 95) if pos.any() else 1.0
        return np.clip(v / max(m, 1e-9), 0, 1)

    s, c, g = norm95(s_v), norm95(c_v), norm95(g_v)
    sox6 = np.array(mcolors.to_rgb(FAMILY_COLORS["Sox6"]))
    calb1 = np.array(mcolors.to_rgb(FAMILY_COLORS["Calb1"]))
    gad2 = np.array(mcolors.to_rgb(FAMILY_COLORS["Gad2"]))
    grey = np.array([0.87, 0.87, 0.87])

    out = np.tile(grey, (len(s), 1))
    idx = np.where((s_v > 0) | (c_v > 0) | (g_v > 0))[0]
    tot = np.where((s + c + g)[idx] > 0, (s + c + g)[idx], 1.0)
    hue = (s[idx] / tot)[:, None] * sox6 + (c[idx] / tot)[:, None] * calb1 + (g[idx] / tot)[:, None] * gad2
    intensity = np.maximum.reduce([s[idx], c[idx], g[idx]])[:, None]
    out[idx] = grey * (1 - intensity) + hue * intensity
    return np.clip(out, 0, 1)


def main():
    df = pd.read_parquet(FROZEN / "siletti_percell.parquet")
    xy = df[["UMAP1", "UMAP2"]].to_numpy()
    n = len(df)
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 11.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03, wspace=0.08, hspace=0.14)

    ax = axes[0, 0]
    fam = df["hmoe_family"].to_numpy()
    o = rng.permutation(n)
    scatter_cat(ax, xy, o, [FAMILY_COLORS.get(fam[i], GREY) for i in o])
    fam_map = {f: FAMILY_COLORS[f] for f in ["Sox6", "Calb1", "Gad2"]}
    cat_legend(ax, fam, fam_map, fs=9, ncol=1)
    ax.set_title("HMoE predicted family", fontweight="bold", fontsize=13)
    roman_label(ax, "i.")

    ax = axes[0, 1]
    sub = df["hmoe_subtype"].to_numpy()
    o = rng.permutation(n)
    scatter_cat(ax, xy, o, [SUBTYPE_COLORS.get(sub[i], GREY) for i in o])
    sub_present = sorted({s for s in sub}, key=lambda s: (infer_family(s), s))
    sub_map = {s: SUBTYPE_COLORS.get(s, GREY) for s in sub_present}
    cat_legend(ax, sub, sub_map, fs=7, ncol=2)
    ax.set_title("HMoE predicted subtype", fontweight="bold", fontsize=13)
    roman_label(ax, "ii.")

    ax = axes[1, 0]
    sil = df["siletti_subtype"].astype(str).to_numpy()
    sil_col = df["siletti_color"].astype(str).to_numpy()
    o = rng.permutation(n)
    ax.scatter(xy[o, 0], xy[o, 1], c=sil_col[o], s=22, alpha=0.85, rasterized=True, edgecolors="none")
    lab_col = {lab: sil_col[sil == lab][0]
               for lab in sorted(set(sil), key=lambda s: (s == "Novel_DA", s))}
    cat_legend(ax, sil, lab_col, fs=7, ncol=2)
    ax.set_title("Siletti subcluster (ground truth)", fontweight="bold", fontsize=13)
    roman_label(ax, "iii.")

    ax = axes[1, 1]
    colors = blend_expression(df["expr_SOX6"].to_numpy(),
                              df["expr_CALB1"].to_numpy(),
                              df["expr_GAD2"].to_numpy())
    o = np.argsort(colors.max(1))
    ax.scatter(xy[o, 0], xy[o, 1], c=colors[o], s=22, rasterized=True, edgecolors="none")
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[f], markersize=8, label=g)
         for f, g in [("Sox6", "SOX6"), ("Calb1", "CALB1"), ("Gad2", "GAD2")]]
    ax.legend(handles=h, fontsize=9, loc="upper right", frameon=False)
    ax.set_title("SOX6 / CALB1 / GAD2 expression", fontweight="bold", fontsize=13)
    roman_label(ax, "iv.")

    for row in axes:
        for a in row:
            clean(a)

    fig.suptitle(f"Siletti human DA neurons – HMoE transfer vs ground truth (n={n:,})",
                 fontweight="bold", fontsize=14.5, y=0.975)
    figio.save_panel(fig, OUT / "siletti_umap")
    print(f"  -> siletti_umap  ({n:,} cells)")


if __name__ == "__main__":
    main()
