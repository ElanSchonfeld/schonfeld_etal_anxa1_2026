#!/usr/bin/env python3
"""Figure 2 - Kamath family confusion matrix (HMoE predicted vs ground truth)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import FAMILY_COLORS         # noqa: E402

mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
                     "mathtext.bf": "Helvetica:bold", "mathtext.it": "Helvetica:italic",
                     "mathtext.default": "bf"})

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
FAMS = ["Sox6", "Calb1"]
GLABEL = {"Sox6": r"$\mathbf{Sox6^{+}}$", "Calb1": r"$\mathbf{Calb1^{+}}$"}


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    df = df[~df["gad2_excluded"].values.astype(bool)]
    gt = np.array(["Sox6" if str(c).upper().startswith("SOX6")
                   else ("Calb1" if str(c).upper().startswith("CALB1") else "Other")
                   for c in df["Cell_Type"].to_numpy()])
    pf = df["pred_family"].astype(str).to_numpy()
    m = np.isin(gt, FAMS)
    gt, pf, n = gt[m], pf[m], int(m.sum())

    cm = np.array([[np.sum((gt == g) & (pf == p)) for p in FAMS] for g in FAMS], float)
    recall = cm / cm.sum(1, keepdims=True) * 100
    support = cm.sum(1).astype(int)
    acc = np.trace(cm) / cm.sum()

    cmap = plt.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    GAP = 0.03

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i in range(2):
        for j in range(2):
            val = recall[i, j]
            ax.add_patch(Rectangle((j - 0.5 + GAP, i - 0.5 + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
                                   facecolor=cmap(norm(val)), edgecolor="none", zorder=1))
            dark = val > 50
            ax.text(j, i - 0.15, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color="white" if dark else "#111827", zorder=3)
            ax.text(j, i + 0.19, f"{val:.1f}%", ha="center", va="center", fontsize=12.5,
                    color="#eaf1fb" if dark else "#6b7280", zorder=3)
            if i == j:
                ax.add_patch(Rectangle((j - 0.5 + GAP, i - 0.5 + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
                                       fill=False, edgecolor="#0f5c2e", lw=1.6, zorder=4))

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([GLABEL[f] for f in FAMS], fontsize=14)
    ax.set_yticklabels([f"{GLABEL[f]}\nn = {s:,}" for f, s in zip(FAMS, support)], fontsize=14,
                       va="center", linespacing=1.6)
    ax.set_xlabel("HMoE predicted family", fontsize=12.5, labelpad=10)
    ax.set_ylabel("Kamath ground-truth family", fontsize=12.5, labelpad=10)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for t, f in zip(ax.get_xticklabels(), FAMS):
        t.set_color(FAMILY_COLORS[f])
    for t, f in zip(ax.get_yticklabels(), FAMS):
        t.set_color(FAMILY_COLORS[f])
    for s in ax.spines.values():
        s.set_visible(False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.05, ticks=[0, 25, 50, 75, 100])
    cb.set_label("Recall (%)", fontsize=11)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#cccccc")

    ax.annotate(f"n = {n:,}   •   overall accuracy = {acc * 100:.1f}%",
                xy=(0.5, -0.06), xycoords="axes fraction", ha="center", va="top",
                fontsize=9.5, color="#374151")

    fig.tight_layout(pad=0.4)
    figio.save_panel(fig, OUT / "kamath_family_confusion")
    print(f"  -> kamath_family_confusion  (n={n:,}, acc={acc * 100:.1f}%)")


if __name__ == "__main__":
    main()
