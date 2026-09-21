#!/usr/bin/env python3
"""Figure 2 - Kamath ground-truth cell type -> HMoE predicted subtype Sankey."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import (                     # noqa: E402
    SUBTYPE_COLORS, KAMATH_CT_COLORS, KAMATH_SOX6_TYPES, KAMATH_CALB1_TYPES,
    infer_family, spell_celltype,
)

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
FAM_ORDER = ["Sox6", "Calb1", "Gad2"]
CT_ORDER = KAMATH_SOX6_TYPES + KAMATH_CALB1_TYPES
SUB_ORDER = sorted(SUBTYPE_COLORS, key=lambda s: (FAM_ORDER.index(infer_family(s)), s))


def _ribbon(ax, x0, x1, yl0, yl1, yr0, yr1, color, alpha):
    xc = (x0 + x1) / 2
    verts = [(x0, yl0), (xc, yl0), (xc, yr0), (x1, yr0),
             (x1, yr1), (xc, yr1), (xc, yl1), (x0, yl1), (x0, yl0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def sankey(d, min_cells=10):
    ct = (pd.crosstab(d["Cell_Type"], d["pred_subtype"])
          .reindex(index=CT_ORDER, columns=SUB_ORDER).fillna(0).astype(int))
    left = [n for n in ct.index if ct.loc[n].sum() > 0]
    right = [r for r in ct.columns if ct[r].sum() > 0]
    ct = ct.loc[left, right]
    M = ct.to_numpy().astype(float)
    lt, rt, N = M.sum(1), M.sum(0), M.sum()

    GAPL, GAPR = 0.006, 0.006
    uL, uR = 1 - GAPL * (len(left) - 1), 1 - GAPR * (len(right) - 1)
    hL, hR = uL * lt / N, uR * rt / N
    topL = np.concatenate([[0], np.cumsum(hL + GAPL)[:-1]])
    topR = np.concatenate([[0], np.cumsum(hR + GAPR)[:-1]])
    cenL, cenR = topL + hL / 2, topR + hR / 2

    offL, offR = topL.copy(), topR.copy()
    segL, segR = {}, {}
    for i in range(len(left)):
        for j in sorted(range(len(right)), key=lambda j: cenR[j]):
            if M[i, j] < min_cells:
                continue
            h = uL * M[i, j] / N
            segL[(i, j)] = (offL[i], offL[i] + h); offL[i] += h
    for j in range(len(right)):
        for i in sorted(range(len(left)), key=lambda i: cenL[i]):
            if M[i, j] < min_cells:
                continue
            h = uR * M[i, j] / N
            segR[(i, j)] = (offR[j], offR[j] + h); offR[j] += h

    fig, ax = plt.subplots(figsize=(11.0, 12.2))
    barw, xL, xR = 0.020, 0.315, 0.685
    kept = 0
    for (i, j), (l0, l1) in segL.items():
        r0, r1 = segR[(i, j)]
        frac = M[i, j] / lt[i]
        alpha = float(np.clip(0.10 + 0.72 * frac, 0.10, 0.80))
        _ribbon(ax, xL, xR, 1 - l1, 1 - l0, 1 - r1, 1 - r0,
                KAMATH_CT_COLORS.get(left[i], "#999"), alpha)
        kept += M[i, j]

    for i in range(len(left)):
        y = 1 - (topL[i] + hL[i])
        ax.add_patch(plt.Rectangle((xL - barw, y), barw, hL[i],
                                   fc=KAMATH_CT_COLORS.get(left[i], "#999"), ec="white", lw=0.5, zorder=3))
        ax.text(xL - barw - 0.010, y + hL[i] / 2, f"{spell_celltype(left[i])}  ({int(lt[i]):,})",
                ha="right", va="center", fontsize=10.5, color="#111", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
    for j in range(len(right)):
        y = 1 - (topR[j] + hR[j])
        ax.add_patch(plt.Rectangle((xR, y), barw, hR[j],
                                   fc=SUBTYPE_COLORS.get(right[j], "#999"), ec="white", lw=0.5, zorder=3))
        ax.text(xR + barw + 0.010, y + hR[j] / 2, f"{right[j]}  ({int(rt[j]):,})",
                ha="left", va="center", fontsize=10.5, color="#111", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    ax.text(xL - barw / 2, 1.028, "Kamath cell type (ground truth)", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(xR + barw / 2, 1.028, "HMoE predicted subtype", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.015, 1.05); ax.axis("off")

    figio.save_panel(fig, OUT / "kamath_groundtruth_subtype_sankey", pad_inches=0.05)
    print(f"  -> kamath_groundtruth_subtype_sankey  ({len(segL)} ribbons, "
          f"{100*kept/N:.1f}% of cells, N={int(N):,})")


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    da = ~df["gad2_excluded"].values.astype(bool)
    d = pd.DataFrame({"Cell_Type": df["Cell_Type"].astype(str).values[da],
                      "pred_subtype": df["pred_subtype"].astype(str).values[da]})
    print(f"DA cells: {len(d):,}")
    sankey(d)


if __name__ == "__main__":
    main()
