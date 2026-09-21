#!/usr/bin/env python3
"""Concordance panels comparing Salmani author labels with HMoE predictions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                   # noqa: E402
import plot_salmani_author_vs_hmoe_umap as S

FAMILY_COLORS = {"Sox6": "#0173B2", "Calb1": "#DE8F05", "Gad2": "#2ca02c"}
OUTDIR = S.OUTDIR
MDA_TERR = list(S.TERR_MDA)
FAM_ORDER = ["Sox6", "Calb1", "Gad2"]
GLABEL = {"Sox6": r"$\mathbf{Sox6^{+}}$", "Calb1": r"$\mathbf{Calb1^{+}}$", "Gad2": r"$\mathbf{Gad2^{+}}$"}
SUB_ORDER = sum([[s for s in S.SUBTYPE_COLORS if s.startswith(f + ":")] for f in FAM_ORDER], [])


def load_da():
    m = S.load()
    d = m[(m["celltype"] == "midbrain_Dopaminergic") & m["pred_subtype"].notna()].copy()
    return d[d["territory"].isin(MDA_TERR)]


def panel_G(d):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
                         "mathtext.bf": "Helvetica:bold", "mathtext.it": "Helvetica:italic",
                         "mathtext.default": "bf", "pdf.fonttype": 42, "svg.fonttype": "none"})
    ct = pd.crosstab(d["territory"], d["pred_family"]).reindex(index=MDA_TERR, columns=FAM_ORDER).fillna(0).astype(int)
    cm = ct.to_numpy().astype(float)
    row = cm / cm.sum(1, keepdims=True) * 100
    support = cm.sum(1).astype(int)
    dom_frac = 100 * cm.max(1).sum() / cm.sum()
    nr = len(MDA_TERR)

    cmap = plt.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    GAP = 0.03
    fig, ax = plt.subplots(figsize=(4.9, 8.4))
    for i in range(nr):
        jmax = int(np.argmax(cm[i]))
        for j in range(3):
            val = row[i, j]
            ax.add_patch(Rectangle((j - 0.5 + GAP, i - 0.5 + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
                                   facecolor=cmap(norm(val)), edgecolor="none", zorder=1))
            dark = val > 50
            ax.text(j, i - 0.14, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=13.5,
                    fontweight="bold", color="white" if dark else "#111827", zorder=3)
            ax.text(j, i + 0.20, f"{val:.1f}%", ha="center", va="center", fontsize=9.5,
                    color="#eaf1fb" if dark else "#6b7280", zorder=3)
            if j == jmax:
                ax.add_patch(Rectangle((j - 0.5 + GAP, i - 0.5 + GAP), 1 - 2 * GAP, 1 - 2 * GAP,
                                       fill=False, edgecolor="#0f5c2e", lw=1.6, zorder=4))

    ax.set_xlim(-0.5, 2.5); ax.set_ylim(nr - 0.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks(range(3)); ax.set_yticks(range(nr))
    ax.set_xticklabels([GLABEL[f] for f in FAM_ORDER], fontsize=14)
    ax.set_yticklabels([f"{t}\nn = {s:,}" for t, s in zip(MDA_TERR, support)], fontsize=11.5,
                       va="center", linespacing=1.5)
    ax.set_xlabel("HMoE predicted family", fontsize=12.5, labelpad=10)
    ax.set_ylabel("Salmani territory (ground truth)", fontsize=12.5, labelpad=10)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for t, f in zip(ax.get_xticklabels(), FAM_ORDER):
        t.set_color(FAMILY_COLORS[f])
    for s in ax.spines.values():
        s.set_visible(False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.05, ticks=[0, 25, 50, 75, 100])
    cb.set_label("% of territory", fontsize=11)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#cccccc")

    fig.tight_layout(pad=0.4)
    stem = OUTDIR / "salmani_panel_G_territory_family_confusion"
    figio.save_panel(fig, stem)
    print(f"  {stem.name}: {dom_frac:.1f}% dominant-family")


def _ribbon(ax, x0, x1, yl0, yl1, yr0, yr1, color, alpha):
    """Filled cubic-Bezier band from left segment [yl0,yl1] at x0 to right segment [yr0,yr1] at x1."""
    xc = (x0 + x1) / 2
    verts = [(x0, yl0), (xc, yl0), (xc, yr0), (x1, yr0),
             (x1, yr1), (xc, yr1), (xc, yl1), (x0, yl1), (x0, yl0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def panel_H(d, min_cells=8):
    NH_COLORS, nh_all = S.build_neigh_colors(d["neighborhood"])
    ct = pd.crosstab(d["neighborhood"], d["pred_subtype"]).reindex(index=nh_all, columns=SUB_ORDER).fillna(0).astype(int)
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
        _ribbon(ax, xL, xR, 1 - l1, 1 - l0, 1 - r1, 1 - r0, NH_COLORS.get(left[i], "#999"), alpha)
        kept += M[i, j]

    for i in range(len(left)):
        y = 1 - (topL[i] + hL[i])
        ax.add_patch(plt.Rectangle((xL - barw, y), barw, hL[i], fc=NH_COLORS.get(left[i], "#999"),
                                   ec="white", lw=0.5, zorder=3))
        ax.text(xL - barw - 0.010, y + hL[i] / 2, f"{left[i]}  ({int(lt[i]):,})", ha="right",
                va="center", fontsize=10.5, color="#111", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
    for j in range(len(right)):
        y = 1 - (topR[j] + hR[j])
        ax.add_patch(plt.Rectangle((xR, y), barw, hR[j], fc=S.SUBTYPE_COLORS.get(right[j], "#999"),
                                   ec="white", lw=0.5, zorder=3))
        ax.text(xR + barw + 0.010, y + hR[j] / 2, f"{right[j]}  ({int(rt[j]):,})", ha="left",
                va="center", fontsize=10.5, color="#111", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    ax.text(xL - barw / 2, 1.028, "Salmani neighborhood", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(xR + barw / 2, 1.028, "HMoE subtype", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.015, 1.05); ax.axis("off")
    stem = OUTDIR / "salmani_panel_H_neighborhood_subtype_sankey"
    figio.save_panel(fig, stem, pad_inches=0.05)
    print(f"  {stem.name}: {len(segL)} ribbons, {100*kept/N:.1f}% of cells")


def main():
    d = load_da()
    print(f"cells: {len(d):,}")
    panel_G(d)
    panel_H(d)


if __name__ == "__main__":
    main()
