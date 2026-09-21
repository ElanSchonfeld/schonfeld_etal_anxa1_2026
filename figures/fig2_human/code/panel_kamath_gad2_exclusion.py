#!/usr/bin/env python3
"""Figure 2 supplementary panel: UMAP and QC metrics of Gad2-routed Kamath nuclei."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common")); import figio  # noqa: E402
OUT = HERE.parent / "output" / "panels"; FROZEN = HERE.parent / "frozen"

DA_MARKERS = ["TH", "SLC6A3", "SLC18A2", "DDC", "NR4A2"]
NONDA_MARKERS = ["GAD1", "GAD2", "SLC32A1", "AQP4"]
DA_PROGRAM = ["SLC6A3", "SLC18A2", "DDC", "NR4A2"]
KEPT_C, EXCL_C = "#3B7AB3", "#D55E00"
FS_STAR, FS_FOLD, FS_PANEL, FS_GRP = 13, 8.5, 15, 9


def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def _one_violin(ax, pos, v, gad):
    for grp, off, col in [(~gad, -0.19, KEPT_C), (gad, 0.19, EXCL_C)]:
        vp = ax.violinplot([v[grp]], positions=[pos + off], widths=0.34, showextrema=False, showmedians=True)
        for b in vp["bodies"]: b.set_facecolor(col); b.set_alpha(0.85); b.set_edgecolor("none")
        vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(1.3)


def _annot(ax, pos, head, p, fold=None, transform=None):
    kw = {"transform": transform} if transform is not None else {}
    ax.text(pos, head, stars(p), ha="center", va="bottom", fontsize=FS_STAR,
            color="#222" if p < 0.05 else "#888", fontweight="bold", **kw)
    if fold is not None:
        ax.text(pos, head, fold, ha="center", va="top", fontsize=FS_FOLD, color="#555", **kw)


def _panel_letter(ax, letter, x=-0.15):
    ax.text(x, 1.05, letter, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top", ha="left")


def main():
    df = pd.read_parquet(FROZEN / "kamath_gad2_percell.parquet")
    st = json.load(open(FROZEN / "kamath_gad2_stats.json"))
    gad = df["gad2_excluded"].values.astype(bool)
    xy = df[["umap_x", "umap_y"]].values
    n_tot = len(df); rng = np.random.RandomState(42)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6))
    fig.subplots_adjust(left=0.075, right=0.94, top=0.93, bottom=0.11, wspace=0.28, hspace=0.30)

    ax = axes[0, 0]; o = rng.permutation(n_tot)
    ax.scatter(xy[o, 0], xy[o, 1], c=np.where(gad[o], EXCL_C, "#d6d6d6"), s=6, alpha=0.7, rasterized=True, edgecolors="none")
    ax.scatter(xy[gad, 0], xy[gad, 1], c=EXCL_C, s=7, alpha=0.9, rasterized=True, edgecolors="none")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(0.85)
    for s in ax.spines.values(): s.set_visible(False)
    _panel_letter(ax, "a", x=-0.08)

    ax = axes[0, 1]; axR = ax.twinx()
    umis, genes, mt, ribo = df["umis"].values, df["genes"].values, df["mt"].values, df["ribo"].values
    lu, lg = np.log10(np.maximum(umis, 1)), np.log10(np.maximum(genes, 1))
    _one_violin(ax, 0, lu, gad); _one_violin(ax, 1, lg, gad)
    _one_violin(axR, 2, mt, gad); _one_violin(axR, 3, ribo, gad)
    lmax = max(lu.max(), lg.max()); lo_L = min(lu.min(), lg.min()) - (lmax) * 0.02
    ax.set_ylim(lo_L, lmax + (lmax - lo_L) * 0.34); axR.set_ylim(0, max(mt.max(), ribo.max()) * 1.34)
    SF, tr = 0.90, ax.get_xaxis_transform()
    for pos, key in [(0, "umis"), (1, "genes"), (2, "mt"), (3, "ribo")]:
        _annot(ax, pos, SF, st["qc"][key]["p"], st["qc"][key]["fold"], tr)
    ax.set_xlim(-0.6, 3.6); ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["UMIs / cell", "genes / cell", "mito %", "ribosomal %"], rotation=20, ha="right")
    ax.set_ylabel(r"log$_{10}$ count"); axR.set_ylabel("% of counts"); ax.axvline(1.5, color="#bbb", lw=1)
    ax.text(0.5, 1.0, "depth / complexity", transform=tr, ha="center", va="bottom", fontsize=FS_GRP, color="#666")
    ax.text(2.5, 1.0, "degradation", transform=tr, ha="center", va="bottom", fontsize=FS_GRP, color="#666")
    ax.spines[["top"]].set_visible(False); axR.spines[["top"]].set_visible(False); _panel_letter(ax, "b")

    ax = axes[1, 0]; markers = DA_MARKERS + NONDA_MARKERS
    ev = [df[g].values for g in markers]; nda = len(DA_MARKERS)
    lo = min(np.percentile(v, 0.5) for v in ev); hi = max(np.percentile(v, 99.5) for v in ev); off = (hi - lo) * 0.05
    da_y = max(ev[i].max() for i in range(nda)) + off
    nd_y = max(ev[i].max() for i in range(nda, len(markers))) + off
    for pos, (g, v) in enumerate(zip(markers, ev)):
        _one_violin(ax, pos, v, gad)
        _annot(ax, pos, da_y if pos < nda else nd_y, st["markers"][g])
    ax.set_xticks(range(len(markers))); ax.set_xticklabels(markers, fontstyle="italic", rotation=30, ha="right")
    ax.set_ylabel("expression (CP10k, log1p)"); ax.set_ylim(lo - (hi - lo) * 0.05, da_y + (hi - lo) * 0.12)
    ax.set_xlim(-0.6, len(markers) - 0.4); ax.axvline(nda - 0.5, color="#bbb", lw=1)
    ax.text((nda - 1) / 2, 1.0, "dopaminergic", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS_GRP, color="#666")
    ax.text(nda + (len(NONDA_MARKERS) - 1) / 2, 1.0, "non-DA / glial", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS_GRP, color="#666")
    ax.spines[["top", "right"]].set_visible(False); _panel_letter(ax, "c")

    ax = axes[1, 1]; slope = st["slopegraph"]
    for s in slope:
        ax.plot([0, 1], [s["kept"], s["excl"]], color="#c8c8c8", lw=1.1, zorder=1)
        ax.scatter([0], [s["kept"]], color=KEPT_C, s=34, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter([1], [s["excl"]], color=EXCL_C, s=34, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="#ddd", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1]); ax.set_xticklabels(["DA\n(kept)", "Gad2-routed\n(excluded)"])
    ax.set_ylabel("DA-program score\n(per-donor mean, z)")
    yhi = max(s["kept"] for s in slope)
    ax.text(0.5, yhi + 0.15, f"{stars(st['da_score_p'])}   n = {st['nd']} donors\npaired Wilcoxon",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")
    ax.set_ylim(None, yhi + 0.9); ax.spines[["top", "right"]].set_visible(False); _panel_letter(ax, "d")

    fig.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor=KEPT_C, markersize=11, label=f"DA, kept (n = {st['n_kept']:,})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=EXCL_C, markersize=11,
               label=f"Gad2-routed, excluded (n = {st['n_excl']:,}, {st['pct_excl']}%)")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.028))
    fig.text(0.5, 0.008, "donor-level pseudobulk: per-donor means, paired Wilcoxon "
             f"(n = {st['nd']} donors) · *p<0.05 **p<0.01 ***p<0.001", ha="center", va="bottom", fontsize=8, color="#777")
    figio.save_panel(fig, OUT / "kamath_gad2_exclusion")


if __name__ == "__main__":
    main()
