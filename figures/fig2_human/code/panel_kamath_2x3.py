#!/usr/bin/env python3
"""Figure 2 - Kamath 2x3: subtype identity + Anxa1 marker feature plots (all DA cells)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

GENE_PANELS = [("TAFA1", 0, 1, "iii."), ("ALDH1A1", 1, 1, "iv."),
               ("PCDH17", 0, 2, "v."), ("TPBG", 1, 2, "vi.")]
PANEL_GENES = [g for g, *_ in GENE_PANELS]
NOTE = {"TAFA1": " (eponymous marker)", "ALDH1A1": "",
        "PCDH17": " (de-novo marker)", "TPBG": " (de-novo marker)"}
ALRA_CSVS = ["Kamath_alra_tafa1_markers.csv", "Kamath_alra_imputed.csv"]
EXPR_CMAP = plt.cm.coolwarm


def load_alra_multi(barcodes, genes):
    """ALRA-imputed values for `genes`, drawn from whichever ALRA csv carries each gene."""
    out = {}
    for csv in ALRA_CSVS:
        df = pd.read_csv(FROZEN / csv)
        lut = {col[:-5]: dict(zip(df["barcode"], df[col]))
               for col in df.columns if col.endswith("_alra")}
        for g in genes:
            if g in out or g not in lut:
                continue
            d = lut[g]
            out[g] = np.array([d.get(b, 0.0) for b in barcodes], dtype=np.float32)
    missing = [g for g in genes if g not in out]
    if missing:
        raise KeyError(f"no ALRA values for {missing}")
    return out


def clean(ax):
    ax.set_box_aspect(0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def roman_label(ax, txt):
    ax.text(-0.04, 1.06, txt, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def scatter_cat(ax, xy, idx_order, colors):
    ax.scatter(xy[idx_order, 0], xy[idx_order, 1], c=colors, s=8, alpha=0.7,
               rasterized=True, edgecolors="none")


def scatter_expr(ax, xy, vals):
    """All cells on coolwarm; p90 clip of positives, high plotted on top; inset colorbar."""
    pos = vals > 0
    vmax = max(np.percentile(vals[pos], 90), 1e-3) if pos.any() else 1.0
    o = np.argsort(vals)
    sm = ax.scatter(xy[o, 0], xy[o, 1], c=vals[o], cmap=EXPR_CMAP,
                    vmin=0, vmax=vmax, s=8, rasterized=True, edgecolors="none")
    cax = ax.inset_axes([0.66, 0.95, 0.31, 0.028])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax / 2, vmax])
    cb.set_ticklabels(["0", f"{vmax/2:.1f}", f"{vmax:.1f}"])
    cax.tick_params(labelsize=8, pad=1)
    cb.set_label("ALRA expr (log1p)", fontsize=8, labelpad=1)


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    df = df[~df["gad2_excluded"].values.astype(bool)]
    xy = df[["umap_x", "umap_y"]].to_numpy()
    sub = df["pred_subtype"].to_numpy()
    ct = df["Cell_Type"].to_numpy()
    alra = load_alra_multi(df["obs_name"].to_numpy(), PANEL_GENES)
    n = len(df)
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(2, 3, figsize=(16.6, 11.1))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, wspace=0.07, hspace=0.13)

    ax = axes[0, 0]
    o = rng.permutation(n)
    scatter_cat(ax, xy, o, [SUBTYPE_COLORS.get(sub[i], "#bbb") for i in o])
    subs = sorted({s for s in sub if "Gad2" not in s}, key=lambda s: (infer_family(s), s))
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SUBTYPE_COLORS.get(s, "#bbb"),
                markersize=6, label=s.split(":")[-1]) for s in subs if (sub == s).sum() > 0]
    ax.legend(handles=h, fontsize=7, ncol=3, loc="upper right",
              handletextpad=0.1, columnspacing=0.5, frameon=False)
    ax.set_title("HMoE predicted subtypes", fontweight="bold", fontsize=13)
    roman_label(ax, "i.")

    ax = axes[1, 0]
    o = rng.permutation(n)
    scatter_cat(ax, xy, o, [KAMATH_CT_COLORS.get(ct[i], "#cfcfcf") for i in o])
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=KAMATH_CT_COLORS.get(c, "#cfcfcf"),
                markersize=6, label=spell_celltype(c))
         for c in KAMATH_SOX6_TYPES + KAMATH_CALB1_TYPES if (ct == c).sum() > 0]
    ax.legend(handles=h, fontsize=7, ncol=2, loc="upper right",
              handletextpad=0.1, columnspacing=0.5, frameon=False)
    ax.set_title("Kamath cell types (ground truth)", fontweight="bold", fontsize=13)
    roman_label(ax, "ii.")

    used = {(0, 0), (1, 0)}
    for gene, r, c, lab in GENE_PANELS:
        ax = axes[r, c]; used.add((r, c))
        scatter_expr(ax, xy, alra[gene])
        ax.set_title(f"{gene}{NOTE.get(gene, ' (Anxa1 marker)')}", fontweight="bold", fontsize=13)
        roman_label(ax, lab)

    for r in range(2):
        for c in range(3):
            if (r, c) not in used:
                axes[r, c].axis("off")
    for row in axes:
        for a in row:
            if a.get_visible() and a.axison:
                clean(a)

    fig.suptitle(f"Kamath et al. 2022 – HMoE on human DA neurons (all DA cells, n={n:,})",
                 fontweight="bold", fontsize=15, y=0.985)
    figio.save_panel(fig, OUT / "kamath_2x3_all")
    print(f"  -> kamath_2x3_all  ({n:,} cells)")


if __name__ == "__main__":
    main()
