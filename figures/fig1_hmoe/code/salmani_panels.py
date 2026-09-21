#!/usr/bin/env python3
"""Figure 1 - Salmani author-vs-HMoE UMAP panels (standalone, one file each)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                   # noqa: E402
import plot_salmani_author_vs_hmoe_umap as S

OUTDIR = S.OUTDIR


def scatter_clean(ax, x, y, colors, order, lims, bg_a, fg_a):
    """Parent draw_umap/draw_expr scatter WITHOUT the title/subtitle (clean panel)."""
    bg = ~order
    ax.scatter(x[bg], y[bg], s=S.PT, c=colors[bg], alpha=bg_a, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(x[order], y[order], s=S.PT, c=colors[order], alpha=fg_a, linewidths=0, rasterized=True, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(*lims); ax.set_ylim(*lims); ax.set_aspect("equal")
    x0, y0 = 0.02, 0.02
    ax.annotate("", xy=(x0 + 0.14, y0), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.annotate("", xy=(x0, y0 + 0.14), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.text(x0 + 0.075, y0 - 0.012, "UMAP1", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#333")
    ax.text(x0 - 0.012, y0 + 0.075, "UMAP2", transform=ax.transAxes, ha="right", va="center",
            fontsize=7.5, color="#333", rotation=90)


def callout(ax, target, boxfrac, label, fs=10):
    """Boxed label at axes-fraction `boxfrac` with a leader line to data-coord `target`."""
    ax.annotate(label, xy=target, xycoords="data", xytext=boxfrac, textcoords="axes fraction",
                fontsize=fs, fontweight="bold", ha="center", va="center", color="#111", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.3),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.4))


def leader(ax, target, boxfrac):
    """Extra leader line from an existing callout box to a second data-coord `target`."""
    ax.annotate("", xy=target, xycoords="data", xytext=boxfrac, textcoords="axes fraction",
                zorder=5, arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.4))


def panel(draw, legend):
    fig = plt.figure(figsize=(6.2, 7.3))
    gs = GridSpec(2, 1, height_ratios=[4.2, 1.0], hspace=0.03,
                  left=0.02, right=0.98, top=0.995, bottom=0.01)
    ax = fig.add_subplot(gs[0, 0]); leg = fig.add_subplot(gs[1, 0])
    draw(ax); legend(fig, leg)
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    m = S.load()
    x, y = m["umap_x"].to_numpy(), m["umap_y"].to_numpy()
    terr = m["territory"].to_numpy(); neigh = m["neighborhood"].to_numpy()
    sub = m["pred_subtype"].to_numpy(dtype=object)
    has_pred = m["pred_subtype"].notna().to_numpy()

    cA = np.array([S.TERR_MDA.get(t, S.GREY) for t in terr]); onA = np.isin(terr, list(S.TERR_MDA))
    NEIGH_COLORS, nh_ordered = S.build_neigh_colors(neigh)
    cB = np.array([NEIGH_COLORS.get(nn, S.GREY) for nn in neigh]); onB = np.isin(neigh, nh_ordered)
    subs = np.array([s if isinstance(s, str) else "" for s in sub])
    cC = np.array([S.SUBTYPE_COLORS.get(s, S.NODA) for s in subs]); onC = has_pred
    FOCUS_D = {**S.FOCUS, "stac": "#cb181d"}
    grp = np.where(np.isin(subs, S.ANXA), "anxa",
          np.where(subs == "Calb1:Stac", "stac",
          np.where(np.char.startswith(subs, "Sox6"), "sox6",
          np.where(np.char.startswith(subs, "Calb1"), "calb1",
          np.where(np.char.startswith(subs, "Gad2"), "gad2", "none")))))
    cD = np.array([FOCUS_D.get(g, S.NODA) for g in grp]); onD = np.isin(grp, ["anxa", "stac"])
    a01, hi_a = S.norm01(m["expr_Anxa1"].to_numpy())
    cF = S.ANXA1_CMAP(a01)[:, :3]; onF = a01 > 0.05
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max()); pad = 0.03 * (hi - lo)
    lims = (lo - pad, hi + pad)

    def med(mask):
        return (float(np.median(x[mask])), float(np.median(y[mask])))
    snc_hi = med(subs == "Sox6:Tafa1")
    snc_lo = med(subs == "Sox6:Vcan")
    vta = med(subs == "Calb1:Stac")
    SNC_BOX = (0.99, 0.60)

    def dA(ax):
        scatter_clean(ax, x, y, cA, onA, lims, S.ABG, 0.9)
        S.centroid_labels(ax, x, y, terr, list(S.TERR_MDA))
    def lAf(fig, leg):
        S.legend_axis(leg, [S.dot(S.TERR_MDA[t]) for t in S.TERR_MDA], list(S.TERR_MDA),
                      ncol=3, title="Territory (ground truth)")
    figio.save_panel(panel(dA, lAf), OUTDIR / "salmani_panel_A_territory")

    def dB(ax):
        scatter_clean(ax, x, y, cB, onB, lims, S.ABG, 0.9)
    def lBf(fig, leg):
        S.legend_axis(leg, [S.dot(NEIGH_COLORS[nn]) for nn in nh_ordered], nh_ordered,
                      ncol=3, title="Neighborhood (ground truth)", fs=7.6)
    figio.save_panel(panel(dB, lBf), OUTDIR / "salmani_panel_B_neighborhood")

    fam_order = ["Sox6", "Calb1", "Gad2"]
    ordered = sum([[s for s in S.SUBTYPE_COLORS if s.startswith(f + ":")] for f in fam_order], [])
    def dC(ax):
        scatter_clean(ax, x, y, cC, onC, lims, S.ABG, 0.9)
    def lCf(fig, leg):
        S.legend_axis(leg, [S.dot(S.SUBTYPE_COLORS[s]) for s in ordered], ordered,
                      ncol=3, title="HMoE subtype", fs=8.2)
    figio.save_panel(panel(dC, lCf), OUTDIR / "salmani_panel_C_hmoe_subtype")

    lblD = [r"$\mathbf{Sox6^{+}\!/\!Anxa1^{+}}$  (Tafa1, Vcan)", r"$\mathbf{Calb1^{Stac}}$  (Anxa1$^{+}$ VTA)",
            r"$\mathbf{Sox6^{+}}$  (non-Anxa1)", r"$\mathbf{Calb1^{+}}$", r"$\mathbf{Gad2^{+}}$"]
    def dD(ax):
        scatter_clean(ax, x, y, cD, onD, lims, S.ABG, 0.9)
        callout(ax, snc_hi, SNC_BOX, "Anxa1$^{+}$\nSNc")
        leader(ax, snc_lo, SNC_BOX)
        callout(ax, vta, (0.20, 0.94), "Anxa1$^{+}$ VTA")
    def lDf(fig, leg):
        S.legend_axis(leg, [S.dot(FOCUS_D[k]) for k in ["anxa", "stac", "sox6", "calb1", "gad2"]], lblD,
                      ncol=1, title="Predicted family / group", fs=11)
    figio.save_panel(panel(dD, lDf), OUTDIR / "salmani_panel_D_vulnerability_focus")

    def dF(ax):
        scatter_clean(ax, x, y, cF, onF, lims, 0.85, 0.95)
        callout(ax, snc_hi, SNC_BOX, "Anxa1$^{+}$\nSNc")
        leader(ax, snc_lo, SNC_BOX)
        callout(ax, vta, (0.20, 0.94), "Anxa1$^{+}$ VTA")
    def lFf(fig, leg):
        leg.axis("off")
        cax = leg.inset_axes([0.28, 0.42, 0.44, 0.22])
        sm = ScalarMappable(norm=mcolors.Normalize(0, hi_a), cmap=S.ANXA1_CMAP)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Anxa1  norm. log expr.", fontsize=9.5, fontweight="bold")
        cb.ax.tick_params(labelsize=8)
    figio.save_panel(panel(dF, lFf), OUTDIR / "salmani_panel_F_anxa1_expr")

    print(f"\n  SNc callouts -> {snc_hi} (Tafa1), {snc_lo} (Vcan)   VTA callout -> {vta}")
    print(f"wrote 5 panels -> {OUTDIR}")


if __name__ == "__main__":
    main()
