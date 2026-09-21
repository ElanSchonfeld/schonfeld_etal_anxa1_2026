#!/usr/bin/env python
"""Figure 4 supplementary panel: ligand-receptor driver ranking across the five datasets."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use("Agg")


def _register_helvetica() -> None:
    from fontTools.ttLib import TTCollection
    ttc = Path("/System/Library/Fonts/Helvetica.ttc")
    if not ttc.exists():
        return
    cache = Path.home() / ".cache" / "m2h_fonts"
    cache.mkdir(parents=True, exist_ok=True)
    want = {"Helvetica": "Helvetica.ttf", "Helvetica Bold": "Helvetica-Bold.ttf"}
    for face in TTCollection(str(ttc)).fonts:
        nm = face["name"].getDebugName(4) or face["name"].getDebugName(1)
        if nm in want:
            out = cache / want[nm]
            if not out.exists():
                face.save(str(out))
            fm.fontManager.addfont(str(out))


_register_helvetica()
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.linewidth"] = 0.7
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch, FancyArrowPatch              # noqa: E402
from matplotlib.gridspec import GridSpec                           # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen" / "lr_drivers"
OUT = HERE.parent / "output" / "panels"

GROUPS = ["Anxa1", "Sox6", "Calb1"]
GCOL = {"Anxa1": "#08306B", "Sox6": "#4292C6", "Calb1": "#DE8F05"}
TOP_N = 14
DATASETS = [("mouse_LRRK2", "LRRK2\n(mouse)",   "Mouse"),
            ("salmani",     "Salmani\n(mouse)", "Mouse"),
            ("macaque",     "HMBA\n(macaque)",  "Macaque"),
            ("human",       "Kamath\n(human)",  "Human"),
            ("kraft",       "Kraft\n(human)",   "Human")]


def _draw_labels(ax, sub) -> None:
    """LIGAND ->[drawn arrow] RECEPTOR laid out right-to-left, ending just left of the axis."""
    yc = np.arange(len(sub))
    trans = ax.get_yaxis_transform()
    r = ax.figure.canvas.get_renderer()
    axw = ax.get_window_extent(r).width
    x_right, gap, arrow_len = -0.02, 0.012, 0.05
    for y, L, R in zip(yc, sub.ligand, sub.receptor):
        tR = ax.text(x_right, y, str(R), ha="right", va="center", fontsize=7, transform=trans, clip_on=False)
        wR = tR.get_window_extent(r).width / axw
        x_arr_r = x_right - wR - gap
        x_arr_l = x_arr_r - arrow_len
        ax.add_patch(FancyArrowPatch((x_arr_l, y), (x_arr_r, y), transform=trans,
                     arrowstyle="-|>", mutation_scale=7, lw=1.0, color="#1a1a1a",
                     shrinkA=0, shrinkB=0, clip_on=False, zorder=6))
        ax.text(x_arr_l - gap, y, str(L), ha="right", va="center", fontsize=7, transform=trans, clip_on=False)


def draw_family(ax, df, title) -> None:
    """3 bars per pair (Anxa1/Sox6/Calb1); rank by max-over-family value."""
    df = df.copy()
    df["rank_val"] = df[GROUPS].max(1)
    sub = df.sort_values("rank_val", ascending=False).head(TOP_N).iloc[::-1].reset_index(drop=True)
    yc = np.arange(len(sub)); bw = 0.26
    for j, g in enumerate(GROUPS):
        ax.barh(yc + (1 - j) * bw, sub[g], height=bw, color=GCOL[g], zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_yticks(yc); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    _draw_labels(ax, sub)
    ax.set_ylim(-0.7, len(sub) - 0.3)
    ax.set_xlim(0, sub[GROUPS].to_numpy().max() * 1.08)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5, linespacing=0.95)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#999"); ax.spines["bottom"].set_color("#999")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)


def species_brackets(fig, axes) -> None:
    spans = {}
    for i, (_, _, sp) in enumerate(DATASETS):
        spans.setdefault(sp, []).append(i)
    for sp, cols in spans.items():
        x0 = axes[min(cols)].get_position().x0
        x1 = axes[max(cols)].get_position().x1
        y = 0.955
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color="#333", lw=1.1, transform=fig.transFigure, clip_on=False))
        fig.text((x0 + x1) / 2, y + 0.006, sp, ha="center", va="bottom", fontsize=11, fontweight="bold")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 8.6))
    gs = GridSpec(1, 5, wspace=0.92, left=0.075, right=0.99, top=0.86, bottom=0.10)
    axes = []
    for col, (key, title, species) in enumerate(DATASETS):
        df = pd.read_csv(FROZEN / f"{key}_magnitude.tsv", sep="\t")
        ax = fig.add_subplot(gs[0, col])
        draw_family(ax, df, title)
        ax.set_xlabel("Composite driver score  (recon × ligand × receptor)", fontsize=7.5)
        axes.append(ax)
    species_brackets(fig, axes)
    fig.suptitle("LR-pair drivers (composite: decomposition × ligand-expr × receptor-expr) "
                 "· 5 datasets, LIANA consensus", fontsize=13, fontweight="bold", y=0.995)
    leg = [Patch(fc=GCOL[g], label=lab) for g, lab in
           [("Anxa1", "Anxa1 (Sox6:Tafa1/Vcan)"), ("Sox6", "Sox6 family"), ("Calb1", "Calb1 family")]]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))
    stem = OUT / "lr_drivers"
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.ai", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png + .ai")


if __name__ == "__main__":
    main()
