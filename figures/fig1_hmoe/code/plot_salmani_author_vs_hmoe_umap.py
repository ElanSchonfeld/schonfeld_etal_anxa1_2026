#!/usr/bin/env python3
"""Shared loader, palettes and legend helpers for the Figure 1 Salmani panels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
OUTDIR = HERE.parent / "output" / "panels"
FONTS = HERE.parents[2] / "common" / "_fonts"
PRED = FROZEN / "YaghmaeianSalmani_percell_gad2allowed.parquet"
AUTH = FROZEN / "salmani_author_annotations.parquet"

for f in ["Helvetica-Regular.ttf", "Helvetica-Bold.ttf"]:
    if (FONTS / f).exists():
        fm.fontManager.addfont(str(FONTS / f))
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
    "mathtext.bf": "Helvetica:bold", "mathtext.it": "Helvetica:italic",
    "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

SUBTYPE_COLORS = {
    "Calb1:Ccdc192": "#1f77b4", "Calb1:Chrm2": "#aec7e8", "Calb1:Gipr": "#ff7f0e",
    "Calb1:Kctd8": "#ffbb78", "Calb1:Lpar1": "#2ca02c", "Calb1:Pde11a": "#d62728",
    "Calb1:Ptprt": "#ff9896", "Calb1:Stac": "#9467bd", "Calb1:Sulf1": "#8c564b",
    "Calb1:Sox6": "#c5b0d5", "Gad2:Ebf2": "#c49c94", "Gad2:Egfr": "#e377c2",
    "Sox6:Arhgap28": "#f7b6d2", "Sox6:Kcnmb2": "#c7c7c7", "Sox6:March3": "#bcbd22",
    "Sox6:Tafa1": "#dbdb8d", "Sox6:Tmem132d": "#17becf", "Sox6:Vcan": "#9edae5"}

TERR_MDA = {"Sox6": "#1f77b4", "Otx2": "#ff7f0e", "Pcsk6": "#2ca02c", "Pdia5": "#d62728",
            "Fbn2": "#9467bd", "Gad2": "#8c564b", "ML_clusters": "#e377c2", "Ebf1": "#17becf"}
GREY = "#dcdcdc"
NODA = "#e9e9e9"

ANXA = ["Sox6:Tafa1", "Sox6:Vcan"]
FOCUS = {"anxa": "#08306b", "sox6": "#4292c6", "calb1": "#DE8F05", "gad2": "#2ca02c"}

ANXA1_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "anxa1", ["#ededed", "#fdae6b", "#e6550d", "#a63603", "#67000d"])

PT = 5.0
ABG = 0.55

FLOATER_KNN = 30
FLOATER_MAX_NONMDA = 0.5


def shades(base, n):
    """N hues from a darker to a lighter variant of `base` (dark -> light)."""
    rgb = np.array(mcolors.to_rgb(base))
    if n == 1:
        return [mcolors.to_hex(rgb)]
    out = []
    for t in np.linspace(-0.34, 0.5, n):
        c = rgb * (1 + t) if t < 0 else rgb + (1 - rgb) * t
        out.append(mcolors.to_hex(np.clip(c, 0, 1)))
    return out


def nh_parent(nh):
    if nh.startswith("nonmDA") or nh == "unassigned":
        return None
    if nh == "ML_clusters":
        return "ML_clusters"
    return nh.split("_NH")[0]


def build_neigh_colors(neighs):
    """Colour each mDA neighbourhood by its parent territory's hue, shaded by NH index."""
    cmap, ordered = {}, []
    for terr, base in TERR_MDA.items():
        nhs = sorted([nh for nh in set(neighs) if nh_parent(nh) == terr])
        for c, nh in zip(shades(base, len(nhs)), nhs):
            cmap[nh] = c
        ordered += nhs
    return cmap, ordered


def norm01(v, lo_pct=0, hi_pct=99):
    """Percentile-clipped min-max to [0,1] for display; returns (scaled, hi_value)."""
    hi = np.percentile(v, hi_pct)
    lo = np.percentile(v, lo_pct)
    if hi <= lo:
        hi = v.max() if v.max() > lo else lo + 1.0
    return np.clip((v - lo) / (hi - lo), 0, 1), hi


def load():
    """Authors' DA atlas only: the celltype==midbrain_Dopaminergic cells (the 8 mDA territories), left-joined to our HMoE predictions by GEO barcode."""
    auth = pd.read_parquet(AUTH)
    is_mda = (auth["celltype"] == "midbrain_Dopaminergic").to_numpy()
    X = auth[["umap_x", "umap_y"]].to_numpy()
    _, idx = NearestNeighbors(n_neighbors=FLOATER_KNN + 1).fit(X).kneighbors(X)
    frac_nonmda = 1.0 - is_mda[idx[:, 1:]].mean(1)
    keep = is_mda & (frac_nonmda <= FLOATER_MAX_NONMDA)
    n_float = int((is_mda & ~keep).sum())
    print(f"authors' DA cells {int(is_mda.sum()):,}  |  dropped {n_float} floating "
          f"(>{int(FLOATER_MAX_NONMDA*100)}% non-mDA neighbours)  |  kept {int(keep.sum()):,}")
    auth = auth[keep].reset_index(drop=True)
    pred = pd.read_parquet(PRED)[["barcode", "pred_subtype", "pred_family"]]
    pred["pred_family"] = pred["pred_subtype"].str.split(":").str[0]
    return auth.merge(pred, on="barcode", how="left")


def centroid_labels(ax, x, y, labels, keep):
    """Place cluster names at per-label medians (only for labels in `keep`), white-haloed."""
    for lab in keep:
        m = labels == lab
        if m.sum() < 30:
            continue
        cx, cy = np.median(x[m]), np.median(y[m])
        ax.text(cx, cy, lab, fontsize=9, fontweight="bold", ha="center", va="center",
                color="#111", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def legend_axis(ax, handles, labels, ncol, title=None, fs=9, title_fs=10.5):
    ax.axis("off")
    leg = ax.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
                    fontsize=fs, handletextpad=0.4, columnspacing=1.0, labelspacing=0.32,
                    borderpad=0.1, bbox_to_anchor=(0.5, 1.02))
    if title:
        leg.set_title(title, prop={"size": title_fs, "weight": "bold"})
    return leg


def dot(color):
    return Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                  markerfacecolor=color, markeredgecolor="none")
