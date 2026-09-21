#!/usr/bin/env python3
"""Figure 2 - cross-species DA family classification benchmark (Accuracy)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                    # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"

N_TOTAL, N_POS = 22048, 15147
MAJORITY_ACC = N_POS / N_TOTAL
REFERENCE = "HMoE"
EFFECT_THRESHOLDS = (0.05, 0.10, 0.20)
XMIN = 0.30

CATEGORY_COLORS = {
    "HMoE (Ours)": "#E8873A", "Classical ML": "#5B7FA5", "KNN Transfer": "#7EAE8D",
    "Bioinformatics methods": "#C86A6A", "Foundation Model": "#9B8BB4",
}
STAR_KEY = ("vs HMoE (non-overlapping 95% CI = significant):   "
            "*** Δ≥0.20    ** Δ≥0.10    * significant (smaller Δ)    ns = CIs overlap")


def wilson_ci(p, n, z=1.96):
    """Wilson 95% CI for a binomial proportion (accuracy over n cells)."""
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def effect_size_stars(delta, significant):
    """Ns if CIs overlap; else *, **, *** by |delta|."""
    if not significant:
        return "ns"
    d = abs(delta)
    if d >= EFFECT_THRESHOLDS[2]:
        return "***"
    if d >= EFFECT_THRESHOLDS[1]:
        return "**"
    return "*"


def accuracy_stars(df):
    """Effect-size stars vs HMoE, gated by non-overlapping accuracy 95% CIs."""
    ref_val = float(df.loc[df["Method"] == REFERENCE, "Accuracy"].iloc[0])
    ref_lo, _ = wilson_ci(ref_val, N_TOTAL)
    stars = {}
    for _, row in df.iterrows():
        if row["Method"] == REFERENCE:
            stars[row["Method"]] = "ref"
            continue
        _, hi = wilson_ci(float(row["Accuracy"]), N_TOTAL)
        stars[row["Method"]] = effect_size_stars(ref_val - float(row["Accuracy"]), hi < ref_lo)
    return stars


def star_style(star):
    if star in ("***", "**", "*"):
        return star, "black", "normal", "bold", 8
    if star in ("ref", "ns"):
        return star, "#999", "italic", "normal", 7
    return "", "#ccc", "normal", "normal", 7


def main():
    df = pd.read_csv(FROZEN / "benchmark_fair_accuracy.csv")
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    methods = df["Method"].tolist()
    colors = [CATEGORY_COLORS[c] for c in df["category"]]
    vals = df["Accuracy"].values
    y = np.arange(len(methods))
    lo = np.array([wilson_ci(v, N_TOTAL)[0] for v in vals])
    hi = np.array([wilson_ci(v, N_TOTAL)[1] for v in vals])

    fig, ax = plt.subplots(figsize=(10, 8.6))
    fig.patch.set_facecolor("white")
    ax.barh(y, vals - XMIN, left=XMIN, color=colors, edgecolor="none", height=0.68, zorder=2)
    ax.errorbar(vals, y, xerr=np.array([vals - lo, hi - vals]), fmt="none",
                ecolor="#333", elinewidth=0.5, capsize=0, zorder=3)
    for i, v in enumerate(vals):
        ax.text(max(v, hi[i]) + 0.004, i, f"{v:.3f}", va="center", ha="left",
                fontsize=7.5, color="#333")

    ref_val = float(df.loc[df["Method"] == REFERENCE, "Accuracy"].iloc[0])
    ax.axvline(x=ref_val, color="#E8873A", linestyle="--", alpha=0.45, linewidth=0.8, zorder=1)
    ax.axvline(x=MAJORITY_ACC, color="#c8c8c8", linestyle=":", linewidth=0.6, zorder=0)

    stars = accuracy_stars(df)
    ax.text(0.95, -0.85, "vs HMoE", transform=ax.get_yaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color="#666")
    for i, m in enumerate(methods):
        txt, color, style, weight, sz = star_style(stars.get(m, ""))
        if txt:
            ax.text(0.95, y[i], txt, ha="center", va="center", transform=ax.get_yaxis_transform(),
                    fontsize=sz - 1, fontweight=weight, fontstyle=style, color=color)

    prev = None
    for i, c in enumerate(df["category"]):
        if prev and c != prev:
            ax.axhline(y=i - 0.5, color="#ddd", linestyle="-", linewidth=0.5)
        prev = c

    ax.set_yticks(y); ax.set_yticklabels([])
    for i, m in enumerate(methods):
        ax.text(-0.01, y[i] - 0.16, m, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9,
                fontweight="bold" if m == REFERENCE else "normal", color="#222")
        ax.text(-0.01, y[i] + 0.26, df["protocol"].iloc[i], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.5, color="#a8a8a8", style="italic")
    ax.invert_yaxis()
    ax.set_xlim(XMIN, 1.0)
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.tick_params(axis="x", labelsize=9); ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(MAJORITY_ACC, len(methods) - 0.35, "majority class", ha="center", va="bottom",
            fontsize=6.5, color="#9a9a9a")

    cats = [c for c in CATEGORY_COLORS if c in set(df["category"])]
    row1, row2 = cats[:3], cats[3:]
    leg1 = fig.legend(handles=[Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in row1],
                      loc="lower center", ncol=len(row1), fontsize=9.5, frameon=False,
                      bbox_to_anchor=(0.5, 0.072), handlelength=1.2,
                      handletextpad=0.5, columnspacing=1.8)
    fig.add_artist(leg1)
    if row2:
        fig.legend(handles=[Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in row2],
                   loc="lower center", ncol=len(row2), fontsize=9.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.030), handlelength=1.2,
                   handletextpad=0.5, columnspacing=1.8)

    plt.tight_layout(rect=[0, 0.115, 1, 0.99])
    fig.text(0.5, 0.005, STAR_KEY, ha="center", va="bottom", fontsize=7, color="#999")

    figio.save_panel(fig, OUT / "benchmark_roc_auc")


if __name__ == "__main__":
    main()
