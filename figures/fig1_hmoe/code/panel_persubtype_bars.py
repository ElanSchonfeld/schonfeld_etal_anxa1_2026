#!/usr/bin/env python3
"""Figure 1 - per-subtype HMoE performance bars."""
import sys
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                   # noqa: E402
from palette import SUBTYPE_COLORS, FAM_LABEL_COLOR

FROZEN = HERE.parent / "frozen"
OUT = HERE.parent / "output" / "panels"

sns.set_theme(style="ticks", context="notebook")
mpl.rcParams.update({"hatch.linewidth": 0.5, "axes.edgecolor": "#333333"})

FAM_ORDER = ["Sox6", "Calb1", "Gad2"]
METRICS = [("recall_sensitivity", "Recall (sensitivity)", ""),
           ("roc_auc", "ROC-AUC", "////"),
           ("pr_auc", "PR-AUC", "....")]


def fam_of(s):
    return s.split(":")[0] if ":" in s else s


def main():
    cv = json.loads((FROZEN / "cv5_metrics_mouse.json").read_text())
    pc = cv["per_class_mean_std"]

    order, centers, fam_span = [], [], {}
    x = 0.0
    for fam in FAM_ORDER:
        fl = sorted([l for l in pc if fam_of(l) == fam],
                    key=lambda l: pc[l]["recall_sensitivity"]["mean"], reverse=True)
        start = x
        for l in fl:
            order.append(l); centers.append(x); x += 1.0
        fam_span[fam] = (start, x - 1.0)
    centers = np.array(centers)

    Y0 = 0.70
    bw, offs = 0.26, [-0.275, 0.0, 0.275]
    fig, ax = plt.subplots(figsize=(14.6, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.30, left=0.055, right=0.99)

    x_left, x_right = centers[0] - 0.5, centers[-1] + 0.5
    for i, fam in enumerate(FAM_ORDER):
        lo, hi = fam_span[fam]
        band_lo = x_left if i == 0 else lo - 0.5
        band_hi = x_right if i == len(FAM_ORDER) - 1 else hi + 0.5
        cc = np.array([int(FAM_LABEL_COLOR[fam][k:k + 2], 16) for k in (1, 3, 5)]) / 255
        ax.axvspan(band_lo, band_hi, color=tuple(cc + (1 - cc) * 0.91), zorder=0)
        ax.text((lo + hi) / 2, 1.072, fam, ha="center", va="center", fontsize=18,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.34", fc=FAM_LABEL_COLOR[fam], ec="none"))

    for (key, _, hatch), off in zip(METRICS, offs):
        for l, c in zip(order, centers):
            m = pc[l][key]["mean"]; s = pc[l][key]["std"]; col = SUBTYPE_COLORS.get(l, "#999")
            ax.bar(c + off, m - Y0, bottom=Y0, width=bw, color=col, hatch=hatch,
                   edgecolor="#3a3a3a", linewidth=0.55, yerr=s, zorder=3,
                   error_kw=dict(ecolor="#3a3a3a", elinewidth=0.8, capsize=1.6, capthick=0.8))

    ax.set_ylim(Y0, 1.11)
    ax.set_yticks(np.arange(Y0, 1.001, 0.05))
    ax.set_ylabel("Score  (5-fold CV mean ± std)", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.axhline(1.0, color="#CFCFCF", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="y", color="#FFFFFF", lw=1.1, zorder=1)
    ax.set_axisbelow(False)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels(order, rotation=42, ha="right", fontsize=12)
    for tick, l in zip(ax.get_xticklabels(), order):
        tick.set_color(SUBTYPE_COLORS.get(l, "#333")); tick.set_fontweight("bold")

    leg = [Patch(facecolor="#BBBBBB", edgecolor="#2a2a2a", hatch=h, label=lab)
           for _, lab, h in METRICS]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=14, handlelength=1.8, handleheight=1.4, columnspacing=2.0)

    figio.save_panel(fig, OUT / "persubtype_metric_bars")


if __name__ == "__main__":
    main()
