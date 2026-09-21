#!/usr/bin/env python3
"""Figure 2 - Kamath Height=5 partition UMAP (ALL DA cells)."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                    # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"


def assign_partition_colors(labels):
    """Tab20 over the sorted unique partition labels."""
    unique = sorted(set(labels))
    palette = cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    return {label: mcolors.to_hex(palette[i]) for i, label in enumerate(unique)}


def main():
    part = json.loads((FROZEN / "kamath_height5_partition.json").read_text())
    leaf_to_hp = part["leaf_to_hp"]
    height = part["height"]

    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    xy = df[["umap_x", "umap_y"]].to_numpy()
    da = ~df["gad2_excluded"].values.astype(bool)
    mask = da

    hp_labels = np.array([leaf_to_hp.get(st, "?") for st in df["pred_subtype"].to_numpy()])
    hp_color_map = assign_partition_colors(hp_labels)

    rng = np.random.RandomState(42)
    order = rng.permutation(np.where(mask)[0])
    n = int(mask.sum())

    fig, ax = plt.subplots(figsize=(8.0, 7.6))
    cols = np.array([hp_color_map.get(hp_labels[i], "#999999") for i in order])
    ax.scatter(xy[order, 0], xy[order, 1], c=cols, s=12, alpha=0.65,
               rasterized=True, edgecolors="none")

    hp_counts = Counter(hp_labels[mask])
    handles = []
    for label, cnt in hp_counts.most_common():
        if "Gad2" in label:
            continue
        short = label.split("(")[0].strip()
        if len(short) > 20:
            short = short[:18] + ".."
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=hp_color_map[label], markersize=7,
                              label=f"{short} ({cnt:,})"))
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              handletextpad=0.1, columnspacing=0.4, ncol=2, frameon=False)
    ax.set_title(f"Predicted Height={height} Partition\n(all DA cells, {n:,} cells)",
                 fontweight="bold", fontsize=14)
    ax.set_box_aspect(0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    figio.save_panel(fig, OUT / "height_5", pad_inches=0.1)
    print(f"  -> height_5  ({n:,} cells)")


if __name__ == "__main__":
    main()
