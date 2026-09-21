#!/usr/bin/env python3
"""Figure 2 (supp) - Kamath within-species subtype marker heatmap (PurpleAndYellow)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import FAMILY_COLORS, infer_family   # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
PURPLE_YELLOW = LinearSegmentedColormap.from_list("PurpleAndYellow",
                                                  ["#FF00FF", "#000000", "#FFFF00"])


def _bounds(labels):
    """Cumulative block boundaries from a per-column label list (runs preserved in order)."""
    order, bounds = [], []
    cur = 0
    for lab in labels:
        if not order or lab != order[-1]:
            if order:
                bounds.append(cur)
            order.append(lab)
        cur += 1
    bounds.append(cur)
    return order, bounds


def main():
    meta = json.loads((FROZEN / "kamath_heatmap_meta.json").read_text())
    genes = meta["genes"]; owner = meta["owner"]; subs = meta["subs"]
    blocks = meta["blocks"]; GRP_ORDER = meta["GRP_ORDER"]; GRP_COL = meta["GRP_COL"]
    SUBTYPE_COLORS = meta["SUBTYPE_COLORS"]; VLO, VHI = meta["VLO"], meta["VHI"]

    Z = pd.read_parquet(FROZEN / "kamath_heatmap_matrix.parquet").loc[genes].to_numpy()
    Za = pd.read_parquet(FROZEN / "kamath_heatmap_agg.parquet").loc[genes].to_numpy()

    sub_run, bounds = _bounds(meta["main_col_subtype"])
    grp_run, agg_bounds = _bounds(meta["agg_col_group"])
    cur, acur = bounds[-1], agg_bounds[-1]
    ng = len(genes)
    row_n = [sum(owner[g] == b for g in genes) for b in blocks]
    norm = TwoSlopeNorm(vmin=min(VLO, -1e-3), vcenter=0.0, vmax=max(VHI, 1e-3))

    starts = [0] + bounds[:-1]
    agg_starts = [0] + agg_bounds[:-1]

    fig = plt.figure(figsize=(15.5, 0.245 * ng + 3.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.02, 1.0], width_ratios=[0.30, 1.0],
                          hspace=0.01, wspace=0.02, left=0.09, right=0.9, top=0.80, bottom=0.03)
    axgb = fig.add_subplot(gs[0, 0]); axg = fig.add_subplot(gs[1, 0])
    axsb = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[1, 1])

    axg.imshow(Za, aspect="auto", cmap=PURPLE_YELLOW, norm=norm, interpolation="nearest")
    for b in agg_bounds[:-1]:
        axg.axvline(b - 0.5, color="white", lw=1.8)
    rc = 0
    for nrow in row_n[:-1]:
        rc += nrow; axg.axhline(rc - 0.5, color="white", lw=1.0)
    axg.set_xticks([]); axg.set_xlim(-0.5, acur - 0.5); axg.set_ylim(ng - 0.5, -0.5)
    axg.set_yticks(range(ng)); axg.set_yticklabels(genes, fontsize=11.5, fontstyle="italic")
    for t, g in zip(axg.get_yticklabels(), genes):
        t.set_color(GRP_COL["Anxa1+"] if owner[g] == "Anxa1+"
                    else FAMILY_COLORS.get(infer_family(owner[g]), "#333"))
    axg.tick_params(length=0)
    for spn in axg.spines.values():
        spn.set_visible(False)
    for st, en, Gn in zip(agg_starts, agg_bounds, GRP_ORDER):
        axgb.axvspan(st, en, color=GRP_COL[Gn], lw=0)
        axgb.text((st + en) / 2, 1.25, Gn, rotation=40, rotation_mode="anchor", ha="left",
                  va="bottom", fontsize=13, fontweight="bold", color=GRP_COL[Gn])
    axgb.set_xlim(-0.5, acur - 0.5); axgb.set_ylim(0, 1); axgb.axis("off")

    im = axh.imshow(Z, aspect="auto", cmap=PURPLE_YELLOW, norm=norm, interpolation="nearest")
    for b in bounds[:-1]:
        axh.axvline(b - 0.5, color="white", lw=0.7)
    rc = 0
    for nrow in row_n[:-1]:
        rc += nrow; axh.axhline(rc - 0.5, color="white", lw=1.0)
    axh.set_xticks([]); axh.set_yticks([]); axh.set_xlim(-0.5, cur - 0.5); axh.set_ylim(ng - 0.5, -0.5)
    axh.tick_params(length=0)
    for spn in axh.spines.values():
        spn.set_visible(False)
    for st, en, s in zip(starts, bounds, subs):
        axsb.axvspan(st, en, color=SUBTYPE_COLORS.get(s, "#bbb"), lw=0)
        axsb.text((st + en) / 2, 1.25, s, rotation=40, rotation_mode="anchor", ha="left",
                  va="bottom", fontsize=11.5, fontweight="bold",
                  color=FAMILY_COLORS.get(infer_family(s), "#333"))
    axsb.set_xlim(-0.5, cur - 0.5); axsb.set_ylim(0, 1); axsb.axis("off")

    cax = axh.inset_axes([1.012, 0.70, 0.016, 0.28])
    cb = plt.colorbar(im, cax=cax); cb.set_label("scaled expression\n(z; p5-p95)", fontsize=10)
    cb.ax.tick_params(labelsize=9); cb.set_ticks([VLO, 0, VHI])
    cb.set_ticklabels([f"{VLO:.1f}", "0", f"{VHI:.1f}"])

    figio.save_panel(fig, OUT / "kamath_subtype_heatmap")
    print(f"  -> kamath_subtype_heatmap  ({ng} genes x {len(subs)} subtypes + 3 pooled cols)")


if __name__ == "__main__":
    main()
