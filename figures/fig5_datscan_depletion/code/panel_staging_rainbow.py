#!/usr/bin/env python3
"""Figure 5 panel: enrichment-weighted % dopamine loss by staging for four projection groups."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from fig5_common import figio, FROZEN, PANELS, GORD, GLAB, SCHEME_TITLE
from vector_cells import draw_cells

mat = pd.read_csv(FROZEN / "staging_rainbow_matrix.csv")
sc = json.load(open(FROZEN / "staging_rainbow_scale.json"))
VMIN, VMAX = sc["vmin"], sc["vmax"]; _lo, _hi = VMIN + 0.15 * (VMAX - VMIN), VMIN + 0.85 * (VMAX - VMIN)

fig = plt.figure(figsize=(13.5, 3.8))
gs = fig.add_gridspec(1, 3, left=0.14, right=0.88, top=0.80, bottom=0.20, wspace=0.09, width_ratios=[7, 4, 4])
im0 = None
for ci, sk in enumerate(["time", "hy", "updrs"]):
    sub = mat[mat.scheme == sk]
    order = sub[sub.group == GORD[0]].stage.tolist()
    M = np.array([[float(sub[(sub.group == g) & (sub.stage == s)].value.iloc[0]) for s in order] for g in GORD])
    ax = fig.add_subplot(gs[0, ci]); im0 = draw_cells(ax, M, 'turbo', VMIN, VMAX)
    for a in range(len(GORD)):
        for b in range(len(order)):
            vv = M[a, b]; ax.text(b, a, f"{vv:.0f}", ha='center', va='center', fontsize=8.5, fontweight='bold',
                                  color='white' if (vv < _lo or vv > _hi) else 'black')
    ax.set_xticks(range(len(order))); ax.set_xticklabels([s.replace('UPDRS ', '').replace('H&Y ', '') for s in order], fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(GORD)))
    if ci == 0:
        ax.set_yticklabels([GLAB[g] for g in GORD], fontsize=9); ax.set_ylabel("Overall (dorsal)", fontsize=11.5, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.set_title(SCHEME_TITLE[sk], fontsize=10.5, fontweight='bold'); ax.tick_params(length=0)
cax = fig.add_axes([0.895, 0.22, 0.012, 0.56])
fig.colorbar(im0, cax=cax).set_label(f"% dopamine loss (PD vs HC)  [p5-p95: {VMIN:.0f}-{VMAX:.0f}%]", fontsize=8.5)
figio.save_panel(fig, PANELS / "staging_rainbow")
