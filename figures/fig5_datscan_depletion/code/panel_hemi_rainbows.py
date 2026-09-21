#!/usr/bin/env python3
"""Figure 5 supplementary panel: hemisphere-aligned staging depletion heatmaps."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from fig5_common import figio, FROZEN, PANELS, GORD, GLAB_S, SCHEME_TITLE
from vector_cells import draw_cells

mat = pd.read_csv(FROZEN / "hemi_rainbow_matrix.csv")
sc = json.load(open(FROZEN / "hemi_rainbow_scale.json"))
VMIN, VMAX = sc["vmin"], sc["vmax"]; _lo, _hi = VMIN + 0.15 * (VMAX - VMIN), VMIN + 0.85 * (VMAX - VMIN)
ROWS = sc["row_order"]

fig = plt.figure(figsize=(9.5, 12.0))
gs = fig.add_gridspec(6, 3, left=0.16, right=0.88, top=0.925, bottom=0.05, wspace=0.09, hspace=0.32, width_ratios=[7, 4, 4])
im0 = None
for ri, rowname in enumerate(ROWS):
    for ci, sk in enumerate(["time", "hy", "updrs"]):
        sub = mat[(mat.row == rowname) & (mat.scheme == sk)]
        order = sub[sub.group == GORD[0]].stage.tolist()
        M = np.array([[float(sub[(sub.group == g) & (sub.stage == s)].value.iloc[0]) for s in order] for g in GORD])
        ax = fig.add_subplot(gs[ri, ci]); im0 = draw_cells(ax, M, 'turbo', VMIN, VMAX)
        for a in range(len(GORD)):
            for b in range(len(order)):
                vv = M[a, b]; ax.text(b, a, f"{vv:.0f}", ha='center', va='center', fontsize=8, fontweight='bold',
                                      color='white' if (vv < _lo or vv > _hi) else 'black')
        ax.set_xticks(range(len(order))); ax.set_xticklabels([s.replace('UPDRS ', '').replace('H&Y ', '') for s in order], fontsize=7.5, rotation=30, ha='right')
        ax.set_yticks(range(len(GORD)))
        if ci == 0:
            ax.set_yticklabels([GLAB_S[g] for g in GORD], fontsize=8.5); ax.set_ylabel(rowname, fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])
        if ri == 0: ax.set_title(SCHEME_TITLE[sk], fontsize=10.5, fontweight='bold')
        ax.tick_params(length=0)
cax = fig.add_axes([0.905, 0.35, 0.014, 0.30]); fig.colorbar(im0, cax=cax).set_label(f"% dopamine loss (PD vs HC)  [p5-p95: {VMIN:.0f}-{VMAX:.0f}%]", fontsize=8.5)
fig.text(0.52, 0.955, "Hemisphere-aligned staging depletion (MA vs LA): more-affected putamen degenerates deeper;\nAnxa1-territory leads within each; caudate spared in both",
         ha='center', fontsize=10, fontweight='bold')
figio.save_panel(fig, PANELS / "hemi_rainbows")
