#!/usr/bin/env python3
"""Figure 5 panel: pairwise group differences in % dopamine loss per structure."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from fig5_common import figio, FROZEN, PANELS, GORD, GLAB
from vector_cells import draw_cells

Sdf = pd.read_csv(FROZEN / "delta_stats.csv"); nsub = json.load(open(FROZEN / "delta_meta.json"))["n_baselines"]
_hasrbc = 'rbc' in Sdf.columns

fig = plt.figure(figsize=(13.5, 4.6))
dtxt = fig.text(0.5, 0.93, f"Pairwise Δ % loss (row - column, median of within-subject paired differences) + significance + rank-biserial (rbc) - per-subject paired Wilcoxon, n={nsub} baselines, Holm-adjusted",
                ha='center', fontsize=10.5, fontweight='bold')
gs = fig.add_gridspec(1, 3, left=0.14, right=0.88, top=0.80, bottom=0.16, wspace=0.16); imd = None
for si, struct in enumerate(["Overall", "Putamen", "Caudate"]):
    sub = Sdf[Sdf.structure == struct]; pdd = {}; sg = {}; rbc = {}
    for _, r in sub.iterrows():
        pdd[(r.group1, r.group2)] = r.median_diff; pdd[(r.group2, r.group1)] = -r.median_diff
        sg[frozenset([r.group1, r.group2])] = r.sig
        if _hasrbc: rbc[(r.group1, r.group2)] = r.rbc
    Dm = np.array([[0.0 if gi == gj else pdd[(gi, gj)] for gj in GORD] for gi in GORD])
    ax = fig.add_subplot(gs[0, si]); imd = draw_cells(ax, Dm, 'RdBu_r', -25, 25, aspect='equal')
    for i in range(4):
        for j in range(4):
            if i == j: ax.text(j, i, '–', ha='center', va='center', fontsize=11, color='#888'); continue
            d = Dm[i, j]; s = sg.get(frozenset([GORD[i], GORD[j]]), '')
            rb = rbc.get((GORD[i], GORD[j]), -rbc.get((GORD[j], GORD[i]), np.nan)) if _hasrbc else np.nan
            txt = f"{d:+.0f}\n{s}" + (f"\nrbc {rb:+.2f}" if np.isfinite(rb) else "")
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.2, fontweight='bold', color='white' if abs(d) > 13 else 'black')
    ax.set_xticks(range(4)); ax.set_xticklabels([GLAB[g] for g in GORD], fontsize=7.5, rotation=30, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels([GLAB[g] for g in GORD] if si == 0 else [], fontsize=7.5)
    ax.set_title(struct, fontsize=11.5, fontweight='bold', pad=4); ax.tick_params(length=0)
cax = fig.add_axes([0.895, 0.30, 0.012, 0.34]); fig.colorbar(imd, cax=cax).set_label("Δ median % loss\n(row - column)", fontsize=8)
figio.save_panel(fig, PANELS / "delta_matrices")
