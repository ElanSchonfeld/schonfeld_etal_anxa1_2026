#!/usr/bin/env python3
"""Figure 5 panel: Anxa1-territory DAT by genetic group in pre-symptomatic prodromal participants."""
import pandas as pd
import matplotlib.pyplot as plt
from fig5_common import figio, FROZEN, PANELS

pts = pd.read_csv(FROZEN / "genetics_driver_points.csv")
grp = pd.read_csv(FROZEN / "genetics_driver_groups.csv").sort_values("idx")
hc_mean = float(grp.loc[grp.group == "HC", "mean"].iloc[0])

fig, ax = plt.subplots(figsize=(5.6, 4.8))
for _, g in grp.iterrows():
    v = pts[pts.group == g.group]
    ax.scatter(v.x, v.y, s=4, color=g.color, alpha=0.30, edgecolor='none')
    ax.plot([g.idx - 0.3, g.idx + 0.3], [g["mean"], g["mean"]], color=g.color, lw=2.6, zorder=3)
    if isinstance(g.ann, str) and g.ann:
        ax.text(g.idx, 2.0, g.ann, ha='center', fontsize=9, color=g.color)
ax.axhline(hc_mean, ls='--', color='k', lw=0.7)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.set_xticks(grp.idx); ax.set_xticklabels(grp.label, fontsize=9.5); ax.tick_params(axis='y', labelsize=9.5)
ax.set_ylabel('Anxa1 putaminal territory (voxel-weighted DAT)', fontsize=10.5); ax.set_ylim(0, 2.2)
ax.set_title('Pre-symptomatic (prodromal, MDS-UPDRS III ≤ 3):\nGBA normal, LRRK2 already reduced',
             loc='left', fontweight='bold', fontsize=10.5)
figio.save_panel(fig, PANELS / "genetics_driver")
