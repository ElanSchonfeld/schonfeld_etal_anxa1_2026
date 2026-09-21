#!/usr/bin/env python3
"""Figure 5 panel: pairwise genetic-stratum contrasts in putaminal DAT under nested adjustment."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig5_common import figio, FROZEN, PANELS

C = pd.read_csv(FROZEN / "genetics_contrasts.csv"); H = pd.read_csv(FROZEN / "genetics_contrasts_headers.csv")
TXX = 0.085

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for _, r in C.iterrows():
    ms = 7.5 if r.marker == 'D' else 7
    ax.plot([r.lo, r.hi], [r.y, r.y], '-', color=r.color, lw=r.lw, solid_capstyle='round')
    ax.plot(r.eff, r.y, r.marker, color=r.color, ms=ms)
    ax.text(TXX, r.y, r.stat_text, va='center', fontsize=7, color=r.color,
            fontweight='bold' if r.marker == 'D' else 'normal', clip_on=False)
for _, h in H.iterrows():
    ax.text(h.x, h.y, h.text, va='center', ha='left', fontsize=7.5, fontweight='bold', color=h.color)
ax.axvline(0, color='k', lw=0.9, ls='--'); ax.axvline(0.065, color='#ddd', lw=0.8, zorder=0)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.set_yticks(C.y); ax.set_yticklabels(C.ylabel, fontsize=7)
for lab in ax.get_yticklabels():
    if lab.get_text() == 'dx-matched': lab.set_color('#c1121f'); lab.set_fontweight('bold')
    if lab.get_text().startswith('+LEDD'): lab.set_color('#7b3294'); lab.set_fontweight('bold')
ax.set_ylim(17.9, -3.4); ax.set_xlim(-0.31, 0.075); ax.set_xticks([-0.3, -0.2, -0.1, 0])
ax.tick_params(labelsize=7)
ax.set_xlabel('Δ putamen SBR (all-visits mixed models)   left = more loss in 2nd group')
ax.set_title('D  pairwise contrasts, all timepoints + p-values', loc='left', fontweight='bold', fontsize=9)
_mk = dict(color='none', markeredgecolor='none')
ax.legend(handles=[Line2D([0], [0], marker='o', markerfacecolor='#333333', ms=6, label='nested mixed model', **_mk),
                   Line2D([0], [0], marker='o', markerfacecolor='#7b3294', ms=6, label='+LEDD (medication)', **_mk),
                   Line2D([0], [0], marker='D', markerfacecolor='#c1121f', ms=6, label='dx-matched (primary)', **_mk)],
          loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=7,
          handletextpad=0.4, columnspacing=1.0)
figio.save_panel(fig, PANELS / "genetics_pairwise_contrasts")
