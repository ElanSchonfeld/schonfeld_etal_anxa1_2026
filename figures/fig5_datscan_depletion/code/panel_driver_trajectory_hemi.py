#!/usr/bin/env python3
"""Figure 5 supplementary panel: Anxa1 putaminal territory DAT by genetic group and hemisphere.

Run: python panel_driver_trajectory_hemi.py
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch
from fig5_common import figio, FROZEN, PANELS

D = pd.read_csv(FROZEN / "driver_trajectory_hemi.csv")
HC = float(D[D.kind == 'hc'].iloc[0]['mean'])
XCAP_PROD = float(D[D.kind == 'xcap'].iloc[0]['x'])

COL = {'idiopathic': '#6b6b6b', 'none': '#8a8a8a', 'LRRK2': '#3f8fd0', 'GBA': '#2ca25f'}
RALAB = 'Anxa1 putaminal territory\n(voxel-weighted DAT)'
FS = dict(tick=10, label=11, title=11.5, legend=10, note=9.5, row=13)


def _traj(kind, side, group):
    r = D[(D.kind == kind) & (D.side == side) & (D.group == group)].sort_values('x')
    return r.x.to_numpy(), r['mean'].to_numpy(), r['sem'].to_numpy()


fig = plt.figure(figsize=(9.5, 8.4))
gs = GridSpec(2, 1, figure=fig, hspace=0.38, left=0.15, right=0.98, top=0.95, bottom=0.08)


def panel(sub, side):
    gsb = GridSpecFromSubplotSpec(1, 2, subplot_spec=sub, wspace=0.0, width_ratios=[1, 1.25])
    a0 = fig.add_subplot(gsb[0, 0]); a1 = fig.add_subplot(gsb[0, 1], sharey=a0); a0.set_facecolor('#faf6ec')
    a0.set_ylim(0.1, 1.4); CONV_X = XCAP_PROD; XEDGE = XCAP_PROD + 2
    a0.axhline(HC, ls='--', color='k', lw=0.8); a0.text(0.3, HC + 0.025, 'HC', fontsize=FS['note'], ha='left'); a1.axhline(HC, ls='--', color='k', lw=0.8)
    pend = {}
    for carr_ in ['none', 'LRRK2', 'GBA']:
        c = COL[carr_]; xs, ys, es = _traj('prod_traj', 'bilateral', carr_)
        if len(xs):
            a0.errorbar(xs, ys, yerr=es, fmt='-o', color=c, ms=5, lw=2.0, capsize=3, zorder=3)
            pend[carr_] = (xs[-1], ys[-1])
    a0.set_xlim(0, XEDGE); a0.set_xticks([t for t in range(0, 13, 2) if t <= XEDGE]); a0.tick_params(labelsize=FS['tick'])
    a0.set_xlabel('MDS-UPDRS III (prodromal)', fontsize=FS['label']); a0.set_ylabel(RALAB, fontsize=FS['label'])
    a0.spines['right'].set_visible(False); a0.set_title('prodromal (bilateral)', fontsize=FS['title'], color='#8a7a4a', fontweight='bold')
    dstart = {}
    for g in ['idiopathic', 'LRRK2', 'GBA']:
        c = COL[g]; xs, ys, es = _traj('diag_traj', side, g)
        if len(xs): a1.errorbar(xs, ys, yerr=es, fmt='-o', color=c, ms=5, lw=2.0, capsize=3, label=g, zorder=3); dstart[g] = (xs[0], ys[0])
    a1.set_xlim(0, 9); a1.set_xticks([0, 2, 4, 6, 8, 8.5]); a1.set_xticklabels(['0', '2', '4', '6', '8', '8+']); a1.tick_params(labelsize=FS['tick'])
    a1.set_xlabel('years since motor onset (diagnosed)', fontsize=FS['label']); a1.set_title(f'diagnosed ({side})', fontsize=FS['title'], color='#555', fontweight='bold')
    a1.legend(frameon=False, fontsize=FS['legend'], loc='upper right'); a1.spines['left'].set_visible(False); a1.tick_params(left=False); plt.setp(a1.get_yticklabels(), visible=False)
    for carr_, g in [('none', 'idiopathic'), ('LRRK2', 'LRRK2'), ('GBA', 'GBA')]:
        if carr_ in pend and g in dstart:
            fig.add_artist(ConnectionPatch(xyA=pend[carr_], coordsA=a0.transData, xyB=dstart[g], coordsB=a1.transData, ls=(0, (2, 1.5)), color=COL[g], lw=1.4, alpha=0.9, zorder=4))
    a1.axvline(0, ls=':', color='#c1121f', lw=2.0, zorder=6, clip_on=False); a1.text(0.2, 1.37, 'motor onset', fontsize=FS['note'], color='#c1121f', fontweight='bold', ha='left', va='top')
    _rp = sub.get_position(fig)
    fig.text(0.03, (_rp.y0 + _rp.y1) / 2, f'{"MORE" if side=="MA" else "LESS"}-AFFECTED\n({side})', rotation=90, ha='center', va='center',
             fontsize=FS['row'], fontweight='bold', color='#c1121f' if side == 'MA' else '#5a5a5a')


for r, side in enumerate(['MA', 'LA']):
    panel(gs[r, 0], side)
figio.save_panel(fig, PANELS / "driver_trajectory_hemi")
