#!/usr/bin/env python3
"""Figure 5 supplementary panel: hemisphere-aligned genetic-stratum comparison."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from fig5_common import figio, FROZEN, PANELS

COL = {'idiopathic': '#6b6b6b', 'LRRK2': '#3f8fd0', 'GBA': '#2ca25f'}
GR = ['idiopathic', 'LRRK2', 'GBA']
BARS = pd.read_csv(FROZEN / "hemi_confound_bars.csv")
CONTR = pd.read_csv(FROZEN / "hemi_confound_contrasts.csv")
CONC = pd.read_csv(FROZEN / "hemi_confound_concordance.csv")
INS = np.load(FROZEN / "hemi_confound_inset.npz")
N_PAIRS = json.load(open(FROZEN / "hemi_confound_meta.json"))["n_pairs"]

fig = plt.figure(figsize=(11.5, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30, left=0.06, right=0.975, top=0.92, bottom=0.075)

def bget(subpanel, key, hemi, group):
    r = BARS[(BARS.subpanel == subpanel) & (BARS.key == key) & (BARS.hemi == hemi) & (BARS.group == group)]
    return r.iloc[0]

axB = fig.add_subplot(gs[0, 0])
XG = [('caudate', 'MA'), ('caudate', 'LA'), ('anxa1', 'MA'), ('anxa1', 'LA')]
w = 0.20
for xi, (rlab, suf) in enumerate(XG):
    idio = float(bget('B', rlab, suf, 'idiopathic').value)
    for j, g in enumerate(GR):
        row = bget('B', rlab, suf, g); x = xi + (j - 1) * w
        val, lo, hi_ = float(row.value), float(row.lo), float(row.hi)
        axB.errorbar(x, val, yerr=[[val - lo], [hi_ - val]], fmt='o', ms=6, color=COL[g],
                     ecolor=COL[g], elinewidth=1.4, capsize=2.5, mec='white', mew=0.9, zorder=3)
        if g != 'idiopathic':
            axB.text(x, lo * 0.955, f"{100*(val-idio)/idio:+.0f}%", ha='center', va='top',
                     fontsize=6.4, color=COL[g], fontweight='bold')
        if isinstance(row.star, str) and row.star and row.star != 'n.s.':
            axB.text(x, hi_ * 1.03, row.star, ha='center', va='bottom', fontsize=7, fontweight='bold', color=COL[g])
axB.set_yscale('log')
axB.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0]); axB.set_yticklabels(['0.2', '0.3', '0.5', '0.7', '1.0'])
axB.set_ylim(0.135, 1.30); axB.minorticks_off()
axB.set_xticks(range(len(XG)))
axB.set_xticklabels([f"caudate\nMA", "caudate\nLA", "Anxa1 territory\nMA", "Anxa1 territory\nLA"], fontsize=7.5)
axB.axvline(1.5, color='#dddddd', lw=0.8, zorder=0)
axB.set_ylabel('adjusted SBR (age+sex+onset+dx)\nlog scale', fontsize=8)
axB.tick_params(labelsize=8)
axB.legend(handles=[Line2D([0], [0], marker='o', ls='none', color=COL[g], mec='white', ms=6, label=g) for g in GR],
           frameon=False, fontsize=7, loc='lower left', ncol=3, columnspacing=0.9, handletextpad=0.3)
axB.set_title('A  Deficit is the Anxa1 putaminal territory, not caudate\n% = change vs idiopathic; stars = GBA vs idio', loc='left', fontweight='bold', fontsize=9)

axD = fig.add_subplot(gs[0, 1]); contrasts = [('idiopathic', 'LRRK2'), ('idiopathic', 'GBA'), ('LRRK2', 'GBA')]
yl = []; ylab = []; yy = 0
for g1, g2 in contrasts:
    for suf, off, mfc in [('MA', 0.16, True), ('LA', -0.16, False)]:
        row = CONTR[(CONTR.g1 == g1) & (CONTR.g2 == g2) & (CONTR.hemi == suf)].iloc[0]
        eff, lo, hi, p, n = float(row.eff), float(row.lo), float(row.hi), float(row.pval), int(row.n)
        c = COL[g2] if g1 == 'idiopathic' else '#7b3294'
        axD.plot([lo, hi], [yy + off, yy + off], '-', color=c, lw=2.6, solid_capstyle='round', alpha=1.0 if suf == 'MA' else 0.5)
        axD.plot(eff, yy + off, 'D', color=c if mfc else 'white', mec=c, ms=8 if suf == 'MA' else 6.5, mew=1.2)
        axD.text(0.075, yy + off, f"{suf}: {eff:+.3f} {row.star} (n={n})", va='center', fontsize=6.6, color=c, clip_on=False, fontweight='bold' if suf == 'MA' else 'normal')
    yl.append(yy); ylab.append(f"{g2} vs {g1}"); yy += 1
axD.axvline(0, color='k', lw=0.9, ls='--'); axD.axvline(0.065, color='#ddd', lw=0.8, zorder=0)
axD.set_yticks(yl); axD.set_yticklabels(ylab, fontsize=8); axD.set_ylim(-0.7, yy - 0.3)
axD.set_xlim(-0.34, 0.07); axD.set_xticks([-0.3, -0.2, -0.1, 0]); axD.tick_params(labelsize=7.5)
axD.set_xlabel('delta Anxa1-putaminal DAT (dx-matched, mediator-free)   left = more loss in 2nd group')
axD.set_title('B  dx-matched contrasts, MA (filled) vs LA (hollow)', loc='left', fontweight='bold', fontsize=9)

elo, ehi = float(INS['elo']), float(INS['ehi'])
gsE = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.10); imE = None
for r, (tag, hlab) in enumerate([('MA', 'MA (floored)'), ('LA', 'LA (dynamic range)')]):
    axG = fig.add_subplot(gsE[r, 0])
    axG.imshow(INS[f'{tag}_bg'], cmap='gray', interpolation='bilinear', aspect='equal')
    imE = axG.imshow(INS[f'{tag}_disp'], cmap='inferno', vmin=elo, vmax=ehi, interpolation='nearest', aspect='equal')
    axG.contour(INS[f'{tag}_put'], levels=[0.5], colors='k', linewidths=0.6)
    axG.set_xticks([]); axG.set_yticks([]); axG.set_ylabel(hlab, fontsize=8, fontweight='bold')
    for sp in axG.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
    if r == 0: axG.set_title('C  GBA excess loss (idiopathic - GBA SUVR)\nz = 0 mm; MA vs LA hemisphere', fontsize=8, fontweight='bold')
pE = gs[1, 0].get_position(fig)
cax = fig.add_axes([pE.x1 + 0.002, pE.y0 + 0.02, 0.007, pE.height * 0.6]); cb = fig.colorbar(imE, cax=cax); cb.set_label('excess deficit', fontsize=6.5); cb.ax.tick_params(labelsize=6)

axF = fig.add_subplot(gs[1, 1])
TERR_COL = {'Anxa1': '#22336b', 'Sox6': '#2f6fb2'}
rows = [(t, terr) for t in ('MA', 'LA') for terr in ('Anxa1', 'Sox6')]
for i, (tag, terr) in enumerate(rows):
    c = CONC[(CONC.hemi == tag) & (CONC.territory == terr)].iloc[0]
    y = len(rows) - 1 - i; col = TERR_COL[terr]
    axF.plot([float(c.lo), float(c.hi)], [y, y], color=col, lw=5.5, solid_capstyle='round',
             zorder=2, alpha=1.0 if tag == 'MA' else 0.55)
    axF.scatter([float(c.rho)], [y], color=col, s=95, zorder=3, edgecolor='w', lw=1.1)
    axF.text(float(c.rho), y + 0.22, f"{float(c.rho):+.2f} {c.star}", ha='center',
             fontsize=8, fontweight='bold', color=col)
axF.axhline(1.5, color='#dddddd', lw=0.8, zorder=0)
axF.axvline(0, color='#444', lw=0.9, ls='--')
axF.set_ylim(-0.6, len(rows) - 0.3); axF.set_xlim(-1.0, 1.0)
axF.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
axF.set_yticklabels([f"{t}  {terr}" for t, terr in rows], fontsize=7.6)
for lbl, (t, _terr) in zip(axF.get_yticklabels(), rows):
    lbl.set_fontweight('bold' if t == 'MA' else 'normal')
axF.tick_params(axis='y', length=0)
axF.set_xlabel('Spearman rho  (projection territory / GBA excess loss)\nblock-bootstrap 95% CI')
axF.set_title(f'D  Anxa1 territory tracks GBA excess loss; Sox6 territory runs\nopposite. Clearest in LA (MA floored)  (n={N_PAIRS} matched GBA pairs)',
              loc='left', fontweight='bold', fontsize=9)

fig.suptitle('Hemisphere-aligned genetic confound (clinical DOMSIDE MA/LA): the more-affected hemisphere is near-floored (uniformly severe loss), so BOTH the GBA excess magnitude (panel B) and its spatial match to the Anxa1 projection territory (panel D) are revealed most clearly in the LESS-affected hemisphere, which retains dynamic range.',
             fontsize=8.7, fontweight='bold', y=0.975)
figio.save_panel(fig, PANELS / "hemi_genetics_confound")
