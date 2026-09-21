#!/usr/bin/env python3
"""Drive-free render: WHERE dopamine is lost, and what the location predicts.

Run: python panel_conversion_generalises.py
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig5_common import figio, FROZEN, PANELS

OR = pd.read_csv(FROZEN / "conversion_generalises_irbd.csv").set_index("pred")
GM = json.load(open(FROZEN / "conversion_generalises_meta.json"))
F = pd.read_csv(FROZEN / "family_fingerprint_striatum_ancova.csv")
WM = pd.read_csv(FROZEN / "where_median.csv").set_index("pred")
WK = pd.read_csv(FROZEN / "where_median_km.csv")
AS = pd.read_csv(FROZEN / "partition_asymmetry_otsu.csv")

ANX, NEG, LIGHT = "#22336b", "#b8332e", "#c4c8d0"
TCOL = {'least': "#9ec4e0", 'mid': "#7f8c9b", 'most': "#22336b"}
PCOL = {'anxa1_str': "#22336b", 'sox6_str': "#7f8c9b", 'calb1_str': "#b8332e"}
FAMK = ['anxa1_str', 'sox6_str', 'calb1_str']
NM = ['Anxa1', 'Sox6-all', 'Calb1']
HILITE = {'motor daily living (UPDRS-II)': ANX, 'PIGD (Stebbins)': ANX, 'rest tremor': ANX,
          'non-motor daily living (UPDRS-IB, 6 item)': ANX, 'bradykinesia-rigidity': ANX,
          'depression (GDS-15)': NEG, 'anxiety (STAI)': NEG,
          'cognitive / behavioral (UPDRS-I rated)': NEG, 'global cognition (MoCA)': NEG}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(16.4, 9.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.72], height_ratios=[1.75, 0.78, 0.82],
                      hspace=0.52, wspace=0.19, left=0.070, right=0.985, top=0.935, bottom=0.068)

ax = fig.add_subplot(gs[0, 0])
for kk in FAMK:
    x = WM.loc[kk]
    g = WK[(WK.pred == kk) & (WK.half == 'high') & (WK.t <= 5)].sort_values('t')
    ax.step(g.t, 100 * (1 - g.surv), where='post', lw=2.9 if kk == 'anxa1_str' else 2.1,
            color=PCOL[kk], alpha=1.0 if kk == 'anxa1_str' else 0.9,
            zorder=5 if kk == 'anxa1_str' else 3,
            label=f"{x['label']:<9s}{x['risk3_high']:5.1f}% at 3 y   HR {x['HR']:.2f}  "
                  + ("p<0.001" if x['p'] < 1e-3 else f"p={x['p']:.3f}"))
ax.axvline(3.0, color="#aaaaaa", lw=0.8, ls=':')
leg = ax.legend(fontsize=7.2, frameon=False, loc='upper left',
                prop={'family': 'DejaVu Sans Mono', 'size': 7.2},
                title="the half with the LARGEST share of the loss in that territory",
                title_fontsize=6.8)
leg.get_title().set_fontfamily('DejaVu Sans Mono')
ax.set_xlabel("years since baseline DaTscan", fontsize=8)
ax.set_ylabel("cumulative conversion to PD (%)", fontsize=8)
ax.set_title("a  WHERE the loss sits - Anxa1 $\\to$ Sox6-all $\\to$ Calb1,\n"
             "     at matched total striatal burden, age and sex", fontsize=9.4, loc='left')
ax.tick_params(labelsize=7.5)

ax = fig.add_subplot(gs[1, 0])
o = OR.reindex(FAMK); yy = np.arange(len(o))[::-1]
for i2, (k2, x2) in enumerate(o.iterrows()):
    c2 = ANX if x2.HR > 1 else NEG
    ax.plot([x2.lo, x2.hi], [yy[i2]] * 2, color=c2, lw=2.0)
    ax.plot(x2.HR, yy[i2], 'o', color=c2, ms=6.0, mec='k', mew=0.7)
    ax.annotate(f"{x2.HR:.2f}  " + (f"p={x2.p:.0e}" if x2.p < 1e-3 else f"p={x2.p:.3f}"),
                xy=(0.985, yy[i2]), xycoords=('axes fraction', 'data'), va='center', ha='right',
                fontsize=6.4, fontweight='bold' if x2.p < 0.05 else 'normal')
ax.axvline(1.0, color="#333333", lw=1.0, ls='--')
ax.set_xscale('log'); ax.set_xlim(0.24, 3.6); ax.set_xticks([0.3, 0.5, 1.0, 2.0, 3.0])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(-0.65, len(o) - 0.35)
ax.set_yticks(yy); ax.set_yticklabels(['Anxa1', 'Sox6-all', 'Calb1'], fontsize=7.0)
ax.get_yticklabels()[0].set_fontweight('bold')
ax.set_xlabel("odds of iRBD per SD of depletion, at matched total burden", fontsize=7.2)
ax.set_title(f"b  WHICH territory is preferentially lost in iRBD\n"
             f"     ({GM['n_irbd']} iRBD of {GM['n']} prodromals)", fontsize=9.4, loc='left')
ax.tick_params(labelsize=7.2)

ax = fig.add_subplot(gs[2, 0])
DES = ['bilateral posterior putamen', 'more-affected hemisphere', 'less-affected hemisphere']
yy = np.arange(len(DES))[::-1]
for k2, de in enumerate(DES):
    r = AS[AS.design == de].iloc[0]; y0 = yy[k2]
    ax.barh(y0, r.lr_fwd, height=0.52, color=ANX, zorder=3)
    ax.barh(y0, -r.lr_rev, height=0.52, color="#c4c8d0", zorder=3)
    ax.text(r.lr_fwd - 0.7, y0, f"$\\chi^2$ {r.lr_fwd:.1f}  p<0.001", va='center', ha='right',
            fontsize=6.9, color='white', fontweight='bold')
    ax.text(-4.25, y0, f"{r.lr_rev:.1f}   p={r.p_rev:.2f}", va='center', ha='right',
            fontsize=6.9, color="#8a8f99")
ax.axvline(0, color="#333333", lw=1.2, zorder=4)
for xv in (3.84, -3.84): ax.axvline(xv, color="#b8332e", lw=0.9, ls='--', zorder=2)
ax.set_xlim(-10, 23); ax.set_ylim(-0.85, len(DES) + 0.55)
ax.set_yticks(yy)
ax.set_yticklabels([d_.replace(' posterior putamen', ' post. put.').replace(' hemisphere', ' hemi.')
                    for d_ in DES], fontsize=6.8)
ax.set_xticks([-3.84, 0, 10, 20]); ax.set_xticklabels(['3.8', '0', '10', '20'], fontsize=7.0)
ax.spines['bottom'].set_bounds(-10, 23)
ax.spines['left'].set_visible(False)
ax.set_xlabel("likelihood-ratio $\\chi^2$(1) for ADDING the second region\n"
              "to a Cox model that already contains the first", fontsize=7.2)
bb = dict(boxstyle='square,pad=0.15', fc='white', ec='none')
ax.text(-4.25, len(DES) + 0.20, "adding the complement", ha='right', va='center',
        fontsize=6.5, color="#8a8f99", fontweight='bold', bbox=bb)
ax.text(9.5, len(DES) + 0.20, "adding the ANXA1-enriched territory", ha='center', va='center',
        fontsize=6.5, color=ANX, fontweight='bold', bbox=bb)
ax.set_title("c  The ANXA1-enriched territory contains what its complementary\n"
             "     posterior putamen knows - Otsu split (319 / 369 voxels)", fontsize=9.4, loc='left')
ax.tick_params(labelsize=7.0)

CL = {'motor': (['bradykinesia-rigidity', 'motor daily living (UPDRS-II)', 'PIGD (Stebbins)',
                 'rest tremor', 'action/postural tremor', 'Schwab & England ADL'], "#22336b"),
      'other non-motor': (['non-motor daily living (UPDRS-IB, 6 item)', 'Pain (UPDRS 1.9)',
                           'RBD symptoms (RBDSQ)', 'daytime sleepiness (ESS)',
                           'autonomic symptoms (SCOPA-AUT 4-21)', 'bulbar: swallow/drool (SCOPA 1-3)'], "#2f8f7a"),
      'cognitive': (['global cognition (MoCA)', 'processing speed (SDMT)', 'verbal learning (HVLT-R)',
                     'visuospatial (Benton JLO)', 'working memory (LNS)', 'semantic fluency'], "#9aa0ad"),
      'neuropsychiatric': (['depression (GDS-15)', 'anxiety (STAI)',
                            'cognitive / behavioral (UPDRS-IA)', 'impulsive-compulsive (QUIP)'], "#b8332e")}
dom2c = {d_: c for _, (ds, c) in CL.items() for d_ in ds}
assert set(F.outcome) == {'ancova_3y'}
AXS = {}
for coh in ['prodromal', 'PD']:
    v = F[F.cohort == coh].pivot(index='domain', columns='territory', values='beta')
    AXS[coh] = v['anxa1_str'] - v['calb1_str']
A = pd.DataFrame(AXS).dropna()

ax = fig.add_subplot(gs[:, 1])
xmin, xmax = float(A.prodromal.min()), float(A.prodromal.max())
ymin, ymax = float(A.PD.min()), float(A.PD.max())
xr0, yr0 = xmax - xmin, ymax - ymin
XL = (xmin - 0.26 * xr0, xmax + 0.20 * xr0)
YL = (ymin - 0.24 * yr0, ymax + 0.36 * yr0)
sx, sy = XL[1] - XL[0], YL[1] - YL[0]
ax.axhspan(0, YL[1], color="#22336b", alpha=0.030, zorder=0)
ax.axhspan(YL[0], 0, color="#b8332e", alpha=0.025, zorder=0)
lo, hi = max(XL[0], YL[0]), min(XL[1], YL[1])
ax.plot([lo, hi], [lo, hi], color="#8a8f99", lw=1.0, zorder=1)
ax.axhline(0, color="#333333", lw=0.9, ls='--', zorder=1)
ax.axvline(0, color="#333333", lw=0.9, ls='--', zorder=1)
ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_box_aspect(1)

pts = [(r['prodromal'], r['PD']) for _, r in A.iterrows()]
SHORT = {'bulbar: swallow/drool': 'swallow / drool',
         'Schwab & England ADL': 'Schwab-England ADL',
         'Pain': 'pain',
         'cognitive / behavioral': 'cognitive/behavioral'}
LFS, AFS, TFS, CFS = 10.5, 12.5, 12.5, 10.5
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
def tw(s_, fs):
    font = matplotlib.font_manager.FontProperties(size=fs)
    w = renderer.get_text_width_height_descent(s_, font, ismath=False)[0]
    return (w / ax.bbox.width + 0.010) * sx
def txh(fs):
    font = matplotlib.font_manager.FontProperties(size=fs)
    h = renderer.get_text_width_height_descent("Xg", font, ismath=False)[1]
    return (h / ax.bbox.height + 0.006) * sy

LABELS = {
    'rest tremor':                              ('left',   'center',  0.0045,  0.0000, False),
    'motor daily living (UPDRS-II)':            ('center', 'bottom',  0.0000,  0.0035, False),
    'working memory (LNS)':                     ('center', 'bottom',  0.0000,  0.0035, False),
    'processing speed (SDMT)':                  ('center', 'bottom',  0.0080,  0.0035, False),
    'visuospatial (Benton JLO)':                ('left',   'bottom', -0.0010,  0.0030, False),
    'bulbar: swallow/drool (SCOPA 1-3)':        ('center', 'bottom',  0.0000,  0.0035, False),
    'action/postural tremor':                   ('left',   'top',    -0.0180, -0.0030, False),
    'bradykinesia-rigidity':                    ('left',   'center',  0.0045,  0.0000, False),
    'non-motor daily living (UPDRS-IB, 6 item)': ('left',  'center',  0.0045,  0.0000, False),
    'autonomic symptoms (SCOPA-AUT 4-21)':      ('left',   'center',  0.0045,  0.0000, False),
    'semantic fluency':                         ('center', 'top',     0.0060, -0.0035, False),
    'PIGD (Stebbins)':                          ('center', 'top',    -0.0030, -0.0035, False),
    'Schwab & England ADL':                     ('center', 'top',     0.0000, -0.0035, False),
    'RBD symptoms (RBDSQ)':                     ('right',  'center', -0.0045,  0.0000, False),
    'depression (GDS-15)':                      ('right',  'center', -0.0060,  0.0000, False),
    'anxiety (STAI)':                           ('left',   'top',     0.0035, -0.0030, False),
    'impulsive-compulsive (QUIP)':              ('center', 'top',    -0.0260, -0.0030, False),
    'Pain (UPDRS 1.9)':                         ('right',  'center', -0.0045,  0.0000, False),
    'daytime sleepiness (ESS)':                 ('center', 'center', -0.0010,  0.0280, True),
    'verbal learning (HVLT-R)':                 ('center', 'center', -0.0350,  0.0105, True),
    'global cognition (MoCA)':                  ('center', 'center', -0.0440,  0.0095, True),
    'cognitive / behavioral (UPDRS-IA)':        ('center', 'center', -0.0440, -0.0100, True),
}
assert set(A.index) == set(dom2c) == set(LABELS), "Every plotted assessment needs a label and category"

for dom, (ha, va, dx, dy, lead) in LABELS.items():
    px, py = float(A.loc[dom, 'prodromal']), float(A.loc[dom, 'PD'])
    c = dom2c[dom]
    txt = SHORT.get(dom.split(' (')[0], dom.split(' (')[0])
    tx, ty = px + dx, py + dy
    ax.text(tx, ty, txt, ha=ha, va=va, fontsize=LFS, color=c, zorder=6, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.15))
    if lead:
        bw, bh = tw(txt, LFS), txh(LFS)
        vx, vy = (tx - px) / sx, (ty - py) / sy
        L = np.hypot(vx, vy); ux, uy = vx / L, vy / L
        t_edge = min((bw / 2 / sx) / abs(ux) if abs(ux) > 1e-9 else np.inf,
                     (bh / 2 / sy) / abs(uy) if abs(uy) > 1e-9 else np.inf)
        t0, t1 = 0.016, L - t_edge - 0.004
        if t1 > t0:
            ax.plot([px + ux * t0 * sx, px + ux * t1 * sx], [py + uy * t0 * sy, py + uy * t1 * sy],
                    color=c, lw=1.1, alpha=0.85, zorder=2, solid_capstyle='round')
for (px, py), dom in zip(pts, A.index):
    ax.plot(px, py, 'o', color=dom2c[dom], ms=9.5, mec='white', mew=1.1, zorder=5)

ax.set_xlabel("Prodromal: relative association with 3-year change\n"
              r"$\beta$(Anxa1 depletion) $-$ $\beta$(Calb1 depletion)", fontsize=AFS)
ax.set_ylabel("PD: relative association with 3-year change\n"
              r"$\beta$(Anxa1 depletion) $-$ $\beta$(Calb1 depletion)", fontsize=AFS)
ax.set_title("d  Territorial associations across cohorts\n"
             f"     All points: change at ~3 y, baseline-adjusted; across-domain r = {A.prodromal.corr(A.PD):.2f}",
             fontsize=TFS, loc='left', pad=14)
for xf, yf, txt, col, ha, va in [
        (0.985, 0.985, "Larger Anxa1 coefficient in both", ANX, 'right', 'top'),
        (0.015, 0.015, "Larger Calb1 coefficient in both", NEG, 'left', 'bottom')]:
    ax.text(xf, yf, txt, transform=ax.transAxes, fontsize=CFS, color=col, ha=ha, va=va,
            fontweight='bold', style='italic')
for nm2, (ds, c) in CL.items():
    ax.plot([], [], 'o', color=c, ms=8, label=nm2)
ax.legend(fontsize=LFS, frameon=False, loc='upper left', bbox_to_anchor=(0.015, 0.94), ncol=2,
          columnspacing=1.0, handletextpad=0.4)
ax.tick_params(labelsize=11)

figio.save_panel(fig, PANELS / "fig5_conversion_generalises")
print("done ->", PANELS / "fig5_conversion_generalises.png")
