#!/usr/bin/env python
"""WHERE the loss sits: conversion by territory share, at matched total burden.

Run: python freeze_fig5_conversion_where.py
"""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from survutil import cox_fit
from conversion_paths import PRIVATE

HERE = os.path.dirname(os.path.abspath(__file__)); FROZEN = os.path.join(HERE, "..", "frozen")
T, E = 'tyears_pd', 'event_pd'
zs = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
FAMK = [('anxa1_str', 'Anxa1'), ('sox6_str', 'Sox6-all'), ('calb1_str', 'Calb1')]

surv = pd.read_parquet(PRIVATE / "cohort_phenoconversion.parquet")
fam = pd.read_parquet(PRIVATE / "family_terr_sbr.parquet")
d = surv.drop(columns=['striatum', 'postput'], errors='ignore').merge(fam, on='PATNO')
d[E] = d[E].astype(int); d = d[d[T] > 0].reset_index(drop=True)
d['age_z'] = zs(d.age); d['sexf'] = d.sex.astype(float)
d['burden'] = -zs(d.striatum)


def km(sub):
    """Kaplan-Meier survival steps for one half."""
    s, out = 1.0, []
    for t in np.sort(sub[T].unique()):
        nr = int((sub[T] >= t).sum()); ne = int(((sub[T] == t) & (sub[E] == 1)).sum())
        if nr > 0 and ne > 0: s *= (1 - ne / nr)
        out.append((float(t), float(s)))
    return out


def risk_at(curve, target=3.0):
    prior = [c for c in curve if c[0] <= target]
    return 100 * (1 - (prior[-1][1] if prior else 1.0))


rows, kmrows = [], []
for k, lab in FAMK:
    dep = -zs(d[k])
    X = sm.add_constant(pd.DataFrame({'burden': d.burden, 'age_z': d.age_z, 'sexf': d.sexf}))
    x = d.copy(); x['z'] = zs(sm.OLS(dep, X).fit().resid)
    x['half'] = np.where(x.z > x.z.median(), 'high', 'low')
    m = cox_fit(x, ['z', 'age_z', 'sexf'], T, E)
    rec = dict(pred=k, label=lab, HR=float(np.exp(m.params[0])), p=float(m.pvalues[0]),
               HR_lo=float(np.exp(m.params[0] - 1.96 * m.bse[0])),
               HR_hi=float(np.exp(m.params[0] + 1.96 * m.bse[0])))
    for h in ['low', 'high']:
        sub = x[x.half == h]; cur = km(sub)
        rec[f'risk3_{h}'] = risk_at(cur); rec[f'n_{h}'] = int(len(sub)); rec[f'ev_{h}'] = int(sub[E].sum())
        for t, s in cur: kmrows.append(dict(pred=k, half=h, t=t, surv=s))
    rec['spread'] = rec['risk3_high'] - rec['risk3_low']
    rows.append(rec)

R = pd.DataFrame(rows)[['pred', 'label', 'HR', 'HR_lo', 'HR_hi', 'p', 'risk3_low', 'n_low', 'ev_low',
                        'risk3_high', 'n_high', 'ev_high', 'spread']]
KM = pd.DataFrame(kmrows)

old = os.path.join(FROZEN, "where_median.csv")
if os.path.exists(old):
    O = pd.read_csv(old).set_index('pred'); N = R.set_index('pred')
    bad = []
    for k, _ in FAMK:
        for c in ['HR', 'p', 'risk3_low', 'risk3_high', 'spread']:
            if not np.isclose(N.loc[k, c], O.loc[k, c], rtol=0, atol=1e-12):
                bad.append(f"{k}.{c}: {N.loc[k, c]!r} != {O.loc[k, c]!r}")
        for c in ['n_low', 'n_high', 'ev_low', 'ev_high']:
            if int(N.loc[k, c]) != int(O.loc[k, c]): bad.append(f"{k}.{c}: {N.loc[k,c]} != {O.loc[k,c]}")
    assert not bad, "where_median.csv drifted from the frozen values:\n  " + "\n  ".join(bad)
    print("  exactness guard: reproduces the frozen table bit-for-bit")

R.to_csv(os.path.join(FROZEN, "where_median.csv"), index=False)
KM.to_csv(os.path.join(FROZEN, "where_median_km.csv"), index=False)
pd.set_option('display.width', 200)
print(R.round(4).to_string(index=False))
print(f"\nwrote where_median.csv and where_median_km.csv ({len(KM)} KM steps)")
