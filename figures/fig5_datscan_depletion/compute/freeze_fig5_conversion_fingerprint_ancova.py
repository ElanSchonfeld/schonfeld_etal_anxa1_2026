#!/usr/bin/env python
"""Whole-striatum family coefficients for the 3-year change in 22 clinical assessments.

Run: python freeze_fig5_conversion_fingerprint_ancova.py
"""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import updrs_util as uu
from math import ceil
from conversion_paths import PRIVATE


def need(cols, frac=0.75):
    """Minimum items required to score a questionnaire sum: 75% of items, rounded up."""
    return int(ceil(frac * len(cols)))

PPMI = os.environ.get("PPMI_ROOT")
if not PPMI:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
SI = f"{PPMI}/study_info"
HERE = os.path.dirname(os.path.abspath(__file__)); FROZEN = os.path.join(HERE, "..", "frozen")
zs = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
load = lambda f: pd.read_csv(f"{SI}/{f}", low_memory=False)
num = lambda d, c: pd.to_numeric(d[c], errors='coerce')
def prep(d):
    d = d.copy(); d['dt'] = pd.to_datetime(d.INFODT, format='%m/%Y', errors='coerce')
    return d.dropna(subset=['dt'])

u3 = prep(load("MDS-UPDRS_Part_III_13Jul2026.csv")); u3 = u3[u3.PDSTATE.isna() | (u3.PDSTATE == 'OFF')]
u3['brady_rig'] = uu.item_score(u3, uu.BRADY, 1.0)
u3['rest_tremor'] = uu.item_score(u3, uu.RTREM, 1.0)
u3['action_tremor'] = uu.item_score(u3, uu.ATREM, 1.0)
PIGD3 = ['NP3GAIT', 'NP3FRZGT', 'NP3PSTBL']
u2 = prep(load("MDS_UPDRS_Part_II__Patient_Questionnaire_13Jul2026.csv")); u2['v'] = num(u2, 'NP2PTOT')
for c in ['NP2WALK', 'NP2FREZ']: u2[c] = uu.sanitize_items(u2, [c])[c]
p3 = u3[['PATNO', 'EVENT_ID', 'dt'] + PIGD3].copy(); p3[PIGD3] = uu.sanitize_items(p3, PIGD3)
u23 = u2[['PATNO', 'EVENT_ID', 'NP2WALK', 'NP2FREZ']].drop_duplicates(['PATNO', 'EVENT_ID']) \
        .merge(p3.drop_duplicates(['PATNO', 'EVENT_ID']), on=['PATNO', 'EVENT_ID'])
u23['pigd'] = u23[['NP2WALK', 'NP2FREZ'] + PIGD3].mean(axis=1, skipna=False)
u1 = prep(load("MDS-UPDRS_Part_I_13Jul2026.csv")); u1['v'] = num(u1, 'NP1RTOT')
u1p = prep(load("MDS-UPDRS_Part_I_Patient_Questionnaire_13Jul2026.csv"))
IB = ['NP1SLPN', 'NP1SLPD', 'NP1PAIN', 'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1FATG']
u1p[IB] = uu.sanitize_items(u1p, IB)
u1p['ib6'] = u1p[[c for c in IB if c != 'NP1PAIN']].sum(axis=1, min_count=6)
u1p['pain'] = u1p['NP1PAIN']
adl = prep(load("Modified_Schwab___England_Activities_of_Daily_Living_13Jul2026.csv")); adl['v'] = num(adl, 'MSEADLG')
moca = prep(load("Montreal_Cognitive_Assessment__MoCA__13Jul2026.csv")); moca['v'] = num(moca, 'MCATOT')
ben = prep(load("Benton_Judgement_of_Line_Orientation_13Jul2026.csv")); ben['v'] = num(ben, 'JLO_TOTRAW')
hvlt = prep(load("Hopkins_Verbal_Learning_Test_-_Revised_13Jul2026.csv")); hvlt['v'] = num(hvlt, 'DVT_TOTAL_RECALL')
lns = prep(load("Letter_-_Number_Sequencing_13Jul2026.csv")); lns['v'] = num(lns, 'LNS_TOTRAW')
sdmt = prep(load("Symbol_Digit_Modalities_Test_13Jul2026.csv")); sdmt['v'] = num(sdmt, 'SDMTOTAL')
sft = prep(load("Modified_Semantic_Fluency_13Jul2026.csv")); sft['v'] = num(sft, 'DVT_SFTANIM')
scp = prep(load("SCOPA-AUT_13Jul2026.csv"))
SCA = [f"SCAU{i}" for i in range(1, 22)]
for c in SCA: scp[c] = num(scp, c).replace(9, np.nan)
AUT = [f"SCAU{i}" for i in range(4, 22)]
scp['v'] = scp[AUT].sum(axis=1, min_count=need(AUT))
scp['bulbar'] = scp[['SCAU1', 'SCAU2', 'SCAU3']].sum(axis=1, min_count=3)
ess = prep(load("Epworth_Sleepiness_Scale_13Jul2026.csv"))
E = [f"ESS{i}" for i in range(1, 9)]
for c in E: ess[c] = num(ess, c)
ess['v'] = ess[E].sum(axis=1, min_count=need(E))
gds = prep(load("Geriatric_Depression_Scale__Short_Version__13Jul2026.csv"))
GREV = ['GDSSATIS', 'GDSGSPIR', 'GDSHAPPY', 'GDSALIVE', 'GDSENRGY']
GFWD = ['GDSDROPD', 'GDSEMPTY', 'GDSBORED', 'GDSAFRAD', 'GDSHLPLS', 'GDSHOME', 'GDSMEMRY',
        'GDSWRTLS', 'GDSHOPLS', 'GDSBETER']
for c in GREV + GFWD: gds[c] = num(gds, c)
gds['v'] = gds[GFWD].sum(axis=1, min_count=need(GFWD)) + (1 - gds[GREV]).sum(axis=1, min_count=need(GREV))
stai = prep(load("State-Trait_Anxiety_Inventory_for_Adults_13Jul2026.csv"))
SREV = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30, 33, 34, 36, 39]
STAI_COLS = [f"STAIAD{i}" for i in range(1, 41)]
for i in range(1, 41):
    c = f"STAIAD{i}"; x = num(stai, c); stai[c] = (5 - x) if i in SREV else x
stai['v'] = stai[STAI_COLS].sum(axis=1, min_count=need(STAI_COLS))
quip = prep(load("QUIP-Current-Short_13Jul2026.csv"))
Q = ['TMGAMBLE', 'CNTRLGMB', 'TMSEX', 'CNTRLSEX', 'TMBUY', 'CNTRLBUY', 'TMEAT', 'CNTRLEAT',
     'TMTORACT', 'TMTMTACT', 'TMTRWD', 'TMDISMED', 'CNTRLDSM']
for c in Q: quip[c] = num(quip, c)
quip['v'] = quip[Q].sum(axis=1, min_count=need(Q))
rbd = prep(load("REM_Sleep_Behavior_Disorder_Screening_Questionnaire_13Jul2026.csv"))
Rq = ['DRMVIVID', 'DRMAGRAC', 'DRMNOCTB', 'SLPLMBMV', 'SLPINJUR', 'DRMVERBL', 'DRMFIGHT',
      'DRMUMV', 'DRMOBJFL', 'MVAWAKEN', 'DRMREMEM', 'SLPDSTRB']
for c in Rq: rbd[c] = num(rbd, c)
rbd['v'] = rbd[Rq].sum(axis=1, min_count=need(Rq))

MOT, NM, COG, NP = 'motor', 'non-motor', 'cognition', 'neuropsychiatric'
DOM = [(u3, 'brady_rig', 'bradykinesia-rigidity', MOT, True),
       (u3, 'rest_tremor', 'rest tremor', MOT, True),
       (u3, 'action_tremor', 'action/postural tremor', MOT, True),
       (u23, 'pigd', 'PIGD (Stebbins)', MOT, True),
       (u2, 'v', 'motor daily living (UPDRS-II)', MOT, True),
       (adl, 'v', 'Schwab & England ADL', MOT, False),
       (u1p, 'ib6', 'non-motor daily living (UPDRS-IB, 6 item)', NM, True),
       (u1p, 'pain', 'Pain (UPDRS 1.9)', NM, True),
       (scp, 'v', 'autonomic symptoms (SCOPA-AUT 4-21)', NM, True),
       (scp, 'bulbar', 'bulbar: swallow/drool (SCOPA 1-3)', NM, True),
       (ess, 'v', 'daytime sleepiness (ESS)', NM, True),
       (rbd, 'v', 'RBD symptoms (RBDSQ)', NM, True),
       (u1, 'v', 'cognitive / behavioral (UPDRS-IA)', NP, True),
       (gds, 'v', 'depression (GDS-15)', NP, True),
       (stai, 'v', 'anxiety (STAI)', NP, True),
       (quip, 'v', 'impulsive-compulsive (QUIP)', NP, True),
       (moca, 'v', 'global cognition (MoCA)', COG, False),
       (ben, 'v', 'visuospatial (Benton JLO)', COG, False),
       (hvlt, 'v', 'verbal learning (HVLT-R)', COG, False),
       (lns, 'v', 'working memory (LNS)', COG, False),
       (sdmt, 'v', 'processing speed (SDMT)', COG, False),
       (sft, 'v', 'semantic fluency', COG, False)]

def nearest_level(tab, col, sdt):
    t = tab.dropna(subset=[col]).copy()
    t['distance'] = (t.dt - t.PATNO.map(sdt)).dt.days.abs() / 365.25
    t['not_baseline'] = t.EVENT_ID.ne('BL')
    srt = ['PATNO', 'distance', 'not_baseline', 'dt'] + (['REC_ID'] if 'REC_ID' in t else [])
    t = t[t.distance <= 0.5].sort_values(srt)
    return t.drop_duplicates('PATNO').set_index('PATNO')[col]

def followup(tab, col, sdt, lo=2.0, hi=5.0, target=3.0):
    t = tab.dropna(subset=[col]).copy(); t['yr'] = (t.dt - t.PATNO.map(sdt)).dt.days / 365.25
    t = t[(t.yr >= lo) & (t.yr <= hi)].copy(); t['d'] = (t.yr - target).abs()
    g = t.sort_values(['PATNO', 'd']).drop_duplicates('PATNO').set_index('PATNO')
    return g[col], g['yr']

T = pd.read_parquet(PRIVATE / "family_terr_sbr.parquet")
prog = pd.read_parquet(PRIVATE / "cohort_progression.parquet")
surv = pd.read_parquet(PRIVATE / "cohort_phenoconversion.parquet")
TERR = [('anxa1_str', 'Anxa1-assoc.'), ('sox6_str', 'Sox6-all'), ('sox6r_str', 'Sox6 non-Anxa1'), ('calb1_str', 'Calb1')]
rows = []
for cname, base in [("prodromal", surv), ("PD", prog[prog.COHORT_DEFINITION == "Parkinson's Disease"])]:
    d = base[['PATNO', 'scan_dt', 'age', 'sex', 'xing_striatum']].merge(T, on='PATNO')
    d['age_z'] = zs(d.age); d['sexf'] = d.sex.astype(float); d['glob'] = -zs(d.xing_striatum)
    for k, _ in TERR: d['dep_' + k] = -zs(d[k])
    sdt = d.set_index('PATNO').scan_dt
    for tab, col, lab, grp, hiw in DOM:
        tb = tab[tab.PATNO.isin(set(d.PATNO))]
        b = nearest_level(tb, col, sdt); fu, yr = followup(tb, col, sdt)
        g = d.copy(); g['b0'] = g.PATNO.map(b)
        g['y'] = g.PATNO.map(fu); g['yr'] = g.PATNO.map(yr)
        g = g.dropna(subset=['y', 'b0', 'yr'])
        if len(g) < 100: continue
        g['yz'] = zs((1.0 if hiw else -1.0) * (g.y - g.b0)); g['b0z'] = zs(g.b0)
        try: qb = pd.qcut(g.b0, 5, labels=False, duplicates='drop')
        except Exception: qb = pd.factorize(g.b0)[0]
        Dm = pd.get_dummies(pd.Series(qb, index=g.index), prefix='b', drop_first=True).astype(float)
        Z = pd.concat([g[['glob', 'b0z', 'age_z', 'sexf', 'yr']], Dm], axis=1)
        for k, klab in TERR:
            m = sm.OLS(g.yz, sm.add_constant(pd.concat([g[['dep_' + k]], Z], axis=1))).fit()
            rows.append(dict(cohort=cname, territory=k, label=klab, domain=lab, group=grp,
                             outcome='ancova_3y', n=len(g), beta=float(m.params['dep_' + k]),
                             se=float(m.bse['dep_' + k]), p=float(m.pvalues['dep_' + k])))
R = pd.DataFrame(rows); R['q'] = np.nan
for (c_, t_), idx in R.groupby(['cohort', 'territory']).groups.items():
    R.loc[idx, 'q'] = multipletests(R.loc[idx, 'p'], method='fdr_bh')[1]
R.to_csv(f"{FROZEN}/family_fingerprint_striatum_ancova.csv", index=False)
pd.set_option('display.width', 200)
A = R[R.territory.isin(['anxa1_str', 'calb1_str'])].pivot_table(index='domain', columns=['cohort', 'territory'], values='beta')
C = pd.DataFrame({c: A[(c, 'anxa1_str')] - A[(c, 'calb1_str')] for c in ['prodromal', 'PD']})
print(C.round(3).sort_values('prodromal', ascending=False).to_string())
print(f"{len(C)} assessments; across-domain r = {C.prodromal.corr(C.PD):.3f}; wrote family_fingerprint_striatum_ancova.csv")
