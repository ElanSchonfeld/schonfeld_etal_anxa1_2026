#!/usr/bin/env python
"""Assemble the two PROSPECTIVE analysis cohorts.

Run: python freeze_fig5_conversion_cohorts.py
"""
import os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from conversion_paths import PRIVATE

PPMI = os.environ.get("PPMI_ROOT")
if not PPMI:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
SI, DEM, DER = f"{PPMI}/study_info", f"{PPMI}/demographics", f"{PPMI}/derived"
DS = f"{DER}/datscan_spatial"
HERE = os.path.dirname(os.path.abspath(__file__)); FROZEN = os.path.join(HERE, "..", "frozen")
log = {}

anx = pd.read_parquet(f"{DS}/anxa1_putaminal_spatial.parquet")
anx = anx[anx.source == 'baseline'][['PATNO', 'anxa1put_L', 'anxa1put_R']]
rest = pd.read_parquet(f"{DS}/anxa_vs_nonanxa_putamen.parquet")
rest = rest[rest.source == 'baseline'][['PATNO', 'restput_L', 'restput_R']]
mat = pd.read_parquet(PRIVATE / "matched_putamen_sbr.parquet")

img = anx.merge(rest, on='PATNO', how='left').merge(mat, on='PATNO', how='left')
img['anxa1'] = (img.anxa1put_L + img.anxa1put_R) / 2
img['restput'] = (img.restput_L + img.restput_R) / 2
img['anxa1_worse'] = img[['anxa1put_L', 'anxa1put_R']].min(axis=1)
img['postput_worse'] = img[['postput_L', 'postput_R']].min(axis=1)
img['putamen_worse'] = img[['putamen_L', 'putamen_R']].min(axis=1)
log['n_baseline_scans'] = int(len(img))

rec = pd.read_csv(f"{DER}/datscan_recon_cohort.csv", low_memory=False)
rec['scan_dt'] = pd.to_datetime(rec.baseline_date, errors='coerce')
img = img.merge(rec[['PATNO', 'cohort', 'scan_dt']], on='PATNO', how='left')

ps = pd.read_csv(f"{DEM}/Participant_Status_13Jul2026.csv", low_memory=False)
img = img.merge(ps[['PATNO', 'COHORT_DEFINITION', 'ENROLL_STATUS', 'ENRLRBD', 'ENRLHPSM',
                    'ENRLLRRK2', 'ENRLGBA', 'ENRLSNCA', 'ENRLPRKN', 'ENRLPINK1']], on='PATNO', how='left')

dem = pd.read_csv(f"{DEM}/Demographics_13Jul2026.csv", low_memory=False).drop_duplicates('PATNO')
dem['birth'] = pd.to_datetime(dem.BIRTHDT, format='%m/%Y', errors='coerce')
img = img.merge(dem[['PATNO', 'birth', 'SEX']], on='PATNO', how='left')
img['age'] = (img.scan_dt - img.birth).dt.days / 365.25
img['sex'] = pd.to_numeric(img.SEX, errors='coerce')

xing = pd.read_csv(f"{SI}/Xing_Core_Lab_-_Quant_SBR_13Jul2026.csv", low_memory=False)
xing['dt'] = pd.to_datetime(xing.DATSCAN_DATE, format='%m/%Y', errors='coerce')
xing = xing.dropna(subset=['dt']).sort_values(['PATNO', 'dt'])
xk = {'PUTAMEN_REF_CWM': 'xing_put', 'POSCOMMISSURAL_PUTAMEN_REF_CWM': 'xing_postput',
      'STRIATUM_REF_CWM': 'xing_striatum', 'CAUDATE_REF_CWM': 'xing_caudate'}


def nearest_xing(patno, scan_dt):
    """The Xing row whose DATSCAN_DATE is closest to this subject's baseline scan (month resolution)."""
    g = xing[xing.PATNO == patno]
    if not len(g) or pd.isna(scan_dt): return {v: np.nan for v in xk.values()}
    i = (g.dt - scan_dt).abs().idxmin()
    return {v: pd.to_numeric(xing.at[i, k], errors='coerce') for k, v in xk.items()}


xrows = [nearest_xing(p, d) for p, d in zip(img.PATNO, img.scan_dt)]
img = pd.concat([img.reset_index(drop=True), pd.DataFrame(xrows)], axis=1)

prd = pd.read_csv(f"{SI}/Primary_Research_Diagnosis_13Jul2026.csv", low_memory=False)
prd['dt'] = pd.to_datetime(prd.INFODT, format='%m/%Y', errors='coerce')
prd = prd.dropna(subset=['dt', 'PRIMDIAG']).sort_values(['PATNO', 'dt'])
prd['PRIMDIAG'] = pd.to_numeric(prd.PRIMDIAG, errors='coerce')

PD_CODES = [1]
LBD_CODES = [1, 5]


def dx_summary(patno, scan_dt, codes):
    """(prevalent_at_baseline, event, time_years) for one subject and one endpoint definition."""
    g = prd[prd.PATNO == patno]
    if not len(g) or pd.isna(scan_dt):
        return dict(prevalent=np.nan, event=np.nan, tyears=np.nan, last_dx=pd.NaT)
    scan_m = scan_dt.to_period('M')
    hit = g[g.PRIMDIAG.isin(codes)]
    prevalent = bool(len(hit) and (hit.dt.dt.to_period('M') <= scan_m).any())
    after = hit[hit.dt.dt.to_period('M') > scan_m]
    last = g.dt.max()
    if prevalent:
        return dict(prevalent=True, event=np.nan, tyears=np.nan, last_dx=last)
    if len(after):
        return dict(prevalent=False, event=1, tyears=(after.dt.min() - scan_dt).days / 365.25, last_dx=last)
    return dict(prevalent=False, event=0, tyears=(last - scan_dt).days / 365.25, last_dx=last)


prod = img[img.COHORT_DEFINITION == 'Prodromal'].copy()
log['n_prodromal_with_scan'] = int(len(prod))

for tag, codes in [('pd', PD_CODES), ('lbd', LBD_CODES)]:
    s = pd.DataFrame([dx_summary(p, d, codes) for p, d in zip(prod.PATNO, prod.scan_dt)], index=prod.index)
    prod[f'prevalent_{tag}'] = s.prevalent; prod[f'event_{tag}'] = s.event; prod[f'tyears_{tag}'] = s.tyears
prod['last_dx'] = [dx_summary(p, d, PD_CODES)['last_dx'] for p, d in zip(prod.PATNO, prod.scan_dt)]

log['n_prodromal_no_dx_record'] = int(prod.event_pd.isna().sum() - (prod.prevalent_pd == True).sum())
log['n_prodromal_prevalent_pd'] = int((prod.prevalent_pd == True).sum())

surv = prod[(prod.prevalent_pd == False) & prod.tyears_pd.notna() & (prod.tyears_pd > 0)].copy()
need = ['anxa1', 'restput', 'putamen', 'postput', 'anxa1flat', 'xing_put', 'xing_postput', 'age', 'sex']
log['n_before_predictor_completeness'] = int(len(surv))
surv = surv.dropna(subset=need)
log['n_cohortA_final'] = int(len(surv))
log['n_events_pd'] = int(surv.event_pd.sum())
log['n_events_lbd'] = int(surv.event_lbd.fillna(0).sum())
log['followup_years_median'] = float(surv.tyears_pd.median())
log['followup_years_total'] = float(surv.tyears_pd.sum())
surv.to_parquet(PRIVATE / "cohort_phenoconversion.parquet", index=False)

u = pd.read_csv(f"{SI}/MDS-UPDRS_Part_III_13Jul2026.csv", low_memory=False)
u['dt'] = pd.to_datetime(u.INFODT, format='%m/%Y', errors='coerce')
u['NP3TOT'] = pd.to_numeric(u.NP3TOT, errors='coerce')
u = u.dropna(subset=['dt', 'NP3TOT'])
u = u[u.PDSTATE.isna() | (u.PDSTATE == 'OFF')]
u = u.merge(img[['PATNO', 'scan_dt', 'COHORT_DEFINITION']], on='PATNO', how='inner')
u['yrs'] = (u.dt - u.scan_dt).dt.days / 365.25
u = u[u.yrs >= -0.5]
u = u.sort_values(['PATNO', 'dt']).drop_duplicates(['PATNO', 'dt'])
u[['PATNO', 'COHORT_DEFINITION', 'dt', 'yrs', 'NP3TOT']].to_parquet(PRIVATE / "updrs_visits.parquet", index=False)


def slope(g):
    if len(g) < 3: return np.nan
    if g.yrs.max() - g.yrs.min() < 2: return np.nan
    return float(np.polyfit(g.yrs, g.NP3TOT, 1)[0])


sl = u.groupby('PATNO').apply(lambda g: pd.Series(dict(
    slope=slope(g), n_visits=len(g), span=g.yrs.max() - g.yrs.min(), np3_base=g.NP3TOT.iloc[0])))
sl = sl.dropna(subset=['slope']).reset_index()
prog = img.merge(sl, on='PATNO', how='inner').dropna(subset=need)
log['n_cohortB_total'] = int(len(prog))
log['n_cohortB_by_cohort'] = prog.COHORT_DEFINITION.value_counts().to_dict()
prog.to_parquet(PRIVATE / "cohort_progression.parquet", index=False)

json.dump(log, open(f"{FROZEN}/cohort_build_log.json", "w"), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))
