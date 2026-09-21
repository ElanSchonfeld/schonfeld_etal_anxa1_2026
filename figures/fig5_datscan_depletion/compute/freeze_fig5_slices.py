#!/usr/bin/env python3
"""Freeze the small 2D arrays drawn by the three DaTscan montage panels.

Run: PPMI_ROOT=/path/to/ppmi python compute/freeze_fig5_slices.py
"""
import glob
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from nilearn import image
from scipy.ndimage import binary_fill_holes, gaussian_filter

from datscan_registration import TIME_ORDER, select_visits, visit_candidates

_ppmi_root = os.environ.get("PPMI_ROOT")
if not _ppmi_root:
    raise RuntimeError("Set PPMI_ROOT to the local PPMI data directory")
PPMI = Path(_ppmi_root).expanduser().resolve()
OUT = PPMI / "derived" / "datscan_spatial"
VOL = OUT / "fpcit_mni"
LVOL = OUT / "fpcit_mni_visits"
FSL = PPMI / "derived" / "av133_spatial" / "mni152_fsl_2mm.nii.gz"
T1_1MM = PPMI / "derived" / "av133_spatial" / "mni152_fsl_1mm.nii.gz"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"
FROZEN.mkdir(exist_ok=True)

aff = nib.load(FSL).affine
osgk = image.resample_to_img(OUT / "osgk_striatum.nii.gz", FSL, interpolation='nearest').get_fdata().astype(int)
STR2 = osgk > 0
_absx = lambda m: np.abs(aff[0,0]*np.argwhere(m)[:,0]+aff[0,3]).mean()
labs = {l:_absx(osgk==l) for l in (1,2,3) if (osgk==l).any()}
CAUD = osgk == min(labs, key=labs.get); PUT = osgk == max(labs, key=labs.get)
POST = PUT.copy(); ANT = PUT.copy(); ym = np.median([aff[1,1]*j+aff[1,3] for j in np.argwhere(PUT)[:,1]])
for j in range(osgk.shape[1]):
    if aff[1,1]*j+aff[1,3] >= ym: POST[:, j, :] = False
    else: ANT[:, j, :] = False

def build_masks(fsl_path, t1_path, labels):
    aff2 = nib.load(fsl_path).affine
    ref1 = nib.load(t1_path)
    aff1 = ref1.affine
    mni1 = ref1.get_fdata()
    str1 = image.resample_to_img(
        nib.Nifti1Image((labels > 0).astype(np.uint8), aff2),
        ref1,
        interpolation="nearest",
    ).get_fdata() > 0.5
    xa = np.argwhere(str1.any((1, 2)))[:, 0]
    ya = np.argwhere(str1.any((0, 2)))[:, 0]
    za = np.argwhere(str1.any((0, 1)))[:, 0]
    xc, yc, zc = (xa.min() + xa.max()) // 2, (ya.min() + ya.max()) // 2, (za.min() + za.max()) // 2
    xhw = (xa.max() - xa.min()) // 2 + 12
    half_height = max(ya.max() - ya.min(), za.max() - za.min()) // 2 + 10

    def clip(start, stop, size):
        start, stop = int(start), int(stop)
        if start < 0:
            return 0, stop - start
        if stop > size:
            return start - (stop - size), size
        return start, stop

    xcrop = clip(xc - xhw, xc + xhw, mni1.shape[0])
    ycrop = clip(yc - half_height, yc + half_height, mni1.shape[1])
    zcrop = clip(zc - half_height, zc + half_height, mni1.shape[2])
    voxel_z = lambda mm: int(round((mm - aff1[2, 3]) / aff1[2, 2]))
    voxel_y = lambda mm: int(round((mm - aff1[1, 3]) / aff1[1, 1]))
    zarea = np.array([str1[:, :, z].sum() for z in range(str1.shape[2])])
    yarea = np.array([str1[:, y, :].sum() for y in range(str1.shape[1])])
    zsub = np.where(zarea > 0.18 * zarea.max())[0]
    ysub = np.where(yarea > 0.18 * yarea.max())[0]
    zlo, zhi = aff1[2, 2] * zsub.min() + aff1[2, 3], aff1[2, 2] * zsub.max() + aff1[2, 3]
    ylo, yhi = aff1[1, 1] * ysub.min() + aff1[1, 3], aff1[1, 1] * ysub.max() + aff1[1, 3]
    required = [-4, 0, 4, 8]
    zmm = sorted(set(required) | set(np.round(np.linspace(zlo, zhi, 3)).astype(int).tolist()))
    ymm = sorted(set(required) | set(np.round(np.linspace(ylo, yhi, 3)).astype(int).tolist()))
    count = max(len(zmm), len(ymm))
    while len(zmm) < count:
        zmm.append(zmm[-1] + 4)
    while len(ymm) < count:
        ymm.append(ymm[-1] + 4)
    return {
        "mni": mni1,
        "aff": aff1,
        "aff2": aff2,
        "ref1": ref1,
        "STR1": str1,
        "X": xcrop,
        "Y": ycrop,
        "Zc": zcrop,
        "zs": [voxel_z(mm) for mm in zmm],
        "ys": [voxel_y(mm) for mm in ymm],
        "zmm": zmm,
        "ymm": ymm,
    }


def canvas(reference, brain):
    brain = binary_fill_holes(brain)
    rgb = np.ones((*reference.shape, 3))
    positive = reference[brain & (reference > 0)]
    if len(positive):
        low, high = np.percentile(positive, (1, 99))
        normalized = np.clip((reference - low) / max(high - low, 1e-9), 0, 1)
    else:
        normalized = np.zeros_like(reference)
    gray = 0.15 + 0.78 * normalized
    rgb[brain] = np.stack([gray[brain]] * 3, -1)
    rgb[brain & (normalized < 0.18)] = [0.06, 0.06, 0.06]
    return rgb


crop = build_masks(FSL, T1_1MM, osgk)


def build_frozen(res, title, out_name, colorbar_label='% depletion (PD vs HC)', pct=(5, 95)):
    """Replicate the locked renderer's construction and freeze only the drawn 2D arrays."""
    mni1, aff2, ref1, STR1 = crop['mni'], crop['aff2'], crop['ref1'], crop['STR1']
    (X0, X1), (Y0, Y1), (Z0, Z1) = crop['X'], crop['Y'], crop['Zc']
    zs, ys, zmm, ymm = crop['zs'], crop['ys'], crop['zmm'], crop['ymm']
    brain = mni1 > (0.10 * mni1.max())
    NC = len(zs); ns = len(res)
    pool = np.concatenate([d['dep'][STR2][~np.isnan(d['dep'][STR2])] for d in res])
    vmin, vmax = float(np.percentile(pool, pct[0])), float(np.percentile(pool, pct[1]))
    STR1s = gaussian_filter(STR1.astype(np.float32), 1.6)

    def to1(dep2):
        d = np.nan_to_num(dep2, nan=-1e4).astype(np.float32)
        r = image.resample_to_img(nib.Nifti1Image(d, aff2), ref1, interpolation='nearest').get_fdata()
        return np.where(r > -9e3, r, np.nan)

    bg_ax  = np.stack([np.rot90(canvas(mni1[X0:X1, Y0:Y1, z], brain[X0:X1, Y0:Y1, z])) for z in zs]).astype(np.float32)
    sm_ax  = np.stack([np.rot90(STR1s[X0:X1, Y0:Y1, z]) for z in zs]).astype(np.float32)
    bg_cor = np.stack([np.rot90(canvas(mni1[X0:X1, y, Z0:Z1], brain[X0:X1, y, Z0:Z1])) for y in ys]).astype(np.float32)
    sm_cor = np.stack([np.rot90(STR1s[X0:X1, y, Z0:Z1]) for y in ys]).astype(np.float32)

    ov_ax = np.full((ns, NC) + bg_ax.shape[1:3], np.nan, np.float32)
    ov_cor = np.full((ns, NC) + bg_cor.shape[1:3], np.nan, np.float32)
    for r, d in enumerate(res):
        dep1 = to1(d['dep'])
        for c, z in enumerate(zs):
            ov_ax[r, c] = np.rot90(np.where(STR1[X0:X1, Y0:Y1, z], dep1[X0:X1, Y0:Y1, z], np.nan))
        for c, y in enumerate(ys):
            ov_cor[r, c] = np.rot90(np.where(STR1[X0:X1, y, Z0:Z1], dep1[X0:X1, y, Z0:Z1], np.nan))

    np.savez_compressed(
        FROZEN / out_name,
        bg_ax=bg_ax, sm_ax=sm_ax, bg_cor=bg_cor, sm_cor=sm_cor, ov_ax=ov_ax, ov_cor=ov_cor,
        zmm=np.array(zmm, int), ymm=np.array(ymm, int),
        stages=np.array([str(d['stage']) for d in res]),
        ns_labels=np.array([str(d['n']) for d in res]),
        colors=np.array([str(d.get('color', 'black')) for d in res]),
        vmin=np.float32(vmin), vmax=np.float32(vmax), pct=np.array(pct, int),
        title=np.array(title), colorbar_label=np.array(colorbar_label),
    )
    sz = (FROZEN / out_name).stat().st_size / 1024
    print(f"  wrote {out_name}  ({sz:.0f} KB)  ns={ns} NC={NC} vmin={vmin:.1f} vmax={vmax:.1f} bgshape={bg_ax.shape[1:]}")


coh = pd.read_csv(PPMI / "derived" / "datscan_recon_cohort.csv"); coh['PATNO'] = coh.PATNO.astype(int)
sd = pd.read_parquet(PPMI / "derived" / "ppmi_scan_datscan.parquet"); sd['PATNO'] = sd.PATNO.astype(int)
meta = sd.sort_values(['PATNO', 'scan_date']).drop_duplicates('PATNO').set_index('PATNO')
HCset = set(coh[coh.cohort == 'Healthy Control'].PATNO)
HCids = [int(os.path.basename(f).split('_')[0]) for f in glob.glob(str(VOL / "*_suvr.nii.gz"))
         if int(os.path.basename(f).split('_')[0]) in HCset]
HCstack = np.clip(np.stack([nib.load(VOL / f"{p}_suvr.nii.gz").get_fdata() for p in HCids]) - 1, 0, None)
hcm = HCstack.mean(0)
hc_age = np.array([meta.AGE_AT_VISIT.get(p, np.nan) for p in HCids])
hc_sex = np.array([meta.SEX.get(p, np.nan) for p in HCids])
_ok = np.isfinite(hc_age) & np.isfinite(hc_sex)
_X = np.c_[np.ones(_ok.sum()), hc_age[_ok], hc_sex[_ok]]
_beta = np.linalg.lstsq(_X, HCstack[_ok].reshape(_ok.sum(), -1), rcond=None)[0].reshape(3, *hcm.shape)
expected_hc = lambda age, sex: _beta[0] + _beta[1] * age + _beta[2] * sex
print(f"HC reference n={len(HCids)}; age+sex model on {_ok.sum()}")


print("\n[final_depletion]")
dep = nib.load(OUT / "datscan_pct_depletion.nii.gz").get_fdata()
dep = np.where(STR2, dep, np.nan)
PDall = set(coh[coh.cohort == "Parkinson's Disease"].PATNO)
n_pd = sum(1 for f in glob.glob(str(VOL / "*_suvr.nii.gz"))
           if int(os.path.basename(f).split('_')[0]) in PDall)
print(f"  n = {len(HCids)} HC / {n_pd} PD (baseline registered volumes)")
res_final = [dict(stage="PD vs HC", n=f"{len(HCids)}/{n_pd}", dep=dep)]
build_frozen(res_final, "DaTscan PD-vs-HC dopamine depletion (age+sex adj, FP-CIT/MNI152)",
             "slices_final_depletion.npz")


print("\n[genetics_spatial]")
PDset = set(coh[(coh.cohort == "Parkinson's Disease") & coh.in_xing_sbr].PATNO)
pdids = [int(os.path.basename(f).split('_')[0]) for f in glob.glob(str(VOL / "*_suvr.nii.gz"))
         if int(os.path.basename(f).split('_')[0]) in PDset]

def carrier(p, col):
    v = meta[col].get(p, 0); return bool(v) if pd.notna(v) else False

pdids = [p for p in pdids if not (carrier(p, 'LRRK2_carrier') and carrier(p, 'GBA_carrier'))]
groups = [('idiopathic PD', lambda p: not carrier(p, 'LRRK2_carrier') and not carrier(p, 'GBA_carrier'), 'black'),
          ('LRRK2 PD',      lambda p: carrier(p, 'LRRK2_carrier'), '#3388cc'),
          ('GBA PD',        lambda p: carrier(p, 'GBA_carrier'),   '#33aa55')]
res_gen = []
for name, sel, color in groups:
    ids = [p for p in pdids if sel(p)]
    vols = [np.clip(nib.load(VOL / f"{p}_suvr.nii.gz").get_fdata() - 1, 0, None) for p in ids]
    if len(vols) < 5: continue
    ma = np.nanmean([meta.AGE_AT_VISIT.get(p, np.nan) for p in ids])
    ms = np.nanmean([meta.SEX.get(p, np.nan) for p in ids])
    exp = expected_hc(ma, ms) if np.isfinite(ma) else hcm
    d = np.where(exp > 0.15, (exp - np.mean(vols, 0)) / exp * 100, np.nan)
    res_gen.append(dict(stage=name, n=len(vols), dep=d, color=color))
    print(f"  {name} (n={len(vols)}, age {ma:.1f})")
build_frozen(res_gen, "DaTscan final depletion by genotype (idiopathic / LRRK2 / GBA, age+sex adj, FP-CIT/MNI)",
             "slices_genetics_spatial.npz")


print("\n[staging_time]")
YRS_ORDER = list(TIME_ORDER)
chosen = select_visits(visit_candidates(PPMI))
chosen = chosen.assign(path=[LVOL / f"{row.PATNO}_{row.date}_suvr.nii.gz" for row in chosen.itertuples()])
proc_df = chosen[[path.is_file() for path in chosen.path]]
print(f"  selected {len(chosen)} visits; {len(proc_df)} registered volumes available")

res_stg = []
for name in YRS_ORDER:
    group = proc_df[proc_df.yrs_stage == name].sort_values(['PATNO', 'date'])
    volumes, ages, sexes = [], [], []
    for row in group.itertuples():
        volumes.append(np.clip(nib.load(row.path).get_fdata() - 1, 0, None))
        ages.append(row.age)
        sexes.append(row.sex)
    if not volumes:
        continue
    mean_age, mean_sex = np.nanmean(ages), np.nanmean(sexes)
    expected = expected_hc(mean_age, mean_sex) if np.isfinite(mean_age) and np.isfinite(mean_sex) else hcm
    depletion = np.where(expected > 0.15, (expected - np.mean(volumes, axis=0)) / expected * 100, np.nan)
    res_stg.append({'stage': name, 'n': len(volumes), 'dep': depletion})
for r in res_stg: print(f"  {r['stage']:8s} n={r['n']:3d}")
build_frozen(res_stg, "DaTscan depletion by TIME since onset (longitudinal, age+sex adj, FP-CIT/MNI)",
             "slices_staging_time.npz")
print("\nDONE")
