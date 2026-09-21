#!/usr/bin/env python3
"""Write the pairwise subtype comparison of projection-weighted DaTscan depletion across PD baseline volumes in putamen, caudate and both together.

Run: PPMI_ROOT=/path/to/ppmi python compute/build_group_stats.py
"""
import warnings
from itertools import combinations

import nibabel as nib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nilearn import image
from scipy.stats import rankdata, wilcoxon

from datscan_registration import ppmi_root
from projection_fields import GROUP_ORDER, build_hmba_projection_frame

STRUCTURES = [("Putamen", "Pu"), ("Caudate", "Ca"), ("Overall", "Ov")]

PPMI = ppmi_root()
SPATIAL = PPMI / "derived" / "datscan_spatial"
VOLUMES = SPATIAL / "fpcit_mni"
FSL = PPMI / "derived" / "av133_spatial" / "mni152_fsl_2mm.nii.gz"


def rank_biserial(first, second):
    delta = np.asarray(first, float) - np.asarray(second, float)
    delta = delta[delta != 0]
    if delta.size == 0:
        return 0.0
    ranks = rankdata(np.abs(delta))
    positive, negative = ranks[delta > 0].sum(), ranks[delta < 0].sum()
    return float((positive - negative) / (positive + negative))


frame = build_hmba_projection_frame(PPMI)
reference = nib.load(FSL)


def to_fsl(values, interpolation):
    source = nib.Nifti1Image(np.nan_to_num(values).astype(np.float32), frame.affine)
    return image.resample_to_img(source, reference, interpolation=interpolation).get_fdata()


weights = {group: np.clip(to_fsl(frame.centered_fields[group], "linear"), 0, None) for group in GROUP_ORDER}
putamen = to_fsl(frame.putamen.astype(float), "nearest") > 0.5
caudate = to_fsl(frame.caudate.astype(float), "nearest") > 0.5
masks = {"Pu": putamen, "Ca": caudate, "Ov": putamen | caudate}
print("weights resampled", {group: int((weights[group] > 0).sum()) for group in GROUP_ORDER})

scans = pd.read_parquet(PPMI / "derived" / "ppmi_scan_datscan.parquet")
scans["PATNO"] = scans.PATNO.astype(int)
by_patno = scans.sort_values(["PATNO", "scan_date"]).drop_duplicates("PATNO").set_index("PATNO")
cohort = pd.read_csv(PPMI / "derived" / "datscan_recon_cohort.csv")
hc_ids = set(cohort[cohort.cohort == "Healthy Control"].PATNO.astype(int))
pd_ids = set(cohort[cohort.cohort == "Parkinson's Disease"].PATNO.astype(int))

hc_files = [path for path in sorted(VOLUMES.glob("*_suvr.nii.gz")) if int(path.name.split("_")[0]) in hc_ids]
hc_stack = np.clip(np.stack([nib.load(path).get_fdata() for path in hc_files]) - 1, 0, None)
hc_mean = hc_stack.mean(0)
hc_patnos = [int(path.name.split("_")[0]) for path in hc_files]
hc_age = np.array([by_patno.AGE_AT_VISIT.get(patno, np.nan) for patno in hc_patnos])
hc_sex = np.array([by_patno.SEX.get(patno, np.nan) for patno in hc_patnos])
usable = np.isfinite(hc_age) & np.isfinite(hc_sex)
design = np.c_[np.ones(usable.sum()), hc_age[usable], hc_sex[usable]]
beta = np.linalg.lstsq(design, hc_stack[usable].reshape(usable.sum(), -1), rcond=None)[0].reshape(3, *hc_mean.shape)
expected_hc = lambda age, sex: beta[0] + beta[1] * age + beta[2] * sex
print(f"HC reference n={len(hc_patnos)}; age+sex model on {usable.sum()}")


def weighted_mean(depletion, group, mask):
    weight = weights[group] * mask
    selected = (weight > 0) & np.isfinite(depletion)
    return float(np.sum(weight[selected] * depletion[selected]) / np.sum(weight[selected])) if selected.any() else np.nan


rows = []
pd_files = [path for path in sorted(VOLUMES.glob("*_suvr.nii.gz")) if int(path.name.split("_")[0]) in pd_ids]
for index, path in enumerate(pd_files):
    patno = int(path.name.split("_")[0])
    age = by_patno.AGE_AT_VISIT.get(patno, np.nan)
    sex = by_patno.SEX.get(patno, np.nan)
    try:
        suvr = nib.load(path).get_fdata()
    except Exception:
        continue
    sbr = np.clip(suvr - 1, 0, None)
    expected = expected_hc(age, sex) if np.isfinite(age) and np.isfinite(sex) else hc_mean
    depletion = np.where(expected > 0.15, (expected - sbr) / expected * 100, np.nan)
    record = {}
    for group in GROUP_ORDER:
        for prefix, mask in masks.items():
            record[f"{prefix}_{group}"] = weighted_mean(depletion, group, mask)
    rows.append(record)
    if (index + 1) % 200 == 0:
        print(f"    {index + 1}/{len(pd_files)}")
subjects = pd.DataFrame(rows).dropna()
print(f"PD baselines: {len(subjects)}")

results = []
for structure, prefix in STRUCTURES:
    for first, second in combinations(GROUP_ORDER, 2):
        left = subjects[f"{prefix}_{first}"].values
        right = subjects[f"{prefix}_{second}"].values
        try:
            _, pvalue = wilcoxon(left, right)
        except Exception:
            pvalue = np.nan
        results.append(dict(
            structure=structure,
            group1=first,
            group2=second,
            n=len(subjects),
            median1=round(np.median(left), 1),
            median2=round(np.median(right), 1),
            median_diff=round(np.median(left - right), 1),
            rbc=round(rank_biserial(left, right), 3),
            p=pvalue,
        ))
stats = pd.DataFrame(results)
for structure in stats.structure.unique():
    index = stats.index[stats.structure == structure]
    pvalues = stats.loc[index, "p"].values
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    for rank, position in enumerate(order):
        adjusted[position] = min(1.0, pvalues[position] * (len(pvalues) - rank))
    stats.loc[index, "p_holm"] = adjusted
stats["sig"] = stats["p_holm"].map(
    lambda value: "***" if value < 1e-3 else "**" if value < 1e-2 else "*" if value < 0.05 else "ns"
)
stats.to_csv(SPATIAL / "fig12_group_stats.csv", index=False)
print(f"fig12_group_stats.csv {stats.shape}")
