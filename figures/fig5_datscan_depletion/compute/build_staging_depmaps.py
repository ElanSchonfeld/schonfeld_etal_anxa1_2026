#!/usr/bin/env python3
"""Write the per-stage percent depletion voxel maps over the registered PD visit volumes, bilateral and aligned on the hemisphere contralateral to the clinical dominant side.

Run: PPMI_ROOT=/path/to/ppmi python compute/build_staging_depmaps.py
"""
import warnings

import nibabel as nib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from datscan_registration import hy_stage, ppmi_root, updrs_stage

MERGED_TIME_BINS = [
    (0, 1, "<1y"), (1, 1.5, "1-1.5y"), (1.5, 3, "1.5-3y"),
    (3, 5, "3-5y"), (5, 8, "5-8y"), (8, 40, "8y+"),
]


def merged_time_stage(years):
    for low, high, name in MERGED_TIME_BINS:
        if low <= years < high:
            return name
    return None


SCHEMES = {
    "time": (merged_time_stage, "yrs_since_onset", [name for _, _, name in MERGED_TIME_BINS]),
    "hy": (hy_stage, "NHY", ["H&Y 1", "H&Y 2", "H&Y 3", "H&Y 4"]),
    "updrs": (updrs_stage, "MDSUPDRS3_total", ["UPDRS 0-20", "UPDRS 20-40", "UPDRS 40-60", "UPDRS 60-80"]),
}

PPMI = ppmi_root()
SPATIAL = PPMI / "derived" / "datscan_spatial"
VOLUMES = SPATIAL / "fpcit_mni"
VISITS = SPATIAL / "fpcit_mni_visits"

scans = pd.read_parquet(PPMI / "derived" / "ppmi_scan_datscan.parquet")
scans["PATNO"] = scans.PATNO.astype(int)
scans["ym"] = pd.to_datetime(scans.scan_date).dt.strftime("%Y-%m")
ordered = scans.sort_values(["PATNO", "scan_date"])
by_visit = ordered.drop_duplicates(["PATNO", "ym"]).set_index(["PATNO", "ym"])
by_patno = ordered.drop_duplicates("PATNO").set_index("PATNO")
affected_side = {int(patno): {1: "R", 2: "L"}.get(value) for patno, value in by_patno.DOMSIDE.items()}

cohort = pd.read_csv(PPMI / "derived" / "datscan_recon_cohort.csv")
hc_ids = set(cohort[cohort.cohort == "Healthy Control"].PATNO.astype(int))
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

rows = []
for path in sorted(VISITS.glob("*_suvr.nii.gz")):
    stem = path.name[: -len("_suvr.nii.gz")]
    patno, date = int(stem.split("_")[0]), stem.split("_", 1)[1]
    try:
        record = by_visit.loc[(patno, date[:7])]
    except KeyError:
        continue
    rows.append({
        "path": path,
        "patno": patno,
        "age": record.AGE_AT_VISIT,
        "sex": record.SEX,
        "yrs_since_onset": record.yrs_since_onset,
        "NHY": record.NHY,
        "MDSUPDRS3_total": record.MDSUPDRS3_total,
        "side": affected_side.get(patno),
    })
visits = pd.DataFrame(rows)
print(f"PD visits {len(visits)}; with dominant side {int(visits.side.notna().sum())}")


def stage_maps(aligned):
    output = {}
    for key, (labeler, column, order) in SCHEMES.items():
        stage = visits[column].map(labeler)
        maps = []
        for name in order:
            group = visits[(stage == name) & (visits.side.notna() if aligned else True)]
            if not len(group):
                continue
            total = np.zeros(hc_mean.shape)
            count = 0
            ages, sexes = [], []
            for row in group.itertuples():
                sbr = np.clip(nib.load(row.path).get_fdata() - 1, 0, None)
                if aligned:
                    sbr = np.flip(sbr, axis=0) if row.side == "L" else sbr
                total += sbr
                count += 1
                ages.append(row.age)
                sexes.append(row.sex)
            if count < 1:
                continue
            mean_age, mean_sex = np.nanmean(ages), np.nanmean(sexes)
            expected = expected_hc(mean_age, mean_sex) if np.isfinite(mean_age) and np.isfinite(mean_sex) else hc_mean
            depletion = np.where(expected > 0.15, (expected - total / count) / expected * 100, np.nan).astype(np.float32)
            maps.append((f"{name}|n={count}", depletion))
            print(f"  [{key}] {name:14s} n={count}")
        output[f"deps_{key}"] = np.array(maps, dtype=object)
        output[f"stages_{key}"] = np.array([label.split("|")[0] for label, _ in maps])
    return output


np.savez_compressed(
    SPATIAL / "fig12_staging_all_depmaps.npz", hcm=hc_mean.astype(np.float32), **stage_maps(False)
)
print("fig12_staging_all_depmaps.npz")
np.savez_compressed(
    SPATIAL / "fig12_staging_all_depmaps_hemi.npz", hcm=hc_mean.astype(np.float32), **stage_maps(True)
)
print("fig12_staging_all_depmaps_hemi.npz")
