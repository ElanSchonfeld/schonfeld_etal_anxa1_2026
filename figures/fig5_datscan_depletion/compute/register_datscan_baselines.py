#!/usr/bin/env python3
"""Register each baseline DaTscan reconstruction to the FP-CIT/MNI grid as a CWM-normalized SUVR volume and write the PD-vs-HC percent depletion map.

Run: PPMI_ROOT=/path/to/ppmi python compute/register_datscan_baselines.py
"""
import warnings

import ants
import nibabel as nib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from datscan_registration import cwm_reference, dcm2niix, normalize, ppmi_root, recon_members, striatal_vois

COHORTS = ("Healthy Control", "Parkinson's Disease", "Prodromal")

PPMI = ppmi_root()
SPATIAL = PPMI / "derived" / "datscan_spatial"
VOLUMES = SPATIAL / "fpcit_mni"
VOLUMES.mkdir(parents=True, exist_ok=True)
FSL = PPMI / "derived" / "av133_spatial" / "mni152_fsl_2mm.nii.gz"

vois = striatal_vois(FSL, SPATIAL / "osgk_striatum.nii.gz")
affine = vois["affine"]
cwm = cwm_reference(FSL, vois["striatum"])

cohort = pd.read_csv(PPMI / "derived" / "datscan_recon_cohort.csv")
cohort["PATNO"] = cohort.PATNO.astype(int)
selected = cohort[cohort.in_xing_sbr & cohort.cohort.isin(COHORTS)].sort_values("PATNO")
wanted = {(int(row.PATNO), row.baseline_date): row.cohort for row in selected.itertuples()}
members = {key: files for key, files in recon_members(PPMI / "images" / "datscan_spect" / "_zips").items() if key in wanted}
print(f"baseline recons selected: {len(members)}")

fixed = ants.image_read(str(SPATIAL / "fpcit_fsl_grid.nii.gz"))
binary = dcm2niix()
registered = []
for index, (key, files) in enumerate(sorted(members.items())):
    out = VOLUMES / f"{key[0]}_suvr.nii.gz"
    if not out.exists() and not normalize(files, str(out), fixed, affine, cwm, binary):
        continue
    registered.append({"PATNO": key[0], "cohort": wanted[key]})
    if (index + 1) % 100 == 0:
        print(f"    {index + 1}/{len(members)}")
table = pd.DataFrame(registered).sort_values("PATNO")
print(f"registered volumes: {len(table)}")


def group_stack(label):
    return np.clip(
        np.stack([nib.load(VOLUMES / f"{patno}_suvr.nii.gz").get_fdata() for patno in table[table.cohort == label].PATNO]) - 1,
        0,
        None,
    )


hc_mean = group_stack("Healthy Control").mean(0)
pd_mean = group_stack("Parkinson's Disease").mean(0)
depletion = np.where(hc_mean > 0.15, (hc_mean - pd_mean) / hc_mean * 100, np.nan)
nib.save(
    nib.Nifti1Image(np.nan_to_num(depletion).astype(np.float32), affine),
    SPATIAL / "datscan_pct_depletion.nii.gz",
)
print("datscan_pct_depletion.nii.gz")
