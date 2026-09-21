#!/usr/bin/env python3
"""Register the staged PD follow-up DaTscan reconstructions to the FP-CIT/MNI grid as CWM-normalized SUVR volumes.

Run: PPMI_ROOT=/path/to/ppmi python compute/register_datscan_visits.py
"""
import warnings

import ants

warnings.filterwarnings("ignore")
from datscan_registration import (
    cwm_reference,
    dcm2niix,
    normalize,
    ppmi_root,
    select_visits,
    striatal_vois,
    visit_candidates,
)

PPMI = ppmi_root()
SPATIAL = PPMI / "derived" / "datscan_spatial"
VISITS = SPATIAL / "fpcit_mni_visits"
VISITS.mkdir(parents=True, exist_ok=True)
FSL = PPMI / "derived" / "av133_spatial" / "mni152_fsl_2mm.nii.gz"

vois = striatal_vois(FSL, SPATIAL / "osgk_striatum.nii.gz")
affine = vois["affine"]
cwm = cwm_reference(FSL, vois["striatum"])

candidates = visit_candidates(PPMI)
chosen = select_visits(candidates)
print(f"candidate visits: {len(candidates)}  selected: {len(chosen)}")

fixed = ants.image_read(str(SPATIAL / "fpcit_fsl_grid.nii.gz"))
binary = dcm2niix()
written = 0
for index, row in enumerate(chosen.itertuples()):
    out = VISITS / f"{row.PATNO}_{row.date}_suvr.nii.gz"
    if out.exists() or normalize(row.files, str(out), fixed, affine, cwm, binary, attempts=3):
        written += 1
    if (index + 1) % 100 == 0:
        print(f"    {index + 1}/{len(chosen)}")
print(f"visit volumes available: {written}")
