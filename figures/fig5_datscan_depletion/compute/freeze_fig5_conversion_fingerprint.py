#!/usr/bin/env python
"""Baseline SBR per participant in each DA-subtype projection-family territory.

Run: python freeze_fig5_conversion_fingerprint.py
"""
import os, numpy as np, pandas as pd, nibabel as nib, warnings; warnings.filterwarnings('ignore')
from nilearn import image
from joblib import Parallel, delayed
from conversion_paths import PRIVATE

PPMI = os.environ.get("PPMI_ROOT")
if not PPMI:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
DS = f"{PPMI}/derived/datscan_spatial"; FSLP = f"{PPMI}/derived/av133_spatial/mni152_fsl_2mm.nii.gz"
HERE = os.path.dirname(os.path.abspath(__file__)); FROZEN = os.path.join(HERE, "..", "frozen")

fsl = nib.load(FSLP); aff = fsl.affine
osgk = image.resample_to_img(f"{DS}/osgk_striatum.nii.gz", fsl, interpolation='nearest').get_fdata().astype(int)
_ax = lambda m: np.abs(aff[0, 0] * np.argwhere(m)[:, 0] + aff[0, 3]).mean()
labs = {l: _ax(osgk == l) for l in (1, 2, 3) if (osgk == l).any()}
PUT = osgk == max(labs, key=labs.get); STR = osgk > 0
POST = PUT.copy(); ymid = np.median([aff[1, 1] * j + aff[1, 3] for j in np.argwhere(PUT)[:, 1]])
for j in range(osgk.shape[1]):
    if aff[1, 1] * j + aff[1, 3] >= ymid: POST[:, j, :] = False

FAM = ['anxa1', 'sox6', 'sox6r', 'calb1']
Wm = {f: nib.load(f"{DS}/subtype_weight_{f}_fsl.nii.gz").get_fdata() for f in FAM}
SPEC = {}
for f in FAM:
    SPEC[f"{f}_put"] = (Wm[f] * PUT, True)
    SPEC[f"{f}_str"] = (Wm[f] * STR, True)
SPEC["postput"] = (POST.astype(float), False)
SPEC["striatum"] = (STR.astype(float), False)

CACHE = PRIVATE / "family_terr_sbr.parquet"
prog = pd.read_parquet(PRIVATE / "cohort_progression.parquet")
surv = pd.read_parquet(PRIVATE / "cohort_phenoconversion.parquet")
need = sorted(set(prog.PATNO) | set(surv.PATNO))

if not os.path.exists(CACHE):
    masks = {k: (w > 0, w[w > 0]) for k, (w, _) in SPEC.items()}

    def one(p):
        try: v = nib.load(f"{DS}/fpcit_mni/{p}_suvr.nii.gz").get_fdata() - 1.0
        except Exception: return None
        r = {"PATNO": p}
        for k, (m, w) in masks.items():
            r[k] = float((v[m] * w).sum() / w.sum())
        return r
    print(f"[load] {len(need)} baseline volumes", flush=True)
    rows = Parallel(n_jobs=8, verbose=5)(delayed(one)(p) for p in need)
    pd.DataFrame([r for r in rows if r]).to_parquet(CACHE, index=False)
T = pd.read_parquet(CACHE)
print(f"[terr] {len(T)} subjects x {len(SPEC)} territories", flush=True)
