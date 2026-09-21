#!/usr/bin/env python
"""Pipeline-MATCHED unweighted putaminal SBR comparators.

Run: python freeze_fig5_conversion_matched_sbr.py
"""
import os, sys, glob, json, numpy as np, nibabel as nib, pandas as pd, warnings; warnings.filterwarnings('ignore')
from nilearn import image
from joblib import Parallel, delayed
from conversion_paths import PRIVATE

PPMI = os.environ.get("PPMI_ROOT")
if not PPMI:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
DS = f"{PPMI}/derived/datscan_spatial"
VOL = f"{DS}/fpcit_mni"
FSL = f"{PPMI}/derived/av133_spatial/mni152_fsl_2mm.nii.gz"
WCACHE = f"{DS}/anxa1_putaminal_weight_fsl.nii.gz"
FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frozen")
os.makedirs(FROZEN, exist_ok=True)

fsl = nib.load(FSL); aff = fsl.affine
osgk = image.resample_to_img(f"{DS}/osgk_striatum.nii.gz", fsl, interpolation='nearest').get_fdata().astype(int)
_absx = lambda m: np.abs(aff[0, 0] * np.argwhere(m)[:, 0] + aff[0, 3]).mean()
labs = {l: _absx(osgk == l) for l in (1, 2, 3) if (osgk == l).any()}
CAUD = osgk == min(labs, key=labs.get)
PUT = osgk == max(labs, key=labs.get)
STR = osgk > 0
POST = PUT.copy()
ymid = np.median([aff[1, 1] * j + aff[1, 3] for j in np.argwhere(PUT)[:, 1]])
for j in range(osgk.shape[1]):
    if aff[1, 1] * j + aff[1, 3] >= ymid: POST[:, j, :] = False

W = nib.load(WCACHE).get_fdata()
II = np.arange(W.shape[0])[:, None, None]; xmm = aff[0, 0] * II + aff[0, 3]
LEFT, RIGHT = xmm > 0, xmm < 0

A = W > 0
geom = dict(
    putamen_voxels=int(PUT.sum()), post_putamen_voxels=int(POST.sum()), anxa1_field_voxels=int(A.sum()),
    anxa1_frac_of_putamen=float(A.sum() / PUT.sum()),
    anxa1_frac_inside_post=float((A & POST).sum() / A.sum()),
    post_frac_covered_by_anxa1=float((A & POST).sum() / POST.sum()),
    dice_anxa1_vs_post=float(2 * (A & POST).sum() / (A.sum() + POST.sum())),
    weight_mass_inside_post=float(W[POST].sum() / W.sum()),
)
json.dump(geom, open(f"{FROZEN}/field_geometry.json", "w"), indent=2)
print("[geometry]", json.dumps(geom, indent=2), flush=True)

masks = {}
for nm, m in [("putamen", PUT), ("postput", POST), ("caudate", CAUD), ("striatum", STR)]:
    masks[f"{nm}_L"] = m & LEFT; masks[f"{nm}_R"] = m & RIGHT
masks["anxa1flat_L"] = A & LEFT; masks["anxa1flat_R"] = A & RIGHT


def one(f):
    p = int(os.path.basename(f).split('_')[0])
    try:
        v = nib.load(f).get_fdata()
    except Exception:
        return None
    row = {"PATNO": p}
    for k, m in masks.items():
        row[k] = float(v[m].mean() - 1.0)
    return row


files = sorted(glob.glob(f"{VOL}/*_suvr.nii.gz"))
print(f"[scan] {len(files)} baseline volumes", flush=True)
rows = Parallel(n_jobs=8, verbose=5)(delayed(one)(f) for f in files)
df = pd.DataFrame([r for r in rows if r is not None])
for nm in ["putamen", "postput", "caudate", "striatum", "anxa1flat"]:
    df[nm] = (df[f"{nm}_L"] + df[f"{nm}_R"]) / 2
df.to_parquet(PRIVATE / "matched_putamen_sbr.parquet", index=False)
print(f"saved -> {PRIVATE / 'matched_putamen_sbr.parquet'}  n={len(df)}", flush=True)
print(df[["putamen", "postput", "caudate", "anxa1flat"]].describe().to_string(), flush=True)
