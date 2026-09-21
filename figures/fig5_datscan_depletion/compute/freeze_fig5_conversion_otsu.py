#!/usr/bin/env python
"""Recompute main-figure panel C using an outcome-independent Otsu partition.

Run: python freeze_fig5_conversion_otsu.py
"""
import json
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import image
from scipy.stats import chi2
from skimage.filters import threshold_otsu

from survutil import cox_fit
from conversion_paths import PRIVATE

warnings.filterwarnings("ignore")

PPMI_VALUE = os.environ.get("PPMI_ROOT")
if not PPMI_VALUE:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PPMI = Path(PPMI_VALUE).expanduser().resolve()
DS = PPMI / "derived/datscan_spatial"
FSLP = PPMI / "derived/av133_spatial/mni152_fsl_2mm.nii.gz"
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
T, E = "tyears_pd", "event_pd"

fsl = nib.load(FSLP)
aff = fsl.affine
osgk = image.resample_to_img(
    DS / "osgk_striatum.nii.gz", fsl, interpolation="nearest"
).get_fdata().astype(int)


def abs_x(mask):
    ijk = np.argwhere(mask)
    return np.abs(aff[0, 0] * ijk[:, 0] + aff[0, 3]).mean()


labs = {lab: abs_x(osgk == lab) for lab in (1, 2, 3) if (osgk == lab).any()}
putamen = osgk == max(labs, key=labs.get)
posterior = putamen.copy()
ijk = np.argwhere(putamen)
y_mid = np.median(aff[1, 1] * ijk[:, 1] + aff[1, 3])
for j in range(osgk.shape[1]):
    if aff[1, 1] * j + aff[1, 3] >= y_mid:
        posterior[:, j, :] = False

weight = nib.load(DS / "anxa1_putaminal_weight_fsl.nii.gz").get_fdata()
cut = float(threshold_otsu(weight[posterior]))
high = posterior & (weight > cut)
complement = posterior & ~high

ii = np.arange(osgk.shape[0])[:, None, None]
x_mm = aff[0, 0] * ii + aff[0, 3]
left, right = x_mm > 0, x_mm < 0
masks = {
    "high": high,
    "complement": complement,
    "high_L": high & left,
    "high_R": high & right,
    "complement_L": complement & left,
    "complement_R": complement & right,
    "postput_L": posterior & left,
    "postput_R": posterior & right,
}
geometry = {
    "method": "two-class Otsu threshold over posterior-putamen ANXA1 weights",
    "threshold": cut,
    "posterior_putamen_voxels": int(posterior.sum()),
    **{f"{name}_voxels": int(mask.sum()) for name, mask in masks.items()},
}
assert geometry["high_voxels"] == 319
assert geometry["complement_voxels"] == 369
print(json.dumps(geometry, indent=2), flush=True)

surv = pd.read_parquet(PRIVATE / "cohort_phenoconversion.parquet")
cache = PRIVATE / "postput_partition_otsu_sbr.parquet"


def one(patno):
    try:
        volume = nib.load(DS / f"fpcit_mni/{patno}_suvr.nii.gz").get_fdata() - 1.0
    except Exception:
        return None
    row = {"PATNO": patno}
    for name, mask in masks.items():
        row[name] = float(volume[mask].mean())
    return row


if cache.exists():
    values = pd.read_parquet(cache)
else:
    rows = Parallel(n_jobs=8, prefer="threads")(
        delayed(one)(patno) for patno in surv.PATNO.tolist()
    )
    values = pd.DataFrame([row for row in rows if row is not None])
    values.to_parquet(cache, index=False)

d = surv[["PATNO", T, E, "age", "sex"]].merge(values, on="PATNO")
d = d[d[T] > 0].copy()
d[E] = d[E].astype(int)
zscore = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
d["age_z"] = zscore(d.age)
d["sexf"] = d.sex.astype(float)

worse_left = d.postput_L < d.postput_R
for level in ("high", "complement"):
    d[f"{level}_worse"] = np.where(worse_left, d[f"{level}_L"], d[f"{level}_R"])
    d[f"{level}_better"] = np.where(worse_left, d[f"{level}_R"], d[f"{level}_L"])

designs = [
    ("bilateral posterior putamen", "high", "complement"),
    ("more-affected hemisphere", "high_worse", "complement_worse"),
    ("less-affected hemisphere", "high_better", "complement_better"),
]
covariates = ["age_z", "sexf"]
rows = []
for label, high_col, complement_col in designs:
    x = d.dropna(subset=[high_col, complement_col]).copy()
    x["A"] = -zscore(x[high_col])
    x["B"] = -zscore(x[complement_col])
    model_a = cox_fit(x, ["A"] + covariates, T, E)
    model_b = cox_fit(x, ["B"] + covariates, T, E)
    model_ab = cox_fit(x, ["A", "B"] + covariates, T, E)
    lr_forward = 2 * (model_ab.llf - model_b.llf)
    lr_reverse = 2 * (model_ab.llf - model_a.llf)
    rows.append({
        "design": label,
        "n": int(len(x)),
        "events": int(x[E].sum()),
        "rho": float(x.A.corr(x.B)),
        "HR_high": float(np.exp(model_a.params[0])),
        "HR_complement": float(np.exp(model_b.params[0])),
        "lr_fwd": float(lr_forward),
        "p_fwd": float(chi2.sf(lr_forward, 1)),
        "lr_rev": float(lr_reverse),
        "p_rev": float(chi2.sf(lr_reverse, 1)),
        "beta_high_joint": float(model_ab.params[0]),
        "beta_complement_joint": float(model_ab.params[1]),
    })

result = pd.DataFrame(rows)
result.to_csv(FROZEN / "partition_asymmetry_otsu.csv", index=False)
with open(FROZEN / "postput_otsu_geometry.json", "w") as handle:
    json.dump(geometry, handle, indent=2)
meta = {
    "n": int(len(d)),
    "events": int(d[E].sum()),
    "threshold": cut,
    "high_voxels": int(high.sum()),
    "complement_voxels": int(complement.sum()),
}
with open(FROZEN / "partition_asymmetry_otsu_meta.json", "w") as handle:
    json.dump(meta, handle, indent=2)

print(result.round(6).to_string(index=False))
