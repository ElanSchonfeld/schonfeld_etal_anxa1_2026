#!/usr/bin/env python3
"""Write the Anxa1 putaminal weight field, the four subtype territory weight fields, and the per-volume Anxa1 territory and Anxa1-versus-rest putamen readouts.

Run: PPMI_ROOT=/path/to/ppmi python compute/build_projection_weights_and_caches.py
"""
import warnings

import nibabel as nib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nilearn import image

from datscan_registration import ppmi_root, striatal_vois
from projection_fields import build_hmba_projection_frame

SUBTYPES = {"anxa1": "Anxa1", "sox6": "Sox6-all", "sox6r": "Sox6-rest", "calb1": "Calb1"}

PPMI = ppmi_root()
SPATIAL = PPMI / "derived" / "datscan_spatial"
VOLUMES = SPATIAL / "fpcit_mni"
VISITS = SPATIAL / "fpcit_mni_visits"
FSL = PPMI / "derived" / "av133_spatial" / "mni152_fsl_2mm.nii.gz"

vois = striatal_vois(FSL, SPATIAL / "osgk_striatum.nii.gz")
affine = vois["affine"]
putamen = vois["putamen"]
striatum = vois["striatum"]
reference = nib.load(FSL)
frame = build_hmba_projection_frame(PPMI)


def to_fsl(field):
    source = nib.Nifti1Image(np.nan_to_num(field), frame.affine)
    return image.resample_to_img(source, reference, interpolation="linear").get_fdata()


weight_path = SPATIAL / "anxa1_putaminal_weight_fsl.nii.gz"
nib.save(
    nib.Nifti1Image((np.clip(to_fsl(frame.centered_fields["Anxa1"]), 0, None) * putamen).astype(np.float32), affine),
    weight_path,
)
anxa1_weight = nib.load(weight_path).get_fdata()
print(f"anxa1_putaminal_weight_fsl.nii.gz {int((anxa1_weight > 0).sum())} voxels")
for short, group in SUBTYPES.items():
    territory = np.clip(to_fsl(frame.centered_fields[group]), 0, None) * striatum
    nib.save(nib.Nifti1Image(territory.astype(np.float32), affine), SPATIAL / f"subtype_weight_{short}_fsl.nii.gz")
    print(f"subtype_weight_{short}_fsl.nii.gz {int((territory > 0).sum())} voxels")

x_mm = affine[0, 0] * np.arange(anxa1_weight.shape[0])[:, None, None] + affine[0, 3]
weight_left = anxa1_weight * (x_mm > 0)
weight_right = anxa1_weight * (x_mm < 0)
mask_left, mask_right = weight_left > 0, weight_right > 0
total_left, total_right = weight_left[mask_left].sum(), weight_right[mask_right].sum()
anxa_putamen = putamen & (anxa1_weight > 0)
rest_putamen = putamen & (anxa1_weight <= 0)
REGIONS = {
    "anxaput": {"L": anxa_putamen & (x_mm > 0), "R": anxa_putamen & (x_mm < 0)},
    "restput": {"L": rest_putamen & (x_mm > 0), "R": rest_putamen & (x_mm < 0)},
}


def sources():
    for path in sorted(VOLUMES.glob("*_suvr.nii.gz")):
        yield "baseline", path, int(path.name.split("_")[0]), None
    for path in sorted(VISITS.glob("*_suvr.nii.gz")):
        stem = path.name[: -len("_suvr.nii.gz")]
        yield "visit", path, int(stem.split("_")[0]), stem.split("_", 1)[1][:7]


weighted_rows = []
territory_rows = []
for source, path, patno, month in sources():
    try:
        volume = nib.load(path).get_fdata()
    except Exception:
        continue
    sbr = np.clip(volume - 1, 0, None)
    weighted_rows.append({
        "source": source,
        "PATNO": patno,
        "ym": month,
        "anxa1put_L": float((weight_left[mask_left] * sbr[mask_left]).sum() / total_left),
        "anxa1put_R": float((weight_right[mask_right] * sbr[mask_right]).sum() / total_right),
    })
    record = {"source": source, "PATNO": patno, "ym": month}
    for name, sides in REGIONS.items():
        for side, mask in sides.items():
            record[f"{name}_{side}"] = float(np.nanmean(np.clip(volume[mask] - 1, 0, None)))
    territory_rows.append(record)

weighted = pd.DataFrame(weighted_rows)
weighted.to_parquet(SPATIAL / "anxa1_putaminal_spatial.parquet", index=False)
print(f"anxa1_putaminal_spatial.parquet {weighted.shape}")
territory = pd.DataFrame(territory_rows)
territory.to_parquet(SPATIAL / "anxa_vs_nonanxa_putamen.parquet", index=False)
print(f"anxa_vs_nonanxa_putamen.parquet {territory.shape}")
