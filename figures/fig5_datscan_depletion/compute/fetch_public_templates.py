#!/usr/bin/env python3
"""Download the MNI reference grids, striatal atlas, FP-CIT template and HMBA-BG annotation, and resample the FP-CIT template onto the FSL 2 mm grid.

Run: PPMI_ROOT=/path/to/ppmi python compute/fetch_public_templates.py
"""
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import nibabel as nib
from nilearn import image

TEMPLATEFLOW = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym"
NEUROVAULT_API = "https://neurovault.org/api/images/406338/"
FPCIT_URL = "https://sourceforge.net/projects/spmtemplates/files/symFPCITtemplate_MNI_norm.nii/download"
HMBA_URL = (
    "https://allen-atlas-assets.s3.amazonaws.com/annotation-sets/"
    "hmba-adult-human-hombabg-annotation/2025/annotations_compressed_700.nii.gz"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}
CHECKSUMS = {
    "mni152_fsl_2mm.nii.gz": "6071a87fd702e99ffc0347b0b1a1f796",
    "mni152_fsl_1mm.nii.gz": "c1c90ea94864dbc7a0aedac6142e364e",
    "osgk_striatum.nii.gz": "f750be1a7a5726e4f567d267a9e608c9",
    "fpcit_template_mni.nii": "9b3bd257465cc3ffa79f08d0cc9d8943",
    "annotations_compressed_700.nii.gz": "b4ec0ae2b3e7869cffd0385cd8be65f8",
}

_root = os.environ.get("PPMI_ROOT")
if not _root:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PPMI = Path(_root).expanduser().resolve()
AV133 = PPMI / "derived" / "av133_spatial"
SPATIAL = PPMI / "derived" / "datscan_spatial"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"
for folder in (AV133, SPATIAL, FROZEN):
    folder.mkdir(parents=True, exist_ok=True)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as handle:
        shutil.copyfileobj(response, handle)


def fetch(url, path):
    path = Path(path)
    expected = CHECKSUMS[path.name]
    if not path.is_file() or md5(path) != expected:
        download(url, path)
    found = md5(path)
    if found != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected}, got {found}")
    print(f"  {path.name} {found}")
    return path


fsl_2mm = fetch(f"{TEMPLATEFLOW}/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz", AV133 / "mni152_fsl_2mm.nii.gz")
fetch(f"{TEMPLATEFLOW}/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz", AV133 / "mni152_fsl_1mm.nii.gz")

osgk = SPATIAL / "osgk_striatum.nii.gz"
if not osgk.is_file() or md5(osgk) != CHECKSUMS[osgk.name]:
    with urllib.request.urlopen(urllib.request.Request(NEUROVAULT_API, headers=HEADERS), timeout=300) as handle:
        source = json.load(handle)["file"].replace("http://", "https://")
    fetch(source, osgk)
else:
    print(f"  {osgk.name} {md5(osgk)}")

fpcit_raw = fetch(FPCIT_URL, SPATIAL / "fpcit_template_mni.nii")
fetch(HMBA_URL, FROZEN / "annotations_compressed_700.nii.gz")

grid = SPATIAL / "fpcit_fsl_grid.nii.gz"
image.resample_to_img(str(fpcit_raw), str(fsl_2mm), interpolation="continuous").to_filename(grid)
print(f"  {grid.name} {nib.load(grid).shape}")
