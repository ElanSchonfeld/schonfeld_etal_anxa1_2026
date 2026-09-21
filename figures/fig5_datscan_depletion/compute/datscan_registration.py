#!/usr/bin/env python3
"""Shared DaTscan inputs: the reconstructed-series index of the DICOM archives, the striatal volumes of interest, the white-matter reference, the FP-CIT/MNI SUVR normalizer and the staged PD visit selection."""
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nilearn import datasets, image
from scipy import ndimage

SERIES = re.compile(r"PPMI/(\d+)/(.+?)/(\d{4}-\d{2}-\d{2})[^/]*/(I\d+)/")
RECONSTRUCTED = re.compile(r"reconstruct", re.I)
SMOOTH_FWHM = 6.0
SHAPE = (91, 109, 91)
PER_BIN = 120
TIME_BINS = [
    (0, 0.5, "<0.5y"), (0.5, 1, "0.5-1y"), (1, 1.5, "1-1.5y"), (1.5, 3, "1.5-3y"),
    (3, 5, "3-5y"), (5, 8, "5-8y"), (8, 40, "8y+"),
]
TIME_ORDER = [name for _, _, name in TIME_BINS]
UPDRS_BINS = [
    (0, 20, "UPDRS 0-20"), (20, 40, "UPDRS 20-40"), (40, 60, "UPDRS 40-60"),
    (60, 80, "UPDRS 60-80"), (80, 100, "UPDRS 80-100"), (100, 200, "UPDRS 100+"),
]
STAGE_COLUMNS = ["yrs_stage", "hy_stage", "updrs_stage"]


def ppmi_root():
    root = os.environ.get("PPMI_ROOT")
    if not root:
        raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
    return Path(root).expanduser().resolve()


def dcm2niix():
    binary = shutil.which("dcm2niix")
    if not binary:
        raise RuntimeError("dcm2niix is not on PATH.")
    return binary


def recon_members(zips_dir):
    members = {}
    for archive in sorted(glob.glob(f"{zips_dir}/*.zip")):
        try:
            names = zipfile.ZipFile(archive).namelist()
        except Exception:
            continue
        for name in names:
            if not name.lower().endswith(".dcm"):
                continue
            match = SERIES.search(name)
            if match and RECONSTRUCTED.search(match.group(2)):
                members.setdefault((int(match.group(1)), match.group(3)), []).append((archive, name))
    return members


def striatal_vois(fsl_path, osgk_path):
    affine = nib.load(fsl_path).affine
    labels = image.resample_to_img(str(osgk_path), str(fsl_path), interpolation="nearest").get_fdata().astype(int)
    mean_abs_x = lambda mask: np.abs(affine[0, 0] * np.argwhere(mask)[:, 0] + affine[0, 3]).mean()
    extents = {label: mean_abs_x(labels == label) for label in (1, 2, 3) if (labels == label).any()}
    return {
        "affine": affine,
        "putamen": labels == max(extents, key=extents.get),
        "striatum": labels > 0,
    }


def cwm_reference(fsl_path, striatum):
    prior = image.resample_to_img(
        datasets.fetch_icbm152_2009()["wm"], str(fsl_path), interpolation="continuous"
    ).get_fdata()
    return ndimage.binary_erosion(prior > 0.9, iterations=2) & ~ndimage.binary_dilation(striatum, iterations=3)


def time_stage(years):
    for low, high, name in TIME_BINS:
        if low <= years < high:
            return name
    return None


def hy_stage(value):
    if pd.isna(value) or value == 101 or value < 0 or value > 5:
        return None
    return f"H&Y {value:g}"


def updrs_stage(value):
    if pd.isna(value) or value < 0 or value > 108:
        return None
    for low, high, name in UPDRS_BINS:
        if low <= value < high:
            return name
    return None


def visit_candidates(ppmi_root):
    cohort = pd.read_csv(ppmi_root / "derived" / "datscan_recon_cohort.csv")
    pd_ids = set(cohort[(cohort.cohort == "Parkinson's Disease") & cohort.in_xing_sbr].PATNO.astype(int))
    scans = pd.read_parquet(
        ppmi_root / "derived" / "ppmi_scan_datscan.parquet",
        columns=["PATNO", "scan_date", "yrs_since_onset", "NHY", "MDSUPDRS3_total", "AGE_AT_VISIT", "SEX"],
    ).dropna(subset=["yrs_since_onset"])
    scans["PATNO"] = scans.PATNO.astype(int)
    scans["ym"] = pd.to_datetime(scans.scan_date, errors="coerce").dt.strftime("%Y-%m")
    clinical = scans.dropna(subset=["ym"]).drop_duplicates(["PATNO", "ym"]).set_index(["PATNO", "ym"])
    rows = []
    for (patno, date), files in recon_members(ppmi_root / "images" / "datscan_spect" / "_zips").items():
        if patno not in pd_ids:
            continue
        try:
            record = clinical.loc[(patno, date[:7])]
        except KeyError:
            continue
        rows.append({
            "PATNO": patno,
            "date": date,
            "yrs": float(record.yrs_since_onset),
            "NHY": record.NHY,
            "updrs": record.MDSUPDRS3_total,
            "age": record.AGE_AT_VISIT,
            "sex": record.SEX,
            "files": files,
        })
    candidates = pd.DataFrame(rows).dropna(subset=["yrs"])
    candidates["yrs_stage"] = candidates.yrs.map(time_stage)
    candidates["hy_stage"] = candidates.NHY.map(hy_stage)
    candidates["updrs_stage"] = candidates.updrs.map(updrs_stage)
    return candidates


def select_visits(candidates):
    selected = set()
    for column in STAGE_COLUMNS:
        for _, group in candidates.dropna(subset=[column]).groupby(column):
            selected.update(group.sort_values(["PATNO", "date"]).head(PER_BIN).index)
    return candidates.loc[sorted(selected)]


def normalize(files, out, fixed, affine, cwm, binary, attempts=1):
    import ants

    for attempt in range(attempts):
        workdir = tempfile.mkdtemp()
        try:
            for archive, member in files:
                with open(f"{workdir}/{os.path.basename(member)}", "wb") as handle:
                    handle.write(zipfile.ZipFile(archive).read(member))
            subprocess.run([binary, "-o", workdir, "-f", "v", "-z", "y", workdir], capture_output=True)
            converted = sorted(glob.glob(f"{workdir}/v*.nii.gz"), key=os.path.getsize, reverse=True)
            if not converted:
                return False
            volume = nib.load(converted[0]).get_fdata().squeeze()
            volume = volume.mean(3) if volume.ndim == 4 else volume
            if volume.shape != SHAPE:
                return False
            volume = volume - np.percentile(volume, 1)
            staged = f"{workdir}/r.nii.gz"
            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), staged)
            registered = ants.registration(
                fixed, ants.image_read(staged), type_of_transform="Affine", aff_metric="mattes"
            )
            smoothed = image.smooth_img(
                nib.Nifti1Image(registered["warpedmovout"].numpy(), affine), fwhm=SMOOTH_FWHM
            ).get_fdata()
            nib.save(nib.Nifti1Image((smoothed / smoothed[cwm].mean()).astype(np.float32), affine), out)
            if os.path.exists(out):
                return True
            raise OSError("write did not persist")
        except Exception as error:
            if attempt == attempts - 1:
                print(f"    skip: {type(error).__name__}")
                return False
            time.sleep(3)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return False
