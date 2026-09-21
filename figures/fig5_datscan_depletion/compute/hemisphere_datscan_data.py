#!/usr/bin/env python3
"""Clinical hemisphere alignment and spatial DaTscan readouts for Figure 5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = [
    "PuR",
    "PuC",
    "PuPV",
    "PuPD",
    "PuAV",
    "PuAD",
    "Putamen",
    "Caudate",
    "Striatum",
]


def _required_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def clinical_more_affected_side(domside) -> str | None:
    """Map dominant motor-symptom side to the contralateral brain hemisphere."""
    if domside == 1:
        return "R"
    if domside == 2:
        return "L"
    return None


def genetic_group(row) -> str:
    """Return the mutually exclusive PD genetic group used in Figure 5."""
    lrrk2 = bool(row.LRRK2_carrier) if pd.notna(row.LRRK2_carrier) else False
    gba = bool(row.GBA_carrier) if pd.notna(row.GBA_carrier) else False
    return "LRRK2" if lrrk2 else ("GBA" if gba else "idiopathic")


def exclude_double_carriers(scans: pd.DataFrame) -> pd.DataFrame:
    """Drop scan rows flagged as carrying both an LRRK2 and a GBA variant."""
    lrrk2 = scans.LRRK2_carrier.fillna(False).astype(bool)
    gba = scans.GBA_carrier.fillna(False).astype(bool)
    return scans[~(lrrk2 & gba)].copy()


def add_clinical_hemisphere_columns(scans: pd.DataFrame) -> pd.DataFrame:
    """Add more-affected and less-affected regional SBR columns in place."""
    scans["ma_hemi"] = scans.DOMSIDE.map(clinical_more_affected_side)
    for region in REGIONS:
        left = scans[f"SBR_{region}_L"]
        right = scans[f"SBR_{region}_R"]
        scans[f"SBR_{region}_MA"] = np.where(
            scans.ma_hemi == "R", right, np.where(scans.ma_hemi == "L", left, np.nan)
        )
        scans[f"SBR_{region}_LA"] = np.where(
            scans.ma_hemi == "R", left, np.where(scans.ma_hemi == "L", right, np.nan)
        )
    return scans


def add_spatial_anxa1_columns(
    scans: pd.DataFrame, ppmi_root: str | Path
) -> pd.DataFrame:
    """Attach the voxel-weighted Anxa1 putaminal readout to the scan table."""
    ppmi_root = Path(ppmi_root).expanduser().resolve()
    spatial_path = _required_file(
        ppmi_root / "derived/datscan_spatial/anxa1_putaminal_spatial.parquet",
        "voxel-weighted Anxa1 putaminal cache",
    )
    cohort_path = _required_file(
        ppmi_root / "derived/datscan_recon_cohort.csv",
        "DaTscan reconstruction cohort",
    )

    for suffix in ("", "_L", "_R", "_MA", "_LA"):
        scans[f"SBR_anxa1put{suffix}"] = np.nan

    spatial = pd.read_parquet(spatial_path)
    spatial["PATNO"] = spatial.PATNO.astype(int)
    scans["ym"] = pd.to_datetime(scans.scan_date, errors="coerce").dt.strftime("%Y-%m")
    visit = spatial[spatial.source == "visit"].set_index(["PATNO", "ym"])[
        ["anxa1put_L", "anxa1put_R"]
    ]
    baseline = (
        spatial[spatial.source == "baseline"]
        .drop_duplicates("PATNO")
        .set_index("PATNO")[["anxa1put_L", "anxa1put_R"]]
    )

    cohort = pd.read_csv(cohort_path)
    cohort["PATNO"] = cohort.PATNO.astype(int)
    try:
        baseline_month = (
            cohort.dropna(subset=["baseline_date"])
            .assign(
                ym=lambda data: pd.to_datetime(data.baseline_date).dt.strftime(
                    "%Y-%m"
                )
            )
            .drop_duplicates("PATNO")
            .set_index("PATNO")["ym"]
        )
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime):
        baseline_month = (
            scans.sort_values("scan_date")
            .drop_duplicates("PATNO")
            .set_index("PATNO")["ym"]
        )

    left = np.full(len(scans), np.nan)
    right = np.full(len(scans), np.nan)
    patients = scans.PATNO.to_numpy()
    months = scans.ym.to_numpy()
    has_visit_maps = scans.COHORT_DEFINITION.eq("Parkinson's Disease").to_numpy()
    for index in range(len(scans)):
        key = (patients[index], months[index])
        if key in visit.index:
            left[index] = visit.at[key, "anxa1put_L"]
            right[index] = visit.at[key, "anxa1put_R"]
        elif patients[index] in baseline.index and (
            baseline_month.get(patients[index]) == months[index]
            or not has_visit_maps[index]
        ):
            left[index] = baseline.at[patients[index], "anxa1put_L"]
            right[index] = baseline.at[patients[index], "anxa1put_R"]

    scans["SBR_anxa1put_L"] = left
    scans["SBR_anxa1put_R"] = right
    scans["SBR_anxa1put"] = np.nanmean(np.c_[left, right], axis=1)
    scans["SBR_anxa1put_MA"] = np.where(
        scans.ma_hemi == "R", right, np.where(scans.ma_hemi == "L", left, np.nan)
    )
    scans["SBR_anxa1put_LA"] = np.where(
        scans.ma_hemi == "R", left, np.where(scans.ma_hemi == "L", right, np.nan)
    )
    return scans


def load_scan_with_hemisphere_fields(ppmi_root: str | Path) -> pd.DataFrame:
    ppmi_root = Path(ppmi_root).expanduser().resolve()
    scan_path = _required_file(
        ppmi_root / "derived/ppmi_scan_datscan.parquet",
        "canonical scan-level DaTscan table",
    )
    scans = pd.read_parquet(scan_path)
    scans["PATNO"] = scans.PATNO.astype(int)
    scans["sex"] = scans.SEX
    add_clinical_hemisphere_columns(scans)
    add_spatial_anxa1_columns(scans, ppmi_root)
    return scans
