#!/usr/bin/env python3
"""Hemisphere-aligned voxel analysis for Figure 5."""
from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy.stats import norm, spearmanr

from hemisphere_datscan_data import (
    clinical_more_affected_side,
    exclude_double_carriers,
    genetic_group,
)
from projection_fields import (
    GROUP_ORDER,
    build_hmba_projection_frame,
    resample_group_fields_to_fsl,
)


def _required_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _aggregate_content_hash(paths: list[Path]) -> str:
    """Hash an ordered input set without exposing participant-bearing filenames."""
    aggregate = hashlib.sha256()
    for path in paths:
        item = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                item.update(block)
        aggregate.update(item.digest())
    return aggregate.hexdigest()


def _canonicalize_volume(values: np.ndarray, more_affected_side: str) -> np.ndarray:
    """Place the clinically more-affected hemisphere at world x less than zero."""
    return np.flip(values, axis=0) if more_affected_side == "L" else values


def _block_bootstrap_correspondence(
    enrichment: dict[str, np.ndarray],
    depletion: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    *,
    seed: int = 42,
    n_bootstrap: int = 500,
) -> dict[str, float | tuple[float, float]]:
    """Estimate paired 6 mm block-bootstrap projection correspondences."""
    ijk = np.array(np.where(mask)).T
    millimetres = (affine[:3, :3] @ ijk.T + affine[:3, 3:]).T
    blocks = np.floor(millimetres / 6.0).astype(np.int64)
    block_keys = blocks[:, 0] * 100003 + blocks[:, 1] * 317 + blocks[:, 2]
    unique_blocks = np.unique(block_keys)
    block_indices = [np.where(block_keys == key)[0] for key in unique_blocks]
    rng = np.random.default_rng(seed)
    samples = [
        np.concatenate(
            [
                block_indices[position]
                for position in rng.integers(
                    0, len(unique_blocks), len(unique_blocks)
                )
            ]
        )
        for _ in range(n_bootstrap)
    ]

    def observed(group: str) -> float:
        valid = np.isfinite(enrichment[group]) & np.isfinite(depletion)
        return float(spearmanr(enrichment[group][valid], depletion[valid])[0])

    def resampled(group: str, sample: np.ndarray) -> float:
        group_values = enrichment[group][sample]
        depletion_values = depletion[sample]
        valid = np.isfinite(group_values) & np.isfinite(depletion_values)
        if valid.sum() <= 20:
            return np.nan
        return float(spearmanr(group_values[valid], depletion_values[valid])[0])

    bootstrap = {
        group: np.array([resampled(group, sample) for sample in samples])
        for group in ("Anxa1", "Sox6-rest")
    }
    difference = bootstrap["Anxa1"] - bootstrap["Sox6-rest"]
    difference = difference[np.isfinite(difference)]
    observed_difference = observed("Anxa1") - observed("Sox6-rest")
    standard_error = difference.std()
    p_value = (
        0.0
        if standard_error <= 0
        else float(2 * norm.sf(abs(observed_difference) / standard_error))
    )
    result: dict[str, float | tuple[float, float]] = {
        "dRho": observed_difference,
        "ci": (
            float(np.percentile(difference, 2.5)),
            float(np.percentile(difference, 97.5)),
        ),
        "p": p_value,
        "rho_anxa1": observed("Anxa1"),
        "rho_sox6all": observed("Sox6-all"),
        "rho_sox6rest": observed("Sox6-rest"),
    }

    bootstrap["Sox6-all"] = np.array(
        [resampled("Sox6-all", sample) for sample in samples]
    )
    for group, key in (
        ("Anxa1", "anxa1"),
        ("Sox6-all", "sox6all"),
        ("Sox6-rest", "sox6rest"),
    ):
        values = bootstrap[group][np.isfinite(bootstrap[group])]
        rho = observed(group)
        standard_error = values.std()
        result[f"ci_{key}"] = (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        result[f"p_{key}"] = (
            0.0
            if standard_error <= 0
            else float(2 * norm.sf(abs(rho) / standard_error))
        )
    return result


def compute_hemisphere_voxel_analysis(
    ppmi_root: str | Path,
    *,
    seed: int = 42,
    n_bootstrap: int = 500,
    locked_bins_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
) -> dict:
    """Return the matched GBA excess map and MA/LA projection concordance."""
    ppmi_root = Path(ppmi_root).expanduser().resolve()
    cohort_path = _required_file(
        ppmi_root / "derived/datscan_recon_cohort.csv",
        "DaTscan reconstruction cohort",
    )
    scan_path = _required_file(
        ppmi_root / "derived/ppmi_scan_datscan.parquet",
        "canonical scan-level DaTscan table",
    )
    volume_dir = ppmi_root / "derived/datscan_spatial/fpcit_mni"
    if not volume_dir.is_dir():
        raise FileNotFoundError(f"Missing registered DaTscan volume directory: {volume_dir}")
    osgk_path = _required_file(
        ppmi_root / "derived/datscan_spatial/osgk_striatum.nii.gz",
        "OSGK striatum atlas",
    )

    frame = build_hmba_projection_frame(
        ppmi_root,
        locked_bins_path=locked_bins_path,
        annotation_path=annotation_path,
    )
    fsl_fields = resample_group_fields_to_fsl(frame)
    fsl_reference = fsl_fields.reference

    cohort = pd.read_csv(cohort_path)
    cohort["PATNO"] = cohort.PATNO.astype(int)
    scans = pd.read_parquet(scan_path)
    scans["PATNO"] = scans.PATNO.astype(int)
    scans = exclude_double_carriers(scans)
    pd_patients = set(
        cohort[
            (cohort.cohort == "Parkinson's Disease") & cohort.in_xing_sbr
        ].PATNO
    )
    baseline = (
        scans[scans.PATNO.isin(pd_patients)]
        .sort_values(["PATNO", "scan_date"])
        .drop_duplicates("PATNO")
        .copy()
    )
    baseline["grp3"] = baseline.apply(genetic_group, axis=1)
    baseline["ma_hemi"] = baseline.DOMSIDE.map(clinical_more_affected_side)
    baseline = baseline.rename(
        columns={"AGE_AT_VISIT": "age", "yrs_since_onset": "dur"}
    )
    baseline["sex"] = baseline.SEX
    baseline = baseline[baseline.ma_hemi.notna()]

    idiopathic = baseline[baseline.grp3 == "idiopathic"]
    used = set()
    pairs = []
    for _, carrier in baseline[baseline.grp3 == "GBA"].iterrows():
        candidates = idiopathic[
            (idiopathic.sex == carrier.sex)
            & (idiopathic.age.sub(carrier.age).abs() <= 5)
            & (idiopathic.dur.sub(carrier.dur).abs() <= 1.5)
            & (~idiopathic.PATNO.isin(used))
        ]
        if len(candidates) == 0:
            continue
        pick = candidates.iloc[
            (
                candidates.age.sub(carrier.age).abs()
                + candidates.dur.sub(carrier.dur).abs()
            ).values.argmin()
        ]
        used.add(pick.PATNO)
        pairs.append(
            (
                int(carrier.PATNO),
                carrier.ma_hemi,
                int(pick.PATNO),
                pick.ma_hemi,
            )
        )

    def load_volume(patient: int, more_affected_side: str):
        path = volume_dir / f"{patient}_suvr.nii.gz"
        if not path.is_file():
            return None, path
        image_data = nib.load(path)
        if image_data.shape != fsl_reference.shape or not np.allclose(
            image_data.affine, fsl_reference.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"Registered DaTscan volume is not on the FSL grid: {path}")
        values = np.clip(image_data.get_fdata() - 1, 0, None)
        return _canonicalize_volume(values, more_affected_side), path

    gba_volumes = []
    idiopathic_volumes = []
    volume_paths = []
    for gba_patient, gba_side, idiopathic_patient, idiopathic_side in pairs:
        gba_volume, gba_path = load_volume(gba_patient, gba_side)
        idiopathic_volume, idiopathic_path = load_volume(
            idiopathic_patient, idiopathic_side
        )
        if gba_volume is None or idiopathic_volume is None:
            continue
        gba_volumes.append(gba_volume)
        idiopathic_volumes.append(idiopathic_volume)
        volume_paths.extend([gba_path, idiopathic_path])
    if not gba_volumes:
        raise RuntimeError("No matched GBA and idiopathic DaTscan volumes were available")

    excess_fsl = np.nanmean(idiopathic_volumes, axis=0) - np.nanmean(
        gba_volumes, axis=0
    )
    osgk = image.resample_to_img(
        osgk_path, fsl_reference, interpolation="nearest"
    ).get_fdata().astype(int)
    striatum_fsl = osgk > 0
    affine_fsl = fsl_reference.affine

    def absolute_x(label_mask: np.ndarray) -> float:
        indices = np.argwhere(label_mask)[:, 0]
        return float(np.abs(affine_fsl[0, 0] * indices + affine_fsl[0, 3]).mean())

    label_distance = {
        label: absolute_x(osgk == label)
        for label in (1, 2, 3)
        if (osgk == label).any()
    }
    putamen_fsl = osgk == max(label_distance, key=label_distance.get)
    axis_zero = np.arange(osgk.shape[0])[:, None, None]
    x_mm = affine_fsl[0, 0] * axis_zero + affine_fsl[0, 3]
    left_fsl = striatum_fsl & (x_mm < 0)
    right_fsl = striatum_fsl & (x_mm > 0)

    excess_hmba = frame.to_h(excess_fsl)
    hmba_axis_zero = np.arange(frame.striatum.shape[0])[:, None, None]
    hmba_x_mm = frame.affine[0, 0] * hmba_axis_zero + frame.affine[0, 3]
    hemisphere_masks = {
        "MA": frame.striatum & (hmba_x_mm < 0),
        "LA": frame.striatum & (hmba_x_mm > 0),
    }
    concordance = {}
    for tag, hemisphere_mask in hemisphere_masks.items():
        valid_mask = hemisphere_mask & np.isfinite(excess_hmba)
        depletion = excess_hmba[valid_mask]
        enrichment = {
            group: frame.noncentered_fields[group][valid_mask]
            for group in GROUP_ORDER
        }
        concordance[tag] = _block_bootstrap_correspondence(
            enrichment,
            depletion,
            valid_mask,
            frame.affine,
            seed=seed,
            n_bootstrap=n_bootstrap,
        )

    return {
        "excess_fsl_canon": excess_fsl,
        "PUT_fsl": putamen_fsl,
        "STR_fsl": striatum_fsl,
        "left_fsl": left_fsl,
        "right_fsl": right_fsl,
        "aff_fsl": affine_fsl,
        "fsl_reference": fsl_reference,
        "MA": concordance["MA"],
        "LA": concordance["LA"],
        "n_pairs": len(gba_volumes),
        "input_volume_count": len(volume_paths),
        "input_volume_set_sha256": _aggregate_content_hash(volume_paths),
        "input_paths": [
            cohort_path,
            scan_path,
            osgk_path,
            frame.fsl_2mm_path,
            frame.locked_bins_path,
            frame.annotation_path,
        ],
    }
