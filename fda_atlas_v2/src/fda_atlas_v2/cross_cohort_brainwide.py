"""Brain-wide selectivity against a shared cross-cohort reference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .selectivity import moderate_selectivity_results


def cross_cohort_brainwide_selectivity(
    focal_expression: np.ndarray,
    focal_detection: np.ndarray,
    *,
    genes: Sequence[str],
    focal_donor_ids: Sequence[str],
    focal_n_cells: int,
    reference_expression: np.ndarray,
    reference_detection: np.ndarray,
    reference_population_ids: Sequence[str],
    reference_n_cells: Sequence[int],
    reference_donor_counts: Sequence[int],
    minimum_focal_detection: float = 0.05,
    confidence_z: float = 1.96,
) -> pd.DataFrame:
    focal = np.asarray(focal_expression, dtype=float)
    focal_detect = np.asarray(focal_detection, dtype=float)
    reference = np.asarray(reference_expression, dtype=float)
    reference_detect = np.asarray(reference_detection, dtype=float)
    donor_ids = tuple(str(value) for value in focal_donor_ids)
    population_ids = tuple(str(value) for value in reference_population_ids)
    genes = tuple(str(value) for value in genes)
    reference_cells = np.asarray(reference_n_cells, dtype=int)
    reference_donors = np.asarray(reference_donor_counts, dtype=int)

    if focal.ndim != 2 or focal_detect.shape != focal.shape:
        raise ValueError("focal expression and detection must have identical 2D shape")
    if reference.ndim != 2 or reference_detect.shape != reference.shape:
        raise ValueError("reference expression and detection must have identical 2D shape")
    if focal.shape[1] != len(genes) or reference.shape[1] != len(genes):
        raise ValueError("gene identifiers do not align with focal and reference matrices")
    if focal.shape[0] != len(donor_ids) or len(set(donor_ids)) != len(donor_ids):
        raise ValueError("focal donor identifiers must be unique and align with rows")
    if reference.shape[0] != len(population_ids):
        raise ValueError("reference population identifiers do not align with rows")
    if len(set(population_ids)) != len(population_ids):
        raise ValueError("reference population identifiers must be unique")
    if len(reference_cells) != len(population_ids) or len(reference_donors) != len(
        population_ids
    ):
        raise ValueError("reference support vectors do not align with populations")
    if focal.shape[0] < 3:
        raise ValueError("cross-cohort selectivity requires at least three focal donors")
    if not 0 <= minimum_focal_detection <= 1:
        raise ValueError("minimum focal detection must be in [0, 1]")
    if (
        not np.isfinite(focal).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(focal_detect).all()
        or not np.isfinite(reference_detect).all()
    ):
        raise ValueError("cross-cohort expression and detection must be finite")
    if (
        (focal_detect < 0).any()
        or (focal_detect > 1).any()
        or (reference_detect < 0).any()
        or (reference_detect > 1).any()
    ):
        raise ValueError("cross-cohort detection values must be fractions")
    if (reference_cells <= 0).any() or (reference_donors <= 0).any():
        raise ValueError("reference populations require positive cell and donor support")

    n_donors, n_genes = focal.shape
    differences = focal[np.newaxis, :, :] - reference[:, np.newaxis, :]
    effects = differences.mean(axis=1)
    errors = differences.std(axis=1, ddof=1) / np.sqrt(n_donors)
    strongest = np.argmin(effects, axis=0)
    columns = np.arange(n_genes)
    raw_effect = effects[strongest, columns]
    raw_error = errors[strongest, columns]
    focal_detection_mean = focal_detect.mean(axis=0)
    detection_difference = (
        focal_detection_mean - reference_detect[strongest, columns]
    )

    alpha = 2 * (1 - stats.norm.cdf(confidence_z))
    degrees = n_donors - 1
    critical = stats.t.ppf(
        1 - alpha / (2 * len(population_ids)),
        degrees,
    )
    lower = effects - critical * errors
    upper = effects + critical * errors
    ci_low = lower.min(axis=0)
    ci_high = upper.min(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = np.divide(
            effects,
            errors,
            out=np.full_like(effects, np.inf),
            where=errors > 0,
        )
    p_values = stats.t.sf(statistics, degrees)
    p_values[(errors == 0) & (effects <= 0)] = 1.0
    p_value = p_values.max(axis=0)

    loo_effects = []
    loo_competitors = []
    for omitted in range(n_donors):
        keep = np.arange(n_donors) != omitted
        loo = differences[:, keep, :].mean(axis=1)
        loo_effects.append(loo.min(axis=0))
        loo_competitors.append(loo.argmin(axis=0))
    loo_effects = np.vstack(loo_effects)
    loo_competitors = np.vstack(loo_competitors)

    status = np.full(n_genes, "estimated", dtype=object)
    status[focal_detection_mean < minimum_focal_detection] = (
        "insufficient_focal_detection"
    )
    result = pd.DataFrame(
        {
            "gene_id": genes,
            "effect": np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "detection_difference": detection_difference,
            "competitor_id": np.asarray(population_ids, dtype=object)[strongest],
            "n_common_donors": n_donors,
            "focal_n_cells": int(focal_n_cells),
            "competitor_n_cells": reference_cells[strongest],
            "focal_detection": focal_detection_mean,
            "raw_effect": raw_effect,
            "raw_standard_error": raw_error,
            "loo_raw_effect_min": loo_effects.min(axis=0),
            "loo_raw_effect_max": loo_effects.max(axis=0),
            "matrix_representation": (
                "focal donor log2 CPM from raw counts versus donor-balanced "
                "cross-cohort non-DA reference"
            ),
            "strongest_competitor_rule": (
                "minimum effect across fixed donor-balanced non-DA reference regions"
            ),
            "interval_method": (
                "simultaneous Bonferroni t interval across reference regions; "
                "not a ranking gate"
            ),
            "selection_uncertainty_status": (
                "leave-one-focal-donor-out competitor stability"
            ),
            "status": status,
            "p_value": p_value,
            "q_value": np.nan,
            "competitor_loo_selection_fraction": (
                loo_competitors == strongest[np.newaxis, :]
            ).mean(axis=0),
            "common_donor_ids": " | ".join(donor_ids),
            "n_eligible_comparators": len(population_ids),
            "reference_n_donors": reference_donors[strongest],
            "reference_is_donor_matched": False,
        }
    )
    return moderate_selectivity_results(result, confidence_z=confidence_z)
