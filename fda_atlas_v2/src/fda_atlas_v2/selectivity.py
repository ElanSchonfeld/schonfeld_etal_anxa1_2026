"""Genome-wide donor-paired selectivity against the strongest population competitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PopulationExpression:
    expression: np.ndarray
    detection: np.ndarray
    metadata: pd.DataFrame
    genes: tuple[str, ...]
    matrix_representation: str


def _validate(data: PopulationExpression) -> None:
    expression = np.asarray(data.expression, dtype=float)
    detection = np.asarray(data.detection, dtype=float)
    if expression.ndim != 2 or detection.shape != expression.shape:
        raise ValueError("expression and detection matrices must have identical two-dimensional shape")
    if len(data.metadata) != expression.shape[0] or len(data.genes) != expression.shape[1]:
        raise ValueError("population metadata or genes do not align with matrices")
    required = {"donor_id", "population_id", "n_cells"}
    if missing := sorted(required - set(data.metadata.columns)):
        raise ValueError(f"population metadata is missing columns: {missing}")
    if not np.isfinite(expression).all():
        raise ValueError("expression values must be finite depth-corrected values")
    if not np.isfinite(detection).all() or ((detection < 0) | (detection > 1)).any():
        raise ValueError("detection values must be finite fractions in [0, 1]")
    if len(data.genes) != len(set(data.genes)):
        raise ValueError("gene identifiers must be unique")
    if data.matrix_representation != "log2_cpm_from_raw_donor_pseudobulk":
        raise ValueError(
            "selectivity requires log2 CPM computed from raw donor pseudobulk counts"
        )


def _paired_rows(
    metadata: pd.DataFrame,
    focal_population: str,
    comparator_population: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    focal = metadata[metadata["population_id"].astype(str).eq(focal_population)].copy()
    comparator = metadata[
        metadata["population_id"].astype(str).eq(comparator_population)
    ].copy()
    if focal["donor_id"].astype(str).duplicated().any() or comparator[
        "donor_id"
    ].astype(str).duplicated().any():
        raise ValueError("each donor must have one pseudobulk per population")
    focal_index = dict(zip(focal["donor_id"].astype(str), focal.index))
    comparator_index = dict(zip(comparator["donor_id"].astype(str), comparator.index))
    donors = sorted(set(focal_index) & set(comparator_index))
    return (
        np.asarray([focal_index[donor] for donor in donors], dtype=int),
        np.asarray([comparator_index[donor] for donor in donors], dtype=int),
        donors,
    )


def moderate_selectivity_results(
    frame: pd.DataFrame,
    *,
    confidence_z: float = 1.96,
) -> pd.DataFrame:
    """Finalize selection-aware effects and compute BH FDR across all genes."""

    required = {"raw_effect", "raw_standard_error", "p_value", "status"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"selectivity results are missing columns: {missing}")
    if confidence_z <= 0:
        raise ValueError("confidence z must be positive")
    output = frame.copy()
    raw_effect = pd.to_numeric(output["raw_effect"], errors="coerce").to_numpy()
    unsupported = ~output["status"].astype(str).eq("estimated").to_numpy()
    effect = raw_effect.copy()
    effect[unsupported] = np.nan
    output["effect"] = effect
    output.loc[unsupported, ["ci_low", "ci_high", "p_value"]] = np.nan
    p_values = pd.to_numeric(output["p_value"], errors="coerce").to_numpy()
    eligible = ~unsupported & np.isfinite(p_values)
    q_values = np.full(len(output), np.nan)
    if eligible.any():
        indices = np.flatnonzero(eligible)
        order = indices[np.argsort(p_values[indices], kind="stable")]
        ranks = np.arange(1, len(order) + 1, dtype=float)
        adjusted = p_values[order] * len(order) / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q_values[order] = np.minimum(adjusted, 1.0)
    output["q_value"] = q_values
    return output


def strongest_competitor_selectivity(
    data: PopulationExpression,
    *,
    focal_population: str,
    comparator_populations: Sequence[str],
    minimum_common_donors: int = 3,
    minimum_cells_per_donor_population: int = 10,
    minimum_focal_detection: float = 0.05,
    confidence_z: float = 1.96,
    apply_moderation: bool = True,
) -> pd.DataFrame:
    """Estimate every gene before any FDA-target filtering or ranking."""

    _validate(data)
    if minimum_common_donors < 3:
        raise ValueError("minimum common donors must be at least three")
    if minimum_cells_per_donor_population <= 0:
        raise ValueError("minimum cells must be positive")
    if not 0 <= minimum_focal_detection <= 1:
        raise ValueError("minimum focal detection must be in [0, 1]")
    metadata = data.metadata.reset_index(drop=True).copy()
    expression = np.asarray(data.expression, dtype=float)
    detection = np.asarray(data.detection, dtype=float)
    comparators = sorted(set(str(value) for value in comparator_populations))
    if focal_population in comparators:
        raise ValueError("focal population cannot be its own comparator")
    n_genes = expression.shape[1]
    supported_metadata = metadata[
        metadata["n_cells"].astype(int).ge(minimum_cells_per_donor_population)
    ]
    focal_supported = supported_metadata[
        supported_metadata["population_id"].astype(str).eq(focal_population)
    ]
    if focal_supported["donor_id"].astype(str).duplicated().any():
        raise ValueError("focal population has duplicate donor pseudobulks")
    records = []
    for comparator in comparators:
        focal_rows, comparator_rows, donors = _paired_rows(
            supported_metadata, focal_population, comparator
        )
        if len(donors) < minimum_common_donors:
            continue
        focal_cells = metadata.loc[focal_rows, "n_cells"].astype(int).to_numpy()
        comparator_cells = metadata.loc[comparator_rows, "n_cells"].astype(int).to_numpy()
        paired_effect = expression[focal_rows] - expression[comparator_rows]
        raw_effect = paired_effect.mean(axis=0)
        raw_se = paired_effect.std(axis=0, ddof=1) / np.sqrt(len(donors))
        focal_detection = detection[focal_rows].mean(axis=0)
        detection_difference = (
            detection[focal_rows] - detection[comparator_rows]
        ).mean(axis=0)
        records.append(
            {
                "population_id": comparator,
                "raw_effect": raw_effect,
                "raw_standard_error": raw_se,
                "paired_effects": paired_effect,
                "donors": tuple(donors),
                "focal_detection": focal_detection,
                "detection_difference": detection_difference,
                "n_common_donors": len(donors),
                "focal_n_cells": int(focal_cells.sum()),
                "comparator_n_cells": int(comparator_cells.sum()),
            }
        )
    if not records:
        return pd.DataFrame(
            {
                "gene_id": list(data.genes),
                "effect": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "detection_difference": np.nan,
                "competitor_id": None,
                "n_common_donors": 0,
                "focal_n_cells": 0,
                "competitor_n_cells": 0,
                "focal_detection": np.nan,
                "raw_effect": np.nan,
                "raw_standard_error": np.nan,
                "loo_raw_effect_min": np.nan,
                "loo_raw_effect_max": np.nan,
                "matrix_representation": data.matrix_representation,
                "strongest_competitor_rule": "minimum donor-paired raw effect",
                "interval_method": "simultaneous Bonferroni t interval for the minimum contrast",
                "selection_uncertainty_status": "no eligible comparator",
                "status": "insufficient_comparator_support",
                "p_value": np.nan,
                "q_value": np.nan,
                "competitor_loo_selection_fraction": np.nan,
                "common_donor_ids": "",
                "n_eligible_comparators": 0,
            }
        )

    paired_raw_effects = np.vstack([record["raw_effect"] for record in records])
    strongest = np.argmin(paired_raw_effects, axis=0)
    column = np.arange(n_genes)

    def choose(key, dtype=float):
        values = np.vstack(
            [
                np.full(n_genes, record[key], dtype=dtype)
                if np.isscalar(record[key])
                else np.asarray(record[key], dtype=dtype)
                for record in records
            ]
        )
        return values[strongest, column]

    raw_effect = choose("raw_effect")
    raw_se = choose("raw_standard_error")
    all_donors = sorted(
        {donor for record in records for donor in record["donors"]}
    )
    loo_selected_effects = []
    loo_selected_competitors = []
    for omitted_donor in all_donors:
        comparator_effects = []
        for record in records:
            keep = np.asarray(
                [donor != omitted_donor for donor in record["donors"]], dtype=bool
            )
            if int(keep.sum()) < 2:
                comparator_effects.append(np.full(n_genes, np.inf))
            else:
                comparator_effects.append(record["paired_effects"][keep].mean(axis=0))
        stacked = np.vstack(comparator_effects)
        loo_selected_effects.append(stacked.min(axis=0))
        loo_selected_competitors.append(stacked.argmin(axis=0))
    loo_selected_effects = np.vstack(loo_selected_effects)
    loo_selected_competitor = np.vstack(loo_selected_competitors)
    loo_raw_effect_min = loo_selected_effects.min(axis=0)
    loo_raw_effect_max = loo_selected_effects.max(axis=0)
    competitor_loo_selection_fraction = (
        loo_selected_competitor == strongest[np.newaxis, :]
    ).mean(axis=0)
    focal_detection = choose("focal_detection")
    detection_difference = choose("detection_difference")
    n_common_donors = choose("n_common_donors", dtype=int)
    focal_n_cells = choose("focal_n_cells", dtype=int)
    comparator_n_cells = choose("comparator_n_cells", dtype=int)
    competitor_ids = np.asarray(
        [records[index]["population_id"] for index in strongest], dtype=object
    )
    common_donor_ids = np.asarray(
        [" | ".join(records[index]["donors"]) for index in strongest],
        dtype=object,
    )
    n_comparators = len(records)
    alpha = 2 * (1 - stats.norm.cdf(confidence_z))
    simultaneous_lower = []
    simultaneous_upper = []
    one_sided_p = []
    for record in records:
        effect_values = np.asarray(record["raw_effect"], dtype=float)
        se_values = np.asarray(record["raw_standard_error"], dtype=float)
        degrees = int(record["n_common_donors"]) - 1
        critical = stats.t.ppf(1 - alpha / (2 * n_comparators), degrees)
        simultaneous_lower.append(effect_values - critical * se_values)
        simultaneous_upper.append(effect_values + critical * se_values)
        statistic = np.divide(
            effect_values,
            se_values,
            out=np.full_like(effect_values, np.inf),
            where=se_values > 0,
        )
        p_value = stats.t.sf(statistic, degrees)
        p_value[(se_values == 0) & (effect_values <= 0)] = 1.0
        one_sided_p.append(p_value)
    ci_low = np.vstack(simultaneous_lower).min(axis=0)
    ci_high = np.vstack(simultaneous_upper).min(axis=0)
    p_value = np.vstack(one_sided_p).max(axis=0)
    status = np.full(n_genes, "estimated", dtype=object)
    unsupported = focal_detection < minimum_focal_detection
    status[unsupported] = "insufficient_focal_detection"
    result = pd.DataFrame(
        {
            "gene_id": list(data.genes),
            "effect": np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "detection_difference": detection_difference,
            "competitor_id": competitor_ids,
            "n_common_donors": n_common_donors,
            "focal_n_cells": focal_n_cells,
            "competitor_n_cells": comparator_n_cells,
            "focal_detection": focal_detection,
            "raw_effect": raw_effect,
            "raw_standard_error": raw_se,
            "loo_raw_effect_min": loo_raw_effect_min,
            "loo_raw_effect_max": loo_raw_effect_max,
            "p_value": p_value,
            "q_value": np.nan,
            "competitor_loo_selection_fraction": competitor_loo_selection_fraction,
            "common_donor_ids": common_donor_ids,
            "n_eligible_comparators": n_comparators,
            "matrix_representation": data.matrix_representation,
            "strongest_competitor_rule": (
                "minimum donor-paired effect over comparator-specific eligible donor sets"
            ),
            "interval_method": "simultaneous Bonferroni t interval for the minimum contrast",
            "selection_uncertainty_status": "competitor reselected in leave-one-donor-out analyses",
            "status": status,
        }
    )
    if apply_moderation:
        return moderate_selectivity_results(result, confidence_z=confidence_z)
    return result
