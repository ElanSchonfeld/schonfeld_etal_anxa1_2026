"""Donor-aware scDRS population effects and conservative lead sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


CELL_COLUMNS = {
    "donor_id",
    "study",
    "condition",
    "population_id",
    "score",
}

HUMAN_INTEGRATED_CONTROL_CELLS = {"Kamath": 15_684, "Siletti": 823}
HUMAN_INTEGRATED_CONTROL_DONORS = {"Kamath": 8, "Siletti": 3}


def validated_human_integrated_control_mask(
    metadata: pd.DataFrame,
    *,
    donor_column: str = "donor",
    study_column: str = "study",
    condition_column: str = "disease_status",
) -> pd.Series:
    required = {donor_column, study_column, condition_column}
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"integrated human metadata lacks columns: {missing}")
    study = metadata[study_column].astype(str)
    condition = metadata[condition_column].astype(str)
    mask = (study.eq("Kamath") & condition.eq("Ctrl")) | (
        study.eq("Siletti") & condition.eq("Control")
    )
    selected = metadata.loc[mask]
    cell_counts = (
        selected[study_column].astype(str).value_counts(sort=False).to_dict()
    )
    donor_counts = (
        selected.assign(_study=selected[study_column].astype(str))
        .groupby("_study", observed=True)[donor_column]
        .nunique()
        .to_dict()
    )
    if cell_counts != HUMAN_INTEGRATED_CONTROL_CELLS:
        raise ValueError(
            "integrated human control cell counts disagree with the frozen cohort: "
            f"{cell_counts}"
        )
    if donor_counts != HUMAN_INTEGRATED_CONTROL_DONORS:
        raise ValueError(
            "integrated human control donor counts disagree with the frozen cohort: "
            f"{donor_counts}"
        )
    return mask


def _mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(clean))
    if len(clean) == 1:
        return mean, np.nan, np.nan, np.nan
    standard_error = float(stats.sem(clean))
    if standard_error == 0:
        return mean, 0.0, mean, mean
    critical = float(stats.t.ppf((1 + confidence) / 2, len(clean) - 1))
    return (
        mean,
        standard_error,
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def _loo_range(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan, np.nan
    estimates = [float(np.mean(np.delete(clean, index))) for index in range(len(clean))]
    return min(estimates), max(estimates)


def _stratified_mean_ci(
    values: pd.Series,
    strata: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Estimate the donor-weighted marginal mean with stratified uncertainty."""

    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "stratum": strata.astype(str),
        }
    ).dropna(subset=["value"])
    if frame.empty:
        return np.nan, np.nan, np.nan, np.nan
    marginal_mean = float(frame["value"].mean())
    if len(frame) == 1:
        return marginal_mean, np.nan, np.nan, np.nan
    grouped = list(frame.groupby("stratum", observed=True, sort=True))
    if len(grouped) == 1 or any(len(block) < 2 for _, block in grouped):
        return _mean_ci(frame["value"], confidence=confidence)
    n_total = len(frame)
    variance_components = []
    denominator_components = []
    for _, block in grouped:
        n_stratum = len(block)
        weight = n_stratum / n_total
        component = weight**2 * float(block["value"].var(ddof=1)) / n_stratum
        variance_components.append(component)
        denominator_components.append(component**2 / (n_stratum - 1))
    variance = float(sum(variance_components))
    standard_error = float(np.sqrt(variance))
    if standard_error == 0:
        return marginal_mean, 0.0, marginal_mean, marginal_mean
    denominator = float(sum(denominator_components))
    degrees = variance**2 / denominator if denominator > 0 else len(frame) - len(grouped)
    critical = float(stats.t.ppf((1 + confidence) / 2, degrees))
    return (
        marginal_mean,
        standard_error,
        marginal_mean - critical * standard_error,
        marginal_mean + critical * standard_error,
    )


def _stratified_loo_range(
    values: pd.Series, strata: pd.Series
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "stratum": strata.astype(str),
        }
    ).dropna(subset=["value"])
    if len(frame) < 2:
        return np.nan, np.nan
    estimates = [
        _stratified_mean_ci(
            frame.drop(frame.index[index])["value"],
            frame.drop(frame.index[index])["stratum"],
        )[0]
        for index in range(len(frame))
    ]
    return float(min(estimates)), float(max(estimates))


def estimate_population_effects(
    cell_scores: pd.DataFrame,
    *,
    min_cells_per_donor_population: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate equal-donor relative scDRS effects."""

    missing = sorted(CELL_COLUMNS - set(cell_scores.columns))
    if missing:
        raise ValueError(f"missing cell-score columns: {missing}")
    if min_cells_per_donor_population < 1:
        raise ValueError("min_cells_per_donor_population must be positive")

    columns = sorted(CELL_COLUMNS | ({"source_donor_id"} if "source_donor_id" in cell_scores else set()))
    frame = cell_scores.loc[:, columns].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["donor_id", "population_id", "score"])

    donor_metadata = frame.groupby("donor_id", observed=True).agg(
        n_studies=("study", "nunique"),
        n_conditions=("condition", "nunique"),
        **(
            {"n_source_donors": ("source_donor_id", "nunique")}
            if "source_donor_id" in frame
            else {}
        ),
    )
    invalid = donor_metadata[
        donor_metadata["n_studies"].ne(1)
        | donor_metadata["n_conditions"].ne(1)
        | donor_metadata.get("n_source_donors", pd.Series(1, index=donor_metadata.index)).ne(1)
    ]
    if not invalid.empty:
        raise ValueError("each donor_id must map to one study and one condition")

    aggregations = {
            "score": ("score", "mean"),
            "n_cells": ("score", "size"),
            "study": ("study", "first"),
            "condition": ("condition", "first"),
    }
    if "source_donor_id" in frame:
        aggregations["source_donor_id"] = ("source_donor_id", "first")
    donor_means = (
        frame.groupby(["donor_id", "population_id"], observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    donor_means = donor_means[
        donor_means["n_cells"].ge(min_cells_per_donor_population)
    ].copy()
    donor_sum = donor_means.groupby("donor_id", observed=True)["score"].transform(
        "sum"
    )
    donor_population_n = donor_means.groupby("donor_id", observed=True)[
        "score"
    ].transform("size")
    donor_means = donor_means[donor_population_n.gt(1)].copy()
    donor_means["relative_effect"] = donor_means["score"] - (
        (donor_sum - donor_means["score"]) / (donor_population_n - 1)
    )
    donor_means["donor_weight"] = 1.0

    rows = []
    for population_id, block in donor_means.groupby(
        "population_id", observed=True, sort=True
    ):
        effect, standard_error, ci_low, ci_high = _stratified_mean_ci(
            block["relative_effect"], block["study"]
        )
        loo_min, loo_max = _stratified_loo_range(
            block["relative_effect"], block["study"]
        )
        n_donors = int(block["donor_id"].nunique())
        study_counts = block["study"].astype(str).value_counts()
        if len(study_counts) == 1:
            inference_model = "equal-donor marginal effect with single-study variance"
        elif study_counts.ge(2).all():
            inference_model = (
                "equal-donor marginal effect with study-stratified variance"
            )
        else:
            inference_model = (
                "equal-donor marginal effect with pooled variance because a study "
                "has fewer than two contributing donors"
            )
        rows.append(
            {
                "population_id": str(population_id),
                "effect": effect,
                "standard_error": standard_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_donors": n_donors,
                "n_studies": int(block["study"].nunique()),
                "n_cells": int(block["n_cells"].sum()),
                "min_cells_per_donor": int(block["n_cells"].min()),
                "median_cells_per_donor": float(block["n_cells"].median()),
                "positive_donor_fraction": float(block["relative_effect"].gt(0).mean()),
                "loo_effect_min": loo_min,
                "loo_effect_max": loo_max,
                "status": "estimated" if n_donors >= 3 else "limited_donors",
                "inference_model": inference_model,
            }
        )
    effects = pd.DataFrame(rows).sort_values(
        ["effect", "population_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    return effects, donor_means.reset_index(drop=True)


def select_population_lead_set(
    effects: pd.DataFrame,
    donor_means: pd.DataFrame,
    *,
    min_common_donors: int = 3,
    min_donors: int = 3,
) -> pd.DataFrame:
    """Select a leader and retain populations not separated from it."""

    if effects.empty:
        return pd.DataFrame(
            columns=[
                "population_id",
                "lead_status",
                "leader_id",
                "delta_from_leader",
                "delta_ci_low",
                "delta_ci_high",
                "n_common_donors",
                "selection_reason",
            ]
        )
    eligible_effects = effects[effects["n_donors"].ge(min_donors)].copy()
    if eligible_effects.empty:
        return pd.DataFrame(
            {
                "population_id": effects["population_id"].astype(str),
                "lead_status": "insufficient_data",
                "leader_id": "unavailable",
                "delta_from_leader": np.nan,
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "n_common_donors": effects["n_donors"].astype(int),
                "selection_reason": "fewer than the required independent donors",
            }
        )
    leader_id = str(
        eligible_effects.sort_values(
            ["effect", "n_donors", "population_id"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]["population_id"]
    )
    wide = donor_means.pivot(
        index="donor_id", columns="population_id", values="score"
    )
    donor_study = donor_means[["donor_id", "study"]].drop_duplicates()
    if donor_study["donor_id"].astype(str).duplicated().any():
        raise ValueError("lead-set donor maps to multiple study strata")
    study_by_donor = donor_study.set_index("donor_id")["study"].astype(str)
    rows = []
    for population_id in effects["population_id"].astype(str):
        population_donors = int(
            effects.loc[
                effects["population_id"].astype(str).eq(population_id), "n_donors"
            ].iloc[0]
        )
        if population_donors < min_donors:
            rows.append(
                {
                    "population_id": population_id,
                    "lead_status": "insufficient_data",
                    "leader_id": leader_id,
                    "delta_from_leader": np.nan,
                    "delta_ci_low": np.nan,
                    "delta_ci_high": np.nan,
                    "n_common_donors": population_donors,
                    "selection_reason": "fewer than the required independent donors",
                }
            )
            continue
        if population_id == leader_id:
            rows.append(
                {
                    "population_id": population_id,
                    "lead_status": "leader",
                    "leader_id": leader_id,
                    "delta_from_leader": 0.0,
                    "delta_ci_low": 0.0,
                    "delta_ci_high": 0.0,
                    "n_common_donors": int(wide[leader_id].notna().sum()),
                    "selection_reason": "highest donor-aware effect",
                }
            )
            continue

        paired = wide[[leader_id, population_id]].dropna()
        delta = paired[population_id] - paired[leader_id]
        mean, _, ci_low, ci_high = _stratified_mean_ci(
            delta, study_by_donor.reindex(delta.index)
        )
        n_common = len(paired)
        if n_common < min_common_donors:
            lead_status = "lead_set"
            reason = "insufficient paired donors to separate from leader"
        elif ci_high >= 0:
            lead_status = "lead_set"
            reason = "paired confidence interval overlaps leader"
        else:
            lead_status = "separated"
            reason = "paired confidence interval is below leader"
        rows.append(
            {
                "population_id": population_id,
                "lead_status": lead_status,
                "leader_id": leader_id,
                "delta_from_leader": mean,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "n_common_donors": n_common,
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def set_primary_cohort_flags(
    frame: pd.DataFrame, expected_primary: dict[str, str]
) -> pd.DataFrame:
    """Set only primary-cohort flags after proving every cohort is present."""

    required = {"species", "analysis_cohort", "is_primary_cohort"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"population table lacks primary-cohort columns: {missing}")
    observed_species = set(frame["species"].astype(str))
    if observed_species != set(expected_primary):
        raise ValueError(
            "population table species disagree with expected primary cohorts: "
            f"{sorted(observed_species)}"
        )
    output = frame.copy()
    output["is_primary_cohort"] = False
    for species, cohort in expected_primary.items():
        species_rows = output["species"].astype(str).eq(species)
        cohort_rows = output["analysis_cohort"].astype(str).eq(cohort)
        if not (species_rows & cohort_rows).any():
            raise ValueError(f"{species} primary cohort is absent: {cohort}")
        output.loc[species_rows & cohort_rows, "is_primary_cohort"] = True
    primary = output.loc[output["is_primary_cohort"].astype(bool)]
    observed = (
        primary.groupby("species", observed=True)["analysis_cohort"]
        .unique()
        .to_dict()
    )
    normalized = {species: values.tolist() for species, values in observed.items()}
    if normalized != {
        species: [cohort] for species, cohort in expected_primary.items()
    }:
        raise ValueError(f"primary-cohort flag repair failed: {normalized}")
    return output
