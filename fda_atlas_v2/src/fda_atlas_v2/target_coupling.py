"""Donor-level target to leave-out scDRS coupling without cell pseudoreplication."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats


INPUT_COLUMNS = {
    "species", "trait_id", "population_id", "target_id", "method",
    "donor_id", "target_expression", "score", "n_cells",
    "study", "condition",
    "stratum_id", "analysis_partition", "analysis_role",
    "gene_mapping_status", "expression_mapping_available",
    "score_availability_status",
    "circularity_status", "is_primary_method", "expression_representation",
    "score_representation",
}

METHOD_CIRCULARITY = {
    "leave_target_out": {
        "target_removed",
        "target_not_in_gene_set",
        "target_not_scored_in_corrected_baseline",
        "not_evaluated_target_aware_leaveout",
    },
    "leave_ld_block_out": {"target_ld_block_removed", "ld_block_mapping_unavailable"},
    "leave_chromosome_out": {"target_chromosome_removed"},
}


def unavailable_rhesus_coupling_rows(
    selectivity: pd.DataFrame,
    *,
    release_id: str,
) -> pd.DataFrame:
    """Materialize explicit unavailable-rhesus coverage rows."""

    required = {
        "species", "trait_id", "population_id", "target_id", "eligible",
        "gene_mapping_status", "within_da_status", "within_da_focal_n_cells",
        "within_da_n_common_donors",
    }
    if missing := sorted(required - set(selectivity.columns)):
        raise ValueError(f"rhesus coupling coverage lacks columns: {missing}")
    source = selectivity[selectivity["species"].astype(str).eq("macaque")].copy()
    rows = []
    for row in source.itertuples(index=False):
        donors = pd.to_numeric(row.within_da_n_common_donors, errors="coerce")
        cells = pd.to_numeric(row.within_da_focal_n_cells, errors="coerce")
        n_donors = int(donors) if np.isfinite(donors) else 0
        expression_available = bool(row.eligible) and str(row.within_da_status) in {
            "estimated", "insufficient_focal_detection",
        }
        study_counts = {"Allen HMBA": n_donors} if n_donors > 0 else {}
        rows.append(
            {
                "release_id": release_id,
                "species": "macaque",
                "trait_id": str(row.trait_id),
                "population_id": str(row.population_id),
                "target_id": str(row.target_id),
                "analysis_partition": "allen_hmba_rhesus_control",
                "method": "leave_target_out",
                "analysis_role": "primary",
                "gene_mapping_status": str(row.gene_mapping_status),
                "expression_mapping_available": expression_available,
                "score_availability_status": "target_aware_leaveout_not_computed",
                "is_primary_method": True,
                "estimate": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "n_donors": n_donors,
                "n_cells": int(cells) if np.isfinite(cells) else 0,
                "n_studies": len(study_counts),
                "study_donor_counts_json": json.dumps(
                    study_counts, separators=(",", ":"), sort_keys=True
                ),
                "n_strata": len(study_counts),
                "residual_degrees_of_freedom": 0,
                "analysis_strata": "Allen HMBA|Control|primary" if study_counts else "",
                "positive_donor_fraction": np.nan,
                "loo_estimate_min": np.nan,
                "loo_estimate_max": np.nan,
                "circularity_status": "not_evaluated_target_aware_leaveout",
                "stability_status": "not_estimated_target_aware_leaveout",
                "analysis_unit": (
                    "Allen HMBA Macaca mulatta donor by HMoE population; target-aware coupling unavailable"
                ),
                "expression_representation": (
                    "log2_cpm_from_raw_rhesus_donor_population_pseudobulk_prior_count_0.5"
                ),
                "score_representation": (
                    "rhesus scDRS available, but per-target leaveout scores are not materialized"
                ),
                "status": "unavailable_score",
                "affects_dual_selectivity_rank": False,
            }
        )
    result = pd.DataFrame(rows)
    validate_target_coupling_semantics(result)
    return result


def validate_selectivity_primary_coverage(
    coupling: pd.DataFrame,
    selectivity: pd.DataFrame,
    *,
    release_id: str,
) -> None:
    """Require one primary coupling result for every released selectivity key."""

    keys = ["release_id", "species", "trait_id", "population_id", "target_id"]
    for label, frame in (("coupling", coupling), ("selectivity", selectivity)):
        if missing := sorted(set(keys) - set(frame.columns)):
            raise ValueError(f"{label} coverage input is missing columns: {missing}")
        observed_release = set(frame["release_id"].dropna().astype(str))
        if observed_release != {str(release_id)} or frame["release_id"].isna().any():
            raise ValueError(f"{label} coverage release ID differs from {release_id}")
    expected = selectivity[keys].drop_duplicates()
    if len(expected) != len(selectivity):
        raise ValueError("selectivity coverage keys are duplicated")
    primary = coupling.loc[
        coupling["analysis_role"].astype(str).eq("primary")
        & coupling["is_primary_method"].astype(bool),
        keys,
    ]
    if primary.duplicated().any():
        raise ValueError("coupling has duplicate primary rows for a selectivity key")
    missing = expected.merge(primary, on=keys, how="left", indicator=True)
    missing_n = int(missing["_merge"].ne("both").sum())
    extra = primary.merge(expected, on=keys, how="left", indicator=True)
    extra_n = int(extra["_merge"].ne("both").sum())
    if missing_n or extra_n:
        raise ValueError(
            "coupling primary-key coverage differs from selectivity: "
            f"missing={missing_n}, extra={extra_n}"
        )


def _bh(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return output
    order = finite.sort_values(kind="stable").index
    ranked = finite.loc[order].to_numpy(dtype=float)
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[order] = np.minimum(adjusted, 1.0)
    return output


def _residualize_strata(
    x: np.ndarray, y: np.ndarray, strata: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    x_residual = np.empty_like(x, dtype=float)
    y_residual = np.empty_like(y, dtype=float)
    unique = sorted(set(strata.astype(str)))
    for value in unique:
        selected = strata.astype(str) == value
        x_residual[selected] = x[selected] - x[selected].mean()
        y_residual[selected] = y[selected] - y[selected].mean()
    return x_residual, y_residual, len(unique)


def _association(
    x: np.ndarray, y: np.ndarray, *, residual_df: int
) -> tuple[float, float, float, float, float]:
    x = (x - x.mean()) / x.std(ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    estimate = float(np.dot(x, y) / (len(x) - 1))
    estimate = float(np.clip(estimate, -1.0, 1.0))
    if residual_df < 1:
        raise ValueError("target coupling has no residual degrees of freedom")
    residual = y - estimate * x
    standard_error = float(
        np.sqrt(np.dot(residual, residual) / residual_df / np.dot(x, x))
    )
    critical = float(stats.t.ppf(0.975, residual_df))
    statistic = estimate / standard_error if standard_error > 0 else np.inf
    p_value = float(2 * stats.t.sf(abs(statistic), residual_df))
    return (
        estimate,
        standard_error,
        estimate - critical * standard_error,
        estimate + critical * standard_error,
        p_value,
    )


def estimate_target_trait_coupling(
    donor_summaries: pd.DataFrame,
    *,
    release_id: str,
    min_donors: int = 3,
    min_cells_per_donor: int = 10,
) -> pd.DataFrame:
    """Estimate equal-donor standardized associations within each DA population."""

    missing = sorted(INPUT_COLUMNS - set(donor_summaries.columns))
    if missing:
        raise ValueError(f"target coupling input is missing columns: {missing}")
    if min_donors < 3 or min_cells_per_donor < 1:
        raise ValueError("target coupling support thresholds are invalid")
    frame = donor_summaries.copy()
    frame["target_expression"] = pd.to_numeric(frame["target_expression"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["n_cells"] = pd.to_numeric(frame["n_cells"], errors="coerce")
    if (
        frame["study"].fillna("").astype(str).str.strip().eq("").any()
        or frame["condition"].fillna("").astype(str).str.strip().eq("").any()
    ):
        raise ValueError("target coupling requires explicit study and condition strata")
    key = [
        "species", "trait_id", "population_id", "target_id",
        "analysis_partition", "method", "donor_id",
    ]
    if frame.duplicated(key).any():
        raise ValueError("target coupling input has duplicate donor summaries")

    group_columns = [
        "species", "trait_id", "population_id", "target_id",
        "analysis_partition", "method",
    ]
    rows = []
    for group_key, block in frame.groupby(group_columns, observed=True, sort=True):
        metadata_columns = [
            "circularity_status", "is_primary_method", "expression_representation",
            "score_representation", "analysis_role",
            "gene_mapping_status", "expression_mapping_available",
            "score_availability_status",
        ]
        if any(block[column].nunique(dropna=False) != 1 for column in metadata_columns):
            raise ValueError("target coupling metadata differs across donor summaries")
        circularity = str(block["circularity_status"].iloc[0])
        method = str(group_key[-1])
        if circularity not in METHOD_CIRCULARITY.get(method, set()):
            raise ValueError(f"invalid circularity status for {method}: {circularity}")
        eligible = block[
            block["target_expression"].notna()
            & block["score"].notna()
            & block["n_cells"].ge(min_cells_per_donor)
        ].copy()
        n_donors = eligible["donor_id"].nunique()
        study_donor_counts = (
            eligible.groupby("study", observed=True)["donor_id"]
            .nunique()
            .sort_index()
            .astype(int)
            .to_dict()
        )
        integrated_human = (
            str(group_key[0]) == "human"
            and str(group_key[4]) == "human_integrated_control"
            and str(block["analysis_role"].iloc[0]) == "primary"
        )
        integrated_study_support = (
            set(study_donor_counts) == {"Kamath", "Siletti"}
            and all(count >= 2 for count in study_donor_counts.values())
        )
        status = "estimated"
        estimate = standard_error = ci_low = ci_high = p_value = np.nan
        loo_min = loo_max = positive_fraction = np.nan
        stability = "not_estimable"
        n_strata = eligible["stratum_id"].astype(str).nunique()
        residual_df = int(n_donors - n_strata - 1)
        if str(block["score_availability_status"].iloc[0]) != "available":
            status = "unavailable_score"
        elif not bool(block["expression_mapping_available"].iloc[0]):
            status = "unavailable_expression_mapping"
        elif n_donors < min_donors:
            status = "insufficient_donors"
        elif integrated_human and not integrated_study_support:
            status = "insufficient_study_support"
        else:
            x = eligible["target_expression"].to_numpy(dtype=float)
            y = eligible["score"].to_numpy(dtype=float)
            strata = eligible["stratum_id"].astype(str).to_numpy()
            x, y, n_strata = _residualize_strata(x, y, strata)
            residual_df = int(n_donors - n_strata - 1)
            if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
                status = "unstable_zero_variance"
            elif residual_df < 1:
                status = "insufficient_donors"
            else:
                estimate, standard_error, ci_low, ci_high, p_value = _association(
                    x, y, residual_df=residual_df
                )
                donor_directions = (x - x.mean()) * (y - y.mean())
                positive_fraction = float(np.mean(donor_directions > 0))
                if n_donors >= 4:
                    loo = []
                    for index in range(n_donors):
                        x_loo = np.delete(x, index)
                        y_loo = np.delete(y, index)
                        strata_loo = np.delete(strata, index)
                        x_loo, y_loo, n_strata_loo = _residualize_strata(
                            x_loo, y_loo, strata_loo
                        )
                        df_loo = len(x_loo) - n_strata_loo - 1
                        if np.std(x_loo, ddof=1) == 0 or np.std(y_loo, ddof=1) == 0:
                            continue
                        if df_loo < 1:
                            continue
                        loo.append(_association(x_loo, y_loo, residual_df=df_loo)[0])
                    if loo:
                        loo_min, loo_max = float(min(loo)), float(max(loo))
                        stability = (
                            "stable_sign" if loo_min > 0 or loo_max < 0 else "loo_sign_unstable"
                        )
                else:
                    stability = "too_few_donors_for_loo"
        rows.append(
            {
                "release_id": release_id,
                **dict(zip(group_columns, group_key)),
                "analysis_role": str(block["analysis_role"].iloc[0]),
                "gene_mapping_status": str(block["gene_mapping_status"].iloc[0]),
                "expression_mapping_available": bool(
                    block["expression_mapping_available"].iloc[0]
                ),
                "score_availability_status": str(
                    block["score_availability_status"].iloc[0]
                ),
                "is_primary_method": bool(block["is_primary_method"].iloc[0]),
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "q_value": np.nan,
                "n_donors": int(n_donors),
                "n_cells": int(eligible["n_cells"].sum()),
                "n_studies": len(study_donor_counts),
                "study_donor_counts_json": json.dumps(
                    study_donor_counts, separators=(",", ":"), sort_keys=True
                ),
                "n_strata": int(n_strata),
                "residual_degrees_of_freedom": int(max(residual_df, 0)),
                "analysis_strata": " | ".join(
                    sorted(eligible["stratum_id"].astype(str).unique())
                ),
                "positive_donor_fraction": positive_fraction,
                "loo_estimate_min": loo_min,
                "loo_estimate_max": loo_max,
                "circularity_status": circularity,
                "stability_status": stability,
                "analysis_unit": "donor by HMoE population summary; equal donor weight; study-condition-cohort-assay strata removed",
                "expression_representation": str(block["expression_representation"].iloc[0]),
                "score_representation": str(block["score_representation"].iloc[0]),
                "status": status,
                "affects_dual_selectivity_rank": False,
            }
        )
    output = pd.DataFrame(rows)
    fdr_groups = [
        "species", "trait_id", "population_id", "analysis_partition", "method"
    ]
    output["q_value"] = output.groupby(fdr_groups, observed=True, sort=False)[
        "p_value"
    ].transform(_bh)
    validate_target_coupling_semantics(output)
    return output.sort_values(group_columns, kind="stable").reset_index(drop=True)


def validate_target_coupling_semantics(frame: pd.DataFrame) -> None:
    """Validate leakage controls, donor inference, FDR, and rank inertness."""

    if frame.empty:
        raise ValueError("target_trait_coupling cannot be empty")
    for method, allowed in METHOD_CIRCULARITY.items():
        values = set(frame.loc[frame["method"].eq(method), "circularity_status"].astype(str))
        invalid = values - allowed
        if invalid:
            raise ValueError(f"invalid circularity status for {method}: {sorted(invalid)}")
    estimated = frame["status"].eq("estimated")
    numeric = frame.loc[
        estimated,
        ["estimate", "standard_error", "ci_low", "ci_high", "p_value", "q_value",
         "positive_donor_fraction"],
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("estimated target coupling rows require finite statistics")
    if (
        (frame.loc[estimated, "n_donors"].astype(int) < 3).any()
        or (frame.loc[estimated, "residual_degrees_of_freedom"].astype(int) < 1).any()
        or (numeric["standard_error"] < 0).any()
        or (~numeric["p_value"].between(0, 1)).any()
        or (~numeric["q_value"].between(0, 1)).any()
        or (~numeric["positive_donor_fraction"].between(0, 1)).any()
        or (numeric["ci_low"] > numeric["estimate"]).any()
        or (numeric["estimate"] > numeric["ci_high"]).any()
    ):
        raise ValueError("estimated target coupling invariants fail")
    if not set(frame["analysis_role"].astype(str)).issubset({"primary", "sensitivity"}):
        raise ValueError("target coupling analysis role must be primary or sensitivity")
    for row in frame[["n_donors", "n_studies", "study_donor_counts_json"]].itertuples(
        index=False
    ):
        try:
            study_counts = json.loads(str(row.study_donor_counts_json))
        except json.JSONDecodeError as exc:
            raise ValueError("target coupling study donor counts are invalid JSON") from exc
        if (
            not isinstance(study_counts, dict)
            or any(not isinstance(value, int) or value < 1 for value in study_counts.values())
            or int(row.n_studies) != len(study_counts)
            or int(row.n_donors) != sum(study_counts.values())
        ):
            raise ValueError("target coupling study donor counts are inconsistent")
    integrated_estimated = (
        estimated
        & frame["species"].astype(str).eq("human")
        & frame["analysis_partition"].astype(str).eq("human_integrated_control")
        & frame["analysis_role"].astype(str).eq("primary")
    )
    for value in frame.loc[integrated_estimated, "study_donor_counts_json"]:
        counts = json.loads(str(value))
        if set(counts) != {"Kamath", "Siletti"} or min(counts.values()) < 2:
            raise ValueError(
                "estimated integrated human coupling lacks Kamath and Siletti support"
            )
    expected_primary_partition = {
        "human": "human_integrated_control",
        "mouse": "mouse_all_conditions",
        "macaque": "allen_hmba_rhesus_control",
    }
    primary_partition = frame["analysis_role"].astype(str).eq("primary")
    for species, expected_partition in expected_primary_partition.items():
        observed = set(
            frame.loc[
                primary_partition & frame["species"].astype(str).eq(species),
                "analysis_partition",
            ].astype(str)
        )
        if observed and observed != {expected_partition}:
            raise ValueError(
                f"{species} primary target coupling must use {expected_partition}: "
                f"{sorted(observed)}"
            )
    base = [
        "release_id", "species", "trait_id", "population_id", "target_id",
        "analysis_partition",
    ]
    primary_counts = frame.groupby(base, observed=True)["is_primary_method"].sum()
    if not primary_counts.eq(1).all():
        raise ValueError("each target-trait-population requires exactly one primary coupling method")
    primary = frame["is_primary_method"].astype(bool)
    primary_score_unavailable = frame.loc[primary, "status"].eq("unavailable_score")
    allowed_explicit_rhesus = (
        frame.loc[primary, "species"].astype(str).eq("macaque")
        & frame.loc[primary, "circularity_status"].astype(str).eq(
            "not_evaluated_target_aware_leaveout"
        )
    )
    if (primary_score_unavailable & ~allowed_explicit_rhesus).any():
        raise ValueError("an unavailable coupling score cannot be the primary method")
    if frame["affects_dual_selectivity_rank"].astype(bool).any():
        raise ValueError("target-trait coupling cannot alter dual-selectivity rank")
