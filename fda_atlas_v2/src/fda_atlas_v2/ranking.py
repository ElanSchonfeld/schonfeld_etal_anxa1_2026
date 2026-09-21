"""Primary dual-selectivity ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["species", "trait_id", "population_id"]
REQUIRED_COLUMNS = [
    *GROUP_COLUMNS,
    "target_id",
    "within_da_effect",
    "brainwide_effect",
    "expression_support",
    "eligible",
]


def compute_dual_selectivity_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with deterministic direction-aware v2 ranks."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing ranking columns: {missing}")

    result = frame.copy().reset_index(drop=True)
    identities = [*GROUP_COLUMNS, "target_id"]
    if result[identities].isna().any().any() or (
        result[identities].astype(str).apply(lambda column: column.str.strip()).eq("")
        .any().any()
    ):
        raise ValueError("ranking identities must be nonblank")
    if result.duplicated(identities).any():
        raise ValueError("ranking target identity is duplicated within an analysis")
    if result["eligible"].isna().any() or not result["eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("eligible must contain strict nonnull booleans")
    result["within_da_percentile"] = np.nan
    result["brainwide_percentile"] = np.nan
    result["dual_selectivity_score"] = np.nan
    result["dual_selectivity_tiebreak"] = np.nan
    result["dual_selectivity_rank"] = pd.array(
        [pd.NA] * len(result), dtype="Int64"
    )

    finite_within = pd.Series(
        np.isfinite(pd.to_numeric(result["within_da_effect"], errors="coerce")),
        index=result.index,
    )
    finite_brainwide = pd.Series(
        np.isfinite(pd.to_numeric(result["brainwide_effect"], errors="coerce")),
        index=result.index,
    )
    finite_effects = finite_within & finite_brainwide
    eligible = result["eligible"].astype(bool)
    expression_support = pd.to_numeric(
        result["expression_support"], errors="coerce"
    )
    rankable = eligible & finite_effects
    if (rankable & ~np.isfinite(expression_support)).any():
        raise ValueError("rankable targets require finite expression support")
    result["rank_status"] = np.select(
        [~eligible, eligible & ~finite_effects],
        ["ineligible", "unavailable"],
        default="ranked",
    )

    for _, indices in result.loc[eligible].groupby(
        GROUP_COLUMNS, sort=False, dropna=False
    ).groups.items():
        group_index = pd.Index(indices)
        for effect_column, percentile_column, finite_axis in (
            ("within_da_effect", "within_da_percentile", finite_within),
            ("brainwide_effect", "brainwide_percentile", finite_brainwide),
        ):
            index = group_index[finite_axis.loc[group_index].to_numpy()]
            if len(index) == 0:
                continue
            effects = pd.to_numeric(result.loc[index, effect_column])
            percentiles = effects.rank(method="average", pct=True)
            percentiles = percentiles.where(effects > 0, 0.0)
            result.loc[index, percentile_column] = percentiles

        index = group_index[rankable.loc[group_index].to_numpy()]
        if len(index) == 0:
            continue
        within = result.loc[index, "within_da_percentile"].astype(float)
        brain = result.loc[index, "brainwide_percentile"].astype(float)
        score = np.minimum(within, brain)
        positive_index = index[score.to_numpy() > 0]
        nonpositive_index = index[score.to_numpy() <= 0]
        result.loc[nonpositive_index, "rank_status"] = (
            "nonpositive_joint_selectivity"
        )
        if len(positive_index) == 0:
            continue
        positive_within = within.loc[positive_index]
        positive_brain = brain.loc[positive_index]
        result.loc[positive_index, "dual_selectivity_score"] = np.minimum(
            positive_within, positive_brain
        )
        result.loc[positive_index, "dual_selectivity_tiebreak"] = np.sqrt(
            positive_within * positive_brain
        )

        ordered = result.loc[positive_index].sort_values(
            [
                "dual_selectivity_score",
                "dual_selectivity_tiebreak",
                "expression_support",
                "target_id",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        )
        result.loc[ordered.index, "dual_selectivity_rank"] = pd.array(
            range(1, len(ordered) + 1), dtype="Int64"
        )

    return result
