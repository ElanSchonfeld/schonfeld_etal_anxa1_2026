import numpy as np
import pandas as pd
import pytest

from fda_atlas_v2.ranking import compute_dual_selectivity_rank


def _ranking_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["human"] * 6,
            "trait_id": ["pd"] * 6,
            "population_id": ["sox6_tafa1"] * 6,
            "target_id": ["A", "B", "C", "D", "E", "F"],
            "within_da_effect": [4.0, 5.0, -1.0, 3.0, np.nan, 2.0],
            "brainwide_effect": [2.0, -0.1, 5.0, 3.0, 1.0, 1.0],
            "expression_support": [8.0, 9.0, 9.0, 7.0, 6.0, 5.0],
            "eligible": [True, True, True, True, True, False],
        }
    )


def test_rank_is_minimum_of_direction_aware_axis_percentiles():
    ranked = compute_dual_selectivity_rank(_ranking_input()).set_index("target_id")

    assert ranked.loc["B", "brainwide_percentile"] == 0
    assert ranked.loc["C", "within_da_percentile"] == 0
    assert ranked.loc["B", "rank_status"] == "nonpositive_joint_selectivity"
    assert ranked.loc["C", "rank_status"] == "nonpositive_joint_selectivity"
    assert pd.isna(ranked.loc["B", "dual_selectivity_score"])
    assert pd.isna(ranked.loc["C", "dual_selectivity_score"])

    expected = ranked[["within_da_percentile", "brainwide_percentile"]].min(axis=1)
    np.testing.assert_allclose(
        ranked.loc[["A", "D"], "dual_selectivity_score"],
        expected.loc[["A", "D"]],
    )
    assert ranked.loc["A", "dual_selectivity_rank"] == 1
    assert ranked.loc["D", "dual_selectivity_rank"] == 2


def test_missing_and_ineligible_targets_are_not_ranked():
    ranked = compute_dual_selectivity_rank(_ranking_input()).set_index("target_id")

    assert ranked.loc["E", "rank_status"] == "unavailable"
    assert ranked.loc["F", "rank_status"] == "ineligible"
    assert pd.isna(ranked.loc["E", "dual_selectivity_rank"])
    assert pd.isna(ranked.loc["F", "dual_selectivity_rank"])


def test_infinite_axis_effect_is_unavailable_not_ranked():
    frame = _ranking_input().iloc[[0]].assign(within_da_effect=np.inf)
    ranked = compute_dual_selectivity_rank(frame).iloc[0]
    assert ranked["rank_status"] == "unavailable"
    assert pd.isna(ranked["dual_selectivity_rank"])


def test_supporting_evidence_cannot_change_rank():
    base = _ranking_input()
    first = compute_dual_selectivity_rank(base)
    decorated = base.assign(
        genetics_score=[0, 100, 0, 0, 1000, 50],
        safety_score=[100, 0, 1000, 0, 0, 50],
        pharmacology_score=[0, 0, 0, 1000, 0, 50],
    )
    second = compute_dual_selectivity_rank(decorated)

    columns = [
        "target_id",
        "within_da_percentile",
        "brainwide_percentile",
        "dual_selectivity_score",
        "dual_selectivity_rank",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_ranking_is_independent_within_species_trait_population():
    base = _ranking_input().iloc[:4]
    mouse = base.assign(species="mouse", within_da_effect=[1.0, 2.0, 3.0, 4.0])
    combined = pd.concat([base, mouse], ignore_index=True)
    ranked = compute_dual_selectivity_rank(combined)

    counts = ranked.groupby(["species", "trait_id", "population_id"])[
        "dual_selectivity_rank"
    ].count()
    assert counts.to_dict() == {
        ("human", "pd", "sox6_tafa1"): 2,
        ("mouse", "pd", "sox6_tafa1"): 3,
    }


def test_ranking_does_not_depend_on_input_index_labels():
    frame = _ranking_input()
    frame.index = [0, 0, 1, 1, 2, 2]
    ranked = compute_dual_selectivity_rank(frame)

    assert len(ranked) == len(frame)
    assert ranked["target_id"].is_unique
    assert ranked.loc[ranked["rank_status"].eq("ranked"), "dual_selectivity_rank"].notna().all()


def test_ranking_rejects_string_boolean_eligibility():
    frame = _ranking_input()
    frame["eligible"] = frame["eligible"].map({True: "True", False: "False"})
    with pytest.raises(ValueError, match="strict nonnull booleans"):
        compute_dual_selectivity_rank(frame)


def test_ranking_rejects_duplicate_target_identity():
    frame = pd.concat([_ranking_input(), _ranking_input().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="identity is duplicated"):
        compute_dual_selectivity_rank(frame)


def test_ranking_rejects_missing_expression_tiebreak_for_rankable_target():
    frame = _ranking_input()
    frame.loc[frame["target_id"].eq("A"), "expression_support"] = np.nan
    with pytest.raises(ValueError, match="finite expression support"):
        compute_dual_selectivity_rank(frame)


def test_unavailable_eligible_target_can_lack_expression_support():
    frame = _ranking_input()
    frame.loc[frame["target_id"].eq("E"), "expression_support"] = np.nan

    ranked = compute_dual_selectivity_rank(frame).set_index("target_id")

    assert ranked.loc["E", "rank_status"] == "unavailable"
    assert pd.isna(ranked.loc["E", "dual_selectivity_rank"])


def test_available_axis_percentile_is_retained_when_other_axis_is_pending():
    frame = _ranking_input().iloc[:3].copy()
    frame["within_da_effect"] = [3.0, 2.0, -1.0]
    frame["brainwide_effect"] = np.nan
    frame["expression_support"] = np.nan

    ranked = compute_dual_selectivity_rank(frame)

    assert ranked["rank_status"].eq("unavailable").all()
    assert ranked["dual_selectivity_rank"].isna().all()
    assert ranked["brainwide_percentile"].isna().all()
    assert ranked["within_da_percentile"].tolist() == [1.0, 2 / 3, 0.0]


def test_all_nonpositive_joint_scores_receive_no_misleading_rank_one():
    frame = _ranking_input().iloc[:3].copy()
    frame["within_da_effect"] = [-1.0, -2.0, -3.0]
    frame["brainwide_effect"] = [1.0, 2.0, 3.0]

    ranked = compute_dual_selectivity_rank(frame)

    assert ranked["rank_status"].eq("nonpositive_joint_selectivity").all()
    assert ranked["dual_selectivity_score"].isna().all()
    assert ranked["dual_selectivity_rank"].isna().all()
