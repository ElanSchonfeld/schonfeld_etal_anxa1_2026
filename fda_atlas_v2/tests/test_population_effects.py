import pandas as pd
import pytest

from fda_atlas_v2.population_effects import (
    estimate_population_effects,
    select_population_lead_set,
    set_primary_cohort_flags,
    validated_human_integrated_control_mask,
)


def _cell_scores() -> pd.DataFrame:
    rows = []
    values = {
        "d1": {"A": 3.0, "B": 0.0, "C": 2.95},
        "d2": {"A": 4.0, "B": 1.0, "C": 3.7},
        "d3": {"A": 5.0, "B": 2.0, "C": 5.0},
        "d4": {"A": 6.0, "B": 3.0, "C": 5.9},
    }
    for donor, populations in values.items():
        study = "study1" if donor in {"d1", "d2"} else "study2"
        for population, score in populations.items():
            n_cells = 100 if donor == "d1" and population == "A" else 10
            rows.extend(
                {
                    "donor_id": f"{study}::{donor}",
                    "source_donor_id": donor,
                    "study": study,
                    "condition": "control",
                    "population_id": population,
                    "score": score,
                }
                for _ in range(n_cells)
            )
    return pd.DataFrame(rows)


def test_population_effects_use_equal_weight_per_donor():
    effects, donor_means = estimate_population_effects(
        _cell_scores(), min_cells_per_donor_population=5
    )
    effects = effects.set_index("population_id")

    assert effects.loc["A", "n_donors"] == 4
    assert effects.loc["A", "effect"] > effects.loc["C", "effect"]
    assert effects.loc["A", "effect"] > 0
    assert effects.loc["B", "effect"] < 0
    assert donor_means.groupby("donor_id")["donor_weight"].first().eq(1.0).all()


def test_population_with_too_few_cells_is_removed_for_that_donor():
    frame = _cell_scores()
    group_order = frame.groupby(["donor_id", "population_id"]).cumcount()
    mask = (
        frame["donor_id"].eq("study2::d4")
        & frame["population_id"].eq("C")
        & group_order.ge(2)
    )
    frame = frame.loc[~mask]
    effects, donor_means = estimate_population_effects(
        frame, min_cells_per_donor_population=5
    )
    effects = effects.set_index("population_id")

    assert effects.loc["C", "n_donors"] == 3


def test_lead_set_keeps_close_population_but_rejects_separated_population():
    effects, donor_means = estimate_population_effects(
        _cell_scores(), min_cells_per_donor_population=5
    )
    leads = select_population_lead_set(effects, donor_means, min_common_donors=3)
    status = leads.set_index("population_id")["lead_status"].to_dict()

    assert status["A"] == "leader"
    assert status["C"] == "lead_set"
    assert status["B"] == "separated"


def test_population_with_insufficient_donors_cannot_become_leader():
    frame = _cell_scores()
    rare = pd.DataFrame(
        {
            "donor_id": ["study1::d1"] * 10,
            "source_donor_id": ["d1"] * 10,
            "study": ["study1"] * 10,
            "condition": ["control"] * 10,
            "population_id": ["X"] * 10,
            "score": [100.0] * 10,
        }
    )
    effects, donor_means = estimate_population_effects(
        pd.concat([frame, rare], ignore_index=True),
        min_cells_per_donor_population=5,
    )
    leads = select_population_lead_set(
        effects, donor_means, min_common_donors=3, min_donors=3
    ).set_index("population_id")

    assert leads.loc["X", "lead_status"] == "insufficient_data"
    assert leads.loc["X", "leader_id"] != "X"


def test_primary_flags_can_select_integrated_human_without_changing_values():
    frame = pd.DataFrame(
        {
            "species": ["human", "human", "mouse"],
            "analysis_cohort": [
                "kamath_control", "control_all", "all_condition_stratified"
            ],
            "is_primary_cohort": [True, False, True],
            "effect": [1.0, 2.0, 3.0],
        }
    )
    repaired = set_primary_cohort_flags(
        frame,
        {"human": "control_all", "mouse": "all_condition_stratified"},
    )
    assert repaired["is_primary_cohort"].tolist() == [False, True, True]
    pd.testing.assert_frame_equal(
        repaired.drop(columns="is_primary_cohort"),
        frame.drop(columns="is_primary_cohort"),
    )


def test_population_uncertainty_is_stratified_while_donors_remain_equal_weight():
    rows = []
    for donor, study, a_score, b_score in (
        ("d1", "Kamath", 1.0, 0.0),
        ("d2", "Kamath", 1.0, 0.0),
        ("d3", "Siletti", 3.0, 0.0),
        ("d4", "Siletti", 3.0, 0.0),
    ):
        for population, score in (("A", a_score), ("B", b_score)):
            rows.extend(
                {
                    "donor_id": f"{study}::{donor}",
                    "source_donor_id": donor,
                    "study": study,
                    "condition": "control",
                    "population_id": population,
                    "score": score,
                }
                for _ in range(10)
            )
    effects, donor_means = estimate_population_effects(
        pd.DataFrame(rows), min_cells_per_donor_population=5
    )
    a = effects.set_index("population_id").loc["A"]
    assert a["effect"] == 2.0
    assert a["standard_error"] == 0.0
    assert a["n_studies"] == 2
    assert a["inference_model"] == (
        "equal-donor marginal effect with study-stratified variance"
    )
    assert set(donor_means["source_donor_id"]) == {"d1", "d2", "d3", "d4"}


def test_population_inference_model_reports_actual_variance_fallback():
    frame = _cell_scores()
    single_study = frame[frame["study"].eq("study1")]
    effects, _ = estimate_population_effects(
        single_study, min_cells_per_donor_population=5
    )
    assert set(effects["inference_model"]) == {
        "equal-donor marginal effect with single-study variance"
    }

    sparse_second_study = frame[
        frame["donor_id"].ne("study2::d4")
    ]
    effects, _ = estimate_population_effects(
        sparse_second_study, min_cells_per_donor_population=5
    )
    assert set(effects["inference_model"]) == {
        "equal-donor marginal effect with pooled variance because a study has fewer "
        "than two contributing donors"
    }


def test_human_integrated_control_requires_exact_study_status_pairs_and_counts():
    metadata = pd.concat(
        [
            pd.DataFrame(
                {
                    "study": "Kamath",
                    "disease_status": "Ctrl",
                    "donor": [f"k{index % 8}" for index in range(15_684)],
                }
            ),
            pd.DataFrame(
                {
                    "study": "Siletti",
                    "disease_status": "Control",
                    "donor": [f"s{index % 3}" for index in range(823)],
                }
            ),
        ],
        ignore_index=True,
    )
    mask = validated_human_integrated_control_mask(metadata)
    assert int(mask.sum()) == 16_507

    mislabeled = metadata.copy()
    mislabeled.loc[15_684, "disease_status"] = "Ctrl"
    with pytest.raises(ValueError, match="cell counts disagree"):
        validated_human_integrated_control_mask(mislabeled)
