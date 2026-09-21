import numpy as np
import pandas as pd
import pytest

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.cross_species import assemble_cross_species_effects


def _row(species, target, effect, status="ranked"):
    available = status == "ranked"
    value = effect if available else np.nan
    return {
        "species": species,
        "trait_id": "pd",
        "population_id": "Sox6:Tafa1",
        "target_id": target,
        "rank_status": status,
        "gene_mapping_status": (
            "human_direct" if species == "human" else "reciprocal_one_to_one_ortholog"
        ),
        "within_da_effect": value,
        "within_da_ci_low": value - 0.2 if available else np.nan,
        "within_da_ci_high": value + 0.2 if available else np.nan,
        "within_da_percentile": 0.8 if available else 0.0,
        "within_da_q_value": 0.01 if available else np.nan,
        "within_da_competitor_id": "competitor_da" if available else "",
        "brainwide_effect": value,
        "brainwide_ci_low": value - 0.3 if available else np.nan,
        "brainwide_ci_high": value + 0.3 if available else np.nan,
        "brainwide_percentile": 0.7 if available else 0.0,
        "brainwide_q_value": 0.02 if available else np.nan,
        "brainwide_competitor_id": "competitor_brain" if available else "",
    }


def test_cross_species_retains_both_species_without_magnitude_subtraction():
    frame = pd.DataFrame(
        [
            _row("human", "HGNC:1", 1.0),
            _row("mouse", "HGNC:1", 0.6),
            _row("macaque", "HGNC:1", np.nan, "axis_unavailable"),
            _row("human", "HGNC:2", 0.5),
            _row("mouse", "HGNC:2", -0.4),
        ]
    )
    result = assemble_cross_species_effects(frame, release_id="r1")
    validate_table("cross_species_effects", result)
    assert len(result) == 8
    human_mouse = result[
        result["source_species"].eq("human")
        & result["target_species"].eq("mouse")
    ]
    target1 = human_mouse[human_mouse["target_id"].eq("HGNC:1")]
    assert set(target1["direction_status"]) == {"same_positive"}
    assert set(target1["availability_status"]) == {"both_available"}
    assert set(target1["source_effect"]) == {1.0}
    assert set(target1["target_effect"]) == {0.6}
    target2 = human_mouse[human_mouse["target_id"].eq("HGNC:2")]
    assert set(target2["direction_status"]) == {"opposite_direction"}
    macaque = result[result["target_species"].eq("macaque")]
    assert set(macaque["availability_status"]) == {"target_unavailable"}
    assert set(macaque["direction_status"]) == {"unavailable"}
    assert "effect_difference" not in result
    assert set(result["magnitude_comparison_status"]) == {
        "within_species_effects_only_no_cross_species_subtraction"
    }
    assert not result["affects_dual_selectivity_rank"].any()


def test_cross_species_rejects_duplicate_species_rows():
    row = _row("human", "HGNC:1", 1.0)
    frame = pd.DataFrame([row, row, _row("mouse", "HGNC:1", 0.5)])
    with pytest.raises(ValueError, match="duplicate"):
        assemble_cross_species_effects(frame, release_id="r1")


def test_cross_species_availability_is_axis_specific_not_overall_rank_status():
    human = _row("human", "HGNC:1", 1.0, status="ranked")
    human["rank_status"] = "unavailable"
    human["brainwide_effect"] = np.nan
    human["brainwide_ci_low"] = np.nan
    human["brainwide_ci_high"] = np.nan
    human["brainwide_percentile"] = np.nan
    human["brainwide_q_value"] = np.nan
    human["brainwide_competitor_id"] = ""
    frame = pd.DataFrame([human, _row("mouse", "HGNC:1", 0.5)])

    result = assemble_cross_species_effects(frame, release_id="r1").set_index(
        "contrast"
    )
    assert result.loc["within_da", "availability_status"] == "both_available"
    assert result.loc["within_da", "direction_status"] == "same_positive"
    assert result.loc["brainwide", "availability_status"] == "source_unavailable"
    assert result.loc["brainwide", "direction_status"] == "unavailable"


def test_cross_species_retains_different_trait_relevant_populations_without_comparing_direction():
    human = _row("human", "HGNC:1", 1.0)
    mouse = _row("mouse", "HGNC:1", -0.5)
    mouse["population_id"] = "Gad2:Ebf2"
    result = assemble_cross_species_effects(
        pd.DataFrame([human, mouse]), release_id="r1"
    )
    assert len(result) == 2
    assert set(result["source_population_id"]) == {"Sox6:Tafa1"}
    assert set(result["target_population_id"]) == {"Gad2:Ebf2"}
    assert set(result["availability_status"]) == {"both_available"}
    assert set(result["direction_status"]) == {
        "different_population_not_comparable"
    }
    assert set(result["mapping_status"]) == {
        "same_human_target_but_different_trait_relevant_hmoe_populations"
    }


def test_cross_species_missing_axis_is_unavailable_before_population_comparability():
    human = _row("human", "HGNC:1", 1.0)
    macaque = _row("macaque", "HGNC:1", np.nan, status="unavailable")
    macaque["population_id"] = "Calb1:Ptprt"

    result = assemble_cross_species_effects(
        pd.DataFrame([human, macaque]), release_id="r1"
    )

    assert set(result["availability_status"]) == {"target_unavailable"}
    assert set(result["direction_status"]) == {"unavailable"}
    assert set(result["mapping_status"]) == {
        "same_human_target_but_different_trait_relevant_hmoe_populations"
    }


def test_cross_species_crosses_all_selected_lead_set_populations():
    first = _row("human", "HGNC:1", 1.0)
    second = first.copy()
    second["population_id"] = "Sox6:Vcan"
    result = assemble_cross_species_effects(
        pd.DataFrame([first, second, _row("mouse", "HGNC:1", 0.5)]),
        release_id="r1",
    )
    assert len(result) == 4
    assert set(result["source_population_id"]) == {"Sox6:Tafa1", "Sox6:Vcan"}
    assert set(result["source_species"]) == {"human"}
    assert set(result["target_species"]) == {"mouse"}


def test_cross_species_can_materialize_only_shared_populations():
    human_shared = _row("human", "HGNC:1", 1.0)
    human_other = human_shared.copy()
    human_other["population_id"] = "Sox6:Vcan"
    result = assemble_cross_species_effects(
        pd.DataFrame([human_shared, human_other, _row("mouse", "HGNC:1", 0.5)]),
        release_id="r1",
        shared_populations_only=True,
    )
    assert len(result) == 2
    assert result["source_population_id"].eq("Sox6:Tafa1").all()
    assert result["target_population_id"].eq("Sox6:Tafa1").all()
