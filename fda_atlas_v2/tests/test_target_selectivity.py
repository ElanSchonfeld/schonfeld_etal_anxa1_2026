import numpy as np
import pandas as pd

from fda_atlas_v2.target_selectivity import (
    assemble_target_population_selectivity,
    build_target_gene_registry,
    combine_target_evidence_universe,
    validate_axis_identity,
    validate_selected_trait_populations,
)
from fda_atlas_v2.contracts import TRAITS


def _axis(gene_effects):
    return pd.DataFrame(
        {
            "gene_id": list(gene_effects),
            "effect": list(gene_effects.values()),
            "ci_low": np.array(list(gene_effects.values())) - 0.2,
            "ci_high": np.array(list(gene_effects.values())) + 0.2,
            "detection_difference": 0.3,
            "competitor_id": "competitor",
            "n_common_donors": 4,
            "focal_n_cells": 100,
            "focal_detection": 0.8,
            "p_value": 0.01,
            "q_value": 0.02,
            "competitor_loo_selection_fraction": 1.0,
            "common_donor_ids": "d1 | d2 | d3 | d4",
            "n_eligible_comparators": 2,
            "status": "estimated",
        }
    )


def test_target_join_occurs_after_genomewide_axes_and_ranks_only_valid_mappings():
    edges = pd.DataFrame(
        {
            "target_id": ["HGNC:1", "HGNC:2", "HGNC:3", "HGNC:4"],
            "human_gene": ["A", "B", "C", "D"],
        }
    )
    orthologs = pd.DataFrame(
        {
            "Human Gene": ["A", "B", "B", "D", "X"],
            "Mouse Gene": ["a", "b1", "b2", "d", "d"],
        }
    )
    registry = build_target_gene_registry(edges, species="mouse", orthologs=orthologs)
    observed = registry.set_index("target_id")
    assert observed.loc["HGNC:1", "gene_mapping_status"] == (
        "reciprocal_one_to_one_mouse_ortholog"
    )
    assert observed.loc["HGNC:2", "gene_mapping_status"] == "multiple_mouse_orthologs"
    assert observed.loc["HGNC:3", "gene_mapping_status"] == "no_mouse_ortholog"
    assert observed.loc["HGNC:4", "gene_mapping_status"] == (
        "nonreciprocal_mouse_ortholog"
    )
    result = assemble_target_population_selectivity(
        _axis({"a": 2.0, "unrelated": 100.0}),
        _axis({"a": 1.0, "unrelated": 100.0}),
        registry,
        release_id="development",
        species="mouse",
        trait_id="pd",
        population_id="Sox6:Tafa1",
    ).set_index("target_id")
    assert result.loc["HGNC:1", "rank_status"] == "ranked"
    assert result.loc["HGNC:1", "dual_selectivity_rank"] == 1
    assert result.loc["HGNC:2", "rank_status"] == "ineligible"
    assert result.loc["HGNC:3", "rank_status"] == "ineligible"
    assert result.loc["HGNC:4", "rank_status"] == "ineligible"


def test_human_target_registry_is_direct_and_complete():
    edges = pd.DataFrame(
        {"target_id": ["HGNC:1", "HGNC:2"], "human_gene": ["A", "B"]}
    )
    registry = build_target_gene_registry(edges, species="human")
    assert registry["species_gene"].tolist() == ["A", "B"]
    assert set(registry["gene_mapping_status"]) == {"human_direct"}


def test_human_target_absent_from_axis_is_retained_as_unavailable():
    edges = pd.DataFrame(
        {"target_id": ["HGNC:1", "HGNC:2"], "human_gene": ["A", "B"]}
    )
    registry = build_target_gene_registry(edges, species="human")

    result = assemble_target_population_selectivity(
        _axis({"A": 1.5}),
        _axis({"A": 0.8}),
        registry,
        release_id="development",
        species="human",
        trait_id="pd",
        population_id="Sox6:Tafa1",
    ).set_index("target_id")

    assert result.loc["HGNC:1", "rank_status"] == "ranked"
    assert result.loc["HGNC:2", "eligible"]
    assert result.loc["HGNC:2", "rank_status"] == "unavailable"
    assert result.loc["HGNC:2", "within_da_status"] == "gene_not_measured_in_axis"
    assert result.loc["HGNC:2", "brainwide_status"] == "gene_not_measured_in_axis"
    assert pd.isna(result.loc["HGNC:2", "dual_selectivity_rank"])


def test_macaque_registry_requires_reciprocal_one_to_one_mapping_for_rank():
    edges = pd.DataFrame(
        {
            "target_id": ["HGNC:1", "HGNC:2", "HGNC:3", "HGNC:4", "HGNC:5"],
            "human_gene": ["A", "B", "C", "D", "E"],
        }
    )
    orthologs = pd.DataFrame(
        {
            "Human Gene": ["A", "B", "B", "D", "X", "E"],
            "Macaque Gene": [
                "A_MAC", "B1_MAC", "B2_MAC", "D_MAC", "D_MAC", "E_MAC"
            ],
            "Orthology Type": [
                "ortholog_one2one", "ortholog_one2many", "ortholog_one2many",
                "ortholog_one2one", "ortholog_one2one", "ortholog_one2one",
            ],
            "Orthology Confidence": ["1", "1", "1", "1", "1", "0"],
        }
    )
    registry = build_target_gene_registry(
        edges, species="macaque", orthologs=orthologs
    )
    observed = registry.set_index("target_id")
    assert observed.loc["HGNC:1", "gene_mapping_status"] == (
        "reciprocal_one_to_one_macaque_ortholog"
    )
    assert observed.loc["HGNC:2", "gene_mapping_status"] == (
        "multiple_macaque_orthologs"
    )
    assert observed.loc["HGNC:3", "gene_mapping_status"] == "no_macaque_ortholog"
    assert observed.loc["HGNC:4", "gene_mapping_status"] == (
        "nonreciprocal_macaque_ortholog"
    )
    assert observed.loc["HGNC:5", "gene_mapping_status"] == (
        "low_confidence_macaque_ortholog"
    )
    result = assemble_target_population_selectivity(
        _axis({"A_MAC": 1.5}),
        _axis({"A_MAC": 0.8}),
        registry,
        release_id="development",
        species="macaque",
        trait_id="pd",
        population_id="Sox6:Tafa1",
    ).set_index("target_id")
    assert result.loc["HGNC:1", "rank_status"] == "ranked"
    assert set(
        result.loc[["HGNC:2", "HGNC:3", "HGNC:4", "HGNC:5"], "rank_status"]
    ) == {
        "ineligible"
    }


def test_axis_identity_refuses_cross_dataset_or_cross_population_reuse():
    axis = _axis({"A": 1.0}).assign(
        release_id="development",
        species="human",
        focal_population_id="Sox6:Tafa1",
        axis_id="within_da",
        dataset_id="kamath_da",
    )
    validate_axis_identity(
        axis,
        release_id="development",
        species="human",
        population_id="Sox6:Tafa1",
        axis_id="within_da",
        dataset_id="kamath_da",
    )
    with np.testing.assert_raises_regex(ValueError, "dataset_id"):
        validate_axis_identity(
            axis,
            release_id="development",
            species="human",
            population_id="Sox6:Tafa1",
            axis_id="within_da",
            dataset_id="allen_whb",
        )


def test_target_universe_retains_substance_only_and_moiety_only_targets():
    moiety = pd.DataFrame(
        {"target_id": ["HGNC:1", "HGNC:2"], "human_gene": ["A", "B"]}
    )
    substance = pd.DataFrame(
        {"target_id": ["HGNC:1", "HGNC:3"], "human_gene": ["A", "C"]}
    )
    universe = combine_target_evidence_universe(moiety, substance).set_index(
        "target_id"
    )
    assert set(universe.index) == {"HGNC:1", "HGNC:2", "HGNC:3"}
    assert universe.loc["HGNC:1", "target_evidence_grains"] == "moiety|substance"
    assert universe.loc["HGNC:2", "target_evidence_grains"] == "moiety"
    assert universe.loc["HGNC:3", "target_evidence_grains"] == "substance"


def test_trait_population_selection_preserves_lead_sets_with_one_default_leader():
    candidates = pd.DataFrame(
        {
            "trait_id": list(TRAITS),
            "hierarchy_level": ["leaf"] * len(TRAITS),
            "population_id": ["Sox6:Tafa1"] * len(TRAITS),
            "selected_resolution": [True] * len(TRAITS),
            "bridge_available": [True] * len(TRAITS),
            "default_leader": [True] * len(TRAITS),
        }
    )
    selected = validate_selected_trait_populations(candidates, species="human")
    assert len(selected) == 11

    lead_set = candidates.iloc[[0]].copy()
    lead_set["population_id"] = "Sox6:Vcan"
    lead_set["default_leader"] = False
    selected = validate_selected_trait_populations(
        pd.concat([candidates, lead_set], ignore_index=True), species="human"
    )
    assert len(selected) == 12

    second_leader = lead_set.copy()
    second_leader["default_leader"] = True
    with np.testing.assert_raises_regex(ValueError, "exactly one default leader"):
        validate_selected_trait_populations(
            pd.concat([candidates, second_leader], ignore_index=True), species="human"
        )
