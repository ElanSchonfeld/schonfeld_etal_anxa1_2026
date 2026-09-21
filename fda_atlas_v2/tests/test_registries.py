import pandas as pd
import pytest

from fda_atlas_v2.registries import (
    build_population_registry,
    collapse_gene_set_duplicate_weights,
    build_gene_set_duplicate_audit,
    build_trait_registry,
    parse_scdrs_gene_set,
)


def test_gene_set_parser_preserves_source_entries_and_audits_scdrs_duplicates(tmp_path):
    path = tmp_path / "pd.gs"
    path.write_text("TRAIT\tGENESET\npd\tSNCA:8.0,LRRK2:7.0\n")
    assert parse_scdrs_gene_set(path, "pd") == [("SNCA", 8.0), ("LRRK2", 7.0)]
    path.write_text("TRAIT\tGENESET\npd\tSNCA:8.0,SNCA:7.0\n")
    assert parse_scdrs_gene_set(path, "pd") == [("SNCA", 8.0), ("SNCA", 7.0)]
    audit = build_gene_set_duplicate_audit("r", {"pd": path})
    assert audit.loc[0, "scdrs_effective_weight"] == 7.0
    assert audit.loc[0, "weights_disagree"]


def test_duplicate_sensitivity_uses_mean_without_changing_unique_genes():
    corrected, changes = collapse_gene_set_duplicate_weights(
        [("A", 4.0), ("B", 2.0), ("A", 2.0)]
    )
    assert corrected == [("A", 3.0), ("B", 2.0)]
    assert changes.loc[0, "gene"] == "A"
    assert changes.loc[0, "original_last_occurrence_weight"] == 2.0
    assert changes.loc[0, "corrected_mean_weight"] == 3.0
    assert changes.loc[0, "relative_weight_change"] == 0.5


def test_trait_registry_requires_all_frozen_traits(tmp_path):
    path = tmp_path / "pd.gs"
    path.write_text("TRAIT\tGENESET\npd\tSNCA:8.0\n")
    with pytest.raises(ValueError, match="frozen 11 traits"):
        build_trait_registry("r", {"pd": path}, {"traits": {}}, magma_version="v1")


def test_population_registry_is_species_specific_and_hierarchical():
    effects = pd.DataFrame(
        {
            "species": ["human", "human", "mouse", "macaque"],
            "hierarchy_level": ["family", "leaf", "family", "family"],
            "population_id": ["Sox6", "Sox6:A", "Sox6", "Sox6"],
            "parent_population_id": ["DA", "Sox6/X", "DA", "DA"],
        }
    )
    registry = build_population_registry("r", effects)
    assert len(registry) == 4
    assert set(registry["species"]) == {"human", "mouse", "macaque"}
    assert registry.set_index(["species", "population_id"]).loc[
        ("human", "Sox6:A"), "parent_population_id"
    ] == "Sox6"
    human_definition = registry.set_index(["species", "population_id"]).loc[
        ("human", "Sox6:A"), "definition_source"
    ]
    assert "Kamath defines subtype ground truth" in human_definition
    assert "Siletti contributes the independent integrated study stratum" in human_definition
    macaque_definition = registry.set_index(["species", "population_id"]).loc[
        ("macaque", "Sox6"), "definition_source"
    ]
    assert "Macaca mulatta" in macaque_definition
    assert "four contributing donors" in macaque_definition
    assert "Chiou rhesus cross-cohort reference" in macaque_definition


def test_population_registry_removes_branches_and_retains_anxa1_for_all_species():
    effects = pd.DataFrame(
        {
            "species": ["human", "mouse", "macaque", "macaque"],
            "hierarchy_level": ["family", "family", "family", "branch"],
            "population_id": ["Anxa1", "Anxa1", "Anxa1", "Sox6/example"],
            "parent_population_id": ["DA", "DA", "DA", "Sox6"],
        }
    )
    registry = build_population_registry("r", effects)
    assert set(registry["species"]) == {"human", "mouse", "macaque"}
    assert set(registry["population_id"]) == {"Anxa1"}
    assert not registry["hierarchy_level"].eq("branch").any()
