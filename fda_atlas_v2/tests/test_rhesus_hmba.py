import pandas as pd
import pytest

from fda_atlas_v2.rhesus_hmba import (
    ANXA1_LEAVES,
    CANONICAL_LEAVES,
    HMBA_DA_GROUPS,
    population_members,
    select_rhesus_da_cells,
    validate_rhesus_donors,
)


def donor_registry():
    return pd.DataFrame(
        {
            "donor_label": ["R1", "R2", "P1"],
            "donor_species": ["NCBITaxon:9544", "NCBITaxon:9544", "NCBITaxon:9545"],
            "species_scientific_name": [
                "Macaca mulatta", "Macaca mulatta", "Macaca nemestrina",
            ],
        }
    )


def test_rhesus_donor_filter_is_exact_and_excludes_pig_tailed_macaque():
    observed = validate_rhesus_donors(donor_registry())
    assert observed["donor_label"].tolist() == ["R1", "R2"]
    assert set(observed["donor_species"]) == {"NCBITaxon:9544"}
    assert set(observed["species_scientific_name"]) == {"Macaca mulatta"}


def test_rhesus_donor_filter_rejects_taxon_name_disagreement():
    donors = donor_registry()
    donors.loc[0, "species_scientific_name"] = "Macaca fascicularis"
    with pytest.raises(ValueError, match="taxon and scientific name disagree"):
        validate_rhesus_donors(donors)


def test_hmba_cell_selection_uses_taxonomy_da_and_verified_rhesus_donors_only():
    membership = pd.DataFrame(
        {
            "cell_label": ["a", "b", "c", "d"],
            "cluster_alias": ["Macaque-1", "Macaque-2", "Macaque-1", "Macaque-3"],
        }
    )
    hierarchy = pd.DataFrame(
        {
            "cluster_alias": ["Macaque-1", "Macaque-2", "Macaque-3"],
            "cluster_annotation_term_set_name": ["Group", "Group", "Group"],
            "cluster_annotation_term_name": [
                HMBA_DA_GROUPS[0], HMBA_DA_GROUPS[1], "not DA",
            ],
        }
    )
    metadata = pd.DataFrame(
        {
            "cell_label": ["a", "b", "c", "d"],
            "donor_label": ["R1", "P1", "R2", "R1"],
            "library_label": ["l1", "l2", "l3", "l4"],
        }
    )
    observed = select_rhesus_da_cells(
        membership, hierarchy, metadata, donor_registry(),
    )
    assert observed["cell_label"].tolist() == ["a", "c"]
    assert set(observed["donor_label"]) == {"R1", "R2"}
    assert set(observed["species_scientific_name"]) == {"Macaca mulatta"}


def test_public_population_members_include_anxa1_and_no_branches():
    members = population_members()
    assert members["Anxa1"] == set(ANXA1_LEAVES)
    assert members["Sox6"] == {
        leaf for leaf in CANONICAL_LEAVES if leaf.startswith("Sox6:")
    }
    assert members["Gad2"] == {
        leaf for leaf in CANONICAL_LEAVES if leaf.startswith("Gad2:")
    }
    assert len(members) == 22
    assert not any("/" in population for population in members)
