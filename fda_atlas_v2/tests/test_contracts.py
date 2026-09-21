import pandas as pd
import pytest

from fda_atlas_v2.contracts import (
    SCHEMA_VERSION,
    TABLE_CONTRACTS,
    ContractError,
    validate_table,
)


EXPECTED_TABLES = {
    "source_citations",
    "dataset_registry",
    "trait_registry",
    "population_registry",
    "cell_membership_registry",
    "cell_partition_audit",
    "scdrs_population_effects",
    "trait_population_leads",
    "target_population_selectivity",
    "target_trait_coupling",
    "causal_genetics_evidence",
    "causal_genetics_credible_set_variants",
    "cross_species_effects",
    "source_row_reconciliation",
    "product_ingredients",
    "regulatory_substances",
    "substance_identity_edges",
    "active_moieties",
    "substance_active_moiety_edges",
    "drug_target_edges",
    "drug_target_action_summary",
    "drug_target_activity_evidence",
    "drug_polypharmacology_summary",
    "regulatory_substance_target_evidence",
    "regulatory_substance_target_coverage",
    "active_moiety_target_coverage",
    "withheld_moiety_target_attribution",
    "regulatory_products",
    "regulatory_labels",
    "regulatory_label_entity_edges",
    "regulatory_label_sections",
    "drug_target_trait_contexts",
    "entity_search",
    "data_dictionary",
}


def _selectivity_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "release_id": ["v2-test"],
            "species": ["human"],
            "trait_id": ["pd"],
            "population_id": ["sox6_tafa1"],
            "target_id": ["HGNC:123"],
            "human_gene": ["GENE1"],
            "species_gene": ["GENE1"],
            "species_gene_identifier_type": ["gene_symbol"],
            "ortholog_count": [1],
            "orthologs_json": ['["GENE1"]'],
            "gene_mapping_status": ["human_direct"],
            "within_da_effect": [1.2],
            "within_da_ci_low": [0.8],
            "within_da_ci_high": [1.6],
            "within_da_detection_difference": [0.4],
            "within_da_competitor_id": ["sox6_vcan"],
            "within_da_percentile": [1.0],
            "within_da_p_value": [0.01],
            "within_da_q_value": [0.02],
            "within_da_competitor_loo_selection_fraction": [1.0],
            "within_da_common_donor_ids": ["d1 | d2 | d3"],
            "within_da_n_eligible_comparators": [2],
            "within_da_status": ["estimated"],
            "brainwide_effect": [0.9],
            "brainwide_ci_low": [0.5],
            "brainwide_ci_high": [1.3],
            "brainwide_detection_difference": [0.3],
            "brainwide_competitor_id": ["astrocyte_1"],
            "brainwide_percentile": [1.0],
            "brainwide_p_value": [0.02],
            "brainwide_q_value": [0.03],
            "brainwide_competitor_loo_selection_fraction": [1.0],
            "brainwide_common_donor_ids": ["d1 | d2 | d3"],
            "brainwide_n_eligible_comparators": [100],
            "brainwide_status": ["estimated"],
            "dual_selectivity_score": [1.0],
            "dual_selectivity_rank": [1],
            "rank_status": ["ranked"],
            "eligible": [True],
            "focal_n_donors": [3],
            "focal_n_cells": [120],
            "expression_support": [8.0],
        }
    )


def test_schema_names_are_frozen():
    assert SCHEMA_VERSION == "2.0.0"
    assert set(TABLE_CONTRACTS) == EXPECTED_TABLES


def test_valid_selectivity_table_passes():
    validate_table("target_population_selectivity", _selectivity_row())


def test_missing_required_column_fails():
    frame = _selectivity_row().drop(columns="brainwide_effect")
    with pytest.raises(ContractError, match="missing required columns"):
        validate_table("target_population_selectivity", frame)


def test_duplicate_primary_key_fails():
    frame = pd.concat([_selectivity_row(), _selectivity_row()], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate primary key"):
        validate_table("target_population_selectivity", frame)


def test_invalid_species_fails():
    frame = _selectivity_row().assign(species="rhesus")
    with pytest.raises(ContractError, match="invalid species"):
        validate_table("target_population_selectivity", frame)


def test_joint_score_must_equal_minimum_axis_percentile():
    frame = _selectivity_row().assign(dual_selectivity_score=0.75)
    with pytest.raises(ContractError, match="dual-selectivity invariant"):
        validate_table("target_population_selectivity", frame)


def test_selectivity_percentiles_and_persisted_rank_are_reconstructed():
    first = _selectivity_row()
    second = _selectivity_row().assign(
        target_id="HGNC:124",
        within_da_effect=0.8,
        brainwide_effect=0.7,
        within_da_percentile=0.5,
        brainwide_percentile=0.5,
        dual_selectivity_score=0.5,
        dual_selectivity_rank=2,
        expression_support=7.0,
    )
    frame = pd.concat([first, second], ignore_index=True)
    validate_table("target_population_selectivity", frame)

    with pytest.raises(ContractError, match="percentile reconstruction"):
        validate_table(
            "target_population_selectivity",
            frame.assign(within_da_percentile=[0.9, 0.6], dual_selectivity_score=[0.9, 0.5]),
        )
    with pytest.raises(ContractError, match="persisted rank reconstruction"):
        validate_table(
            "target_population_selectivity",
            frame.assign(dual_selectivity_rank=[2, 1]),
        )


def test_unranked_selectivity_rows_require_explicit_empty_rank_fields():
    frame = _selectivity_row().assign(
        eligible=False,
        species_gene="",
        ortholog_count=0,
        orthologs_json="[]",
        gene_mapping_status="no_mouse_ortholog",
        species="mouse",
        rank_status="ineligible",
        within_da_effect=float("nan"),
        brainwide_effect=float("nan"),
        within_da_status="not_evaluated_ineligible_mapping",
        brainwide_status="not_evaluated_ineligible_mapping",
        within_da_percentile=float("nan"),
        brainwide_percentile=float("nan"),
        dual_selectivity_score=float("nan"),
        dual_selectivity_rank=pd.NA,
    )
    validate_table("target_population_selectivity", frame)
    with pytest.raises(ContractError, match="unranked rows contain dual_selectivity_rank"):
        validate_table(
            "target_population_selectivity", frame.assign(dual_selectivity_rank=1)
        )


def test_unavailable_joint_rank_can_retain_one_available_axis_percentile():
    frame = _selectivity_row().assign(
        brainwide_effect=float("nan"),
        brainwide_status="pending_external_brainwide_rna",
        rank_status="unavailable",
        within_da_percentile=1.0,
        brainwide_percentile=float("nan"),
        dual_selectivity_score=float("nan"),
        dual_selectivity_rank=pd.NA,
        expression_support=float("nan"),
    )
    validate_table("target_population_selectivity", frame)


def test_measured_nonpositive_joint_selectivity_is_retained_without_rank():
    frame = _selectivity_row().assign(
        within_da_effect=-0.2,
        within_da_percentile=0.0,
        brainwide_percentile=1.0,
        rank_status="nonpositive_joint_selectivity",
        dual_selectivity_score=float("nan"),
        dual_selectivity_rank=pd.NA,
    )

    validate_table("target_population_selectivity", frame)


def test_selectivity_axis_status_must_explain_effect_availability():
    frame = _selectivity_row().assign(
        within_da_effect=float("nan"),
        rank_status="unavailable",
        within_da_percentile=float("nan"),
        brainwide_percentile=1.0,
        dual_selectivity_score=float("nan"),
        dual_selectivity_rank=pd.NA,
    )
    with pytest.raises(ContractError, match="status disagrees with effect availability"):
        validate_table("target_population_selectivity", frame)

    frame["within_da_status"] = "gene_not_measured_in_axis"
    validate_table("target_population_selectivity", frame)


def test_selectivity_eligibility_requires_exact_species_mapping():
    frame = _selectivity_row().assign(
        species="mouse",
        gene_mapping_status="nonreciprocal_mouse_ortholog",
    )
    with pytest.raises(ContractError, match="eligibility disagrees"):
        validate_table("target_population_selectivity", frame)

    frame["gene_mapping_status"] = "reciprocal_one_to_one_mouse_ortholog"
    validate_table("target_population_selectivity", frame)

    with pytest.raises(ContractError, match="ortholog array and count disagree"):
        validate_table(
            "target_population_selectivity",
            frame.assign(ortholog_count=2),
        )


def test_causal_genetics_is_source_grained_and_rank_inert():
    frame = pd.DataFrame(
        {
            "release_id": ["v2-test"],
            "trait_id": ["pd"],
            "target_id": ["HGNC:1"],
            "human_gene": ["A"],
            "hgnc_id": ["HGNC:1"],
            "ensembl_gene_id": ["ENSG000001"],
            "source": ["Open Targets Platform"],
            "source_release": ["26.06"],
            "study_id": ["GCST1"],
            "study_trait": ["Parkinson disease"],
            "locus_id": ["locus-1"],
            "lead_variant_id": ["1_1_A_G"],
            "region": ["1:1-2"],
            "finemapping_method": ["SuSiE"],
            "credible_set_confidence": ["high"],
            "evidence_type": ["locus_to_gene"],
            "evidence_value": [0.4],
            "evidence_value_type": ["L2G score"],
            "effect_direction": ["not_interpreted"],
            "direction_basis": ["not therapeutic direction"],
            "biosample_id": [""],
            "biosample_name": [""],
            "biosample_relevance": ["not_available"],
            "biosample_ontology_ancestors_json": ["[]"],
            "clpp": [float("nan")],
            "h3": [float("nan")],
            "h4": [float("nan")],
            "beta_ratio_sign_average": [float("nan")],
            "source_record_id": ["ot:l2g:one"],
            "status": ["above_0.05_support_threshold"],
            "affects_dual_selectivity_rank": [False],
        }
    )
    validate_table("causal_genetics_evidence", frame)
    with pytest.raises(ContractError, match="cannot alter rank"):
        validate_table(
            "causal_genetics_evidence",
            frame.assign(affects_dual_selectivity_rank=True),
        )


def test_credible_set_variants_are_rank_inert():
    frame = pd.DataFrame(
        {
            "release_id": ["v2-test"], "trait_id": ["pd"],
            "study_id": ["G1"], "locus_id": ["L1"], "variant_id": ["1_1_A_G"],
            "rs_ids_json": ["[]"], "chromosome": ["1"], "position": [1],
            "reference_allele": ["A"], "alternate_allele": ["G"],
            "posterior_probability": [0.9], "is_95_credible_set": [True],
            "is_99_credible_set": [True], "log_bayes_factor": [2.0],
            "beta": [0.1], "standard_error": [0.02],
            "p_value_mantissa": [1.0], "p_value_exponent": [-8],
            "source": ["Open Targets Platform"], "source_release": ["26.06"],
            "source_record_id": ["ot:credible_set_variant:one"],
            "affects_dual_selectivity_rank": [False],
        }
    )
    validate_table("causal_genetics_credible_set_variants", frame)
    with pytest.raises(ContractError, match="cannot alter rank"):
        validate_table(
            "causal_genetics_credible_set_variants",
            frame.assign(affects_dual_selectivity_rank=True),
        )
