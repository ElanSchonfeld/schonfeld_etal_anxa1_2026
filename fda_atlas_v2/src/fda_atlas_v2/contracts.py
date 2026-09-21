"""Frozen long-table contracts for the atlas v2 release."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping

import numpy as np
import pandas as pd


SCHEMA_VERSION = "2.0.0"
SPECIES = frozenset({"human", "mouse", "macaque"})
TRAITS = frozenset(
    {
        "pd",
        "scz",
        "adhd",
        "nicotine_dep",
        "bipolar",
        "chronic_pain",
        "mdd",
        "oud",
        "ocd",
        "anxiety_any",
        "fibromyalgia",
    }
)

EXPECTED_DUPLICATE_SYMBOLS = 34
EXPECTED_DUPLICATE_TRAITS = 8
RANK_STATUS = frozenset(
    {"ranked", "ineligible", "unavailable", "nonpositive_joint_selectivity"}
)


class ContractError(ValueError):
    """Raised when a release table violates its frozen contract."""


@dataclass(frozen=True)
class TableContract:
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    enum_columns: Mapping[str, frozenset[str]] = field(default_factory=dict)


def _contract(required, key, enums=None) -> TableContract:
    return TableContract(tuple(required), tuple(key), enums or {})


TABLE_CONTRACTS = {
    "source_citations": _contract(
        ["release_id", "source_id", "source_category", "source_name",
         "source_release", "citation", "doi", "pmid", "accession",
         "source_url", "license_name", "license_url", "license_scope",
         "reuse_status", "used_for", "release_tables_json",
         "rank_input_role", "affects_dual_selectivity_rank",
         "availability_status"],
        ["release_id", "source_id"],
        {
            "source_category": frozenset(
                {"primary_expression", "regulatory", "target_pharmacology",
                 "genetics", "identity", "method"}
            ),
            "reuse_status": frozenset(
                {"open_license", "public_domain", "terms_apply",
                 "controlled_terms_apply", "license_not_explicit"}
            ),
            "availability_status": frozenset({"available", "pending_external"}),
            "rank_input_role": frozenset(
                {"rank_driving_expression", "pending_rank_driving_expression",
                 "trait_population_selection", "regulatory_denominator",
                 "identity_mapping", "orthology_mapping", "supporting_evidence"}
            ),
        },
    ),
    "dataset_registry": _contract(
        ["release_id", "dataset_id", "species", "dataset_role", "assay",
         "anatomy_scope", "source_name", "source_release",
         "source_paths_json", "source_manifest_path",
         "source_manifest_sha256", "matrix_representation",
         "normalization_or_transformation", "n_cells", "n_features",
         "rank_cohort_n_cells", "rank_cohort_n_donors",
         "availability_status", "rank_role", "identity_role", "sha256",
         "checksum_status"],
        ["release_id", "dataset_id"],
        {
            "species": SPECIES,
            "availability_status": frozenset(
                {"available_verified", "inputs_ready_resource_gated",
                 "pending_external_release"}
            ),
            "dataset_role": frozenset(
                {"integrated_da_atlas", "whole_brain_comparator",
                 "taxonomy_anchor", "focal_da_atlas", "future_primary_atlas",
                 "regional_brainwide_atlas", "anatomical_breadth_sensitivity"}
            ),
            "rank_role": frozenset(
                 {"primary_within_da", "primary_brainwide",
                 "pending_primary_brainwide",
                 "brainwide_sensitivity_only"}
            ),
            "checksum_status": frozenset(
                {"manifest_recorded_content_sha256",
                 "manifest_recorded_component_set_sha256",
                 "component_content_sha256_not_computed_resource_gate",
                 "component_content_sha256_not_recomputed",
                 "unavailable_pending_release"}
            ),
        },
    ),
    "trait_registry": _contract(
        ["release_id", "trait_id", "display_name", "gene_set_version", "n_genes"],
        ["release_id", "trait_id"],
        {"trait_id": TRAITS},
    ),
    "population_registry": _contract(
        ["release_id", "species", "population_id", "population_name",
         "parent_population_id", "da_status", "definition_source"],
        ["release_id", "species", "population_id"],
        {"species": SPECIES, "da_status": frozenset({"da", "non_da", "ambiguous"})},
    ),
    "cell_membership_registry": _contract(
        ["release_id", "dataset_id", "cell_id", "species", "anatomy",
         "da_status", "population_id", "membership_source",
         "exclude_from_non_da"],
        ["release_id", "dataset_id", "cell_id"],
        {"species": SPECIES, "da_status": frozenset({"da", "non_da", "ambiguous"})},
    ),
    "cell_partition_audit": _contract(
        ["release_id", "dataset_id", "species", "anatomy", "donor_id",
         "library_id", "parent_n", "da_n", "nonda_n", "ambiguous_n",
         "intersection_n", "union_n", "partition_complete"],
        ["release_id", "dataset_id", "anatomy", "donor_id", "library_id"],
        {"species": SPECIES},
    ),
    "scdrs_population_effects": _contract(
        ["release_id", "species", "trait_id", "hierarchy_level",
         "population_id", "parent_population_id", "analysis_cohort", "effect",
         "standard_error", "ci_low", "ci_high", "n_donors", "n_studies",
         "loo_effect_min", "loo_effect_max", "status"],
        ["release_id", "species", "trait_id", "hierarchy_level",
         "population_id", "analysis_cohort"],
        {"species": SPECIES, "trait_id": TRAITS},
    ),
    "trait_population_leads": _contract(
        ["release_id", "species", "trait_id", "hierarchy_level",
         "analysis_cohort", "population_id", "lead_status", "leader_id",
         "delta_from_leader", "delta_ci_low", "delta_ci_high",
         "n_common_donors", "selection_reason"],
        ["release_id", "species", "trait_id", "hierarchy_level",
         "analysis_cohort", "population_id"],
        {"species": SPECIES, "trait_id": TRAITS},
    ),
    "target_population_selectivity": _contract(
        ["release_id", "species", "trait_id", "population_id", "target_id",
         "human_gene", "species_gene", "species_gene_identifier_type",
         "ortholog_count", "orthologs_json", "gene_mapping_status",
         "within_da_effect", "within_da_ci_low", "within_da_ci_high",
         "within_da_detection_difference", "within_da_competitor_id",
         "within_da_percentile", "within_da_p_value", "within_da_q_value",
         "within_da_competitor_loo_selection_fraction",
         "within_da_common_donor_ids", "within_da_n_eligible_comparators",
         "within_da_status",
         "brainwide_effect", "brainwide_ci_low",
         "brainwide_ci_high", "brainwide_detection_difference",
         "brainwide_competitor_id", "brainwide_percentile", "brainwide_p_value",
         "brainwide_q_value", "brainwide_competitor_loo_selection_fraction",
         "brainwide_common_donor_ids", "brainwide_n_eligible_comparators",
         "brainwide_status",
         "dual_selectivity_score", "dual_selectivity_rank", "rank_status",
         "eligible", "focal_n_donors", "focal_n_cells", "expression_support"],
        ["release_id", "species", "trait_id", "population_id", "target_id"],
        {
            "species": SPECIES,
            "trait_id": TRAITS,
            "rank_status": RANK_STATUS,
            "species_gene_identifier_type": frozenset(
                {"gene_symbol", "ensembl_gene_id"}
            ),
            "within_da_status": frozenset(
                {"estimated", "single_donor_descriptive", "insufficient_focal_detection",
                 "insufficient_comparator_support", "gene_not_measured_in_axis",
                 "not_evaluated_ineligible_mapping"}
            ),
            "brainwide_status": frozenset(
                {"estimated", "insufficient_focal_detection",
                 "insufficient_comparator_support", "gene_not_measured_in_axis",
                 "not_evaluated_ineligible_mapping", "pending_external_brainwide_rna",
                 "not_estimable_all_donor_axis", "brainwide_axis_unavailable"}
            ),
        },
    ),
    "target_trait_coupling": _contract(
        ["release_id", "species", "trait_id", "population_id", "target_id",
         "analysis_partition", "analysis_role", "method", "is_primary_method", "estimate", "standard_error",
         "ci_low", "ci_high", "p_value", "q_value", "n_donors", "n_cells",
         "n_studies", "study_donor_counts_json", "n_strata",
         "residual_degrees_of_freedom", "analysis_strata",
         "positive_donor_fraction", "loo_estimate_min", "loo_estimate_max",
         "circularity_status", "stability_status", "analysis_unit",
         "gene_mapping_status", "expression_mapping_available",
         "score_availability_status",
         "expression_representation", "score_representation", "status",
         "affects_dual_selectivity_rank"],
        ["release_id", "species", "trait_id", "population_id", "target_id",
         "analysis_partition", "method"],
        {
            "species": SPECIES,
            "trait_id": TRAITS,
            "method": frozenset(
                {"leave_target_out", "leave_ld_block_out", "leave_chromosome_out"}
            ),
            "analysis_role": frozenset({"primary", "sensitivity"}),
            "status": frozenset(
                {"estimated", "insufficient_donors", "insufficient_study_support",
                 "unstable_zero_variance",
                 "unavailable_score", "unavailable_expression_mapping"}
            ),
        },
    ),
    "causal_genetics_evidence": _contract(
        ["release_id", "trait_id", "target_id", "human_gene", "hgnc_id",
         "ensembl_gene_id", "source", "source_release", "study_id",
         "study_trait", "locus_id", "lead_variant_id", "region",
         "finemapping_method", "credible_set_confidence", "evidence_type",
         "evidence_value", "evidence_value_type", "effect_direction",
         "direction_basis", "biosample_id", "biosample_name",
         "biosample_relevance", "biosample_ontology_ancestors_json", "clpp", "h3",
         "h4", "beta_ratio_sign_average", "source_record_id", "status",
         "affects_dual_selectivity_rank"],
        ["release_id", "source_record_id"],
        {
            "trait_id": TRAITS,
            "biosample_relevance": frozenset(
                {"dopaminergic_neuron", "neuronal", "brain_tissue",
                 "nervous_system", "other", "not_available"}
            ),
        },
    ),
    "causal_genetics_credible_set_variants": _contract(
        ["release_id", "trait_id", "study_id", "locus_id", "variant_id",
         "rs_ids_json", "chromosome", "position", "reference_allele",
         "alternate_allele", "posterior_probability", "is_95_credible_set",
         "is_99_credible_set", "log_bayes_factor", "beta", "standard_error",
         "p_value_mantissa", "p_value_exponent", "source", "source_release",
         "source_record_id", "affects_dual_selectivity_rank"],
        ["release_id", "locus_id", "variant_id"],
        {"trait_id": TRAITS},
    ),
    "cross_species_effects": _contract(
        ["release_id", "trait_id", "target_id", "source_species",
         "target_species", "source_population_id", "target_population_id",
         "contrast", "source_effect", "source_ci_low", "source_ci_high",
         "source_percentile", "source_q_value", "source_competitor_id",
         "source_rank_status", "source_gene_mapping_status",
         "target_effect", "target_ci_low", "target_ci_high",
         "target_percentile", "target_q_value", "target_competitor_id",
         "target_rank_status", "target_gene_mapping_status",
         "availability_status", "direction_status", "mapping_status",
         "magnitude_comparison_status", "affects_dual_selectivity_rank"],
        ["release_id", "trait_id", "target_id", "source_species",
         "target_species", "source_population_id", "target_population_id", "contrast"],
        {
            "trait_id": TRAITS,
            "source_species": SPECIES,
            "target_species": SPECIES,
            "contrast": frozenset({"within_da", "brainwide"}),
            "availability_status": frozenset(
                {"both_available", "source_unavailable", "target_unavailable",
                 "both_unavailable"}
            ),
            "direction_status": frozenset(
                {"same_positive", "same_negative", "opposite_direction",
                 "includes_zero", "unavailable",
                 "different_population_not_comparable"}
            ),
            "mapping_status": frozenset(
                {"same_human_target_and_shared_hmoe_population",
                 "same_human_target_but_different_trait_relevant_hmoe_populations"}
            ),
            "magnitude_comparison_status": frozenset(
                {"within_species_effects_only_no_cross_species_subtraction"}
            ),
        },
    ),
    "source_row_reconciliation": _contract(
        ["release_id", "regulatory_source", "source_release", "source_row_id",
         "source_section", "record_type", "parent_source_row_id",
         "natural_key", "source_location", "source_record_sha256",
         "disposition", "product_id", "reason", "output_table", "output_id"],
        ["release_id", "regulatory_source", "source_release", "source_row_id"],
        {"disposition": frozenset({"retained", "excluded", "container"})},
    ),
    "product_ingredients": _contract(
        ["release_id", "product_id", "ingredient_index",
         "product_ingredient_id", "source_row_id", "ingredient_role",
         "ingredient_text", "substance_id", "strength", "dosage_form",
         "route", "duplicate_ordinal", "resolution_status"],
        ["release_id", "product_id", "ingredient_index"],
    ),
    "regulatory_substances": _contract(
        ["release_id", "substance_id", "preferred_name", "normalized_name",
         "identity_status", "identity_source"],
        ["release_id", "substance_id"],
    ),
    "substance_identity_edges": _contract(
        ["release_id", "source_substance_id", "target_substance_id",
         "relationship_type", "evidence_source", "source_record_id",
         "query_id", "source_name_role", "match_status", "match_method",
         "match_confidence", "gsrs_source_version"],
        ["release_id", "source_substance_id", "target_substance_id",
         "relationship_type", "evidence_source", "source_record_id"],
    ),
    "active_moieties": _contract(
        ["release_id", "moiety_id", "preferred_name", "normalization_status",
         "resolution_class", "unii", "gsrs_uuid", "linking_id",
         "substance_class", "gsrs_source_version"],
        ["release_id", "moiety_id"],
    ),
    "substance_active_moiety_edges": _contract(
        ["release_id", "substance_id", "moiety_id", "relationship_type",
         "evidence_source", "source_record_id", "query_ids",
         "gsrs_record_ids", "gsrs_source_version"],
        ["release_id", "substance_id", "moiety_id", "relationship_type",
         "evidence_source", "source_record_id"],
    ),
    "drug_target_edges": _contract(
        ["release_id", "moiety_id", "target_id", "evidence_source",
         "evidence_kind", "action_type", "directness", "target_scope",
         "confidence", "source_record_id"],
        ["release_id", "moiety_id", "target_id", "evidence_source", "source_record_id"],
    ),
    "drug_target_action_summary": _contract(
        ["release_id", "entity_type", "entity_id", "target_id", "human_gene",
         "n_source_rows", "n_exact_scope_rows", "n_directional_exact_rows",
         "n_directional_broad_rows", "n_ambiguous_exact_rows",
         "n_ambiguous_broad_rows", "activation_actions_json",
         "inhibition_actions_json", "ambiguous_actions_json",
         "activation_source_families_json", "inhibition_source_families_json",
         "source_databases_json", "target_scopes_json", "action_consensus",
         "conflict_evidence_status", "exact_action_coverage",
         "clinical_action_status",
         "affects_dual_selectivity_rank"],
        ["release_id", "entity_type", "entity_id", "target_id"],
        {
            "entity_type": frozenset({"active_moiety", "regulatory_substance"}),
            "action_consensus": frozenset(
                {"activation_only", "inhibition_only",
                 "conflicting_activation_and_inhibition",
                 "broad_scope_direction_only", "directional_action_unavailable"}
            ),
            "conflict_evidence_status": frozenset(
                {"not_conflicting", "independent_source_family_conflict",
                 "single_source_family_conflict"}
            ),
            "exact_action_coverage": frozenset(
                {"complete_directional_exact", "partial_directional_exact",
                 "no_directional_exact"}
            ),
            "clinical_action_status": frozenset(
                {"source_only_unadjudicated", "source_conflict_unadjudicated",
                 "direction_unavailable_unadjudicated"}
            ),
        },
    ),
    "drug_target_activity_evidence": _contract(
        ["release_id", "entity_type", "entity_id", "target_id", "human_gene",
         "source_record_id", "source_database", "source_release",
         "source_drug_id", "source_drug_name", "source_target_id",
         "source_target_name", "target_organism", "activity_endpoint_type",
         "reported_activity_value", "activity_value_numeric", "reported_unit",
         "activity_relation", "activity_source", "activity_source_url",
         "activity_comment", "activity_availability_status", "activity_scale",
         "comparability_status", "action_type", "target_scope", "evidence_kind",
         "affects_dual_selectivity_rank"],
        ["release_id", "entity_type", "entity_id", "target_id", "source_record_id"],
        {
            "entity_type": frozenset({"active_moiety", "regulatory_substance"}),
            "activity_availability_status": frozenset(
                {"source_reported", "not_reported_by_source"}
            ),
            "activity_scale": frozenset(
                {"source_reported_unitless", "not_reported"}
            ),
            "comparability_status": frozenset({"not_harmonized_across_assays"}),
        },
    ),
    "drug_polypharmacology_summary": _contract(
        ["release_id", "entity_type", "entity_id", "preferred_name",
         "target_coverage_status", "n_target_evidence_rows", "n_targets",
         "n_targets_activation_only", "n_targets_inhibition_only",
         "n_targets_conflicting", "n_targets_broad_scope_only",
         "n_targets_direction_unavailable", "n_targets_partial_directional",
         "polypharmacology_status", "benefit_liability_status",
         "affects_dual_selectivity_rank"],
        ["release_id", "entity_type", "entity_id"],
        {
            "entity_type": frozenset({"active_moiety", "regulatory_substance"}),
            "polypharmacology_status": frozenset(
                {"no_resolved_human_targets", "target_action_conflict_present",
                 "target_actions_fully_directional_no_conflicts",
                 "target_actions_partially_unavailable",
                 "directional_actions_unavailable"}
            ),
        },
    ),
    "regulatory_substance_target_evidence": _contract(
        ["release_id", "substance_id", "target_id", "evidence_source",
         "evidence_kind", "action_type", "directness", "target_scope",
         "confidence", "source_record_id", "bridge_status", "human_gene",
         "evidence_tier", "source_database", "source_release"],
        ["release_id", "substance_id", "target_id", "evidence_source",
         "source_record_id"],
    ),
    "regulatory_substance_target_coverage": _contract(
        ["release_id", "substance_id", "preferred_name", "identity_status",
         "has_explicit_active_moiety_relationship", "n_target_edges",
         "n_human_targets", "target_coverage_status"],
        ["release_id", "substance_id"],
    ),
    "active_moiety_target_coverage": _contract(
        ["release_id", "moiety_id", "preferred_name", "resolution_class",
         "n_target_edges", "n_human_targets", "target_coverage_status"],
        ["release_id", "moiety_id"],
    ),
    "withheld_moiety_target_attribution": _contract(
        ["release_id", "substance_id", "moiety_id", "target_id",
         "evidence_source", "source_record_id", "source_drug_id",
         "source_drug_name", "moiety_preferred_name",
         "n_moieties_for_substance", "moiety_attribution_status"],
        ["release_id", "substance_id", "moiety_id", "target_id",
         "evidence_source", "source_record_id"],
    ),
    "regulatory_products": _contract(
        ["release_id", "product_id", "regulatory_source", "source_row_id",
         "source_release", "source_record_sha256", "application_number",
         "product_number", "proprietary_name", "nonproprietary_name",
         "ingredient_text", "strength", "dosage_form", "route",
         "product_presentation", "marketing_status", "licensure_status",
         "currently_marketed", "discontinued", "revoked", "approval_date",
         "product_approval_date_availability",
         "earliest_approved_orig_submission_date", "applicant",
         "reference_product", "reference_standard", "resolution_status"],
        ["release_id", "product_id"],
    ),
    "regulatory_labels": _contract(
        ["release_id", "label_id", "set_id", "label_version", "effective_time",
         "source_release", "source_partition", "source_record_sha256",
         "application_number_json", "brand_name_json", "generic_name_json",
         "substance_name_json", "unii_json", "route_json", "product_type_json",
         "affects_dual_selectivity_rank"],
        ["release_id", "label_id"],
    ),
    "regulatory_label_entity_edges": _contract(
        ["release_id", "label_entity_edge_id", "label_id", "entity_type",
         "entity_id", "match_method", "source_value",
         "propagation_source_entity_id", "identity_scope", "evidence_source",
         "source_record_id", "affects_dual_selectivity_rank"],
        ["release_id", "label_entity_edge_id"],
        {"entity_type": frozenset(
            {"regulatory_product", "regulatory_substance", "active_moiety"}
        )},
    ),
    "regulatory_label_sections": _contract(
        ["release_id", "label_id", "section_name", "section_index",
         "section_category", "section_text", "section_text_sha256",
         "evidence_source", "source_record_id", "trait_adjudication_status",
         "affects_dual_selectivity_rank"],
        ["release_id", "label_id", "section_name", "section_index"],
    ),
    "drug_target_trait_contexts": _contract(
        ["release_id", "entity_type", "entity_id", "target_id", "trait_id", "context_type",
         "evidence_source", "value", "direction", "status", "source_record_id",
         "missing_reason", "label_id", "label_section_source_record_id",
         "label_section_name", "identity_path_json",
         "polypharmacology_target_ids_json", "adjudication_method",
         "direction_evidence_source", "direction_known",
         "affects_dual_selectivity_rank"],
        ["release_id", "entity_type", "entity_id", "target_id", "trait_id", "context_type",
         "evidence_source", "source_record_id"],
        {
            "trait_id": TRAITS,
            "entity_type": frozenset({"active_moiety", "regulatory_substance"}),
        },
    ),
    "entity_search": _contract(
        ["release_id", "entity_type", "entity_id", "display_name", "search_term",
         "source"],
        ["release_id", "entity_type", "entity_id", "search_term", "source"],
    ),
    "data_dictionary": _contract(
        ["release_id", "table_name", "column_name", "description", "dtype",
         "nullable"],
        ["release_id", "table_name", "column_name"],
    ),
}


def validate_table(table_name: str, frame: pd.DataFrame) -> None:
    """Validate required columns, keys, enums, and selectivity invariants."""

    if table_name not in TABLE_CONTRACTS:
        raise ContractError(f"unknown release table: {table_name}")
    contract = TABLE_CONTRACTS[table_name]
    missing = sorted(set(contract.required_columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{table_name}: missing required columns {missing}")

    if frame[list(contract.primary_key)].isna().any(axis=None):
        raise ContractError(f"{table_name}: null value in primary key")
    duplicate = frame.duplicated(list(contract.primary_key), keep=False)
    if duplicate.any():
        raise ContractError(
            f"{table_name}: duplicate primary key in {int(duplicate.sum())} rows"
        )

    for column, allowed in contract.enum_columns.items():
        values = set(frame[column].dropna().astype(str))
        invalid = sorted(values - set(allowed))
        if invalid:
            label = column.replace("_", " ")
            raise ContractError(f"{table_name}: invalid {label} values {invalid}")

    if table_name == "dataset_registry":
        from .dataset_registry import validate_dataset_registry_semantics

        validate_dataset_registry_semantics(frame)
    if table_name == "source_citations":
        from .source_citations import validate_source_citations_semantics

        validate_source_citations_semantics(frame)
    if table_name == "cell_partition_audit":
        from .cell_partition_audit import validate_cell_partition_audit_semantics

        validate_cell_partition_audit_semantics(frame)
    if table_name == "target_trait_coupling":
        from .target_coupling import validate_target_coupling_semantics

        validate_target_coupling_semantics(frame)
    if table_name == "drug_target_trait_contexts":
        from .contexts import validate_context_semantics

        validate_context_semantics(frame)
    if table_name == "target_population_selectivity":
        _validate_selectivity(frame)
    if table_name == "causal_genetics_credible_set_variants":
        posterior = pd.to_numeric(frame["posterior_probability"], errors="coerce")
        if posterior.isna().any() or not posterior.between(0, 1).all():
            raise ContractError(
                "causal_genetics_credible_set_variants: posterior probability outside [0, 1]"
            )
        if (
            frame["is_95_credible_set"].astype(bool)
            & ~frame["is_99_credible_set"].astype(bool)
        ).any():
            raise ContractError(
                "causal_genetics_credible_set_variants: 95 percent set is not within 99 percent set"
            )
    if table_name in {
        "causal_genetics_evidence", "causal_genetics_credible_set_variants",
        "drug_target_action_summary", "drug_target_activity_evidence",
        "drug_polypharmacology_summary",
        "cross_species_effects", "target_trait_coupling",
        "regulatory_labels", "regulatory_label_entity_edges",
        "regulatory_label_sections", "drug_target_trait_contexts",
    } and frame["affects_dual_selectivity_rank"].astype(bool).any():
        raise ContractError(f"{table_name}: supporting evidence cannot alter rank")


def _validate_selectivity(frame: pd.DataFrame) -> None:
    for column in ("human_gene", "gene_mapping_status"):
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ContractError(
                f"target_population_selectivity: {column} contains missing values"
            )
    for prefix in ("within_da", "brainwide"):
        effect = pd.to_numeric(frame[f"{prefix}_effect"], errors="coerce")
        available = frame[f"{prefix}_status"].astype(str).isin(
            {"estimated", "single_donor_descriptive"}
        )
        if not available.eq(np.isfinite(effect)).all():
            raise ContractError(
                f"target_population_selectivity: {prefix} status disagrees with effect availability"
            )

    for column in ("within_da_percentile", "brainwide_percentile"):
        values = pd.to_numeric(frame[column], errors="coerce")
        outside = values.notna() & ~values.between(0, 1)
        if outside.any():
            raise ContractError(f"target_population_selectivity: {column} outside [0, 1]")

    eligible = frame["eligible"]
    if eligible.isna().any():
        raise ContractError("target_population_selectivity: eligible contains missing values")
    eligible = eligible.astype(bool)
    expected_mapping = frame["species"].astype(str).map(
        {
            "human": "human_direct",
            "mouse": "reciprocal_one_to_one_mouse_ortholog",
            "macaque": "reciprocal_one_to_one_macaque_ortholog",
        }
    )
    expected_eligible = frame["gene_mapping_status"].astype(str).eq(expected_mapping)
    if not eligible.eq(expected_eligible).all():
        raise ContractError(
            "target_population_selectivity: eligibility disagrees with species gene mapping"
        )
    ortholog_count = pd.to_numeric(frame["ortholog_count"], errors="coerce")
    if ortholog_count.isna().any() or (ortholog_count < 0).any() or not np.equal(
        ortholog_count, np.floor(ortholog_count)
    ).all():
        raise ContractError(
            "target_population_selectivity: ortholog_count is not a nonnegative integer"
        )
    for row in frame[
        ["orthologs_json", "ortholog_count", "species_gene", "eligible"]
    ].itertuples(index=False):
        try:
            orthologs = json.loads(str(row.orthologs_json))
        except json.JSONDecodeError as exc:
            raise ContractError(
                "target_population_selectivity: orthologs_json is invalid"
            ) from exc
        if (
            not isinstance(orthologs, list)
            or any(not isinstance(value, str) or not value for value in orthologs)
            or orthologs != sorted(set(orthologs))
            or len(orthologs) != int(row.ortholog_count)
        ):
            raise ContractError(
                "target_population_selectivity: ortholog array and count disagree"
            )
        if bool(row.eligible) and (
            len(orthologs) != 1 or str(row.species_gene) != orthologs[0]
        ):
            raise ContractError(
                "target_population_selectivity: eligible species gene is not the unique ortholog"
            )
    within_effect = pd.to_numeric(frame["within_da_effect"], errors="coerce")
    brain_effect = pd.to_numeric(frame["brainwide_effect"], errors="coerce")
    finite_effects = np.isfinite(within_effect) & np.isfinite(brain_effect)
    expected_status = pd.Series(np.select(
        [~eligible, eligible & ~finite_effects],
        ["ineligible", "unavailable"],
        default="ranked",
    ), index=frame.index, dtype=object)

    group_columns = ["species", "trait_id", "population_id"]
    if frame.loc[~eligible, ["within_da_percentile", "brainwide_percentile"]].notna().any(axis=None):
        raise ContractError(
            "target_population_selectivity: ineligible rows contain axis percentiles"
        )
    for identity, group in frame.loc[eligible].groupby(
        group_columns, sort=False, dropna=False
    ):
        for effect_column, percentile_column in (
            ("within_da_effect", "within_da_percentile"),
            ("brainwide_effect", "brainwide_percentile"),
        ):
            effects = pd.to_numeric(group[effect_column], errors="coerce")
            finite = np.isfinite(effects)
            expected = effects.loc[finite].rank(method="average", pct=True).where(
                effects.loc[finite] > 0, 0.0
            )
            observed = pd.to_numeric(group[percentile_column], errors="coerce")
            if observed.loc[~finite].notna().any() or not np.allclose(
                observed.loc[finite], expected, rtol=0, atol=1e-12, equal_nan=False
            ):
                raise ContractError(
                    "target_population_selectivity: direction-aware percentile "
                    f"reconstruction failed for {identity} {percentile_column}"
                )

    measured = eligible & finite_effects
    measured_joint = np.minimum(
        pd.to_numeric(frame["within_da_percentile"], errors="coerce"),
        pd.to_numeric(frame["brainwide_percentile"], errors="coerce"),
    )
    expected_status.loc[measured & measured_joint.le(0)] = (
        "nonpositive_joint_selectivity"
    )
    if not frame["rank_status"].astype(str).eq(expected_status).all():
        raise ContractError(
            "target_population_selectivity: rank status disagrees with eligibility, "
            "axis availability, and positive joint selectivity"
        )

    ranked = frame["rank_status"].eq("ranked")
    unranked = ~ranked
    for column in ("dual_selectivity_score", "dual_selectivity_rank"):
        if frame.loc[unranked, column].notna().any():
            raise ContractError(
                f"target_population_selectivity: unranked rows contain {column}"
            )

    for identity, group in frame.loc[ranked].groupby(
        group_columns, sort=False, dropna=False
    ):
        within = pd.to_numeric(group["within_da_percentile"], errors="coerce")
        brain = pd.to_numeric(group["brainwide_percentile"], errors="coerce")
        expected_score = np.minimum(within, brain)
        score = pd.to_numeric(group["dual_selectivity_score"], errors="coerce")
        if not np.allclose(score, expected_score, rtol=0, atol=1e-12, equal_nan=False):
            raise ContractError(
                "target_population_selectivity: dual-selectivity invariant failed"
            )
        expression_support = pd.to_numeric(
            group["expression_support"], errors="coerce"
        )
        if not np.isfinite(expression_support).all():
            raise ContractError(
                "target_population_selectivity: ranked expression support is unavailable"
            )
        ordered = group.assign(
            _expected_score=expected_score,
            _expected_tiebreak=np.sqrt(within * brain),
            _expression_support=expression_support,
        ).sort_values(
            ["_expected_score", "_expected_tiebreak", "_expression_support", "target_id"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        expected_rank = pd.Series(
            np.arange(1, len(ordered) + 1), index=ordered.index, dtype=float
        ).reindex(group.index)
        observed_rank = pd.to_numeric(group["dual_selectivity_rank"], errors="coerce")
        if not np.array_equal(observed_rank.to_numpy(dtype=float), expected_rank.to_numpy()):
            raise ContractError(
                f"target_population_selectivity: persisted rank reconstruction failed for {identity}"
            )
