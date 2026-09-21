import pandas as pd

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.polypharmacology import (
    summarize_portfolios,
    summarize_target_actions,
)


def _edges():
    return pd.DataFrame(
        {
            "moiety_id": ["m1", "m1", "m1", "m1"],
            "target_id": ["HGNC:1", "HGNC:1", "HGNC:2", "HGNC:3"],
            "human_gene": ["A", "A", "B", "C"],
            "action_type": ["AGONIST", "INHIBITOR", "AGONIST", "unspecified"],
            "target_scope": [
                "single_protein", "single_protein", "family_group_component",
                "single_protein",
            ],
            "source_family": ["ChEMBL", "DrugCentral", "GtoPdb", "DrugCentral"],
            "source_database": ["ChEMBL", "DrugCentral", "GtoPdb primary", "DrugCentral"],
        }
    )


def test_exact_action_conflict_and_broad_scope_are_not_conflated():
    summary = summarize_target_actions(
        _edges(), release_id="r1", entity_type="active_moiety", entity_column="moiety_id"
    )
    validate_table("drug_target_action_summary", summary)
    by_target = summary.set_index("target_id")
    assert by_target.loc["HGNC:1", "action_consensus"] == (
        "conflicting_activation_and_inhibition"
    )
    assert by_target.loc["HGNC:1", "conflict_evidence_status"] == (
        "independent_source_family_conflict"
    )
    assert by_target.loc["HGNC:1", "clinical_action_status"] == (
        "source_conflict_unadjudicated"
    )
    assert by_target.loc["HGNC:2", "action_consensus"] == "broad_scope_direction_only"
    assert by_target.loc["HGNC:2", "n_directional_exact_rows"] == 0
    assert by_target.loc["HGNC:3", "action_consensus"] == "directional_action_unavailable"
    assert by_target.loc["HGNC:3", "clinical_action_status"] == (
        "direction_unavailable_unadjudicated"
    )
    assert not summary["affects_dual_selectivity_rank"].any()


def test_portfolio_retains_zero_target_entities_and_never_claims_benefit():
    target_summary = summarize_target_actions(
        _edges(), release_id="r1", entity_type="active_moiety", entity_column="moiety_id"
    )
    coverage = pd.DataFrame(
        {
            "moiety_id": ["m1", "m2"],
            "preferred_name": ["ONE", "TWO"],
            "target_coverage_status": ["resolved_human_targets", "no_resolved_human_target"],
        }
    )
    portfolio = summarize_portfolios(
        target_summary, _edges(), coverage,
        release_id="r1", entity_type="active_moiety", entity_id_column="moiety_id",
    )
    validate_table("drug_polypharmacology_summary", portfolio)
    by_id = portfolio.set_index("entity_id")
    assert by_id.loc["m1", "polypharmacology_status"] == "target_action_conflict_present"
    assert by_id.loc["m2", "polypharmacology_status"] == "no_resolved_human_targets"
    assert by_id.loc["m2", "n_targets"] == 0
    assert set(portfolio["benefit_liability_status"]) == {
        "not_interpretable_without_trait_direction_and_complete_portfolio"
    }
    assert not portfolio["affects_dual_selectivity_rank"].any()
