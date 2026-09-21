import pandas as pd
import pytest

from fda_atlas_v2.contexts import (
    build_pending_context_coverage,
    prepare_label_context_candidates,
)
from fda_atlas_v2.contracts import ContractError, validate_table


def test_context_coverage_is_complete_explicit_and_nonranking():
    moiety_edges = pd.DataFrame(
        {
            "moiety_id": ["m1", "m1", "m2"],
            "target_id": ["t1", "t1", "t2"],
        }
    )
    substance_edges = pd.DataFrame(
        {
            "substance_id": ["s1", "s1", "s2"],
            "target_id": ["t1", "t1", "t3"],
        }
    )
    contexts = build_pending_context_coverage("r", moiety_edges, substance_edges)
    assert len(contexts) == 44
    assert contexts[
        ["entity_type", "entity_id", "target_id", "trait_id"]
    ].drop_duplicates().shape[0] == 44
    assert set(contexts["entity_type"]) == {
        "active_moiety", "regulatory_substance"
    }
    assert set(contexts["value"]) == {"not_available"}
    assert set(contexts["direction"]) == {"unknown"}
    assert not contexts["direction_known"].any()
    assert contexts["identity_path_json"].eq("[]").all()
    assert contexts.loc[
        contexts["entity_id"].eq("m1"), "polypharmacology_target_ids_json"
    ].eq('["t1"]').all()
    assert not contexts["affects_dual_selectivity_rank"].any()


def test_label_context_candidates_preserve_drug_grain_and_complete_portfolio():
    label_edges = pd.DataFrame(
        {
            "label_id": ["l1", "l1"],
            "entity_type": ["active_moiety", "regulatory_product"],
            "entity_id": ["m1", "p1"],
            "match_method": ["exact_active_moiety_unii", "exact_application_number"],
            "source_value": ["U1", "NDA1"],
            "propagation_source_entity_id": ["", ""],
            "identity_scope": ["direct_active_moiety_match", "application_level_product_match"],
            "source_record_id": ["edge:m1", "edge:p1"],
        }
    )
    sections = pd.DataFrame(
        {
            "label_id": ["l1"],
            "section_name": ["indications_and_usage"],
            "section_index": [1],
            "section_category": ["indication_or_use"],
            "section_text_sha256": ["a" * 64],
            "source_record_id": ["section:l1:1"],
            "trait_adjudication_status": ["not_adjudicated"],
        }
    )
    moiety_edges = pd.DataFrame(
        {"moiety_id": ["m1", "m1"], "target_id": ["HGNC:2", "HGNC:1"]}
    )
    substance_edges = pd.DataFrame(
        {"substance_id": ["s1"], "target_id": ["HGNC:3"]}
    )
    candidates = prepare_label_context_candidates(
        "r", label_edges, sections, moiety_edges, substance_edges
    )
    assert len(candidates) == 1
    assert candidates.loc[0, "entity_id"] == "m1"
    assert candidates.loc[0, "polypharmacology_target_ids_json"] == '["HGNC:1","HGNC:2"]'
    assert candidates.loc[0, "n_portfolio_targets"] == 2
    assert candidates.loc[0, "identity_path_json"].startswith('[{"identity_scope"')
    assert {"target_id", "trait_id", "section_text"}.isdisjoint(candidates.columns)
    assert not candidates["affects_dual_selectivity_rank"].any()


def test_context_contract_rejects_incomplete_portfolio_or_unadjudicated_claim():
    contexts = build_pending_context_coverage(
        "r",
        pd.DataFrame({"moiety_id": ["m1", "m1"], "target_id": ["t1", "t2"]}),
        pd.DataFrame({"substance_id": [], "target_id": []}),
    )
    broken = contexts.copy()
    broken.loc[broken["target_id"].eq("t2"), "polypharmacology_target_ids_json"] = '["t1"]'
    with pytest.raises(ContractError, match="target is absent"):
        validate_table("drug_target_trait_contexts", broken)

    broken = contexts.copy()
    broken.loc[0, "value"] = "approved_indication"
    with pytest.raises(ContractError, match="adjudication method"):
        validate_table("drug_target_trait_contexts", broken)
