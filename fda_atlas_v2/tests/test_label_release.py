import pandas as pd
import pytest

from fda_atlas_v2.label_release import normalize_compact_labels


def compact_record():
    return {
        "label_id": "label-1",
        "set_id": "set-1",
        "version": "3",
        "effective_time": "20260101",
        "source_partition": "part-1.zip",
        "matched": True,
        "product_ids": ["p1"],
        "substance_ids": ["s1"],
        "moiety_ids": ["m1"],
        "match_methods": ["exact_application_number"],
        "match_evidence": [
            {
                "entity_type": "regulatory_product",
                "entity_id": "p1",
                "match_method": "exact_application_number",
                "source_value": "NDA1",
                "propagation_source_entity_id": "",
            },
            {
                "entity_type": "regulatory_substance",
                "entity_id": "s1",
                "match_method": "propagated_from_product_match",
                "source_value": "NDA1",
                "propagation_source_entity_id": "p1",
            },
        ],
        "openfda": {
            "application_number": ["NDA1"],
            "brand_name": ["BRAND"],
            "generic_name": ["GENERIC"],
            "substance_name": ["INGREDIENT"],
            "unii": ["UNII1"],
            "route": ["ORAL"],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
        },
        "sections": {
            "indications_and_usage": ["For treatment of a specified disease."],
            "boxed_warning": ["Serious warning text."],
        },
    }


def test_label_release_preserves_drug_grain_and_never_attributes_text_to_target():
    labels, edges, sections = normalize_compact_labels(
        "r1", "openfda_label_2026-07-10", [compact_record()]
    )
    assert labels["label_id"].tolist() == ["label-1"]
    assert set(edges["entity_type"]) == {"regulatory_product", "regulatory_substance"}
    assert edges.loc[
        edges["entity_type"].eq("regulatory_product"), "identity_scope"
    ].tolist() == ["application_level_product_match"]
    assert "target_id" not in labels
    assert "target_id" not in edges
    assert "target_id" not in sections
    assert set(sections["section_category"]) == {
        "indication_or_use", "safety_or_use_constraint"
    }
    assert not labels["affects_dual_selectivity_rank"].any()
    assert not edges["affects_dual_selectivity_rank"].any()
    assert not sections["affects_dual_selectivity_rank"].any()


def test_label_release_requires_separate_trait_adjudication():
    _, _, sections = normalize_compact_labels(
        "r1", "openfda_label_2026-07-10", [compact_record()]
    )
    assert sections["trait_adjudication_status"].eq("not_adjudicated").all()


def test_label_release_rejects_duplicate_or_unknown_sections():
    record = compact_record()
    with pytest.raises(ValueError, match="duplicate openFDA"):
        normalize_compact_labels("r1", "source", [record, record])
    record = compact_record()
    record["sections"]["invented"] = ["x"]
    with pytest.raises(ValueError, match="unknown compact label section"):
        normalize_compact_labels("r1", "source", [record])


def test_supporting_label_contract_rejects_rank_mutation():
    labels, _, _ = normalize_compact_labels(
        "r1", "openfda_label_2026-07-10", [compact_record()]
    )
    labels.loc[0, "affects_dual_selectivity_rank"] = True
    from fda_atlas_v2.contracts import ContractError, validate_table

    with pytest.raises(ContractError, match="cannot alter rank"):
        validate_table("regulatory_labels", labels)
