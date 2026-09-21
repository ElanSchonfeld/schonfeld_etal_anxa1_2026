import io
import json

import pandas as pd

from fda_atlas_v2.openfda_labels import (
    build_match_indices,
    iter_json_results,
    match_label_record,
    summarize_match_indices,
    validate_download_metadata,
)


def test_streaming_results_parser_handles_chunk_boundaries():
    payload = json.dumps(
        {"meta": {"x": 1}, "results": [{"id": 1, "name": "café"}, {"id": 2}]},
        ensure_ascii=False,
    )
    records = list(iter_json_results(io.BytesIO(payload.encode()), chunk_size=7))
    assert records == [{"id": 1, "name": "café"}, {"id": 2}]


def test_label_match_prefers_exact_application_and_unii_links():
    indices = build_match_indices(
        pd.DataFrame(
            {
                "product_id": ["p1"], "application_number": ["NDA1"],
                "proprietary_name": ["BRAND"], "nonproprietary_name": ["GENERIC"],
                "ingredient_text": ["INGREDIENT"],
            }
        ),
        pd.DataFrame({"product_id": ["p1"], "substance_id": ["s1"]}),
        pd.DataFrame(
            {"substance_id": ["s1"], "preferred_name": ["INGREDIENT"], "name_variants": [""]}
        ),
        pd.DataFrame({"moiety_id": ["m1"], "unii": ["UNII1"]}),
        pd.DataFrame({"substance_id": ["s1"], "moiety_id": ["m1"]}),
    )
    match = match_label_record(
        {"openfda": {"application_number": ["NDA1"], "unii": ["UNII1"]}},
        indices,
    )
    assert match["matched"]
    assert match["product_ids"] == ["p1"]
    assert match["substance_ids"] == ["s1"]
    assert match["moiety_ids"] == ["m1"]
    assert set(match["match_methods"]) == {
        "exact_application_number", "exact_active_moiety_unii"
    }
    evidence = {
        (row["entity_type"], row["entity_id"], row["match_method"],
         row["propagation_source_entity_id"])
        for row in match["match_evidence"]
    }
    assert ("regulatory_product", "p1", "exact_application_number", "") in evidence
    assert ("active_moiety", "m1", "exact_active_moiety_unii", "") in evidence
    assert (
        "regulatory_substance", "s1", "propagated_from_product_match", "p1"
    ) in evidence
    assert (
        "regulatory_substance", "s1",
        "propagated_from_active_moiety_match", "m1"
    ) in evidence


def test_name_matching_preserves_qualifier_words_and_withholds_collisions():
    indices = build_match_indices(
        pd.DataFrame(
            {
                "product_id": ["p1", "p2"],
                "application_number": ["NDA1", "NDA2"],
                "proprietary_name": ["Albumin Human", "Albumin (Human)"],
                "nonproprietary_name": ["PRODUCT A", "PRODUCT B"],
                "ingredient_text": ["Albumin Human", "Albumin (Human)"],
            }
        ),
        pd.DataFrame(
            {"product_id": ["p1", "p2"], "substance_id": ["s1", "s2"]}
        ),
        pd.DataFrame(
            {
                "substance_id": ["s1", "s2", "s3", "s4"],
                "preferred_name": [
                    "Albumin Human",
                    "Albumin (Human)",
                    "Antihemophilic Factor (Human)",
                    "Antihemophilic Factor (Recombinant)",
                ],
                "name_variants": ["", "", "", ""],
            }
        ),
        pd.DataFrame({"moiety_id": ["m1"], "unii": ["UNII1"]}),
        pd.DataFrame({"substance_id": ["s1"], "moiety_id": ["m1"]}),
    )
    ambiguous = match_label_record(
        {"openfda": {"substance_name": ["ALBUMIN HUMAN"]}}, indices
    )
    assert not ambiguous["matched"]

    qualified = match_label_record(
        {
            "openfda": {
                "substance_name": ["ANTIHEMOPHILIC FACTOR HUMAN"]
            }
        },
        indices,
    )
    assert qualified["substance_ids"] == ["s3"]
    assert qualified["match_methods"] == [
        "unique_punctuation_insensitive_substance_name_to_substance"
    ]
    audit = summarize_match_indices(indices)
    assert audit["ambiguous_substance_name_keys_withheld"] == 1
    assert audit["ambiguous_product_name_keys_withheld"] == 1
    assert audit["ambiguous_active_moiety_unii_keys"] == 0


def test_download_metadata_reconciles_partition_counts():
    payload = {
        "results": {"drug": {"label": {
            "export_date": "2026-07-10", "total_records": 3,
            "partitions": [
                {"file": "https://download.open.fda.gov/drug/label/a.zip", "records": 2, "size_mb": "1.0"},
                {"file": "https://download.open.fda.gov/drug/label/b.zip", "records": 1, "size_mb": "0.5"},
            ],
        }}}
    }
    audit = validate_download_metadata(payload, "2026-07-10")
    assert audit["total_records"] == 3
    assert audit["total_size_mb"] == 1.5
