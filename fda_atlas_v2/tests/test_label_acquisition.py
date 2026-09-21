import hashlib
import json

import pandas as pd
import pytest

from fda_atlas_v2.label_acquisition import verify_completed_label_acquisition
from fda_atlas_v2.openfda_labels import build_match_indices, summarize_match_indices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquisition(tmp_path):
    frames = {
        "regulatory_products": pd.DataFrame(
            {
                "product_id": ["p1"], "application_number": ["NDA1"],
                "proprietary_name": ["BRAND"], "nonproprietary_name": ["GENERIC"],
                "ingredient_text": ["INGREDIENT"],
            }
        ),
        "product_ingredients": pd.DataFrame(
            {"product_id": ["p1"], "substance_id": ["s1"]}
        ),
        "regulatory_substances": pd.DataFrame(
            {
                "substance_id": ["s1"], "preferred_name": ["INGREDIENT"],
                "name_variants": [""],
            }
        ),
        "active_moieties": pd.DataFrame(
            {"moiety_id": ["m1"], "unii": ["UNII1"]}
        ),
        "substance_active_moiety_edges": pd.DataFrame(
            {"substance_id": ["s1"], "moiety_id": ["m1"]}
        ),
    }
    matching_inputs = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        matching_inputs[name] = {"path": str(path), "sha256": digest(path)}
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("pass\n")
    matched = []
    partitions = []
    for index in range(2):
        path = tmp_path / f"matched{index}.jsonl.gz"
        path.write_bytes(f"matched-{index}".encode())
        matched.append(path)
        partitions.append(
            {
                "url": f"https://download.open.fda.gov/drug/label/part{index}.zip",
                "source_records": 5,
                "matched_labels": 2,
                "zip_retained": False,
                "zip_sha256": "a" * 64,
                "zip_size_bytes": 100,
                "source_json_member": "drug-label.json",
                "matched_path": str(path),
                "output_sha256": digest(path),
            }
        )
    index_audit = summarize_match_indices(
        build_match_indices(
            frames["regulatory_products"], frames["product_ingredients"],
            frames["regulatory_substances"], frames["active_moieties"],
            frames["substance_active_moiety_edges"],
        )
    )
    payload = {
        "status": "complete",
        "partition_results": partitions,
        "matching_inputs": matching_inputs,
        "matching_input_sha256": hashlib.sha256(
            json.dumps(matching_inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "matching_index_audit": index_audit,
        "pipeline_files": {str(pipeline): digest(pipeline)},
    }
    manifest = tmp_path / "acquisition.json"
    manifest.write_text(json.dumps(payload))
    return manifest, payload


def test_completed_acquisition_verifies_every_source_and_output(tmp_path):
    manifest, _ = acquisition(tmp_path)
    result = verify_completed_label_acquisition(
        manifest, expected_partitions=2, expected_source_records=10
    )
    assert result["source_records"] == 10
    assert result["matched_records"] == 4
    assert len(result["matched_paths"]) == 2


def test_acquisition_rejects_missing_zip_provenance(tmp_path):
    manifest, payload = acquisition(tmp_path)
    payload["partition_results"][0]["source_json_member"] = ""
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="JSON member provenance"):
        verify_completed_label_acquisition(
            manifest, expected_partitions=2, expected_source_records=10
        )


def test_acquisition_rejects_stale_matching_graph(tmp_path):
    manifest, payload = acquisition(tmp_path)
    path = tmp_path / "regulatory_products.parquet"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="matching input is absent or stale"):
        verify_completed_label_acquisition(
            manifest, expected_partitions=2, expected_source_records=10
        )
