import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fda_atlas_v2.openfda_labels import build_match_indices, summarize_match_indices


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_openfda_label_release.py"
SPEC = importlib.util.spec_from_file_location("openfda_label_release_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_fixture(directory):
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
            {"substance_id": ["s1"], "preferred_name": ["INGREDIENT"], "name_variants": [""]}
        ),
        "active_moieties": pd.DataFrame({"moiety_id": ["m1"], "unii": ["UNII1"]}),
        "substance_active_moiety_edges": pd.DataFrame(
            {"substance_id": ["s1"], "moiety_id": ["m1"]}
        ),
    }
    records = {}
    for name, frame in frames.items():
        path = directory / f"matching_{name}.parquet"
        frame.to_parquet(path, index=False)
        records[name] = {"path": str(path), "sha256": digest(path)}
    audit = summarize_match_indices(
        build_match_indices(
            frames["regulatory_products"], frames["product_ingredients"],
            frames["regulatory_substances"], frames["active_moieties"],
            frames["substance_active_moiety_edges"],
        )
    )
    return records, audit


def compact(index):
    label_id = f"label-{index:02d}"
    return {
        "label_id": label_id,
        "set_id": f"set-{index:02d}",
        "version": "1",
        "effective_time": "20260101",
        "source_partition": f"part-{index:02d}.zip",
        "matched": True,
        "product_ids": ["p1"],
        "substance_ids": [],
        "moiety_ids": [],
        "match_methods": ["exact_application_number"],
        "match_evidence": [
            {
                "entity_type": "regulatory_product",
                "entity_id": "p1",
                "match_method": "exact_application_number",
                "source_value": "NDA1",
                "propagation_source_entity_id": "",
            }
        ],
        "openfda": {"application_number": ["NDA1"]},
        "sections": {"indications_and_usage": [f"Indication {index}"]},
    }


def test_builder_streams_verified_partitions_and_reports_identity_coverage(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    matched_dir = cache / "matched"
    matched_dir.mkdir(parents=True)
    links = tmp_path / "release"
    links.mkdir()
    pd.DataFrame({"product_id": ["p1"]}).to_parquet(
        links / "regulatory_products.parquet", index=False
    )
    pd.DataFrame({"substance_id": ["s1"]}).to_parquet(
        links / "regulatory_substances.parquet", index=False
    )
    pd.DataFrame({"moiety_id": ["m1"]}).to_parquet(
        links / "active_moieties.parquet", index=False
    )
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("pass\n")
    partitions = []
    for index in range(14):
        path = matched_dir / f"part-{index:02d}.matched.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(compact(index)) + "\n")
        partitions.append(
            {
                "url": f"https://download.open.fda.gov/drug/label/part-{index:02d}.zip",
                "source_records": 20_000 if index < 13 else 334,
                "matched_labels": 1,
                "zip_retained": False,
                "zip_sha256": "a" * 64,
                "zip_size_bytes": 100,
                "source_json_member": "drug-label.json",
                "matched_path": str(path),
                "output_sha256": digest(path),
            }
        )
    matching_inputs, matching_index_audit = matching_fixture(tmp_path)
    acquisition = {
        "status": "complete",
        "export_date": "2026-07-10",
        "partition_results": partitions,
        "matching_inputs": matching_inputs,
        "matching_input_sha256": hashlib.sha256(
            json.dumps(matching_inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "matching_index_audit": matching_index_audit,
        "pipeline_files": {str(pipeline): digest(pipeline)},
    }
    metadata = cache / "download.json"
    metadata.write_text(json.dumps({"results": {"drug": {"label": {
        "export_date": "2026-07-10",
        "partitions": [
            {"file": row["url"], "records": row["source_records"], "size_mb": "1"}
            for row in partitions
        ],
        "total_records": 260_334,
    }}}}))
    acquisition["metadata_path"] = str(metadata)
    acquisition["metadata_sha256"] = digest(metadata)
    (cache / "acquisition_manifest.json").write_text(json.dumps(acquisition))
    monkeypatch.setattr(MODULE, "large_data_preflight", lambda **kwargs: {})
    output = tmp_path / "output"
    MODULE.build(
        SimpleNamespace(
            execute=True,
            chunk_records=3,
            cache_dir=cache,
            output_dir=output,
            release_link_dir=links,
            overwrite=False,
            release_id="test",
        )
    )
    manifest = json.loads((links / "openfda_label_release_manifest.json").read_text())
    assert manifest["tables"]["regulatory_labels"]["rows"] == 14
    assert manifest["identity_coverage"]["regulatory_product"] == {
        "matched_entities": 1,
        "release_entities": 1,
        "fraction": 1.0,
    }
    assert manifest["section_coverage"] == {
        "labels_with_selected_sections": 14,
        "labels_without_selected_sections": 0,
    }
    for name in MODULE.TABLE_NAMES:
        assert (links / f"{name}.parquet").is_symlink()
        assert pd.read_parquet(links / f"{name}.parquet").shape[0] > 0
