"""Fail-closed verification of a completed openFDA label acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .openfda_labels import (
    build_match_indices,
    summarize_match_indices,
    validate_download_metadata,
)


MATCHING_INPUT_TABLES = (
    "regulatory_products",
    "product_ingredients",
    "regulatory_substances",
    "active_moieties",
    "substance_active_moiety_edges",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_completed_label_acquisition(
    manifest_path: Path,
    *,
    expected_partitions: int | None = None,
    expected_source_records: int | None = None,
) -> dict:
    """Verify all matched partitions, source denominators, inputs, and code hashes."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != "complete":
        raise ValueError("openFDA label acquisition is not complete")
    partitions = payload.get("partition_results", [])
    metadata_path = Path(str(payload.get("metadata_path", ""))).expanduser().resolve()
    if expected_partitions is None or expected_source_records is None:
        if (
            not metadata_path.is_file()
            or sha256(metadata_path) != payload.get("metadata_sha256")
        ):
            raise ValueError("openFDA pinned download metadata is absent or hash-invalid")
        audit = validate_download_metadata(
            json.loads(metadata_path.read_text()), str(payload.get("export_date", ""))
        )
        expected_partitions = audit["n_partitions"]
        expected_source_records = audit["total_records"]
        expected_by_url = {
            str(row["file"]): int(row["records"]) for row in audit["partitions"]
        }
        observed_by_url = {
            str(row.get("url", "")): int(row.get("source_records", -1))
            for row in partitions
        }
        if observed_by_url != expected_by_url:
            raise ValueError("openFDA partition record counts differ from pinned metadata")
    assert expected_partitions is not None and expected_source_records is not None
    if len(partitions) != expected_partitions:
        raise ValueError(
            f"expected {expected_partitions} openFDA label partitions, observed {len(partitions)}"
        )
    urls = [str(record.get("url", "")) for record in partitions]
    if any(not url.startswith("https://download.open.fda.gov/drug/label/") for url in urls):
        raise ValueError("openFDA acquisition contains an unexpected partition URL")
    if len(urls) != len(set(urls)):
        raise ValueError("openFDA acquisition contains duplicate partition URLs")

    source_records = 0
    matched_records = 0
    matched_paths = []
    for record in partitions:
        source_count = int(record.get("source_records", -1))
        matched_count = int(record.get("matched_labels", -1))
        if source_count <= 0 or matched_count < 0 or matched_count > source_count:
            raise ValueError("openFDA partition source or matched count is invalid")
        source_records += source_count
        matched_records += matched_count
        if record.get("zip_retained") is not False:
            raise ValueError("openFDA temporary source ZIP was not deleted")
        zip_digest = str(record.get("zip_sha256", ""))
        if len(zip_digest) != 64 or int(record.get("zip_size_bytes", 0)) <= 0:
            raise ValueError("openFDA source ZIP provenance is incomplete")
        if not str(record.get("source_json_member", "")).endswith(".json"):
            raise ValueError("openFDA source JSON member provenance is incomplete")
        path = Path(str(record.get("matched_path", ""))).expanduser().resolve()
        if not path.is_file() or sha256(path) != record.get("output_sha256"):
            raise ValueError(f"matched label partition is absent or hash-invalid: {path}")
        matched_paths.append(path)
    if source_records != expected_source_records:
        raise ValueError(
            f"openFDA source record denominator expected {expected_source_records}, "
            f"observed {source_records}"
        )
    if len(matched_paths) != len(set(matched_paths)):
        raise ValueError("openFDA acquisition reuses a matched output path")

    matching_inputs = payload.get("matching_inputs", {})
    encoded = json.dumps(
        matching_inputs, sort_keys=True, separators=(",", ":")
    ).encode()
    if not matching_inputs or hashlib.sha256(encoded).hexdigest() != payload.get(
        "matching_input_sha256"
    ):
        raise ValueError("openFDA regulatory matching identity is invalid")
    if set(matching_inputs) != set(MATCHING_INPUT_TABLES):
        raise ValueError("openFDA acquisition does not bind the complete matching graph")
    matching_frames = {}
    for name, record in sorted(matching_inputs.items()):
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"openFDA matching input is absent or stale: {name}")
        matching_frames[name] = pd.read_parquet(path)
    observed_index_audit = summarize_match_indices(
        build_match_indices(
            matching_frames["regulatory_products"],
            matching_frames["product_ingredients"],
            matching_frames["regulatory_substances"],
            matching_frames["active_moieties"],
            matching_frames["substance_active_moiety_edges"],
        )
    )
    if payload.get("matching_index_audit") != observed_index_audit:
        raise ValueError("openFDA acquisition matching-index audit is absent or stale")

    pipeline_files = payload.get("pipeline_files", {})
    if not pipeline_files:
        raise ValueError("openFDA acquisition pipeline hashes are missing")
    for path_text, digest in sorted(pipeline_files.items()):
        path = Path(path_text).expanduser().resolve()
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"openFDA acquisition pipeline is absent or stale: {path}")
    return {
        "payload": payload,
        "matched_paths": matched_paths,
        "source_records": source_records,
        "matched_records": matched_records,
        "matching_index_audit": observed_index_audit,
    }
