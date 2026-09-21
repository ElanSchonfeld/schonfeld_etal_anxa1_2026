#!/usr/bin/env python3
"""Stream matched openFDA JSONL into normalized, drug-grain release tables."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_atlas_v2.label_release import normalize_compact_labels  # noqa: E402
from fda_atlas_v2.label_acquisition import (  # noqa: E402
    verify_completed_label_acquisition,
)
from fda_atlas_v2.resource_safety import large_data_preflight  # noqa: E402


DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
DEFAULT_CACHE = DATA / "cache/fda/openfda_drug_label/2026-07-10"
DEFAULT_OUTPUT = DATA / "derived/m2h_fda_scdrs_atlas_v2/openfda_label_release/2026-07-10"
DEFAULT_LINKS = ROOT / "release/development"
TABLE_NAMES = (
    "regulatory_labels",
    "regulatory_label_entity_edges",
    "regulatory_label_sections",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def input_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            yield record


def link_atomic(source: Path, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial-link")
    if partial.exists() or partial.is_symlink():
        partial.unlink()
    os.symlink(source, partial)
    partial.replace(destination)


def build(args: argparse.Namespace) -> None:
    if not args.execute:
        raise RuntimeError("pass --execute after the resource preflight is safe")
    if args.chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    large_data_preflight(estimated_peak_memory_gib=2.0)
    cache = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    link_dir = args.release_link_dir.expanduser().resolve()
    acquisition_path = cache / "acquisition_manifest.json"
    if not acquisition_path.is_file():
        raise FileNotFoundError(acquisition_path)
    verified_acquisition = verify_completed_label_acquisition(acquisition_path)
    acquisition = verified_acquisition["payload"]
    export_date = str(acquisition.get("export_date", ""))
    source_release = f"openfda_drug_label_{export_date}"
    matched_paths = verified_acquisition["matched_paths"]
    expected_records = int(verified_acquisition["matched_records"])

    output_dir.mkdir(parents=True, exist_ok=True)
    link_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / f"{name}.parquet" for name in TABLE_NAMES}
    partial_paths = {
        name: path.with_suffix(".parquet.partial") for name, path in output_paths.items()
    }
    if not args.overwrite and any(
        path.exists() or partial_paths[name].exists()
        for name, path in output_paths.items()
    ):
        raise FileExistsError("label release output exists; pass --overwrite")
    for path in partial_paths.values():
        path.unlink(missing_ok=True)

    writers: dict[str, pq.ParquetWriter] = {}
    schemas: dict[str, pa.Schema] = {}
    counts = {name: 0 for name in TABLE_NAMES}
    seen_labels: set[str] = set()
    seen_edges: set[str] = set()
    seen_sections: set[tuple[str, str, int]] = set()
    labels_with_edges: set[str] = set()
    labels_with_sections: set[str] = set()
    matched_entities = {
        "regulatory_product": set(),
        "regulatory_substance": set(),
        "active_moiety": set(),
    }
    valid_entities = {
        "regulatory_product": set(
            pd.read_parquet(link_dir / "regulatory_products.parquet", columns=["product_id"])[
                "product_id"
            ].astype(str)
        ),
        "regulatory_substance": set(
            pd.read_parquet(
                link_dir / "regulatory_substances.parquet", columns=["substance_id"]
            )["substance_id"].astype(str)
        ),
        "active_moiety": set(
            pd.read_parquet(link_dir / "active_moieties.parquet", columns=["moiety_id"])[
                "moiety_id"
            ].astype(str)
        ),
    }

    def write_frame(name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in writers:
            schemas[name] = table.schema
            writers[name] = pq.ParquetWriter(
                partial_paths[name], table.schema, compression="zstd"
            )
        else:
            table = table.cast(schemas[name])
        writers[name].write_table(table, row_group_size=args.chunk_records)
        counts[name] += len(frame)

    def process_chunk(records: list[dict]) -> None:
        frames = normalize_compact_labels(args.release_id, source_release, records)
        for edge in frames[1].itertuples(index=False):
            if edge.label_entity_edge_id in seen_edges:
                raise ValueError("duplicate label entity edge")
            seen_edges.add(edge.label_entity_edge_id)
            if edge.entity_id not in valid_entities[edge.entity_type]:
                raise ValueError(
                    f"label edge has unknown {edge.entity_type}: {edge.entity_id}"
                )
            labels_with_edges.add(str(edge.label_id))
            matched_entities[str(edge.entity_type)].add(str(edge.entity_id))
        for section in frames[2].itertuples(index=False):
            key = (section.label_id, section.section_name, int(section.section_index))
            if key in seen_sections:
                raise ValueError("duplicate label section key")
            seen_sections.add(key)
            labels_with_sections.add(str(section.label_id))
        for name, frame in zip(TABLE_NAMES, frames):
            write_frame(name, frame)

    chunk = []
    observed_records = 0
    try:
        for matched_path in matched_paths:
            for record in input_records(matched_path):
                label_id = str(record.get("label_id", "")).strip()
                if label_id in seen_labels:
                    raise ValueError(f"duplicate label across partitions: {label_id}")
                seen_labels.add(label_id)
                chunk.append(record)
                observed_records += 1
                if len(chunk) < args.chunk_records:
                    continue
                process_chunk(chunk)
                chunk = []
        if chunk:
            process_chunk(chunk)
    finally:
        for writer in writers.values():
            writer.close()
    if observed_records != expected_records:
        raise ValueError(
            f"matched label count expected {expected_records:,}, observed {observed_records:,}"
        )
    if set(writers) != set(TABLE_NAMES) or counts["regulatory_labels"] != observed_records:
        raise ValueError("normalized label release is incomplete")
    if labels_with_edges != seen_labels:
        raise ValueError(
            f"normalized label release has {len(seen_labels - labels_with_edges)} labels "
            "without an exact regulatory identity edge"
        )
    for name, path in output_paths.items():
        partial_paths[name].replace(path)
        link_atomic(path, link_dir / path.name)

    manifest = {
        "release_id": args.release_id,
        "source_release": source_release,
        "scope": "normalized openFDA labels retained at drug identity and section grain",
        "input": {
            "acquisition_manifest": str(acquisition_path),
            "sha256": sha256(acquisition_path),
            "matched_partitions": len(matched_paths),
            "matched_records": observed_records,
        },
        "scientific_rules": {
            "label_text_target_attribution": False,
            "trait_adjudication": "separate required bridge",
            "affects_dual_selectivity_rank": False,
        },
        "identity_coverage": {
            entity_type: {
                "matched_entities": len(matched_entities[entity_type]),
                "release_entities": len(valid_entities[entity_type]),
                "fraction": (
                    len(matched_entities[entity_type]) / len(valid_entities[entity_type])
                    if valid_entities[entity_type] else 0.0
                ),
            }
            for entity_type in sorted(valid_entities)
        },
        "section_coverage": {
            "labels_with_selected_sections": len(labels_with_sections),
            "labels_without_selected_sections": len(seen_labels - labels_with_sections),
        },
        "pipeline_files": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/fda_atlas_v2/label_release.py",
                ROOT / "src/fda_atlas_v2/label_acquisition.py",
                ROOT / "src/fda_atlas_v2/contracts.py",
            )
        },
        "tables": {
            name: {
                "path": str(output_paths[name]),
                "release_link": str(link_dir / output_paths[name].name),
                "rows": counts[name],
                "sha256": sha256(output_paths[name]),
            }
            for name in TABLE_NAMES
        },
    }
    manifest_path = link_dir / "openfda_label_release_manifest.json"
    atomic_json(manifest, output_dir / manifest_path.name)
    atomic_json(manifest, manifest_path)
    print(json.dumps({"tables": counts, "matched_records": observed_records}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-link-dir", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--chunk-records", type=int, default=1_000)
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())
