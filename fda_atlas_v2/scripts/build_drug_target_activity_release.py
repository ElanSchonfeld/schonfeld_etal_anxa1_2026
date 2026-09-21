#!/usr/bin/env python3
"""Build the DrugCentral target activity release tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fda_atlas_v2.activity import (  # noqa: E402
    build_drug_target_activity_evidence,
    reconstruct_drugcentral_activity,
)
from fda_atlas_v2.contracts import validate_table  # noqa: E402


DEFAULT_RELEASE = ROOT / "release/development"
DEFAULT_LEGACY_MANIFEST = (
    Path(os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent))
    / "results/fda_scdrs_targets_comprehensive"
    / "fda_comprehensive_target_registry_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def build(args: argparse.Namespace) -> None:
    release_dir = args.release_dir.expanduser().resolve()
    legacy_manifest_path = args.legacy_manifest.expanduser().resolve()
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    source_record = legacy_manifest["inputs"]["drugcentral"]
    source_path = Path(source_record["path"]).expanduser().resolve()
    if sha256(source_path) != source_record["sha256"]:
        raise ValueError("pinned DrugCentral source hash differs from legacy manifest")
    paths = {
        "moiety_edges": release_dir / "drug_target_edges.parquet",
        "substance_edges": release_dir / "regulatory_substance_target_evidence.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    interactions = pd.read_csv(
        source_path, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    source_activity = reconstruct_drugcentral_activity(interactions)
    evidence = build_drug_target_activity_evidence(
        pd.read_parquet(paths["moiety_edges"]),
        pd.read_parquet(paths["substance_edges"]),
        interactions,
        release_id=args.release_id,
    )
    validate_table("drug_target_activity_evidence", evidence)
    output = release_dir / "drug_target_activity_evidence.parquet"
    manifest_path = release_dir / "drug_target_activity_manifest.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    atomic_parquet(evidence, output)
    results = {
        "raw_drugcentral_rows": len(interactions),
        "reconstructed_source_records": len(source_activity),
        "activity_evidence_rows": len(evidence),
        "active_moiety_rows": int(evidence["entity_type"].eq("active_moiety").sum()),
        "regulatory_substance_rows": int(
            evidence["entity_type"].eq("regulatory_substance").sum()
        ),
        "source_reported_activity_rows": int(
            evidence["activity_availability_status"].eq("source_reported").sum()
        ),
        "activity_not_reported_rows": int(
            evidence["activity_availability_status"].eq(
                "not_reported_by_source"
            ).sum()
        ),
        "distinct_activity_endpoints": int(
            evidence.loc[
                evidence["activity_endpoint_type"].ne(""), "activity_endpoint_type"
            ].nunique()
        ),
    }
    manifest = {
        "release_id": args.release_id,
        "scope": "source-reported DrugCentral activity for every retained DrugCentral target edge",
        "scientific_rules": {
            "source_values_are_converted": False,
            "source_values_are_aggregated": False,
            "missing_activity_is_negative_evidence": False,
            "cross_assay_comparison_allowed": False,
            "activity_value_interpretation": "unitless source-reported value from the pinned DrugCentral export",
            "affects_dual_selectivity_rank": False,
        },
        "inputs": {
            "drugcentral": {
                "path": str(source_path),
                "sha256": sha256(source_path),
                "bytes": source_path.stat().st_size,
            },
            "legacy_manifest": {
                "path": str(legacy_manifest_path),
                "sha256": sha256(legacy_manifest_path),
            },
            **{
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in paths.items()
            },
        },
        "results": results,
        "endpoint_counts": {
            str(key): int(value)
            for key, value in evidence["activity_endpoint_type"]
            .replace("", "not_reported").value_counts().sort_index().items()
        },
        "output": {
            "path": str(output),
            "rows": len(evidence),
            "sha256": sha256(output),
        },
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/activity.py"): sha256(
                SRC / "fda_atlas_v2/activity.py"
            ),
        },
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())
