#!/usr/bin/env python3
"""Build the DA/non-DA cell partition table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.cell_partition_audit import (  # noqa: E402
    summarize_membership_parquets_with_identity,
)
from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.resource_safety import large_data_preflight  # noqa: E402


DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
DEFAULT_MANIFESTS = (
    DATA / "derived/m2h_fda_scdrs_atlas_v2/human_whb_membership/human_whb_cell_membership_manifest.json",
    DATA / "derived/m2h_fda_scdrs_atlas_v2/mouse_whb_membership/mouse_whb_cell_membership_manifest.json",
)
DEFAULT_OUTPUT = ROOT / "release/development/cell_partition_audit.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_membership_outputs(manifest_paths: list[Path]) -> tuple[list[Path], list[dict]]:
    paths = []
    provenance = []
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        outputs = manifest.get("outputs", {})
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"membership manifest has no outputs: {manifest_path}")
        pipeline = manifest.get("pipeline", {})
        pipeline_path = Path(str(pipeline.get("path", ""))).expanduser().resolve()
        if (
            not pipeline_path.is_file()
            or pipeline.get("sha256") != sha256(pipeline_path)
        ):
            raise ValueError(f"membership pipeline provenance is stale: {manifest_path}")
        pipeline_files = manifest.get("pipeline_files", {})
        if not isinstance(pipeline_files, dict) or not pipeline_files:
            raise ValueError(
                f"membership pipeline-file provenance is missing: {manifest_path}"
            )
        for path_text, digest in pipeline_files.items():
            path = Path(str(path_text)).expanduser().resolve()
            if not path.is_file() or sha256(path) != str(digest):
                raise ValueError(
                    f"membership pipeline-file provenance is stale: {path}"
                )
        manifest_rows = 0
        for key, record in sorted(outputs.items()):
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != str(record.get("sha256", "")):
                raise ValueError(f"membership output hash mismatch: {path}")
            rows = int(record.get("rows", -1))
            if rows <= 0:
                raise ValueError(f"membership output has invalid row count: {path}")
            if pq.ParquetFile(path).metadata.num_rows != rows:
                raise ValueError(f"membership output row count mismatch: {path}")
            paths.append(path)
            manifest_rows += rows
        provenance.append(
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "release_id": str(manifest.get("release_id", "")),
                "dataset_id": str(manifest.get("dataset_id", "")),
                "outputs": len(outputs),
                "rows": manifest_rows,
            }
        )
    if len({record["dataset_id"] for record in provenance}) != len(provenance):
        raise ValueError("membership manifests contain duplicate dataset IDs")
    return paths, provenance


def build(args: argparse.Namespace) -> None:
    large_data_preflight(estimated_peak_memory_gib=1.0)
    output_path = args.output.expanduser().resolve()
    manifest_path = output_path.with_name("cell_partition_audit_manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("cell partition audit output already exists")
    paths, provenance = verified_membership_outputs(args.membership_manifest)
    observed_releases = {record["release_id"] for record in provenance}
    if observed_releases != {args.release_id}:
        raise ValueError(
            f"membership release IDs {sorted(observed_releases)} do not match {args.release_id}"
        )
    audit, unique_by_dataset = summarize_membership_parquets_with_identity(
        paths, batch_size=args.batch_size
    )
    validate_table("cell_partition_audit", audit)
    parent_by_dataset = audit.groupby("dataset_id", sort=True)["parent_n"].sum().to_dict()
    da_by_dataset = audit.groupby("dataset_id", sort=True)["da_n"].sum().to_dict()
    nonda_by_dataset = audit.groupby("dataset_id", sort=True)["nonda_n"].sum().to_dict()
    ambiguous_by_dataset = (
        audit.groupby("dataset_id", sort=True)["ambiguous_n"].sum().to_dict()
    )
    expected_parent = {
        "allen_human_whb_10xv3_20240330": 3_369_219,
        "allen_mouse_wmb_10xv3_20230630": 2_349_544,
    }
    expected_da = {
        "allen_human_whb_10xv3_20240330": 2_026,
        "allen_mouse_wmb_10xv3_20230630": 8_409,
    }
    expected_nonda = {
        "allen_human_whb_10xv3_20240330": 3_367_193,
        "allen_mouse_wmb_10xv3_20230630": 2_328_133,
    }
    expected_ambiguous = {
        "allen_human_whb_10xv3_20240330": 0,
        "allen_mouse_wmb_10xv3_20230630": 13_002,
    }
    if parent_by_dataset != expected_parent:
        raise ValueError(f"whole-brain parent counts disagree: {parent_by_dataset}")
    if da_by_dataset != expected_da:
        raise ValueError(f"whole-brain DA counts disagree: {da_by_dataset}")
    if unique_by_dataset != expected_parent:
        raise ValueError(f"whole-brain unique cell counts disagree: {unique_by_dataset}")
    if nonda_by_dataset != expected_nonda:
        raise ValueError(f"whole-brain non-DA counts disagree: {nonda_by_dataset}")
    if ambiguous_by_dataset != expected_ambiguous:
        raise ValueError(
            f"whole-brain ambiguous counts disagree: {ambiguous_by_dataset}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".parquet.partial")
    audit.to_parquet(partial, index=False, compression="zstd")
    partial.replace(output_path)
    payload = {
        "release_id": args.release_id,
        "scope": "complete human and mouse donor-by-anatomy DA/non-DA partition audit",
        "inputs": provenance,
        "counts": {
            "rows": len(audit),
            "datasets": int(audit["dataset_id"].nunique()),
            "parent_by_dataset": parent_by_dataset,
            "unique_cells_by_dataset": unique_by_dataset,
            "da_by_dataset": da_by_dataset,
            "nonda_by_dataset": nonda_by_dataset,
            "ambiguous_by_dataset": ambiguous_by_dataset,
            "intersection_cells": int(audit["intersection_n"].sum()),
            "incomplete_partitions": int((~audit["partition_complete"].astype(bool)).sum()),
        },
        "rules": {
            "partition": "every parent cell is exactly one of DA, non-DA, or ambiguous",
            "cell_id_uniqueness": "global within each dataset across every membership file",
            "da_membership": "provenance registry only; no single-marker classification",
            "ambiguous_disposition": "excluded from non-DA comparator",
            "library_id": "source feature_matrix_label",
        },
        "output": {
            "path": str(output_path),
            "rows": len(audit),
            "sha256": sha256(output_path),
        },
        "pipeline": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "pipeline_files": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                SRC / "fda_atlas_v2/cell_partition_audit.py",
                SRC / "fda_atlas_v2/contracts.py",
                SRC / "fda_atlas_v2/resource_safety.py",
            )
        },
    }
    manifest_partial = manifest_path.with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(manifest_path)
    print(json.dumps(payload["counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--membership-manifest", type=Path, action="append",
        default=None,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.membership_manifest is None:
        args.membership_manifest = list(DEFAULT_MANIFESTS)
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
