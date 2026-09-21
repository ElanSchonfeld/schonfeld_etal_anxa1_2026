#!/usr/bin/env python3
"""Build the public source-citation and reuse-terms registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.source_citations import (  # noqa: E402
    build_source_citations,
    validate_source_citations_semantics,
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
    os.replace(partial, path)


def atomic_parquet(frame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def build(args: argparse.Namespace) -> None:
    config = args.config.expanduser().resolve()
    release = args.release_dir.expanduser().resolve()
    output = release / "source_citations.parquet"
    manifest_path = release / "source_citations_manifest.json"
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("source citation outputs exist; pass --overwrite")
    records = json.loads(config.read_text())
    frame = build_source_citations(args.release_id, records)
    validate_table("source_citations", frame)
    validate_source_citations_semantics(frame)
    atomic_parquet(frame, output)
    pipeline_files = [
        Path(__file__).resolve(),
        ROOT / "src/fda_atlas_v2/source_citations.py",
        ROOT / "src/fda_atlas_v2/contracts.py",
    ]
    manifest = {
        "release_id": args.release_id,
        "schema_version": "2.0.0",
        "scope": "all external data, identity, pharmacology, genetics, and rank-driving method sources",
        "input": {"path": str(config), "sha256": sha256(config)},
        "pipeline_files": {
            str(path): sha256(path) for path in pipeline_files
        },
        "counts": {
            "sources": len(frame),
            "rank_driving_expression_sources": int(
                frame["rank_input_role"].isin(
                    ["rank_driving_expression", "pending_rank_driving_expression"]
                ).sum()
            ),
            "pending_external_sources": int(
                frame["availability_status"].eq("pending_external").sum()
            ),
        },
        "output": {
            "path": str(output), "rows": len(frame), "sha256": sha256(output)
        },
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/source_citations.json"
    )
    parser.add_argument(
        "--release-dir", type=Path, default=ROOT / "release/development"
    )
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
