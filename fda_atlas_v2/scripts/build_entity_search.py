#!/usr/bin/env python3
"""Build the compact public search index for genes, drugs, and products."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.entity_search import build_entity_search  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def build(args: argparse.Namespace) -> None:
    directory = args.release_dir.expanduser().resolve()
    inputs = {
        "drug_target_edges": directory / "drug_target_edges.parquet",
        "regulatory_substance_target_evidence": (
            directory / "regulatory_substance_target_evidence.parquet"
        ),
        "active_moieties": directory / "active_moieties.parquet",
        "regulatory_substances": directory / "regulatory_substances.parquet",
        "regulatory_products": directory / "regulatory_products.parquet",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frames = {name: pd.read_parquet(path) for name, path in inputs.items()}
    search = build_entity_search(args.release_id, **frames)
    output = directory / "entity_search.parquet"
    manifest_path = directory / "entity_search_manifest.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
    atomic_parquet(search, output)
    manifest = {
        "release_id": args.release_id,
        "scope": "target, active-moiety, substance, product, brand, generic, and application search",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "counts": {
            "rows": len(search),
            "entities": int(search[["entity_type", "entity_id"]].drop_duplicates().shape[0]),
            "rows_by_entity_type": {
                str(key): int(value)
                for key, value in search["entity_type"].value_counts().sort_index().items()
            },
            "entities_by_entity_type": {
                str(key): int(value)
                for key, value in search.drop_duplicates(
                    ["entity_type", "entity_id"]
                )["entity_type"].value_counts().sort_index().items()
            },
        },
        "table": {"path": str(output), "sha256": sha256(output), "rows": len(search)},
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/entity_search.py"): sha256(
                SRC / "fda_atlas_v2/entity_search.py"
            ),
        },
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
