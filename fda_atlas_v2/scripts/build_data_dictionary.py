#!/usr/bin/env python3
"""Build the public machine-readable release data dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.data_dictionary import build_data_dictionary  # noqa: E402
from fda_atlas_v2 import contracts as contracts_module  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def build(args: argparse.Namespace) -> None:
    directory = args.release_dir.expanduser().resolve()
    dictionary = build_data_dictionary(args.release_id, directory)
    output = directory / "data_dictionary.parquet"
    manifest_path = directory / "data_dictionary_manifest.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
    atomic_parquet(dictionary, output)
    contracted_names = set(dictionary["table_name"]) - {"data_dictionary"}
    contracted_inputs = sorted(directory / f"{name}.parquet" for name in contracted_names)
    manifest = {
        "release_id": args.release_id,
        "scope": "all fields in currently materialized contracted release tables",
        "counts": {
            "rows": len(dictionary),
            "tables": int(dictionary["table_name"].nunique()),
            "curated_descriptions": int(dictionary["description_status"].eq("curated").sum()),
            "generated_plain_language_descriptions": int(
                dictionary["description_status"].eq("generated_plain_language").sum()
            ),
        },
        "inputs": {
            path.stem: {"path": str(path), "sha256": sha256(path)}
            for path in contracted_inputs
        },
        "table": {"path": str(output), "rows": len(dictionary), "sha256": sha256(output)},
        "pipeline_file": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "module_file": {
            "path": str(Path(build_data_dictionary.__code__.co_filename).resolve()),
            "sha256": sha256(Path(build_data_dictionary.__code__.co_filename).resolve()),
        },
        "contract_file": {
            "path": str(Path(contracts_module.__file__).resolve()),
            "sha256": sha256(Path(contracts_module.__file__).resolve()),
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
