#!/usr/bin/env python3
"""Build same-target, same-trait cross-species evidence from released axis rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_atlas_v2.cross_species import assemble_cross_species_effects  # noqa: E402
from fda_atlas_v2.contracts import validate_table  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> None:
    directory = args.release_dir.expanduser().resolve()
    source = directory / "target_population_selectivity.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    selectivity = pd.read_parquet(source)
    validate_table("target_population_selectivity", selectivity)
    if set(selectivity["release_id"].astype(str)) != {args.release_id}:
        raise ValueError("selectivity release ID differs from requested cross-species release")
    result = assemble_cross_species_effects(
        selectivity,
        release_id=args.release_id,
        shared_populations_only=True,
    )
    output = directory / "cross_species_effects.parquet"
    manifest_path = directory / "cross_species_effects_manifest.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    partial = output.with_suffix(output.suffix + ".partial")
    result.to_parquet(partial, index=False, compression="zstd")
    partial.replace(output)
    manifest = {
        "release_id": args.release_id,
        "scope": "same-trait same-target cross-species evidence with explicit HMoE population comparability",
        "scientific_rules": {
            "absolute_expression_compared_across_species": False,
            "effect_difference_computed": False,
            "within_species_effects_and_percentiles_retained": True,
            "direction_compared_only_for_identical_hmoe_population_ids": True,
            "shared_hmoe_populations_only": True,
            "different_population_cross_products_materialized": False,
            "figure_or_supporting_evidence_reranks_targets": False,
        },
        "input": {"path": str(source), "rows": len(selectivity), "sha256": sha256(source)},
        "counts": {
            "rows": len(result),
            "traits": int(result["trait_id"].nunique()),
            "targets": int(result["target_id"].nunique()),
            "species_pairs": int(
                result[["source_species", "target_species"]].drop_duplicates().shape[0]
            ),
            "both_available_rows": int(
                result["availability_status"].eq("both_available").sum()
            ),
            "both_available_within_da_rows": int(
                (
                    result["availability_status"].eq("both_available")
                    & result["contrast"].eq("within_da")
                ).sum()
            ),
            "both_available_brainwide_rows": int(
                (
                    result["availability_status"].eq("both_available")
                    & result["contrast"].eq("brainwide")
                ).sum()
            ),
            "opposite_direction_rows": int(
                result["direction_status"].eq("opposite_direction").sum()
            ),
            "different_population_not_comparable_rows": int(
                result["direction_status"].eq(
                    "different_population_not_comparable"
                ).sum()
            ),
        },
        "output": {"path": str(output), "rows": len(result), "sha256": sha256(output)},
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(ROOT / "src/fda_atlas_v2/cross_species.py"): sha256(
                ROOT / "src/fda_atlas_v2/cross_species.py"
            ),
            str(ROOT / "src/fda_atlas_v2/contracts.py"): sha256(
                ROOT / "src/fda_atlas_v2/contracts.py"
            ),
        },
    }
    manifest_partial = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(manifest_path)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())
