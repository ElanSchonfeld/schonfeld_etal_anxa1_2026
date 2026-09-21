#!/usr/bin/env python3
"""Build the frozen trait and available DA-population release registries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.contracts import TRAITS  # noqa: E402
from fda_atlas_v2.registries import (  # noqa: E402
    build_gene_set_duplicate_audit,
    build_population_registry,
    build_trait_registry,
    sha256,
)


MANUSCRIPT = Path(
    os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent)
).expanduser()
DEFAULT_GENE_SETS = MANUSCRIPT / "scdrs_final/gs/per_trait"
DEFAULT_GWAS = MANUSCRIPT / "scdrs_final/gwas_manifest.json"
DEFAULT_MAGMA = MANUSCRIPT / "scdrs_final/metadata/magma_params.json"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    effects_path = args.population_effects.expanduser().resolve()
    gwas_path = args.gwas_manifest.expanduser().resolve()
    magma_path = args.magma_manifest.expanduser().resolve()
    gene_set_dir = args.gene_set_dir.expanduser().resolve()
    gene_sets = {trait: gene_set_dir / f"{trait}.gs" for trait in TRAITS}
    for path in [effects_path, gwas_path, magma_path, *gene_sets.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    gwas = json.loads(gwas_path.read_text())
    magma = json.loads(magma_path.read_text())
    effects = pd.read_parquet(effects_path)
    traits = build_trait_registry(
        args.release_id,
        gene_sets,
        gwas,
        magma_version=str(magma["magma_version"]),
    )
    duplicate_audit = build_gene_set_duplicate_audit(args.release_id, gene_sets)
    populations = build_population_registry(args.release_id, effects)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "trait_registry": traits,
        "population_registry": populations,
        "gene_set_duplicate_audit": duplicate_audit,
    }
    paths = {name: output_dir / f"{name}.parquet" for name in tables}
    for name, path in paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild")
        atomic_parquet(tables[name], path)

    manifest = {
        "release_id": args.release_id,
        "scope": "frozen 11-trait and currently available DA population registries",
        "inputs": {
            "population_effects": {"path": str(effects_path), "sha256": sha256(effects_path)},
            "gwas_manifest": {"path": str(gwas_path), "sha256": sha256(gwas_path)},
            "magma_manifest": {"path": str(magma_path), "sha256": sha256(magma_path)},
            "gene_sets": {
                trait: {"path": str(path), "sha256": sha256(path)}
                for trait, path in sorted(gene_sets.items())
            },
        },
        "method": {
            "trait_gene_sets": "parse exact weighted scDRS input rows",
            "population_source": "unique species and hierarchy definitions in population effects",
            "macaque_status": (
                "Allen HMBA rhesus within-DA populations with a DA-excluded "
                "Chiou cross-cohort brain-wide reference"
            ),
        },
        "counts": {
            "traits": len(traits),
            "trait_genes": dict(zip(traits["trait_id"], traits["n_genes"])),
            "populations": len(populations),
            "duplicate_gene_symbols": len(duplicate_audit),
            "traits_with_duplicate_gene_symbols": int(
                duplicate_audit["trait_id"].nunique()
            ),
            "populations_by_species": {
                str(key): int(value)
                for key, value in populations["species"].value_counts().sort_index().items()
            },
        },
        "tables": {
            name: {"path": str(path), "rows": len(tables[name]), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/registries.py"): sha256(
                SRC / "fda_atlas_v2/registries.py"
            ),
        },
    }
    manifest_path = output_dir / "release_registries_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--gene-set-dir", type=Path, default=DEFAULT_GENE_SETS)
    parser.add_argument("--gwas-manifest", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--magma-manifest", type=Path, default=DEFAULT_MAGMA)
    parser.add_argument(
        "--population-effects",
        type=Path,
        default=ROOT / "release/development/scdrs_population_effects.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
