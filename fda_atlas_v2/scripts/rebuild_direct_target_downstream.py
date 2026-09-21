#!/usr/bin/env python3
"""Rebuild edge-dependent tables for the strict direct-target FDA atlas."""

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

from fda_atlas_v2.target_evidence import build_moiety_target_coverage  # noqa: E402


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
    staged = args.release_dir.expanduser().resolve()
    source = args.source_release.expanduser().resolve()
    required = {
        "edges": staged / "drug_target_edges.parquet",
        "labels": staged / "direct_label_target_evidence.parquet",
        "moieties": source / "active_moieties.parquet",
        "selectivity": staged / "target_population_selectivity.parquet",
        "region_contrasts": source / "brainwide_region_contrasts.parquet",
        "macaque_region_contrasts": (
            ROOT
            / "results/direct_target_rebuild/"
            "macaque_brainwide_region_contrasts.parquet"
        ),
    }
    required.update(
        {
            f"{species}_registry": staged / f"{species}_target_gene_registry.parquet"
            for species in ("human", "mouse", "macaque")
        }
    )
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    edges = pd.read_parquet(required["edges"])
    moieties = pd.read_parquet(required["moieties"])
    coverage = build_moiety_target_coverage(moieties, edges)
    atomic_parquet(coverage, staged / "active_moiety_target_coverage.parquet")

    selectivity = pd.read_parquet(
        required["selectivity"], columns=["species", "population_id"]
    ).drop_duplicates()
    full_regions = pd.read_parquet(required["region_contrasts"])
    region_parts = []
    for species in ("human", "mouse"):
        populations = set(
            selectivity.loc[
                selectivity["species"].astype(str).eq(species), "population_id"
            ].astype(str)
        )
        registry = pd.read_parquet(required[f"{species}_registry"])
        genes = set(registry["species_gene"].dropna().astype(str))
        part = full_regions[
            full_regions["species"].astype(str).eq(species)
            & full_regions["population_id"].astype(str).isin(populations)
            & full_regions["gene_id"].astype(str).isin(genes)
        ].copy()
        region_parts.append(part)
    macaque_populations = set(
        selectivity.loc[
            selectivity["species"].astype(str).eq("macaque"), "population_id"
        ].astype(str)
    )
    macaque_registry = pd.read_parquet(required["macaque_registry"])
    macaque_genes = set(macaque_registry["species_gene"].dropna().astype(str))
    macaque_regions = pd.read_parquet(required["macaque_region_contrasts"])
    macaque_part = macaque_regions[
        macaque_regions["population_id"].astype(str).isin(macaque_populations)
        & macaque_regions["gene_id"].astype(str).isin(macaque_genes)
    ].copy()
    region_parts.append(macaque_part)
    regions = pd.concat(region_parts, ignore_index=True)
    regions.insert(0, "release_id", args.release_id)
    regions = regions.sort_values(
        ["species", "population_id", "gene_id"], kind="stable"
    ).reset_index(drop=True)
    if regions.duplicated(["species", "population_id", "gene_id"]).any():
        raise ValueError("FDA region contrasts contain duplicate population-gene rows")
    if regions["population_id"].astype(str).str.contains("/").any():
        raise ValueError("branch populations leaked into FDA region contrasts")
    atomic_parquet(regions, staged / "brainwide_region_contrasts_fda.parquet")

    base_manifest_path = source / "drug_target_evidence_manifest.json"
    target_manifest = json.loads(base_manifest_path.read_text())
    target_manifest["status"] = "direct_single_gene_fda_label_expansion"
    target_manifest["scope"] = (
        "pinned source evidence plus direct single-gene mechanisms verified in "
        "current FDA prescribing information"
    )
    target_manifest["inputs"]["direct_label_target_evidence"] = {
        "path": str(required["labels"]),
        "rows": len(pd.read_parquet(required["labels"])),
        "sha256": sha256(required["labels"]),
    }
    output_paths = {
        "active_moiety_target_coverage": staged / "active_moiety_target_coverage.parquet",
        "drug_target_edges": required["edges"],
        "regulatory_substance_target_coverage": source / "regulatory_substance_target_coverage.parquet",
        "regulatory_substance_target_evidence": source / "regulatory_substance_target_evidence.parquet",
        "regulatory_substance_target_evidence_bridge": source / "regulatory_substance_target_evidence_bridge.parquet",
        "withheld_moiety_target_attribution": source / "withheld_moiety_target_attribution.parquet",
    }
    target_manifest["outputs"] = {
        name: {"path": str(path), "rows": len(pd.read_parquet(path)), "sha256": sha256(path)}
        for name, path in output_paths.items()
    }
    target_manifest["counts"].update(
        {
            "moiety_target_edges": len(edges),
            "moieties_with_resolved_human_target": int(
                coverage["n_human_targets"].gt(0).sum()
            ),
            "moieties_without_resolved_human_target": int(
                coverage["n_human_targets"].eq(0).sum()
            ),
            "unique_human_targets": int(edges["target_id"].nunique()),
            "direct_label_target_edges_added": len(pd.read_parquet(required["labels"])),
        }
    )
    target_manifest["pipeline_files"] = {
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        str(SRC / "fda_atlas_v2/target_evidence.py"): sha256(
            SRC / "fda_atlas_v2/target_evidence.py"
        ),
    }
    atomic_json(target_manifest, staged / "drug_target_evidence_manifest.json")

    manifest = {
        "release_id": args.release_id,
        "scope": "edge-dependent coverage and brain-wide regional support for 533 direct targets",
        "scientific_rules": {
            "family_group_attribution_in_analytical_universe": False,
            "branch_populations_included": False,
            "macaque_brainwide_materialized": True,
            "macaque_brainwide_reference_is_cross_cohort": True,
            "macaque_brainwide_reference_is_donor_matched": False,
            "all_chiou_source_da_cells_excluded": True,
        },
        "counts": {
            "coverage_rows": len(coverage),
            "moieties_with_targets": int(coverage["n_human_targets"].gt(0).sum()),
            "region_rows": len(regions),
            "region_targets_human": int(
                regions.loc[regions["species"].eq("human"), "gene_id"].nunique()
            ),
            "region_targets_mouse": int(
                regions.loc[regions["species"].eq("mouse"), "gene_id"].nunique()
            ),
            "region_targets_macaque": int(
                regions.loc[regions["species"].eq("macaque"), "gene_id"].nunique()
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in required.items()
        },
        "outputs": {
            name: {"path": str(path), "rows": len(pd.read_parquet(path)), "sha256": sha256(path)}
            for name, path in {
                "active_moiety_target_coverage": staged / "active_moiety_target_coverage.parquet",
                "brainwide_region_contrasts_fda": staged / "brainwide_region_contrasts_fda.parquet",
            }.items()
        },
    }
    atomic_json(manifest, staged / "direct_target_downstream_manifest.json")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--source-release", type=Path, default=ROOT / "release/development"
    )
    parser.add_argument("--release-id", default="development")
    build(parser.parse_args())
