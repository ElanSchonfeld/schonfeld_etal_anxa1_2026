#!/usr/bin/env python3
"""Build target-action and complete portfolio summaries for every drug identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.polypharmacology import (  # noqa: E402
    ACTION_DIRECTION,
    summarize_portfolios,
    summarize_target_actions,
)


DEFAULT_RELEASE = ROOT / "release/development"


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
    inputs = {
        "moiety_edges": release_dir / "drug_target_edges.parquet",
        "substance_edges": release_dir / "regulatory_substance_target_evidence.parquet",
        "moiety_coverage": release_dir / "active_moiety_target_coverage.parquet",
        "substance_coverage": release_dir / "regulatory_substance_target_coverage.parquet",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    moiety_edges = pd.read_parquet(inputs["moiety_edges"])
    substance_edges = pd.read_parquet(inputs["substance_edges"])
    moiety_coverage = pd.read_parquet(inputs["moiety_coverage"])
    substance_coverage = pd.read_parquet(inputs["substance_coverage"])
    moiety_targets = summarize_target_actions(
        moiety_edges,
        release_id=args.release_id,
        entity_type="active_moiety",
        entity_column="moiety_id",
    )
    substance_targets = summarize_target_actions(
        substance_edges,
        release_id=args.release_id,
        entity_type="regulatory_substance",
        entity_column="substance_id",
    )
    target_summary = pd.concat(
        [moiety_targets, substance_targets], ignore_index=True
    ).sort_values(
        ["entity_type", "entity_id", "target_id"], kind="stable"
    ).reset_index(drop=True)
    moiety_portfolios = summarize_portfolios(
        moiety_targets,
        moiety_edges,
        moiety_coverage,
        release_id=args.release_id,
        entity_type="active_moiety",
        entity_id_column="moiety_id",
    )
    substance_portfolios = summarize_portfolios(
        substance_targets,
        substance_edges,
        substance_coverage,
        release_id=args.release_id,
        entity_type="regulatory_substance",
        entity_id_column="substance_id",
    )
    portfolios = pd.concat(
        [moiety_portfolios, substance_portfolios], ignore_index=True
    ).sort_values(["entity_type", "entity_id"], kind="stable").reset_index(drop=True)
    validate_table("drug_target_action_summary", target_summary)
    validate_table("drug_polypharmacology_summary", portfolios)
    if len(moiety_portfolios) != 2_191 or len(substance_portfolios) != 3_107:
        raise ValueError("polypharmacology summaries do not retain every drug identity")
    if int(target_summary["n_source_rows"].sum()) != len(moiety_edges) + len(
        substance_edges
    ):
        raise ValueError("target action summaries do not reconcile all source evidence rows")
    expected_pairs = len(
        moiety_edges[["moiety_id", "target_id"]].drop_duplicates()
    ) + len(substance_edges[["substance_id", "target_id"]].drop_duplicates())
    if len(target_summary) != expected_pairs:
        raise ValueError("target action summaries do not reconcile entity-target pairs")
    if int(portfolios["n_targets"].sum()) != len(target_summary):
        raise ValueError("portfolio target counts do not reconcile target summaries")
    paths = {
        "target_actions": release_dir / "drug_target_action_summary.parquet",
        "portfolios": release_dir / "drug_polypharmacology_summary.parquet",
        "manifest": release_dir / "polypharmacology_action_manifest.json",
    }
    for key, frame in (("target_actions", target_summary), ("portfolios", portfolios)):
        if paths[key].exists() and not args.overwrite:
            raise FileExistsError(f"{paths[key]} exists; pass --overwrite")
        atomic_parquet(frame, paths[key])
    action_counts = target_summary["action_consensus"].value_counts().sort_index()
    conflict_counts = target_summary["conflict_evidence_status"].value_counts().sort_index()
    portfolio_counts = portfolios["polypharmacology_status"].value_counts().sort_index()
    manifest = {
        "release_id": args.release_id,
        "scope": "source-grained target actions and complete drug polypharmacology portfolios",
        "scientific_rules": {
            "exact_target_direction_scope": "single_protein only",
            "family_and_complex_direction_role": "broad_scope_support only",
            "unknown_action_is_negative_evidence": False,
            "benefit_or_liability_inferred": False,
            "action_consensus_interpretation": "unadjudicated source report",
            "source_conflicts_are_overridden": False,
            "clinical_action_adjudication": "pending complete FDA label acquisition",
            "complete_zero_target_entities_retained": True,
            "affects_dual_selectivity_rank": False,
            "action_direction_vocabulary": ACTION_DIRECTION,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "results": {
            "source_evidence_rows": len(moiety_edges) + len(substance_edges),
            "entity_target_rows": len(target_summary),
            "active_moiety_target_rows": len(moiety_targets),
            "regulatory_substance_target_rows": len(substance_targets),
            "portfolio_rows": len(portfolios),
            "active_moiety_portfolios": len(moiety_portfolios),
            "regulatory_substance_portfolios": len(substance_portfolios),
            "action_consensus_counts": {
                str(key): int(value) for key, value in action_counts.items()
            },
            "conflict_evidence_counts": {
                str(key): int(value) for key, value in conflict_counts.items()
            },
            "portfolio_status_counts": {
                str(key): int(value) for key, value in portfolio_counts.items()
            },
        },
        "tables": {
            key: {"path": str(paths[key]), "rows": len(frame), "sha256": sha256(paths[key])}
            for key, frame in (("target_actions", target_summary), ("portfolios", portfolios))
        },
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/polypharmacology.py"): sha256(
                SRC / "fda_atlas_v2/polypharmacology.py"
            ),
        },
    }
    atomic_json(manifest, paths["manifest"])
    print(json.dumps(manifest["results"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())
