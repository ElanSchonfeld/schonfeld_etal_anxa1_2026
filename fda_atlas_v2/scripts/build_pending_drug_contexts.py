#!/usr/bin/env python3
"""Build pending context coverage for every entity-target-trait tuple."""

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

from fda_atlas_v2.contexts import build_pending_context_coverage  # noqa: E402
from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.cns_exposure import materialize_b3db_contexts  # noqa: E402
from fda_atlas_v2.drugcentral_context import (  # noqa: E402
    materialize_drugcentral_trait_contexts,
)


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
    paths = {
        "moiety_edges": directory / "drug_target_edges.parquet",
        "substance_edges": directory / "regulatory_substance_target_evidence.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    edges = {name: pd.read_parquet(path) for name, path in paths.items()}
    contexts = build_pending_context_coverage(
        args.release_id, edges["moiety_edges"], edges["substance_edges"]
    )
    candidates = pd.DataFrame()
    if args.drugcentral_candidates.is_file():
        candidates = pd.read_parquet(args.drugcentral_candidates)
        drugcentral_contexts = materialize_drugcentral_trait_contexts(
            args.release_id,
            candidates,
            edges["moiety_edges"],
            edges["substance_edges"],
        )
        contexts = pd.concat([contexts, drugcentral_contexts], ignore_index=True)
        contexts = contexts.sort_values(
            [
                "entity_type", "entity_id", "target_id", "trait_id",
                "context_type", "evidence_source", "source_record_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
        validate_table("drug_target_trait_contexts", contexts)
    b3db_evidence = pd.DataFrame()
    if args.b3db_evidence.is_file():
        b3db_evidence = pd.read_parquet(args.b3db_evidence)
        b3db_contexts = materialize_b3db_contexts(
            args.release_id,
            b3db_evidence,
            edges["moiety_edges"],
            edges["substance_edges"],
        )
        contexts = pd.concat([contexts, b3db_contexts], ignore_index=True)
        contexts = contexts.sort_values(
            [
                "entity_type", "entity_id", "target_id", "trait_id",
                "context_type", "evidence_source", "source_record_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
        validate_table("drug_target_trait_contexts", contexts)
    output = directory / "drug_target_trait_contexts.parquet"
    manifest_path = directory / "drug_target_trait_contexts_manifest.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
    atomic_parquet(contexts, output)
    manifest = {
        "release_id": args.release_id,
        "scope": "complete pending coverage plus reviewed structured drug-grain trait context",
        "status": (
            "partial_supporting_evidence_label_adjudication_pending"
            if (
                len(candidates)
                and (directory / "label_context_adjudication_candidates.parquet").is_file()
            )
            else (
                "partial_supporting_evidence_openfda_pending"
                if len(candidates)
                else "pending_supporting_evidence_acquisition"
            )
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "rows": len(edges[name])}
            for name, path in paths.items()
        },
        "counts": {
            "rows": len(contexts),
            "active_moieties": int(
                contexts.loc[
                    contexts["entity_type"].eq("active_moiety"), "entity_id"
                ].nunique()
            ),
            "regulatory_substances": int(
                contexts.loc[
                    contexts["entity_type"].eq("regulatory_substance"), "entity_id"
                ].nunique()
            ),
            "targets": int(contexts["target_id"].nunique()),
            "entity_target_pairs": int(
                contexts[["entity_type", "entity_id", "target_id"]]
                .drop_duplicates().shape[0]
            ),
            "traits": int(contexts["trait_id"].nunique()),
            "available_context_rows": int(contexts["value"].ne("not_available").sum()),
            "drugcentral_candidate_rows": len(candidates),
            "drugcentral_context_rows": int(
                contexts["evidence_source"].astype(str).str.startswith(
                    "drugcentral-public-db-observed-"
                ).sum()
            ),
            "b3db_entity_evidence_rows": len(b3db_evidence),
            "b3db_context_rows": int(
                contexts["evidence_source"].astype(str).str.startswith(
                    "b3db-github-"
                ).sum()
            ),
        },
        "interpretation": (
            "these rows document missing supporting evidence; they do not imply a "
            "negative result and cannot alter dual-selectivity rank"
        ),
        "table": {"path": str(output), "sha256": sha256(output), "rows": len(contexts)},
        "scientific_rules": {
            "indication_jurisdiction_resolved": False,
            "fda_specific_indication_claims_created": False,
            "off_label_kept_separate": True,
            "complete_polypharmacology_retained": True,
            "b3db_identity_match": "exact_stereochemical_inchikey",
            "b3db_predictions_included": False,
            "b3db_physicochemical_proxies_included": False,
            "affects_dual_selectivity_rank": False,
            "openfda_labels_acquired": (
                directory / "label_context_adjudication_candidates.parquet"
            ).is_file(),
            "openfda_label_trait_adjudication_complete": False,
        },
        "pipeline_files": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                SRC / "fda_atlas_v2/contexts.py",
                SRC / "fda_atlas_v2/drugcentral_context.py",
                SRC / "fda_atlas_v2/cns_exposure.py",
            )
        },
    }
    if len(candidates):
        manifest["inputs"]["drugcentral_candidates"] = {
            "path": str(args.drugcentral_candidates.resolve()),
            "sha256": sha256(args.drugcentral_candidates),
            "rows": len(candidates),
        }
    if len(b3db_evidence):
        manifest["inputs"]["b3db_entity_evidence"] = {
            "path": str(args.b3db_evidence.resolve()),
            "sha256": sha256(args.b3db_evidence),
            "rows": len(b3db_evidence),
        }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument(
        "--drugcentral-candidates",
        type=Path,
        default=(
            ROOT / "release/development/supporting/drugcentral_context/"
            "drugcentral_trait_context_candidates.parquet"
        ),
    )
    parser.add_argument("--release-id", default="development")
    parser.add_argument(
        "--b3db-evidence",
        type=Path,
        default=(
            ROOT / "release/development/supporting/b3db_cns_exposure/"
            "b3db_entity_exposure_evidence.parquet"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
