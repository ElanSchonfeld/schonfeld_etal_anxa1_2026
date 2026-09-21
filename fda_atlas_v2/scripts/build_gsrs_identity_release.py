#!/usr/bin/env python3
"""Build the regulatory identity and active-moiety release tables."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.contracts import SCHEMA_VERSION, validate_table  # noqa: E402
from fda_atlas_v2.gsrs_identity import build_gsrs_identity_release  # noqa: E402


MANUSCRIPT = Path(
    os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent)
).expanduser()
DEFAULT_CROSSWALK = (
    MANUSCRIPT / "results/fda_scdrs_identity/fda_gsrs_identity_crosswalk.tsv"
)
DEFAULT_CROSSWALK_MANIFEST = (
    MANUSCRIPT / "results/fda_scdrs_identity/fda_gsrs_identity_manifest.json"
)
EXPECTED = {
    "regulatory_substances": 3_107,
    "gsrs_queries": 3_458,
    "resolved_queries": 3_172,
    "resolved_regulatory_substances": 2_864,
    "unresolved_regulatory_substances": 243,
    "active_moieties": 2_191,
    "substance_active_moiety_edges": 2_630,
    "substances_with_explicit_active_moiety": 2_594,
    "resolved_without_explicit_active_moiety": 270,
    "inferred_self_edges": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> None:
    substances_path = args.substances.expanduser().resolve()
    crosswalk_path = args.crosswalk.expanduser().resolve()
    crosswalk_manifest_path = args.crosswalk_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for source in (substances_path, crosswalk_path, crosswalk_manifest_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream = json.loads(crosswalk_manifest_path.read_text())
    recorded_crosswalk_hash = upstream["outputs"][crosswalk_path.name]["sha256"]
    observed_crosswalk_hash = sha256(crosswalk_path)
    if observed_crosswalk_hash != recorded_crosswalk_hash:
        raise ValueError("GSRS crosswalk hash does not match its pinned manifest")

    substances = pd.read_parquet(substances_path)
    crosswalk = pd.read_csv(
        crosswalk_path, sep="\t", dtype=str, keep_default_na=False
    )
    result = build_gsrs_identity_release(
        substances,
        crosswalk,
        release_id=args.release_id,
    )
    if result.audit != EXPECTED:
        mismatches = {
            key: {"expected": value, "observed": result.audit.get(key)}
            for key, value in EXPECTED.items()
            if result.audit.get(key) != value
        }
        raise ValueError(f"GSRS identity count mismatch: {mismatches}")

    frames = {
        "regulatory_substances": result.substances.sort_values(
            "substance_id", kind="stable"
        ).reset_index(drop=True),
        "substance_identity_edges": result.identity_edges.sort_values(
            ["source_substance_id", "query_id"], kind="stable"
        ).reset_index(drop=True),
        "active_moieties": result.active_moieties.sort_values(
            "moiety_id", kind="stable"
        ).reset_index(drop=True),
        "substance_active_moiety_edges": result.active_moiety_edges.sort_values(
            ["substance_id", "moiety_id"], kind="stable"
        ).reset_index(drop=True),
    }
    for table_name, frame in frames.items():
        validate_table(table_name, frame)

    if frames["substance_active_moiety_edges"].duplicated(
        ["substance_id", "moiety_id", "relationship_type"]
    ).any():
        raise ValueError("duplicate exact substance-to-active-moiety edge")
    status_counts = frames["regulatory_substances"]["identity_status"].value_counts()
    expected_status = {
        "resolved_explicit_active_moiety": 2_594,
        "resolved_no_explicit_active_moiety_relationship": 270,
        "unresolved_identity": 243,
    }
    if status_counts.to_dict() != expected_status:
        raise ValueError(
            f"unexpected regulatory identity status counts: {status_counts.to_dict()}"
        )

    table_paths = {
        name: output_dir / f"{name}.parquet"
        for name in frames
    }
    for name, path in table_paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild")
        atomic_parquet(frames[name], path)

    code_paths = [
        Path(__file__).resolve(),
        SRC / "fda_atlas_v2/gsrs_identity.py",
        SRC / "fda_atlas_v2/contracts.py",
    ]
    manifest = {
        "release_id": args.release_id,
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "exact regulatory substance identity and explicit GSRS active-moiety edges"
        ),
        "inputs": {
            "regulatory_substances_exact": {
                "path": str(substances_path),
                "sha256": sha256(substances_path),
            },
            "gsrs_crosswalk": {
                "path": str(crosswalk_path),
                "sha256": observed_crosswalk_hash,
            },
            "gsrs_crosswalk_manifest": {
                "path": str(crosswalk_manifest_path),
                "sha256": sha256(crosswalk_manifest_path),
            },
            "gsrs_snapshot": upstream["gsrs_snapshot"],
        },
        "identity_contract": {
            "join_key": "regulatory_name_exact_key",
            "allowed_query_roles": ["active_ingredient", "proper_name"],
            "ambiguous_or_unresolved_selected": False,
            "active_moiety_source": "explicit GSRS ACTIVE MOIETY relationships only",
            "self_edges_inferred": False,
            "unresolved_substances_retained": True,
        },
        "expected_counts": EXPECTED,
        "observed_counts": result.audit,
        "tables": {
            name: {
                "path": str(path),
                "rows": len(frames[name]),
                "columns": list(frames[name].columns),
                "sha256": sha256(path),
            }
            for name, path in table_paths.items()
        },
        "pipeline_files": {
            str(path): sha256(path)
            for path in code_paths
        },
    }
    manifest_path = output_dir / "gsrs_identity_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")
    atomic_json(manifest, manifest_path)
    print(json.dumps(result.audit, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--substances",
        type=Path,
        default=ROOT / "release/development/regulatory_substances_exact.parquet",
    )
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument(
        "--crosswalk-manifest", type=Path, default=DEFAULT_CROSSWALK_MANIFEST
    )
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
