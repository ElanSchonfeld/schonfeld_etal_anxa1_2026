#!/usr/bin/env python3
"""Build the source-complete pinned Drugs@FDA and Purple Book denominator."""

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
from fda_atlas_v2.regulatory_sources import (  # noqa: E402
    build_regulatory_substances,
    load_drugsfda_archive,
    load_purple_book,
)


DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
DEFAULT_DRUGSFDA = (
    DATA
    / "derived/m2h_druggable_scdrs/cache/fda/"
    "2026-07-09_drug-drugsfda-0001-of-0001.json.zip"
)
DEFAULT_PURPLE = (
    DATA
    / "cache/fda/purplebook/2026-06/"
    "purplebook-search-June-data-download.csv"
)
EXPECTED = {
    "drugsfda_applications": 29_192,
    "drugsfda_product_nodes": 51_473,
    "drugsfda_ingredient_nodes": 60_006,
    "drugsfda_products_retained": 50_028,
    "drugsfda_ingredients_retained": 58_243,
    "drugsfda_tentative_products": 1_445,
    "drugsfda_tentative_ingredients": 1_763,
    "purple_change_rows": 23,
    "purple_products_retained": 2_205,
    "purple_revoked_products": 233,
    "total_products": 52_233,
    "total_product_substance_edges": 60_448,
    "total_source_nodes": 142_899,
    "exact_regulatory_substances": 3_107,
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


def _require(label: str, observed: int) -> None:
    expected = EXPECTED[label]
    if observed != expected:
        raise ValueError(f"{label}: expected {expected}, observed {observed}")


def build(args: argparse.Namespace) -> None:
    drugs_path = args.drugsfda.expanduser().resolve()
    purple_path = args.purple_book.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for source in (drugs_path, purple_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    drugs = load_drugsfda_archive(
        drugs_path,
        release_id=args.release_id,
        source_release="2026-07-09",
    )
    purple = load_purple_book(
        purple_path,
        release_id=args.release_id,
        source_release="2026-06",
    )

    products = pd.concat([drugs.products, purple.products], ignore_index=True).sort_values(
        ["regulatory_source", "application_number", "product_number", "product_id"],
        kind="stable",
    ).reset_index(drop=True)
    ingredients = pd.concat(
        [drugs.ingredients, purple.ingredients], ignore_index=True
    ).sort_values(["product_id", "ingredient_index"], kind="stable").reset_index(drop=True)
    reconciliation = pd.concat(
        [drugs.reconciliation, purple.reconciliation], ignore_index=True
    ).sort_values(
        ["regulatory_source", "source_release", "source_row_id"], kind="stable"
    ).reset_index(drop=True)
    substances = build_regulatory_substances(
        ingredients, release_id=args.release_id
    ).sort_values("substance_id", kind="stable").reset_index(drop=True)

    _require("drugsfda_applications", drugs.audit["application_nodes"])
    _require("drugsfda_product_nodes", drugs.audit["product_nodes"])
    _require("drugsfda_ingredient_nodes", drugs.audit["ingredient_nodes"])
    _require("drugsfda_products_retained", len(drugs.products))
    _require("drugsfda_ingredients_retained", len(drugs.ingredients))
    _require("drugsfda_tentative_products", drugs.audit["tentative_products_excluded"])
    _require(
        "drugsfda_tentative_ingredients",
        drugs.audit["tentative_ingredient_nodes_excluded"],
    )
    _require("purple_change_rows", purple.audit["change_summary_rows"])
    _require("purple_products_retained", len(purple.products))
    _require("purple_revoked_products", purple.audit["revoked_products"])
    _require("total_products", len(products))
    _require("total_product_substance_edges", len(ingredients))
    _require("total_source_nodes", len(reconciliation))
    _require("exact_regulatory_substances", len(substances))

    for table_name, frame in {
        "regulatory_products": products,
        "product_ingredients": ingredients,
        "regulatory_substances": substances,
        "source_row_reconciliation": reconciliation,
    }.items():
        validate_table(table_name, frame)

    edge_counts = ingredients.groupby("product_id", sort=False).size()
    products_without_edges = products.loc[
        ~products["product_id"].isin(edge_counts.index),
        ["product_id", "resolution_status"],
    ]
    expected_exception = pd.DataFrame(
        [{
            "product_id": "drugsfda:NDA221337:001",
            "resolution_status": "no_ingredient_in_source",
        }]
    )
    if not products_without_edges.reset_index(drop=True).equals(expected_exception):
        raise ValueError(
            "unexpected included product without a product-substance edge: "
            f"{products_without_edges.to_dict('records')}"
        )

    table_paths = {
        "regulatory_products": output_dir / "regulatory_products.parquet",
        "product_ingredients": output_dir / "product_ingredients.parquet",
        "regulatory_substances": output_dir / "regulatory_substances_exact.parquet",
        "source_row_reconciliation": output_dir / "source_row_reconciliation.parquet",
    }
    frames = {
        "regulatory_products": products,
        "product_ingredients": ingredients,
        "regulatory_substances": substances,
        "source_row_reconciliation": reconciliation,
    }
    for name, path in table_paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild")
        atomic_parquet(frames[name], path)

    code_paths = [
        Path(__file__).resolve(),
        SRC / "fda_atlas_v2/regulatory_sources.py",
        SRC / "fda_atlas_v2/contracts.py",
    ]
    manifest = {
        "release_id": args.release_id,
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "source-complete Drugs@FDA and Purple Book regulatory product denominator"
        ),
        "sources": {
            "drugsfda": {
                "release": "2026-07-09",
                "path": str(drugs_path),
                "sha256": sha256(drugs_path),
                "audit": drugs.audit,
            },
            "purple_book": {
                "release": "2026-06",
                "path": str(purple_path),
                "sha256": sha256(purple_path),
                "audit": purple.audit,
            },
        },
        "transformations": {
            "drugsfda_tentative_rule": (
                "exclude only marketing_status == None (Tentative Approval)"
            ),
            "purple_full_table_rule": "parse data after the last recognized header",
            "substance_identity_rule": (
                "NFKC, HTML unescape, whitespace collapse, and casefold only; "
                "punctuation and parenthetical qualifiers are preserved"
            ),
            "product_approval_date_rule": (
                "never infer a product approval date from application submissions"
            ),
            "purple_proper_name_role": (
                "one proper_name edge per product; not asserted to be a conventional ingredient"
            ),
        },
        "expected_counts": EXPECTED,
        "observed_counts": {
            "products": len(products),
            "product_substance_edges": len(ingredients),
            "regulatory_substances": len(substances),
            "source_nodes_reconciled": len(reconciliation),
            "products_without_ingredient": len(products_without_edges),
        },
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
    manifest_path = output_dir / "regulatory_source_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["observed_counts"], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drugsfda", type=Path, default=DEFAULT_DRUGSFDA)
    parser.add_argument("--purple-book", type=Path, default=DEFAULT_PURPLE)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
