#!/usr/bin/env python3
"""Build rank-inert donor-level target to leave-out scDRS coupling evidence."""

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

from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.target_coupling import (  # noqa: E402
    estimate_target_trait_coupling,
    validate_selectivity_primary_coverage,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/development")
    parser.add_argument(
        "--selectivity",
        type=Path,
        default=ROOT / "release/development/target_population_selectivity.parquet",
    )
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--min-donors", type=int, default=3)
    parser.add_argument("--min-cells-per-donor", type=int, default=10)
    args = parser.parse_args()
    sources = [path.expanduser().resolve() for path in args.input]
    selectivity_path = args.selectivity.expanduser().resolve()
    release_dir = args.release_dir.expanduser().resolve()
    for source in (*sources, selectivity_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    summaries = pd.concat(
        [pd.read_parquet(source) for source in sources], ignore_index=True
    )
    selectivity_full = pd.read_parquet(selectivity_path)
    validate_table("target_population_selectivity", selectivity_full)
    if set(selectivity_full["release_id"].astype(str)) != {args.release_id}:
        raise ValueError("selectivity release ID differs from requested coupling release")
    selectivity = selectivity_full[
        ~selectivity_full["population_id"].astype(str).str.contains("/", regex=False)
    ].copy()
    key_columns = ["species", "trait_id", "population_id", "target_id"]
    selectivity_keys = selectivity[key_columns].drop_duplicates()
    summaries = summaries.merge(
        selectivity_keys,
        on=["species", "trait_id", "population_id", "target_id"],
        how="inner",
        validate="many_to_one",
    )
    if summaries.empty:
        raise ValueError("coupling donor inputs do not overlap released selectivity keys")
    output = estimate_target_trait_coupling(
        summaries,
        release_id=args.release_id,
        min_donors=args.min_donors,
        min_cells_per_donor=args.min_cells_per_donor,
    )
    validate_table("target_trait_coupling", output)
    validate_selectivity_primary_coverage(
        output,
        selectivity,
        release_id=args.release_id,
    )
    path = release_dir / "target_trait_coupling.parquet"
    partial = path.with_suffix(".parquet.partial")
    output.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    manifest = {
        "release_id": args.release_id,
        "scope": "rank-inert family/subtype DA genetic-liability-program coexpression",
        "inputs": [
            {"path": str(source), "sha256": sha256(source)} for source in sources
        ],
        "selectivity_key_filter": {
            "path": str(selectivity_path),
            "keys": len(selectivity_keys),
            "sha256": sha256(selectivity_path),
            "population_scope": "families and terminal subtypes; branches excluded; Anxa1 included",
        },
        "donor_input_rows_after_selectivity_filter": len(summaries),
        "method": {
            "analysis_unit": "donor by HMoE population summary",
            "cell_pseudoreplication": False,
            "fdr_scope": "species by trait by population by source partition by leave-out method",
            "primary": "leave LD block out when mapped; otherwise leave target out",
            "source_partitions": (
                "Kamath and Siletti scoring strata form one integrated primary human "
                "analysis; mouse and macaque (Allen HMBA control) are the second and "
                "third primary species analyses"
            ),
            "affects_dual_selectivity_rank": False,
        },
        "output": {"path": str(path), "rows": len(output), "sha256": sha256(path)},
        "status_counts": output["status"].value_counts().sort_index().to_dict(),
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/target_coupling.py"): sha256(
                SRC / "fda_atlas_v2/target_coupling.py"
            ),
            str(SRC / "fda_atlas_v2/contracts.py"): sha256(
                SRC / "fda_atlas_v2/contracts.py"
            ),
        },
    }
    manifest_path = release_dir / "target_trait_coupling_manifest.json"
    manifest_partial = manifest_path.with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
