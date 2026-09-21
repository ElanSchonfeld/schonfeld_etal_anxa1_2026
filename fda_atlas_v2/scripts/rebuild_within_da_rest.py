#!/usr/bin/env python3
"""Recompute the within-DA selectivity axis as focal-vs-POOLED-rest-of-DA for every focal population and splice it into the target-population-selectivity table.

Run:  python scripts/rebuild_within_da_rest.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.pseudobulk import (  # noqa: E402
    SparsePseudobulkResult,
    load_sparse_pseudobulk_checkpoint,
)
from fda_atlas_v2.selectivity_pipeline import (  # noqa: E402
    chunked_strongest_competitor_selectivity,
)
from fda_atlas_v2.target_selectivity import _axis_table  # noqa: E402
from fda_atlas_v2.ranking import compute_dual_selectivity_rank  # noqa: E402
from fda_atlas_v2.contracts import validate_table  # noqa: E402

DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
PB = DATA / "derived/m2h_fda_scdrs_atlas_v2/pseudobulk"
CHECKPOINT = {
    ("human", "family"): "human_integrated_control_da_family",
    ("human", "leaf"): "human_integrated_control_da_leaf",
    ("mouse", "family"): "mouse_da_all_conditions_family",
    ("mouse", "leaf"): "mouse_da_all_conditions_leaf",
}
ANXA1_LEAVES = ("Sox6:Tafa1", "Sox6:Vcan")
KW = dict(
    gene_chunk_size=2_000,
    minimum_common_donors=3,
    minimum_cells_per_donor_population=10,
    minimum_focal_detection=0.05,
    confidence_z=1.96,
)
SHIPPED = ROOT / "release/development/target_population_selectivity.parquet"
OUTPUT = ROOT / "release/development/target_population_selectivity.rest_da.parquet"

WITHIN_COLS = [
    "within_da_effect", "within_da_ci_low", "within_da_ci_high",
    "within_da_detection_difference", "within_da_competitor_id",
    "within_da_n_common_donors", "within_da_focal_n_cells",
    "within_da_focal_detection", "within_da_p_value", "within_da_q_value",
    "within_da_competitor_loo_selection_fraction", "within_da_common_donor_ids",
    "within_da_n_eligible_comparators", "within_da_status",
]
WITHIN_FLOAT = [
    "within_da_effect", "within_da_ci_low", "within_da_ci_high",
    "within_da_detection_difference", "within_da_n_common_donors",
    "within_da_focal_n_cells", "within_da_focal_detection", "within_da_p_value",
    "within_da_q_value", "within_da_competitor_loo_selection_fraction",
    "within_da_n_eligible_comparators",
]
WITHIN_OBJECT = [
    "within_da_competitor_id", "within_da_common_donor_ids", "within_da_status",
]


def build_pooled_rest_pseudobulk(
    result: SparsePseudobulkResult,
    focal_populations: set[str],
    focal_name: str,
) -> SparsePseudobulkResult:
    """Collapse each donor into a focal row and a pooled ``rest_DA`` row."""
    md = result.metadata.reset_index(drop=True)
    pop = md["population_id"].astype(str)
    donor = md["donor_id"].astype(str)
    is_focal = pop.isin(focal_populations).to_numpy()
    new_pop = np.where(is_focal, focal_name, "rest_DA")
    keys = pd.DataFrame({"donor_id": donor.to_numpy(), "population_id": new_pop})
    unique = (
        keys.drop_duplicates()
        .sort_values(["donor_id", "population_id"], kind="stable")
        .reset_index(drop=True)
    )
    group_index = pd.MultiIndex.from_frame(unique)
    codes = group_index.get_indexer(pd.MultiIndex.from_frame(keys))
    if (codes < 0).any():
        raise ValueError("failed to assign pooled donor-population group")
    indicator = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float64), (codes, np.arange(len(codes)))),
        shape=(len(unique), len(codes)),
    )
    counts = (indicator @ result.counts).tocsr()
    detected = (indicator.astype(np.int64) @ result.detected_cells).tocsr()
    merged = unique.copy()
    merged["n_cells"] = (
        np.asarray(indicator @ md["n_cells"].to_numpy(np.float64))
        .reshape(-1)
        .astype(np.int64)
    )
    merged["library_size"] = np.asarray(
        indicator @ md["library_size"].to_numpy(np.float64)
    ).reshape(-1)
    return SparsePseudobulkResult(
        counts=counts,
        detected_cells=detected,
        metadata=merged,
        feature_indices=result.feature_indices,
    )


def load_checkpoints() -> dict:
    loaded = {}
    for (species, level), name in CHECKPOINT.items():
        loaded[(species, level)] = load_sparse_pseudobulk_checkpoint(
            PB / f"{name}.manifest.json"
        )
    return loaded


def population_levels(loaded: dict) -> dict:
    """Map (species, population_id) -> hierarchy level from checkpoint metadata."""
    levels = {}
    for (species, level), (result, _names) in loaded.items():
        for population in result.metadata["population_id"].astype(str).unique():
            key = (species, population)
            if key in levels and levels[key] != level:
                raise ValueError(f"{key} appears at multiple levels")
            levels[key] = level
    return levels


def rest_axis(loaded, species, population) -> pd.DataFrame:
    """Genome-wide within_da axis for one focal vs pooled rest of DA."""
    if population == "Anxa1":
        level = "leaf"
        focal_set = set(ANXA1_LEAVES)
    else:
        level = population_levels(loaded)[(species, population)]
        focal_set = {population}
    result, names = loaded[(species, level)]
    pb = build_pooled_rest_pseudobulk(result, focal_set, population)
    return chunked_strongest_competitor_selectivity(
        pb,
        names,
        focal_population=population,
        comparator_populations=["rest_DA"],
        **KW,
    )


def main(source: Path = SHIPPED, output: Path = OUTPUT) -> None:
    shipped = pd.read_parquet(source)
    n_rows = len(shipped)
    loaded = load_checkpoints()

    parts = []
    for species in ("human", "mouse"):
        active = shipped[
            shipped["species"].eq(species)
            & ~shipped["population_id"].astype(str).str.contains("/", regex=False)
        ]
        pops = sorted(active["population_id"].astype(str).unique())
        for population in pops:
            axis = rest_axis(loaded, species, population)
            table = _axis_table(axis, "within_da")
            table.insert(0, "species", species)
            table.insert(1, "population_id", population)
            parts.append(table)
            est = int(axis["status"].eq("estimated").sum())
            print(f"  built {species:6s} {population:16s} estimated={est}")
    new = pd.concat(parts, ignore_index=True)
    key = ["species", "population_id", "species_gene"]
    newkey = new.set_index(key).sort_index()
    if not newkey.index.is_unique:
        raise ValueError("reconstructed within_da axis keys are not unique")

    out = shipped.copy()
    idx = pd.MultiIndex.from_arrays(
        [out["species"].to_numpy(), out["population_id"].to_numpy(),
         out["species_gene"].to_numpy()]
    )
    hm = out["species"].isin(("human", "mouse")).to_numpy()
    for column in WITHIN_COLS:
        mapped = newkey[column].reindex(idx).to_numpy()
        out[column] = np.where(hm, mapped, out[column].to_numpy(dtype=object))
    eligible = out["eligible"].astype(bool).to_numpy()
    missing = out["within_da_status"].isna().to_numpy() & hm
    out.loc[missing & eligible, "within_da_status"] = "gene_not_measured_in_axis"
    out.loc[missing & ~eligible, "within_da_status"] = "not_evaluated_ineligible_mapping"

    for column in WITHIN_FLOAT:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    for column in WITHIN_OBJECT:
        out[column] = out[column].astype(object)
        out.loc[out[column].isna(), column] = None

    out["focal_n_donors"] = out[
        ["within_da_n_common_donors", "brainwide_n_common_donors"]
    ].min(axis=1, skipna=False)
    out["focal_n_cells"] = out[
        ["within_da_focal_n_cells", "brainwide_focal_n_cells"]
    ].min(axis=1, skipna=False)
    out["expression_support"] = out[
        ["within_da_focal_detection", "brainwide_focal_detection"]
    ].min(axis=1, skipna=False)

    out = compute_dual_selectivity_rank(out)
    out = out[list(shipped.columns)]

    validate_table("target_population_selectivity", out)
    if len(out) != n_rows:
        raise ValueError(f"row count changed: {len(out)} != {n_rows}")

    partial = output.with_suffix(".parquet.partial")
    out.to_parquet(partial, index=False)
    partial.replace(output)
    print(f"wrote {output}  rows={len(out)}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=SHIPPED,
        help=(
            "table supplying the (species, population, target) row skeleton; only its "
            "shape and brainwide_* columns are used, every within_da_* column is "
            "recomputed. Point this at the 22-population strict grid, not at a table "
            "produced by the axis-registry route, which covers fewer populations."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    main(arguments.source.expanduser().resolve(), arguments.output.expanduser().resolve())
