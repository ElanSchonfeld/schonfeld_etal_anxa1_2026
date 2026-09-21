#!/usr/bin/env python3
"""Map the 823-cell Siletti DA set to WHB cell labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import cellxgene_census
import numpy as np
import pandas as pd


CENSUS_VERSION = "2023-12-15"
HERE = Path(__file__).resolve().parent.parent
INPUT_ROOT = Path(os.environ.get("DOPABASE_INPUT_DIR", HERE / "inputs"))
SILETTI_INPUT_DIR = Path(
    os.environ.get("DOPABASE_SILETTI_INPUT_DIR", INPUT_ROOT / "siletti")
)

SELECTED_DA = HERE / "data" / "siletti_da_neurons.h5ad"
SOURCE_CACHE = Path(
    os.environ.get(
        "DOPABASE_CENSUS_CACHE",
        INPUT_ROOT / "cellxgene_census" / CENSUS_VERSION / "h5ads",
    )
)
WHB_METADATA = Path(
    os.environ.get(
        "DOPABASE_WHB_METADATA",
        INPUT_ROOT / "allen_whb" / "WHB-10Xv3" / "20241115" / "cell_metadata.csv",
    )
)
BROAD_CANDIDATES = Path(
    os.environ.get(
        "DOPABASE_SILETTI_BROAD_H5AD",
        SILETTI_INPUT_DIR / "Siletti_Dopamine_Neurons.h5ad",
    )
)
AUTHOR_DA = Path(
    os.environ.get(
        "DOPABASE_SILETTI_REFERENCE_H5AD",
        SILETTI_INPUT_DIR / "Siletti_DA_raw_counts.h5ad",
    )
)
WHB_TAXONOMY = Path(
    os.environ.get(
        "DOPABASE_WHB_TAXONOMY",
        INPUT_ROOT
        / "allen_whb"
        / "WHB-taxonomy"
        / "20240330"
        / "cluster_to_cluster_annotation_membership.csv",
    )
)
OUTPUT = HERE / "results" / "siletti_da_whb_crosswalk.csv"
EXCLUSION_OUTPUT = HERE / "results" / "siletti_whb_da_exclusion_registry.csv"
MANIFEST = HERE / "results" / "siletti_da_whb_crosswalk_manifest.json"

CENSUS_COLUMNS = (
    "soma_joinid",
    "dataset_id",
    "donor_id",
    "cell_type",
    "raw_sum",
    "tissue",
)
WHB_COLUMNS = (
    "cell_label",
    "feature_matrix_label",
    "donor_label",
    "cluster_alias",
    "region_of_interest_label",
    "anatomical_division_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--metadata-chunk-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected() -> tuple[pd.DataFrame, list[str]]:
    selected = ad.read_h5ad(SELECTED_DA, backed="r")
    try:
        query_ids = pd.to_numeric(selected.obs_names, errors="raise").astype("int64")
        frame = pd.DataFrame(
            {
                "census_query_row": query_ids,
                "saved_dataset_id": selected.obs["dataset_id"].astype(str).to_numpy(),
                "saved_donor_id": selected.obs["donor_id"].astype(str).to_numpy(),
                "saved_dissection": selected.obs["dissection"].astype(str).to_numpy(),
                "saved_raw_sum": selected.obs["raw_sum"].to_numpy(dtype=float),
                "saved_da_score": selected.obs["DA_score"].to_numpy(dtype=float),
            }
        )
        dataset_ids = selected.obs["dataset_id"].astype(str).drop_duplicates().tolist()
    finally:
        selected.file.close()
    if len(frame) != 823 or frame["census_query_row"].nunique() != 823:
        raise ValueError("the frozen Siletti DA object is not the expected 823 unique rows")
    return frame, dataset_ids


def ensure_source_h5ads(dataset_ids: list[str], download_missing: bool) -> None:
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    for dataset_id in dataset_ids:
        path = SOURCE_CACHE / f"{dataset_id}.h5ad"
        if path.exists():
            continue
        if not download_missing:
            raise FileNotFoundError(
                f"missing {path}; rerun with --download-missing to fetch the pinned source H5AD"
            )
        cellxgene_census.download_source_h5ad(
            dataset_id,
            str(path),
            census_version=CENSUS_VERSION,
            progress_bar=True,
        )


def read_census_rows(obs, value_filter: str) -> pd.DataFrame:
    return (
        obs.read(value_filter=value_filter, column_names=list(CENSUS_COLUMNS))
        .concat()
        .to_pandas()
        .sort_values("soma_joinid", kind="stable")
        .reset_index(drop=True)
    )


def map_to_source_labels(
    selected: pd.DataFrame,
    dataset_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    try:
        obs = census["census_data"]["homo_sapiens"].obs
        registry = census["census_info"]["datasets"].read().concat().to_pandas()
        query = read_census_rows(
            obs,
            "dataset_id in " + str(dataset_ids) + " and cell_type == 'neuron'",
        )
        query_ids = selected["census_query_row"].to_numpy(dtype=np.int64)
        if query_ids.min() < 0 or query_ids.max() >= len(query):
            raise ValueError("saved query-local row number is outside the reconstructed query")
        chosen = query.iloc[query_ids].reset_index(drop=True)
        for saved_column, query_column in (
            ("saved_dataset_id", "dataset_id"),
            ("saved_donor_id", "donor_id"),
            ("saved_raw_sum", "raw_sum"),
        ):
            left = selected[saved_column].to_numpy()
            right = chosen[query_column].to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"reconstructed query disagrees at {saved_column}")
        chosen = pd.concat([selected.reset_index(drop=True), chosen], axis=1)

        parts: list[pd.DataFrame] = []
        source_audit: list[dict[str, object]] = []
        for dataset_id in dataset_ids:
            path = SOURCE_CACHE / f"{dataset_id}.h5ad"
            source = ad.read_h5ad(path, backed="r")
            try:
                dataset_rows = read_census_rows(obs, f"dataset_id == '{dataset_id}'")
                if len(dataset_rows) != source.n_obs:
                    raise ValueError(f"source row count mismatch for {dataset_id}")
                if not np.array_equal(
                    dataset_rows["donor_id"].astype(str).to_numpy(),
                    source.obs["donor_id"].astype(str).to_numpy(),
                ):
                    raise ValueError(f"source donor order mismatch for {dataset_id}")
                if not np.array_equal(
                    dataset_rows["cell_type"].astype(str).to_numpy(),
                    source.obs["cell_type"].astype(str).to_numpy(),
                ):
                    raise ValueError(f"source cell-type order mismatch for {dataset_id}")
                delta = (
                    source.obs["total_UMIs"].to_numpy(dtype=float)
                    - dataset_rows["raw_sum"].to_numpy(dtype=float)
                )
                if np.any(delta < 0) or float(delta.max()) > 100:
                    raise ValueError(f"source count-order audit failed for {dataset_id}")

                global_to_row = pd.Series(
                    np.arange(len(dataset_rows), dtype=np.int64),
                    index=dataset_rows["soma_joinid"].to_numpy(dtype=np.int64),
                )
                block = chosen[chosen["dataset_id"].eq(dataset_id)].copy()
                source_rows = global_to_row.loc[
                    block["soma_joinid"].to_numpy(dtype=np.int64)
                ].to_numpy(dtype=np.int64)
                block["source_row_index"] = source_rows
                block["whb_cell_label"] = source.obs_names[source_rows].astype(str)
                block["source_donor_id"] = (
                    source.obs["donor_id"].astype(str).to_numpy()[source_rows]
                )
                block["source_dissection"] = (
                    source.obs["dissection"].astype(str).to_numpy()[source_rows]
                )
                block["source_cluster_id"] = (
                    source.obs["cluster_id"].astype(str).to_numpy()[source_rows]
                )
                block["source_subcluster_id"] = (
                    source.obs["subcluster_id"].astype(str).to_numpy()[source_rows]
                )
                block["source_supercluster"] = (
                    source.obs["supercluster_term"].astype(str).to_numpy()[source_rows]
                )
                block["source_total_umis"] = (
                    source.obs["total_UMIs"].to_numpy(dtype=float)[source_rows]
                )
                parts.append(block)
                source_audit.append(
                    {
                        "dataset_id": dataset_id,
                        "source_rows": int(source.n_obs),
                        "selected_rows": int(len(block)),
                        "source_file": str(path),
                        "source_size_bytes": path.stat().st_size,
                        "source_sha256": file_sha256(path),
                        "max_source_minus_census_umi": float(delta.max()),
                    }
                )
            finally:
                source.file.close()

        mapped = (
            pd.concat(parts, ignore_index=True)
            .sort_values("census_query_row", kind="stable")
            .reset_index(drop=True)
        )
        registry = registry[registry["dataset_id"].isin(dataset_ids)].copy()
        mapped = mapped.merge(
            registry[["dataset_id", "dataset_version_id", "dataset_title"]],
            on="dataset_id",
            how="left",
            validate="many_to_one",
        )
        return mapped, pd.DataFrame(source_audit)
    finally:
        census.close()


def attach_whb_metadata(mapped: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    labels = set(mapped["whb_cell_label"])
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        WHB_METADATA,
        usecols=list(WHB_COLUMNS),
        dtype=str,
        chunksize=chunk_size,
    ):
        keep = chunk["cell_label"].isin(labels)
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    if not frames:
        raise ValueError("none of the mapped labels occur in WHB metadata")
    metadata = pd.concat(frames, ignore_index=True)
    if len(metadata) != 823 or metadata["cell_label"].nunique() != 823:
        raise ValueError("the mapped labels are not a complete unique 823-cell WHB set")
    result = mapped.merge(
        metadata,
        left_on="whb_cell_label",
        right_on="cell_label",
        how="left",
        validate="one_to_one",
    ).drop(columns="cell_label")
    if not result["feature_matrix_label"].eq("WHB-10Xv3-Neurons").all():
        raise ValueError("a mapped Siletti DA cell is absent from the WHB neuron matrix")
    if not np.array_equal(
        result["source_donor_id"].to_numpy(), result["donor_label"].to_numpy()
    ):
        raise ValueError("source and WHB donor labels disagree")
    return result


def attach_reference_membership(mapped: pd.DataFrame) -> pd.DataFrame:
    broad = ad.read_h5ad(BROAD_CANDIDATES, backed="r")
    author = ad.read_h5ad(AUTHOR_DA, backed="r")
    try:
        broad_meta = pd.DataFrame(
            {
                "whb_cell_label": broad.obs_names.astype(str),
                "broad_cluster_id": broad.obs["cluster_id"].astype(str).to_numpy(),
                "broad_subcluster_id": broad.obs["subcluster_id"].astype(str).to_numpy(),
            }
        )
        author_labels = set(author.obs_names.astype(str))
        broad_labels = set(broad_meta["whb_cell_label"])
    finally:
        broad.file.close()
        author.file.close()
    result = mapped.merge(
        broad_meta,
        on="whb_cell_label",
        how="left",
        validate="one_to_one",
    )
    result["in_broad_1853"] = result["whb_cell_label"].isin(broad_labels)
    result["in_author_da_998"] = result["whb_cell_label"].isin(author_labels)
    result["source_cluster_395"] = result["source_cluster_id"].eq("395")
    return result


def build_exclusion_registry(
    mapped: pd.DataFrame,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    broad = ad.read_h5ad(BROAD_CANDIDATES, backed="r")
    author = ad.read_h5ad(AUTHOR_DA, backed="r")
    try:
        selected_labels = set(mapped["whb_cell_label"])
        broad_labels = set(broad.obs_names.astype(str))
        author_labels = set(author.obs_names.astype(str))
    finally:
        broad.file.close()
        author.file.close()

    taxonomy = pd.read_csv(WHB_TAXONOMY, dtype=str)
    taxonomy_da = taxonomy[
        taxonomy["cluster_annotation_term_set_name"].eq("neurotransmitter")
        & taxonomy["cluster_annotation_term_name"].eq("DA VGLUT2")
    ]
    taxonomy_aliases = set(taxonomy_da["cluster_alias"])
    exact_reference_labels = selected_labels | broad_labels | author_labels

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        WHB_METADATA,
        usecols=list(WHB_COLUMNS),
        dtype=str,
        chunksize=chunk_size,
    ):
        keep = chunk["cell_label"].isin(exact_reference_labels) | chunk[
            "cluster_alias"
        ].isin(taxonomy_aliases)
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    if not frames:
        raise ValueError("no Siletti DA exclusion labels occur in WHB metadata")

    registry = pd.concat(frames, ignore_index=True).rename(
        columns={"cell_label": "whb_cell_label"}
    )
    if registry["whb_cell_label"].duplicated().any():
        raise ValueError("WHB metadata contains duplicate DA exclusion labels")
    missing = exact_reference_labels - set(registry["whb_cell_label"])
    if missing:
        raise ValueError(f"{len(missing)} exact DA reference labels are absent from WHB")
    if not registry["feature_matrix_label"].eq("WHB-10Xv3-Neurons").all():
        raise ValueError("a Siletti DA exclusion cell is absent from the WHB neuron matrix")

    registry["frozen_siletti_823"] = registry["whb_cell_label"].isin(
        selected_labels
    )
    registry["broad_candidate_1853"] = registry["whb_cell_label"].isin(
        broad_labels
    )
    registry["author_da_998"] = registry["whb_cell_label"].isin(author_labels)
    registry["allen_taxonomy_da_vglut2"] = registry["cluster_alias"].isin(
        taxonomy_aliases
    )
    registry["eligible_focal_reference"] = registry["frozen_siletti_823"]
    registry["exclude_from_non_da"] = True
    registry["exclusion_basis"] = registry.apply(
        lambda row: ";".join(
            name
            for name, present in (
                ("frozen_siletti_823", row["frozen_siletti_823"]),
                ("broad_candidate_1853", row["broad_candidate_1853"]),
                ("author_da_998", row["author_da_998"]),
                ("allen_taxonomy_da_vglut2", row["allen_taxonomy_da_vglut2"]),
            )
            if present
        ),
        axis=1,
    )
    registry = registry.sort_values("whb_cell_label", kind="stable").reset_index(
        drop=True
    )
    counts = {
        "exact_reference_exclusion_union_before_taxonomy": len(
            exact_reference_labels
        ),
        "allen_taxonomy_da_vglut2_cells": int(
            registry["allen_taxonomy_da_vglut2"].sum()
        ),
        "allen_taxonomy_da_vglut2_outside_reference_union": int(
            (
                registry["allen_taxonomy_da_vglut2"]
                & ~registry["whb_cell_label"].isin(exact_reference_labels)
            ).sum()
        ),
        "final_whb_da_exclusion_cells": len(registry),
    }
    return registry, counts


def main() -> None:
    args = parse_args()
    require_files(
        [SELECTED_DA, WHB_METADATA, BROAD_CANDIDATES, AUTHOR_DA, WHB_TAXONOMY]
    )
    if not args.overwrite and (
        OUTPUT.exists() or EXCLUSION_OUTPUT.exists() or MANIFEST.exists()
    ):
        raise FileExistsError("crosswalk outputs exist; pass --overwrite")

    selected, dataset_ids = load_selected()
    ensure_source_h5ads(dataset_ids, args.download_missing)
    mapped, source_audit = map_to_source_labels(selected, dataset_ids)
    mapped = attach_whb_metadata(mapped, args.metadata_chunk_size)
    mapped = attach_reference_membership(mapped)
    exclusion_registry, exclusion_counts = build_exclusion_registry(
        mapped, args.metadata_chunk_size
    )

    if len(mapped) != 823:
        raise ValueError("final crosswalk does not contain 823 rows")
    for column in ("census_query_row", "soma_joinid", "whb_cell_label"):
        if mapped[column].nunique() != 823:
            raise ValueError(f"final crosswalk is not one-to-one at {column}")

    keep = [
        "census_query_row",
        "soma_joinid",
        "whb_cell_label",
        "dataset_id",
        "dataset_version_id",
        "dataset_title",
        "source_row_index",
        "saved_donor_id",
        "source_donor_id",
        "donor_label",
        "saved_dissection",
        "source_dissection",
        "region_of_interest_label",
        "anatomical_division_label",
        "source_supercluster",
        "source_cluster_id",
        "source_subcluster_id",
        "cluster_alias",
        "saved_raw_sum",
        "raw_sum",
        "source_total_umis",
        "saved_da_score",
        "in_broad_1853",
        "in_author_da_998",
        "source_cluster_395",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    mapped[keep].to_csv(partial, index=False)
    os.replace(partial, OUTPUT)
    partial_exclusion = EXCLUSION_OUTPUT.with_suffix(
        EXCLUSION_OUTPUT.suffix + ".partial"
    )
    exclusion_registry.to_csv(partial_exclusion, index=False)
    os.replace(partial_exclusion, EXCLUSION_OUTPUT)
    manifest = {
        "census_version": CENSUS_VERSION,
        "selected_da_input": str(SELECTED_DA.resolve()),
        "whb_metadata": str(WHB_METADATA.resolve()),
        "counts": {
            "selected_da": 823,
            "mapped_global_census_ids": int(mapped["soma_joinid"].nunique()),
            "mapped_whb_cell_labels": int(mapped["whb_cell_label"].nunique()),
            "overlap_broad_1853": int(mapped["in_broad_1853"].sum()),
            "overlap_author_da_998": int(mapped["in_author_da_998"].sum()),
            "source_cluster_395": int(mapped["source_cluster_395"].sum()),
            **exclusion_counts,
        },
        "source_datasets": source_audit.to_dict(orient="records"),
        "output": {
            "crosswalk": {
                "path": str(OUTPUT.resolve()),
                "rows": len(mapped),
                "sha256": file_sha256(OUTPUT),
            },
            "exclusion_registry": {
                "path": str(EXCLUSION_OUTPUT.resolve()),
                "rows": len(exclusion_registry),
                "sha256": file_sha256(EXCLUSION_OUTPUT),
            },
        },
    }
    partial_manifest = MANIFEST.with_suffix(MANIFEST.suffix + ".partial")
    partial_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(partial_manifest, MANIFEST)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"wrote {OUTPUT}")
    print(f"wrote {EXCLUSION_OUTPUT}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
