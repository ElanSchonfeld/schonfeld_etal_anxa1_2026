"""Sparse donor-pseudobulk utilities for depth-corrected transfer evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .hmoe_preprocessing import validate_raw_counts


@dataclass(frozen=True)
class PseudobulkResult:
    counts: np.ndarray
    metadata: pd.DataFrame
    feature_indices: np.ndarray


@dataclass(frozen=True)
class SparsePseudobulkResult:
    counts: sp.csr_matrix
    detected_cells: sp.csr_matrix
    metadata: pd.DataFrame
    feature_indices: np.ndarray


@dataclass(frozen=True)
class DiskPseudobulkResult:
    counts: np.memmap
    detected_cells: np.memmap
    metadata: pd.DataFrame
    feature_indices: np.ndarray


@dataclass(frozen=True)
class PartitionedDiskPseudobulkResult:
    partitions: tuple[DiskPseudobulkResult | SparsePseudobulkResult, ...]
    global_row_by_partition: tuple[np.ndarray, ...]
    metadata: pd.DataFrame
    feature_indices: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_sparse_pseudobulk_checkpoint(
    pseudobulk: SparsePseudobulkResult,
    feature_names: Sequence[str],
    output_dir: Path,
    *,
    checkpoint_id: str,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    names = [str(value) for value in feature_names]
    if not checkpoint_id.strip():
        raise ValueError("checkpoint ID must be nonempty")
    if pseudobulk.counts.shape != pseudobulk.detected_cells.shape:
        raise ValueError("pseudobulk count and detection matrices do not align")
    if len(pseudobulk.metadata) != pseudobulk.counts.shape[0]:
        raise ValueError("pseudobulk metadata does not align with matrix rows")
    if len(names) != pseudobulk.counts.shape[1]:
        raise ValueError("feature names do not align with matrix columns")
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    if len(pseudobulk.feature_indices) != len(names):
        raise ValueError("source feature indices do not align with feature names")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "counts": output_dir / f"{checkpoint_id}.counts.npz",
        "detected_cells": output_dir / f"{checkpoint_id}.detected_cells.npz",
        "metadata": output_dir / f"{checkpoint_id}.metadata.parquet",
        "features": output_dir / f"{checkpoint_id}.features.parquet",
        "manifest": output_dir / f"{checkpoint_id}.manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"checkpoint outputs already exist: {existing}")

    sparse_outputs = {
        "counts": pseudobulk.counts.tocsr(),
        "detected_cells": pseudobulk.detected_cells.tocsr(),
    }
    for key, matrix in sparse_outputs.items():
        partial = paths[key].with_name(paths[key].stem + ".partial.npz")
        sp.save_npz(partial, matrix, compressed=True)
        partial.replace(paths[key])

    metadata_partial = paths["metadata"].with_suffix(".parquet.partial")
    pseudobulk.metadata.to_parquet(metadata_partial, index=False)
    metadata_partial.replace(paths["metadata"])
    features = pd.DataFrame(
        {
            "feature_position": np.arange(len(names), dtype=np.int64),
            "feature_name": names,
            "source_feature_index": np.asarray(
                pseudobulk.feature_indices, dtype=np.int64
            ),
        }
    )
    features_partial = paths["features"].with_suffix(".parquet.partial")
    features.to_parquet(features_partial, index=False)
    features_partial.replace(paths["features"])

    manifest = {
        "checkpoint_id": checkpoint_id,
        "matrix_representation": "raw_count_sums_by_donor_population",
        "depth_correction_status": "not_applied_until_expression_iteration",
        "rows": int(pseudobulk.counts.shape[0]),
        "features": int(pseudobulk.counts.shape[1]),
        "counts_nnz": int(pseudobulk.counts.nnz),
        "detected_cells_nnz": int(pseudobulk.detected_cells.nnz),
        "files": {
            key: {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    manifest_partial = paths["manifest"].with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(paths["manifest"])
    return paths["manifest"]


def load_sparse_pseudobulk_checkpoint(
    manifest_path: Path,
) -> tuple[SparsePseudobulkResult, tuple[str, ...]]:
    """Load a checkpoint only after verifying every recorded file hash."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("matrix_representation") != "raw_count_sums_by_donor_population":
        raise ValueError("checkpoint is not raw donor-population pseudobulk counts")
    paths = {}
    for key in ("counts", "detected_cells", "metadata", "features"):
        record = manifest.get("files", {}).get(key, {})
        path = manifest_path.parent / str(record.get("name", ""))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"checkpoint file is missing or hash-invalid: {key}")
        paths[key] = path
    counts = sp.load_npz(paths["counts"]).tocsr()
    detected = sp.load_npz(paths["detected_cells"]).tocsr()
    metadata = pd.read_parquet(paths["metadata"])
    features = pd.read_parquet(paths["features"]).sort_values(
        "feature_position", kind="stable"
    )
    expected_positions = np.arange(len(features), dtype=np.int64)
    if not np.array_equal(features["feature_position"].to_numpy(), expected_positions):
        raise ValueError("checkpoint feature positions are not contiguous")
    names = tuple(features["feature_name"].astype(str))
    result = SparsePseudobulkResult(
        counts=counts,
        detected_cells=detected,
        metadata=metadata,
        feature_indices=features["source_feature_index"].to_numpy(dtype=np.int64),
    )
    if counts.shape != detected.shape or counts.shape != (
        int(manifest["rows"]),
        int(manifest["features"]),
    ):
        raise ValueError("checkpoint matrix shapes do not match the manifest")
    if len(metadata) != counts.shape[0] or len(names) != counts.shape[1]:
        raise ValueError("checkpoint metadata or features do not align")
    return result, names


def aggregate_pseudobulk(
    counts: Any,
    observations: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    feature_indices: Sequence[int],
    minimum_cells: int,
) -> PseudobulkResult:
    validate_raw_counts(counts)
    if len(observations) != counts.shape[0]:
        raise ValueError("observation rows do not match the count matrix")
    if minimum_cells <= 0:
        raise ValueError("minimum cells must be positive")
    group_columns = [str(value) for value in group_columns]
    if missing := sorted(set(group_columns) - set(observations.columns)):
        raise ValueError(f"pseudobulk group columns are missing: {missing}")
    indices = np.asarray(feature_indices, dtype=int)
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
        raise ValueError("feature indices must be a unique one-dimensional vector")
    if len(indices) and (indices.min() < 0 or indices.max() >= counts.shape[1]):
        raise ValueError("pseudobulk feature index is outside the matrix")

    grouping = observations[group_columns].astype(str).reset_index(drop=True)
    unique_groups = grouping.drop_duplicates().sort_values(
        group_columns, kind="stable"
    ).reset_index(drop=True)
    group_index = pd.MultiIndex.from_frame(unique_groups)
    cell_index = pd.MultiIndex.from_frame(grouping)
    codes = group_index.get_indexer(cell_index)
    if (codes < 0).any():
        raise ValueError("failed to assign a pseudobulk group")
    indicator = sp.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float64),
            (codes, np.arange(len(codes), dtype=int)),
        ),
        shape=(len(unique_groups), len(codes)),
    )
    selected = counts[:, indices]
    if not sp.issparse(selected):
        selected = sp.csr_matrix(np.asarray(selected))
    summed = indicator @ selected
    cell_library = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    library_sizes = np.asarray(indicator @ cell_library).reshape(-1)
    n_cells = np.asarray(indicator.sum(axis=1)).reshape(-1).astype(int)
    keep = n_cells >= minimum_cells
    metadata = unique_groups.loc[keep].reset_index(drop=True)
    metadata["n_cells"] = n_cells[keep]
    metadata["library_size"] = library_sizes[keep]
    return PseudobulkResult(
        counts=np.asarray(summed[keep].toarray(), dtype=np.float64),
        metadata=metadata,
        feature_indices=indices,
    )


def aggregate_sparse_pseudobulk(
    counts: Any,
    observations: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    feature_indices: Sequence[int] | None = None,
    feature_collapse: sp.spmatrix | None = None,
    eligible_cells: Sequence[bool] | None = None,
    minimum_cells: int,
    cell_chunk_size: int = 20_000,
) -> SparsePseudobulkResult:
    """Stream raw cells into sparse pseudobulk counts and detection counts."""

    if len(observations) != counts.shape[0]:
        raise ValueError("observation rows do not match the count matrix")
    if minimum_cells <= 0 or cell_chunk_size <= 0:
        raise ValueError("minimum cells and cell chunk size must be positive")
    group_columns = [str(value) for value in group_columns]
    if missing := sorted(set(group_columns) - set(observations.columns)):
        raise ValueError(f"pseudobulk group columns are missing: {missing}")
    if feature_indices is not None and feature_collapse is not None:
        raise ValueError("feature indices and feature collapse are mutually exclusive")
    collapse = None
    if feature_collapse is not None:
        collapse = sp.csr_matrix(feature_collapse)
        if collapse.shape[0] != counts.shape[1] or collapse.shape[1] <= 0:
            raise ValueError("feature collapse does not align with count columns")
        if (collapse.data < 0).any() or not np.array_equal(
            np.asarray(collapse.sum(axis=1)).reshape(-1),
            np.ones(counts.shape[1]),
        ):
            raise ValueError("feature collapse must map every source feature exactly once")
        indices = np.arange(collapse.shape[1], dtype=int)
    elif feature_indices is None:
        indices = np.arange(counts.shape[1], dtype=int)
    else:
        indices = np.asarray(feature_indices, dtype=int)
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
        raise ValueError("feature indices must be a unique one-dimensional vector")
    if len(indices) and (indices.min() < 0 or indices.max() >= counts.shape[1]):
        raise ValueError("pseudobulk feature index is outside the matrix")

    if eligible_cells is None:
        eligible_mask = np.ones(counts.shape[0], dtype=bool)
    else:
        eligible_mask = np.asarray(eligible_cells, dtype=bool)
        if eligible_mask.ndim != 1 or len(eligible_mask) != counts.shape[0]:
            raise ValueError("eligible-cell mask does not align with count rows")
    if not eligible_mask.any():
        raise ValueError("eligible-cell mask selects no cells")

    grouping = observations.loc[eligible_mask, group_columns].astype(str).reset_index(drop=True)
    unique_groups = grouping.drop_duplicates().sort_values(
        group_columns, kind="stable"
    ).reset_index(drop=True)
    group_index = pd.MultiIndex.from_frame(unique_groups)
    eligible_codes = group_index.get_indexer(pd.MultiIndex.from_frame(grouping))
    if (eligible_codes < 0).any():
        raise ValueError("failed to assign a pseudobulk group")
    group_cell_counts = np.bincount(eligible_codes, minlength=len(unique_groups))
    keep = group_cell_counts >= minimum_cells
    remap = np.full(len(unique_groups), -1, dtype=int)
    remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    kept_codes = np.full(counts.shape[0], -1, dtype=int)
    kept_codes[eligible_mask] = remap[eligible_codes]
    metadata = unique_groups.loc[keep].reset_index(drop=True)
    metadata["n_cells"] = group_cell_counts[keep]
    library_sizes = np.zeros(len(metadata), dtype=np.float64)
    summed = sp.csr_matrix((len(metadata), len(indices)), dtype=np.float64)
    detected = sp.csr_matrix((len(metadata), len(indices)), dtype=np.int64)

    for start in range(0, counts.shape[0], cell_chunk_size):
        stop = min(start + cell_chunk_size, counts.shape[0])
        chunk_codes = kept_codes[start:stop]
        eligible = chunk_codes >= 0
        if not eligible.any():
            continue
        raw = counts[start:stop]
        if not sp.issparse(raw):
            raw = sp.csr_matrix(np.asarray(raw))
        else:
            raw = raw.tocsr()
        validate_raw_counts(raw)
        raw = raw[eligible]
        active_codes = chunk_codes[eligible]
        indicator = sp.csr_matrix(
            (
                np.ones(len(active_codes), dtype=np.float64),
                (active_codes, np.arange(len(active_codes), dtype=int)),
            ),
            shape=(len(metadata), len(active_codes)),
        )
        full_library = np.asarray(raw.sum(axis=1), dtype=np.float64).reshape(-1)
        library_sizes += np.asarray(indicator @ full_library).reshape(-1)
        selected = raw @ collapse if collapse is not None else raw[:, indices]
        summed = summed + indicator @ selected
        binary = selected.copy()
        binary.eliminate_zeros()
        binary.data = np.ones_like(binary.data, dtype=np.int64)
        detected = detected + indicator.astype(np.int64) @ binary

    metadata["library_size"] = library_sizes
    if (library_sizes <= 0).any():
        raise ValueError("eligible pseudobulk has zero full-library counts")
    summed.eliminate_zeros()
    detected.eliminate_zeros()
    return SparsePseudobulkResult(
        counts=summed.tocsr(),
        detected_cells=detected.tocsr(),
        metadata=metadata,
        feature_indices=indices,
    )


def merge_sparse_pseudobulks(
    pseudobulks: Sequence[SparsePseudobulkResult],
    partition_feature_names: Sequence[Sequence[str]],
    *,
    group_columns: Sequence[str] = ("donor_id", "population_id"),
) -> SparsePseudobulkResult:
    values = list(pseudobulks)
    if not values:
        raise ValueError("at least one pseudobulk is required")
    names_by_partition = [tuple(str(name) for name in names) for names in partition_feature_names]
    if len(names_by_partition) != len(values):
        raise ValueError("each pseudobulk partition requires a feature-name vector")
    columns = [str(value) for value in group_columns]
    if not columns:
        raise ValueError("merge group columns must be nonempty")
    reference_indices = np.asarray(values[0].feature_indices, dtype=np.int64)
    reference_names = names_by_partition[0]
    n_features = values[0].counts.shape[1]
    for value, names in zip(values, names_by_partition):
        if value.counts.shape != value.detected_cells.shape:
            raise ValueError("pseudobulk count and detection matrices do not align")
        if value.counts.shape[1] != n_features or not np.array_equal(
            np.asarray(value.feature_indices, dtype=np.int64), reference_indices
        ):
            raise ValueError("pseudobulk partitions do not have identical feature order")
        if names != reference_names or len(names) != n_features:
            raise ValueError("pseudobulk partitions do not have identical feature identities")
        required = set(columns) | {"n_cells", "library_size"}
        if missing := sorted(required - set(value.metadata.columns)):
            raise ValueError(f"pseudobulk partition metadata is missing: {missing}")
        if len(value.metadata) != value.counts.shape[0]:
            raise ValueError("pseudobulk partition metadata does not align with rows")

    metadata = pd.concat(
        [value.metadata[columns + ["n_cells", "library_size"]] for value in values],
        ignore_index=True,
    )
    keys = metadata[columns].astype(str)
    unique = keys.drop_duplicates().sort_values(columns, kind="stable").reset_index(drop=True)
    group_index = pd.MultiIndex.from_frame(unique)
    codes = group_index.get_indexer(pd.MultiIndex.from_frame(keys))
    if (codes < 0).any():
        raise ValueError("failed to assign merged pseudobulk groups")
    indicator = sp.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float64),
            (codes, np.arange(len(codes), dtype=int)),
        ),
        shape=(len(unique), len(codes)),
    )
    counts = indicator @ sp.vstack([value.counts for value in values], format="csr")
    detected = indicator.astype(np.int64) @ sp.vstack(
        [value.detected_cells for value in values], format="csr"
    )
    merged = unique.copy()
    merged["n_cells"] = np.asarray(
        indicator @ metadata["n_cells"].to_numpy(dtype=np.int64)
    ).reshape(-1).astype(np.int64)
    merged["library_size"] = np.asarray(
        indicator @ metadata["library_size"].to_numpy(dtype=np.float64)
    ).reshape(-1)
    return SparsePseudobulkResult(
        counts=counts.tocsr(),
        detected_cells=detected.tocsr(),
        metadata=merged,
        feature_indices=reference_indices,
    )


def iter_population_expression(
    pseudobulk: (
        SparsePseudobulkResult
        | DiskPseudobulkResult
        | PartitionedDiskPseudobulkResult
    ),
    feature_names: Sequence[str],
    *,
    gene_chunk_size: int = 2_000,
    prior_count: float = 0.5,
):
    """Yield bounded dense gene chunks after explicit full-library depth correction."""

    from .selectivity import PopulationExpression

    names = [str(value) for value in feature_names]
    n_features = (
        len(pseudobulk.feature_indices)
        if isinstance(pseudobulk, PartitionedDiskPseudobulkResult)
        else pseudobulk.counts.shape[1]
    )
    if len(names) != n_features:
        raise ValueError("feature names do not align with pseudobulk columns")
    if gene_chunk_size <= 0:
        raise ValueError("gene chunk size must be positive")
    libraries = pseudobulk.metadata["library_size"].to_numpy(dtype=float)
    cells = pseudobulk.metadata["n_cells"].to_numpy(dtype=float)
    for start in range(0, len(names), gene_chunk_size):
        stop = min(start + gene_chunk_size, len(names))
        if isinstance(pseudobulk, PartitionedDiskPseudobulkResult):
            raw = np.zeros((len(pseudobulk.metadata), stop - start), dtype=np.float64)
            detected = np.zeros_like(raw)
            for partition, global_rows in zip(
                pseudobulk.partitions, pseudobulk.global_row_by_partition
            ):
                partition_raw = partition.counts[:, start:stop]
                partition_detected = partition.detected_cells[:, start:stop]
                raw[global_rows] += (
                    partition_raw.toarray()
                    if sp.issparse(partition_raw)
                    else np.asarray(partition_raw)
                )
                detected[global_rows] += (
                    partition_detected.toarray()
                    if sp.issparse(partition_detected)
                    else np.asarray(partition_detected)
                )
        else:
            raw_block = pseudobulk.counts[:, start:stop]
            raw = (
                raw_block.toarray()
                if sp.issparse(raw_block)
                else np.asarray(raw_block)
            )
            detected_block = pseudobulk.detected_cells[:, start:stop]
            detected = (
                detected_block.toarray()
                if sp.issparse(detected_block)
                else np.asarray(detected_block)
            )
        expression = log2_cpm(raw, libraries, prior_count=prior_count)
        detection = detected / cells[:, None]
        yield PopulationExpression(
            expression=expression,
            detection=detection,
            metadata=pseudobulk.metadata,
            genes=tuple(names[start:stop]),
            matrix_representation="log2_cpm_from_raw_donor_pseudobulk",
        )


def log2_cpm(
    pseudobulk_counts: np.ndarray,
    library_sizes: Sequence[float],
    *,
    prior_count: float = 0.5,
) -> np.ndarray:
    """Apply explicit library-depth correction to raw pseudobulk counts."""

    matrix = np.asarray(pseudobulk_counts, dtype=np.float64)
    libraries = np.asarray(library_sizes, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or len(libraries) != matrix.shape[0]:
        raise ValueError("pseudobulk counts and library sizes do not align")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("pseudobulk counts must be finite and nonnegative")
    if not np.isfinite(libraries).all() or (libraries <= 0).any():
        raise ValueError("pseudobulk library sizes must be finite and positive")
    if prior_count <= 0:
        raise ValueError("prior count must be positive")
    return np.log2((matrix + prior_count) / libraries[:, None] * 1e6)


def center_within_study(
    expression: np.ndarray,
    studies: Sequence[str],
) -> np.ndarray:
    """Remove study-wide gene means while retaining population differences."""

    matrix = np.asarray(expression, dtype=np.float64)
    labels = np.asarray(studies).astype(str)
    if matrix.ndim != 2 or len(labels) != matrix.shape[0]:
        raise ValueError("expression rows and study labels do not align")
    centered = np.empty_like(matrix)
    for study in sorted(set(labels)):
        mask = labels == study
        centered[mask] = matrix[mask] - matrix[mask].mean(axis=0, keepdims=True)
    return centered


def population_centroids(
    expression: np.ndarray,
    populations: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute equal-donor median centroids for each population."""

    matrix = np.asarray(expression, dtype=np.float64)
    labels = np.asarray(populations).astype(str)
    if matrix.ndim != 2 or len(labels) != matrix.shape[0]:
        raise ValueError("expression rows and population labels do not align")
    ordered = sorted(set(labels))
    centroids = np.vstack(
        [np.median(matrix[labels == population], axis=0) for population in ordered]
    )
    return centroids, ordered


def pearson_correlation_matrix(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Correlate rows after per-profile centering with zero-variance handling."""

    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference matrices must share feature columns")
    query_centered = query - query.mean(axis=1, keepdims=True)
    reference_centered = reference - reference.mean(axis=1, keepdims=True)
    numerator = query_centered @ reference_centered.T
    denominator = np.linalg.norm(query_centered, axis=1, keepdims=True) * np.linalg.norm(
        reference_centered, axis=1, keepdims=True
    ).T
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
