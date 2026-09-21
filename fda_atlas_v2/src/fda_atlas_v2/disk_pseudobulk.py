"""Resumable disk-backed pseudobulks for nearly dense whole-brain summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .hmoe_preprocessing import validate_raw_counts
from .pseudobulk import (
    DiskPseudobulkResult,
    PartitionedDiskPseudobulkResult,
    SparsePseudobulkResult,
    _sha256,
)


UINT32_MAX = np.iinfo(np.uint32).max


def _prepare_sparse_row_updates(destination: np.memmap, group_rows: np.ndarray, matrix):
    matrix = matrix.tocsr()
    updates = []
    for local_row, group_row in enumerate(group_rows):
        left = int(matrix.indptr[local_row])
        right = int(matrix.indptr[local_row + 1])
        indices = matrix.indices[left:right]
        if len(indices) == 0:
            continue
        values = matrix.data[left:right].astype(np.uint64, copy=False)
        updated = destination[group_row, indices].astype(np.uint64) + values
        if (updated > UINT32_MAX).any():
            raise OverflowError(f"disk pseudobulk uint32 overflow in group {group_row}")
        updates.append((int(group_row), indices.copy(), updated.astype(np.uint32)))
    return updates


def _apply_sparse_row_updates(destination: np.memmap, updates) -> None:
    for group_row, indices, values in updates:
        destination[group_row, indices] = values


def accumulate_sparse_raw_chunk(
    destination_counts: np.memmap,
    destination_detected_cells: np.memmap,
    raw_counts,
    group_rows: Sequence[int],
    accumulated_n_cells: np.ndarray,
    accumulated_library_sizes: np.ndarray,
    *,
    feature_collapse: sp.csr_matrix | None = None,
) -> int:
    """Add one raw cell chunk to disk arrays without densifying its genes."""

    if destination_counts.shape != destination_detected_cells.shape:
        raise ValueError("disk count and detection destinations do not align")
    if destination_counts.dtype != np.uint32 or destination_detected_cells.dtype != np.uint32:
        raise ValueError("disk pseudobulk destinations must be uint32")
    if len(accumulated_n_cells) != destination_counts.shape[0] or len(
        accumulated_library_sizes
    ) != destination_counts.shape[0]:
        raise ValueError("disk pseudobulk support accumulators do not align")
    if accumulated_n_cells.dtype != np.uint64 or accumulated_library_sizes.dtype != np.uint64:
        raise ValueError("disk pseudobulk support accumulators must be uint64")
    validate_raw_counts(raw_counts)
    if feature_collapse is None:
        if raw_counts.shape[1] != destination_counts.shape[1]:
            raise ValueError("raw cell chunk features do not align with destination")
    elif feature_collapse.shape != (raw_counts.shape[1], destination_counts.shape[1]):
        raise ValueError("feature-collapse matrix does not align with raw and destination genes")
    codes = np.asarray(group_rows, dtype=np.int64)
    if codes.ndim != 1 or len(codes) != raw_counts.shape[0]:
        raise ValueError("group rows do not align with raw cell chunk")
    eligible = codes >= 0
    if not eligible.any():
        return 0
    active_codes = codes[eligible]
    if active_codes.max() >= destination_counts.shape[0]:
        raise ValueError("group row is outside disk pseudobulk destination")
    raw = raw_counts if sp.issparse(raw_counts) else sp.csr_matrix(np.asarray(raw_counts))
    raw = raw.tocsr()[eligible].astype(np.int64)
    raw.eliminate_zeros()
    if feature_collapse is not None:
        raw = (raw @ feature_collapse).tocsr()
        raw.eliminate_zeros()
    unique_groups, inverse = np.unique(active_codes, return_inverse=True)
    assignment = sp.csr_matrix(
        (
            np.ones(len(inverse), dtype=np.int64),
            (inverse, np.arange(len(inverse), dtype=np.int64)),
        ),
        shape=(len(unique_groups), len(inverse)),
    )
    summed = assignment @ raw
    binary = raw.copy()
    binary.data = np.ones_like(binary.data, dtype=np.int64)
    detected = assignment @ binary
    count_updates = _prepare_sparse_row_updates(
        destination_counts, unique_groups, summed
    )
    detection_updates = _prepare_sparse_row_updates(
        destination_detected_cells, unique_groups, detected
    )
    _apply_sparse_row_updates(destination_counts, count_updates)
    _apply_sparse_row_updates(destination_detected_cells, detection_updates)

    cell_increment = np.bincount(
        active_codes, minlength=destination_counts.shape[0]
    ).astype(np.uint64)
    cell_libraries = np.asarray(raw.sum(axis=1), dtype=np.uint64).reshape(-1)
    library_increment = np.zeros(destination_counts.shape[0], dtype=np.uint64)
    np.add.at(library_increment, active_codes, cell_libraries)
    accumulated_n_cells += cell_increment
    accumulated_library_sizes += library_increment
    return int(eligible.sum())


def _paths(output_dir: Path, checkpoint_id: str) -> dict[str, Path]:
    return {
        "counts_partial": output_dir / f"{checkpoint_id}.counts.partial.npy",
        "detected_partial": output_dir / f"{checkpoint_id}.detected_cells.partial.npy",
        "counts": output_dir / f"{checkpoint_id}.counts.npy",
        "detected_cells": output_dir / f"{checkpoint_id}.detected_cells.npy",
        "metadata": output_dir / f"{checkpoint_id}.metadata.parquet",
        "features": output_dir / f"{checkpoint_id}.features.parquet",
        "state": output_dir / f"{checkpoint_id}.state.json",
        "manifest": output_dir / f"{checkpoint_id}.manifest.json",
    }


def create_disk_pseudobulk_workspace(
    output_dir: Path,
    *,
    checkpoint_id: str,
    shape: tuple[int, int],
    partition_ids: Sequence[str],
    overwrite: bool = False,
) -> Path:
    """Create zeroed uint32 arrays and an atomic partition-progress state."""

    output_dir = Path(output_dir).expanduser().resolve()
    if not checkpoint_id.strip():
        raise ValueError("checkpoint ID must be nonempty")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("disk pseudobulk shape must be two-dimensional and positive")
    partitions = [str(value) for value in partition_ids]
    if not partitions or len(partitions) != len(set(partitions)):
        raise ValueError("partition IDs must be nonempty and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir, checkpoint_id)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"disk pseudobulk outputs already exist: {existing}")
    if overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)

    counts = np.lib.format.open_memmap(
        paths["counts_partial"], mode="w+", dtype=np.uint32, shape=shape
    )
    detected = np.lib.format.open_memmap(
        paths["detected_partial"], mode="w+", dtype=np.uint32, shape=shape
    )
    counts.flush()
    detected.flush()
    state = {
        "checkpoint_id": checkpoint_id,
        "status": "accumulating",
        "shape": [int(shape[0]), int(shape[1])],
        "dtype": "uint32",
        "partition_ids": partitions,
        "completed_partition_ids": [],
    }
    partial = paths["state"].with_suffix(".json.partial")
    partial.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    partial.replace(paths["state"])
    return paths["state"]


def open_disk_pseudobulk_workspace(
    state_path: Path,
) -> tuple[np.memmap, np.memmap, dict]:
    """Reopen an incomplete workspace without resetting accumulated values."""

    state_path = Path(state_path).expanduser().resolve()
    state = json.loads(state_path.read_text())
    if state.get("status") != "accumulating":
        raise ValueError("disk pseudobulk workspace is not accumulating")
    paths = _paths(state_path.parent, str(state["checkpoint_id"]))
    if not paths["manifest"].exists():
        for partial_key, final_key in (
            ("counts_partial", "counts"),
            ("detected_partial", "detected_cells"),
        ):
            if not paths[partial_key].exists() and paths[final_key].exists():
                paths[final_key].replace(paths[partial_key])
    shape = tuple(int(value) for value in state["shape"])
    counts = np.lib.format.open_memmap(paths["counts_partial"], mode="r+")
    detected = np.lib.format.open_memmap(paths["detected_partial"], mode="r+")
    if counts.shape != shape or detected.shape != shape:
        raise ValueError("disk pseudobulk workspace arrays do not match state shape")
    if counts.dtype != np.uint32 or detected.dtype != np.uint32:
        raise ValueError("disk pseudobulk workspace arrays are not uint32")
    return counts, detected, state


def mark_disk_partition_complete(state_path: Path, partition_id: str) -> None:
    """Atomically record a fully flushed partition so restart can skip it."""

    state_path = Path(state_path).expanduser().resolve()
    state = json.loads(state_path.read_text())
    partition_id = str(partition_id)
    if partition_id not in state.get("partition_ids", []):
        raise ValueError(f"unknown disk pseudobulk partition: {partition_id}")
    completed = list(state.get("completed_partition_ids", []))
    if partition_id in completed:
        raise ValueError(f"disk pseudobulk partition is already complete: {partition_id}")
    completed.append(partition_id)
    state["completed_partition_ids"] = completed
    partial = state_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    partial.replace(state_path)


def finalize_disk_pseudobulk_checkpoint(
    state_path: Path,
    metadata: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    source_feature_indices: Sequence[int] | None = None,
    row_chunk_size: int = 128,
) -> Path:
    """Validate full raw sums, finalize arrays, and write a hashed manifest."""

    if row_chunk_size <= 0:
        raise ValueError("row chunk size must be positive")
    state_path = Path(state_path).expanduser().resolve()
    counts, detected, state = open_disk_pseudobulk_workspace(state_path)
    expected_partitions = list(state["partition_ids"])
    completed = list(state.get("completed_partition_ids", []))
    if len(completed) != len(expected_partitions) or set(completed) != set(
        expected_partitions
    ):
        raise ValueError("disk pseudobulk partitions are incomplete")
    names = [str(value) for value in feature_names]
    if len(names) != counts.shape[1] or len(names) != len(set(names)):
        raise ValueError("disk pseudobulk feature names are misaligned or duplicated")
    required = {"n_cells", "library_size"}
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"disk pseudobulk metadata is missing: {missing}")
    if len(metadata) != counts.shape[0]:
        raise ValueError("disk pseudobulk metadata does not align with rows")
    cell_values = pd.to_numeric(metadata["n_cells"], errors="raise").to_numpy(dtype=float)
    library_values = pd.to_numeric(
        metadata["library_size"], errors="raise"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(cell_values).all()
        or not np.isfinite(library_values).all()
        or (cell_values <= 0).any()
        or (library_values <= 0).any()
        or not np.allclose(cell_values, np.rint(cell_values), rtol=0, atol=0)
        or not np.allclose(library_values, np.rint(library_values), rtol=0, atol=0)
    ):
        raise ValueError("disk pseudobulk support and library sizes must be positive")
    n_cells = cell_values.astype(np.uint64)
    libraries = library_values.astype(np.uint64)
    for start in range(0, counts.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, counts.shape[0])
        observed_library = counts[start:stop].sum(axis=1, dtype=np.uint64)
        if not np.array_equal(observed_library, libraries[start:stop]):
            raise ValueError("disk pseudobulk full-count sums disagree with library sizes")
        if (detected[start:stop] > n_cells[start:stop, None]).any():
            raise ValueError("detected-cell counts exceed pseudobulk cell support")
    counts.flush()
    detected.flush()
    del counts, detected

    paths = _paths(state_path.parent, str(state["checkpoint_id"]))
    indices = (
        np.arange(len(names), dtype=np.int64)
        if source_feature_indices is None
        else np.asarray(source_feature_indices, dtype=np.int64)
    )
    if (
        indices.ndim != 1
        or len(indices) != len(names)
        or (indices < 0).any()
        or len(indices) != len(set(indices.tolist()))
    ):
        raise ValueError("source feature indices do not align with feature names")
    features = pd.DataFrame(
        {
            "feature_position": np.arange(len(names), dtype=np.int64),
            "feature_name": names,
            "source_feature_index": indices,
        }
    )
    metadata_partial = paths["metadata"].with_suffix(".parquet.partial")
    features_partial = paths["features"].with_suffix(".parquet.partial")
    metadata.to_parquet(metadata_partial, index=False)
    features.to_parquet(features_partial, index=False)
    staged = {
        "counts": paths["counts_partial"],
        "detected_cells": paths["detected_partial"],
        "metadata": metadata_partial,
        "features": features_partial,
    }
    manifest = {
        "checkpoint_id": state["checkpoint_id"],
        "matrix_representation": "disk_backed_raw_count_sums_by_donor_population",
        "depth_correction_status": "not_applied_until_expression_iteration",
        "shape": list(state["shape"]),
        "dtype": "uint32",
        "partition_ids": expected_partitions,
        "files": {
            key: {
                "name": paths[key].name,
                "size_bytes": staged[key].stat().st_size,
                "sha256": _sha256(staged[key]),
            }
            for key in ("counts", "detected_cells", "metadata", "features")
        },
    }
    manifest_partial = paths["manifest"].with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["counts_partial"].replace(paths["counts"])
    paths["detected_partial"].replace(paths["detected_cells"])
    metadata_partial.replace(paths["metadata"])
    features_partial.replace(paths["features"])
    manifest_partial.replace(paths["manifest"])
    state_path.unlink()
    return paths["manifest"]


def load_disk_pseudobulk_checkpoint(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> tuple[DiskPseudobulkResult, tuple[str, ...]]:
    """Open finalized arrays as read-only memmaps after integrity checks."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("matrix_representation") != (
        "disk_backed_raw_count_sums_by_donor_population"
    ):
        raise ValueError("checkpoint is not disk-backed raw pseudobulk counts")
    paths = {}
    for key in ("counts", "detected_cells", "metadata", "features"):
        record = manifest.get("files", {}).get(key, {})
        path = manifest_path.parent / str(record.get("name", ""))
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"disk pseudobulk file is missing or size-invalid: {key}")
        if verify_hashes and _sha256(path) != record.get("sha256"):
            raise ValueError(f"disk pseudobulk file is hash-invalid: {key}")
        paths[key] = path
    counts = np.load(paths["counts"], mmap_mode="r")
    detected = np.load(paths["detected_cells"], mmap_mode="r")
    metadata = pd.read_parquet(paths["metadata"])
    features = pd.read_parquet(paths["features"]).sort_values(
        "feature_position", kind="stable"
    )
    shape = tuple(int(value) for value in manifest["shape"])
    if counts.shape != shape or detected.shape != shape:
        raise ValueError("disk pseudobulk arrays do not match manifest shape")
    if counts.dtype != np.uint32 or detected.dtype != np.uint32:
        raise ValueError("disk pseudobulk finalized arrays are not uint32")
    if len(metadata) != shape[0] or len(features) != shape[1]:
        raise ValueError("disk pseudobulk metadata or features do not align")
    positions = features["feature_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(shape[1], dtype=np.int64)):
        raise ValueError("disk pseudobulk feature positions are not contiguous")
    names = tuple(features["feature_name"].astype(str))
    return (
        DiskPseudobulkResult(
            counts=counts,
            detected_cells=detected,
            metadata=metadata,
            feature_indices=features["source_feature_index"].to_numpy(dtype=np.int64),
        ),
        names,
    )


def combine_disk_pseudobulk_partitions(
    partitions: Sequence[DiskPseudobulkResult | SparsePseudobulkResult],
    partition_feature_names: Sequence[Sequence[str]],
    *,
    group_columns: Sequence[str] = ("donor_id", "population_id"),
) -> tuple[PartitionedDiskPseudobulkResult, tuple[str, ...]]:
    """Create a virtual sparse/disk merge for only the active gene block."""

    values = list(partitions)
    names_by_partition = [tuple(str(name) for name in names) for names in partition_feature_names]
    if not values or len(values) != len(names_by_partition):
        raise ValueError("each disk partition requires a feature-name vector")
    columns = [str(value) for value in group_columns]
    if not columns:
        raise ValueError("partition merge group columns must be nonempty")
    reference_names = names_by_partition[0]
    reference_indices = np.asarray(values[0].feature_indices, dtype=np.int64)
    metadata_parts = []
    for value, names in zip(values, names_by_partition):
        if names != reference_names or len(names) != value.counts.shape[1]:
            raise ValueError("disk partitions do not have identical feature identities")
        if not np.array_equal(
            np.asarray(value.feature_indices, dtype=np.int64), reference_indices
        ):
            raise ValueError("disk partitions do not have identical feature order")
        required = set(columns) | {"n_cells", "library_size"}
        if missing := sorted(required - set(value.metadata.columns)):
            raise ValueError(f"disk partition metadata is missing: {missing}")
        keys = value.metadata[columns].astype(str)
        if keys.duplicated().any():
            raise ValueError("a disk partition has duplicate donor-population rows")
        metadata_parts.append(
            value.metadata[columns + ["n_cells", "library_size"]].copy()
        )

    concatenated = pd.concat(metadata_parts, ignore_index=True)
    key_frame = concatenated[columns].astype(str)
    global_keys = key_frame.drop_duplicates().sort_values(
        columns, kind="stable"
    ).reset_index(drop=True)
    global_index = pd.MultiIndex.from_frame(global_keys)
    mappings = []
    offset = 0
    for metadata in metadata_parts:
        stop = offset + len(metadata)
        mapping = global_index.get_indexer(
            pd.MultiIndex.from_frame(key_frame.iloc[offset:stop])
        )
        if (mapping < 0).any():
            raise ValueError("failed to map a disk partition into global groups")
        mappings.append(mapping.astype(np.int64))
        offset = stop
    codes = global_index.get_indexer(pd.MultiIndex.from_frame(key_frame))
    merged = global_keys.copy()
    merged["n_cells"] = np.bincount(
        codes,
        weights=pd.to_numeric(concatenated["n_cells"], errors="raise"),
        minlength=len(global_keys),
    ).astype(np.int64)
    merged["library_size"] = np.bincount(
        codes,
        weights=pd.to_numeric(concatenated["library_size"], errors="raise"),
        minlength=len(global_keys),
    )
    return (
        PartitionedDiskPseudobulkResult(
            partitions=tuple(values),
            global_row_by_partition=tuple(mappings),
            metadata=merged,
            feature_indices=reference_indices,
        ),
        reference_names,
    )
