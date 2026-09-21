"""Streaming summaries that prove DA, non-DA, and ambiguous cell partitions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


GROUP_COLUMNS = (
    "release_id", "dataset_id", "species", "anatomy", "donor_id", "library_id"
)
MEMBERSHIP_COLUMNS = (
    "release_id", "dataset_id", "cell_id", "species", "anatomy", "donor_id",
    "feature_matrix_label", "da_status", "exclude_from_non_da",
)


def _validate_membership_chunk(frame: pd.DataFrame) -> None:
    missing = sorted(set(MEMBERSHIP_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"cell membership chunk is missing columns: {missing}")
    status = frame["da_status"].astype(str)
    if invalid := sorted(set(status) - {"da", "non_da", "ambiguous"}):
        raise ValueError(f"cell membership chunk has invalid DA statuses: {invalid}")
    excluded = frame["exclude_from_non_da"].astype(bool)
    if (status.eq("da") & ~excluded).any():
        raise ValueError("a DA cell is not excluded from the non-DA comparator")
    if (status.eq("ambiguous") & ~excluded).any():
        raise ValueError("an ambiguous cell is not excluded from the non-DA comparator")
    if (status.eq("non_da") & excluded).any():
        raise ValueError("a non-DA cell is unexpectedly excluded from its comparator")


def accumulate_partition_counts(frame: pd.DataFrame, counts: Counter | None = None) -> Counter:
    _validate_membership_chunk(frame)
    counts = Counter() if counts is None else counts
    work = frame.copy()
    work["library_id"] = work["feature_matrix_label"].fillna("").astype(str)
    for column in GROUP_COLUMNS[:-1]:
        work[column] = work[column].fillna("").astype(str)
    grouped = work.groupby(list(GROUP_COLUMNS), dropna=False, sort=False)
    for key, block in grouped:
        status = block["da_status"].astype(str)
        counts[(*key, "parent_n")] += int(len(block))
        counts[(*key, "da_n")] += int(status.eq("da").sum())
        counts[(*key, "nonda_n")] += int(status.eq("non_da").sum())
        counts[(*key, "ambiguous_n")] += int(status.eq("ambiguous").sum())
    return counts


def finalize_partition_counts(counts: Counter) -> pd.DataFrame:
    keys = sorted({key[:-1] for key in counts})
    rows = []
    for key in keys:
        parent_n = int(counts[(*key, "parent_n")])
        da_n = int(counts[(*key, "da_n")])
        nonda_n = int(counts[(*key, "nonda_n")])
        ambiguous_n = int(counts[(*key, "ambiguous_n")])
        intersection_n = 0
        union_n = da_n + nonda_n + ambiguous_n
        rows.append(
            dict(
                zip(GROUP_COLUMNS, key),
                parent_n=parent_n,
                da_n=da_n,
                nonda_n=nonda_n,
                ambiguous_n=ambiguous_n,
                intersection_n=intersection_n,
                union_n=union_n,
                partition_complete=(union_n == parent_n),
            )
        )
    output = pd.DataFrame(rows)
    validate_cell_partition_audit_semantics(output)
    return output


def summarize_membership_parquets(
    paths: list[Path], *, batch_size: int = 250_000
) -> pd.DataFrame:
    """Stream membership Parquets without loading a whole brain into memory."""

    audit, _ = summarize_membership_parquets_with_identity(
        paths, batch_size=batch_size
    )
    return audit


def summarize_membership_parquets_with_identity(
    paths: list[Path], *, batch_size: int = 250_000
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not paths:
        raise ValueError("at least one cell membership Parquet is required")
    counts = Counter()
    seen_by_dataset: dict[str, set[str]] = {}
    for path in paths:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=list(MEMBERSHIP_COLUMNS)
        ):
            frame = batch.to_pandas()
            for dataset_id, block in frame.groupby("dataset_id", sort=False):
                values = block["cell_id"].astype("string")
                invalid = values.isna() | values.str.strip().eq("")
                if invalid.any():
                    raise ValueError(
                        f"cell membership has {int(invalid.sum())} blank cell IDs"
                    )
                cell_ids = values.astype(str)
                if cell_ids.duplicated().any():
                    raise ValueError(
                        f"cell membership batch has duplicate IDs: {dataset_id}"
                    )
                seen = seen_by_dataset.setdefault(str(dataset_id), set())
                overlap = set(cell_ids) & seen
                if overlap:
                    raise ValueError(
                        f"cell membership repeats {len(overlap)} IDs: {dataset_id}"
                    )
                seen.update(cell_ids)
            accumulate_partition_counts(frame, counts)
    audit = finalize_partition_counts(counts)
    unique_counts = {
        dataset_id: len(values) for dataset_id, values in sorted(seen_by_dataset.items())
    }
    parent_counts = audit.groupby("dataset_id", sort=True)["parent_n"].sum().to_dict()
    if unique_counts != parent_counts:
        raise ValueError(
            f"unique cell counts {unique_counts} do not equal parents {parent_counts}"
        )
    return audit, unique_counts


def validate_cell_partition_audit_semantics(frame: pd.DataFrame) -> None:
    required = set(GROUP_COLUMNS) | {
        "parent_n", "da_n", "nonda_n", "ambiguous_n", "intersection_n",
        "union_n", "partition_complete",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"cell partition audit is missing columns: {missing}")
    if frame.empty:
        raise ValueError("cell partition audit cannot be empty")
    count_columns = [
        "parent_n", "da_n", "nonda_n", "ambiguous_n", "intersection_n", "union_n"
    ]
    counts = frame[count_columns].apply(pd.to_numeric, errors="coerce")
    if counts.isna().any(axis=None) or (counts < 0).any(axis=None):
        raise ValueError("cell partition audit counts must be finite and nonnegative")
    if not counts.eq(counts.round()).all(axis=None):
        raise ValueError("cell partition audit counts must be integers")
    expected_union = counts["da_n"] + counts["nonda_n"] + counts["ambiguous_n"]
    if not counts["union_n"].eq(expected_union).all():
        raise ValueError("cell partition audit union does not equal DA plus non-DA plus ambiguous")
    if not counts["parent_n"].eq(counts["union_n"]).all():
        raise ValueError("cell partition audit does not cover every parent cell")
    if counts["intersection_n"].ne(0).any():
        raise ValueError("cell partition audit has DA/non-DA intersections")
    if not frame["partition_complete"].astype(bool).all():
        raise ValueError("cell partition audit contains an incomplete partition")
