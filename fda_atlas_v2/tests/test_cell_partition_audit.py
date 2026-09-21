from collections import Counter

import pandas as pd
import pytest

from fda_atlas_v2.cell_partition_audit import (
    accumulate_partition_counts,
    finalize_partition_counts,
    summarize_membership_parquets,
    summarize_membership_parquets_with_identity,
    validate_cell_partition_audit_semantics,
)
from fda_atlas_v2.contracts import validate_table


def membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "release_id": ["r1"] * 6,
            "dataset_id": ["wholebrain"] * 6,
            "cell_id": [f"cell-{index}" for index in range(6)],
            "species": ["human"] * 6,
            "anatomy": ["midbrain"] * 4 + ["cortex"] * 2,
            "donor_id": ["d1"] * 4 + ["d2"] * 2,
            "feature_matrix_label": ["neurons"] * 4 + ["nonneurons"] * 2,
            "da_status": ["da", "non_da", "ambiguous", "non_da", "non_da", "ambiguous"],
            "exclude_from_non_da": [True, False, True, False, False, True],
        }
    )


def test_partition_audit_exactly_covers_da_nonda_and_ambiguous(tmp_path):
    frame = membership()
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame.iloc[:3].to_parquet(first, index=False)
    frame.iloc[3:].to_parquet(second, index=False)
    audit = summarize_membership_parquets([first, second], batch_size=2)
    validate_table("cell_partition_audit", audit)
    assert audit["parent_n"].sum() == 6
    assert audit["da_n"].sum() == 1
    assert audit["nonda_n"].sum() == 3
    assert audit["ambiguous_n"].sum() == 2
    assert audit["intersection_n"].sum() == 0
    assert audit["union_n"].sum() == 6
    assert audit["partition_complete"].all()


def test_partition_audit_rejects_membership_leakage():
    frame = membership()
    frame.loc[0, "exclude_from_non_da"] = False
    with pytest.raises(ValueError, match="DA cell"):
        accumulate_partition_counts(frame)

    frame = membership()
    frame.loc[1, "exclude_from_non_da"] = True
    with pytest.raises(ValueError, match="non-DA cell"):
        accumulate_partition_counts(frame)


def test_partition_audit_semantics_reject_incomplete_union():
    counts = accumulate_partition_counts(membership(), Counter())
    audit = finalize_partition_counts(counts)
    audit.loc[0, "ambiguous_n"] = 0
    with pytest.raises(ValueError, match="union does not equal"):
        validate_cell_partition_audit_semantics(audit)


def test_partition_audit_rejects_duplicate_ids_across_files(tmp_path):
    frame = membership()
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame.iloc[:3].to_parquet(first, index=False)
    changed = frame.iloc[3:].copy()
    changed.loc[changed.index[0], "cell_id"] = "cell-0"
    changed.to_parquet(second, index=False)
    with pytest.raises(ValueError, match="repeats 1 IDs"):
        summarize_membership_parquets_with_identity([first, second], batch_size=2)
