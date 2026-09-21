"""Memory-bounded genome-wide selectivity from pseudobulk checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .pseudobulk import (
    DiskPseudobulkResult,
    PartitionedDiskPseudobulkResult,
    SparsePseudobulkResult,
    iter_population_expression,
)
from .selectivity import (
    moderate_selectivity_results,
    strongest_competitor_selectivity,
)


def chunked_strongest_competitor_selectivity(
    pseudobulk: (
        SparsePseudobulkResult
        | DiskPseudobulkResult
        | PartitionedDiskPseudobulkResult
    ),
    feature_names: Sequence[str],
    *,
    focal_population: str,
    comparator_populations: Sequence[str],
    gene_chunk_size: int = 2_000,
    minimum_common_donors: int = 3,
    minimum_cells_per_donor_population: int = 10,
    minimum_focal_detection: float = 0.05,
    confidence_z: float = 1.96,
) -> pd.DataFrame:
    """Select competitors in chunks, then compute genome-wide BH FDR once."""

    chunks = []
    for expression in iter_population_expression(
        pseudobulk,
        feature_names,
        gene_chunk_size=gene_chunk_size,
    ):
        chunks.append(
            strongest_competitor_selectivity(
                expression,
                focal_population=focal_population,
                comparator_populations=comparator_populations,
                minimum_common_donors=minimum_common_donors,
                minimum_cells_per_donor_population=minimum_cells_per_donor_population,
                minimum_focal_detection=minimum_focal_detection,
                confidence_z=confidence_z,
                apply_moderation=False,
            )
        )
    if not chunks:
        raise ValueError("pseudobulk checkpoint contains no features")
    raw = pd.concat(chunks, ignore_index=True)
    return moderate_selectivity_results(raw, confidence_z=confidence_z)
