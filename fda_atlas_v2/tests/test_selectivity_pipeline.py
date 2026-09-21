import numpy as np
import pandas as pd
import scipy.sparse as sp

from fda_atlas_v2.pseudobulk import SparsePseudobulkResult
from fda_atlas_v2.selectivity_pipeline import (
    chunked_strongest_competitor_selectivity,
)


def test_chunked_selectivity_is_invariant_to_chunk_size():
    metadata = pd.DataFrame(
        {
            "donor_id": ["d1", "d2", "d3"] * 3,
            "population_id": ["focal"] * 3 + ["a"] * 3 + ["b"] * 3,
            "n_cells": [20] * 9,
            "library_size": [1000.0] * 9,
        }
    )
    counts = sp.csr_matrix(
        np.array(
            [
                [100, 20, 10], [110, 21, 11], [90, 19, 9],
                [10, 30, 5], [12, 31, 6], [8, 29, 4],
                [30, 10, 2], [32, 11, 3], [28, 9, 1],
            ],
            dtype=float,
        )
    )
    pseudobulk = SparsePseudobulkResult(
        counts=counts,
        detected_cells=sp.csr_matrix(np.full(counts.shape, 20, dtype=int)),
        metadata=metadata,
        feature_indices=np.arange(3),
    )
    one = chunked_strongest_competitor_selectivity(
        pseudobulk,
        ["g1", "g2", "g3"],
        focal_population="focal",
        comparator_populations=["a", "b"],
        gene_chunk_size=1,
    )
    three = chunked_strongest_competitor_selectivity(
        pseudobulk,
        ["g1", "g2", "g3"],
        focal_population="focal",
        comparator_populations=["a", "b"],
        gene_chunk_size=3,
    )
    pd.testing.assert_frame_equal(one, three)
