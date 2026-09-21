import numpy as np

from fda_atlas_v2.cross_cohort_brainwide import (
    cross_cohort_brainwide_selectivity,
)


def test_cross_cohort_reference_is_explicitly_not_donor_matched():
    focal = np.array(
        [
            [5.0, 1.0],
            [5.2, 1.1],
            [4.8, 0.9],
            [5.1, 1.2],
        ]
    )
    result = cross_cohort_brainwide_selectivity(
        focal,
        np.full_like(focal, 0.5),
        genes=["A", "B"],
        focal_donor_ids=["allen-1", "allen-2", "allen-3", "allen-4"],
        focal_n_cells=100,
        reference_expression=np.array([[2.0, 2.0], [3.0, 0.5]]),
        reference_detection=np.array([[0.2, 0.3], [0.3, 0.1]]),
        reference_population_ids=["chiou-region-1", "chiou-region-2"],
        reference_n_cells=[1000, 2000],
        reference_donor_counts=[5, 3],
    ).set_index("gene_id")

    assert result["reference_is_donor_matched"].eq(False).all()
    assert set(result["common_donor_ids"]) == {
        "allen-1 | allen-2 | allen-3 | allen-4"
    }
    assert result.loc["A", "competitor_id"] == "chiou-region-2"
    assert result.loc["B", "competitor_id"] == "chiou-region-1"
    assert result.loc["A", "effect"] > 0
    assert result.loc["B", "effect"] < 0


def test_detection_difference_is_reported_but_not_a_gate():
    result = cross_cohort_brainwide_selectivity(
        np.array([[3.0], [3.1], [2.9], [3.2]]),
        np.array([[0.2], [0.2], [0.2], [0.2]]),
        genes=["A"],
        focal_donor_ids=["d1", "d2", "d3", "d4"],
        focal_n_cells=40,
        reference_expression=np.array([[1.0]]),
        reference_detection=np.array([[0.8]]),
        reference_population_ids=["region"],
        reference_n_cells=[100],
        reference_donor_counts=[5],
        minimum_focal_detection=0.05,
    )

    assert result.loc[0, "detection_difference"] < 0
    assert result.loc[0, "status"] == "estimated"
    assert np.isfinite(result.loc[0, "effect"])
