import numpy as np
import pandas as pd

from fda_atlas_v2.selectivity import (
    PopulationExpression,
    moderate_selectivity_results,
    strongest_competitor_selectivity,
)


def _data():
    metadata = pd.DataFrame(
        {
            "donor_id": ["d1", "d2", "d3"] * 3,
            "population_id": ["focal"] * 3 + ["comp_a"] * 3 + ["comp_b"] * 3,
            "n_cells": [20] * 9,
        }
    )
    expression = np.array(
        [
            [5.0, 2.0, 1.0], [5.2, 2.1, 1.1], [4.8, 1.9, 0.9],
            [1.0, 3.0, 0.5], [1.2, 3.1, 0.6], [0.8, 2.9, 0.4],
            [3.0, 1.0, 0.2], [3.2, 1.1, 0.3], [2.8, 0.9, 0.1],
        ]
    )
    detection = np.full_like(expression, 0.8)
    detection[:, 2] = 0.01
    return PopulationExpression(
        expression, detection, metadata, ("g1", "g2", "g3"),
        "log2_cpm_from_raw_donor_pseudobulk",
    )


def test_strongest_competitor_is_gene_specific_and_donor_paired():
    result = strongest_competitor_selectivity(
        _data(), focal_population="focal", comparator_populations=["comp_a", "comp_b"]
    ).set_index("gene_id")
    assert result.loc["g1", "competitor_id"] == "comp_b"
    assert result.loc["g2", "competitor_id"] == "comp_a"
    assert result.loc["g1", "raw_effect"] == 2.0
    assert result.loc["g2", "raw_effect"] == -1.0
    assert result.loc["g1", "n_common_donors"] == 3
    assert result.loc["g1", "focal_n_cells"] == 60
    assert result.loc["g1", "loo_raw_effect_min"] == 2.0
    assert result.loc["g1", "loo_raw_effect_max"] == 2.0


def test_low_detection_remains_visible_but_unestimated():
    result = strongest_competitor_selectivity(
        _data(), focal_population="focal", comparator_populations=["comp_a", "comp_b"]
    ).set_index("gene_id")
    assert result.loc["g3", "status"] == "insufficient_focal_detection"
    assert np.isnan(result.loc["g3", "effect"])
    assert result.loc["g3", "competitor_id"] in {"comp_a", "comp_b"}


def test_additive_depth_corrected_shift_does_not_change_selectivity():
    data = _data()
    baseline = strongest_competitor_selectivity(
        data, focal_population="focal", comparator_populations=["comp_a", "comp_b"]
    )
    shifted = PopulationExpression(
        data.expression + 4.0, data.detection, data.metadata, data.genes,
        data.matrix_representation,
    )
    repeated = strongest_competitor_selectivity(
        shifted, focal_population="focal", comparator_populations=["comp_a", "comp_b"]
    )
    np.testing.assert_allclose(baseline["raw_effect"], repeated["raw_effect"])
    np.testing.assert_allclose(baseline["effect"], repeated["effect"], equal_nan=True)


def test_insufficient_population_support_returns_all_genes_with_reasons():
    result = strongest_competitor_selectivity(
        _data(), focal_population="focal", comparator_populations=["missing"]
    )
    assert len(result) == 3
    assert set(result["status"]) == {"insufficient_comparator_support"}


def test_selectivity_requires_at_least_three_paired_donors():
    with np.testing.assert_raises_regex(ValueError, "at least three"):
        strongest_competitor_selectivity(
            _data(),
            focal_population="focal",
            comparator_populations=["comp_a"],
            minimum_common_donors=2,
        )


def test_integrated_or_scaled_expression_is_rejected():
    data = _data()
    invalid = PopulationExpression(
        data.expression, data.detection, data.metadata, data.genes, "scvi_integrated"
    )
    try:
        strongest_competitor_selectivity(
            invalid, focal_population="focal", comparator_populations=["comp_a"]
        )
    except ValueError as exc:
        assert "raw donor pseudobulk" in str(exc)
    else:
        raise AssertionError("integrated expression must be rejected")


def test_genomewide_remoderation_is_invariant_to_gene_chunk_boundaries():
    data = _data()
    full = strongest_competitor_selectivity(
        data,
        focal_population="focal",
        comparator_populations=["comp_a", "comp_b"],
    )
    chunks = []
    for indices in ([0], [1, 2]):
        chunk = PopulationExpression(
            expression=data.expression[:, indices],
            detection=data.detection[:, indices],
            metadata=data.metadata,
            genes=tuple(data.genes[index] for index in indices),
            matrix_representation=data.matrix_representation,
        )
        chunks.append(
            strongest_competitor_selectivity(
                chunk,
                focal_population="focal",
                comparator_populations=["comp_a", "comp_b"],
            )
        )
    remoderated = moderate_selectivity_results(pd.concat(chunks, ignore_index=True))
    columns = [
        "gene_id",
        "effect",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
    ]
    pd.testing.assert_frame_equal(
        full[columns].reset_index(drop=True),
        remoderated[columns].reset_index(drop=True),
    )


def test_low_detection_gene_does_not_influence_fdr_or_final_effects():
    base = pd.DataFrame(
        {
            "gene_id": ["g1", "g2"],
            "raw_effect": [1.0, 2.0],
            "raw_standard_error": [0.2, 0.3],
            "p_value": [0.01, 0.04],
            "ci_low": [0.5, 1.0],
            "ci_high": [1.5, 3.0],
            "status": ["estimated", "estimated"],
        }
    )
    augmented = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "gene_id": ["dropout"],
                    "raw_effect": [1_000.0],
                    "raw_standard_error": [0.01],
                    "p_value": [1e-100],
                    "ci_low": [999.0],
                    "ci_high": [1001.0],
                    "status": ["insufficient_focal_detection"],
                }
            ),
        ],
        ignore_index=True,
    )
    observed = moderate_selectivity_results(augmented)
    expected = moderate_selectivity_results(base)
    np.testing.assert_allclose(
        observed.loc[:1, "effect"], expected["effect"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        observed.loc[:1, "q_value"], expected["q_value"], rtol=0, atol=0
    )
    assert np.isnan(observed.loc[2, "effect"])
    assert np.isnan(observed.loc[2, "q_value"])


def test_low_cell_donor_is_dropped_without_discarding_supported_comparator():
    metadata = pd.DataFrame(
        {
            "donor_id": ["d1", "d2", "d3", "d4"] * 3,
            "population_id": ["focal"] * 4 + ["comp_a"] * 4 + ["comp_b"] * 4,
            "n_cells": [20] * 11 + [5],
        }
    )
    expression = np.asarray(
        [[5.0]] * 4 + [[1.0]] * 4 + [[4.0]] * 4
    )
    data = PopulationExpression(
        expression=expression,
        detection=np.full_like(expression, 0.8),
        metadata=metadata,
        genes=("g1",),
        matrix_representation="log2_cpm_from_raw_donor_pseudobulk",
    )
    result = strongest_competitor_selectivity(
        data,
        focal_population="focal",
        comparator_populations=["comp_a", "comp_b"],
        minimum_common_donors=3,
        minimum_cells_per_donor_population=10,
    ).iloc[0]

    assert result["competitor_id"] == "comp_b"
    assert result["raw_effect"] == 1.0
    assert result["n_common_donors"] == 3
    assert result["common_donor_ids"] == "d1 | d2 | d3"
    assert result["n_eligible_comparators"] == 2
    assert 0 <= result["competitor_loo_selection_fraction"] <= 1
