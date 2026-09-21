import numpy as np
import pytest
import scipy.sparse as sp

from fda_atlas_v2.hmoe_preprocessing import (
    audit_feature_order,
    fit_fixed_pearson_state,
    fixed_state_frame,
    fixed_state_from_frame,
    fixed_pearson_residuals,
)


def _state_and_prediction_counts():
    training = sp.csr_matrix(
        np.array(
            [
                [4, 0, 2, 1],
                [1, 5, 0, 2],
                [3, 1, 4, 0],
                [0, 2, 1, 7],
            ],
            dtype=np.float32,
        )
    )
    state = fit_fixed_pearson_state(training, ["GeneA", "GeneB", "GeneC", "GeneD"])
    prediction = sp.csr_matrix(
        np.array(
            [
                [8, 0, 1, 3],
                [0, 4, 2, 1],
                [2, 2, 0, 9],
                [3, 1, 5, 0],
                [1, 7, 0, 2],
            ],
            dtype=np.float32,
        )
    )
    return state, prediction


def test_fixed_residuals_are_chunk_and_batch_composition_invariant():
    state, prediction = _state_and_prediction_counts()
    all_at_once = fixed_pearson_residuals(prediction, state, chunk_size=100)
    one_at_a_time = fixed_pearson_residuals(prediction, state, chunk_size=1)
    np.testing.assert_array_equal(all_at_once, one_at_a_time)

    unrelated = sp.csr_matrix(np.array([[100, 1, 0, 0], [0, 0, 50, 2]]))
    extended = sp.vstack([prediction, unrelated], format="csr")
    extended_result = fixed_pearson_residuals(extended, state, chunk_size=2)
    np.testing.assert_array_equal(all_at_once, extended_result[: prediction.shape[0]])


def test_fixed_state_table_round_trip_preserves_transform():
    state, prediction = _state_and_prediction_counts()
    restored = fixed_state_from_frame(
        fixed_state_frame(state),
        n_training_cells=state.n_training_cells,
        total_training_counts=state.total_training_counts,
        theta=state.theta,
        clip_value=state.clip_value,
    )
    np.testing.assert_array_equal(
        fixed_pearson_residuals(prediction, state),
        fixed_pearson_residuals(prediction, restored),
    )


def test_fixed_residuals_are_row_order_invariant():
    state, prediction = _state_and_prediction_counts()
    order = np.array([3, 0, 4, 1, 2])
    expected = fixed_pearson_residuals(prediction, state)
    permuted = fixed_pearson_residuals(prediction[order], state)
    inverse = np.argsort(order)
    np.testing.assert_array_equal(expected, permuted[inverse])


def test_compact_selected_feature_transform_matches_full_transform():
    state, prediction = _state_and_prediction_counts()
    selected = [3, 1]
    full = fixed_pearson_residuals(
        prediction, state, feature_indices=selected, chunk_size=2
    )
    libraries = np.asarray(prediction.sum(axis=1)).reshape(-1)
    compact = fixed_pearson_residuals(
        prediction[:, selected],
        state,
        feature_indices=selected,
        library_sizes=libraries,
        chunk_size=3,
    )
    np.testing.assert_array_equal(full, compact)


def test_feature_audit_distinguishes_case_only_from_positional_mismatch():
    case_only = audit_feature_order(
        ["Atp6v1h", "Col23a1"], ["Atp6V1H", "Col23A1"]
    )
    assert not case_only["exact_positional_match"]
    assert case_only["casefold_positional_match"]
    assert case_only["case_only_mismatch_count"] == 2

    reordered = audit_feature_order(
        ["Atp6v1h", "Col23a1"], ["Col23A1", "Atp6V1H"]
    )
    assert not reordered["casefold_positional_match"]


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.array([[1.5, 0.0]]), "integer-like"),
        (np.array([[-1.0, 0.0]]), "negative"),
        (np.array([[np.nan, 0.0]]), "non-finite"),
    ],
)
def test_training_state_rejects_non_raw_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        fit_fixed_pearson_state(matrix, ["a", "b"])
