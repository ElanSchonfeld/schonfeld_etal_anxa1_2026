import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import retrain_hmoe_sct_hybrid as training


def test_reference_topology_matches_hybrid_gate_representations():
    meta = json.loads(training.MODEL_TEMPLATE.read_text())

    assert meta["pipeline"] == "v4_improved_unified"
    assert len(meta["gate_nodes"]) == 17
    assert sum(len(meta["gates"][gate]) for gate in meta["gate_nodes"]) == 34
    for gate in meta["gate_nodes"]:
        expected = "raw" if training.gate_depth(gate) <= training.SHALLOW_DEPTH else "sct"
        assert meta["gate_repr"][gate] == expected

    reference_dir = training.MODEL_TEMPLATE.parent / "child_models"
    assert len(list(reference_dir.rglob("*.safe.pt"))) == 34


def test_sct_transform_is_finite_and_deterministic():
    counts = np.array(
        [[0, 1, 3, 0], [2, 0, 1, 4], [1, 2, 0, 1]],
        dtype=np.float32,
    )

    first = training.sct_transform(counts)
    second = training.sct_transform(counts)

    assert first.shape == counts.shape
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_calb1_sox6_uses_taxonomy_prefix():
    assert training.infer_family("Calb1:Sox6") == "Calb1"


def test_training_output_cannot_overwrite_checked_model_by_default():
    checked_model = training.PROJECT_ROOT / "hmoe_annotate" / "model"
    assert checked_model not in training.OUT_DIR.parents
