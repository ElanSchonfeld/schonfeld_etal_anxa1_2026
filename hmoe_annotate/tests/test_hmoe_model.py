from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hmoe_annotate import hmoe_model as hm  # noqa: E402


def test_model_loads_and_returns_normalized_leaf_probabilities():
    bundle = hm.load_model(ROOT / "hmoe_annotate" / "model")
    rng = np.random.default_rng(42)
    counts = rng.poisson(0.1, size=(4, len(bundle["feature_names"]))).astype(
        np.float32
    )

    probabilities = hm.predict_proba(bundle, counts)

    assert probabilities.shape == (4, len(bundle["leaves"]))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    assert {hm.infer_family(value) for value in bundle["leaves"]} == {
        "Calb1",
        "Gad2",
        "Sox6",
    }
