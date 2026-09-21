"""Training-reference Pearson residuals for HMoE transfer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def feature_names_hash(values: Sequence[str], *, casefold: bool = False) -> str:
    digest = hashlib.sha256()
    for value in values:
        text = str(value).casefold() if casefold else str(value)
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_feature_order(
    training_features: Sequence[str], model_features: Sequence[str]
) -> dict[str, Any]:
    training = [str(value) for value in training_features]
    model = [str(value) for value in model_features]
    return {
        "training_n_features": len(training),
        "model_n_features": len(model),
        "training_feature_hash": feature_names_hash(training),
        "model_feature_hash": feature_names_hash(model),
        "training_casefold_hash": feature_names_hash(training, casefold=True),
        "model_casefold_hash": feature_names_hash(model, casefold=True),
        "exact_positional_match": training == model,
        "casefold_positional_match": (
            [value.casefold() for value in training]
            == [value.casefold() for value in model]
        ),
        "case_only_mismatch_count": (
            sum(
                left != right and left.casefold() == right.casefold()
                for left, right in zip(training, model)
            )
            if len(training) == len(model)
            else None
        ),
    }


@dataclass(frozen=True)
class FixedPearsonState:
    feature_names: tuple[str, ...]
    gene_rates: np.ndarray
    n_training_cells: int
    total_training_counts: float
    theta: float
    clip_value: float
    feature_hash: str
    feature_casefold_hash: str

    def __post_init__(self) -> None:
        rates = np.asarray(self.gene_rates, dtype=np.float64)
        if rates.ndim != 1 or len(rates) != len(self.feature_names):
            raise ValueError("gene rates do not match the feature-name vector")
        if not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError("gene rates must be finite and nonnegative")
        if not np.isclose(rates.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError(f"gene rates sum to {rates.sum()}, not 1")
        if self.n_training_cells <= 0 or self.total_training_counts <= 0:
            raise ValueError("training state must contain cells and counts")
        if self.theta <= 0 or self.clip_value <= 0:
            raise ValueError("theta and clip value must be positive")
        if self.feature_hash != feature_names_hash(self.feature_names):
            raise ValueError("feature hash does not match feature names")
        if self.feature_casefold_hash != feature_names_hash(
            self.feature_names, casefold=True
        ):
            raise ValueError("casefold feature hash does not match feature names")
        object.__setattr__(self, "gene_rates", rates)


def fixed_state_frame(state: FixedPearsonState) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_index": np.arange(len(state.feature_names), dtype=np.int64),
            "feature_name": list(state.feature_names),
            "gene_rate": state.gene_rates,
        }
    )


def fixed_state_from_frame(
    frame: pd.DataFrame,
    *,
    n_training_cells: int,
    total_training_counts: float,
    theta: float,
    clip_value: float,
) -> FixedPearsonState:
    required = {"feature_index", "feature_name", "gene_rate"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"fixed state table missing columns: {missing}")
    ordered = frame.sort_values("feature_index", kind="stable").reset_index(drop=True)
    expected = np.arange(len(ordered), dtype=np.int64)
    observed = pd.to_numeric(ordered["feature_index"], errors="raise").to_numpy()
    if not np.array_equal(expected, observed):
        raise ValueError("fixed state feature indices are not contiguous from zero")
    names = tuple(ordered["feature_name"].astype(str))
    return FixedPearsonState(
        feature_names=names,
        gene_rates=pd.to_numeric(ordered["gene_rate"], errors="raise").to_numpy(),
        n_training_cells=int(n_training_cells),
        total_training_counts=float(total_training_counts),
        theta=float(theta),
        clip_value=float(clip_value),
        feature_hash=feature_names_hash(names),
        feature_casefold_hash=feature_names_hash(names, casefold=True),
    )


def _count_values(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.data)
    return np.asarray(matrix)


def validate_raw_counts(matrix: Any) -> None:
    if len(matrix.shape) != 2:
        raise ValueError(f"count matrix must be 2D, got {matrix.shape}")
    values = _count_values(matrix)
    if not np.isfinite(values).all():
        raise ValueError("count matrix contains non-finite values")
    if (values < 0).any():
        raise ValueError("count matrix contains negative values")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError("count matrix is not integer-like raw counts")


def fit_fixed_pearson_state(
    training_counts: Any,
    feature_names: Sequence[str],
    *,
    theta: float = 100.0,
) -> FixedPearsonState:
    """Fit the state used by the deployed full-training HMoE SCT gates."""

    validate_raw_counts(training_counts)
    names = tuple(str(value) for value in feature_names)
    if training_counts.shape[1] != len(names):
        raise ValueError("training matrix columns do not match feature names")
    feature_sums = np.asarray(training_counts.sum(axis=0), dtype=np.float64).reshape(-1)
    total = float(feature_sums.sum())
    if total <= 0:
        raise ValueError("training matrix has zero total counts")
    n_cells = int(training_counts.shape[0])
    return FixedPearsonState(
        feature_names=names,
        gene_rates=feature_sums / total,
        n_training_cells=n_cells,
        total_training_counts=total,
        theta=float(theta),
        clip_value=float(np.sqrt(n_cells)),
        feature_hash=feature_names_hash(names),
        feature_casefold_hash=feature_names_hash(names, casefold=True),
    )


def fixed_pearson_residuals(
    counts: Any,
    state: FixedPearsonState,
    *,
    feature_indices: Sequence[int] | None = None,
    library_sizes: Sequence[float] | None = None,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Apply frozen training parameters independent of batch and chunking."""

    validate_raw_counts(counts)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    indices = (
        np.arange(len(state.feature_names), dtype=int)
        if feature_indices is None
        else np.asarray(feature_indices, dtype=int)
    )
    if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
        raise ValueError("feature indices must be a unique one-dimensional vector")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(state.feature_names)):
        raise ValueError("feature index is outside the training feature space")

    full_input = counts.shape[1] == len(state.feature_names)
    compact_input = feature_indices is not None and counts.shape[1] == len(indices)
    if not full_input and not compact_input:
        raise ValueError(
            "counts must contain the full training feature space or exactly the "
            "requested compact feature vector"
        )
    if library_sizes is None:
        if not full_input:
            raise ValueError("compact counts require full-feature library sizes")
        libraries = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    else:
        libraries = np.asarray(library_sizes, dtype=np.float64).reshape(-1)
        if len(libraries) != counts.shape[0]:
            raise ValueError("library-size vector does not match count rows")
        if not np.isfinite(libraries).all() or (libraries < 0).any():
            raise ValueError("library sizes must be finite and nonnegative")

    output = np.empty((counts.shape[0], len(indices)), dtype=np.float32)
    rates = state.gene_rates[indices]
    for start in range(0, counts.shape[0], chunk_size):
        stop = min(start + chunk_size, counts.shape[0])
        block = counts[start:stop]
        if full_input:
            block = block[:, indices]
        if sp.issparse(block):
            block = block.toarray()
        observed = np.asarray(block, dtype=np.float64)
        mu = libraries[start:stop, None] * rates[None, :]
        denominator = np.sqrt(mu + mu**2 / state.theta)
        denominator[denominator < 1e-12] = 1.0
        residuals = (observed - mu) / denominator
        np.clip(
            residuals,
            -state.clip_value,
            state.clip_value,
            out=residuals,
        )
        residuals[~np.isfinite(residuals)] = 0.0
        output[start:stop] = residuals.astype(np.float32)
    return output
