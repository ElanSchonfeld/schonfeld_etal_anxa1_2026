"""Temperature calibration utilities for MoE v4 outputs."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .routing_spec import EXCLUDED_LEAVES, filter_excluded_leaves


def _logsumexp(x: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_sum = np.exp(x - x_max).sum(axis=axis, keepdims=True)
    out = x_max + np.log(np.maximum(exp_sum, 1e-12))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def apply_temperature_to_probs(P: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """Apply temperature scaling to a probability matrix."""

    if T <= 0:
        raise ValueError("Temperature must be > 0")
    P = np.asarray(P, dtype=float)
    logits = np.log(np.maximum(P, eps))
    scaled = logits / float(T)
    log_probs = scaled - _logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(log_probs)


def fit_temperature_from_probs(
    P: np.ndarray,
    y_true: np.ndarray,
    labels: List[str],
    excluded_leaves: Optional[List[str]] = None,
) -> float:
    """Fit a temperature by minimizing NLL with a simple grid search."""

    excluded = excluded_leaves or EXCLUDED_LEAVES
    kept, _ = filter_excluded_leaves(list(labels), excluded)
    if not kept:
        raise ValueError("All labels excluded; cannot calibrate.")
    idx = [labels.index(s) for s in kept]

    P = np.asarray(P, dtype=float)[:, idx]
    y_true = np.asarray(y_true).astype(str)
    label_to_idx = {label: i for i, label in enumerate(kept)}
    y_idx = np.array([label_to_idx[str(lbl)] for lbl in y_true], dtype=int)

    logits = np.log(np.maximum(P, 1e-12))

    best_T = 1.0
    best_nll = np.inf
    for T in np.linspace(0.5, 5.0, 46):
        scaled = logits / float(T)
        log_probs = scaled - _logsumexp(scaled, axis=1, keepdims=True)
        nll = -np.mean(log_probs[np.arange(log_probs.shape[0]), y_idx])
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return best_T
