"""Stacker utilities for MoE v4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .checkpointing import load_state_dict_only, save_state_dict_only


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def build_stacker_features(
    gate_outputs: Dict[str, Dict[str, np.ndarray]],
    node_outputs: Dict[str, Dict[str, Any]],
    leaf_order: List[str],
    node_order: Optional[List[str]] = None,
    *,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build stacker feature matrix from gate and node outputs."""

    if feature_spec is None:
        feature_spec = {
            "gate_features": "logits",
            "node_features": "logits",
            "gate_order": list(gate_outputs.keys()),
            "node_order": node_order or list(node_outputs.keys()),
            "node_to_leaves": {k: v.get("leaf_order", []) for k, v in node_outputs.items()},
            "leaf_order": list(leaf_order),
        }

    gate_features = feature_spec.get("gate_features", "logits")
    node_features = feature_spec.get("node_features", "logits")
    gate_order = feature_spec.get("gate_order") or list(gate_outputs.keys())
    node_order = feature_spec.get("node_order") or list(node_outputs.keys())

    parts: List[np.ndarray] = []

    for gate_name in gate_order:
        gate = gate_outputs.get(gate_name)
        if gate is None:
            raise ValueError(
                f"Missing gate outputs for {gate_name}. Available gates: {sorted(gate_outputs.keys())}"
            )
        if gate_features in gate:
            arr = np.asarray(gate[gate_features], dtype=float)
        elif gate_features == "probs" and "logits" in gate:
            arr = _softmax(np.asarray(gate["logits"], dtype=float))
        else:
            raise ValueError(f"Gate outputs missing '{gate_features}' for {gate_name}")
        parts.append(arr)

    node_rep = None
    missing_nodes: List[str] = []
    for node_name in node_order:
        node = node_outputs.get(node_name)
        if node is None:
            missing_nodes.append(node_name)
            continue
        if "logits" in node or "scores" in node:
            if node_rep is None:
                node_rep = "logits"
        elif "probs" in node:
            if node_rep is None:
                node_rep = "probs"
        else:
            raise ValueError(f"Node outputs missing logits/scores/probs for {node_name}. Keys={list(node.keys())}")
    if missing_nodes:
        raise ValueError(f"Missing node outputs for: {', '.join(missing_nodes)}")

    if node_rep is None:
        raise ValueError("No node features found for stacker construction")

    if node_rep == "logits":
        for node_name in node_order:
            node = node_outputs.get(node_name, {})
            if "logits" in node or "scores" in node:
                continue
            if "probs" in node:
                node_rep = "probs"
                break
            raise ValueError(f"Node outputs missing logits/scores/probs for {node_name}. Keys={list(node.keys())}")

    for node_name in node_order:
        node = node_outputs.get(node_name)
        if node is None:
            raise ValueError(f"Missing node outputs for {node_name}")
        if node_rep == "logits":
            if "logits" in node:
                arr = np.asarray(node["logits"], dtype=float)
            elif "scores" in node:
                arr = np.asarray(node["scores"], dtype=float)
            else:
                raise ValueError(
                    f"Node outputs missing logits/scores for {node_name}. Keys={list(node.keys())}"
                )
        else:
            if "probs" in node:
                arr = np.asarray(node["probs"], dtype=float)
            elif "logits" in node:
                arr = _softmax(np.asarray(node["logits"], dtype=float))
            elif "scores" in node:
                arr = _softmax(np.asarray(node["scores"], dtype=float))
            else:
                raise ValueError(
                    f"Node outputs missing probs/logits/scores for {node_name}. Keys={list(node.keys())}"
                )
        parts.append(arr)

    X_meta = np.concatenate(parts, axis=1) if parts else np.empty((0, 0))
    return X_meta, feature_spec


def train_leaf_stacker(
    X_meta_train: np.ndarray,
    y_leaf_train: np.ndarray,
    leaf_order: List[str],
    *,
    reg: float = 1e-4,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 0,
    feature_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a linear softmax stacker for leaf prediction."""

    X = np.asarray(X_meta_train, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_meta_train must be 2D, got shape={X.shape}")
    y = np.asarray(y_leaf_train).astype(str)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_leaf_train length must match X_meta_train rows")
    if not leaf_order:
        raise ValueError("leaf_order must be non-empty")

    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaf_order)}
    y_idx = np.array([leaf_to_idx[str(lbl)] for lbl in y], dtype=int)

    n_samples, n_features = X.shape
    n_classes = len(leaf_order)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(n_features, n_classes))
    b = np.zeros((n_classes,), dtype=float)

    y_onehot = np.zeros((n_samples, n_classes), dtype=float)
    y_onehot[np.arange(n_samples), y_idx] = 1.0

    for _ in range(int(epochs)):
        logits = X @ W + b
        probs = _softmax(logits)
        grad_logits = (probs - y_onehot) / float(n_samples)
        grad_W = X.T @ grad_logits + reg * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return {
        "kind": "linear_softmax",
        "W": W.astype(np.float32),
        "b": b.astype(np.float32),
        "leaf_order": list(leaf_order),
        "feature_spec": feature_spec or {},
        "input_dim": int(n_features),
        "reg": float(reg),
        "lr": float(lr),
        "epochs": int(epochs),
    }


def predict_leaf_stacker(stacker_bundle: Dict[str, Any], X_meta: np.ndarray) -> np.ndarray:
    """Predict leaf probabilities using a trained stacker bundle."""

    W = np.asarray(stacker_bundle.get("W"))
    b = np.asarray(stacker_bundle.get("b"))
    if W.ndim != 2 or b.ndim != 1:
        raise ValueError("stacker_bundle must contain 2D W and 1D b")

    X = np.asarray(X_meta, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_meta must be 2D, got shape={X.shape}")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X_meta feature dimension does not match stacker weights")

    logits = X @ W + b
    return _softmax(logits)


def save_stacker_bundle(stacker_bundle: Dict[str, Any], path: str) -> None:
    """Save a stacker bundle using safe checkpointing utilities."""

    state = {
        "W": np.asarray(stacker_bundle["W"]),
        "b": np.asarray(stacker_bundle["b"]),
    }
    meta = {
        "kind": str(stacker_bundle.get("kind", "linear_softmax")),
        "leaf_order": list(stacker_bundle.get("leaf_order", [])),
        "feature_spec": stacker_bundle.get("feature_spec", {}),
        "input_dim": int(stacker_bundle.get("input_dim", state["W"].shape[0])),
    }
    save_state_dict_only(state, path, meta=meta)


def load_stacker_bundle(path: str) -> Dict[str, Any]:
    """Load a stacker bundle saved with save_stacker_bundle."""

    bundle = load_state_dict_only(path)
    state = bundle.get("state_dict", {})
    meta = bundle.get("meta", {})

    W = np.asarray(state.get("W"))
    b = np.asarray(state.get("b"))
    if W.ndim != 2 or b.ndim != 1:
        raise ValueError("Stacker checkpoint missing W/b")

    return {
        "kind": str(meta.get("kind", "linear_softmax")),
        "W": W,
        "b": b,
        "leaf_order": list(meta.get("leaf_order", [])),
        "feature_spec": meta.get("feature_spec", {}),
        "input_dim": int(meta.get("input_dim", W.shape[0])),
    }
