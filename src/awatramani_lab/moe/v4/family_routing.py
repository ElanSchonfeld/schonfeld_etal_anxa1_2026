"""Family-gate-injected hierarchical routing for MoE v4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .compat import predict_family_gate_safe
from .hierarchy import get_node
from .pipelines import _compute_gate_outputs as _compute_gate_outputs_pipelines
from .roc_router import (
    _compute_gate_outputs as _compute_gate_outputs_roc,
    predict_router_hard,
)
from .stacker import build_stacker_features, predict_leaf_stacker


def max_depth_from_bundle(bundle: Dict[str, Any]) -> int:
    """Return the maximum depth of internal nodes in a bundle."""
    all_internal = [
        k
        for k in bundle["node_to_leaves"].keys()
        if isinstance(k, str) and k.startswith("node|")
    ]
    return max(k.count("|") for k in all_internal) if all_internal else 0


def _resolve_gate_outputs(
    bundle: Dict[str, Any], X: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute gate outputs using the appropriate backend."""
    gates = bundle.get("gates", {})
    return _compute_gate_outputs_pipelines(gates, X)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def fit_ddc_calibration(
    family_gate_safe: Dict[str, Any],
    X_human: np.ndarray,
    y_human_family: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """Fit Platt calibration (temperature + bias) on the Ddc gate."""
    P_fam, fams = predict_family_gate_safe(family_gate_safe, X_human)
    fam_to_i = {f: i for i, f in enumerate(fams)}
    p_sox = P_fam[:, fam_to_i["Sox6"]]
    p_calb = P_fam[:, fam_to_i["Calb1"]]

    mask = np.isin(y_human_family, ["Sox6", "Calb1"])
    p_sox_m = p_sox[mask]
    p_calb_m = p_calb[mask]
    y_m = (np.asarray(y_human_family)[mask] == "Sox6").astype(int)

    denom = np.clip(p_sox_m + p_calb_m, 1e-7, None)
    p_sox_given = p_sox_m / denom
    raw_logit = _logit(p_sox_given)

    temps = np.concatenate([
        np.arange(0.4, 1.0, 0.025),
        np.arange(1.0, 2.51, 0.05),
    ])
    biases = np.arange(-1.5, 1.51, 0.025)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.zeros(len(y_m), dtype=int)
    for fold_i, (_, val_idx) in enumerate(skf.split(raw_logit, y_m)):
        fold_ids[val_idx] = fold_i

    best_score = -1.0
    best_t, best_b = 1.0, 0.0

    for t in temps:
        for b in biases:
            cal_p = _sigmoid(raw_logit / t + b)
            pred = (cal_p >= 0.5).astype(int)
            scores = []
            for fold in range(n_splits):
                val_mask = fold_ids == fold
                if val_mask.sum() == 0:
                    continue
                y_val, p_val = y_m[val_mask], pred[val_mask]
                r0 = (p_val[y_val == 0] == 0).mean() if (y_val == 0).sum() > 0 else 0.0
                r1 = (p_val[y_val == 1] == 1).mean() if (y_val == 1).sum() > 0 else 0.0
                scores.append((r0 + r1) / 2.0)
            mean_score = float(np.mean(scores)) if scores else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_t, best_b = float(t), float(b)

    return {
        "temperature": best_t,
        "bias": best_b,
        "cv_balanced_accuracy": best_score,
        "n_cells": int(mask.sum()),
    }


def fit_root_calibration(
    family_gate_safe: Dict[str, Any],
    X_human: np.ndarray,
    y_human_family: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """Fit Platt calibration for the root gate (Gad2 vs Ddc)."""
    P_fam, fams = predict_family_gate_safe(family_gate_safe, X_human)
    fam_to_i = {f: i for i, f in enumerate(fams)}
    p_gad = P_fam[:, fam_to_i["Gad2"]]

    mask = np.isin(y_human_family, ["Sox6", "Calb1", "Gad2"])
    p_gad_m = p_gad[mask]
    y_m = (np.asarray(y_human_family)[mask] == "Gad2").astype(int)

    raw_logit = _logit(p_gad_m)

    temps = np.concatenate([
        np.arange(0.2, 1.0, 0.025),
        np.arange(1.0, 4.01, 0.05),
    ])
    biases = np.arange(-4.0, 1.51, 0.025)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.zeros(len(y_m), dtype=int)
    for fold_i, (_, val_idx) in enumerate(skf.split(raw_logit, y_m)):
        fold_ids[val_idx] = fold_i

    best_score = -1.0
    best_t, best_b = 1.0, 0.0

    for t in temps:
        for b in biases:
            cal_p = _sigmoid(raw_logit / t + b)
            pred = (cal_p >= 0.5).astype(int)
            scores = []
            for fold in range(n_splits):
                val_mask = fold_ids == fold
                if val_mask.sum() == 0:
                    continue
                y_val, p_val = y_m[val_mask], pred[val_mask]
                r0 = (p_val[y_val == 0] == 0).mean() if (y_val == 0).sum() > 0 else 0.0
                r1 = (p_val[y_val == 1] == 1).mean() if (y_val == 1).sum() > 0 else 0.0
                scores.append((r0 + r1) / 2.0)
            mean_score = float(np.mean(scores)) if scores else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_t, best_b = float(t), float(b)

    return {
        "temperature": best_t,
        "bias": best_b,
        "cv_balanced_accuracy": best_score,
        "n_cells": int(mask.sum()),
    }


def override_gate_outputs_with_family(
    router_bundle: Dict[str, Any],
    gate_outputs: Dict[str, Dict[str, np.ndarray]],
    X: np.ndarray,
    *,
    family_gate_safe: Dict[str, Any],
    ddc_calibration: Optional[Dict[str, float]] = None,
    root_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Replace the root and Ddc gate probabilities with family-gate outputs."""
    leaves = list(router_bundle["leaves"])
    node_to_leaves = router_bundle["node_to_leaves"]
    excluded = set(router_bundle.get("excluded_leaves", []))

    P_fam, fams = predict_family_gate_safe(family_gate_safe, X)
    fam_to_i = {f: i for i, f in enumerate(fams)}
    p_calb = P_fam[:, fam_to_i["Calb1"]].astype(np.float64)
    p_sox = P_fam[:, fam_to_i["Sox6"]].astype(np.float64)
    p_gad = P_fam[:, fam_to_i["Gad2"]].astype(np.float64)

    def set_binary_gate(
        node_key: str, p_pos: np.ndarray, pos_is_child0: bool
    ) -> None:
        if pos_is_child0:
            probs = np.vstack([p_pos, 1.0 - p_pos]).T
        else:
            probs = np.vstack([1.0 - p_pos, p_pos]).T
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        gate_outputs[node_key]["probs"] = probs
        gate_outputs[node_key]["logits"] = np.log(probs)

    root_gate = "node|rest_of_dendrogram"
    if root_gate in gate_outputs:
        p_gad_eff = p_gad
        if root_calibration is not None:
            T_root = root_calibration.get("temperature", 1.0)
            b_root = root_calibration.get("bias", 0.0)
            p_gad_eff = _sigmoid(_logit(p_gad_eff) / T_root + b_root)
        child0, child1 = gate_outputs[root_gate]["nodes"]
        gad_leaves = {l for l in leaves if l.startswith("Gad2:")} - excluded
        L0 = set(node_to_leaves.get(child0, [])) - excluded
        pos_is_child0 = len(L0 & gad_leaves) > 0
        set_binary_gate(root_gate, p_gad_eff, pos_is_child0)

    ddc_gate = "node|rest_of_dendrogram|Ddc-high,Slc6a3-high"
    if ddc_gate in gate_outputs:
        denom = np.clip(p_sox + p_calb, 1e-6, None)
        p_sox_given = p_sox / denom
        if ddc_calibration is not None:
            T = ddc_calibration.get("temperature", 1.0)
            b = ddc_calibration.get("bias", 0.0)
            p_sox_given = _sigmoid(_logit(p_sox_given) / T + b)
        child0, child1 = gate_outputs[ddc_gate]["nodes"]
        sox_leaves = {l for l in leaves if l.startswith("Sox6:")} - excluded
        L0 = set(node_to_leaves.get(child0, [])) - excluded
        pos_is_child0 = len(L0 & sox_leaves) > 0
        set_binary_gate(ddc_gate, p_sox_given, pos_is_child0)

    return gate_outputs


def predict_router_soft_family(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    family_gate_safe: Dict[str, Any],
    normalize: bool = True,
    ddc_calibration: Optional[Dict[str, float]] = None,
    root_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Soft routing with family-gate-overridden top-level gates."""
    tree = router_bundle["tree"]
    root_key = router_bundle["root_key"]
    leaves = list(router_bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    X = np.asarray(X, dtype=np.float32)
    training_lib_size = router_bundle.get("training_library_size")
    if training_lib_size is not None:
        row_sums = X.sum(axis=1, keepdims=True)
        scale = np.float32(training_lib_size) / np.maximum(
            row_sums, np.float32(1.0)
        )
        X = X * scale

    gate_outputs = _resolve_gate_outputs(router_bundle, X)
    gate_outputs = override_gate_outputs_with_family(
        router_bundle, gate_outputs, X, family_gate_safe=family_gate_safe,
        ddc_calibration=ddc_calibration, root_calibration=root_calibration,
    )

    X_arr = np.asarray(X, dtype=float)
    n_samples = X_arr.shape[0]
    P_leaf = np.zeros((n_samples, len(leaves)), dtype=float)

    def _route(path: List[str], mass: np.ndarray) -> None:
        node_path = "|".join(path)
        sub = get_node(tree, path)
        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_full = [f"{node_path}|{key}" for key in child_keys]
            if not child_keys:
                return
            gate = gate_outputs.get(node_path)
            if gate is None:
                probs = np.full(
                    (n_samples, len(child_keys)), 1.0 / float(len(child_keys))
                )
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != child_full:
                    idx = [gate_nodes.index(ch) for ch in child_full]
                    probs = probs[:, idx]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            for j, child in enumerate(child_keys):
                _route(path + [child], mass * probs[:, j])

        elif isinstance(sub, list):
            leaf_labels = list(sub)
            if not leaf_labels:
                return
            gate = gate_outputs.get(node_path)
            if gate is None or len(leaf_labels) == 1:
                probs = np.full(
                    (n_samples, len(leaf_labels)),
                    1.0 / float(len(leaf_labels)),
                )
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != leaf_labels:
                    idx = [gate_nodes.index(ch) for ch in leaf_labels]
                    probs = probs[:, idx]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            for j, leaf in enumerate(leaf_labels):
                k = leaf_to_idx.get(leaf)
                if k is not None:
                    P_leaf[:, k] += mass * probs[:, j]
        else:
            raise TypeError(
                f"Unexpected subtree type at {node_path}: {type(sub)}"
            )

    _route([root_key], np.ones(n_samples, dtype=float))
    if normalize and P_leaf.size:
        P_leaf = P_leaf / np.maximum(P_leaf.sum(axis=1, keepdims=True), 1e-12)

    hard = predict_router_hard_family(
        router_bundle,
        X,
        family_gate_safe=family_gate_safe,
        gate_outputs=gate_outputs,
        root_calibration=root_calibration,
    )
    return {
        "P_sub": P_leaf,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "leaves": leaves,
    }


def predict_router_hard_family(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    family_gate_safe: Dict[str, Any],
    gate_outputs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ddc_calibration: Optional[Dict[str, float]] = None,
    root_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Hard routing with family-gate-overridden top-level gates."""
    if gate_outputs is None:
        gate_outputs = _resolve_gate_outputs(router_bundle, X)
        gate_outputs = override_gate_outputs_with_family(
            router_bundle, gate_outputs, X, family_gate_safe=family_gate_safe,
            ddc_calibration=ddc_calibration, root_calibration=root_calibration,
        )
    return predict_router_hard(router_bundle, X, gate_outputs=gate_outputs)


def predict_router_stacked_family(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    family_gate_safe: Dict[str, Any],
    ddc_calibration: Optional[Dict[str, float]] = None,
    root_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Stacked routing with family-gate-overridden top-level gates."""
    if "stacker" not in router_bundle:
        raise ValueError("router_bundle missing stacker")

    gate_outputs = _resolve_gate_outputs(router_bundle, X)
    gate_outputs = override_gate_outputs_with_family(
        router_bundle, gate_outputs, X, family_gate_safe=family_gate_safe,
        ddc_calibration=ddc_calibration, root_calibration=root_calibration,
    )

    node_outputs = {
        node: {"probs": gate_outputs[node]["probs"]} for node in gate_outputs
    }
    stacker = router_bundle["stacker"]

    X_meta, _ = build_stacker_features(
        gate_outputs,
        node_outputs,
        list(router_bundle["leaves"]),
        node_order=list(gate_outputs.keys()),
        feature_spec=stacker.get("feature_spec"),
    )
    P_sub = predict_leaf_stacker(stacker, X_meta)

    hard = predict_router_hard(router_bundle, X, gate_outputs=gate_outputs)
    return {
        "P_sub": P_sub,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "leaves": list(router_bundle["leaves"]),
    }
