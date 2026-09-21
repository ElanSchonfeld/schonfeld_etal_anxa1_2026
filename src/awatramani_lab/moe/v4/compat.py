"""Compatibility shims for MoE v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def train_moe_v2_compat(*args: Any, **kwargs: Any) -> Any:
    """Compatibility entry point for v2-style training (placeholder)."""
    raise NotImplementedError("train_moe_v2_compat is not implemented yet.")


def predict_moe_v2_compat(*args: Any, **kwargs: Any) -> Any:
    """Compatibility entry point for v2-style prediction (placeholder)."""
    raise NotImplementedError("predict_moe_v2_compat is not implemented yet.")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _robust_minmax_transform(
    X: np.ndarray,
    q_low_: np.ndarray,
    scale_: np.ndarray,
    clip: bool,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Z = (X - q_low_) / scale_
    if clip:
        Z = np.clip(Z, 0.0, 1.0)
    return Z.astype(np.float32)


def _l2_norm_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(denom, 1e-12)


def load_family_gate_safe(safe_dir: Path) -> Dict[str, Any]:
    """Load a v3-style SAFE family gate from *safe_dir*."""
    safe_dir = Path(safe_dir)
    meta = json.loads((safe_dir / "meta.json").read_text())

    st = np.load(safe_dir / meta["stacker_file"], allow_pickle=False)
    stacker = {
        "coef": st["coef"].astype(np.float32),
        "intercept": st["intercept"].astype(np.float32),
        "classes_raw": st["classes"].tolist(),
    }

    experts: Dict[str, Dict[str, Any]] = {}
    for cname in meta["class_names"]:
        ex_dir = safe_dir / "experts" / cname
        g = np.load(ex_dir / "genes.npz", allow_pickle=False)
        m = np.load(ex_dir / "model.npz", allow_pickle=False)
        smeta = json.loads((ex_dir / "scaler_meta.json").read_text())
        s = np.load(ex_dir / "scaler.npz", allow_pickle=False)
        emeta = json.loads((ex_dir / "expert_meta.json").read_text())

        experts[cname] = {
            "gene_idx": g["gene_idx"].astype(int),
            "weights": m["weights"].astype(np.float32),
            "bias": float(np.asarray(m["bias"]).reshape(-1)[0]),
            "temperature": float(emeta["temperature"]),
            "scaler_meta": smeta,
            "q_low_": s["q_low_"].astype(np.float32),
            "scale_": s["scale_"].astype(np.float32),
        }

    id_to_fam = meta.get("id_to_fam", {})
    id_to_fam = {int(k): str(v) for k, v in id_to_fam.items()}

    feature_file = safe_dir / "features.json"
    feature_names: Optional[List[str]] = None
    if feature_file.exists():
        feature_names = json.loads(feature_file.read_text())

    return {
        "meta": meta,
        "stacker": stacker,
        "experts": experts,
        "id_to_fam": id_to_fam,
        "feature_names": feature_names,
    }


def predict_family_gate_safe(
    gate_safe: Dict[str, Any],
    X: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Predict family probabilities using a SAFE family gate."""
    X = np.asarray(X, dtype=np.float32)
    class_names: List[str] = gate_safe["meta"]["class_names"]
    use_logit: bool = bool(gate_safe["meta"]["use_logit_features"])

    P_cols: List[np.ndarray] = []
    for cname in class_names:
        ex = gate_safe["experts"][cname]
        idx = ex["gene_idx"]
        Xs = X[:, idx].astype(np.float32)

        smeta = ex["scaler_meta"]
        clip = bool(smeta.get("clip", True))
        Xs = _robust_minmax_transform(Xs, ex["q_low_"], ex["scale_"], clip=clip)

        post_norm = str(smeta.get("post_norm", "none")).lower()
        if post_norm == "l2":
            Xs = _l2_norm_rows(Xs)

        w = ex["weights"].reshape(-1)
        b = float(ex["bias"])
        logit_val = Xs @ w + b
        p = _sigmoid(logit_val)

        T = float(ex["temperature"]) if ex["temperature"] is not None else 1.0
        if abs(T - 1.0) > 1e-6:
            p = _sigmoid(_logit(p) / max(T, 1e-6))

        P_cols.append(p.astype(np.float32))

    P = np.vstack(P_cols).T
    P = np.clip(P, 1e-6, 1.0 - 1e-6)
    Fm = _logit(P) if use_logit else P

    st = gate_safe["stacker"]
    classes_raw = st["classes_raw"]

    stacker_names = [gate_safe["id_to_fam"][int(i)] for i in classes_raw]
    idx_order = [stacker_names.index(c) for c in class_names]

    coef = st["coef"][idx_order][:, idx_order]
    intercept = st["intercept"][idx_order]

    logits = Fm @ coef.T + intercept
    probs = _softmax(logits).astype(np.float32)
    return probs, class_names
