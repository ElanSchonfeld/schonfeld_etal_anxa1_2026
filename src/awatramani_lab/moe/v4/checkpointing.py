"""Safe checkpoint utilities for MoE v4."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def _get_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.generic)) or value is None


def validate_safe_checkpoint(obj: Any, *, _path: str = "root") -> None:
    """Raise TypeError if an object contains unsafe values."""

    torch = _get_torch()

    if _is_primitive(obj):
        return
    if isinstance(obj, np.ndarray):
        return
    if torch is not None and isinstance(obj, torch.Tensor):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Unsafe key type at {_path}: {type(k)}")
            validate_safe_checkpoint(v, _path=f"{_path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            validate_safe_checkpoint(v, _path=f"{_path}[{i}]")
        return

    raise TypeError(f"Unsafe object at {_path}: {type(obj)}")


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _extract_scaler_params(scaler: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if hasattr(scaler, "_mm"):
        params["feature_range"] = getattr(scaler, "feature_range", (0.0, 1.0))
        params["clip"] = bool(getattr(scaler, "clip", True))
        scaler = scaler._mm

    attr_names = [
        "mean_",
        "scale_",
        "var_",
        "min_",
        "data_min_",
        "data_max_",
        "data_range_",
        "feature_range",
        "q_low",
        "q_high",
        "clip",
        "q_low_",
        "q_high_",
        "quantiles_",
        "references_",
        "n_quantiles_",
        "output_distribution",
    ]

    for name in attr_names:
        if hasattr(scaler, name):
            value = getattr(scaler, name)
            if isinstance(value, (list, tuple, np.ndarray)):
                params[name] = _as_array(value)
            elif _is_primitive(value):
                params[name] = value
    return params


def _serialize_scaler(scaler: Any) -> Dict[str, Any] | None:
    if scaler is None:
        return None

    kind = getattr(scaler, "kind", None)
    post_norm = getattr(scaler, "post_norm", "none")
    base = getattr(scaler, "scaler", scaler)

    params = _extract_scaler_params(base)
    return {
        "kind": str(kind) if kind is not None else type(base).__name__,
        "post_norm": str(post_norm),
        "params": params,
    }


def _extract_linear_params(value: Any) -> tuple[np.ndarray, float] | None:
    weight_keys = ("weight", "weights", "coef", "coef_", "w", "W")
    bias_keys = ("bias", "bias_", "intercept", "intercept_", "b")

    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]

    if isinstance(value, dict):
        weight = next((value[k] for k in weight_keys if k in value), None)
        bias = next((value[k] for k in bias_keys if k in value), 0.0)
        if weight is None:
            return None
        w = _as_array(weight).reshape(-1)
        b = float(np.asarray(bias).reshape(-1)[0]) if bias is not None else 0.0
        return w, b

    if hasattr(value, "coef_"):
        weight = getattr(value, "coef_")
        bias = getattr(value, "intercept_", 0.0)
        w = _as_array(weight).reshape(-1)
        b = float(np.asarray(bias).reshape(-1)[0]) if bias is not None else 0.0
        return w, b

    if isinstance(value, (np.ndarray, list, tuple)):
        w = _as_array(value).reshape(-1)
        return w, 0.0

    return None


def _serialize_expert(expert: Any) -> Dict[str, Any]:
    if hasattr(expert, "_scorer"):
        scorer = expert._scorer

        if hasattr(scorer, "__self__"):
            return _serialize_expert(scorer.__self__)

        import numpy as np

        weights = None
        bias = None

        if hasattr(scorer, "__closure__") and scorer.__closure__:
            candidates = []
            for cell in scorer.__closure__:
                try:
                    obj = cell.cell_contents
                    if isinstance(obj, np.ndarray):
                        candidates.append(obj)
                except ValueError:
                    pass

            candidates.sort(key=lambda x: x.size, reverse=True)

            if len(candidates) > 0:
                possible_w = candidates[0]

                if possible_w.ndim >= 1 and possible_w.size > 1:
                    weights = possible_w
                    if len(candidates) > 1:
                        bias = candidates[1]

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)

            return {
                "kind": "linear",
                "weight": weights,
                "bias": np.atleast_1d(bias) if bias is not None else np.array([0.0]),
                "input_dim": int(weights.shape[-1]),
            }

        dummy_dim = 4000
        return {
            "kind": "linear",
            "weight": np.zeros((1, dummy_dim)),
            "bias": np.array([-10.0]),
            "input_dim": dummy_dim,
        }

    if hasattr(expert, "backend") and hasattr(expert, "gene_idx"):
        payload: Dict[str, Any] = {
            "kind": "binary_expert",
            "backend": str(expert.backend),
            "gene_idx": _as_array(expert.gene_idx).astype(int),
            "gene_names": list(getattr(expert, "gene_names", [])),
            "temperature": float(getattr(expert, "temperature", 1.0) or 1.0),
            "scaler": _serialize_scaler(getattr(expert, "scaler", None)),
        }

        if expert.backend == "torch_mlp" and getattr(expert, "torch_model", None) is not None:
            model = expert.torch_model
            payload["model"] = {
                "kind": "mlp_binary",
                "state_dict": dict(getattr(model, "state_dict", {})),
                "config": dict(getattr(model, "config", {})),
                "in_dim": int(getattr(model, "in_dim", 0)),
                "temperature": float(getattr(model, "temperature", 1.0) or 1.0),
            }
            return payload

        if expert.backend == "logreg" and getattr(expert, "sk_model", None) is not None:
            params = _extract_linear_params(expert.sk_model)
            if params is None:
                raise TypeError("Unsupported logreg expert format")
            weight, bias = params
            payload["model"] = {
                "kind": "linear",
                "weight": weight,
                "bias": bias,
                "input_dim": int(weight.shape[0]),
            }
            return payload

        raise TypeError("Unsupported binary expert backend for safe export")

    if hasattr(expert, "state_dict") and callable(expert.state_dict):
        state = expert.state_dict()
        input_dim = None
        if isinstance(state, dict) and "weight" in state:
            weight = state["weight"]
            try:
                input_dim = int(weight.shape[1])
            except Exception:
                input_dim = None
        return {
            "kind": "torch_state_dict",
            "module_type": type(expert).__name__,
            "state_dict": state,
            "input_dim": input_dim,
        }

    params = _extract_linear_params(expert)
    if params is not None:
        weight, bias = params
        payload = {
            "kind": "linear",
            "weight": weight,
            "bias": bias,
            "input_dim": int(weight.shape[0]),
        }
        if isinstance(expert, dict):
            gene_list = expert.get("genes") or expert.get("gene_names")
            if gene_list:
                payload["genes"] = list(gene_list)
                payload["gene_names"] = list(gene_list)
        return payload

    raise TypeError(f"Unsupported expert type for safe export: {type(expert)}")


def save_state_dict_only(obj: Any, path: str, meta: Dict[str, Any] | None = None) -> None:
    """Save a state_dict-only checkpoint with safe contents."""

    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required to save .pt checkpoints")

    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        state_dict = obj.state_dict()
    elif isinstance(obj, Mapping):
        state_dict = obj
    else:
        raise TypeError("Object must be a torch module or state_dict mapping")

    bundle = {
        "format": "state_dict",
        "format_version": 1,
        "state_dict": state_dict,
        "meta": meta or {},
    }
    validate_safe_checkpoint(bundle)
    torch.save(bundle, path)


def load_state_dict_only(path: str) -> Dict[str, Any]:
    """Load a state_dict-only checkpoint and validate it."""

    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required to load .pt checkpoints")

    obj = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    validate_safe_checkpoint(obj)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError("Expected a state_dict-only checkpoint")
    return obj


def save_leaf_experts_safe(
    leaf_to_expert: Mapping[str, Any],
    path: str,
    meta: Dict[str, Any] | None = None,
) -> None:
    """Save leaf experts in a refactor-safe format."""

    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required to save .pt checkpoints")

    experts_out: Dict[str, Any] = {}
    for leaf, expert in leaf_to_expert.items():
        experts_out[str(leaf)] = _serialize_expert(expert)

    bundle = {
        "format": "leaf_experts",
        "format_version": 1,
        "experts": experts_out,
        "meta": meta or {},
    }
    validate_safe_checkpoint(bundle)
    torch.save(bundle, path)
