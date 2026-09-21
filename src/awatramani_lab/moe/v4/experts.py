"""Leaf-level expert wrappers for MoE v4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .checkpointing import save_leaf_experts_safe
from .progress import progress
from .routing_spec import node_dir_name

ScoreFn = Callable[[np.ndarray], np.ndarray]

__all__ = [
    "ExpertAdapter",
    "adapt_expert",
    "load_expert_from_pt",
    "load_v3_leaf_experts",
    "load_leaf_experts_safe",
    "train_leaf_experts_ovr",
    "train_node_specific_leaf_experts",
    "filter_leaf_candidates",
    "build_leaf_score_matrix",
    "apply_candidate_mask",
    "score_leaf_experts",
    "combine_node_weights_with_leaf_scores",
]


class ExpertAdapter:
    """Uniform wrapper with a score(X) -> np.ndarray interface."""

    def __init__(
        self,
        scorer: ScoreFn,
        name: str | None = None,
        *,
        gene_idx: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self._scorer = scorer
        self.gene_idx = None if gene_idx is None else np.asarray(gene_idx, dtype=int)
        self.gene_names = list(gene_names) if gene_names is not None else None

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.gene_idx is not None:
            X = np.asarray(X, dtype=float)[:, self.gene_idx]
        return np.asarray(self._scorer(X), dtype=float).reshape(-1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.score(X)


def filter_leaf_candidates(
    leaf_candidates: Iterable[Iterable[str]] | Iterable[str] | None,
    allowed_leaves: Iterable[str],
    n_samples: int,
) -> List[List[str]]:
    """Normalize candidate leaves into a per-sample list and filter to allowed leaves."""

    allowed = list(allowed_leaves)
    allowed_set = set(allowed)

    if leaf_candidates is None:
        return [allowed[:] for _ in range(n_samples)]

    if isinstance(leaf_candidates, (list, tuple)) and leaf_candidates and isinstance(
        next(iter(leaf_candidates)), str
    ):
        candidates = [list(leaf_candidates) for _ in range(n_samples)]
    else:
        candidates = [list(c) for c in leaf_candidates]

    out: List[List[str]] = []
    for c in candidates:
        out.append([s for s in c if s in allowed_set])
    return out


def _score_model(model: Any, X: np.ndarray) -> np.ndarray:
    """Score samples for a single leaf expert."""

    if hasattr(model, "score"):
        scores = model.score(X)
    elif callable(model):
        scores = model(X)
    elif hasattr(model, "predict_proba_pos"):
        scores = model.predict_proba_pos(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        scores = np.asarray(scores)
        if scores.ndim == 2 and scores.shape[1] >= 2:
            scores = scores[:, -1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict"):
        scores = model.predict(X)
    else:
        raise TypeError("Expert model must be callable or implement predict_proba/decision_function/predict.")

    scores = np.asarray(scores, dtype=float).reshape(-1)
    return scores


def adapt_expert(expert: Any, name: str | None = None) -> ExpertAdapter:
    """Adapt a callable or model-like expert into an ExpertAdapter."""

    return ExpertAdapter(lambda X: _score_model(expert, X), name=name)


def load_expert_from_pt(path: str, device: str = "cpu") -> ExpertAdapter:
    """Load a serialized expert from a .pt file and adapt it."""

    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is required to load .pt expert files") from exc

    obj = torch.load(path, map_location=torch.device(device))
    return adapt_expert(obj, name=path)


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.generic)) or value is None


def _is_safe_container(value: Any, torch_module: Any) -> bool:
    if _is_primitive(value):
        return True
    if isinstance(value, np.ndarray):
        return True
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) for k in value.keys()) and all(
            _is_safe_container(v, torch_module) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_safe_container(v, torch_module) for v in value)
    return False


def _to_numpy_1d(value: Any, torch_module: Any) -> np.ndarray:
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr


def _post_norm(X: np.ndarray, mode: str | None) -> np.ndarray:
    if mode in (None, "none"):
        return X
    if mode == "l2":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        return (X / denom).astype(np.float32)
    raise ValueError(f"Unknown post_norm: {mode}")


def _apply_scaler(X: np.ndarray, scaler_info: Dict[str, Any] | None) -> np.ndarray:
    if scaler_info is None:
        return X

    kind = scaler_info.get("kind")
    params = scaler_info.get("params", {})
    post_norm = scaler_info.get("post_norm", "none")

    X = np.asarray(X, dtype=np.float32)

    if kind in {"minmax", "minmax_clip", "MinMaxScaler", "MinMaxClipScaler"}:
        if "scale_" in params and "min_" in params:
            X = X * params["scale_"] + params["min_"]
        elif "data_min_" in params and "data_max_" in params:
            data_min = params["data_min_"]
            data_max = params["data_max_"]
            data_range = np.where(data_max - data_min < 1e-8, 1.0, data_max - data_min)
            fr = params.get("feature_range", (0.0, 1.0))
            X = (X - data_min) / data_range
            X = X * (fr[1] - fr[0]) + fr[0]
        if params.get("clip") or kind == "minmax_clip":
            fr = params.get("feature_range", (0.0, 1.0))
            X = np.clip(X, fr[0], fr[1])
        return _post_norm(X, post_norm)

    if kind in {"robust_minmax", "robust_minmax_clip"}:
        q_low = params.get("q_low_")
        scale = params.get("scale_")
        if q_low is None and "q_low" in params:
            q_low = params.get("q_low")
        if scale is None and "q_high_" in params and q_low is not None:
            scale = params["q_high_"] - q_low
        if q_low is None or scale is None:
            raise ValueError("robust_minmax scaler params are incomplete")
        X = (X - q_low) / np.where(scale < 1e-8, 1.0, scale)
        if params.get("clip") or kind == "robust_minmax_clip":
            X = np.clip(X, 0.0, 1.0)
        return _post_norm(X, post_norm)

    if kind in {"standard_mouse", "StandardScaler", "standard"}:
        mean = params.get("mean_", 0.0)
        scale = params.get("scale_", 1.0)
        X = (X - mean) / np.where(scale == 0.0, 1.0, scale)
        return _post_norm(X, post_norm)

    if kind in {"quantile", "QuantileTransformer"}:
        quantiles = params.get("quantiles_")
        refs = params.get("references_")
        if quantiles is None or refs is None:
            raise ValueError("quantile scaler params are incomplete")
        quantiles = np.asarray(quantiles, dtype=np.float32)
        refs = np.asarray(refs, dtype=np.float32)
        X_out = np.zeros_like(X, dtype=np.float32)
        for j in range(X.shape[1]):
            X_out[:, j] = np.interp(X[:, j], quantiles[:, j], refs, left=refs[0], right=refs[-1])
        if params.get("output_distribution") == "normal":
            try:
                from numpy import erfinv  # type: ignore
            except Exception as exc:
                raise ValueError("quantile normal output requires numpy.erfinv") from exc
            X_out = np.sqrt(2.0) * erfinv(2.0 * X_out - 1.0)
        return _post_norm(X_out, post_norm)

    raise ValueError(f"Unsupported scaler kind: {kind}")


def _extract_linear_params(value: Any, torch_module: Any) -> tuple[np.ndarray, float] | None:
    weight_keys = ("weight", "weights", "coef", "coef_", "w", "W")
    bias_keys = ("bias", "bias_", "intercept", "intercept_", "b")

    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]

    if isinstance(value, dict):
        weight = next((value[k] for k in weight_keys if k in value), None)
        bias = next((value[k] for k in bias_keys if k in value), 0.0)
        if weight is None:
            return None
        w = _to_numpy_1d(weight, torch_module)
        b = float(np.asarray(bias).reshape(-1)[0]) if bias is not None else 0.0
        return w, b

    if isinstance(value, (np.ndarray, list, tuple)) or (
        torch_module is not None and isinstance(value, torch_module.Tensor)
    ):
        w = _to_numpy_1d(value, torch_module)
        return w, 0.0

    return None


def _make_linear_scorer(weights: np.ndarray, bias: float) -> ScoreFn:
    w = np.asarray(weights, dtype=float).reshape(-1)
    b = float(bias)

    def score(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != w.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, expected {w.shape[0]}")
        return X @ w + b

    return score


def _resolve_gene_idx(
    required_genes: List[str],
    global_gene_names: List[str],
    *,
    label: str,
) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(global_gene_names)}
    missing = [g for g in required_genes if g not in mapping]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            f"{label} expects {len(missing)} genes not found in current gene_names: {preview}{suffix}"
        )
    return np.asarray([mapping[g] for g in required_genes], dtype=int)


def _build_safe_scorer(payload: Any, torch_module: Any) -> ScoreFn | None:
    if isinstance(payload, dict) and payload.get("kind") == "binary_expert":
        backend = payload.get("backend")
        gene_idx = payload.get("gene_idx")
        scaler_info = payload.get("scaler")
        model = payload.get("model", {})
        temperature = float(model.get("temperature", payload.get("temperature", 1.0) or 1.0))

        def prepare_X(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            if gene_idx is not None:
                idx = np.asarray(gene_idx, dtype=int)
                X = X[:, idx]
            return _apply_scaler(X, scaler_info)

        if backend == "logreg" or model.get("kind") == "linear":
            params = _extract_linear_params(model, torch_module)
            if params is None:
                return None
            weight, bias = params

            def score(X: np.ndarray) -> np.ndarray:
                Xs = prepare_X(X)
                logits = Xs @ weight + bias
                logits = logits / max(temperature, 1e-6)
                return 1.0 / (1.0 + np.exp(-logits))

            return score

        if backend == "torch_mlp" and model.get("kind") == "mlp_binary":
            state_dict = model.get("state_dict")
            config = model.get("config", {})
            in_dim = int(model.get("in_dim") or 0)
            if state_dict is None or in_dim <= 0:
                return None

            class MLPBinary(torch_module.nn.Module):
                def __init__(self, in_dim: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.3):
                    super().__init__()
                    self.net = torch_module.nn.Sequential(
                        torch_module.nn.Linear(in_dim, hidden1),
                        torch_module.nn.GELU(),
                        torch_module.nn.Dropout(dropout),
                        torch_module.nn.Linear(hidden1, hidden2),
                        torch_module.nn.GELU(),
                        torch_module.nn.Dropout(dropout),
                        torch_module.nn.Linear(hidden2, 1),
                    )

                def forward(self, x: torch_module.Tensor) -> torch_module.Tensor:
                    return self.net(x).squeeze(-1)

            hidden1 = int(config.get("hidden1", 256))
            hidden2 = int(config.get("hidden2", 128))
            dropout = float(config.get("dropout", 0.3))
            model_obj = MLPBinary(in_dim=in_dim, hidden1=hidden1, hidden2=hidden2, dropout=dropout)
            model_obj.load_state_dict(state_dict)
            model_obj.eval()

            def score(X: np.ndarray) -> np.ndarray:
                Xs = prepare_X(X)
                with torch_module.no_grad():
                    xb = torch_module.tensor(Xs, dtype=torch_module.float32)
                    logits = model_obj(xb)
                    logits = logits / max(temperature, 1e-6)
                    p = torch_module.sigmoid(logits).detach().cpu().numpy()
                return p.astype(np.float32)

            return score

    params = _extract_linear_params(payload, torch_module)
    if params is None:
        return None
    weight, bias = params
    return _make_linear_scorer(weight, bias)


def _infer_safe_path(path: str) -> str:
    p = Path(path)
    if p.suffix == ".pt":
        candidates = [
            p.with_suffix(".safe.pt"),
            p.with_name(f"{p.stem}_state_dict.pt"),
            p.with_name(f"{p.stem}.state_dict.pt"),
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
    return path


def load_v3_leaf_experts(
    moe_pt_path: str,
    leaves: Optional[List[str]] = None,
    *,
    strict: bool = True,
) -> Dict[str, ExpertAdapter]:
    """Load leaf experts from a v3-era .pt file only if it is in a safe format."""

    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is required to load v3 MoE bundles") from exc

    resolved_path = _infer_safe_path(moe_pt_path)
    try:
        obj = torch.load(resolved_path, map_location=torch.device("cpu"))
    except Exception as exc:
        msg = str(exc)
        if isinstance(exc, (ModuleNotFoundError, AttributeError, ImportError)) or (
            "No module named" in msg or "Can't get attribute" in msg
        ):
            raise ValueError(
                "This .pt file appears to be a pickled object tied to the old v3 code layout "
                "and cannot be loaded safely. Use per-leaf checkpoints under models/ovr/... "
                "or export a v4-safe state_dict-only artifact."
            ) from exc
        raise

    if isinstance(obj, dict) and any(k in obj for k in ("subtype_models", "gate", "moe")):
        raise ValueError(
            "This .pt file looks like a full MoE bundle, not leaf experts. "
            "Use per-leaf checkpoints under models/ovr/... or export a v4-safe "
            "state_dict-only artifact."
        )

    if not _is_safe_container(obj, torch):
        raise ValueError(
            "This .pt file contains pickled Python objects and is not a safe leaf-expert "
            "artifact. Use per-leaf checkpoints under models/ovr/... or export a v4-safe "
            "state_dict-only artifact."
        )

    leaf_dict = obj
    if isinstance(obj, dict) and "experts" in obj and isinstance(obj["experts"], dict):
        leaf_dict = obj["experts"]

    if not isinstance(leaf_dict, dict):
        raise ValueError(
            "Expected a dict of leaf -> weights for experts. "
            "Use per-leaf checkpoints under models/ovr/... or export a v4-safe artifact."
        )

    experts: Dict[str, ExpertAdapter] = {}
    unsupported: List[str] = []
    for leaf, blob in leaf_dict.items():
        scorer = _build_safe_scorer(blob, torch)
        if scorer is None:
            if strict:
                unsupported.append(str(leaf))
            continue
        experts[str(leaf)] = ExpertAdapter(scorer, name=str(leaf))

    if unsupported and strict:
        raise ValueError(
            "Unsupported leaf expert format for: "
            f"{', '.join(unsupported)}. Export state_dict-only experts or use models/ovr/."
        )
    if not experts:
        raise ValueError(
            "No supported leaf experts found. Export state_dict-only experts or use models/ovr/."
        )

    if leaves is not None:
        missing = [leaf for leaf in leaves if leaf not in experts]
        if missing:
            raise ValueError(f"Missing requested leaf experts: {', '.join(missing)}")

    return experts


def load_leaf_experts_safe(
    path: str,
    leaves: Optional[List[str]] = None,
    *,
    strict: bool = True,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, ExpertAdapter]:
    """Load v4-safe leaf experts produced by save_leaf_experts_safe."""

    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is required to load leaf expert checkpoints") from exc

    obj = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(obj, dict) or obj.get("format") != "leaf_experts":
        raise ValueError("Expected a v4-safe leaf expert checkpoint (format=leaf_experts)")
    if not _is_safe_container(obj, torch):
        raise ValueError("Leaf expert checkpoint contains unsafe objects")

    leaf_dict = obj.get("experts")
    if not isinstance(leaf_dict, dict):
        raise ValueError("Leaf expert checkpoint missing experts dict")

    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    global_gene_names = gene_names or meta.get("gene_names")
    if global_gene_names is not None:
        global_gene_names = list(global_gene_names)

    experts: Dict[str, ExpertAdapter] = {}
    unsupported: List[str] = []
    for leaf, payload in leaf_dict.items():
        if isinstance(payload, dict) and global_gene_names is not None:
            payload_gene_names = payload.get("genes") or payload.get("gene_names") or meta.get("gene_names")
            gene_idx = None
            if payload_gene_names:
                gene_idx = _resolve_gene_idx(
                    list(payload_gene_names),
                    global_gene_names,
                    label=f"expert '{leaf}'",
                )
            elif "gene_idx" in payload:
                gene_idx = np.asarray(payload.get("gene_idx"), dtype=int)
                if gene_idx.size and gene_idx.max(initial=-1) >= len(global_gene_names):
                    raise ValueError(
                        f"expert '{leaf}' expects genes not found in current gene_names"
                    )
            else:
                input_dim = payload.get("input_dim")
                if input_dim is not None and int(input_dim) != len(global_gene_names):
                    raise ValueError(
                        f"missing gene metadata for expert '{leaf}' "
                        f"(input_dim={input_dim} vs gene_names length={len(global_gene_names)}). "
                        "Old artifacts without gene metadata must be regenerated."
                    )
            if gene_idx is not None:
                payload = dict(payload)
                payload["gene_idx"] = gene_idx
        else:
            gene_idx = None

        scorer = _build_safe_scorer(payload, torch)
        if scorer is None:
            if strict:
                unsupported.append(str(leaf))
            continue
        adapter_gene_idx = None
        if isinstance(payload, dict) and payload.get("kind") != "binary_expert":
            adapter_gene_idx = payload.get("gene_idx")
        experts[str(leaf)] = ExpertAdapter(
            scorer,
            name=str(leaf),
            gene_idx=adapter_gene_idx,
            gene_names=payload.get("genes") if isinstance(payload, dict) and payload.get("genes") else (
                payload.get("gene_names") if isinstance(payload, dict) else None
            ),
        )

    if unsupported and strict:
        raise ValueError(
            "Unsupported leaf expert format for: "
            f"{', '.join(unsupported)}. Export state_dict-only experts."
        )
    if not experts:
        raise ValueError("No supported leaf experts found in checkpoint.")

    if leaves is not None:
        missing = [leaf for leaf in leaves if leaf not in experts]
        if missing:
            raise ValueError(f"Missing requested leaf experts: {', '.join(missing)}")

    return experts


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _train_ovr_weighted(
    X: np.ndarray,
    y_bin: np.ndarray,
    *,
    reg: float,
    lr: float,
    epochs: int,
    seed: int,
    pos_weight: float,
    neg_weight: float,
) -> tuple[np.ndarray, float]:
    """Train a weighted logistic regressor for a single OVR leaf."""

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(n_features,))
    b = 0.0

    weights = np.where(y_bin == 1.0, pos_weight, neg_weight).astype(float)

    for _ in range(int(epochs)):
        logits = X @ W + b
        p = _sigmoid(logits)
        grad = weights * (p - y_bin) / float(n_samples)
        grad_w = X.T @ grad + reg * W
        grad_b = grad.sum()
        W -= lr * grad_w
        b -= lr * grad_b

    return W.astype(np.float32), float(b)


def train_leaf_experts_ovr(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    leaves: List[str],
    *,
    out_dir: str | Path | None = None,
    save: bool = True,
    gene_names: Optional[List[str]] = None,
    reg: float = 1e-4,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Train one-vs-rest linear experts and save a v4-safe checkpoint."""

    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    if gene_names is not None and len(gene_names) != X.shape[1]:
        raise ValueError("gene_names length must match X_train feature dimension")
    y = np.asarray(y_sub_train)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_sub_train length must match X_train rows")
    if not leaves:
        raise ValueError("leaves must be a non-empty list")

    y_str = y.astype(str)
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    experts: Dict[str, Dict[str, Any]] = {}
    for leaf in progress(leaves, total=len(leaves), desc="Leaf experts"):
        y_bin = (y_str == str(leaf)).astype(float)
        W = rng.normal(scale=0.01, size=(n_features,))
        b = 0.0

        for _ in range(int(epochs)):
            logits = X @ W + b
            p = _sigmoid(logits)
            grad = (p - y_bin) / float(n_samples)
            grad_w = X.T @ grad + reg * W
            grad_b = grad.sum()
            W -= lr * grad_w
            b -= lr * grad_b

        gene_list = list(gene_names) if gene_names else []
        experts[str(leaf)] = {
            "weight": W.astype(np.float32),
            "bias": float(b),
            "genes": gene_list,
            "gene_names": gene_list,
            "input_dim": int(len(gene_list) if gene_list else W.shape[0]),
        }

    if save:
        base = Path(__file__).resolve().parents[4]
        model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")
        out_path = model_dir / "leaf_experts.safe.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"gene_names": list(gene_names)} if gene_names else {}
        save_leaf_experts_safe(experts, str(out_path), meta=meta)
    return experts


def train_node_specific_leaf_experts(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    node_to_leaves: Dict[str, List[str]],
    node_name: str,
    *,
    out_dir: str | Path | None = None,
    save: bool = True,
    gene_names: Optional[List[str]] = None,
    reg: float = 1e-4,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Train OVR experts for leaves under a single routing node."""

    leaves = list(node_to_leaves.get(node_name, []))
    if not leaves:
        raise ValueError(f"No leaves found for node '{node_name}'")

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_sub_train).astype(str)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_sub_train length must match X_train rows")
    if gene_names is not None and len(gene_names) != X.shape[1]:
        raise ValueError("gene_names length must match X_train feature dimension")

    mask = np.isin(y, np.asarray(leaves, dtype=object))
    if not np.any(mask):
        raise ValueError(f"No training samples found for node '{node_name}'")

    X_node = X[mask]
    y_node = y[mask]

    leaf_counts = {leaf: int(np.sum(y_node == str(leaf))) for leaf in leaves}
    n_samples = int(y_node.shape[0])

    experts: Dict[str, Dict[str, Any]] = {}
    for leaf in progress(leaves, total=len(leaves), desc=f"Leaf experts: {node_name}"):
        pos = leaf_counts.get(str(leaf), 0)
        neg = n_samples - pos
        if pos <= 0 or neg <= 0:
            pos_weight = 1.0
            neg_weight = 1.0
        else:
            pos_weight = n_samples / (2.0 * pos)
            neg_weight = n_samples / (2.0 * neg)

        y_bin = (y_node == str(leaf)).astype(float)
        W, b = _train_ovr_weighted(
            X_node,
            y_bin,
            reg=reg,
            lr=lr,
            epochs=epochs,
            seed=seed,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )
        gene_list = list(gene_names) if gene_names else []
        experts[str(leaf)] = {
            "weight": W,
            "bias": float(b),
            "genes": gene_list,
            "gene_names": gene_list,
            "input_dim": int(len(gene_list) if gene_list else W.shape[0]),
        }

    if save:
        base = Path(__file__).resolve().parents[4]
        model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")
        node_dir = model_dir / "experts" / node_dir_name(node_name)
        node_dir.mkdir(parents=True, exist_ok=True)
        out_path = node_dir / "leaf_experts.safe.pt"
        meta = {"gene_names": list(gene_names)} if gene_names else {}
        save_leaf_experts_safe(experts, str(out_path), meta=meta)

    return experts


def build_leaf_score_matrix(
    X: np.ndarray,
    experts: Dict[str, Any],
    leaf_order: List[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Score all leaves and return (scores, leaf_order)."""

    if leaf_order is None:
        leaf_order = list(experts.keys())

    scores = np.zeros((X.shape[0], len(leaf_order)), dtype=float)
    for j, leaf in enumerate(leaf_order):
        if leaf not in experts:
            scores[:, j] = -np.inf
            continue
        scores[:, j] = _score_model(experts[leaf], X)
    return scores, leaf_order


def apply_candidate_mask(
    scores: np.ndarray,
    leaf_order: List[str],
    leaf_candidates: Iterable[Iterable[str]],
    fill_value: float = -np.inf,
) -> np.ndarray:
    """Mask scores for leaves not in the per-sample candidates."""

    scores = np.asarray(scores, dtype=float)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaf_order)}
    masked = np.full_like(scores, fill_value, dtype=float)

    for i, cand in enumerate(leaf_candidates):
        for leaf in cand:
            idx = leaf_to_idx.get(leaf)
            if idx is not None:
                masked[i, idx] = scores[i, idx]
    return masked


def score_leaf_experts(
    X: np.ndarray,
    experts: Dict[str, Any],
    leaf_candidates: Iterable[Iterable[str]] | Iterable[str] | None = None,
    leaf_order: List[str] | None = None,
    fill_value: float = -np.inf,
) -> Dict[str, Any]:
    """Score leaf experts and return a consistent output dict."""

    scores, leaf_order = build_leaf_score_matrix(X, experts, leaf_order=leaf_order)
    candidates = filter_leaf_candidates(leaf_candidates, leaf_order, n_samples=X.shape[0])
    masked = apply_candidate_mask(scores, leaf_order, candidates, fill_value=fill_value)
    return {"scores": masked, "leaves": leaf_order, "candidates": candidates}


def combine_node_weights_with_leaf_scores(
    leaf_score_matrix: np.ndarray,
    leaves: List[str],
    node_weights: List[Dict[str, float]] | Dict[str, float],
    node_to_leaves: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Combine node-level weights with leaf-level scores."""

    scores_2d = np.asarray(leaf_score_matrix, dtype=float)
    if scores_2d.ndim == 1:
        scores_2d = scores_2d[None, :]
    if scores_2d.ndim != 2:
        raise ValueError(f"leaf_score_matrix must be 1D or 2D, got {scores_2d.shape}")
    if scores_2d.shape[1] != len(leaves):
        raise ValueError("leaf_score_matrix columns must match len(leaves)")

    if isinstance(node_weights, dict):
        node_weights_list = [node_weights for _ in range(scores_2d.shape[0])]
    else:
        node_weights_list = list(node_weights)
    if len(node_weights_list) != scores_2d.shape[0]:
        raise ValueError("node_weights length must match number of samples")

    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}
    leaf_weight_matrix = np.zeros_like(scores_2d, dtype=float)

    for i, nw in enumerate(node_weights_list):
        for node, weight in nw.items():
            for leaf in node_to_leaves.get(node, []):
                idx = leaf_to_idx.get(leaf)
                if idx is not None:
                    leaf_weight_matrix[i, idx] += float(weight)

    masked_scores = np.where(leaf_weight_matrix > 0.0, scores_2d, 0.0)
    combined = masked_scores * leaf_weight_matrix
    combined = np.where(leaf_weight_matrix > 0.0, combined, -np.inf)
    return {"combined": combined, "leaf_weights": leaf_weight_matrix}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            loaded = load_v3_leaf_experts(path)
            print(f"Loaded {len(loaded)} leaf experts from a safe dict format.")
        except ValueError as exc:
            print(str(exc))
        raise SystemExit(0)

    from .gates import nodes_to_leaf_candidates, route_nodes, route_nodes_at_height
    from .hierarchy import height_partitions, node_to_leaves_from_subtype_paths, walk_paths

    toy_tree = {
        "node": {
            "A": ["A1", "A2"],
            "B": {"B1": ["B1a"]},
        }
    }

    subtype_to_path = walk_paths(toy_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    partitions = list(height_partitions(toy_tree, root_key="node"))
    nodes_h1 = partitions[1]

    node_scores = np.array([[2.0, 0.2], [0.1, 1.7]], dtype=float)
    selected_nodes, selected_weights = route_nodes(node_scores, nodes_h1, top_k=1)
    leaf_candidates = nodes_to_leaf_candidates(selected_nodes, node_to_leaves)

    X = np.array([[1.0, 0.5, 0.2], [0.1, 0.8, 1.2]], dtype=float)
    experts = {
        "A1": lambda x: x[:, 0],
        "A2": lambda x: x[:, 1],
        "B1a": lambda x: x[:, 2],
    }

    out = score_leaf_experts(X, experts, leaf_candidates=leaf_candidates)
    print("leaf_candidates:", leaf_candidates)
    print("leaves:", out["leaves"])
    print("scores:", out["scores"])

    node_weights = [
        {node: float(w) for node, w in zip(nodes, weights)}
        for nodes, weights in zip(selected_nodes, selected_weights)
    ]
    safe_scores = np.where(np.isfinite(out["scores"]), out["scores"], 0.0)
    combined = combine_node_weights_with_leaf_scores(
        leaf_score_matrix=safe_scores,
        leaves=out["leaves"],
        node_weights=node_weights,
        node_to_leaves=node_to_leaves,
    )
    print("leaf_weights:", combined["leaf_weights"])
    print("combined_scores:", combined["combined"])
