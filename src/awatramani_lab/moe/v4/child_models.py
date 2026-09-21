"""Child model backends for hierarchical routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .checkpointing import load_state_dict_only, save_state_dict_only
from .torch_utils import resolve_torch_device


_DUMMY_LOGIT = -1e9


@dataclass
class DummyModel:
    """Placeholder model used when a child classifier is missing on disk."""

    logit_value: float = _DUMMY_LOGIT

    def predict_child_logits(self, X: np.ndarray, *, gene_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        X = np.asarray(X)
        n = int(X.shape[0])
        return np.full((n,), float(self.logit_value), dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _subset_by_idx(X: np.ndarray, gene_idx: Optional[Sequence[int]], *, label: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if gene_idx is None:
        return X
    idx = np.asarray(list(gene_idx), dtype=int)
    if idx.size == 0:
        return X
    if idx.max(initial=-1) >= X.shape[1]:
        raise ValueError(f"{label} gene_idx out of range for X with shape {X.shape}")
    return X[:, idx]


def _train_logreg_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    reg: float,
    epochs: int,
    seed: int,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(X.shape[1],))
    b = 0.0

    pos = float(y.sum())
    neg = float(y.size - pos)
    pos_weight = neg / max(pos, 1.0)
    weights = np.where(y > 0, pos_weight, 1.0).astype(float)

    for _ in range(int(epochs)):
        logits = X @ W + b
        p = _sigmoid(logits)
        grad = (p - y) * weights / float(X.shape[0])
        grad_w = X.T @ grad + reg * W
        grad_b = grad.sum()
        W -= lr * grad_w
        b -= lr * grad_b

    return {"weights": W.astype(np.float32), "bias": float(b)}


def _build_ffn(input_dim: int, hidden_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    class BinaryFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(1)

    return BinaryFFN()


def _build_cnn(input_len: int, conv1: int, conv2: int, kernel: int):
    import torch
    import torch.nn as nn

    class BinaryCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(1, conv1, kernel, padding=pad)
            self.conv2 = nn.Conv1d(conv1, conv2, kernel, padding=pad)
            self.relu = nn.ReLU()
            self.out = nn.Linear(conv2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.mean(dim=2)
            return self.out(x).squeeze(1)

    return BinaryCNN()


def train_child_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    gene_idx: Optional[Sequence[int]],
    seed: int,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a binary child classifier and return a model bundle."""

    model_type = str(model_type).lower()
    hyperparams = hyperparams or {}
    X = _subset_by_idx(X, gene_idx, label="train_child_model")
    y = np.asarray(y, dtype=int).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length")

    if model_type == "logreg":
        fix_iter_scale = bool(hyperparams.get("fix_iter_scale", False))
        max_iter = int(hyperparams.get("max_iter", 2000 if fix_iter_scale else 500))

        scaler_meta = None
        X_fit = X
        if fix_iter_scale:
            mu = X.mean(axis=0, dtype=np.float64)
            sigma = X.std(axis=0, dtype=np.float64)
            sigma = np.where(sigma > 0, sigma, 1.0)
            X_fit = (X - mu) / sigma
            scaler_meta = {
                "kind": "standard",
                "mean": np.asarray(mu, dtype=np.float32),
                "scale": np.asarray(sigma, dtype=np.float32),
            }

        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore

            clf = LogisticRegression(
                penalty="l2",
                C=float(hyperparams.get("C", 1.0)),
                solver=str(hyperparams.get("solver", "lbfgs")),
                max_iter=max_iter,
                class_weight=hyperparams.get("class_weight", "balanced"),
                random_state=seed,
            )
            clf.fit(X_fit, y)
            weights = np.asarray(clf.coef_, dtype=float).reshape(-1)
            bias = float(np.asarray(clf.intercept_, dtype=float).reshape(-1)[0])

        except Exception:
            params = _train_logreg_numpy(
                X_fit,
                y,
                lr=float(hyperparams.get("lr", 0.1)),
                reg=float(hyperparams.get("reg", 1e-4)),
                epochs=int(hyperparams.get("epochs", 200)),
                seed=seed,
            )
            weights = params["weights"]
            bias = params["bias"]

        out = {
            "kind": "logreg",
            "weights": np.asarray(weights, dtype=np.float32),
            "bias": float(bias),
            "gene_idx": list(gene_idx) if gene_idx is not None else [],
            "input_dim": int(X.shape[1]),
        }
        if scaler_meta is not None:
            out["scaler"] = scaler_meta
            out["max_iter"] = max_iter
        return out

    if model_type == "ffn":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("torch is required for ffn child models") from exc
        torch.manual_seed(seed)
        np.random.seed(seed)

        hidden_dim = int(hyperparams.get("hidden_dim", 64))
        dropout = float(hyperparams.get("dropout", 0.1))
        lr = float(hyperparams.get("lr", 1e-3))
        epochs = int(hyperparams.get("epochs", 40))
        batch_size = int(hyperparams.get("batch_size", 256))

        device = resolve_torch_device(hyperparams.get("device"))
        model = _build_ffn(X.shape[1], hidden_dim, dropout).to(device)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        pos = float(y.sum())
        neg = float(y.size - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)
        n_samples = X_tensor.shape[0]
        batch_size = min(batch_size, n_samples)

        for _ in range(epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]
                logits = model(xb).view(-1, 1)
                loss = loss_fn(logits, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()

        return {
            "kind": "ffn",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden_dim": hidden_dim,
            "dropout": float(dropout),
            "gene_idx": list(gene_idx) if gene_idx is not None else [],
            "input_dim": int(X.shape[1]),
        }

    if model_type == "cnn":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("torch is required for cnn child models") from exc
        torch.manual_seed(seed)
        np.random.seed(seed)

        conv1 = int(hyperparams.get("conv1", 16))
        conv2 = int(hyperparams.get("conv2", 32))
        kernel = int(hyperparams.get("kernel", 5))
        lr = float(hyperparams.get("lr", 1e-3))
        epochs = int(hyperparams.get("epochs", 40))
        batch_size = int(hyperparams.get("batch_size", 256))

        device = resolve_torch_device(hyperparams.get("device"))
        model = _build_cnn(X.shape[1], conv1, conv2, kernel).to(device)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        pos = float(y.sum())
        neg = float(y.size - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tensor = torch.from_numpy(X).float().unsqueeze(1).to(device)
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)
        n_samples = X_tensor.shape[0]
        batch_size = min(batch_size, n_samples)

        for _ in range(epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]
                logits = model(xb).view(-1, 1)
                loss = loss_fn(logits, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()

        return {
            "kind": "cnn",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "conv1": conv1,
            "conv2": conv2,
            "kernel": kernel,
            "input_len": int(X.shape[1]),
            "gene_idx": list(gene_idx) if gene_idx is not None else [],
            "input_dim": int(X.shape[1]),
        }

    if model_type == "xgboost":
        try:
            import xgboost  # type: ignore
        except Exception as exc:
            raise ImportError("xgboost is required for xgboost child models") from exc
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": int(hyperparams.get("max_depth", 4)),
            "eta": float(hyperparams.get("eta", 0.1)),
            "subsample": float(hyperparams.get("subsample", 0.9)),
            "colsample_bytree": float(hyperparams.get("colsample_bytree", 0.9)),
            "seed": int(seed),
        }
        dtrain = xgboost.DMatrix(X, label=y)
        booster = xgboost.train(params, dtrain, num_boost_round=int(hyperparams.get("rounds", 200)))
        return {
            "kind": "xgboost",
            "booster": booster,
            "gene_idx": list(gene_idx) if gene_idx is not None else [],
            "input_dim": int(X.shape[1]),
        }

    raise ValueError(f"Unsupported model_type: {model_type}")

def _apply_scaler_if_present(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    scaler = model_bundle.get("scaler")
    if not scaler:
        return X
    if scaler.get("kind") == "standard":
        mu = np.asarray(scaler["mean"], dtype=np.float32)
        sigma = np.asarray(scaler["scale"], dtype=np.float32)
        return (X - mu) / sigma
    return X

def predict_child_logits(
    model_bundle: Any,
    X: np.ndarray,
    *,
    gene_idx: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Predict logits for a child model bundle."""

    if isinstance(model_bundle, DummyModel):
        return model_bundle.predict_child_logits(X, gene_idx=gene_idx)
    if isinstance(model_bundle, dict) and model_bundle.get("kind") == "dummy":
        n = int(np.asarray(X).shape[0])
        return np.full((n,), float(model_bundle.get("logit_value", _DUMMY_LOGIT)), dtype=float)

    if not isinstance(model_bundle, dict):
        raise TypeError("model_bundle must be a dict or DummyModel")

    kind = str(model_bundle.get("kind", "")).lower()
    model_gene_idx = model_bundle.get("gene_idx") or []
    use_idx = list(gene_idx) if gene_idx is not None else model_gene_idx
    X_use = _subset_by_idx(X, use_idx, label=f"child model ({kind})")

    if kind == "logreg":
        X_use = _apply_scaler_if_present(model_bundle, X_use)
        weights = np.asarray(model_bundle.get("weights"), dtype=float).reshape(-1)
        bias = float(np.asarray(model_bundle.get("bias", 0.0)).reshape(-1)[0])
        if X_use.shape[1] != weights.shape[0]:
            raise ValueError("logreg weights do not match feature dimension")
        return X_use @ weights + bias

    if kind == "ffn":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("torch is required for ffn child models") from exc
        hidden_dim = int(model_bundle.get("hidden_dim", 64))
        dropout = float(model_bundle.get("dropout", 0.1))
        model = _build_ffn(X_use.shape[1], hidden_dim, dropout)
        model.load_state_dict(model_bundle.get("state_dict", {}))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_use).float()).cpu().numpy().reshape(-1)
        return logits

    if kind == "cnn":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("torch is required for cnn child models") from exc
        conv1 = int(model_bundle.get("conv1", 16))
        conv2 = int(model_bundle.get("conv2", 32))
        kernel = int(model_bundle.get("kernel", 5))
        model = _build_cnn(X_use.shape[1], conv1, conv2, kernel)
        model.load_state_dict(model_bundle.get("state_dict", {}))
        model.eval()
        X_tensor = torch.from_numpy(X_use).float().unsqueeze(1)
        with torch.no_grad():
            logits = model(X_tensor).cpu().numpy().reshape(-1)
        return logits

    if kind == "xgboost":
        try:
            import xgboost  # type: ignore
        except Exception as exc:
            raise ImportError("xgboost is required for xgboost child models") from exc
        booster = model_bundle.get("booster")
        if booster is None:
            raise ValueError("xgboost model bundle missing booster")
        dtest = xgboost.DMatrix(X_use)
        probs = booster.predict(dtest)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.log(probs / (1.0 - probs))

    raise ValueError(f"Unsupported model bundle kind: {kind}")


def save_child_model(model_bundle: Dict[str, Any], path: str | Path) -> None:
    """Save a child model bundle in a safe checkpoint."""

    path = str(path)
    payload = dict(model_bundle)
    if payload.get("kind") == "xgboost":
        booster = payload.get("booster")
        if booster is None:
            raise ValueError("xgboost model bundle missing booster")
        model_path = Path(path)
        booster_path = model_path.with_suffix(".json")
        booster.save_model(str(booster_path))
        payload["booster_path"] = str(booster_path)
        payload.pop("booster", None)
    save_state_dict_only(payload, path, meta={"kind": model_bundle.get("kind", "")})


def load_child_model(path: str | Path) -> Any:
    """Load a child model bundle; return DummyModel if missing."""

    path = Path(path)
    if not path.exists():
        return DummyModel()
    bundle = load_state_dict_only(str(path))
    state = bundle.get("state_dict", {})
    if isinstance(state, dict) and state.get("kind") is None and bundle.get("meta", {}).get("kind"):
        state["kind"] = bundle.get("meta", {}).get("kind")
    if isinstance(state, dict) and state.get("kind") == "xgboost":
        booster_path = state.get("booster_path")
        if booster_path:
            try:
                import xgboost  # type: ignore
            except Exception as exc:
                raise ImportError("xgboost is required to load xgboost child models") from exc
            booster = xgboost.Booster()
            booster.load_model(str(booster_path))
            state["booster"] = booster
    return state
