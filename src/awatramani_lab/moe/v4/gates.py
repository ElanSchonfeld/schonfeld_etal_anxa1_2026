"""Node routing helpers for MoE v4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .checkpointing import load_state_dict_only, save_state_dict_only
from .hierarchy import height_partitions, node_to_leaves_from_subtype_paths, walk_paths
from .progress import progress
from .torch_utils import resolve_torch_device
from .tree import child_to_leaves, node_dir_name, TreeIndex


def _as_2d(scores: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Normalize scores to 2D; return (scores_2d, was_1d)."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return scores[None, :], True
    if scores.ndim != 2:
        raise ValueError(f"scores must be 1D or 2D, got shape={scores.shape}")
    return scores, False


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax with optional temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = x / temperature
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _normalize_labels(y: np.ndarray, nodes: List[str]) -> np.ndarray:
    """Map string or integer labels to node indices."""

    if y.dtype.kind in {"U", "S", "O"}:
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx = []
        missing = set()
        for label in y.astype(str):
            if label not in node_to_idx:
                missing.add(label)
                continue
            idx.append(node_to_idx[label])
        if missing:
            raise ValueError(f"Unknown node labels in y_node_train: {sorted(missing)[:5]}")
        return np.asarray(idx, dtype=int)

    y_int = y.astype(int)
    if y_int.min(initial=0) < 0 or y_int.max(initial=-1) >= len(nodes):
        raise ValueError("y_node_train indices out of range for provided nodes")
    return y_int


def train_node_gate(
    X_train: np.ndarray,
    y_node_train: np.ndarray,
    nodes: List[str],
    *,
    reg: float = 1e-4,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train a linear softmax gate for node routing (numpy-only)."""

    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    y = np.asarray(y_node_train)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_node_train length must match X_train rows")
    if not nodes:
        raise ValueError("nodes must be a non-empty list")

    y_idx = _normalize_labels(y, nodes)
    n_samples, n_features = X.shape
    n_nodes = len(nodes)

    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(n_features, n_nodes))
    b = np.zeros((n_nodes,), dtype=float)

    y_onehot = np.zeros((n_samples, n_nodes), dtype=float)
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
        "nodes": list(nodes),
        "W": W.astype(np.float32),
        "b": b.astype(np.float32),
        "input_dim": int(n_features),
        "reg": float(reg),
        "lr": float(lr),
        "epochs": int(epochs),
    }


def _select_genes(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: List[str],
    *,
    k_genes: int,
    method: str = "f_classif",
) -> Tuple[np.ndarray, List[str]]:
    """Select top-k genes by ANOVA F-stat or mutual information."""

    try:
        if method == "f_classif":
            from sklearn.feature_selection import f_classif

            scores, _ = f_classif(X, y)
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif

            scores = mutual_info_classif(X, y)
        else:
            raise ValueError(f"Unknown gene selection method: {method}")
    except Exception as exc:  # pragma: no cover - sklearn optional
        raise ImportError("sklearn is required for gene selection") from exc

    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    k = min(int(k_genes), scores.size)
    idx = np.argsort(scores)[::-1][:k]
    genes = [gene_names[i] for i in idx]
    return idx.astype(int), genes


def _resolve_gene_idx_from_names(
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
        raise ValueError(f"{label} expects genes not found in current gene_names: {preview}{suffix}")
    return np.asarray([mapping[g] for g in required_genes], dtype=int)


def _build_binary_cnn(input_len: int, *, conv1: int = 16, conv2: int = 32, kernel: int = 5):
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


def train_node_ovr_gate(
    node_id: str,
    children: List[str],
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    gene_names: List[str],
    *,
    tree: Optional[TreeIndex] = None,
    out_dir: str | Path | None = None,
    save: bool = True,
    k_genes: int = 256,
    selector: str = "f_classif",
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 0,
    conv1: int = 16,
    conv2: int = 32,
    kernel: int = 5,
    device: str | None = None,
) -> Dict[str, Any]:
    """Train one-vs-rest child classifiers for a routing node (CNN per child)."""

    if not children:
        raise ValueError("children must be non-empty")
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_sub_train).astype(str)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_sub_train length must match X_train rows")
    if len(gene_names) != X.shape[1]:
        raise ValueError("gene_names length must match X_train feature dimension")

    if tree is None:
        node_leaves = list(children)
        child_leaves = {child: [child] for child in children}
    else:
        node_leaves = tree.get_leaves(node_id)
        child_leaves = {child: child_to_leaves(tree, child) for child in children}

    mask = np.isin(y, np.asarray(node_leaves, dtype=object))
    if not np.any(mask):
        raise ValueError(f"No training samples under node '{node_id}'")

    X_node = X[mask]
    y_node = y[mask]

    torch = None
    device_obj = None

    child_models: List[Dict[str, Any]] = []
    for child in progress(children, total=len(children), desc=f"Gate {node_id}"):
        leaves = child_leaves.get(child, [])
        if not leaves:
            raise ValueError(f"No leaves found for child '{child}' under '{node_id}'")

        y_bin = np.isin(y_node, np.asarray(leaves, dtype=object)).astype(int)
        pos = int(y_bin.sum())
        neg = int(y_bin.size - pos)
        if pos == 0 or neg == 0:
            const_prob = 0.0 if pos == 0 else 1.0
            child_models.append(
                {
                    "child": str(child),
                    "state_dict": {},
                    "gene_indices": np.array([], dtype=int),
                    "gene_names": [],
                    "model": {
                        "kind": "constant",
                        "prob": float(const_prob),
                        "reason": "no_pos" if pos == 0 else "no_neg",
                    },
                }
            )
            continue

        if torch is None:
            try:
                import torch  # type: ignore
            except Exception as exc:  # pragma: no cover - torch optional
                raise ImportError("torch is required for CNN gates") from exc
            torch.manual_seed(seed)
            np.random.seed(seed)
            device_obj = resolve_torch_device(device)
        if device_obj is None:
            device_obj = resolve_torch_device(device)

        gene_idx, gene_list = _select_genes(
            X_node,
            y_bin,
            gene_names,
            k_genes=k_genes,
            method=selector,
        )
        X_child = X_node[:, gene_idx]

        model = _build_binary_cnn(input_len=X_child.shape[1], conv1=conv1, conv2=conv2, kernel=kernel).to(device_obj)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device_obj)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tensor = torch.from_numpy(X_child).float().unsqueeze(1).to(device_obj)
        y_tensor = torch.from_numpy(y_bin).float().view(-1, 1).to(device_obj)

        n_samples = X_tensor.shape[0]
        if batch_size >= n_samples:
            batch_size = n_samples

        for _ in range(int(epochs)):
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

        child_models.append(
            {
                "child": str(child),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "gene_indices": gene_idx.astype(int),
                "gene_names": gene_list,
                "model": {
                    "kind": "cnn_1d",
                    "conv1": int(conv1),
                    "conv2": int(conv2),
                    "kernel": int(kernel),
                    "input_len": int(X_child.shape[1]),
                },
            }
        )

    gate_bundle = {
        "kind": "ovr_cnn",
        "node_id": str(node_id),
        "children": list(children),
        "nodes": list(children),
        "child_models": child_models,
        "k_genes": int(k_genes),
        "selector": str(selector),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "input_dim": int(X.shape[1]),
        "gene_names": list(gene_names),
        "note": "CNN uses fixed gene ordering saved per child.",
    }

    if save:
        base = Path(__file__).resolve().parents[4]
        model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")
        gate_dir = model_dir / "gates" / node_dir_name(node_id)
        gate_dir.mkdir(parents=True, exist_ok=True)
        save_state_dict_only(gate_bundle, str(gate_dir / "gate.safe.pt"))

    return gate_bundle


def load_node_ovr_gate(path: str) -> Dict[str, Any]:
    """Load a node OVR CNN gate bundle saved with train_node_ovr_gate."""

    bundle = load_state_dict_only(path)
    if bundle.get("kind") != "ovr_cnn":
        raise ValueError("Not an OVR CNN gate bundle")
    return bundle


def predict_node_child_probs(
    gate_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    as_logits: bool = False,
    gene_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Predict per-child probabilities or logits for an OVR CNN gate."""

    if gate_bundle.get("kind") != "ovr_cnn":
        raise ValueError("gate_bundle must be kind='ovr_cnn'")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    child_models = gate_bundle.get("child_models", [])
    if not child_models:
        raise ValueError("gate_bundle missing child_models")

    preds: List[np.ndarray] = []
    global_gene_names = list(gene_names) if gene_names is not None else None
    torch = None
    for child_entry in child_models:
        state = child_entry.get("state_dict", {})
        model_meta = child_entry.get("model", {})
        model_kind = model_meta.get("kind")
        if model_kind == "constant":
            prob = float(model_meta.get("prob", 0.0))
            if as_logits:
                prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
                preds.append(np.full((X.shape[0],), np.log(prob / (1.0 - prob)), dtype=float))
            else:
                preds.append(np.full((X.shape[0],), prob, dtype=float))
            continue

        gene_idx = np.asarray(child_entry.get("gene_idx", []), dtype=int)
        if gene_idx.size == 0:
            gene_idx = np.asarray(child_entry.get("gene_indices", []), dtype=int)
        if global_gene_names is not None:
            child_genes = child_entry.get("gene_names") or []
            if child_genes:
                gene_idx = _resolve_gene_idx_from_names(
                    list(child_genes),
                    global_gene_names,
                    label=f"gate child '{child_entry.get('child', '')}'",
                )
                child_entry["gene_idx"] = gene_idx
        if gene_idx.size == 0:
            raise ValueError("child gate missing gene indices")

        if torch is None:
            try:
                import torch  # type: ignore
            except Exception as exc:  # pragma: no cover - torch optional
                raise ImportError("torch is required for CNN gates") from exc

        model = _build_binary_cnn(
            input_len=int(model_meta.get("input_len", len(gene_idx))),
            conv1=int(model_meta.get("conv1", 16)),
            conv2=int(model_meta.get("conv2", 32)),
            kernel=int(model_meta.get("kernel", 5)),
        )
        model.load_state_dict(state)
        model.eval()

        X_child = X[:, gene_idx]
        X_tensor = torch.from_numpy(X_child).float().unsqueeze(1)
        with torch.no_grad():
            logits = model(X_tensor).cpu().numpy().reshape(-1)
        if as_logits:
            preds.append(logits)
        else:
            preds.append(1.0 / (1.0 + np.exp(-logits)))

    return np.stack(preds, axis=1)


def train_binary_gate(
    X_train: np.ndarray,
    y_binary: np.ndarray,
    *,
    node_name: str | None = None,
    label_names: Tuple[str, str] = ("neg", "pos"),
    nodes: List[str] | None = None,
    gene_names: Optional[List[str]] = None,
    reg: float = 1e-4,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train a binary logistic gate for a named routing node."""

    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    if gene_names is not None and len(gene_names) != X.shape[1]:
        raise ValueError("gene_names length must match X_train feature dimension")
    y = np.asarray(y_binary).astype(float).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y_binary length must match X_train rows")

    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(X.shape[1],))
    b = 0.0

    for _ in range(int(epochs)):
        logits = X @ W + b
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = (p - y) / float(X.shape[0])
        grad_w = X.T @ grad + reg * W
        grad_b = grad.sum()
        W -= lr * grad_w
        b -= lr * grad_b

    if nodes is None:
        if node_name is None:
            nodes = [label_names[0], label_names[1]]
        else:
            nodes = [f"{node_name}|{label_names[0]}", f"{node_name}|{label_names[1]}"]

    return {
        "kind": "binary_logreg",
        "node_name": node_name,
        "label_names": list(label_names),
        "nodes": list(nodes),
        "W": W.astype(np.float32),
        "b": float(b),
        "input_dim": int(X.shape[1]),
        "gene_names": list(gene_names) if gene_names else [],
        "reg": float(reg),
        "lr": float(lr),
        "epochs": int(epochs),
    }


def predict_binary_gate(
    gate_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    as_prob: bool = False,
    gene_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Predict logits or probabilities for a binary gate."""

    W = np.asarray(gate_bundle.get("W"))
    b = float(gate_bundle.get("b", 0.0))
    if W.ndim != 1:
        raise ValueError("binary gate expects 1D weight vector")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if gene_names is not None and gate_bundle.get("gene_names"):
        gate_genes = list(gate_bundle.get("gene_names") or [])
        if gate_genes:
            gene_idx = _resolve_gene_idx_from_names(gate_genes, list(gene_names), label="binary gate")
            X = X[:, gene_idx]
    if X.shape[1] != W.shape[0]:
        raise ValueError("X feature dimension does not match gate weights")

    logits = X @ W + b
    if as_prob:
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - p, p], axis=1)
    return np.stack([-logits, logits], axis=1)


def predict_node_scores(
    gate_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    gene_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute node logits from a trained gate bundle."""

    W = np.asarray(gate_bundle.get("W"))
    b = np.asarray(gate_bundle.get("b"))
    if W.ndim != 2 or b.ndim != 1:
        raise ValueError("gate_bundle must contain 2D W and 1D b")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if gene_names is not None and gate_bundle.get("gene_names"):
        gate_genes = list(gate_bundle.get("gene_names") or [])
        if gate_genes:
            gene_idx = _resolve_gene_idx_from_names(gate_genes, list(gene_names), label="node gate")
            X = X[:, gene_idx]
    if X.shape[1] != W.shape[0]:
        raise ValueError("X feature dimension does not match gate weights")

    return X @ W + b


def save_node_gate(gate_bundle: Dict[str, Any], path: str) -> None:
    """Save a node gate bundle using safe checkpointing utilities."""

    state = {
        "W": np.asarray(gate_bundle["W"]),
        "b": np.asarray(gate_bundle["b"]),
    }
    meta = {
        "kind": str(gate_bundle.get("kind", "linear_softmax")),
        "nodes": list(gate_bundle.get("nodes", [])),
        "input_dim": int(gate_bundle.get("input_dim", state["W"].shape[0])),
        "node_name": gate_bundle.get("node_name"),
        "label_names": gate_bundle.get("label_names"),
        "height": gate_bundle.get("height"),
    }
    save_state_dict_only(state, path, meta=meta)


def save_binary_gate(gate_bundle: Dict[str, Any], path: str) -> None:
    """Save a binary gate bundle using safe checkpointing utilities."""

    state = {
        "W": np.asarray(gate_bundle["W"]),
        "b": np.asarray(gate_bundle.get("b", 0.0)),
    }
    meta = {
        "kind": str(gate_bundle.get("kind", "binary_logreg")),
        "nodes": list(gate_bundle.get("nodes", [])),
        "label_names": list(gate_bundle.get("label_names", ["neg", "pos"])),
        "node_name": gate_bundle.get("node_name"),
        "input_dim": int(gate_bundle.get("input_dim", state["W"].shape[0])),
    }
    save_state_dict_only(state, path, meta=meta)


def load_node_gate(path: str) -> Dict[str, Any]:
    """Load a node gate bundle saved with save_node_gate."""

    bundle = load_state_dict_only(path)
    state = bundle.get("state_dict", {})
    meta = bundle.get("meta", {})

    nodes = meta.get("nodes")
    if not nodes:
        raise ValueError("Missing nodes in gate checkpoint metadata")

    W = np.asarray(state.get("W"))
    b = np.asarray(state.get("b"))
    if W.ndim != 2 or b.ndim != 1:
        raise ValueError("Gate checkpoint missing W/b parameters")
    if W.shape[1] != len(nodes):
        raise ValueError("Gate checkpoint nodes length does not match W shape")

    input_dim = int(meta.get("input_dim", W.shape[0]))
    if W.shape[0] != input_dim:
        raise ValueError("Gate checkpoint input_dim does not match W shape")

    return {
        "kind": str(meta.get("kind", "linear_softmax")),
        "nodes": list(nodes),
        "W": W,
        "b": b,
        "input_dim": input_dim,
        "node_name": meta.get("node_name"),
        "label_names": meta.get("label_names"),
        "height": meta.get("height"),
    }


def load_binary_gate(path: str) -> Dict[str, Any]:
    """Load a binary gate bundle saved with save_binary_gate."""

    bundle = load_state_dict_only(path)
    state = bundle.get("state_dict", {})
    meta = bundle.get("meta", {})

    W = np.asarray(state.get("W"))
    b = float(np.asarray(state.get("b", 0.0)).reshape(-1)[0])
    if W.ndim != 1:
        raise ValueError("Binary gate checkpoint missing 1D W parameter")

    nodes = meta.get("nodes", [])
    label_names = meta.get("label_names", ["neg", "pos"])

    return {
        "kind": str(meta.get("kind", "binary_logreg")),
        "node_name": meta.get("node_name"),
        "label_names": list(label_names),
        "nodes": list(nodes),
        "W": W,
        "b": b,
        "input_dim": int(meta.get("input_dim", W.shape[0])),
    }


def save_gate_bundle(gate_bundle: Dict[str, Any], path: str) -> None:
    """Save any supported gate bundle."""

    kind = gate_bundle.get("kind")
    if kind == "binary_logreg":
        save_binary_gate(gate_bundle, path)
    elif kind == "ovr_cnn":
        save_state_dict_only(gate_bundle, path)
    else:
        save_node_gate(gate_bundle, path)


def load_gate_bundle(path: str) -> Dict[str, Any]:
    """Load any supported gate bundle."""

    bundle = load_state_dict_only(path)
    meta = bundle.get("meta", {})
    kind = meta.get("kind", "linear_softmax")
    state = bundle.get("state_dict", {})
    if isinstance(state, dict) and state.get("kind") == "ovr_cnn":
        return state
    if kind == "binary_logreg":
        return load_binary_gate(path)
    return load_node_gate(path)


def route_nodes(
    scores: np.ndarray,
    nodes: List[str],
    top_k: int = 1,
    normalize: bool = True,
    temperature: float = 1.0,
) -> Tuple[List[List[str]], np.ndarray]:
    """Select top-k nodes and return their weights."""

    scores_2d, _ = _as_2d(scores)
    if scores_2d.shape[1] != len(nodes):
        raise ValueError("scores columns must match len(nodes)")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    weights = _softmax(scores_2d, temperature=temperature) if normalize else scores_2d
    k = min(top_k, weights.shape[1])
    idx = np.argsort(weights, axis=1)[:, ::-1][:, :k]

    selected_nodes = [[nodes[i] for i in row] for row in idx]
    selected_weights = np.take_along_axis(weights, idx, axis=1)
    return selected_nodes, selected_weights


def route_nodes_at_height(
    tree: Dict[str, Any],
    height: int,
    scores: np.ndarray,
    root_key: str = "node",
    top_k: int = 1,
    normalize: bool = True,
    temperature: float = 1.0,
) -> Tuple[List[List[str]], np.ndarray, List[str]]:
    """Route using nodes from a specific height partition."""

    partitions = list(height_partitions(tree, root_key=root_key))
    if height < 0 or height >= len(partitions):
        raise ValueError(f"height out of range: {height}")
    nodes = partitions[height]
    selected_nodes, selected_weights = route_nodes(
        scores=scores,
        nodes=nodes,
        top_k=top_k,
        normalize=normalize,
        temperature=temperature,
    )
    return selected_nodes, selected_weights, nodes


def nodes_to_leaf_candidates(
    selected_nodes: Iterable[Iterable[str]],
    node_to_leaves: Dict[str, List[str]],
    dedupe: bool = True,
) -> List[List[str]]:
    """Map selected nodes to leaf subtype candidates."""

    out: List[List[str]] = []
    for nodes in selected_nodes:
        leaves: List[str] = []
        for node in nodes:
            leaves.extend(node_to_leaves.get(node, []))
        if dedupe:
            seen = set()
            leaves = [s for s in leaves if not (s in seen or seen.add(s))]
        out.append(leaves)
    return out


def node_weights_to_leaf_weights(
    selected_nodes: Iterable[Iterable[str]],
    selected_weights: np.ndarray,
    node_to_leaves: Dict[str, List[str]],
) -> List[Dict[str, float]]:
    """Distribute node weights to leaf candidates (sum over selected nodes)."""

    weights_2d, _ = _as_2d(selected_weights)
    out: List[Dict[str, float]] = []
    for i, nodes in enumerate(selected_nodes):
        leaf_weights: Dict[str, float] = {}
        for j, node in enumerate(nodes):
            w = float(weights_2d[i, j])
            for leaf in node_to_leaves.get(node, []):
                leaf_weights[leaf] = leaf_weights.get(leaf, 0.0) + w
        out.append(leaf_weights)
    return out


if __name__ == "__main__":
    from collections import Counter

    rng = np.random.default_rng(0)
    n, d, k = 200, 4, 3
    true_W = rng.normal(size=(d, k))
    X_demo = rng.normal(size=(n, d))
    logits = X_demo @ true_W
    y_demo = logits.argmax(axis=1)
    nodes_demo = [f"node|N{i}" for i in range(k)]

    gate = train_node_gate(X_demo, y_demo, nodes_demo, reg=1e-3, lr=0.2, epochs=200, seed=0)
    scores = predict_node_scores(gate, X_demo)
    preds = scores.argmax(axis=1)
    acc = (preds == y_demo).mean()
    counts = Counter(preds.tolist())
    print("gate scores shape:", scores.shape)
    print("gate scores min/max:", float(scores.min()), float(scores.max()))
    print("node counts:", dict(counts))
    print("first 5 preds:", preds[:5].tolist())
    print("demo accuracy:", float(acc))

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = f"{tmpdir}/node_gate.safe.pt"
            save_node_gate(gate, ckpt)
            gate2 = load_node_gate(ckpt)
            scores2 = predict_node_scores(gate2, X_demo)
            print("reload delta:", float(np.abs(scores - scores2).max()))
    except Exception as exc:
        print("checkpoint demo skipped:", exc)

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

    scores = np.array([[2.0, 0.2], [0.1, 1.7]], dtype=float)
    selected_nodes, selected_weights = route_nodes(scores, nodes_h1, top_k=1)
    leaf_candidates = nodes_to_leaf_candidates(selected_nodes, node_to_leaves)
    leaf_weights = node_weights_to_leaf_weights(selected_nodes, selected_weights, node_to_leaves)

    flat_nodes = [sel[0] for sel in selected_nodes]
    counts = Counter(flat_nodes)
    print("route nodes_h1:", nodes_h1)
    print("route weights shape:", selected_weights.shape)
    print("route counts:", dict(counts))
    print("first 5 nodes:", flat_nodes[:5])
