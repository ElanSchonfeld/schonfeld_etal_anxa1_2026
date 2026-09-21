"""ROC-based hierarchical router for dendrogram-guided MoE (Option 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import numpy as np

from .child_models import DummyModel, load_child_model, predict_child_logits, save_child_model, train_child_model
from .hierarchy import get_node, height_partitions, node_to_leaves_from_subtype_paths, walk_paths
from .roc_markers import select_top_markers, seurat_roc_markers
from .routing_spec import EXCLUDED_LEAVES
from .stacker import (
    build_stacker_features,
    load_stacker_bundle,
    predict_leaf_stacker,
    save_stacker_bundle,
    train_leaf_stacker,
)
from .progress import progress


def _safe_id(value: str) -> str:
    return str(value).replace("|", "__").replace(":", "_").replace("/", "_")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _prune_tree_to_leaves(tree: Dict[str, Any], allowed_leaves: Iterable[str], *, root_key: str) -> Dict[str, Any]:
    allowed = set(allowed_leaves)

    def _prune(sub: Any) -> Any:
        if isinstance(sub, list):
            return [leaf for leaf in sub if leaf in allowed]
        if isinstance(sub, dict):
            out: Dict[str, Any] = {}
            for key, child in sub.items():
                pruned = _prune(child)
                if isinstance(pruned, list) and not pruned:
                    continue
                if isinstance(pruned, dict) and not pruned:
                    continue
                if pruned is None:
                    continue
                out[key] = pruned
            return out
        return None

    if root_key not in tree:
        raise KeyError(f"root_key '{root_key}' not found in tree")
    pruned = _prune(tree[root_key])
    return {root_key: pruned or {}}


def _iter_node_paths(subtree: Any, path: List[str]):
    yield path, subtree
    if isinstance(subtree, dict):
        for key, child in subtree.items():
            yield from _iter_node_paths(child, path + [key])


def train_router_at_height(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    tree: Dict[str, Any],
    *,
    root_key: str,
    height_train: int,
    gene_names: List[str],
    model_type: str = "logreg",
    topk_markers: int = 50,
    min_pct: float = 0.1,
    min_diff_pct: Optional[float] = None,
    min_samples_per_class: int = 1,
    seed: int = 0,
    model_hyperparams: Optional[Dict[str, Any]] = None,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train per-child classifiers for all nodes at a single height."""

    excluded = list(excluded_leaves or EXCLUDED_LEAVES)
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_sub_train).astype(str)

    keep_mask = ~np.isin(y, np.asarray(excluded, dtype=object))
    X = X[keep_mask]
    y = y[keep_mask]

    subtype_to_path = walk_paths(tree)
    kept_leaves = [leaf for leaf in subtype_to_path.keys() if leaf not in excluded]
    pruned_tree = _prune_tree_to_leaves(tree, kept_leaves, root_key=root_key)
    subtype_to_path = walk_paths(pruned_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)

    height_nodes = list(height_partitions(pruned_tree, root_key=root_key))
    if height_train <= 0 or height_train > len(height_nodes):
        raise ValueError(f"height_train={height_train} out of range (max={len(height_nodes)})")

    parent_idx = min(height_train - 1, len(height_nodes) - 1)
    parent_nodes = height_nodes[parent_idx]
    log: Dict[str, Any] = {"height": height_train, "nodes": {}}
    gates: Dict[str, Any] = {}

    base_dir = Path(out_dir) if out_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for path, sub in _iter_node_paths(pruned_tree[root_key], [root_key]):
        node_path = "|".join(path)
        if node_path not in parent_nodes:
            continue

        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_paths = [path + [child] for child in child_keys]
            child_names = ["|".join(p) for p in child_paths]
        elif isinstance(sub, list):
            child_keys = list(sub)
            child_paths = []
            child_names = list(child_keys)
        else:
            continue

        if len(child_names) < 2:
            continue

        parent_leaves = node_to_leaves.get(node_path, [])
        if not parent_leaves:
            continue
        parent_mask = np.isin(y, np.asarray(parent_leaves, dtype=object))
        X_parent = X[parent_mask]
        y_parent = y[parent_mask]

        node_log: Dict[str, Any] = {"children": list(child_names), "child_status": {}}
        child_models: Dict[str, Any] = {}
        child_priors: Dict[str, float] = {}
        child_model_paths: Dict[str, str] = {}

        child_counts: Dict[str, int] = {}
        for child_name, child_path in zip(child_names, child_paths or child_names):
            if child_paths:
                leaves = node_to_leaves.get("|".join(child_path), [])
            else:
                leaves = [child_path]
            child_counts[child_name] = int(np.sum(np.isin(y_parent, np.asarray(leaves, dtype=object))))

        total = float(sum(child_counts.values()))
        for child_name, count in child_counts.items():
            child_priors[child_name] = (count / total) if total > 0 else 1.0 / float(len(child_names))

        for child_name, child_path in zip(child_names, child_paths or child_names):
            if child_paths:
                leaves = node_to_leaves.get("|".join(child_path), [])
            else:
                leaves = [child_path]

            y_bin = np.isin(y_parent, np.asarray(leaves, dtype=object)).astype(int)
            pos = int(y_bin.sum())
            neg = int(y_bin.size - pos)

            parent_dir = _safe_id(node_path)
            child_id = _safe_id(child_name)
            marker_dir = None
            model_dir = None
            if base_dir is not None:
                marker_dir = base_dir / "markers" / parent_dir
                model_dir = base_dir / "child_models" / parent_dir
                marker_dir.mkdir(parents=True, exist_ok=True)
                model_dir.mkdir(parents=True, exist_ok=True)

            if pos < min_samples_per_class or neg < min_samples_per_class:
                node_log["child_status"][child_name] = {
                    "status": "skipped",
                    "reason": "insufficient_samples",
                    "n_pos": pos,
                    "n_neg": neg,
                    "n_genes": 0,
                }
                child_models[child_name] = DummyModel()
                if marker_dir is not None:
                    csv_path = marker_dir / f"{child_id}_roc_markers.csv"
                    _write_marker_csv(csv_path, [])
                    _write_gene_list(marker_dir / f"{child_id}_genes.json", [])
                continue

            markers = seurat_roc_markers(
                X_parent,
                y_bin,
                gene_names,
                min_pct=min_pct,
                min_diff_pct=min_diff_pct,
            )
            selected = select_top_markers(
                markers,
                topk=topk_markers,
                min_pct=min_pct,
                min_diff_pct=min_diff_pct,
            )

            if marker_dir is not None:
                csv_path = marker_dir / f"{child_id}_roc_markers.csv"
                markers.to_csv(csv_path, index=False)
                _write_gene_list(marker_dir / f"{child_id}_genes.json", selected)

            if not selected:
                node_log["child_status"][child_name] = {
                    "status": "skipped",
                    "reason": "no_markers",
                    "n_pos": pos,
                    "n_neg": neg,
                    "n_genes": 0,
                }
                child_models[child_name] = DummyModel()
                continue

            gene_idx = [gene_names.index(g) for g in selected if g in gene_names]
            model = train_child_model(
                model_type,
                X_parent,
                y_bin,
                gene_idx,
                seed=seed,
                hyperparams=model_hyperparams,
            )
            child_models[child_name] = model
            node_log["child_status"][child_name] = {
                "status": "trained",
                "reason": None,
                "n_pos": pos,
                "n_neg": neg,
                "n_genes": int(len(selected)),
            }
            if model_dir is not None:
                model_path = model_dir / f"{child_id}.safe.pt"
                save_child_model(model, model_path)
                child_model_paths[child_name] = str(model_path.relative_to(base_dir))

        gate = {
            "kind": "ovr_child",
            "node": node_path,
            "children": list(child_names),
            "child_models": child_models,
            "child_priors": child_priors,
            "child_model_paths": child_model_paths,
        }
        gates[node_path] = gate
        log["nodes"][node_path] = node_log

    meta = {
        "root_key": root_key,
        "height_train": int(height_train),
        "model_type": str(model_type),
        "topk_markers": int(topk_markers),
        "min_pct": float(min_pct),
        "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
        "min_samples_per_class": int(min_samples_per_class),
    }
    if base_dir is not None:
        meta_path = base_dir / "meta.json"
        meta_path.write_text(json.dumps({"meta": meta, "log": log}, indent=2))

    return {"gates": gates, "log": log, "meta": meta, "tree": pruned_tree}


def _write_marker_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    headers = ["gene", "auc", "power", "direction", "mean_pos", "mean_neg", "pct_pos", "pct_neg"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_gene_list(path: Path, genes: List[str]) -> None:
    path.write_text(json.dumps({"genes": list(genes)}, indent=2))


def _gate_nodes_from_tree(pruned_tree: Dict[str, Any], *, root_key: str, height_train: int) -> List[str]:
    gate_nodes: List[str] = []
    for path, sub in _iter_node_paths(pruned_tree[root_key], [root_key]):
        depth = len(path) - 1
        if depth >= height_train:
            continue
        if isinstance(sub, dict) and len(sub.keys()) >= 2:
            gate_nodes.append("|".join(path))
        elif isinstance(sub, list) and len(sub) >= 2:
            gate_nodes.append("|".join(path))
    return list(dict.fromkeys(gate_nodes))


def _compute_gate_outputs(
    gates: Dict[str, Any],
    X: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    gate_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for node, gate in gates.items():
        children = list(gate.get("children", []))
        if not children:
            continue
        logits_list: List[np.ndarray] = []
        for child in children:
            model = gate.get("child_models", {}).get(child, DummyModel())
            logits_list.append(predict_child_logits(model, X, gene_idx=None))
        logits = np.stack(logits_list, axis=1)
        probs = _softmax(logits)
        gate_outputs[node] = {"logits": logits, "probs": probs, "nodes": children}
    return gate_outputs


def predict_router_soft(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    normalize: bool = True,
    force_gates: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Soft routing: propagate probabilities down the tree."""

    tree = router_bundle["tree"]
    root_key = router_bundle["root_key"]
    gates = router_bundle.get("gates", {})
    leaves = list(router_bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    X = np.asarray(X, dtype=np.float32)

    training_lib_size = router_bundle.get("training_library_size")
    if training_lib_size is not None:
        row_sums = X.sum(axis=1, keepdims=True)
        scale = np.float32(training_lib_size) / np.maximum(row_sums, np.float32(1.0))
        X *= scale

    gate_outputs = _compute_gate_outputs(gates, X)

    if force_gates:
        n = X.shape[0]
        for gate_path, target_child in force_gates.items():
            go = gate_outputs.get(gate_path)
            if go is None:
                continue
            nodes = list(go.get("nodes", []))
            if target_child in nodes:
                forced = np.zeros((n, len(nodes)), dtype=float)
                forced[:, nodes.index(target_child)] = 1.0
                go["probs"] = forced
    n_samples = X.shape[0]
    P_leaf = np.zeros((n_samples, len(leaves)), dtype=float)

    def _route(path: List[str], mass: np.ndarray):
        node_path = "|".join(path)
        sub = get_node(tree, path)
        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_full = [f"{node_path}|{key}" for key in child_keys]
            if not child_keys:
                return
            gate = gate_outputs.get(node_path)
            if gate is None:
                probs = np.full((n_samples, len(child_keys)), 1.0 / float(len(child_keys)))
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
                probs = np.full((n_samples, len(leaf_labels)), 1.0 / float(len(leaf_labels)))
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != leaf_labels:
                    idx = [gate_nodes.index(ch) for ch in leaf_labels]
                    probs = probs[:, idx]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            for j, leaf in enumerate(leaf_labels):
                idx = leaf_to_idx.get(leaf)
                if idx is not None:
                    P_leaf[:, idx] += mass * probs[:, j]
        else:
            raise TypeError(f"Unexpected subtree type at {node_path}: {type(sub)}")

    _route([root_key], np.ones((n_samples,), dtype=float))

    if normalize and P_leaf.size:
        P_leaf = P_leaf / np.maximum(P_leaf.sum(axis=1, keepdims=True), 1e-12)

    hard = predict_router_hard(router_bundle, X, gate_outputs=gate_outputs)
    return {
        "P_sub": P_leaf,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "leaves": leaves,
    }


def predict_router_hard(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    gate_outputs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Hard routing: argmax selection at each node."""

    tree = router_bundle["tree"]
    root_key = router_bundle["root_key"]
    gates = router_bundle.get("gates", {})
    leaves = list(router_bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    X = np.asarray(X, dtype=np.float32)

    training_lib_size = router_bundle.get("training_library_size")
    if training_lib_size is not None:
        row_sums = X.sum(axis=1, keepdims=True)
        scale = np.float32(training_lib_size) / np.maximum(row_sums, np.float32(1.0))
        X *= scale

    if gate_outputs is None:
        gate_outputs = _compute_gate_outputs(gates, X)
    n_samples = X.shape[0]
    selected_paths: List[List[str]] = []
    final_nodes: List[str] = []

    def _route_indices(path: List[str], idx: np.ndarray):
        if idx.size == 0:
            return
        node_path = "|".join(path)
        sub = get_node(tree, path)
        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_full = [f"{node_path}|{key}" for key in child_keys]
            if not child_keys:
                for i in idx:
                    selected_paths[i] = list(path)
                    final_nodes[i] = node_path
                return
            gate = gate_outputs.get(node_path)
            if gate is None:
                probs = np.full((idx.size, len(child_keys)), 1.0 / float(len(child_keys)))
            else:
                probs = np.asarray(gate["probs"], dtype=float)[idx]
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != child_full:
                    cols = [gate_nodes.index(ch) for ch in child_full]
                    probs = probs[:, cols]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            j = probs.argmax(axis=1)
            for ci, child in enumerate(child_keys):
                take = idx[j == ci]
                if take.size:
                    _route_indices(path + [child], take)
        elif isinstance(sub, list):
            leaf_labels = list(sub)
            if not leaf_labels:
                for i in idx:
                    selected_paths[i] = list(path)
                    final_nodes[i] = node_path
                return
            gate = gate_outputs.get(node_path)
            if gate is None or len(leaf_labels) == 1:
                probs = np.full((idx.size, len(leaf_labels)), 1.0 / float(len(leaf_labels)))
            else:
                probs = np.asarray(gate["probs"], dtype=float)[idx]
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != leaf_labels:
                    cols = [gate_nodes.index(ch) for ch in leaf_labels]
                    probs = probs[:, cols]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            j = probs.argmax(axis=1)
            for k, leaf in enumerate(leaf_labels):
                take = idx[j == k]
                if take.size:
                    for i in take:
                        selected_paths[i] = list(path) + [leaf]
                        final_nodes[i] = leaf
                        _ = leaf_to_idx.get(leaf)
        else:
            raise TypeError(f"Unexpected subtree type at {node_path}: {type(sub)}")

    selected_paths = [[] for _ in range(n_samples)]
    final_nodes = ["" for _ in range(n_samples)]
    _route_indices([root_key], np.arange(n_samples, dtype=int))
    selected_nodes = [[node] for node in final_nodes]

    P_leaf = np.zeros((n_samples, len(leaves)), dtype=float)
    for i, leaf in enumerate(final_nodes):
        if leaf in leaf_to_idx:
            P_leaf[i, leaf_to_idx[leaf]] = 1.0

    return {
        "P_sub": P_leaf,
        "selected_paths": selected_paths,
        "selected_nodes": selected_nodes,
        "final_nodes": final_nodes,
        "gate_outputs": gate_outputs,
        "leaves": leaves,
    }


def predict_router_stacked(
    router_bundle: Dict[str, Any],
    X: np.ndarray,
) -> Dict[str, Any]:
    """Stacked final classifier: gate logits + gate probabilities."""

    if "stacker" not in router_bundle:
        raise ValueError("router_bundle missing stacker")
    gate_outputs = _compute_gate_outputs(router_bundle.get("gates", {}), X)
    node_outputs = {node: {"probs": gate_outputs[node]["probs"]} for node in gate_outputs}
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


def train_roc_router(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    tree: Dict[str, Any],
    *,
    root_key: str = "node",
    height_train: int,
    gene_names: List[str],
    model_type: str = "logreg",
    topk_markers: int = 50,
    min_pct: float = 0.1,
    min_diff_pct: Optional[float] = None,
    min_samples_per_class: int = 1,
    seed: int = 0,
    model_hyperparams: Optional[Dict[str, Any]] = None,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
    train_stacker: bool = False,
    stacker_folds: int = 5,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
) -> Dict[str, Any]:
    """Train a ROC router across heights up to height_train (mouse only)."""

    excluded = list(excluded_leaves or EXCLUDED_LEAVES)
    y_all = np.asarray(y_sub_train).astype(str)
    keep_mask = ~np.isin(y_all, np.asarray(excluded, dtype=object))
    X = np.asarray(X_train, dtype=float)[keep_mask]
    y = y_all[keep_mask]

    subtype_to_path = walk_paths(tree)
    kept_leaves = [leaf for leaf in subtype_to_path.keys() if leaf not in excluded]
    pruned_tree = _prune_tree_to_leaves(tree, kept_leaves, root_key=root_key)
    subtype_to_path = walk_paths(pruned_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)

    max_height = len(height_partitions(pruned_tree, root_key=root_key))
    if height_train > max_height:
        height_train = max_height

    leaf_order = sorted(list(subtype_to_path.keys()))
    gate_nodes = _gate_nodes_from_tree(pruned_tree, root_key=root_key, height_train=height_train)

    gates: Dict[str, Any] = {}
    logs: Dict[str, Any] = {}
    if out_dir is not None:
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

    for h in progress(range(1, height_train + 1), total=height_train, desc="Training ROC router"):
        height_dir = None
        if out_dir is not None:
            height_dir = Path(out_dir) / f"height_{h}"
            height_dir.mkdir(parents=True, exist_ok=True)
        out = train_router_at_height(
            X,
            y,
            pruned_tree,
            root_key=root_key,
            height_train=h,
            gene_names=gene_names,
            model_type=model_type,
            topk_markers=topk_markers,
            min_pct=min_pct,
            min_diff_pct=min_diff_pct,
            min_samples_per_class=min_samples_per_class,
            seed=seed,
            model_hyperparams=model_hyperparams,
            excluded_leaves=[],
            out_dir=height_dir,
        )
        gates.update(out["gates"])
        logs[f"height_{h}"] = out["log"]

    config = {
        "topk_markers": int(topk_markers),
        "min_pct": float(min_pct),
        "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
        "min_samples_per_class": int(min_samples_per_class),
    }

    router_bundle: Dict[str, Any] = {
        "kind": "roc_router",
        "tree": pruned_tree,
        "root_key": root_key,
        "height_train": int(height_train),
        "gates": gates,
        "gate_nodes": gate_nodes,
        "leaves": leaf_order,
        "node_to_leaves": node_to_leaves,
        "subtype_to_path": subtype_to_path,
        "excluded_leaves": excluded,
        "model_type": str(model_type),
        "config": config,
    }

    if train_stacker:
        stacker = _train_router_stacker(
            X,
            y,
            router_bundle,
            gene_names=gene_names,
            stacker_folds=stacker_folds,
            stacker_reg=stacker_reg,
            stacker_lr=stacker_lr,
            stacker_epochs=stacker_epochs,
            seed=seed,
        )
        router_bundle["stacker"] = stacker

    if out_dir is not None:
        gate_children = {node: gate.get("children", []) for node, gate in gates.items()}
        child_model_paths = {node: gate.get("child_model_paths", {}) for node, gate in gates.items()}
        meta = {
            "kind": "roc_router",
            "root_key": root_key,
            "height_train": int(height_train),
            "model_type": str(model_type),
            "excluded_leaves": excluded,
            "tree": pruned_tree,
            "gate_nodes": gate_nodes,
            "gene_names": list(gene_names),
            "config": config,
            "logs": logs,
            "gates": gate_children,
            "child_model_paths": child_model_paths,
        }
        if "stacker" in router_bundle:
            stacker_dir = Path(out_dir) / "stacker"
            stacker_dir.mkdir(parents=True, exist_ok=True)
            stacker_path = stacker_dir / "stacker.safe.pt"
            save_stacker_bundle(router_bundle["stacker"], str(stacker_path))
            meta["stacker_path"] = str(stacker_path.relative_to(Path(out_dir)))
        Path(out_dir, "meta.json").write_text(json.dumps(meta, indent=2))

    return router_bundle


def _train_router_stacker(
    X: np.ndarray,
    y: np.ndarray,
    router_bundle: Dict[str, Any],
    *,
    gene_names: List[str],
    stacker_folds: int,
    stacker_reg: float,
    stacker_lr: float,
    stacker_epochs: int,
    seed: int,
) -> Dict[str, Any]:
    try:
        from sklearn.model_selection import StratifiedKFold  # type: ignore
    except Exception as exc:
        raise ImportError("sklearn is required for OOF stacker training") from exc

    y = np.asarray(y).astype(str)
    min_count = int(np.min(np.bincount(np.unique(y, return_inverse=True)[1]))) if y.size else 0
    if min_count < 2:
        raise ValueError("Not enough samples per class for OOF stacker training")
    n_splits = min(int(stacker_folds), min_count)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_meta_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    config = router_bundle.get("config", {})
    height_train = int(router_bundle.get("height_train", 1))
    model_type = str(router_bundle.get("model_type", "logreg"))

    for train_idx, val_idx in skf.split(X, y):
        fold_bundle = train_roc_router(
            X[train_idx],
            y[train_idx],
            router_bundle["tree"],
            root_key=router_bundle["root_key"],
            height_train=height_train,
            gene_names=gene_names,
            model_type=model_type,
            topk_markers=int(config.get("topk_markers", 50)),
            min_pct=float(config.get("min_pct", 0.1)),
            min_diff_pct=config.get("min_diff_pct"),
            min_samples_per_class=int(config.get("min_samples_per_class", 1)),
            seed=seed,
            excluded_leaves=router_bundle.get("excluded_leaves", []),
            out_dir=None,
            train_stacker=False,
        )
        gate_outputs = _compute_gate_outputs(fold_bundle.get("gates", {}), X[val_idx])
        gate_order = list(router_bundle.get("gate_nodes", gate_outputs.keys()))
        node_outputs = {node: {"probs": gate_outputs[node]["probs"]} for node in gate_order}
        feature_spec = {
            "gate_features": "logits",
            "node_features": "probs",
            "gate_order": gate_order,
            "node_order": gate_order,
            "leaf_order": list(router_bundle["leaves"]),
        }
        X_meta, feature_spec = build_stacker_features(
            gate_outputs,
            node_outputs,
            list(router_bundle["leaves"]),
            node_order=gate_order,
            feature_spec=feature_spec,
        )
        X_meta_parts.append(X_meta)
        y_parts.append(y[val_idx])

    X_meta_all = np.vstack(X_meta_parts)
    y_all = np.concatenate(y_parts)
    stacker = train_leaf_stacker(
        X_meta_all,
        y_all,
        list(router_bundle["leaves"]),
        reg=stacker_reg,
        lr=stacker_lr,
        epochs=stacker_epochs,
        seed=seed,
        feature_spec=feature_spec,
    )
    return stacker


def load_roc_router(out_dir: str | Path) -> Dict[str, Any]:
    """Load a ROC router bundle from disk, returning DummyModels for missing children."""

    base = Path(out_dir)
    meta_path = base / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())

    pruned_tree = meta["tree"]
    root_key = meta["root_key"]
    subtype_to_path = walk_paths(pruned_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    leaves = sorted(list(subtype_to_path.keys()))

    gates: Dict[str, Any] = {}
    child_model_paths = meta.get("child_model_paths", {})
    for node, children in meta.get("gates", {}).items():
        child_models: Dict[str, Any] = {}
        paths_for_node = child_model_paths.get(node, {})
        for child in children:
            rel_path = paths_for_node.get(child)
            if rel_path:
                child_models[child] = load_child_model(base / rel_path)
            else:
                child_models[child] = DummyModel()
        gates[node] = {
            "kind": "ovr_child",
            "node": node,
            "children": list(children),
            "child_models": child_models,
        }

    bundle = {
        "kind": "roc_router",
        "tree": pruned_tree,
        "root_key": root_key,
        "gates": gates,
        "leaves": leaves,
        "node_to_leaves": node_to_leaves,
        "subtype_to_path": subtype_to_path,
        "excluded_leaves": meta.get("excluded_leaves", []),
        "gate_nodes": meta.get("gate_nodes", []),
        "height_train": meta.get("height_train"),
        "model_type": meta.get("model_type", "logreg"),
        "config": meta.get("config", {}),
    }
    stacker_path = meta.get("stacker_path")
    if isinstance(stacker_path, str):
        stacker_full = base / stacker_path
        if stacker_full.exists():
            bundle["stacker"] = load_stacker_bundle(str(stacker_full))
    return bundle


def summarize_router(router_bundle: Dict[str, Any]) -> "pd.DataFrame":
    """Summarize trained routers per node for quick inspection."""

    import pandas as pd

    rows: List[Dict[str, Any]] = []
    gates = router_bundle.get("gates", {})
    for node, gate in gates.items():
        children = list(gate.get("children", []))
        child_models = gate.get("child_models", {})
        n_trained = sum(
            1 for child in children if not isinstance(child_models.get(child), DummyModel)
        )
        rows.append(
            {
                "node": node,
                "n_children": len(children),
                "n_trained": n_trained,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_children", "node"]).reset_index(drop=True)
