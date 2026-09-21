"""Parallel v4 pipelines: notebook ROC and improved hierarchical routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import numpy as np

from .api import compute_feature_signature
from .child_models import DummyModel, load_child_model, predict_child_logits, save_child_model, train_child_model
from .hierarchy import get_node, height_partitions, node_to_leaves_from_subtype_paths, walk_paths
from .roc_markers import seurat_roc_markers, select_top_markers, shared_markers_from_children
from .routing_spec import EXCLUDED_LEAVES, EXCLUSION_RATIONALE
from .progress import progress
from .stacker import build_stacker_features, predict_leaf_stacker, save_stacker_bundle, train_leaf_stacker


def _safe_id(value: str) -> str:
    return str(value).replace("|", "__").replace(":", "_").replace("/", "_")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


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


def _gate_nodes_for_height(tree: Dict[str, Any], *, root_key: str, height_train: int) -> List[str]:
    gate_nodes: List[str] = []
    node_iter = list(_iter_node_paths(tree[root_key], [root_key]))
    for path, sub in progress(node_iter, total=len(node_iter), desc="Training gates"):
        depth = len(path) - 1
        if depth >= height_train:
            continue
        if isinstance(sub, dict) and len(sub.keys()) >= 2:
            gate_nodes.append("|".join(path))
        elif isinstance(sub, list) and len(sub) >= 2:
            gate_nodes.append("|".join(path))
    return list(dict.fromkeys(gate_nodes))


def _knee_topk(scores_sorted: np.ndarray, min_k: int = 20, max_k: int = 500) -> int:
    """Find the knee/elbow in sorted (descending) feature scores."""
    n = len(scores_sorted)
    if n <= min_k:
        return n
    k_range = min(n, max_k)
    y = scores_sorted[:k_range]
    y_min, y_max = y[-1], y[0]
    if y_max - y_min < 1e-12:
        return min_k
    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, k_range)
    x0, y0 = 0.0, 1.0
    x1, y1 = 1.0, y_norm[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.sqrt(dx * dx + dy * dy)
    dist = np.abs(dy * x_norm - dx * y_norm + x1 * y0 - y1 * x0) / denom
    knee = int(np.argmax(dist)) + 1
    return max(min_k, min(knee, max_k))


def _select_features_binary(
    X: np.ndarray,
    y_bin: np.ndarray,
    gene_names: List[str],
    *,
    method: str,
    topk: int,
) -> List[str]:
    try:
        if method == "f_classif":
            from sklearn.feature_selection import f_classif  # type: ignore

            scores, _ = f_classif(X, y_bin)
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif  # type: ignore

            scores = mutual_info_classif(X, y_bin)
        else:
            raise ValueError(f"Unknown feature method: {method}")
    except Exception as exc:
        raise ImportError("sklearn is required for f_classif/mutual_info selection") from exc

    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    sorted_scores = np.sort(scores)[::-1]

    if topk <= 0:
        k = _knee_topk(sorted_scores, min_k=20, max_k=500)
    else:
        k = int(topk)

    idx = np.argsort(scores)[::-1][:k]
    return [gene_names[i] for i in idx]


def _select_features_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: List[str],
    *,
    method: str,
    topk: int,
) -> List[str]:
    return _select_features_binary(X, y, gene_names, method=method, topk=topk)


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


def _compute_gate_outputs(gates: Dict[str, Any], X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
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
        prob_mode = gate.get("prob_mode")
        if prob_mode is None and gate.get("kind") == "ovr_child":
            prob_mode = "sigmoid_norm"
        prob_mode = str(prob_mode or "softmax").lower()
        if prob_mode == "sigmoid_norm":
            probs = _sigmoid(logits)
            row_sums = probs.sum(axis=1, keepdims=True)
            probs = probs / np.maximum(row_sums, 1e-12)
            if probs.shape[1] > 0:
                zero_rows = row_sums.squeeze(1) <= 1e-12
                if np.any(zero_rows):
                    probs[zero_rows] = 1.0 / float(probs.shape[1])
        else:
            probs = _softmax(logits)
        gate_outputs[node] = {"logits": logits, "probs": probs, "nodes": children}
    return gate_outputs


def _train_node_gates(
    X: np.ndarray,
    y: np.ndarray,
    tree: Dict[str, Any],
    *,
    root_key: str,
    height_train: int,
    gene_names: List[str],
    model_type: str,
    feature_method: str,
    topk: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    shared_features: bool,
    min_samples_per_class: int,
    seed: int,
    model_hyperparams: Optional[Dict[str, Any]],
    out_dir: Optional[Path],
    log: Dict[str, Any],
) -> Dict[str, Any]:
    gates: Dict[str, Any] = {}
    node_to_leaves = node_to_leaves_from_subtype_paths(walk_paths(tree))
    gate_nodes = set(_gate_nodes_for_height(tree, root_key=root_key, height_train=height_train))

    for path, sub in _iter_node_paths(tree[root_key], [root_key]):
        node_path = "|".join(path)
        if node_path not in gate_nodes:
            continue

        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_names = [f"{node_path}|{key}" for key in child_keys]
        elif isinstance(sub, list):
            child_keys = list(sub)
            child_names = list(child_keys)
        else:
            continue

        if len(child_names) < 2:
            continue

        node_leaves = node_to_leaves.get(node_path, [])
        if not node_leaves:
            continue
        mask = np.isin(y, np.asarray(node_leaves, dtype=object))
        X_node = X[mask]
        y_node = y[mask]

        node_log: Dict[str, Any] = {"children": list(child_names), "child_status": {}}
        child_models: Dict[str, Any] = {}
        child_gene_idx: Dict[str, List[int]] = {}

        child_leaf_sets = []
        for child_name, child_key in zip(child_names, child_keys):
            if isinstance(sub, dict):
                leaves = node_to_leaves.get(child_name, [])
            else:
                leaves = [child_key]
            child_leaf_sets.append((child_name, leaves))

        shared_genes: Optional[List[str]] = None
        if shared_features and feature_method != "roc":
            y_multi = np.zeros((y_node.shape[0],), dtype=int)
            for ci, (child_name, leaves) in enumerate(child_leaf_sets):
                y_multi[np.isin(y_node, np.asarray(leaves, dtype=object))] = ci
            if feature_method != "none":
                shared_genes = _select_features_multiclass(
                    X_node,
                    y_multi,
                    gene_names,
                    method=feature_method,
                    topk=topk,
                )
            else:
                shared_genes = list(gene_names)

        per_child_markers: Dict[str, Any] = {}
        if shared_features and feature_method == "roc":
            roc_tables = []
            for child_name, leaves in child_leaf_sets:
                y_bin = np.isin(y_node, np.asarray(leaves, dtype=object)).astype(int)
                markers = seurat_roc_markers(
                    X_node, y_bin, gene_names, min_pct=min_pct, min_diff_pct=min_diff_pct
                )
                per_child_markers[child_name] = markers
                roc_tables.append(markers)
            shared_genes = shared_markers_from_children(
                roc_tables,
                topk=topk,
                min_pct=min_pct,
                min_diff_pct=min_diff_pct,
                balanced=True,
            )
            if not shared_genes:
                y_multi = np.zeros((y_node.shape[0],), dtype=int)
                for ci, (child_name, leaves) in enumerate(child_leaf_sets):
                    y_multi[np.isin(y_node, np.asarray(leaves, dtype=object))] = ci
                try:
                    shared_genes = _select_features_multiclass(
                        X_node,
                        y_multi,
                        gene_names,
                        method="f_classif",
                        topk=topk,
                    )
                except Exception:
                    shared_genes = list(gene_names)

        for child_name, leaves in child_leaf_sets:
            y_bin = np.isin(y_node, np.asarray(leaves, dtype=object)).astype(int)
            pos = int(y_bin.sum())
            neg = int(y_bin.size - pos)

            parent_dir = _safe_id(node_path)
            child_id = _safe_id(child_name)
            marker_dir = None
            model_dir = None
            if out_dir is not None:
                marker_dir = out_dir / "markers" / parent_dir
                model_dir = out_dir / "child_models" / parent_dir
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
                    _write_marker_csv(marker_dir / f"{child_id}_roc_markers.csv", [])
                    _write_gene_list(marker_dir / f"{child_id}_genes.json", [])
                continue

            fallback_method = None
            if feature_method == "roc":
                if shared_genes is None:
                    markers = seurat_roc_markers(
                        X_node, y_bin, gene_names, min_pct=min_pct, min_diff_pct=min_diff_pct
                    )
                    selected = select_top_markers(
                        markers, topk=topk, min_pct=min_pct, min_diff_pct=min_diff_pct
                    )
                    if marker_dir is not None:
                        markers.to_csv(marker_dir / f"{child_id}_roc_markers.csv", index=False)
                else:
                    selected = list(shared_genes)
                    if marker_dir is not None:
                        markers = per_child_markers.get(child_name)
                        if markers is not None:
                            markers.to_csv(marker_dir / f"{child_id}_roc_markers.csv", index=False)
                        else:
                            _write_marker_csv(marker_dir / f"{child_id}_roc_markers.csv", [])
            elif feature_method == "none":
                selected = list(gene_names)
            else:
                if shared_genes is None:
                    selected = _select_features_binary(
                        X_node, y_bin, gene_names, method=feature_method, topk=topk
                    )
                else:
                    selected = list(shared_genes)

            if marker_dir is not None:
                _write_gene_list(marker_dir / f"{child_id}_genes.json", selected)

            if not selected:
                try:
                    selected = _select_features_binary(
                        X_node, y_bin, gene_names, method="f_classif", topk=topk
                    )
                    fallback_method = "f_classif"
                except Exception:
                    selected = list(gene_names)
                    fallback_method = "all_genes"
                if marker_dir is not None:
                    _write_gene_list(marker_dir / f"{child_id}_genes.json", selected)

            gene_idx = [gene_names.index(g) for g in selected if g in gene_names]
            model = train_child_model(
                model_type,
                X_node,
                y_bin,
                gene_idx,
                seed=seed,
                hyperparams=model_hyperparams,
            )
            child_models[child_name] = model
            child_gene_idx[child_name] = list(gene_idx)
            node_log["child_status"][child_name] = {
                "status": "trained_fallback" if fallback_method else "trained",
                "reason": f"fallback_{fallback_method}" if fallback_method else None,
                "n_pos": pos,
                "n_neg": neg,
                "n_genes": int(len(gene_idx)),
            }
            if model_dir is not None:
                model_path = model_dir / f"{child_id}.safe.pt"
                save_child_model(model, model_path)

        gates[node_path] = {
            "kind": "ovr_child",
            "prob_mode": "sigmoid_norm",
            "node": node_path,
            "children": list(child_names),
            "child_models": child_models,
            "child_gene_idx": child_gene_idx,
            "feature_method": feature_method,
            "shared_features": shared_features,
        }
        log["nodes"][node_path] = node_log

    return gates


def _train_stacker(
    X: np.ndarray,
    y: np.ndarray,
    router_bundle: Dict[str, Any],
    *,
    stacker_folds: int,
    stacker_reg: float,
    stacker_lr: float,
    stacker_epochs: int,
    seed: int,
    train_fn,
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
    gate_order = list(router_bundle.get("gate_nodes", []))

    for train_idx, val_idx in skf.split(X, y):
        fold_bundle = train_fn(X[train_idx], y[train_idx])
        gate_outputs = _compute_gate_outputs(fold_bundle.get("gates", {}), X[val_idx])
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
    return train_leaf_stacker(
        X_meta_all,
        y_all,
        list(router_bundle["leaves"]),
        reg=stacker_reg,
        lr=stacker_lr,
        epochs=stacker_epochs,
        seed=seed,
        feature_spec=feature_spec,
    )


def train_v4_notebook(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    root_key: str = "node",
    gene_names: List[str],
    model_type: str,
    topk_markers: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    height_train: int,
    seed: int = 0,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
    train_stacker: bool = False,
    stacker_folds: int = 5,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    model_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Notebook-style ROC routing (per-child markers)."""

    excluded = list(EXCLUDED_LEAVES if excluded_leaves is None else excluded_leaves)
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_sub_train).astype(str)

    keep_mask = ~np.isin(y, np.asarray(excluded, dtype=object))
    X = X[keep_mask]
    y = y[keep_mask]

    subtype_to_path = walk_paths(hierarchy_tree)
    kept_leaves = [leaf for leaf in subtype_to_path.keys() if leaf not in excluded]
    pruned_tree = _prune_tree_to_leaves(hierarchy_tree, kept_leaves, root_key=root_key)
    subtype_to_path = walk_paths(pruned_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    max_height = len(height_partitions(pruned_tree, root_key=root_key))
    height_train = min(height_train, max_height)

    log: Dict[str, Any] = {"height": int(height_train), "nodes": {}}
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    gates = _train_node_gates(
        X,
        y,
        pruned_tree,
        root_key=root_key,
        height_train=height_train,
        gene_names=gene_names,
        model_type=model_type,
        feature_method="roc",
        topk=topk_markers,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        shared_features=False,
        min_samples_per_class=1,
        seed=seed,
        model_hyperparams=model_hyperparams,
        out_dir=out_path,
        log=log,
    )

    gate_nodes = _gate_nodes_for_height(pruned_tree, root_key=root_key, height_train=height_train)
    feature_signature = compute_feature_signature(X, feature_names=gene_names)

    bundle: Dict[str, Any] = {
        "pipeline": "v4_notebook",
        "tree": pruned_tree,
        "root_key": root_key,
        "gates": gates,
        "gate_nodes": gate_nodes,
        "leaves": sorted(list(subtype_to_path.keys())),
        "node_to_leaves": node_to_leaves,
        "subtype_to_path": subtype_to_path,
        "excluded_leaves": excluded,
        "excluded_rationale": EXCLUSION_RATIONALE,
        "feature_signature": feature_signature,
        "config": {
            "model_type": model_type,
            "topk_markers": int(topk_markers),
            "min_pct": float(min_pct),
            "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
            "height_train": int(height_train),
            "seed": int(seed),
        },
        "log": log,
    }

    if train_stacker:
        train_fn = lambda Xf, yf: train_v4_notebook(
            Xf,
            yf,
            pruned_tree,
            root_key=root_key,
            gene_names=gene_names,
            model_type=model_type,
            topk_markers=topk_markers,
            min_pct=min_pct,
            min_diff_pct=min_diff_pct,
            height_train=height_train,
            seed=seed,
            excluded_leaves=[],
            out_dir=None,
            train_stacker=False,
            model_hyperparams=model_hyperparams,
        )
        bundle["stacker"] = _train_stacker(
            X,
            y,
            bundle,
            stacker_folds=stacker_folds,
            stacker_reg=stacker_reg,
            stacker_lr=stacker_lr,
            stacker_epochs=stacker_epochs,
            seed=seed,
            train_fn=train_fn,
        )

    if out_path is not None:
        gates_meta = {node: gate.get("children", []) for node, gate in gates.items()}
        meta = {
            "pipeline": "v4_notebook",
            "tree": pruned_tree,
            "root_key": root_key,
            "gate_nodes": gate_nodes,
            "gates": gates_meta,
            "leaves": bundle["leaves"],
            "excluded_leaves": excluded,
            "excluded_rationale": EXCLUSION_RATIONALE,
            "feature_signature": feature_signature,
            "config": bundle["config"],
            "log": log,
        }
        if "stacker" in bundle:
            stacker_dir = out_path / "stacker"
            stacker_dir.mkdir(parents=True, exist_ok=True)
            save_stacker_bundle(bundle["stacker"], str(stacker_dir / "stacker.safe.pt"))
            meta["stacker_path"] = str(Path("stacker") / "stacker.safe.pt")
        (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    return bundle


def train_v4_improved(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    root_key: str = "node",
    gene_names: List[str],
    model_type: str,
    feature_method: str,
    topk: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    height_train: int,
    shared_features: bool = True,
    seed: int = 0,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
    train_stacker: bool = False,
    stacker_folds: int = 5,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    model_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Improved routing with configurable feature selection and shared features."""

    excluded = list(EXCLUDED_LEAVES if excluded_leaves is None else excluded_leaves)
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_sub_train).astype(str)

    keep_mask = ~np.isin(y, np.asarray(excluded, dtype=object))
    X = X[keep_mask]
    y = y[keep_mask]

    subtype_to_path = walk_paths(hierarchy_tree)
    kept_leaves = [leaf for leaf in subtype_to_path.keys() if leaf not in excluded]
    pruned_tree = _prune_tree_to_leaves(hierarchy_tree, kept_leaves, root_key=root_key)
    subtype_to_path = walk_paths(pruned_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    max_height = len(height_partitions(pruned_tree, root_key=root_key))
    height_train = min(height_train, max_height)

    log: Dict[str, Any] = {"height": int(height_train), "nodes": {}}
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    gates = _train_node_gates(
        X,
        y,
        pruned_tree,
        root_key=root_key,
        height_train=height_train,
        gene_names=gene_names,
        model_type=model_type,
        feature_method=feature_method,
        topk=topk,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        shared_features=shared_features,
        min_samples_per_class=1,
        seed=seed,
        model_hyperparams=model_hyperparams,
        out_dir=out_path,
        log=log,
    )

    gate_nodes = _gate_nodes_for_height(pruned_tree, root_key=root_key, height_train=height_train)
    feature_signature = compute_feature_signature(X, feature_names=gene_names)

    bundle: Dict[str, Any] = {
        "pipeline": "v4_improved",
        "tree": pruned_tree,
        "root_key": root_key,
        "gates": gates,
        "gate_nodes": gate_nodes,
        "leaves": sorted(list(subtype_to_path.keys())),
        "node_to_leaves": node_to_leaves,
        "subtype_to_path": subtype_to_path,
        "excluded_leaves": excluded,
        "excluded_rationale": EXCLUSION_RATIONALE,
        "feature_signature": feature_signature,
        "config": {
            "model_type": model_type,
            "feature_method": feature_method,
            "shared_features": shared_features,
            "topk": int(topk),
            "min_pct": float(min_pct),
            "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
            "height_train": int(height_train),
            "seed": int(seed),
        },
        "log": log,
    }

    if train_stacker:
        train_fn = lambda Xf, yf: train_v4_improved(
            Xf,
            yf,
            pruned_tree,
            root_key=root_key,
            gene_names=gene_names,
            model_type=model_type,
            feature_method=feature_method,
            topk=topk,
            min_pct=min_pct,
            min_diff_pct=min_diff_pct,
            height_train=height_train,
            shared_features=shared_features,
            seed=seed,
            excluded_leaves=[],
            out_dir=None,
            train_stacker=False,
            model_hyperparams=model_hyperparams,
        )
        bundle["stacker"] = _train_stacker(
            X,
            y,
            bundle,
            stacker_folds=stacker_folds,
            stacker_reg=stacker_reg,
            stacker_lr=stacker_lr,
            stacker_epochs=stacker_epochs,
            seed=seed,
            train_fn=train_fn,
        )

    if out_path is not None:
        gates_meta = {node: gate.get("children", []) for node, gate in gates.items()}
        meta = {
            "pipeline": "v4_improved",
            "tree": pruned_tree,
            "root_key": root_key,
            "gate_nodes": gate_nodes,
            "gates": gates_meta,
            "leaves": bundle["leaves"],
            "excluded_leaves": excluded,
            "excluded_rationale": EXCLUSION_RATIONALE,
            "feature_signature": feature_signature,
            "config": bundle["config"],
            "log": log,
        }
        if "stacker" in bundle:
            stacker_dir = out_path / "stacker"
            stacker_dir.mkdir(parents=True, exist_ok=True)
            save_stacker_bundle(bundle["stacker"], str(stacker_dir / "stacker.safe.pt"))
            meta["stacker_path"] = str(Path("stacker") / "stacker.safe.pt")
        (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    return bundle


def _signature_mismatch(
    stored: Dict[str, Any],
    current: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    for key in ("input_dim", "feature_names_hash", "preprocess_meta"):
        if stored.get(key) != current.get(key):
            reasons.append(f"feature_signature.{key} mismatch")
    return reasons


def ensure_v4_notebook(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    root_key: str = "node",
    gene_names: List[str],
    model_type: str,
    topk_markers: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    height_train: int,
    seed: int = 0,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
    train_stacker: bool = False,
    stacker_folds: int = 5,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    model_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load or train the notebook ROC router with signature checks."""

    if out_dir is not None:
        meta_path = Path(out_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            signature = compute_feature_signature(X_train, feature_names=gene_names)
            reasons = _signature_mismatch(meta.get("feature_signature", {}), signature)
            cfg = meta.get("config", {})
            expected = {
                "model_type": model_type,
                "topk_markers": int(topk_markers),
                "min_pct": float(min_pct),
                "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
                "height_train": int(height_train),
            }
            for key, value in expected.items():
                if cfg.get(key) != value:
                    reasons.append(f"config.{key} mismatch")
            if reasons:
                print("v4_notebook: artifacts incompatible; retraining.")
                for reason in reasons:
                    print(f"  - {reason}")
            else:
                return load_v4_router(out_dir)

    return train_v4_notebook(
        X_train,
        y_sub_train,
        hierarchy_tree,
        root_key=root_key,
        gene_names=gene_names,
        model_type=model_type,
        topk_markers=topk_markers,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        height_train=height_train,
        seed=seed,
        excluded_leaves=excluded_leaves,
        out_dir=out_dir,
        train_stacker=train_stacker,
        stacker_folds=stacker_folds,
        stacker_reg=stacker_reg,
        stacker_lr=stacker_lr,
        stacker_epochs=stacker_epochs,
        model_hyperparams=model_hyperparams,
    )


def ensure_v4_improved(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    root_key: str = "node",
    gene_names: List[str],
    model_type: str,
    feature_method: str,
    topk: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    height_train: int,
    shared_features: bool = True,
    seed: int = 0,
    excluded_leaves: Optional[List[str]] = None,
    out_dir: Optional[str | Path] = None,
    train_stacker: bool = False,
    stacker_folds: int = 5,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    model_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load or train the improved router with signature checks."""

    if out_dir is not None:
        meta_path = Path(out_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            signature = compute_feature_signature(X_train, feature_names=gene_names)
            reasons = _signature_mismatch(meta.get("feature_signature", {}), signature)
            cfg = meta.get("config", {})
            expected = {
                "model_type": model_type,
                "feature_method": feature_method,
                "shared_features": shared_features,
                "topk": int(topk),
                "min_pct": float(min_pct),
                "min_diff_pct": float(min_diff_pct) if min_diff_pct is not None else None,
                "height_train": int(height_train),
            }
            for key, value in expected.items():
                if cfg.get(key) != value:
                    reasons.append(f"config.{key} mismatch")
            if reasons:
                print("v4_improved: artifacts incompatible; retraining.")
                for reason in reasons:
                    print(f"  - {reason}")
            else:
                return load_v4_router(out_dir)

    return train_v4_improved(
        X_train,
        y_sub_train,
        hierarchy_tree,
        root_key=root_key,
        gene_names=gene_names,
        model_type=model_type,
        feature_method=feature_method,
        topk=topk,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        height_train=height_train,
        shared_features=shared_features,
        seed=seed,
        excluded_leaves=excluded_leaves,
        out_dir=out_dir,
        train_stacker=train_stacker,
        stacker_folds=stacker_folds,
        stacker_reg=stacker_reg,
        stacker_lr=stacker_lr,
        stacker_epochs=stacker_epochs,
        model_hyperparams=model_hyperparams,
    )


def load_v4_router(out_dir: str | Path) -> Dict[str, Any]:
    """Load a saved router bundle, returning DummyModels for missing children."""

    base = Path(out_dir)
    meta_path = base / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())

    tree = meta["tree"]
    root_key = meta["root_key"]
    subtype_to_path = walk_paths(tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    leaves = sorted(list(subtype_to_path.keys()))

    gates: Dict[str, Any] = {}
    for node, children in meta.get("gates", {}).items():
        child_models = {}
        for child in children:
            model_path = base / "child_models" / _safe_id(node) / f"{_safe_id(child)}.safe.pt"
            child_models[child] = load_child_model(model_path)
        gates[node] = {
            "kind": "ovr_child",
            "node": node,
            "children": list(children),
            "child_models": child_models,
        }

    bundle = {
        **meta,
        "gates": gates,
        "node_to_leaves": node_to_leaves,
        "subtype_to_path": subtype_to_path,
        "leaves": leaves,
    }
    stacker_path = meta.get("stacker_path")
    if isinstance(stacker_path, str):
        from .stacker import load_stacker_bundle

        full = base / stacker_path
        if full.exists():
            bundle["stacker"] = load_stacker_bundle(str(full))
    return bundle


def predict_router_hard(bundle: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    """Hard routing: argmax child at each node."""

    tree = bundle["tree"]
    root_key = bundle["root_key"]
    gates = bundle.get("gates", {})
    leaves = list(bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    gate_outputs = _compute_gate_outputs(gates, X)
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    selected_paths: List[List[str]] = [[] for _ in range(n_samples)]
    final_nodes: List[str] = ["" for _ in range(n_samples)]

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


def predict_router_soft(
    bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    force_gates: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Soft routing: propagate probabilities down the tree."""

    tree = bundle["tree"]
    root_key = bundle["root_key"]
    gates = bundle.get("gates", {})
    leaves = list(bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    gate_outputs = _compute_gate_outputs(gates, X)

    if force_gates:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        for gate_path, target_child in force_gates.items():
            go = gate_outputs.get(gate_path)
            if go is None:
                continue
            nodes = list(go.get("nodes", []))
            if target_child in nodes:
                forced = np.zeros((n, len(nodes)), dtype=float)
                forced[:, nodes.index(target_child)] = 1.0
                go["probs"] = forced

    X = np.asarray(X, dtype=float)
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
    if P_leaf.size:
        P_leaf = P_leaf / np.maximum(P_leaf.sum(axis=1, keepdims=True), 1e-12)

    hard = predict_router_hard(bundle, X)
    return {
        "P_sub": P_leaf,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "leaves": leaves,
    }


def predict_router_stacked(bundle: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    """Stacked final classifier: gate logits + node outputs."""

    if "stacker" not in bundle:
        raise ValueError("bundle missing stacker")
    gate_outputs = _compute_gate_outputs(bundle.get("gates", {}), X)
    node_outputs = {node: {"probs": gate_outputs[node]["probs"]} for node in bundle.get("gate_nodes", [])}
    X_meta, _ = build_stacker_features(
        gate_outputs,
        node_outputs,
        list(bundle["leaves"]),
        node_order=list(bundle.get("gate_nodes", [])),
        feature_spec=bundle["stacker"].get("feature_spec"),
    )
    P_sub = predict_leaf_stacker(bundle["stacker"], X_meta)
    hard = predict_router_hard(bundle, X)
    return {
        "P_sub": P_sub,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "leaves": list(bundle["leaves"]),
    }
