"""Minimal public API for MoE v4 (node-based routing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .experts import (
    combine_node_weights_with_leaf_scores,
    load_leaf_experts_safe,
    score_leaf_experts,
    train_leaf_experts_ovr,
    train_node_specific_leaf_experts,
)
from .gates import (
    load_gate_bundle,
    load_node_gate,
    nodes_to_leaf_candidates,
    predict_binary_gate,
    predict_node_child_probs,
    predict_node_scores,
    route_nodes,
    save_gate_bundle,
    save_node_gate,
    train_binary_gate,
    train_node_gate,
    train_node_ovr_gate,
)
from .hierarchy import height_partitions, node_to_leaves_from_subtype_paths, walk_paths
from .moe_types import ModelBundle
from .routing_spec import (
    CALB_NODE,
    EXCLUDED_LEAVES,
    EXCLUSION_RATIONALE,
    GAD_NEG_NODE,
    GAD_POS_NODE,
    ROOT_NODE,
    SOX6_NODE,
    assign_gadneg_branch,
    build_node_to_leaves,
    filter_excluded_leaves,
    is_gad_positive,
    node_dir_name,
)
from .stacker import (
    build_stacker_features,
    load_stacker_bundle,
    predict_leaf_stacker,
    save_stacker_bundle,
    train_leaf_stacker,
)
from .tree import (
    TreeIndex,
    build_tree_index_from_hierarchy,
    child_to_leaves,
    internal_nodes,
    leaf_edges,
    node_dir_name as tree_node_dir_name,
    node_edges,
)


def build_node_moe(config: Dict[str, Any]) -> ModelBundle:
    """Assemble a minimal v4 model bundle from explicit inputs."""

    if "tree" not in config:
        raise ValueError("config must include a 'tree' key")

    tree = config["tree"]
    root_key = config.get("root_key", "node")
    height = config.get("height")

    subtype_to_path = walk_paths(tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    partitions = list(height_partitions(tree, root_key=root_key))

    bundle: ModelBundle = {
        "hierarchy": {
            "tree": tree,
            "root_key": root_key,
            "subtype_to_path": subtype_to_path,
            "node_to_leaves": node_to_leaves,
            "partitions": partitions,
        },
        "metadata": {"height": height},
    }

    if "experts" in config:
        bundle["experts"] = config["experts"]

    return bundle


def _hash_values(values: List[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _coerce_meta(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _coerce_meta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_meta(v) for v in value]
    return value


def compute_feature_signature(
    X: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    feature_names_hint: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a feature signature for v4 artifacts."""

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D to compute signature, got {X.shape}")
    input_dim = int(X.shape[1])

    if feature_names is not None:
        feature_names_hash = _hash_values([str(x) for x in feature_names])
    elif feature_names_hint is not None:
        feature_names_hash = _hash_values([str(feature_names_hint)])
    else:
        feature_names_hash = _hash_values([f"input_dim:{input_dim}"])

    meta = _coerce_meta(preprocess_meta or {})
    return {
        "input_dim": input_dim,
        "feature_names_hash": feature_names_hash,
        "preprocess_meta": meta,
    }


def _diff_signature(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diffs: Dict[str, Dict[str, Any]] = {}
    for key in ("input_dim", "feature_names_hash", "preprocess_meta"):
        if old.get(key) != new.get(key):
            diffs[key] = {"old": old.get(key), "new": new.get(key)}
    return diffs


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _logsumexp(x: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_sum = np.exp(x - x_max).sum(axis=axis, keepdims=True)
    out = x_max + np.log(np.maximum(exp_sum, 1e-12))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - _logsumexp(x, axis=1, keepdims=True)


def save_v4_bundle(bundle: ModelBundle, out_dir: str | Path) -> Path:
    """Save a v4 bundle as safe artifacts under out_dir."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if "hierarchy" not in bundle:
        raise ValueError("bundle must include 'hierarchy' for saving")

    hierarchy = bundle["hierarchy"]
    metadata = bundle.get("metadata", {})
    height = metadata.get("height", 0)
    partitions = hierarchy.get("partitions") or []
    nodes = metadata.get("nodes")
    if isinstance(height, int):
        if not isinstance(nodes, list) or not nodes:
            nodes = partitions[height] if 0 <= height < len(partitions) else []
    else:
        nodes = nodes if isinstance(nodes, list) else []
    leaves = metadata.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        leaves = sorted(hierarchy.get("subtype_to_path", {}).keys())

    meta_path = out_path / "meta.json"

    if "gates" in bundle or "experts_by_node" in bundle or "experts_raw_by_node" in bundle:
        gates = bundle.get("gates")
        experts_payload_by_node = bundle.get("experts_raw_by_node") or bundle.get("experts_by_node")
        if not isinstance(gates, dict):
            raise ValueError("bundle must include a 'gates' dict for hierarchical saving")
        if not isinstance(experts_payload_by_node, dict):
            raise ValueError("bundle must include experts_by_node for hierarchical saving")
        if not gates or not experts_payload_by_node:
            raise ValueError("hierarchical bundle must include at least one gate and one expert node")

        gates_dir = out_path / "gates"
        experts_dir = out_path / "experts"
        gates_dir.mkdir(parents=True, exist_ok=True)
        experts_dir.mkdir(parents=True, exist_ok=True)

        gate_files: Dict[str, str] = {}
        gate_nodes: Dict[str, List[str]] = {}
        for node, gate_bundle in gates.items():
            gate_dir = gates_dir / tree_node_dir_name(str(node))
            gate_dir.mkdir(parents=True, exist_ok=True)
            fname = gate_dir / "gate.safe.pt"
            save_gate_bundle(gate_bundle, str(fname))
            gate_files[str(node)] = str(Path("gates") / tree_node_dir_name(str(node)) / "gate.safe.pt")
            gate_nodes[str(node)] = list(gate_bundle.get("nodes", []))

        expert_files: Dict[str, str] = {}
        for node, payload in experts_payload_by_node.items():
            if not isinstance(payload, dict):
                raise ValueError(f"experts_by_node payload for {node} must be a dict")
            node_dir = experts_dir / node_dir_name(str(node))
            node_dir.mkdir(parents=True, exist_ok=True)
            out_path_node = node_dir / "leaf_experts.safe.pt"
            from .checkpointing import save_leaf_experts_safe

            save_leaf_experts_safe(payload, str(out_path_node))
            expert_files[str(node)] = str(Path("experts") / node_dir_name(str(node)) / "leaf_experts.safe.pt")

        routing = bundle.get("routing", {})
        node_to_leaves = routing.get("node_to_leaves") if isinstance(routing, dict) else None
        if isinstance(node_to_leaves, dict):
            leaf_set = {leaf for leaves in node_to_leaves.values() for leaf in leaves}
            if leaf_set:
                leaves = sorted(leaf_set)
        elif isinstance(bundle.get("leaves"), list):
            leaves = list(bundle["leaves"])

        meta = {
            "format": "moe_v4_bundle",
            "tree": hierarchy.get("tree"),
            "root_key": hierarchy.get("root_key", "node"),
            "height": height,
            "nodes_at_height": nodes,
            "leaves": leaves,
            "gates": gate_files,
            "gate_nodes": gate_nodes,
            "experts_by_node": expert_files,
            "expert_nodes": list(experts_payload_by_node.keys()),
            "routing": _coerce_meta(routing) if isinstance(routing, dict) else {},
            "hyperparams": metadata.get("hyperparams", {}),
            "feature_signature": metadata.get("feature_signature"),
            "gene_names": metadata.get("gene_names") or bundle.get("gene_names") or [],
            "excluded_leaves": metadata.get("excluded_leaves", []),
            "excluded_rationale": metadata.get("excluded_rationale"),
        }
        stacker_payload = bundle.get("stacker_raw") or bundle.get("stacker")
        if isinstance(stacker_payload, dict):
            stacker_dir = out_path / "stacker"
            stacker_dir.mkdir(parents=True, exist_ok=True)
            stacker_path = stacker_dir / "stacker.safe.pt"
            save_stacker_bundle(stacker_payload, str(stacker_path))
            meta["stacker"] = str(Path("stacker") / "stacker.safe.pt")
    else:
        if "gate" not in bundle:
            raise ValueError("bundle must include a 'gate' for saving")

        gate_path = out_path / "node_gate.safe.pt"
        experts_path = out_path / "leaf_experts.safe.pt"

        save_node_gate(bundle["gate"], str(gate_path))

        experts_payload = bundle.get("experts_raw")
        if experts_payload is None:
            experts_payload = bundle.get("experts")
        if not isinstance(experts_payload, dict):
            raise ValueError("bundle must include serializable leaf experts")

        from .checkpointing import save_leaf_experts_safe

        save_leaf_experts_safe(experts_payload, str(experts_path))

        meta = {
            "format": "moe_v4_bundle",
            "tree": hierarchy.get("tree"),
            "root_key": hierarchy.get("root_key", "node"),
            "height": height,
            "nodes_at_height": nodes,
            "leaves": leaves,
            "hyperparams": metadata.get("hyperparams", {}),
            "feature_signature": metadata.get("feature_signature"),
            "gene_names": metadata.get("gene_names") or bundle.get("gene_names") or [],
            "excluded_leaves": metadata.get("excluded_leaves", []),
            "excluded_rationale": metadata.get("excluded_rationale"),
        }

    import json
    from datetime import datetime, timezone

    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))
    return out_path


def load_v4_bundle(out_dir: str | Path, *, gene_names: Optional[List[str]] = None) -> ModelBundle:
    """Load a v4 bundle from safe artifacts under out_dir."""

    out_path = Path(out_dir)
    meta_path = out_path / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {out_path}")

    import json

    meta = json.loads(meta_path.read_text())
    tree = meta.get("tree")
    root_key = meta.get("root_key", "node")
    height = meta.get("height", 0)
    meta_nodes = meta.get("nodes_at_height") or meta.get("nodes", [])
    meta_leaves = meta.get("leaves") or []
    if tree is None:
        raise ValueError("meta.json missing hierarchy tree")

    subtype_to_path = walk_paths(tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    partitions = list(height_partitions(tree, root_key=root_key))

    routing_meta = meta.get("routing", {}) if isinstance(meta.get("routing", {}), dict) else {}
    global_gene_names = gene_names or meta.get("gene_names")
    if "gates" in meta or "experts_by_node" in meta:
        gate_files = meta.get("gates")
        expert_files = meta.get("experts_by_node")
        if not isinstance(gate_files, dict) or not isinstance(expert_files, dict):
            raise ValueError("meta.json missing hierarchical gate/experts file mapping")

        gates: Dict[str, Any] = {}
        for node, rel_path in gate_files.items():
            gate_path = out_path / rel_path
            if not gate_path.exists():
                raise FileNotFoundError(f"Missing gate checkpoint for {node}: {gate_path}")
            gates[str(node)] = load_gate_bundle(str(gate_path))

        if global_gene_names is not None:
            name_to_idx = {name: i for i, name in enumerate(global_gene_names)}
            for gate_bundle in gates.values():
                if gate_bundle.get("kind") != "ovr_cnn":
                    continue
                for child_entry in gate_bundle.get("child_models", []):
                    child_genes = child_entry.get("gene_names") or []
                    if not child_genes:
                        continue
                    missing = [g for g in child_genes if g not in name_to_idx]
                    if missing:
                        preview = ", ".join(missing[:8])
                        suffix = "..." if len(missing) > 8 else ""
                        raise ValueError(
                            "gate child expects genes not found in current gene_names: "
                            f"{preview}{suffix}"
                        )
                    child_entry["gene_idx"] = np.asarray([name_to_idx[g] for g in child_genes], dtype=int)

        node_to_leaves_routing = routing_meta.get("node_to_leaves", {})
        experts_by_node: Dict[str, Any] = {}
        for node, rel_path in expert_files.items():
            exp_path = out_path / rel_path
            if not exp_path.exists():
                raise FileNotFoundError(f"Missing expert checkpoint for {node}: {exp_path}")
            leaves = None
            if isinstance(node_to_leaves_routing, dict):
                leaves = node_to_leaves_routing.get(node)
            experts_by_node[str(node)] = load_leaf_experts_safe(
                str(exp_path),
                leaves=leaves,
                strict=True,
                gene_names=global_gene_names,
            )
        gate_nodes = meta.get("gate_nodes")
        if isinstance(gate_nodes, dict):
            for node, nodes_list in gate_nodes.items():
                if node in gates:
                    gates[node]["nodes"] = list(nodes_list)

        stacker = None
        stacker_rel = meta.get("stacker")
        if isinstance(stacker_rel, str):
            stacker_path = out_path / stacker_rel
            if stacker_path.exists():
                stacker = load_stacker_bundle(str(stacker_path))

        bundle: ModelBundle = {
            "hierarchy": {
                "tree": tree,
                "root_key": root_key,
                "subtype_to_path": subtype_to_path,
                "node_to_leaves": node_to_leaves,
                "partitions": partitions,
            },
            "gates": gates,
            "experts_by_node": experts_by_node,
            "routing": routing_meta,
            "metadata": {
                "height": height,
                "nodes": meta_nodes,
                "gate_nodes": meta.get("gate_nodes", {}),
                "expert_nodes": meta.get("expert_nodes", []),
                "hyperparams": meta.get("hyperparams", {}),
                "feature_signature": meta.get("feature_signature"),
                "gene_names": list(global_gene_names) if global_gene_names is not None else [],
                "excluded_leaves": meta.get("excluded_leaves", []),
                "excluded_rationale": meta.get("excluded_rationale"),
            },
            "nodes": meta_nodes,
            "leaves": meta_leaves,
            "gene_names": list(global_gene_names) if global_gene_names is not None else [],
        }
        if stacker is not None:
            bundle["stacker"] = stacker
        return bundle

    gate_path = out_path / "node_gate.safe.pt"
    experts_path = out_path / "leaf_experts.safe.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing node_gate.safe.pt in {out_path}")
    if not experts_path.exists():
        raise FileNotFoundError(f"Missing leaf_experts.safe.pt in {out_path}")

    gate = load_node_gate(str(gate_path))
    experts = load_leaf_experts_safe(
        str(experts_path),
        leaves=sorted(subtype_to_path.keys()),
        strict=True,
        gene_names=global_gene_names,
    )

    bundle: ModelBundle = {
        "hierarchy": {
            "tree": tree,
            "root_key": root_key,
            "subtype_to_path": subtype_to_path,
            "node_to_leaves": node_to_leaves,
            "partitions": partitions,
        },
        "experts": experts,
        "gate": gate,
        "metadata": {
            "height": height,
            "nodes": meta_nodes,
            "hyperparams": meta.get("hyperparams", {}),
            "feature_signature": meta.get("feature_signature"),
            "gene_names": list(global_gene_names) if global_gene_names is not None else [],
            "excluded_leaves": meta.get("excluded_leaves", []),
            "excluded_rationale": meta.get("excluded_rationale"),
        },
        "nodes": meta_nodes,
        "leaves": meta_leaves,
        "gene_names": list(global_gene_names) if global_gene_names is not None else [],
    }
    return bundle


def ensure_v4_bundle(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    height: int,
    *,
    out_dir: str | Path,
    root_key: str = "node",
    feature_names: Optional[List[str]] = None,
    feature_names_hint: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, Any]] = None,
    excluded_leaves: Optional[List[str]] = None,
    gate_reg: float = 1e-4,
    gate_lr: float = 0.1,
    gate_epochs: int = 200,
    expert_reg: float = 1e-4,
    expert_lr: float = 0.1,
    expert_epochs: int = 200,
    seed: int = 0,
) -> ModelBundle:
    """Load a compatible v4 bundle or retrain if the feature signature mismatches."""

    out_path = Path(out_dir)
    meta_path = out_path / "meta.json"
    signature = compute_feature_signature(
        X_train,
        feature_names=feature_names,
        feature_names_hint=feature_names_hint,
        preprocess_meta=preprocess_meta,
    )
    excluded_use = list(excluded_leaves or EXCLUDED_LEAVES)

    nodes_at_height = list(height_partitions(hierarchy_tree, root_key=root_key))[height]

    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        old_sig = meta.get("feature_signature") or {}
        diffs = _diff_signature(old_sig, signature)

        bundle = load_v4_bundle(out_path, gene_names=gene_names)
        reasons: List[str] = []
        if bundle.get("metadata", {}).get("height") != height:
            reasons.append("height mismatch")
        if not bundle.get("leaves"):
            reasons.append("missing leaves")
        gate_nodes = bundle.get("gate", {}).get("nodes", [])
        if height > 0 and (not gate_nodes or len(gate_nodes) <= 1):
            reasons.append("gate nodes degenerate")
        if gate_nodes and list(gate_nodes) != list(nodes_at_height):
            reasons.append("gate nodes do not match nodes_at_height")
        if diffs:
            reasons.append("feature signature mismatch")
        meta_excluded = meta.get("excluded_leaves", [])
        if sorted(meta_excluded) != sorted(excluded_use):
            reasons.append("excluded leaves mismatch")

        if not reasons:
            return bundle

        print("v4: invalid or incompatible artifacts; retraining v4 bundle.")
        for reason in reasons:
            print(f"  - {reason}")
        if diffs:
            for key, change in diffs.items():
                print(f"    {key}: {change['old']} -> {change['new']}")
    else:
        print("v4: no artifacts found; training new v4 bundle.")

    return train_node_moe(
        X_train,
        y_sub_train,
        hierarchy_tree,
        height,
        out_dir=out_path,
        root_key=root_key,
        gate_reg=gate_reg,
        gate_lr=gate_lr,
        gate_epochs=gate_epochs,
        expert_reg=expert_reg,
        expert_lr=expert_lr,
        expert_epochs=expert_epochs,
        seed=seed,
        feature_signature=signature,
    )


def ensure_hierarchical_v4_bundle(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    out_dir: str | Path,
    root_key: str = "node",
    subtype_to_family: Optional[Dict[str, str]] = None,
    default_gadneg_branch: str = "calb",
    excluded_leaves: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    feature_names_hint: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, Any]] = None,
    gate_reg: float = 1e-4,
    gate_lr: float = 0.1,
    gate_epochs: int = 200,
    expert_reg: float = 1e-4,
    expert_lr: float = 0.1,
    expert_epochs: int = 200,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    seed: int = 0,
    require_stacker: bool = False,
) -> ModelBundle:
    """Load or train a hierarchical v4 bundle with GAD/SOX6/CALB routing."""

    out_path = Path(out_dir)
    meta_path = out_path / "meta.json"
    signature = compute_feature_signature(
        X_train,
        feature_names=feature_names,
        feature_names_hint=feature_names_hint,
        preprocess_meta=preprocess_meta,
    )
    excluded_use = list(excluded_leaves or EXCLUDED_LEAVES)

    expected_root = [GAD_NEG_NODE, GAD_POS_NODE]
    expected_gad_neg = [CALB_NODE, SOX6_NODE]

    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        old_sig = meta.get("feature_signature") or {}
        diffs = _diff_signature(old_sig, signature)

        bundle = load_v4_bundle(out_path)
        reasons: List[str] = []
        gates = bundle.get("gates", {})
        experts_by_node = bundle.get("experts_by_node", {})
        routing = bundle.get("routing", {})

        if not gates or not experts_by_node:
            reasons.append("missing hierarchical gates/experts")

        gate_root = gates.get(ROOT_NODE)
        gate_gad_neg = gates.get(GAD_NEG_NODE)
        if gate_root is None or gate_gad_neg is None:
            reasons.append("missing required gates")
        else:
            if list(gate_root.get("nodes", [])) != expected_root:
                reasons.append("root gate nodes mismatch")
            if list(gate_gad_neg.get("nodes", [])) != expected_gad_neg:
                reasons.append("gad_neg gate nodes mismatch")

        for node in (GAD_POS_NODE, SOX6_NODE, CALB_NODE):
            if node not in experts_by_node:
                reasons.append(f"missing experts for {node}")

        leaves = bundle.get("leaves", [])
        if not leaves:
            reasons.append("missing leaves")

        node_to_leaves = routing.get("node_to_leaves") if isinstance(routing, dict) else None
        if not isinstance(node_to_leaves, dict):
            reasons.append("missing routing node_to_leaves")

        if diffs:
            reasons.append("feature signature mismatch")
        if require_stacker:
            stacker = bundle.get("stacker")
            if stacker is None:
                reasons.append("missing stacker")
            else:
                feature_spec = stacker.get("feature_spec", {}) if isinstance(stacker, dict) else {}
                gate_order = feature_spec.get("gate_order") or list(gates.keys())
                node_order = feature_spec.get("node_order") or list(experts_by_node.keys())
                leaf_order = feature_spec.get("leaf_order") or stacker.get("leaf_order")
                missing_gates = [gate for gate in gate_order if gate not in gates]
                missing_nodes = [node for node in node_order if node not in experts_by_node]
                if missing_gates or missing_nodes:
                    reasons.append("stacker gate/node order mismatch")
                if leaf_order and list(leaf_order) != list(leaves):
                    reasons.append("stacker leaf order mismatch")
        meta_excluded = bundle.get("metadata", {}).get("excluded_leaves", [])
        if sorted(meta_excluded) != sorted(excluded_use):
            reasons.append("excluded leaves mismatch")

        if not reasons:
            return bundle

        print("v4: invalid or incompatible hierarchical artifacts; retraining.")
        for reason in reasons:
            print(f"  - {reason}")
        if diffs:
            for key, change in diffs.items():
                print(f"    {key}: {change['old']} -> {change['new']}")
    else:
        print("v4: no hierarchical artifacts found; training new bundle.")

    return train_hierarchical_v4(
        X_train,
        y_sub_train,
        hierarchy_tree,
        out_dir=out_path,
        root_key=root_key,
        subtype_to_family=subtype_to_family,
        default_gadneg_branch=default_gadneg_branch,
        excluded_leaves=excluded_use,
        gate_reg=gate_reg,
        gate_lr=gate_lr,
        gate_epochs=gate_epochs,
        expert_reg=expert_reg,
        expert_lr=expert_lr,
        expert_epochs=expert_epochs,
        stacker_reg=stacker_reg,
        stacker_lr=stacker_lr,
        stacker_epochs=stacker_epochs,
        train_stacker=require_stacker,
        seed=seed,
        feature_signature=signature,
    )


def train_node_moe(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    height: int,
    *,
    out_dir: str | Path | None = None,
    root_key: str = "node",
    feature_signature: Optional[Dict[str, Any]] = None,
    gate_reg: float = 1e-4,
    gate_lr: float = 0.1,
    gate_epochs: int = 200,
    expert_reg: float = 1e-4,
    expert_lr: float = 0.1,
    expert_epochs: int = 200,
    seed: int = 0,
) -> ModelBundle:
    """Train a v4 node MoE (gate + leaf experts) and save safe artifacts."""

    subtype_to_path = walk_paths(hierarchy_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    partitions = list(height_partitions(hierarchy_tree, root_key=root_key))

    if height < 0 or height >= len(partitions):
        raise ValueError(f"height out of range: {height}")
    nodes_at_height = partitions[height]
    leaves = sorted(subtype_to_path.keys())

    y_sub = np.asarray(y_sub_train).astype(str)
    if y_sub.shape[0] != np.asarray(X_train).shape[0]:
        raise ValueError("y_sub_train length must match X_train rows")

    node_to_idx = {n: i for i, n in enumerate(nodes_at_height)}
    y_node_train: List[int] = []
    missing_subtypes: List[str] = []
    for st in y_sub:
        path = subtype_to_path.get(str(st))
        if path is None:
            missing_subtypes.append(str(st))
            continue
        node = "|".join(path[: height + 1]) if height < len(path) else "|".join(path)
        if node not in node_to_idx:
            raise ValueError(f"Node {node} not found in height partition {height}")
        y_node_train.append(node_to_idx[node])

    if missing_subtypes:
        raise ValueError(f"Unknown subtypes in y_sub_train: {sorted(set(missing_subtypes))[:5]}")

    gate_bundle = train_node_gate(
        X_train,
        np.asarray(y_node_train),
        nodes_at_height,
        reg=gate_reg,
        lr=gate_lr,
        epochs=gate_epochs,
        seed=seed,
    )
    gate_bundle["nodes"] = list(nodes_at_height)
    gate_bundle["height"] = int(height)

    experts_payload = train_leaf_experts_ovr(
        X_train,
        y_sub_train,
        leaves,
        out_dir=out_dir,
        save=False,
        reg=expert_reg,
        lr=expert_lr,
        epochs=expert_epochs,
        seed=seed,
    )

    base = Path(__file__).resolve().parents[4]
    model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")

    bundle: ModelBundle = {
        "hierarchy": {
            "tree": hierarchy_tree,
            "root_key": root_key,
            "subtype_to_path": subtype_to_path,
            "node_to_leaves": node_to_leaves,
            "partitions": partitions,
        },
        "gate": gate_bundle,
        "metadata": {
            "height": height,
            "nodes": nodes_at_height,
            "leaves": leaves,
            "hyperparams": {
                "gate_reg": gate_reg,
                "gate_lr": gate_lr,
                "gate_epochs": gate_epochs,
                "expert_reg": expert_reg,
                "expert_lr": expert_lr,
                "expert_epochs": expert_epochs,
                "seed": seed,
            },
            "feature_signature": feature_signature,
        },
        "experts_raw": experts_payload,
        "leaves": leaves,
    }
    save_v4_bundle(bundle, model_dir)
    bundle["experts"] = load_leaf_experts_safe(str(model_dir / "leaf_experts.safe.pt"), leaves=leaves, strict=True)
    return bundle


def train_hierarchical_v4(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    root_key: str = "node",
    subtype_to_family: Optional[Dict[str, str]] = None,
    default_gadneg_branch: str = "calb",
    excluded_leaves: Optional[List[str]] = None,
    feature_signature: Optional[Dict[str, Any]] = None,
    gate_reg: float = 1e-4,
    gate_lr: float = 0.1,
    gate_epochs: int = 200,
    expert_reg: float = 1e-4,
    expert_lr: float = 0.1,
    expert_epochs: int = 200,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    train_stacker: bool = False,
    seed: int = 0,
) -> ModelBundle:
    """Train a hierarchical MoE v4 with GAD and SOX6/CALB routing."""

    X = np.asarray(X_train, dtype=float)
    y_sub = np.asarray(y_sub_train).astype(str)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    if y_sub.shape[0] != X.shape[0]:
        raise ValueError("y_sub_train length must match X_train rows")

    excluded_use = list(excluded_leaves or EXCLUDED_LEAVES)
    subtype_to_path = walk_paths(hierarchy_tree)
    all_leaves = sorted(subtype_to_path.keys())
    leaves, removed = filter_excluded_leaves(all_leaves, excluded_use)
    if not leaves:
        raise ValueError("All leaves excluded; cannot train hierarchical v4.")
    if removed:
        print(f"v4: excluded leaves: {sorted(removed)}")
    node_to_leaves, unknown = build_node_to_leaves(
        leaves,
        subtype_to_family=subtype_to_family,
        default_gadneg_branch=default_gadneg_branch,
    )

    keep_mask = ~np.isin(y_sub, np.asarray(excluded_use, dtype=object))
    if not np.any(keep_mask):
        raise ValueError("Training data contains only excluded leaves.")
    X = X[keep_mask]
    y_sub = y_sub[keep_mask]

    y_gad = np.array(
        [1.0 if is_gad_positive(st, subtype_to_family=subtype_to_family) else 0.0 for st in y_sub],
        dtype=float,
    )
    gate_root = train_binary_gate(
        X,
        y_gad,
        node_name=ROOT_NODE,
        label_names=("gad_neg", "gad_pos"),
        reg=gate_reg,
        lr=gate_lr,
        epochs=gate_epochs,
        seed=seed,
    )

    gad_neg_mask = y_gad == 0.0
    if not np.any(gad_neg_mask):
        raise ValueError("No gad_neg samples found; cannot train gad_neg gate.")

    y_gadneg = np.array(
        [
            1.0
            if assign_gadneg_branch(
                st, subtype_to_family=subtype_to_family, default_branch=default_gadneg_branch
            )
            == "sox6"
            else 0.0
            for st in y_sub[gad_neg_mask]
        ],
        dtype=float,
    )
    gate_gad_neg = train_binary_gate(
        X[gad_neg_mask],
        y_gadneg,
        node_name=GAD_NEG_NODE,
        label_names=("calb", "sox6"),
        reg=gate_reg,
        lr=gate_lr,
        epochs=gate_epochs,
        seed=seed,
    )

    base = Path(__file__).resolve().parents[4]
    model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")

    experts_raw_by_node: Dict[str, Dict[str, Any]] = {}
    for node_name in (GAD_POS_NODE, SOX6_NODE, CALB_NODE):
        experts_raw_by_node[node_name] = train_node_specific_leaf_experts(
            X,
            y_sub,
            node_to_leaves,
            node_name,
            out_dir=model_dir,
            save=True,
            reg=expert_reg,
            lr=expert_lr,
            epochs=expert_epochs,
            seed=seed,
        )

    stacker_raw = None
    if train_stacker:
        gate_outputs = _compute_gate_outputs(X, {ROOT_NODE: gate_root, GAD_NEG_NODE: gate_gad_neg})
        node_outputs = _compute_node_outputs(X, experts_raw_by_node, node_to_leaves)
        node_order = [GAD_POS_NODE, SOX6_NODE, CALB_NODE]
        for node_name in node_order:
            node_out = node_outputs.get(node_name, {})
            if not all(key in node_out for key in ("logits", "probs", "leaf_order")):
                raise ValueError(f"node_outputs missing keys for {node_name}: {list(node_out.keys())}")
        X_meta, feature_spec = build_stacker_features(
            gate_outputs,
            node_outputs,
            leaves,
            node_order=node_order,
            feature_spec=None,
        )
        stacker_raw = train_leaf_stacker(
            X_meta,
            y_sub,
            leaves,
            reg=stacker_reg,
            lr=stacker_lr,
            epochs=stacker_epochs,
            seed=seed,
            feature_spec=feature_spec,
        )

    bundle: ModelBundle = {
        "hierarchy": {
            "tree": hierarchy_tree,
            "root_key": root_key,
            "subtype_to_path": subtype_to_path,
            "node_to_leaves": node_to_leaves,
            "partitions": list(height_partitions(hierarchy_tree, root_key=root_key)),
        },
        "gates": {ROOT_NODE: gate_root, GAD_NEG_NODE: gate_gad_neg},
        "experts_raw_by_node": experts_raw_by_node,
        "metadata": {
            "height": None,
            "leaves": leaves,
            "excluded_leaves": excluded_use,
            "excluded_rationale": EXCLUSION_RATIONALE,
            "hyperparams": {
                "gate_reg": gate_reg,
                "gate_lr": gate_lr,
                "gate_epochs": gate_epochs,
                "expert_reg": expert_reg,
                "expert_lr": expert_lr,
                "expert_epochs": expert_epochs,
                "seed": seed,
            },
            "feature_signature": feature_signature,
        },
        "routing": {
            "root_node": ROOT_NODE,
            "gad_pos_node": GAD_POS_NODE,
            "gad_neg_node": GAD_NEG_NODE,
            "sox6_node": SOX6_NODE,
            "calb_node": CALB_NODE,
            "node_to_leaves": node_to_leaves,
            "unknown_leaves": unknown,
            "default_gadneg_branch": default_gadneg_branch,
            "excluded_leaves": excluded_use,
            "excluded_removed": removed,
        },
        "leaves": leaves,
    }
    if stacker_raw is not None:
        bundle["stacker_raw"] = stacker_raw

    save_v4_bundle(bundle, model_dir)
    return load_v4_bundle(model_dir)


def ensure_tree_v4_bundle(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    hierarchy_tree: Dict[str, Any],
    *,
    root_key: str = "node",
    gate_nodes: Optional[List[str]] = None,
    stop_nodes: Optional[List[str]] = None,
    gene_names: Optional[List[str]] = None,
    out_dir: str | Path | None = None,
    k_genes: int = 256,
    selector: str = "f_classif",
    gate_lr: float = 1e-3,
    gate_epochs: int = 30,
    gate_batch_size: int = 256,
    expert_reg: float = 1e-4,
    expert_lr: float = 0.1,
    expert_epochs: int = 200,
    seed: int = 0,
    require_stacker: bool = True,
    stacker_reg: float = 1e-4,
    stacker_lr: float = 0.1,
    stacker_epochs: int = 200,
    excluded_leaves: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    feature_names_hint: Optional[str] = None,
    preprocess_meta: Optional[Dict[str, Any]] = None,
) -> ModelBundle:
    """Train/load a generalized tree-based v4 bundle with OVR CNN gates."""

    excluded = excluded_leaves or EXCLUDED_LEAVES
    if gene_names is None:
        raise ValueError("gene_names must be provided for CNN gates")

    tree = build_tree_index_from_hierarchy(
        hierarchy_tree,
        root_key=root_key,
        excluded_leaves=excluded,
    )

    if gate_nodes is None:
        gate_nodes = [node for node in internal_nodes(tree) if len(tree.get_children(node)) > 1]
    if stop_nodes is None:
        stop_nodes = [node for node in gate_nodes if all(tree.is_leaf(c) for c in tree.get_children(node))]

    base = Path(__file__).resolve().parents[4]
    model_dir = Path(out_dir) if out_dir is not None else (base / "models" / "moe_v4")
    model_dir.mkdir(parents=True, exist_ok=True)

    gates: Dict[str, Any] = {}
    for node in gate_nodes:
        gate_path = model_dir / "gates" / tree_node_dir_name(node) / "gate.safe.pt"
        if gate_path.exists():
            gates[node] = load_gate_bundle(str(gate_path))
        else:
            gates[node] = train_node_ovr_gate(
                node,
                tree.get_children(node),
                X_train,
                y_sub_train,
                gene_names,
                tree=tree,
                out_dir=model_dir,
                k_genes=k_genes,
                selector=selector,
                lr=gate_lr,
                epochs=gate_epochs,
                batch_size=gate_batch_size,
                seed=seed,
            )

    experts_by_node: Dict[str, Dict[str, Any]] = {}
    for node in stop_nodes:
        expert_path = model_dir / "experts" / tree_node_dir_name(node) / "leaf_experts.safe.pt"
        if expert_path.exists():
            try:
                experts_by_node[node] = load_leaf_experts_safe(
                    str(expert_path),
                    gene_names=gene_names,
                )
            except ValueError as exc:
                msg = str(exc)
                if gene_names and ("missing gene metadata" in msg or "expects" in msg):
                    print(f"v4: expert artifact incompatible for {node}; retraining. ({msg})")
                    experts_by_node[node] = train_node_specific_leaf_experts(
                        X_train,
                        y_sub_train,
                        tree.node_to_leaves,
                        node,
                        out_dir=model_dir,
                        save=True,
                        gene_names=gene_names,
                        reg=expert_reg,
                        lr=expert_lr,
                        epochs=expert_epochs,
                        seed=seed,
                    )
                else:
                    raise
        else:
            experts_by_node[node] = train_node_specific_leaf_experts(
                X_train,
                y_sub_train,
                tree.node_to_leaves,
                node,
                out_dir=model_dir,
                save=True,
                gene_names=gene_names,
                reg=expert_reg,
                lr=expert_lr,
                epochs=expert_epochs,
                seed=seed,
            )

    stacker_raw = None
    if require_stacker:
        stacker_path = model_dir / "stacker" / "stacker.safe.pt"
        if stacker_path.exists():
            stacker_raw = load_stacker_bundle(str(stacker_path))
            feature_spec = stacker_raw.get("feature_spec", {}) if isinstance(stacker_raw, dict) else {}
            gate_order = feature_spec.get("gate_order") or list(gates.keys())
            node_order = feature_spec.get("node_order") or list(experts_by_node.keys())
            leaf_order = feature_spec.get("leaf_order") or stacker_raw.get("leaf_order")

            missing_gates = [gate for gate in gate_order if gate not in gates]
            missing_nodes = [node for node in node_order if node not in experts_by_node]
            mismatch_reasons = []
            if missing_gates:
                mismatch_reasons.append(f"missing gate outputs for {', '.join(missing_gates)}")
            if missing_nodes:
                mismatch_reasons.append(f"missing node outputs for {', '.join(missing_nodes)}")
            if leaf_order and list(leaf_order) != list(tree.leaves):
                mismatch_reasons.append("leaf order mismatch")
            if mismatch_reasons:
                print("v4: stacker artifacts incompatible with current gates/experts; retraining stacker.")
                for reason in mismatch_reasons:
                    print(f"  - {reason}")
                stacker_raw = None

        if stacker_raw is None:
            gate_outputs = _compute_gate_outputs(X_train, gates, gene_names=gene_names)
            node_outputs = _compute_node_outputs(
                X_train,
                experts_by_node,
                tree.node_to_leaves,
                gene_names=gene_names,
            )
            X_meta, feature_spec = build_stacker_features(
                gate_outputs,
                node_outputs,
                tree.leaves,
                node_order=list(experts_by_node.keys()),
            )
            stacker_raw = train_leaf_stacker(
                X_meta,
                y_sub_train,
                tree.leaves,
                reg=stacker_reg,
                lr=stacker_lr,
                epochs=stacker_epochs,
                seed=seed,
                feature_spec=feature_spec,
            )
            stacker_dir = model_dir / "stacker"
            stacker_dir.mkdir(parents=True, exist_ok=True)
            save_stacker_bundle(stacker_raw, str(stacker_dir / "stacker.safe.pt"))

    feature_signature = compute_feature_signature(
        X_train,
        feature_names=feature_names or gene_names,
        feature_names_hint=feature_names_hint,
        preprocess_meta=preprocess_meta,
    )

    subtype_to_path = tree.subtype_to_path
    hierarchy = {
        "tree": hierarchy_tree,
        "root_key": root_key,
        "subtype_to_path": subtype_to_path,
        "node_to_leaves": tree.node_to_leaves,
        "partitions": list(height_partitions(hierarchy_tree, root_key=root_key)),
    }

    bundle: ModelBundle = {
        "hierarchy": hierarchy,
        "tree_index": tree,
        "gates": gates,
        "experts_by_node": experts_by_node,
        "metadata": {
            "height": None,
            "nodes": gate_nodes,
            "leaves": tree.leaves,
            "excluded_leaves": excluded,
            "excluded_rationale": EXCLUSION_RATIONALE,
            "gene_names": list(gene_names) if gene_names else [],
            "hyperparams": {
                "gate_lr": gate_lr,
                "gate_epochs": gate_epochs,
                "gate_batch_size": gate_batch_size,
                "expert_reg": expert_reg,
                "expert_lr": expert_lr,
                "expert_epochs": expert_epochs,
                "stacker_reg": stacker_reg,
                "stacker_lr": stacker_lr,
                "stacker_epochs": stacker_epochs,
                "seed": seed,
            },
            "feature_signature": feature_signature,
        },
        "routing": {
            "node_to_leaves": tree.node_to_leaves,
            "gate_nodes": gate_nodes,
            "stop_nodes": stop_nodes,
            "excluded_leaves": excluded,
        },
        "leaves": tree.leaves,
        "gene_names": list(gene_names) if gene_names else [],
    }
    if stacker_raw is not None:
        bundle["stacker_raw"] = stacker_raw

    save_v4_bundle(bundle, model_dir)
    loaded = load_v4_bundle(model_dir, gene_names=gene_names)
    loaded["tree_index"] = tree
    return loaded


def _compute_gate_outputs(
    X: np.ndarray,
    gates: Dict[str, Any],
    *,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    gate_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for gate_name, gate_bundle in gates.items():
        kind = gate_bundle.get("kind")
        if kind == "binary_logreg":
            logits = predict_binary_gate(gate_bundle, X, as_prob=False, gene_names=gene_names)
            probs = predict_binary_gate(gate_bundle, X, as_prob=True, gene_names=gene_names)
            nodes = list(gate_bundle.get("nodes", []))
        elif kind == "ovr_cnn":
            logits = predict_node_child_probs(gate_bundle, X, as_logits=True, gene_names=gene_names)
            probs = predict_node_child_probs(gate_bundle, X, as_logits=False, gene_names=gene_names)
            nodes = list(gate_bundle.get("children", []))
        else:
            logits = predict_node_scores(gate_bundle, X, gene_names=gene_names)
            probs = _softmax_rows(logits)
            nodes = list(gate_bundle.get("nodes", []))

        gate_outputs[gate_name] = {
            "logits": logits,
            "probs": probs,
            "nodes": nodes,
        }
    return gate_outputs


def _compute_node_outputs(
    X: np.ndarray,
    experts_by_node: Dict[str, Dict[str, Any]],
    node_to_leaves: Dict[str, List[str]],
    *,
    fill_value: float = -np.inf,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    node_outputs: Dict[str, Dict[str, Any]] = {}
    gene_idx_map = {name: i for i, name in enumerate(gene_names)} if gene_names else None
    for node_name, experts in experts_by_node.items():
        leaf_order = list(node_to_leaves.get(node_name, list(experts.keys())))
        use_raw = bool(experts) and all(
            isinstance(v, dict) and "weight" in v for v in experts.values()
        )
        if use_raw:
            scores = np.full((X.shape[0], len(leaf_order)), fill_value, dtype=float)
            for j, leaf in enumerate(leaf_order):
                params = experts.get(leaf, {})
                w = np.asarray(params.get("weight", []), dtype=float).reshape(-1)
                if w.size == 0:
                    raise ValueError(f"Missing weights for node {node_name} leaf {leaf}")
                b = float(np.asarray(params.get("bias", 0.0)).reshape(-1)[0])
                gene_idx = None
                if isinstance(params, dict) and "gene_idx" in params:
                    gene_idx = np.asarray(params.get("gene_idx"), dtype=int)
                elif gene_idx_map is not None and (params.get("genes") or params.get("gene_names")):
                    gene_list = list(params.get("genes") or params.get("gene_names") or [])
                    missing = [g for g in gene_list if g not in gene_idx_map]
                    if missing:
                        preview = ", ".join(missing[:8])
                        suffix = "..." if len(missing) > 8 else ""
                        raise ValueError(
                            f"expert '{leaf}' expects genes not found in current gene_names: {preview}{suffix}"
                        )
                    gene_idx = np.asarray([gene_idx_map[g] for g in gene_list], dtype=int)
                X_use = np.asarray(X, dtype=float)
                if gene_idx is not None:
                    X_use = X_use[:, gene_idx]
                if X_use.shape[1] != w.shape[0]:
                    raise ValueError(
                        f"expert '{leaf}' expects {w.shape[0]} features, got {X_use.shape[1]}"
                    )
                scores[:, j] = X_use @ w + b
        else:
            scores = score_leaf_experts(
                X,
                experts,
                leaf_candidates=leaf_order,
                leaf_order=leaf_order,
                fill_value=fill_value,
            )["scores"]
        safe_scores = np.where(np.isfinite(scores), scores, -1e9)
        probs = _softmax_rows(safe_scores)
        node_outputs[node_name] = {
            "scores": scores,
            "logits": scores,
            "probs": probs,
        "leaf_order": leaf_order,
    }
    return node_outputs


def _normalize_child_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    denom = probs.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return probs / denom


def _gate_child_probs(
    tree: TreeIndex,
    gate_outputs: Dict[str, Dict[str, np.ndarray]],
    node: str,
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    children = tree.get_children(node)
    if stop_nodes and node in stop_nodes:
        if n_samples is None:
            n_samples = next(iter(gate_outputs.values()))["probs"].shape[0] if gate_outputs else 0
        return [node], np.ones((n_samples, 1), dtype=float)
    if stop_depth is not None and tree.node_depth(node) >= stop_depth:
        if n_samples is None:
            n_samples = next(iter(gate_outputs.values()))["probs"].shape[0] if gate_outputs else 0
        return [node], np.ones((n_samples, 1), dtype=float)
    gate = gate_outputs.get(node)
    if gate is None:
        raise ValueError(f"Missing gate outputs for node '{node}'")
    probs = np.asarray(gate.get("probs"))
    gate_nodes = list(gate.get("nodes", []))
    if gate_nodes and children and gate_nodes != children:
        idx = [gate_nodes.index(child) for child in children]
        probs = probs[:, idx]
    probs = _normalize_child_probs(probs)
    return children, probs


def _tree_index_from_bundle(
    bundle: ModelBundle, *, excluded_leaves: Optional[List[str]] = None
) -> TreeIndex:
    if "tree_index" in bundle and isinstance(bundle["tree_index"], TreeIndex):
        return bundle["tree_index"]
    hierarchy = bundle.get("hierarchy", {})
    subtype_to_path = hierarchy.get("subtype_to_path")
    tree = hierarchy.get("tree")
    root_key = hierarchy.get("root_key", "node")
    excluded = excluded_leaves or bundle.get("metadata", {}).get("excluded_leaves", [])
    if isinstance(subtype_to_path, dict):
        from .tree import build_tree_index_from_subtype_paths

        return build_tree_index_from_subtype_paths(
            subtype_to_path,
            root_key=root_key,
            excluded_leaves=excluded,
        )
    if isinstance(tree, dict):
        return build_tree_index_from_hierarchy(
            tree,
            root_key=root_key,
            excluded_leaves=excluded,
        )
    raise ValueError("bundle missing hierarchy tree/subtype_to_path for tree routing")


def _gated_nodes_for_routing(
    tree: TreeIndex,
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
) -> List[str]:
    stop_set = set(stop_nodes or [])
    gated: List[str] = []
    for node in internal_nodes(tree):
        if node in stop_set:
            continue
        if stop_depth is not None and tree.node_depth(node) >= stop_depth:
            continue
        if tree.get_children(node):
            gated.append(node)
    return gated


def route_tree_hard(
    X: np.ndarray,
    tree: TreeIndex,
    gates: Dict[str, Any],
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Hard routing over a tree using argmax child selection at each node."""

    gate_outputs = _compute_gate_outputs(X, gates, gene_names=gene_names)
    n_samples = X.shape[0]
    selected_paths: List[List[str]] = []
    final_nodes: List[str] = []
    final_leaves: List[str | None] = []

    for i in range(n_samples):
        node = tree.root
        path = [node]
        depth = 0
        while True:
            if stop_nodes and node in stop_nodes:
                final_nodes.append(node)
                final_leaves.append(None)
                break
            if stop_depth is not None and depth >= stop_depth:
                final_nodes.append(node)
                final_leaves.append(None)
                break

            children = tree.get_children(node)
            if not children:
                final_nodes.append(node)
                final_leaves.append(None)
                break

            child_list, probs = _gate_child_probs(
                tree,
                gate_outputs,
                node,
                stop_nodes=stop_nodes,
                stop_depth=stop_depth,
                n_samples=n_samples,
            )
            p = probs[i]
            child = child_list[int(np.argmax(p))]
            path.append(child)
            if tree.is_leaf(child):
                final_nodes.append(child)
                final_leaves.append(child)
                break
            node = child
            depth += 1

        selected_paths.append(path)

    selected_nodes = [[node] for node in final_nodes]
    return {
        "selected_paths": selected_paths,
        "final_nodes": final_nodes,
        "final_leaves": final_leaves,
        "selected_nodes": selected_nodes,
        "gate_outputs": gate_outputs,
    }


def route_tree_soft(
    X: np.ndarray,
    tree: TreeIndex,
    gates: Dict[str, Any],
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Soft routing by propagating normalized child probabilities."""

    gate_outputs = _compute_gate_outputs(X, gates, gene_names=gene_names)
    n_samples = X.shape[0]

    terminal_nodes: List[str]
    if stop_nodes:
        terminal_nodes = list(stop_nodes)
    elif stop_depth is not None:
        terminal_nodes = [
            node
            for node in tree.node_to_children.keys()
            if tree.node_depth(node) == stop_depth
        ]
    else:
        terminal_nodes = list(tree.leaves)

    log_weights = np.full((n_samples, len(terminal_nodes)), -np.inf, dtype=float)
    for j, term in enumerate(terminal_nodes):
        if tree.is_leaf(term):
            edges = leaf_edges(tree, term)
        else:
            edges = node_edges(tree, term)
        log_p = np.zeros((n_samples,), dtype=float)
        for parent, child in edges:
            children, probs = _gate_child_probs(
                tree,
                gate_outputs,
                parent,
                stop_nodes=stop_nodes,
                stop_depth=stop_depth,
                n_samples=n_samples,
            )
            idx = children.index(child)
            log_p += np.log(np.maximum(probs[:, idx], 1e-12))
        log_weights[:, j] = log_p

    out: Dict[str, Any] = {
        "gate_outputs": gate_outputs,
        "terminal_nodes": terminal_nodes,
        "log_weights": log_weights,
    }
    if terminal_nodes and terminal_nodes[0] in tree.leaves:
        log_norm = _logsumexp(log_weights, axis=1, keepdims=True)
        out["P_leaf"] = np.exp(log_weights - log_norm)
        out["leaves"] = terminal_nodes
    else:
        weights = np.exp(log_weights)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        out["node_weights"] = weights

    hard = route_tree_hard(
        X,
        tree,
        gates,
        stop_nodes=stop_nodes,
        stop_depth=stop_depth,
        gene_names=gene_names,
    )
    out["selected_paths"] = hard["selected_paths"]
    out["final_nodes"] = hard["final_nodes"]
    out["selected_nodes"] = hard["selected_nodes"]
    return out


def train_tree_gates(
    X_train: np.ndarray,
    y_sub_train: np.ndarray,
    tree: TreeIndex,
    gate_nodes: Optional[List[str]],
    *,
    gene_names: List[str],
    out_dir: str | Path | None = None,
    k_genes: int = 256,
    selector: str = "f_classif",
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train OVR CNN gates for the specified internal nodes."""

    if gate_nodes is None:
        gate_nodes = [node for node in internal_nodes(tree) if len(tree.get_children(node)) > 1]

    gates: Dict[str, Any] = {}
    for node in gate_nodes:
        children = tree.get_children(node)
        if len(children) < 2:
            continue
        gate = train_node_ovr_gate(
            node,
            children,
            X_train,
            y_sub_train,
            gene_names,
            tree=tree,
            out_dir=out_dir,
            k_genes=k_genes,
            selector=selector,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        gates[node] = gate
    return gates


def load_tree_gates(out_dir: str | Path, gate_nodes: List[str]) -> Dict[str, Any]:
    """Load OVR gate bundles for the specified nodes."""

    base = Path(out_dir)
    gates: Dict[str, Any] = {}
    for node in gate_nodes:
        gate_path = base / "gates" / tree_node_dir_name(node) / "gate.safe.pt"
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing gate checkpoint: {gate_path}")
        gates[node] = load_gate_bundle(str(gate_path))
    return gates


def predict_tree_hard(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Hard routing + node-specific experts."""

    tree = _tree_index_from_bundle(model_bundle)
    gates = model_bundle.get("gates", {})
    experts_by_node = model_bundle.get("experts_by_node", {})
    node_to_leaves = tree.node_to_leaves

    gene_names_use = gene_names or model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    routing = route_tree_hard(
        X,
        tree,
        gates,
        stop_nodes=stop_nodes,
        stop_depth=stop_depth,
        gene_names=gene_names_use,
    )
    final_nodes = routing["final_nodes"]

    leaves_all = list(leaf_order or tree.leaves)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves_all)}
    P_out = np.full((X.shape[0], len(leaves_all)), fill_value, dtype=float)

    if not experts_by_node or (stop_nodes is None and stop_depth is None):
        for i, leaf in enumerate(routing["final_leaves"]):
            if leaf is None:
                continue
            col = leaf_to_idx.get(leaf)
            if col is not None:
                P_out[i, col] = 1.0
        return {
            "P_sub": P_out,
            "combined_scores": P_out,
            "selected_paths": routing["selected_paths"],
            "selected_nodes": routing["selected_nodes"],
            "final_nodes": final_nodes,
            "gate_outputs": routing["gate_outputs"],
            "leaves": leaves_all,
        }

    experts_flat: Dict[str, Any] = {}
    for node, experts in experts_by_node.items():
        for leaf, expert in experts.items():
            experts_flat[str(leaf)] = expert

    leaf_candidates = [
        node_to_leaves.get(node, [node]) if not tree.is_leaf(node) else [node] for node in final_nodes
    ]
    leaf_scores_out = score_leaf_experts(
        X,
        experts_flat,
        leaf_candidates=leaf_candidates,
        leaf_order=leaves_all,
        fill_value=fill_value,
    )

    return {
        "P_sub": leaf_scores_out["scores"],
        "combined_scores": leaf_scores_out["scores"],
        "selected_paths": routing["selected_paths"],
        "selected_nodes": routing["selected_nodes"],
        "final_nodes": final_nodes,
        "leaf_candidates": leaf_candidates,
        "gate_outputs": routing["gate_outputs"],
        "leaves": leaves_all,
    }


def predict_tree_soft(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Soft routing + node-specific experts (log-space mixture)."""

    tree = _tree_index_from_bundle(model_bundle)
    gates = model_bundle.get("gates", {})
    experts_by_node = model_bundle.get("experts_by_node", {})
    node_to_leaves = tree.node_to_leaves

    gene_names_use = gene_names or model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    routing = route_tree_soft(
        X,
        tree,
        gates,
        stop_nodes=stop_nodes,
        stop_depth=stop_depth,
        gene_names=gene_names_use,
    )
    leaves_all = list(leaf_order or tree.leaves)
    leaves_all = [leaf for leaf in leaves_all if leaf in set(tree.leaves)]

    if "P_leaf" in routing and not experts_by_node:
        return {
            "P_sub": routing["P_leaf"],
            "combined_scores": routing["P_leaf"],
            "selected_paths": routing["selected_paths"],
            "selected_nodes": routing["selected_nodes"],
            "final_nodes": routing["final_nodes"],
            "gate_outputs": routing["gate_outputs"],
            "leaves": routing.get("leaves", leaves_all),
        }

    if "node_weights" not in routing:
        raise ValueError("Soft routing requires stop_nodes/stop_depth to combine experts")

    node_outputs = _compute_node_outputs(
        X,
        experts_by_node,
        node_to_leaves,
        fill_value=fill_value,
        gene_names=gene_names_use,
    )
    node_weights = routing["node_weights"]
    terminal_nodes = routing["terminal_nodes"]

    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves_all)}
    combined_log = np.full((X.shape[0], len(leaves_all)), -np.inf, dtype=float)

    for j, node in enumerate(terminal_nodes):
        weight = node_weights[:, j]
        log_weight = np.log(np.maximum(weight, 1e-12))
        node_scores = node_outputs[node]["scores"]
        node_log_probs = _log_softmax(node_scores)
        node_outputs[node]["probs"] = np.exp(node_log_probs)
        for k, leaf in enumerate(node_outputs[node]["leaf_order"]):
            col = leaf_to_idx.get(leaf)
            if col is not None:
                combined_log[:, col] = np.logaddexp(combined_log[:, col], log_weight + node_log_probs[:, k])

    log_norm = _logsumexp(combined_log, axis=1, keepdims=True)
    combined = np.exp(combined_log - log_norm)

    return {
        "P_sub": combined,
        "combined_scores": combined,
        "selected_paths": routing["selected_paths"],
        "selected_nodes": routing["selected_nodes"],
        "final_nodes": routing["final_nodes"],
        "node_weights": {"nodes": terminal_nodes, "weights": node_weights},
        "gate_outputs": routing["gate_outputs"],
        "node_outputs": node_outputs,
        "leaves": leaves_all,
    }


def predict_tree_stacked(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    leaf_order: Optional[List[str]] = None,
    stop_nodes: Optional[List[str]] = None,
    stop_depth: Optional[int] = None,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Stacked classifier on top of gate + node expert outputs."""

    if "stacker" not in model_bundle:
        raise ValueError("model_bundle missing stacker")

    tree = _tree_index_from_bundle(model_bundle)
    gates = model_bundle.get("gates", {})
    experts_by_node = model_bundle.get("experts_by_node", {})
    node_to_leaves = tree.node_to_leaves
    leaves_all = list(leaf_order or tree.leaves)

    gene_names_use = gene_names or model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    gated_nodes = _gated_nodes_for_routing(tree, stop_nodes=stop_nodes, stop_depth=stop_depth)
    missing_gates = [node for node in gated_nodes if node not in gates]
    if missing_gates:
        raise ValueError(f"Missing gates for routed nodes: {', '.join(missing_gates)}")
    gates_use = {node: gates[node] for node in gated_nodes}
    gate_outputs = _compute_gate_outputs(X, gates_use, gene_names=gene_names_use)
    node_outputs = _compute_node_outputs(
        X,
        experts_by_node,
        node_to_leaves,
        gene_names=gene_names_use,
    )

    feature_spec = model_bundle["stacker"].get("feature_spec")
    if isinstance(feature_spec, dict):
        gate_order = feature_spec.get("gate_order") or list(gate_outputs.keys())
        missing_gate_outputs = [g for g in gate_order if g not in gate_outputs]
        if missing_gate_outputs:
            raise ValueError(
                "Stacker expects gate outputs beyond stop_nodes/stop_depth. "
                "Retrain the stacker with the same routing cutoff."
            )
        node_order = feature_spec.get("node_order") or list(node_outputs.keys())
        missing_nodes = [n for n in node_order if n not in node_outputs]
        if missing_nodes:
            raise ValueError(f"Stacker expects node outputs for: {', '.join(missing_nodes)}")
        feature_spec = dict(feature_spec)
        feature_spec["gate_order"] = gate_order
        feature_spec["node_order"] = node_order

    X_meta, _ = build_stacker_features(
        gate_outputs,
        node_outputs,
        leaves_all,
        node_order=list(experts_by_node.keys()),
        feature_spec=feature_spec,
    )
    P_sub = predict_leaf_stacker(model_bundle["stacker"], X_meta)

    hard = route_tree_hard(
        X,
        tree,
        gates,
        stop_nodes=stop_nodes,
        stop_depth=stop_depth,
        gene_names=gene_names_use,
    )
    return {
        "P_sub": P_sub,
        "combined_scores": P_sub,
        "selected_paths": hard["selected_paths"],
        "selected_nodes": hard["selected_nodes"],
        "final_nodes": hard["final_nodes"],
        "gate_outputs": gate_outputs,
        "node_outputs": node_outputs,
        "leaves": leaves_all,
    }


def _final_node_from_leaf(leaf: str, node_to_leaves: Dict[str, List[str]]) -> str | None:
    for node, leaves in node_to_leaves.items():
        if leaf in leaves:
            return node
    return None


def _path_from_final_node(node: str) -> List[str]:
    if node == GAD_POS_NODE:
        return [ROOT_NODE, GAD_POS_NODE]
    if node in (SOX6_NODE, CALB_NODE):
        return [ROOT_NODE, GAD_NEG_NODE, node]
    return [ROOT_NODE, node]


def _filter_node_to_leaves(
    node_to_leaves: Dict[str, List[str]],
    excluded_leaves: List[str],
) -> Dict[str, List[str]]:
    excluded = set(excluded_leaves)
    return {node: [leaf for leaf in leaves if leaf not in excluded] for node, leaves in node_to_leaves.items()}


def predict_hierarchical_v4_hard(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
) -> Dict[str, Any]:
    """Hard routing: argmax gates + node-specific experts."""

    if "gates" not in model_bundle or "experts_by_node" not in model_bundle:
        raise ValueError("model_bundle must include 'gates' and 'experts_by_node'")

    gates = model_bundle["gates"]
    experts_by_node = model_bundle["experts_by_node"]
    routing = model_bundle.get("routing", {})
    node_to_leaves = routing.get("node_to_leaves") if isinstance(routing, dict) else None
    excluded = list(model_bundle.get("metadata", {}).get("excluded_leaves", []))

    if not isinstance(node_to_leaves, dict):
        leaves = model_bundle.get("leaves") or []
        if not leaves:
            raise ValueError("model_bundle missing leaves/node_to_leaves for routing")
        node_to_leaves, _ = build_node_to_leaves(list(leaves))
    node_to_leaves = _filter_node_to_leaves(node_to_leaves, excluded)

    leaves_all = list(
        leaf_order or model_bundle.get("leaves") or sorted({l for v in node_to_leaves.values() for l in v})
    )
    leaves_all, _ = filter_excluded_leaves(leaves_all, excluded)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves_all)}

    gate_root = gates.get(ROOT_NODE)
    gate_gad_neg = gates.get(GAD_NEG_NODE)
    if gate_root is None or gate_gad_neg is None:
        raise ValueError("Missing required gates in model_bundle")

    gene_names_use = model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    gate_outputs = _compute_gate_outputs(
        X,
        {ROOT_NODE: gate_root, GAD_NEG_NODE: gate_gad_neg},
        gene_names=gene_names_use,
    )
    node_outputs = _compute_node_outputs(
        X,
        experts_by_node,
        node_to_leaves,
        fill_value=fill_value,
        gene_names=gene_names_use,
    )

    root_nodes = gate_outputs[ROOT_NODE]["nodes"] or [GAD_NEG_NODE, GAD_POS_NODE]
    root_scores = gate_outputs[ROOT_NODE]["logits"]
    root_selected, _ = route_nodes(root_scores, root_nodes, top_k=1, normalize=True)

    n_samples = X.shape[0]
    final_nodes: List[str] = []
    selected_paths: List[List[str]] = []

    gad_neg_mask = np.array([sel[0] == GAD_NEG_NODE for sel in root_selected], dtype=bool)
    neg_selected = [None] * n_samples

    if np.any(gad_neg_mask):
        neg_nodes = gate_outputs[GAD_NEG_NODE]["nodes"] or [CALB_NODE, SOX6_NODE]
        neg_scores = gate_outputs[GAD_NEG_NODE]["logits"][gad_neg_mask]
        neg_selected_sub, _ = route_nodes(neg_scores, neg_nodes, top_k=1, normalize=True)
        idx = np.flatnonzero(gad_neg_mask)
        for i, sel in zip(idx, neg_selected_sub):
            neg_selected[i] = sel[0]

    for i in range(n_samples):
        root_choice = root_selected[i][0]
        if root_choice == GAD_POS_NODE:
            final = GAD_POS_NODE
        else:
            final = neg_selected[i] or CALB_NODE
        final_nodes.append(final)
        selected_paths.append(_path_from_final_node(final))

    combined = np.full((n_samples, len(leaves_all)), fill_value, dtype=float)
    for node in sorted(set(final_nodes)):
        idx = np.where(np.asarray(final_nodes) == node)[0]
        if idx.size == 0:
            continue
        node_scores = node_outputs[node]["scores"]
        node_leaves = node_outputs[node]["leaf_order"]
        for j, leaf in enumerate(node_leaves):
            col = leaf_to_idx.get(leaf)
            if col is not None:
                combined[idx, col] = node_scores[idx, j]

    selected_nodes = [[node] for node in final_nodes]
    leaf_candidates = [node_to_leaves.get(node, []) for node in final_nodes]

    return {
        "P_sub": combined,
        "combined_scores": combined,
        "selected_paths": selected_paths,
        "selected_nodes": selected_nodes,
        "selected_node": final_nodes,
        "final_nodes": final_nodes,
        "leaf_candidates": leaf_candidates,
        "node_outputs": node_outputs,
        "gate_outputs": gate_outputs,
        "leaves": leaves_all,
    }


def predict_hierarchical_v4_soft(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
) -> Dict[str, Any]:
    """Soft routing: gate probabilities + within-node softmax experts."""

    if "gates" not in model_bundle or "experts_by_node" not in model_bundle:
        raise ValueError("model_bundle must include 'gates' and 'experts_by_node'")

    gates = model_bundle["gates"]
    experts_by_node = model_bundle["experts_by_node"]
    routing = model_bundle.get("routing", {})
    node_to_leaves = routing.get("node_to_leaves") if isinstance(routing, dict) else None
    excluded = list(model_bundle.get("metadata", {}).get("excluded_leaves", []))

    if not isinstance(node_to_leaves, dict):
        leaves = model_bundle.get("leaves") or []
        if not leaves:
            raise ValueError("model_bundle missing leaves/node_to_leaves for routing")
        node_to_leaves, _ = build_node_to_leaves(list(leaves))
    node_to_leaves = _filter_node_to_leaves(node_to_leaves, excluded)

    leaves_all = list(
        leaf_order or model_bundle.get("leaves") or sorted({l for v in node_to_leaves.values() for l in v})
    )
    leaves_all, _ = filter_excluded_leaves(leaves_all, excluded)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves_all)}

    gate_root = gates.get(ROOT_NODE)
    gate_gad_neg = gates.get(GAD_NEG_NODE)
    if gate_root is None or gate_gad_neg is None:
        raise ValueError("Missing required gates in model_bundle")

    gene_names_use = model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    gate_outputs = _compute_gate_outputs(
        X,
        {ROOT_NODE: gate_root, GAD_NEG_NODE: gate_gad_neg},
        gene_names=gene_names_use,
    )
    node_outputs = _compute_node_outputs(
        X,
        experts_by_node,
        node_to_leaves,
        fill_value=fill_value,
        gene_names=gene_names_use,
    )

    root_nodes = gate_outputs[ROOT_NODE]["nodes"] or [GAD_NEG_NODE, GAD_POS_NODE]
    root_logits = gate_outputs[ROOT_NODE]["logits"]
    root_log_probs = _log_softmax(root_logits)
    root_idx = {n: i for i, n in enumerate(root_nodes)}
    log_w_gad_neg = root_log_probs[:, root_idx.get(GAD_NEG_NODE, 0)]
    log_w_gad_pos = root_log_probs[:, root_idx.get(GAD_POS_NODE, 1)]

    neg_nodes = gate_outputs[GAD_NEG_NODE]["nodes"] or [CALB_NODE, SOX6_NODE]
    neg_logits = gate_outputs[GAD_NEG_NODE]["logits"]
    neg_log_probs = _log_softmax(neg_logits)
    neg_idx = {n: i for i, n in enumerate(neg_nodes)}
    log_w_calb = log_w_gad_neg + neg_log_probs[:, neg_idx.get(CALB_NODE, 0)]
    log_w_sox6 = log_w_gad_neg + neg_log_probs[:, neg_idx.get(SOX6_NODE, 1)]

    log_w_final = {
        GAD_POS_NODE: log_w_gad_pos,
        SOX6_NODE: log_w_sox6,
        CALB_NODE: log_w_calb,
    }
    node_weight_order = [GAD_POS_NODE, SOX6_NODE, CALB_NODE]
    node_weights = np.stack([np.exp(log_w_final[node]) for node in node_weight_order], axis=1)

    combined_log = np.full((X.shape[0], len(leaves_all)), -np.inf, dtype=float)
    for node, log_weight in log_w_final.items():
        node_scores = node_outputs[node]["scores"]
        node_log_probs = _log_softmax(node_scores)
        node_outputs[node]["probs"] = np.exp(node_log_probs)
        node_leaves = node_outputs[node]["leaf_order"]
        for j, leaf in enumerate(node_leaves):
            col = leaf_to_idx.get(leaf)
            if col is not None:
                combined_log[:, col] = np.logaddexp(combined_log[:, col], log_weight + node_log_probs[:, j])

    log_norm = _logsumexp(combined_log, axis=1, keepdims=True)
    combined = np.exp(combined_log - log_norm)

    root_selected, _ = route_nodes(root_logits, root_nodes, top_k=1, normalize=True)
    n_samples = X.shape[0]
    selected_final: List[str] = []
    selected_paths: List[List[str]] = []

    gad_neg_mask = np.array([sel[0] == GAD_NEG_NODE for sel in root_selected], dtype=bool)
    neg_selected = [None] * n_samples
    if np.any(gad_neg_mask):
        neg_nodes = gate_outputs[GAD_NEG_NODE]["nodes"] or [CALB_NODE, SOX6_NODE]
        neg_scores = gate_outputs[GAD_NEG_NODE]["logits"][gad_neg_mask]
        neg_selected_sub, _ = route_nodes(neg_scores, neg_nodes, top_k=1, normalize=True)
        idx = np.flatnonzero(gad_neg_mask)
        for i, sel in zip(idx, neg_selected_sub):
            neg_selected[i] = sel[0]

    for i in range(n_samples):
        root_choice = root_selected[i][0]
        if root_choice == GAD_POS_NODE:
            final = GAD_POS_NODE
        else:
            final = neg_selected[i] or CALB_NODE
        selected_final.append(final)
        selected_paths.append(_path_from_final_node(final))

    selected_nodes = [[node] for node in selected_final]
    leaf_candidates = [node_to_leaves.get(node, []) for node in selected_final]

    return {
        "P_sub": combined,
        "combined_scores": combined,
        "selected_paths": selected_paths,
        "selected_nodes": selected_nodes,
        "selected_node": selected_final,
        "final_nodes": selected_final,
        "leaf_candidates": leaf_candidates,
        "node_outputs": node_outputs,
        "gate_outputs": gate_outputs,
        "node_weights": {"nodes": node_weight_order, "weights": node_weights},
        "leaves": leaves_all,
    }


def predict_hierarchical_v4_stacked(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    leaf_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Stacked final classifier: gate + expert logits -> stacker prediction."""

    if "stacker" not in model_bundle:
        raise ValueError("model_bundle missing stacker")
    if "gates" not in model_bundle or "experts_by_node" not in model_bundle:
        raise ValueError("model_bundle must include 'gates' and 'experts_by_node'")

    stacker = model_bundle["stacker"]
    gates = model_bundle["gates"]
    experts_by_node = model_bundle["experts_by_node"]
    routing = model_bundle.get("routing", {})
    node_to_leaves = routing.get("node_to_leaves") if isinstance(routing, dict) else None
    excluded = list(model_bundle.get("metadata", {}).get("excluded_leaves", []))
    if not isinstance(node_to_leaves, dict):
        leaves = model_bundle.get("leaves") or []
        if not leaves:
            raise ValueError("model_bundle missing leaves/node_to_leaves for routing")
        node_to_leaves, _ = build_node_to_leaves(list(leaves))
    node_to_leaves = _filter_node_to_leaves(node_to_leaves, excluded)

    leaves_all = list(leaf_order or model_bundle.get("leaves") or stacker.get("leaf_order", []))
    leaves_all, _ = filter_excluded_leaves(leaves_all, excluded)
    if not leaves_all:
        raise ValueError("leaf_order must be provided or present in stacker bundle")

    gene_names_use = model_bundle.get("gene_names") or model_bundle.get("metadata", {}).get("gene_names")
    gate_outputs = _compute_gate_outputs(X, gates, gene_names=gene_names_use)
    node_outputs = _compute_node_outputs(
        X,
        experts_by_node,
        node_to_leaves,
        gene_names=gene_names_use,
    )
    X_meta, _ = build_stacker_features(
        gate_outputs,
        node_outputs,
        leaves_all,
        node_order=list(node_outputs.keys()),
        feature_spec=stacker.get("feature_spec"),
    )

    P_sub = predict_leaf_stacker(stacker, X_meta)

    root_nodes = gate_outputs[ROOT_NODE]["nodes"] or [GAD_NEG_NODE, GAD_POS_NODE]
    root_scores = gate_outputs[ROOT_NODE]["logits"]
    root_selected, _ = route_nodes(root_scores, root_nodes, top_k=1, normalize=True)

    n_samples = X.shape[0]
    final_nodes: List[str] = []
    selected_paths: List[List[str]] = []

    gad_neg_mask = np.array([sel[0] == GAD_NEG_NODE for sel in root_selected], dtype=bool)
    neg_selected = [None] * n_samples

    if np.any(gad_neg_mask):
        neg_nodes = gate_outputs[GAD_NEG_NODE]["nodes"] or [CALB_NODE, SOX6_NODE]
        neg_scores = gate_outputs[GAD_NEG_NODE]["logits"][gad_neg_mask]
        neg_selected_sub, _ = route_nodes(neg_scores, neg_nodes, top_k=1, normalize=True)
        idx = np.flatnonzero(gad_neg_mask)
        for i, sel in zip(idx, neg_selected_sub):
            neg_selected[i] = sel[0]

    for i in range(n_samples):
        root_choice = root_selected[i][0]
        if root_choice == GAD_POS_NODE:
            final = GAD_POS_NODE
        else:
            final = neg_selected[i] or CALB_NODE
        final_nodes.append(final)
        selected_paths.append(_path_from_final_node(final))

    selected_nodes = [[node] for node in final_nodes]
    leaf_candidates = [node_to_leaves.get(node, []) for node in final_nodes]

    return {
        "P_sub": P_sub,
        "combined_scores": P_sub,
        "selected_paths": selected_paths,
        "selected_nodes": selected_nodes,
        "selected_node": final_nodes,
        "final_nodes": final_nodes,
        "leaf_candidates": leaf_candidates,
        "node_outputs": node_outputs,
        "gate_outputs": gate_outputs,
        "leaves": leaves_all,
    }


def predict_hierarchical_v4(
    X: np.ndarray,
    model_bundle: ModelBundle,
    *,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
) -> Dict[str, Any]:
    return predict_hierarchical_v4_hard(X, model_bundle, leaf_order=leaf_order, fill_value=fill_value)


def predict_node_moe(
    X: np.ndarray,
    model_bundle: ModelBundle,
    node_scores: Optional[np.ndarray] = None,
    height: Optional[int] = None,
    top_k: int = 1,
    normalize: bool = True,
    temperature: float = 1.0,
    leaf_order: Optional[List[str]] = None,
    fill_value: float = -np.inf,
) -> Dict[str, Any]:
    """Route by nodes then score leaf experts."""

    if "hierarchy" not in model_bundle:
        raise ValueError("model_bundle must include 'hierarchy'")
    if "experts" not in model_bundle:
        raise ValueError("model_bundle must include 'experts'")

    hierarchy = model_bundle["hierarchy"]
    partitions = hierarchy.get("partitions")
    if partitions is None:
        partitions = list(height_partitions(hierarchy["tree"], root_key=hierarchy.get("root_key", "node")))

    meta = model_bundle.get("metadata", {})
    if height is None:
        height = meta.get("height", 0)
    if height is None:
        height = 0
    if height < 0 or height >= len(partitions):
        raise ValueError(f"height out of range: {height}")

    nodes = partitions[height]
    meta_nodes = meta.get("nodes")
    if isinstance(meta_nodes, list) and meta_nodes:
        nodes = list(meta_nodes)
    elif "gate" in model_bundle:
        gate_nodes = model_bundle.get("gate", {}).get("nodes")
        if isinstance(gate_nodes, list) and gate_nodes:
            nodes = list(gate_nodes)
    node_to_leaves = hierarchy["node_to_leaves"]

    if node_scores is None:
        if "gate" not in model_bundle:
            raise ValueError("model_bundle must include 'gate' when node_scores is None")
        gate_bundle = model_bundle["gate"]
        scores_raw = predict_node_scores(gate_bundle, X)
        gate_nodes = gate_bundle.get("nodes", nodes)
        if gate_nodes != nodes:
            gate_idx = {n: i for i, n in enumerate(gate_nodes)}
            try:
                reorder = [gate_idx[n] for n in nodes]
            except KeyError as exc:
                raise ValueError("Gate nodes do not match hierarchy nodes") from exc
            node_scores = scores_raw[:, reorder]
        else:
            node_scores = scores_raw

    selected_nodes, selected_weights = route_nodes(
        scores=node_scores,
        nodes=nodes,
        top_k=top_k,
        normalize=normalize,
        temperature=temperature,
    )
    leaf_candidates = nodes_to_leaf_candidates(selected_nodes, node_to_leaves)

    leaf_scores_out = score_leaf_experts(
        X,
        model_bundle["experts"],
        leaf_candidates=leaf_candidates,
        leaf_order=leaf_order,
        fill_value=fill_value,
    )

    node_weights = [
        {node: float(w) for node, w in zip(nodes_sel, weights_sel)}
        for nodes_sel, weights_sel in zip(selected_nodes, selected_weights)
    ]
    combined = combine_node_weights_with_leaf_scores(
        leaf_score_matrix=leaf_scores_out["scores"],
        leaves=leaf_scores_out["leaves"],
        node_weights=node_weights,
        node_to_leaves=node_to_leaves,
    )

    return {
        "selected_nodes": selected_nodes,
        "selected_weights": selected_weights,
        "leaf_candidates": leaf_candidates,
        "leaf_scores": leaf_scores_out["scores"],
        "leaves": leaf_scores_out["leaves"],
        "combined_scores": combined["combined"],
        "leaf_weights": combined["leaf_weights"],
    }


if __name__ == "__main__":
    from collections import Counter

    toy_tree = {
        "node": {
            "A": ["A1", "A2"],
            "B": {"B1": ["B1a"]},
        }
    }

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y_sub = np.array(["A1" if x[0] > 0 else "A2" for x in X], dtype=object)
    y_sub[:50] = "B1a"

    bundle = train_node_moe(
        X,
        y_sub,
        toy_tree,
        height=1,
        gate_epochs=200,
        expert_epochs=200,
        seed=0,
    )

    out = predict_node_moe(X, bundle, node_scores=None, top_k=1)
    selected = [sel[0] for sel in out["selected_nodes"]]
    counts = Counter(selected)
    print("combined_scores shape:", out["combined_scores"].shape)
    print("combined_scores min/max:", float(np.nanmin(out["combined_scores"])), float(np.nanmax(out["combined_scores"])))
    print("node counts:", dict(counts))
    print("first 5 nodes:", selected[:5])

    toy_tree2 = {
        "node": {
            "gad_pos": ["G1"],
            "gad_neg": {"sox6": ["S1"], "calb": ["C1"]},
        }
    }
    gene_names = ["g0", "g1", "g2", "g3"]
    X2 = rng.normal(size=(30, len(gene_names)))
    y_sub2 = np.array(["G1", "S1", "C1"] * 10, dtype=object)

    gates = {
        "node": {"W": rng.normal(size=(4, 2)), "b": np.zeros(2), "nodes": ["node|gad_neg", "node|gad_pos"]},
        "node|gad_neg": {
            "W": rng.normal(size=(4, 2)),
            "b": np.zeros(2),
            "nodes": ["node|gad_neg|sox6", "node|gad_neg|calb"],
        },
    }
    experts_by_node = {
        "node|gad_pos": {"G1": {"weight": rng.normal(size=4), "bias": 0.0, "genes": gene_names}},
        "node|gad_neg|sox6": {"S1": {"weight": rng.normal(size=4), "bias": 0.0, "genes": gene_names}},
        "node|gad_neg|calb": {"C1": {"weight": rng.normal(size=4), "bias": 0.0, "genes": gene_names}},
    }
    subtype_to_path = walk_paths(toy_tree2)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    tree = build_tree_index_from_hierarchy(toy_tree2, root_key="node")
    gate_out = _compute_gate_outputs(X2, gates, gene_names=gene_names)
    node_out = _compute_node_outputs(X2, experts_by_node, node_to_leaves, gene_names=gene_names)
    X_meta, feature_spec = build_stacker_features(gate_out, node_out, list(subtype_to_path.keys()))
    stacker = train_leaf_stacker(X_meta, y_sub2, list(subtype_to_path.keys()), feature_spec=feature_spec)

    bundle2 = {
        "hierarchy": {
            "tree": toy_tree2,
            "root_key": "node",
            "subtype_to_path": subtype_to_path,
            "node_to_leaves": node_to_leaves,
        },
        "gates": gates,
        "experts_by_node": experts_by_node,
        "stacker": stacker,
        "leaves": list(subtype_to_path.keys()),
        "metadata": {"gene_names": gene_names},
    }
    stop_nodes = ["node|gad_pos", "node|gad_neg|sox6", "node|gad_neg|calb"]
    stacked = predict_tree_stacked(X2, bundle2, stop_nodes=stop_nodes, stop_depth=2, gene_names=gene_names)
    print("stacked final nodes:", dict(Counter(stacked["final_nodes"])))
