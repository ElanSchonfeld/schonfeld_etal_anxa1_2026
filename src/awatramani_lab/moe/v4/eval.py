"""Evaluation utilities for MoE v4 routing and per-node diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .routing_spec import EXCLUDED_LEAVES, assign_gadneg_branch, is_gad_positive
from .tree import TreeIndex, child_to_leaves


def _apply_exclusions(
    y_true: np.ndarray,
    P_pred: np.ndarray | None,
    labels: List[str],
    excluded_leaves: List[str] | None,
) -> tuple[np.ndarray, np.ndarray | None, List[str], np.ndarray]:
    excluded = set(excluded_leaves or EXCLUDED_LEAVES)
    mask = ~np.isin(y_true, np.asarray(list(excluded), dtype=object))
    y_true = y_true[mask]
    labels_keep = [label for label in labels if label not in excluded]
    if P_pred is None:
        return y_true, None, labels_keep, mask
    idx = [labels.index(label) for label in labels_keep if label in labels]
    P_pred = np.asarray(P_pred)[mask][:, idx] if idx else np.zeros((y_true.shape[0], 0), dtype=float)
    return y_true, P_pred, labels_keep, mask


def routing_summary(selected_nodes: Iterable[Iterable[str]]) -> Dict[str, Any]:
    """Summarize routing counts, fractions, and entropy."""

    flat = [nodes[0] for nodes in selected_nodes if nodes]
    total = len(flat)
    counts = Counter(flat)
    fractions = {k: (v / total if total else 0.0) for k, v in counts.items()}
    probs = np.array(list(fractions.values()), dtype=float)
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if probs.size else 0.0
    return {"total": total, "counts": dict(counts), "fractions": fractions, "entropy": entropy}


def hierarchical_gate_summary(
    selected_paths: Iterable[Iterable[str]],
    *,
    gad_pos_node: str,
    gad_neg_node: str,
    sox6_node: str,
    calb_node: str,
) -> Dict[str, Any]:
    """Summarize routing fractions at the gad and gad_neg gates."""

    root_counts = Counter()
    gad_neg_counts = Counter()
    total = 0

    for path in selected_paths:
        path = list(path)
        total += 1
        if gad_pos_node in path:
            root_counts["gad_pos"] += 1
            continue
        if gad_neg_node in path:
            root_counts["gad_neg"] += 1
            if sox6_node in path:
                gad_neg_counts["sox6"] += 1
            elif calb_node in path:
                gad_neg_counts["calb"] += 1
            else:
                gad_neg_counts["unknown"] += 1
            continue
        root_counts["unknown"] += 1

    root_total = sum(root_counts.values())
    gad_neg_total = sum(gad_neg_counts.values())

    return {
        "root": {
            "counts": dict(root_counts),
            "fractions": {k: (v / root_total if root_total else 0.0) for k, v in root_counts.items()},
        },
        "gad_neg": {
            "counts": dict(gad_neg_counts),
            "fractions": {k: (v / gad_neg_total if gad_neg_total else 0.0) for k, v in gad_neg_counts.items()},
        },
    }


def coverage_report(
    node_to_leaves: Dict[str, List[str]],
    selected_nodes: Iterable[Iterable[str]],
) -> Dict[str, Any]:
    """Report sample and leaf coverage per node."""

    counts = Counter([nodes[0] for nodes in selected_nodes if nodes])
    report: Dict[str, Any] = {}
    for node, leaves in node_to_leaves.items():
        report[node] = {
            "n_samples": int(counts.get(node, 0)),
            "n_leaves": int(len(leaves)),
        }
    return report


def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else None
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else None
    if rec_neg is None or rec_pos is None:
        return None
    return float(0.5 * (rec_neg + rec_pos))


def gate_metrics_binary(y_true: np.ndarray, probs_pos: np.ndarray) -> Dict[str, Any]:
    """Compute binary gate metrics (acc, bal acc, ROC-AUC, PR-AUC, confusion)."""

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(probs_pos, dtype=float).reshape(-1)
    y_pred = (probs >= 0.5).astype(int)

    acc = float((y_pred == y_true).mean()) if y_true.size else None
    bal_acc = _balanced_accuracy(y_true, y_pred)
    conf = _confusion_matrix_binary(y_true, y_pred)

    roc = None
    pr = None
    reason = None
    if np.unique(y_true).size < 2:
        reason = "only one class present"
    else:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

            roc = float(roc_auc_score(y_true, probs))
            pr = float(average_precision_score(y_true, probs))
        except Exception as exc:
            reason = str(exc)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion_matrix": conf,
        "reason": reason,
    }


def evaluate_gate_metrics(
    y_true_subtype: Iterable[str],
    gate_outputs: Dict[str, Dict[str, np.ndarray]],
    *,
    subtype_to_family: Dict[str, str] | None = None,
    excluded_leaves: List[str] | None = None,
) -> Dict[str, Any]:
    """Evaluate root and gad_neg gate metrics from subtype labels."""

    y_true = np.asarray(list(y_true_subtype)).astype(str)
    y_true, _, _, _ = _apply_exclusions(y_true, None, [], excluded_leaves)
    y_gad = np.array(
        [1 if is_gad_positive(st, subtype_to_family=subtype_to_family) else 0 for st in y_true],
        dtype=int,
    )

    root = gate_outputs.get("node")
    if root is None or "probs" not in root:
        raise ValueError("gate_outputs missing root gate probabilities")
    root_probs = np.asarray(root["probs"], dtype=float)
    root_nodes = root.get("nodes", ["node|gad_neg", "node|gad_pos"])
    root_idx = {n: i for i, n in enumerate(root_nodes)}
    gad_pos_idx = root_idx.get("node|gad_pos", 1)
    root_metrics = gate_metrics_binary(y_gad, root_probs[:, gad_pos_idx])

    gad_neg_mask = y_gad == 0
    gad_neg = gate_outputs.get("node|gad_neg")
    if gad_neg is None or "probs" not in gad_neg:
        raise ValueError("gate_outputs missing gad_neg gate probabilities")
    gad_neg_probs = np.asarray(gad_neg["probs"], dtype=float)
    gad_neg_nodes = gad_neg.get("nodes", ["node|gad_neg|calb", "node|gad_neg|sox6"])
    gad_neg_idx = {n: i for i, n in enumerate(gad_neg_nodes)}
    sox6_idx = gad_neg_idx.get("node|gad_neg|sox6", 1)
    y_sox6 = np.array(
        [
            1
            if assign_gadneg_branch(st, subtype_to_family=subtype_to_family) == "sox6"
            else 0
            for st in y_true[gad_neg_mask]
        ],
        dtype=int,
    )
    gad_neg_metrics = gate_metrics_binary(y_sox6, gad_neg_probs[gad_neg_mask][:, sox6_idx])

    return {"root": root_metrics, "gad_neg": gad_neg_metrics}


def overall_metrics(
    y_true: Iterable[str],
    P_pred: np.ndarray,
    labels: List[str],
    *,
    excluded_leaves: List[str] | None = None,
) -> Dict[str, Any]:
    """Compute overall metrics for leaf predictions."""

    y_true = np.asarray(list(y_true)).astype(str)
    y_true, P, labels, _ = _apply_exclusions(y_true, P_pred, labels, excluded_leaves)
    P = np.asarray(P, dtype=float)
    pred_idx = P.argmax(axis=1)
    pred_labels = np.asarray(labels, dtype=object)[pred_idx]

    acc = float((pred_labels == y_true).mean()) if y_true.size else None
    class_metrics = per_class_metrics(y_true, pred_labels, labels)

    support = {label: int(np.sum(y_true == label)) for label in labels}
    auc_vals: List[float] = []
    pr_vals: List[float] = []
    reason = None
    for i, label in enumerate(labels):
        if support.get(label, 0) == 0:
            continue
        y_bin = (y_true == label).astype(int)
        roc, pr, _reason = _auc_scores(y_bin, P[:, i])
        if roc is not None:
            auc_vals.append(float(roc))
        if pr is not None:
            pr_vals.append(float(pr))
        if _reason and reason is None:
            reason = _reason

    return {
        "accuracy": acc,
        "macro_f1": class_metrics.get("macro_f1"),
        "macro_precision": class_metrics.get("macro_precision"),
        "macro_recall": class_metrics.get("macro_recall"),
        "macro_roc_auc": float(np.mean(auc_vals)) if auc_vals else None,
        "macro_pr_auc": float(np.mean(pr_vals)) if pr_vals else None,
        "auc_reason": reason,
        "support": support,
    }


def write_mode_outputs(
    out_dir: str,
    mode_name: str,
    metrics: Dict[str, Any],
    per_node: Dict[str, Any],
    *,
    excluded_leaves: List[str] | None = None,
) -> Tuple[str, str]:
    """Write metrics JSON and per-node CSV for a given mode."""

    import csv
    import json
    from pathlib import Path

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / f"metrics_{mode_name}.json"
    csv_path = base / f"per_node_metrics_{mode_name}.csv"

    json_path.write_text(
        json.dumps(
            {"metrics": metrics, "per_node": per_node, "excluded_leaves": excluded_leaves or []},
            indent=2,
        )
    )

    rows = per_node_table(per_node)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["node", "n", "frac", "top1_acc", "macro_f1", "macro_roc_auc", "macro_pr_auc"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(json_path), str(csv_path)


def write_gate_metrics(out_dir: str, gate_metrics: Dict[str, Any], *, excluded_leaves: List[str] | None = None) -> str:
    """Write gate metrics JSON."""

    import json
    from pathlib import Path

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "gate_metrics.json"
    payload = {"gate_metrics": gate_metrics, "excluded_leaves": excluded_leaves or []}
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def write_height_metrics(
    out_dir: str,
    mode_name: str,
    height: int,
    height_report: Dict[str, Any],
    *,
    excluded_leaves: List[str] | None = None,
) -> str:
    """Write per-node metrics at a dendrogram height."""

    import csv
    import json
    from pathlib import Path

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / f"per_node_metrics_height{height}_{mode_name}.csv"
    json_path = base / f"per_node_metrics_height{height}_{mode_name}.json"

    per_node = height_report.get("per_node", {})
    rows = per_node_table(per_node)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["node", "n", "frac", "top1_acc", "macro_f1", "macro_roc_auc", "macro_pr_auc"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(
        json.dumps(
            {"height": height, "per_node": per_node, "excluded_leaves": excluded_leaves or []},
            indent=2,
        )
    )
    return str(csv_path)


def top1_accuracy(y_true: Iterable[str], P_pred: np.ndarray, labels: List[str]) -> float:
    """Compute top-1 accuracy for predicted probabilities."""

    y_true = np.asarray(list(y_true)).astype(str)
    pred_idx = np.asarray(P_pred).argmax(axis=1)
    pred_labels = np.asarray(labels, dtype=object)[pred_idx]
    return float((pred_labels == y_true).mean())


def per_class_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
) -> Dict[str, Any]:
    """Compute precision/recall/F1 per class and macro averages."""

    y_true = np.asarray(list(y_true)).astype(str)
    y_pred = np.asarray(list(y_pred)).astype(str)
    metrics: Dict[str, Dict[str, Any]] = {}
    f1_vals: List[float] = []
    prec_vals: List[float] = []
    rec_vals: List[float] = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
        if f1 is not None:
            f1_vals.append(float(f1))
        if precision is not None:
            prec_vals.append(float(precision))
        if recall is not None:
            rec_vals.append(float(recall))

    macro_f1 = float(np.mean(f1_vals)) if f1_vals else None
    macro_precision = float(np.mean(prec_vals)) if prec_vals else None
    macro_recall = float(np.mean(rec_vals)) if rec_vals else None
    return {
        "per_class": metrics,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def _binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC for binary labels using rank statistics."""

    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC undefined for a single class")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[y_true == 1]
    auc = (pos_ranks.sum() - n_pos * (n_pos - 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _binary_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision for binary labels."""

    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        raise ValueError("PR-AUC undefined with no positives")

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else 0.0
    return ap


def _auc_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float | None, float | None, str | None]:
    """Compute ROC-AUC and PR-AUC for binary labels using sklearn if available."""

    if np.unique(y_true).size < 2:
        return None, None, "only one class present"

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

        roc = float(roc_auc_score(y_true, scores))
        pr = float(average_precision_score(y_true, scores))
        return roc, pr, None
    except Exception:
        try:
            roc = _binary_roc_auc(y_true, scores)
            pr = _binary_pr_auc(y_true, scores)
            return roc, pr, "sklearn unavailable"
        except Exception as exc:
            return None, None, str(exc)


def per_node_metrics(
    y_true_subtype: Iterable[str],
    P_sub: np.ndarray,
    all_subtypes: List[str],
    node_to_leaves: Dict[str, List[str]],
    selected_nodes: Iterable[Iterable[str]] | None,
    *,
    excluded_leaves: List[str] | None = None,
    final_nodes: Iterable[str] | None = None,
    selected_paths: Iterable[Iterable[str]] | None = None,
) -> Dict[str, Any]:
    """Compute per-node routing metrics using selected nodes."""

    y_true = np.asarray(list(y_true_subtype)).astype(str)
    y_true, P_sub, all_subtypes, mask = _apply_exclusions(y_true, P_sub, all_subtypes, excluded_leaves)
    if selected_nodes is not None:
        selected = [nodes[0] for nodes in selected_nodes if nodes]
    elif final_nodes is not None:
        selected = [str(node) for node in final_nodes]
    elif selected_paths is not None:
        selected = []
        for path in selected_paths:
            path = list(path)
            selected.append(path[-1] if path else "unknown")
    else:
        raise ValueError("selected_nodes, final_nodes, or selected_paths must be provided")

    if len(selected) != len(mask):
        raise ValueError("selected nodes length must match y_true_subtype length before exclusions")

    selected = [node for node, keep in zip(selected, mask) if keep]
    node_to_leaves = {k: [leaf for leaf in v if leaf in set(all_subtypes)] for k, v in node_to_leaves.items()}
    if len(selected) != len(y_true):
        raise ValueError("selected_nodes length must match y_true_subtype length after exclusions")

    leaf_idx = {leaf: i for i, leaf in enumerate(all_subtypes)}
    metrics: Dict[str, Any] = {}
    routing = Counter(selected)
    total = len(selected)

    for node, count in routing.items():
        idx = [i for i, n in enumerate(selected) if n == node]
        leaves = node_to_leaves.get(node, [])
        if not leaves:
            metrics[node] = {"n": count, "frac": count / total, "reason": "no leaves for node"}
            continue

        col_idx = [leaf_idx[l] for l in leaves if l in leaf_idx]
        P_node = P_sub[idx][:, col_idx]
        y_node = y_true[idx]

        pred_idx = P_node.argmax(axis=1) if P_node.size else np.array([], dtype=int)
        pred_labels = np.array([leaves[i] for i in pred_idx], dtype=object)

        leaf_counts = {leaf: int(np.sum(y_node == leaf)) for leaf in leaves}
        leaf_total = float(len(y_node)) if len(y_node) else 0.0
        leaf_fractions = {
            leaf: (count / leaf_total if leaf_total else 0.0) for leaf, count in leaf_counts.items()
        }

        acc = float((pred_labels == y_node).mean()) if len(pred_labels) else None
        class_metrics = per_class_metrics(y_node, pred_labels, leaves)

        roc_auc: Dict[str, Any] = {}
        pr_auc: Dict[str, Any] = {}
        roc_vals: List[float] = []
        pr_vals: List[float] = []
        for leaf in leaves:
            scores = P_sub[idx, leaf_idx[leaf]]
            y_bin = (y_node == leaf).astype(int)
            roc, pr, reason = _auc_scores(y_bin, scores)
            roc_auc[leaf] = {"value": roc, "reason": reason}
            pr_auc[leaf] = {"value": pr, "reason": reason}
            if roc is not None:
                roc_vals.append(float(roc))
            if pr is not None:
                pr_vals.append(float(pr))

        metrics[node] = {
            "n": int(count),
            "frac": float(count / total) if total else 0.0,
            "leaves": leaves,
            "top1_acc": acc,
            "macro_f1": class_metrics["macro_f1"],
            "leaf_counts": leaf_counts,
            "leaf_fractions": leaf_fractions,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "macro_roc_auc": float(np.mean(roc_vals)) if roc_vals else None,
            "macro_pr_auc": float(np.mean(pr_vals)) if pr_vals else None,
        }

    return metrics


def per_height_metrics(
    y_true_subtype: Iterable[str],
    P_sub: np.ndarray,
    all_subtypes: List[str],
    subtype_to_path: Dict[str, List[str]],
    height: int,
    *,
    excluded_leaves: List[str] | None = None,
) -> Dict[str, Any]:
    """Compute per-node metrics at a dendrogram height, independent of routing."""

    y_true = np.asarray(list(y_true_subtype)).astype(str)
    y_true, P_sub, all_subtypes, mask = _apply_exclusions(y_true, P_sub, all_subtypes, excluded_leaves)

    def node_at_height(path: List[str]) -> str:
        if height < len(path):
            return "|".join(path[: height + 1])
        return "|".join(path)

    selected_nodes: List[List[str]] = []
    for st in y_true:
        path = subtype_to_path.get(str(st))
        if not path:
            selected_nodes.append(["unknown"])
        else:
            selected_nodes.append([node_at_height(path)])

    node_to_leaves: Dict[str, List[str]] = {}
    for leaf in all_subtypes:
        path = subtype_to_path.get(str(leaf))
        if not path:
            continue
        node = node_at_height(path)
        node_to_leaves.setdefault(node, []).append(leaf)

    per_node = per_node_metrics(
        y_true,
        P_sub,
        all_subtypes,
        node_to_leaves,
        selected_nodes,
        excluded_leaves=[],
    )

    return {
        "height": height,
        "node_to_leaves": node_to_leaves,
        "per_node": per_node,
    }


def per_node_table(per_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten per-node metrics into a table of summaries."""

    rows: List[Dict[str, Any]] = []
    for node, stats in per_node.items():
        rows.append(
            {
                "node": node,
                "n": stats.get("n"),
                "frac": stats.get("frac"),
                "top1_acc": stats.get("top1_acc"),
                "macro_f1": stats.get("macro_f1"),
                "macro_roc_auc": stats.get("macro_roc_auc"),
                "macro_pr_auc": stats.get("macro_pr_auc"),
            }
        )
    return rows


def evaluate_node_gate_metrics(
    y_true_subtype: Iterable[str],
    gate_outputs: Dict[str, Dict[str, np.ndarray]],
    tree: TreeIndex,
    *,
    excluded_leaves: List[str] | None = None,
) -> Dict[str, Any]:
    """Evaluate per-node OVR child classifiers and hard child selection."""

    y_true_all = np.asarray(list(y_true_subtype)).astype(str)
    y_true, _, _, mask = _apply_exclusions(y_true_all, None, list(tree.leaves), excluded_leaves)

    per_node: Dict[str, Any] = {}
    for node, gate in gate_outputs.items():
        children = list(gate.get("nodes", []))
        if not children:
            per_node[node] = {"reason": "no children", "n": 0}
            continue

        probs = np.asarray(gate.get("probs"))
        probs = probs[mask]
        node_leaves = tree.get_leaves(node)
        node_mask = np.isin(y_true, np.asarray(node_leaves, dtype=object))
        n_node = int(node_mask.sum())
        if n_node == 0:
            per_node[node] = {"reason": "no samples", "n": 0, "children": children}
            continue

        probs_node = probs[node_mask]
        y_node = y_true[node_mask]

        child_metrics: Dict[str, Any] = {}
        roc_vals: List[float] = []
        pr_vals: List[float] = []

        leaf_to_child: Dict[str, str] = {}
        for child in children:
            leaves = child_to_leaves(tree, child)
            for leaf in leaves:
                leaf_to_child[str(leaf)] = str(child)

        true_child = np.array([leaf_to_child.get(lbl, "unknown") for lbl in y_node], dtype=object)
        pred_idx = probs_node.argmax(axis=1) if probs_node.size else np.array([], dtype=int)
        pred_child = np.array([children[i] for i in pred_idx], dtype=object)

        conf = np.zeros((len(children), len(children)), dtype=int)
        for i, c_true in enumerate(children):
            for j, c_pred in enumerate(children):
                conf[i, j] = int(np.sum((true_child == c_true) & (pred_child == c_pred)))

        for j, child in enumerate(children):
            y_bin = (true_child == child).astype(int)
            scores = probs_node[:, j]
            roc, pr, reason = _auc_scores(y_bin, scores)
            acc = float(np.mean((scores >= 0.5).astype(int) == y_bin)) if y_bin.size else None
            bal_acc = _balanced_accuracy(y_bin, (scores >= 0.5).astype(int))
            if roc is not None:
                roc_vals.append(float(roc))
            if pr is not None:
                pr_vals.append(float(pr))
            child_metrics[child] = {
                "roc_auc": roc,
                "pr_auc": pr,
                "accuracy_0.5": acc,
                "balanced_accuracy": bal_acc,
                "reason": reason,
                "n_pos": int(y_bin.sum()),
                "n_neg": int(y_bin.size - y_bin.sum()),
            }

        per_node[node] = {
            "n": n_node,
            "children": children,
            "child_metrics": child_metrics,
            "macro_roc_auc": float(np.mean(roc_vals)) if roc_vals else None,
            "macro_pr_auc": float(np.mean(pr_vals)) if pr_vals else None,
            "confusion_matrix": conf.tolist(),
        }

    return {"per_node": per_node, "excluded_leaves": excluded_leaves or []}


def node_gate_metrics_table(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_node = metrics.get("per_node", {})
    for node, stats in per_node.items():
        child_metrics = stats.get("child_metrics", {})
        for child, cm in child_metrics.items():
            rows.append(
                {
                    "node": node,
                    "child": child,
                    "n": stats.get("n", 0),
                    "n_pos": cm.get("n_pos"),
                    "n_neg": cm.get("n_neg"),
                    "accuracy_0.5": cm.get("accuracy_0.5"),
                    "balanced_accuracy": cm.get("balanced_accuracy"),
                    "roc_auc": cm.get("roc_auc"),
                    "pr_auc": cm.get("pr_auc"),
                    "macro_roc_auc": stats.get("macro_roc_auc"),
                    "macro_pr_auc": stats.get("macro_pr_auc"),
                }
            )
    return rows


def write_node_gate_metrics(out_dir: str, metrics: Dict[str, Any]) -> Tuple[str, str]:
    """Write per-node gate metrics to JSON and CSV."""

    import json
    from pathlib import Path

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "node_gate_metrics.json"
    csv_path = base / "node_gate_metrics.csv"

    json_path.write_text(json.dumps(metrics, indent=2))

    rows = node_gate_metrics_table(metrics)
    if rows:
        import csv

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return str(json_path), str(csv_path)


def write_eval_outputs(
    out_dir: str,
    run_name: str,
    routing: Dict[str, Any],
    per_node: Dict[str, Any],
    overall_acc: float | None,
) -> Tuple[str, str]:
    """Write JSON and CSV evaluation outputs."""

    import csv
    import json
    from pathlib import Path

    base = Path(out_dir) / run_name
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / "v4_eval.json"
    csv_path = base / "v4_eval_nodes.csv"

    json_path.write_text(
        json.dumps(
            {
                "routing": routing,
                "overall_top1_acc": overall_acc,
                "per_node": per_node,
            },
            indent=2,
        )
    )

    rows = per_node_table(per_node)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["node", "n", "frac", "top1_acc", "macro_f1", "macro_roc_auc", "macro_pr_auc"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(json_path), str(csv_path)
