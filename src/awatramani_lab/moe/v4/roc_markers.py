"""Seurat-style ROC marker selection for hierarchical routing."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sorted_a = a[order]
    _, start_idx, counts = np.unique(sorted_a, return_index=True, return_counts=True)
    for start, count in zip(start_idx, counts):
        if count <= 1:
            continue
        end = start + count
        avg = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg
    return ranks


def _roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(scores)
    pos_ranks = ranks[y_true == 1]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true, scores))
    except Exception:
        return _roc_auc_manual(y_true, scores)


def seurat_roc_markers(
    X: np.ndarray,
    y_bin: np.ndarray,
    gene_names: List[str],
    *,
    min_pct: float = 0.1,
    min_diff_pct: Optional[float] = None,
) -> "pd.DataFrame":
    """Compute Seurat-style ROC markers for a binary split."""

    import pandas as pd

    X = np.asarray(X, dtype=float)
    y = np.asarray(y_bin, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length")
    if len(gene_names) != X.shape[1]:
        raise ValueError("gene_names length must match X columns")

    pos_mask = y == 1
    neg_mask = y == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return pd.DataFrame(
            columns=["gene", "auc", "power", "direction", "mean_pos", "mean_neg", "pct_pos", "pct_neg"]
        )

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    mean_pos = X_pos.mean(axis=0)
    mean_neg = X_neg.mean(axis=0)
    pct_pos = (X_pos > 0).mean(axis=0)
    pct_neg = (X_neg > 0).mean(axis=0)

    auc_vals: List[float] = []
    power_vals: List[float] = []
    direction_vals: List[str] = []
    for g in range(X.shape[1]):
        auc = _roc_auc(y, X[:, g])
        power = abs(auc - 0.5) * 2.0
        direction = "cells.1_high" if auc > 0.5 else "cells.2_high"
        auc_vals.append(auc)
        power_vals.append(power)
        direction_vals.append(direction)

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "auc": auc_vals,
            "power": power_vals,
            "direction": direction_vals,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "pct_pos": pct_pos,
            "pct_neg": pct_neg,
        }
    )

    if min_pct is not None:
        df["pass_min_pct"] = np.maximum(df["pct_pos"], df["pct_neg"]) >= float(min_pct)
    if min_diff_pct is not None:
        df["pass_min_diff_pct"] = np.abs(df["pct_pos"] - df["pct_neg"]) >= float(min_diff_pct)
    return df


def select_top_markers(
    markers: "pd.DataFrame",
    *,
    topk: int,
    min_pct: float,
    min_diff_pct: Optional[float],
) -> List[str]:
    """Select top markers by predictive power after Seurat-like filtering."""

    if markers.empty:
        return []
    mask = np.maximum(markers["pct_pos"], markers["pct_neg"]) >= float(min_pct)
    if min_diff_pct is not None:
        mask &= np.abs(markers["pct_pos"] - markers["pct_neg"]) >= float(min_diff_pct)
    filtered = markers[mask].sort_values("power", ascending=False)
    if filtered.empty:
        return []
    return list(filtered.head(int(topk))["gene"].astype(str))


def shared_markers_from_children(
    per_child: List["pd.DataFrame"],
    *,
    topk: int,
    min_pct: float,
    min_diff_pct: Optional[float],
    balanced: bool = True,
) -> List[str]:
    """Aggregate per-child ROC tables into a shared marker list."""
    import pandas as pd

    if not per_child:
        return []

    n_children = len(per_child)

    if balanced and n_children > 1:
        per_child_k = max(1, topk // n_children)
        remainder = topk - (per_child_k * n_children)

        all_genes = set()
        for i, child_df in enumerate(per_child):
            if child_df.empty:
                continue
            mask = np.maximum(child_df["pct_pos"], child_df["pct_neg"]) >= float(min_pct)
            if min_diff_pct is not None:
                mask &= np.abs(child_df["pct_pos"] - child_df["pct_neg"]) >= float(min_diff_pct)
            filtered = child_df[mask].sort_values("power", ascending=False)

            k = per_child_k + (1 if i < remainder else 0)
            child_genes = list(filtered.head(k)["gene"].astype(str))
            all_genes.update(child_genes)

        if len(all_genes) < topk:
            merged = pd.concat(per_child, ignore_index=True)
            merged = merged.groupby("gene", as_index=False).agg(
                power=("power", "max"),
                pct_pos=("pct_pos", "max"),
                pct_neg=("pct_neg", "max"),
            )
            mask = np.maximum(merged["pct_pos"], merged["pct_neg"]) >= float(min_pct)
            if min_diff_pct is not None:
                mask &= np.abs(merged["pct_pos"] - merged["pct_neg"]) >= float(min_diff_pct)
            merged = merged[mask].sort_values("power", ascending=False)
            for gene in merged["gene"].astype(str):
                if len(all_genes) >= topk:
                    break
                all_genes.add(gene)

        return list(all_genes)

    else:
        merged = pd.concat(per_child, ignore_index=True)
        merged = merged.groupby("gene", as_index=False).agg(
            power=("power", "max"),
            pct_pos=("pct_pos", "max"),
            pct_neg=("pct_neg", "max"),
        )
        mask = np.maximum(merged["pct_pos"], merged["pct_neg"]) >= float(min_pct)
        if min_diff_pct is not None:
            mask &= np.abs(merged["pct_pos"] - merged["pct_neg"]) >= float(min_diff_pct)
        merged = merged[mask].sort_values("power", ascending=False)
        return list(merged.head(int(topk))["gene"].astype(str))
