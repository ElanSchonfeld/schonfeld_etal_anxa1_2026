"""Visualization helpers for MoE v4."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .routing_spec import EXCLUDED_LEAVES, filter_excluded_leaves

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional
    sp = None


def filter_probs(
    P_sub: np.ndarray,
    subtypes: List[str],
    excluded_leaves: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    excluded = excluded_leaves or EXCLUDED_LEAVES
    kept, _ = filter_excluded_leaves(list(subtypes), excluded)
    if not kept:
        raise ValueError("All subtypes excluded; cannot plot.")
    idx = [subtypes.index(s) for s in kept]
    P_sub = np.asarray(P_sub)[:, idx]
    row_sum = np.maximum(P_sub.sum(axis=1, keepdims=True), 1e-12)
    P_sub = P_sub / row_sum
    return P_sub, kept


def argmax_labels(
    P_sub: np.ndarray,
    subtypes: List[str],
    excluded_leaves: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return predicted labels and pmax after filtering excluded leaves."""

    P_use, subtypes_use = filter_probs(P_sub, subtypes, excluded_leaves)
    pred_idx = P_use.argmax(axis=1) if P_use.size else np.zeros((P_use.shape[0],), dtype=int)
    pred_labels = np.asarray(subtypes_use, dtype=object)[pred_idx]
    pmax = P_use[np.arange(P_use.shape[0]), pred_idx] if P_use.size else np.zeros((P_use.shape[0],), dtype=float)
    return pred_labels, pmax, subtypes_use


def pred_leaf_labels(P_sub: np.ndarray, leaf_order: List[str]) -> np.ndarray:
    """Return argmax leaf labels for a probability matrix."""

    P_sub = np.asarray(P_sub, dtype=float)
    if P_sub.ndim != 2:
        raise ValueError(f"P_sub must be 2D, got shape={P_sub.shape}")
    if len(leaf_order) != P_sub.shape[1]:
        raise ValueError("leaf_order length must match P_sub columns")
    pred_idx = P_sub.argmax(axis=1) if P_sub.size else np.zeros((P_sub.shape[0],), dtype=int)
    return np.asarray(leaf_order, dtype=object)[pred_idx]


def entropy(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute entropy per row for probability matrix."""

    P = np.asarray(P, dtype=float)
    P = np.maximum(P, eps)
    return -np.sum(P * np.log(P), axis=1)


def shorten_node_label(node: str) -> str:
    """Compact node labels for plots (strip prefix, replace separators)."""

    text = str(node)
    if text.startswith("node|"):
        text = text[len("node|") :]
    return text.replace("|", "->")


def standard_node_label(
    node: str,
    node_to_leaves: Optional[dict] = None,
    *,
    include_terminal: bool = True,
) -> str:
    """Format a dendrogram node label with family-first naming."""

    parts = str(node).split("|")
    if parts and parts[0] == "node":
        parts = parts[1:]

    def _clean_marker(marker: str) -> str:
        marker = marker.strip()
        if marker.endswith("+"):
            marker = marker[:-1]
        return marker

    def _pick_marker(segment: str, prefer: Optional[str] = None) -> Optional[str]:
        markers = [m.strip() for m in segment.split(",") if m.strip()]
        if prefer:
            for m in markers:
                if prefer in m:
                    return m
        return markers[0] if markers else None

    start_idx = 0
    family = None
    for i, seg in enumerate(parts):
        if "Gad2" in seg:
            start_idx = i
            family = "Gad2"
            break
        if "Sox6" in seg:
            start_idx = i
            family = "Sox6"
            break
        if "Calb1" in seg:
            start_idx = i
            family = "Calb1"
            break
        if "Lef1" in seg:
            start_idx = i
            family = "Lef1"
            break

    label_parts: List[str] = []
    for i, seg in enumerate(parts[start_idx:]):
        prefer = family if i == 0 and family else None
        marker = _pick_marker(seg, prefer=prefer)
        if marker is None:
            continue
        label_parts.append(_clean_marker(marker))

    label = "/".join(label_parts) if label_parts else str(node).replace("|", "/")
    if include_terminal and node_to_leaves:
        leaves = node_to_leaves.get(str(node), [])
        if len(leaves) == 1:
            label = f"{label} ({leaves[0]})"
    return label


def _row_normalize(mat):
    if sp is not None and sp.issparse(mat):
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_sum)
        return inv @ mat
    mat = np.asarray(mat, dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return mat / row_sum


def _knn_connectivities(X: np.ndarray, k: int = 15):
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - optional
        raise ImportError("sklearn is required to build kNN connectivities") from exc

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
    _, indices = nbrs.kneighbors(X)

    rows = np.repeat(np.arange(n), indices.shape[1])
    cols = indices.ravel()
    data = np.ones_like(rows, dtype=float)
    if sp is not None:
        mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        mat = mat.maximum(mat.T)
        mat.setdiag(1.0)
        return mat

    mat = np.zeros((n, n), dtype=float)
    mat[rows, cols] = 1.0
    np.fill_diagonal(mat, 1.0)
    return mat


def smooth_probs_graph(
    adata,
    P: np.ndarray,
    *,
    alpha: float = 0.7,
    n_steps: int = 3,
    use_umap: bool = True,
    k: int = 15,
) -> np.ndarray:
    """Smooth probabilities over a graph using diffusion (unsupervised)."""

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    P0 = np.asarray(P, dtype=float)
    if P0.shape[0] != adata.n_obs:
        raise ValueError("P rows must match adata.n_obs")

    conn = None
    if hasattr(adata, "obsp") and "connectivities" in adata.obsp:
        conn = adata.obsp["connectivities"]
    if conn is None:
        X = None
        if use_umap and hasattr(adata, "obsm") and "X_umap" in adata.obsm:
            X = np.asarray(adata.obsm["X_umap"])
        if X is None:
            X = np.asarray(getattr(adata, "X"))
        conn = _knn_connectivities(X, k=k)

    A = _row_normalize(conn)
    P_s = P0.copy()
    for _ in range(n_steps):
        if sp is not None and sp.issparse(A):
            P_s = alpha * P0 + (1.0 - alpha) * A.dot(P_s)
        else:
            P_s = alpha * P0 + (1.0 - alpha) * (A @ P_s)

    P_s = np.maximum(P_s, 1e-12)
    P_s = P_s / np.sum(P_s, axis=1, keepdims=True)
    return P_s


def attach_pred_columns_from_probs(
    adata,
    P_sub: np.ndarray,
    subtypes: List[str],
    prefix: str,
    *,
    excluded_leaves: Optional[List[str]] = None,
) -> None:
    """Attach predicted label and pmax columns to adata.obs."""

    pred_labels, pmax, _ = argmax_labels(P_sub, subtypes, excluded_leaves)
    adata.obs[f"{prefix}_label"] = pred_labels
    adata.obs[f"{prefix}_pmax"] = pmax


def plot_umap_continuous(
    adata,
    values: np.ndarray,
    title: str,
    out_path: str,
    *,
    s: float = 10.0,
    cmap: str = "viridis",
) -> None:
    """Plot a continuous value over UMAP coordinates."""

    import matplotlib.pyplot as plt

    X_umap = np.asarray(adata.obsm.get("X_umap"))
    if X_umap.shape[1] != 2:
        raise ValueError("adata.obsm['X_umap'] must be Nx2 for plotting")
    vals = np.asarray(values, dtype=float)
    if vals.shape[0] != X_umap.shape[0]:
        raise ValueError("values length must match adata.n_obs")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=vals, s=s, linewidths=0, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def plot_umap_categorical(
    adata,
    labels: Iterable[str],
    title: str,
    out_path: str,
    *,
    s: float = 10.0,
    legend: bool = True,
    cmap: str = "tab20",
) -> None:
    """Plot categorical labels over UMAP coordinates."""

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    X_umap = np.asarray(adata.obsm.get("X_umap"))
    if X_umap.shape[1] != 2:
        raise ValueError("adata.obsm['X_umap'] must be Nx2 for plotting")
    labels = np.asarray(list(labels)).astype(str)
    if labels.shape[0] != X_umap.shape[0]:
        raise ValueError("labels length must match adata.n_obs")

    cats = np.unique(labels)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    colors = np.array([cat_to_idx[v] for v in labels])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=s, linewidths=0, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    if legend:
        norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
        cm = plt.get_cmap(cmap)
        handles = []
        for i, label in enumerate(cats):
            color = cm(norm(i))
            handles.append(plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color))
        ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def truth_family_from_cell_type(cell_type: str) -> str:
    """Map a human cell type string to SOX6/CALB1/OTHER."""

    val = str(cell_type)
    if "SOX6" in val or val.startswith("SOX6"):
        return "SOX6"
    if "CALB1" in val or val.startswith("CALB1"):
        return "CALB1"
    return "OTHER"


def pred_family_from_final_node(final_node: str) -> str:
    """Map a final routing node to SOX6/CALB1/GAD."""

    node = str(final_node)
    if node.endswith("gad_pos"):
        return "GAD"
    if node.endswith("sox6"):
        return "SOX6"
    if node.endswith("calb"):
        return "CALB1"
    return "OTHER"


def pred_family_from_leaf(leaf: str) -> str:
    """Map a leaf subtype string to SOX6/CALB1/GAD/OTHER."""

    val = str(leaf)
    if val.startswith("Sox6:"):
        return "SOX6"
    if val.startswith("Calb1:"):
        return "CALB1"
    if val.startswith("Gad2:"):
        return "GAD"
    return "OTHER"
