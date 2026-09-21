"""Comprehensive plotting knobs, confidence filtering, and metrics for MoE v4."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture

from .compat import predict_family_gate_safe


FAM_ORDER: List[str] = ["CALB1", "SOX6", "GAD"]

COOLOR_SUBTYPE_PALETTE: Dict[str, str] = {
    "Calb1:Ccdc192": "#1f77b4",
    "Calb1:Chrm2": "#aec7e8",
    "Calb1:Gipr": "#ff7f0e",
    "Calb1:Kctd8": "#ffbb78",
    "Calb1:Lpar1": "#2ca02c",
    "Calb1:Pde11a": "#d62728",
    "Calb1:Ptprt": "#ff9896",
    "Calb1:Stac": "#9467bd",
    "Calb1:Sulf1": "#8c564b",
    "Gad2:Ebf2": "#c49c94",
    "Gad2:Egfr": "#e377c2",
    "Sox6:Arhgap28": "#f7b6d2",
    "Sox6:Kcnmb2": "#c7c7c7",
    "Sox6:March3": "#bcbd22",
    "Sox6:Tafa1": "#dbdb8d",
    "Sox6:Tmem132d": "#17becf",
    "Sox6:Vcan": "#9edae5",
}

FAMILY_PALETTE: Dict[str, Tuple[float, float, float]] = {
    "CALB1": (0.12, 0.47, 0.71),
    "SOX6": (0.17, 0.63, 0.17),
    "GAD": (0.84, 0.15, 0.16),
}

_CANON_FAMS: List[str] = ["CALB1", "SOX6", "GAD"]

MOUSE_TRUE_SUB_COL = "subtype"
HUMAN_TRUE_SUB_COL = "Cell_Type"
HUMAN_TRUE_FAM_COL = "true_family"
SAVED_PRED_COL = "pred_family_final_safe"
SAVED_PMAX_COL = "pred_family_final_safe_pmax"


def default_plot_knobs() -> Dict[str, Any]:
    """Return a fresh copy of the default ``PLOT_KNOBS`` dictionary."""
    knobs: Dict[str, Any] = {
        "smallest_on_top": True,

        "confidence_filter": {
            "enabled": False,
            "threshold": 0.8,
            "print_stats": True,
            "apply_to_mouse": True,
            "apply_to_human": True,
            "apply_to_true": False,
        },

        "statistical_filter": {
            "enabled": False,
            "method": "gmm",
            "percentile": 10,
            "iqr_factor": 1.5,
            "by_family": True,
            "min_cells": 50,
        },

        "alpha": 0.7,
        "edgecolor": "none",
        "sort_by_confidence": True,
        "show_confidence_hist": True,

        "reporting": {
            "mouse_family": True,
            "mouse_subtype": True,
            "human_family": True,
            "human_subtype": True,
        },

        "family_checks": {
            "enabled": True,
            "gate_source": "safe_gate",
            "use_gate_feature_names": True,
            "print_pred_family_counts": True,
            "force_recompute_saved_family": True,
            "normalize_true_family": True,
            "normalize_saved_family": True,
            "normalize_pred_family_improved": True,
            "allow_unknown": False,
            "min_accuracy": 0.6,
            "halt_on_fail": True,
            "print_confusion": True,
        },

        "plots": {
            "family_locked_subtypes": True,
            "run_grids": True,
            "include_improved_v2": True,
            "run_example_subtype_grid": True,
            "confidence_variant": "v_neo",
            "entropy_variant": "v_neo",
            "height_ref": "v_neo",
            "show_entropy": True,
            "saved_family_show_legend": True,
            "saved_family_legend_ncol": 6,
            "save_plots": {
                "enabled": True,
                "root": "plots",
                "sets": ["backup", "current", "wip"],
                "write_sets": ["current", "wip"],
                "dpi": 200,
                "purge_extra": False,
            },
        },

        "grid": {
            "use_locked_palette": True,
            "panel_w": 10,
            "panel_h": 8,
            "legend_pad": 3.0,
            "legend_fontsize": 24,
            "legend_markersize": 24,
            "legend_ncol": None,
            "legend_y": 0.03,
            "legend_loc": "lower center",
            "legend_title": None,
            "legend_title_size": None,
            "downsample": {
                "enabled": False,
                "max_points": 20000,
                "random_state": 0,
            },
        },

        "vneo": {
            "use_family_reweight": True,
            "family_reweight_alpha": 0.7,
        },

        "recompute_duplicates": True,
    }
    return knobs


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize a 2-D array so each row sums to 1."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return x / row_sums


def _canon_fam(x: str) -> str:
    if not isinstance(x, str):
        return "UNKNOWN"
    u = x.upper()
    if u.startswith("CALB1"):
        return "CALB1"
    if u.startswith("SOX6"):
        return "SOX6"
    if u.startswith("GAD2") or u.startswith("GAD"):
        return "GAD"
    if u.startswith("LEF1"):
        return "LEF1"
    return "UNKNOWN"


def _pmax(P: np.ndarray) -> np.ndarray:
    """Return per-row max probability."""
    return np.asarray(P, dtype=float).max(axis=1)


def _normalize_label(s: str) -> str:
    """Normalize label formatting for matching."""
    s = str(s).strip()
    return s.replace(": ", ":").replace(" : ", ":").replace(" :", ":")


def infer_family_from_subtype_local(subtype: str) -> str:
    """Infer family from subtype string."""
    s = str(subtype).upper()
    if s.startswith("CALB1") or ":CALB1" in s:
        return "CALB1"
    elif s.startswith("SOX6") or ":SOX6" in s:
        return "SOX6"
    elif s.startswith("GAD") or ":GAD" in s or "GAD2" in s:
        return "GAD"
    return "UNKNOWN"


def format_height_label(
    node: str,
    node_to_leaves: Dict[str, List[str]],
) -> str:
    """Return a human-readable height label for *node*."""
    leaves = node_to_leaves.get(node, [])
    fam = infer_family_from_subtype_local(leaves[0]) if leaves else "UNKNOWN"
    fam_disp = {
        "CALB1": "Calb1",
        "SOX6": "Sox6",
        "GAD": "Gad2",
        "GAD2": "Gad2",
        "LEF1": "Lef1",
    }.get(fam, fam)
    seg = str(node).split("|")[-1]
    seg_up = seg.upper()
    if fam != "UNKNOWN" and fam in seg_up:
        return seg
    return f"{fam_disp}/{seg}"


def _height_labels_from_nodes(
    nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
    *,
    moe_module: Any = None,
) -> List[str]:
    """Compute short height labels for a list of tree *nodes*."""
    if moe_module is not None and hasattr(moe_module, "short_node_label_with_terminal"):
        return [moe_module.short_node_label_with_terminal(n, node_to_leaves) for n in nodes]
    return [format_height_label(n, node_to_leaves) for n in nodes]


def _canon_family_label(label: str) -> str:
    """Canonicalize a family label to upper-case ``CALB1/SOX6/GAD``."""
    s = str(label).strip()
    if s == "" or s.lower() == "nan":
        return "UNKNOWN"
    s_upper = s.upper()
    if s_upper.startswith("CALB1") or "CALB1" in s_upper:
        return "CALB1"
    if s_upper.startswith("SOX6") or "SOX6" in s_upper:
        return "SOX6"
    if s_upper.startswith("GAD") or "GAD2" in s_upper or "GAD" in s_upper:
        return "GAD"
    return s_upper


def normalize_family_values(
    values: Sequence[str],
    categories: Sequence[str] = FAM_ORDER,
    allow_unknown: bool = False,
) -> Tuple[pd.Categorical, int]:
    """Normalize family labels into a ``pd.Categorical``."""
    normalized = []
    unknown = 0
    for v in values:
        fam = _canon_family_label(v)
        if fam in categories:
            normalized.append(fam)
        else:
            unknown += 1
            normalized.append("UNKNOWN" if allow_unknown else np.nan)
    cats = list(categories) + (["UNKNOWN"] if allow_unknown else [])
    return pd.Categorical(normalized, categories=cats), unknown


def normalize_family_obs(
    ad: Any,
    col: str,
    categories: Sequence[str] = FAM_ORDER,
    allow_unknown: bool = False,
) -> int:
    """In-place normalize a family column in ``ad.obs``; returns unknown count."""
    if col not in ad.obs:
        return 0
    values = ad.obs[col].astype(str).values
    normalized, unknown = normalize_family_values(values, categories, allow_unknown)
    ad.obs[col] = normalized
    return unknown


def family_sanity_report(
    ad: Any,
    true_col: str,
    pred_col: str,
    label: str,
    *,
    categories: Sequence[str] = FAM_ORDER,
    print_confusion: bool = True,
) -> Optional[float]:
    """Print a family accuracy report and optionally a confusion table."""
    if true_col not in ad.obs or pred_col not in ad.obs:
        print(f"{label}: missing {true_col} or {pred_col}")
        return None
    true = pd.Series(ad.obs[true_col])
    pred = pd.Series(ad.obs[pred_col])
    mask = true.notna() & pred.notna()
    if mask.sum() == 0:
        print(f"{label}: no comparable rows (all missing)")
        return None
    true = true[mask].astype(str)
    pred = pred[mask].astype(str)
    acc = float((true.values == pred.values).mean())
    print(f"{label}: n={mask.sum()} acc={acc:.3f}")
    if print_confusion:
        ct = pd.crosstab(true, pred, dropna=False)
        try:
            from IPython.display import display  # type: ignore[import-untyped]
            display(ct)
        except ImportError:
            print(ct)
    return acc


def select_family_gate(
    knobs: Dict[str, Any],
    *,
    safe_gate: Any = None,
    saved_gate: Any = None,
    safe_dir: Optional[str] = None,
    load_family_gate_safe_fn: Any = None,
) -> Any:
    """Select the family gate object based on *knobs* configuration."""
    source = knobs["family_checks"]["gate_source"]
    if source == "safe_gate":
        if safe_gate is not None:
            return safe_gate
        if safe_dir is not None and load_family_gate_safe_fn is not None:
            return load_family_gate_safe_fn(safe_dir)
        raise ValueError("safe_gate not found. Pass it explicitly or set safe_dir + load_family_gate_safe_fn.")
    if source == "saved_gate":
        if saved_gate is not None:
            return saved_gate
        raise ValueError("saved_gate not found. Pass it explicitly.")
    raise ValueError(f"Unknown gate_source: {source}")


def build_gate_X(
    adata: Any,
    X_fallback: np.ndarray,
    gate: Dict[str, Any],
    name: str,
    knobs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build the feature matrix for the family gate, aligning features if needed."""
    if knobs is None:
        knobs = default_plot_knobs()

    if scipy.sparse.issparse(X_fallback):
        X_fallback = X_fallback.toarray()
    X_fallback = np.asarray(X_fallback, dtype=np.float32)
    if X_fallback.shape[0] != adata.n_obs:
        raise ValueError(
            f"{name}: X rows ({X_fallback.shape[0]}) != adata.n_obs ({adata.n_obs})."
        )
    if not knobs["family_checks"]["use_gate_feature_names"]:
        return X_fallback
    feature_names = gate.get("feature_names")
    if not feature_names:
        print(f"{name}: gate has no feature_names; using provided X.")
        return X_fallback
    var_names = list(adata.var_names)
    missing = [g for g in feature_names if g not in var_names]
    if missing:
        print(f"{name}: {len(missing)} gate features missing in adata.var_names; using provided X.")
        return X_fallback
    X_gate = adata[:, feature_names].X
    if scipy.sparse.issparse(X_gate):
        X_gate = X_gate.toarray()
    return np.asarray(X_gate, dtype=np.float32)


def _node_probs_for_nodes_local(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
) -> np.ndarray:
    """Aggregate leaf probabilities into node-level probabilities."""
    st2i = {s: i for i, s in enumerate(subtypes)}
    cols = []
    for node in nodes:
        leaves = node_to_leaves.get(node, [])
        idx = [st2i[s] for s in leaves if s in st2i]
        if len(idx) == 0:
            cols.append(np.zeros((P_sub.shape[0],), dtype=float))
        else:
            cols.append(P_sub[:, idx].sum(axis=1))
    return np.vstack(cols).T


def _true_height_labels_from_subtypes(
    true_labels: Sequence[str],
    nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
    use_short_labels: bool = True,
) -> np.ndarray:
    """Assign each true subtype label to its enclosing height node."""
    st2node: Dict[str, str] = {}
    for node in nodes:
        for st in node_to_leaves.get(node, []):
            st2node[str(st)] = node
    labels = np.array([st2node.get(str(st), "Unknown") for st in true_labels], dtype=object)
    if use_short_labels:
        node2short = {n: s for n, s in zip(nodes, _height_labels_from_nodes(nodes, node_to_leaves))}
        labels = np.array([node2short.get(l, l) for l in labels], dtype=object)
    return labels


def _pred_height_labels(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
    use_short_labels: bool = True,
) -> np.ndarray:
    """Predict height-node labels from leaf probabilities."""
    P_nodes = _node_probs_for_nodes_local(P_sub, subtypes, nodes, node_to_leaves)
    idx = np.asarray(P_nodes).argmax(axis=1)
    labels = np.array([nodes[i] for i in idx], dtype=object)
    if use_short_labels:
        node2short = {n: s for n, s in zip(nodes, _height_labels_from_nodes(nodes, node_to_leaves))}
        labels = np.array([node2short.get(l, l) for l in labels], dtype=object)
    return labels


def _ensure_height_available(
    groups_by_height: Optional[Dict[int, Any]],
    H: int,
    label: str,
) -> int:
    """Return *H* if available, otherwise fall back to the highest available."""
    if groups_by_height is None:
        return H
    if H in groups_by_height:
        return H
    avail = sorted(groups_by_height.keys())
    if not avail:
        return H
    new_h = avail[-1]
    print(f"{label}: requested H={H} not found; using H={new_h} instead.")
    return new_h


def compute_confidence_stats_v2(
    P_sub: np.ndarray,
    y_true: Sequence[str],
    subtypes: Sequence[str],
    threshold: Optional[float] = None,
    by_family: bool = False,
) -> pd.DataFrame:
    """Compute confidence statistics per category."""
    pmax = P_sub.max(axis=1)
    pred_idx = P_sub.argmax(axis=1)
    _ = np.array([subtypes[i] for i in pred_idx])

    y_true_norm = np.array([_normalize_label(y) for y in y_true])

    if by_family:
        categories = FAM_ORDER
        cat_map = np.array([infer_family_from_subtype_local(y) for y in y_true_norm])
    else:
        categories = np.unique(y_true_norm)
        cat_map = y_true_norm

    rows: List[Dict[str, Any]] = []
    for cat in categories:
        mask = cat_map == cat
        if mask.sum() == 0:
            continue
        cat_pmax = pmax[mask]
        row: Dict[str, Any] = {
            "category": cat,
            "n_cells": int(mask.sum()),
            "mean_conf": float(np.mean(cat_pmax)),
            "median_conf": float(np.median(cat_pmax)),
            "std_conf": float(np.std(cat_pmax)),
            "min_conf": float(np.min(cat_pmax)),
            "max_conf": float(np.max(cat_pmax)),
            "q25_conf": float(np.percentile(cat_pmax, 25)),
            "q75_conf": float(np.percentile(cat_pmax, 75)),
        }
        if threshold is not None:
            above = cat_pmax >= threshold
            row["pct_above_threshold"] = float(above.mean() * 100)
            row["n_filtered_out"] = int((~above).sum())
            row["pct_filtered_out"] = float((~above).mean() * 100)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)


def compute_family_injected_outputs_safe(
    bundle: Dict[str, Any],
    X_full: np.ndarray,
    *,
    mode: str,
    saved_gate_safe: Any,
    X_gate: np.ndarray,
    compute_family_injected_outputs_fn: Any,
) -> Dict[str, Any]:
    """Wrapper around ``compute_family_injected_outputs`` that gracefully passes ``X_gate`` when the underlying function supports it."""
    try:
        sig = inspect.signature(compute_family_injected_outputs_fn)
        if "X_gate" in sig.parameters:
            return compute_family_injected_outputs_fn(
                bundle, X_full, mode=mode, saved_gate_safe=saved_gate_safe, X_gate=X_gate
            )
    except (TypeError, ValueError):
        pass
    return compute_family_injected_outputs_fn(
        bundle, X_full, mode=mode, saved_gate_safe=saved_gate_safe
    )


def _gmm_threshold(vals: np.ndarray, fallback: float = 0.3) -> float:
    """Compute a GMM-based threshold (midpoint between two component means)."""
    vals = np.asarray(vals, dtype=float)
    if vals.size < 10:
        return fallback
    try:
        gm = GaussianMixture(n_components=2, random_state=0)
        gm.fit(vals.reshape(-1, 1))
        means = gm.means_.ravel()
        means = np.sort(means)
        if len(means) < 2:
            return float(means[0]) if means.size else fallback
        return float((means[0] + means[1]) / 2.0)
    except Exception:
        return fallback


def compute_prediction_uncertainty(
    P_sub: np.ndarray,
    method: str = "entropy_gmm",
) -> Dict[str, Any]:
    """Compute cell-level uncertainty using entropy + GMM separation."""
    P = np.asarray(P_sub, dtype=np.float64)
    N, K = P.shape

    eps = 1e-12
    P_safe = np.maximum(P, eps)
    entropy = -np.sum(P_safe * np.log(P_safe), axis=1)
    max_entropy = np.log(K)
    entropy_norm = entropy / max_entropy

    confidence = P.max(axis=1)

    if method == "entropy_gmm":
        vals = entropy_norm
        try:
            gm = GaussianMixture(n_components=2, random_state=0)
            gm.fit(vals.reshape(-1, 1))
            means = gm.means_.ravel()
            hi_idx = int(np.argmax(means))
            posteriors = gm.predict_proba(vals.reshape(-1, 1))[:, hi_idx]
            uncertain = posteriors > 0.5
            boundary_cells = np.abs(posteriors - 0.5) < 0.1
            if boundary_cells.sum() > 0:
                threshold = float(np.median(vals[boundary_cells]))
            else:
                threshold = float(np.mean(means))
        except Exception:
            threshold = float(np.percentile(vals, 90))
            uncertain = vals >= threshold

    elif method == "entropy_mad":
        med = float(np.median(entropy_norm))
        mad = float(np.median(np.abs(entropy_norm - med)))
        threshold = med + 2.0 * mad * 1.4826
        uncertain = entropy_norm >= threshold

    elif method == "confidence_gmm":
        threshold = _gmm_threshold(confidence, fallback=0.3)
        uncertain = confidence < threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "confidence": confidence,
        "uncertain_mask": uncertain,
        "threshold": float(threshold),
        "n_uncertain": int(uncertain.sum()),
        "pct_uncertain": float(uncertain.mean() * 100),
        "method": method,
    }


def compute_adaptive_thresholds(
    P_sub: np.ndarray,
    y_true: Sequence[str],
    subtypes: Sequence[str],
    method: str = "percentile",
    percentile: int = 10,
    iqr_factor: float = 1.5,
    by_family: bool = True,
    min_cells: int = 50,
) -> Dict[str, float]:
    """Compute adaptive confidence thresholds, optionally per-family."""
    pmax = P_sub.max(axis=1)

    if by_family:
        y_fam = np.array([infer_family_from_subtype_local(y) for y in y_true])
        fams = ["CALB1", "SOX6", "GAD"]
    else:
        y_fam = np.array(["ALL"] * len(pmax), dtype=object)
        fams = ["ALL"]

    thresholds: Dict[str, float] = {}
    for fam in fams:
        mask = y_fam == fam
        if mask.sum() < min_cells:
            thresholds[fam] = 0.3
            continue
        fam_pmax = pmax[mask]
        if method == "percentile":
            thresholds[fam] = float(np.percentile(fam_pmax, percentile))
        elif method == "iqr":
            q1, q3 = np.percentile(fam_pmax, [25, 75])
            iqr = q3 - q1
            thresholds[fam] = max(0.1, float(q1 - iqr_factor * iqr))
        elif method == "gmm":
            thresholds[fam] = _gmm_threshold(fam_pmax, fallback=0.3)
        else:
            thresholds[fam] = 0.3

    return thresholds


def print_confidence_histogram(P_sub: np.ndarray, title: str = "") -> None:
    """Print a text histogram of confidence values."""
    pmax = P_sub.max(axis=1)
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    hist, _ = np.histogram(pmax, bins=bins)
    total = len(pmax)

    print(f"\n{'=' * 70}")
    print(f"Confidence Histogram: {title}")
    print(f"{'=' * 70}")
    print(f"Mean: {pmax.mean():.3f}, Median: {np.median(pmax):.3f}, Std: {pmax.std():.3f}")
    print("-" * 50)
    for i in range(len(bins) - 1):
        pct = hist[i] / total * 100
        bar = "#" * int(pct / 2)
        print(f"[{bins[i]:.2f}-{bins[i + 1]:.2f}): {hist[i]:6d} ({pct:5.1f}%) {bar}")
    print(f"Total: {total}")


def compute_subtype_metrics_v2(
    y_true: Sequence[str],
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    title: str = "",
    topn: int = 25,
) -> pd.DataFrame:
    """Compute per-subtype accuracy with proper label normalization."""
    y_true_norm = np.array([_normalize_label(y) for y in y_true])
    subtypes_norm = [_normalize_label(s) for s in subtypes]
    sub_to_idx = {s: i for i, s in enumerate(subtypes_norm)}

    pmax = P_sub.max(axis=1)
    pred_idx = P_sub.argmax(axis=1)
    pred_labels = np.array([subtypes_norm[i] for i in pred_idx])

    rows: List[Dict[str, Any]] = []
    for subtype in np.unique(y_true_norm):
        mask = y_true_norm == subtype
        n = int(mask.sum())
        if n == 0:
            continue
        if subtype not in sub_to_idx:
            print(f"  Warning: '{subtype}' not in model subtypes")
            continue
        correct = pred_labels[mask] == subtype
        row: Dict[str, Any] = {
            "subtype": subtype,
            "n": n,
            "acc": float(correct.mean()),
            "mean_pmax": float(pmax[mask].mean()),
            "correct_pmax": float(pmax[mask][correct].mean()) if correct.sum() > 0 else np.nan,
            "incorrect_pmax": float(pmax[mask][~correct].mean()) if (~correct).sum() > 0 else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n", ascending=False)

    valid_mask = np.isin(y_true_norm, subtypes_norm)
    if valid_mask.sum() > 0:
        overall_acc = (pred_labels[valid_mask] == y_true_norm[valid_mask]).mean()
        print(f"\n{title}")
        print(f"Overall Accuracy: {overall_acc:.3f} (n={valid_mask.sum()})")
        print(f"Excluded (unknown subtypes): {(~valid_mask).sum()}")

    return df.head(topn)


def apply_confidence_filter(
    P_sub: np.ndarray,
    y_true: Sequence[str],
    knobs: Dict[str, Any],
    subtypes: Sequence[str],
    dataset_name: str = "",
) -> np.ndarray:
    """Apply confidence filtering."""
    pmax = P_sub.max(axis=1)
    n_total = len(pmax)

    if knobs["statistical_filter"]["enabled"]:
        sf = knobs["statistical_filter"]
        thresholds = compute_adaptive_thresholds(
            P_sub, y_true, subtypes,
            method=sf.get("method", "percentile"),
            percentile=sf.get("percentile", 10),
            iqr_factor=sf.get("iqr_factor", 1.5),
            by_family=sf.get("by_family", True),
            min_cells=sf.get("min_cells", 50),
        )
        print(f"\n{dataset_name} Adaptive Thresholds: {thresholds}")

        if sf.get("by_family", True):
            y_fam = np.array([infer_family_from_subtype_local(y) for y in y_true])
            keep_mask = np.ones(n_total, dtype=bool)
            for fam, thresh in thresholds.items():
                fam_mask = y_fam == fam
                keep_mask[fam_mask] = pmax[fam_mask] >= thresh
        else:
            thresh = thresholds.get("ALL", 0.3)
            keep_mask = pmax >= thresh

    elif knobs["confidence_filter"]["enabled"]:
        thresh = knobs["confidence_filter"]["threshold"]
        keep_mask = pmax >= thresh

    else:
        keep_mask = np.ones(n_total, dtype=bool)

    n_filtered = (~keep_mask).sum()
    if n_filtered > 0 and knobs["confidence_filter"]["print_stats"]:
        print(f"\n{dataset_name}: Filtered {n_filtered}/{n_total} cells ({n_filtered / n_total * 100:.1f}%)")
        y_fam = np.array([infer_family_from_subtype_local(y) for y in y_true])
        for fam in ["CALB1", "SOX6", "GAD"]:
            fam_mask = y_fam == fam
            fam_filtered = (~keep_mask & fam_mask).sum()
            fam_total = fam_mask.sum()
            if fam_total > 0:
                print(f"  {fam}: {fam_filtered}/{fam_total} filtered ({fam_filtered / fam_total * 100:.1f}%)")

    return keep_mask


def apply_knobs_to_umap_data(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    knobs: Dict[str, Any],
    y_true: Optional[Sequence[str]] = None,
    dataset_name: str = "",
    labels_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply all knobs to prepare indices for plotting."""
    pmax = P_sub.max(axis=1)
    pred_idx = P_sub.argmax(axis=1)
    pred_labels = np.array([subtypes[i] for i in pred_idx])

    sort_labels = pred_labels
    if labels_override is not None:
        sort_labels = np.array(labels_override)
        if sort_labels.shape[0] != pred_labels.shape[0]:
            raise ValueError(
                f"labels_override length mismatch: {sort_labels.shape[0]} != {pred_labels.shape[0]}"
            )

    stats: Dict[str, Any] = {"total": len(pmax)}

    dataset_lower = (dataset_name or "").lower()
    is_true_panel = "true" in dataset_lower
    apply_filter = (
        (dataset_lower.startswith("mouse") and knobs["confidence_filter"].get("apply_to_mouse", True))
        or (dataset_lower.startswith("human") and knobs["confidence_filter"].get("apply_to_human", True))
    )
    if is_true_panel and not knobs["confidence_filter"].get("apply_to_true", False):
        apply_filter = False

    if apply_filter and (knobs["confidence_filter"]["enabled"] or knobs["statistical_filter"]["enabled"]):
        if y_true is not None:
            keep_mask = apply_confidence_filter(P_sub, y_true, knobs, subtypes, dataset_name)
        else:
            keep_mask = pmax >= knobs["confidence_filter"]["threshold"]
    else:
        keep_mask = np.ones(len(pmax), dtype=bool)

    stats["filtered_out"] = int((~keep_mask).sum())
    stats["pct_filtered"] = float((~keep_mask).mean() * 100)

    indices = np.arange(len(pmax))[keep_mask]

    if knobs["smallest_on_top"] or knobs["sort_by_confidence"]:
        cat_counts = pd.Series(sort_labels).value_counts()
        sort_keys = []
        for idx in indices:
            cat = sort_labels[idx]
            count = cat_counts.get(cat, 0)
            conf = pmax[idx]
            primary = -count if knobs["smallest_on_top"] else count
            secondary = conf if knobs["sort_by_confidence"] else 0
            sort_keys.append((primary, secondary, idx))
        sort_keys.sort()
        indices = np.array([k[2] for k in sort_keys])

    stats["n_plotted"] = len(indices)
    stats["mean_conf_plotted"] = float(pmax[indices].mean()) if len(indices) > 0 else 0

    return indices, stats


def attach_saved_family_gate_preds_v2(
    ad: Any,
    X: np.ndarray,
    saved_gate: Any,
    prefix: str = "pred_family_saved",
    *,
    force: bool = False,
    categories: Sequence[str] = FAM_ORDER,
    allow_unknown: bool = False,
) -> None:
    """Attach saved family gate predictions to AnnData."""
    if (not force) and prefix in ad.obs:
        if categories is not None:
            normalize_family_obs(ad, prefix, categories=categories, allow_unknown=allow_unknown)
        print(f"'{prefix}' already in ad.obs (normalized)")
        return

    P_fam, fams = predict_family_gate_safe(saved_gate, X)
    pred_idx = P_fam.argmax(axis=1)
    pmax = P_fam.max(axis=1)

    pred_labels = [fams[i] for i in pred_idx]
    pred_fam, unknown = normalize_family_values(pred_labels, categories=categories, allow_unknown=allow_unknown)
    if unknown:
        print(f"Warning: {unknown} {prefix} labels outside {categories}")

    ad.obs[prefix] = pred_fam
    ad.obs[f"{prefix}_pmax"] = pmax
    print(f"Attached '{prefix}' to {ad.shape[0]} cells")


def safe_accuracy(y_true: Any, y_pred: Any) -> float:
    """Compute accuracy handling Categorical types."""
    y_true = pd.Series(y_true).astype(str).to_numpy()
    y_pred = pd.Series(y_pred).astype(str).to_numpy()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}")
    return float((y_true == y_pred).mean())


def _hard_mask_subtypes_by_family(
    P_sub: Optional[np.ndarray],
    subtypes: Sequence[str],
    P_fam: Optional[np.ndarray],
    fams: Sequence[str],
    *,
    hard: bool = True,
    power: float = 1.0,
) -> Optional[np.ndarray]:
    """Mask subtype probabilities by family predictions (hard or soft)."""
    if P_sub is None or P_fam is None:
        return P_sub
    P_sub = np.asarray(P_sub, dtype=float)
    P_fam = np.asarray(P_fam, dtype=float)

    fams_norm = [_canon_fam(f) for f in fams]
    fam_to_cols: Dict[str, List[int]] = {}
    for i, fam in enumerate(fams_norm):
        fam_to_cols.setdefault(fam, []).append(i)
    fam_order = list(fam_to_cols.keys())
    P_fam_norm = np.zeros((P_fam.shape[0], len(fam_order)), dtype=float)
    for j, fam in enumerate(fam_order):
        cols = fam_to_cols[fam]
        P_fam_norm[:, j] = P_fam[:, cols].sum(axis=1)

    sub_fams = [infer_family_from_subtype_local(st) for st in subtypes]
    if hard:
        fam_idx = P_fam_norm.argmax(axis=1)
        fam_pred = np.array([fam_order[i] for i in fam_idx], dtype=object)
        mask = np.zeros_like(P_sub, dtype=float)
        for j, fam in enumerate(sub_fams):
            if fam in fam_to_cols:
                mask[:, j] = (fam_pred == fam).astype(float)
        return _normalize_rows(P_sub * mask)

    fam_to_idx = {fam: i for i, fam in enumerate(fam_order)}
    weights = np.ones_like(P_sub, dtype=float)
    for j, fam in enumerate(sub_fams):
        if fam in fam_to_idx:
            weights[:, j] = np.maximum(P_fam_norm[:, fam_to_idx[fam]], 1e-6) ** float(power)
    return _normalize_rows(P_sub * weights)


def _hard_mask_subtypes_by_height(
    P_sub: Optional[np.ndarray],
    subtypes: Sequence[str],
    nodes: Optional[Sequence[str]],
    node_to_leaves: Dict[str, List[str]],
) -> Optional[np.ndarray]:
    """Mask subtype probabilities by height-node predictions."""
    if P_sub is None or nodes is None:
        return P_sub
    P_sub = np.asarray(P_sub, dtype=float)
    st_to_node: Dict[str, Optional[str]] = {}
    for node in nodes:
        for st in node_to_leaves.get(node, []):
            st_to_node[str(st)] = node
    st_nodes = [st_to_node.get(str(st)) for st in subtypes]
    P_nodes = _node_probs_for_nodes_local(P_sub, subtypes, nodes, node_to_leaves)
    top_idx = np.asarray(P_nodes).argmax(axis=1)
    top_nodes = np.array([nodes[i] for i in top_idx], dtype=object)
    mask = np.zeros_like(P_sub, dtype=float)
    for j, node in enumerate(st_nodes):
        if node is None:
            continue
        mask[:, j] = (top_nodes == node).astype(float)
    return _normalize_rows(P_sub * mask)


def reweight_subtypes_with_family_prior(
    P_sub: Optional[np.ndarray],
    subtypes: Sequence[str],
    P_fam: Optional[np.ndarray],
    fams: Sequence[str],
    alpha: float = 0.5,
) -> Optional[np.ndarray]:
    """Reweight subtype probabilities using family prior."""
    if P_sub is None or P_fam is None:
        return P_sub
    P_sub = np.asarray(P_sub, dtype=float)
    P_fam = np.asarray(P_fam, dtype=float)
    fam_to_idx = {f: i for i, f in enumerate(fams)}
    weights = np.ones_like(P_sub)
    for j, st in enumerate(subtypes):
        fam = infer_family_from_subtype_local(st)
        if fam in fam_to_idx:
            weights[:, j] = np.maximum(P_fam[:, fam_to_idx[fam]], 1e-6) ** alpha
    P_adj = P_sub * weights
    return _normalize_rows(P_adj)


def get_bundle(
    name: str,
    *,
    v4_bundles: Optional[Dict[str, Any]] = None,
    results_v4: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if v4_bundles is not None and name in v4_bundles:
        return v4_bundles[name]
    if results_v4 is not None and name in results_v4 and "bundle" in results_v4[name]:
        return results_v4[name]["bundle"]
    raise KeyError(f"Missing bundle for variant: {name}")


def get_mode_res(
    results_one: Dict[str, Any],
    mode: str,
    dataset: str,
) -> Any:
    """Extract mode-specific results for a dataset from *results_one*."""
    key = mode if dataset == "mouse" else f"{mode}_human"
    if key not in results_one or results_one[key] is None:
        raise ValueError(f"Missing results_one['{key}']")
    return results_one[key]


def predict_v4(
    bundle: Dict[str, Any],
    X: np.ndarray,
    mode: str,
    *,
    v4_pipe: Any,
) -> Dict[str, Any]:
    if mode == "soft":
        return v4_pipe.predict_router_soft(bundle, X)
    if mode == "hard":
        return v4_pipe.predict_router_hard(bundle, X)
    if mode == "stacked":
        return v4_pipe.predict_router_stacked(bundle, X)
    raise ValueError(f"Unsupported mode: {mode}")


def get_v4_outputs(
    variant: str,
    bundle: Dict[str, Any],
    X: np.ndarray,
    mode: str,
    dataset: str,
    *,
    v4_pipe: Any,
    results_v4: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Retrieve or compute v4 outputs for *variant*."""
    if results_v4 is not None and variant in results_v4:
        key = mode if dataset == "mouse" else f"{mode}_human"
        res = results_v4[variant].get(key)
        cached_bundle = results_v4[variant].get("bundle")
        if res is not None and cached_bundle is not None:
            same_cfg = cached_bundle.get("config") == bundle.get("config")
            same_pipe = cached_bundle.get("pipeline") == bundle.get("pipeline")
            if same_cfg and same_pipe:
                return res["P_sub"], list(res["leaves"])
    res = predict_v4(bundle, X, mode, v4_pipe=v4_pipe)
    return res["P_sub"], list(res["leaves"])


def bundle_signature(bundle: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return a hashable signature for a bundle config."""
    cfg = bundle.get("config", {}) or {}
    return (bundle.get("pipeline"), tuple(sorted(cfg.items())))


def maybe_refresh_P_sub(
    bundle: Dict[str, Any],
    X: np.ndarray,
    existing: Optional[np.ndarray],
    mode: str,
    *,
    v4_pipe: Any,
    label: str = "",
    recompute_duplicates: bool = True,
) -> Optional[np.ndarray]:
    """If notebook and improved probabilities are identical, optionally recompute."""
    if existing is None:
        return existing
    if not recompute_duplicates:
        return existing
    try:
        refreshed = predict_v4(bundle, X, mode, v4_pipe=v4_pipe)["P_sub"]
        if not np.allclose(refreshed, existing):
            print(f"{label}: refreshed P_sub (was identical).")
            return refreshed
    except Exception as e:
        print(f"{label}: refresh attempt failed ({e}); keeping existing.")
    return existing


def family_probs_from_Psub_local(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    """Aggregate leaf probabilities into family-level probabilities."""
    fams = ["CALB1", "SOX6", "GAD"]
    idx: Dict[str, List[int]] = {f: [] for f in fams}
    for j, st in enumerate(subtypes):
        fam = infer_family_from_subtype_local(st)
        if fam in idx:
            idx[fam].append(j)
    P_sub = np.asarray(P_sub, dtype=float)
    P_fam = np.vstack([
        P_sub[:, idx[f]].sum(axis=1) if idx[f] else np.zeros(P_sub.shape[0])
        for f in fams
    ]).T
    P_fam = _normalize_rows(P_fam)
    return fams, P_fam


def align_family_probs(
    P: np.ndarray,
    fam_names: Sequence[str],
    target_fams: Sequence[str] = ("CALB1", "SOX6", "GAD"),
) -> np.ndarray:
    """Re-order / zero-pad family probability columns to match *target_fams*."""
    fam_map = {str(f).upper(): i for i, f in enumerate(fam_names)}
    cols = []
    for fam in target_fams:
        idx = fam_map.get(fam, None)
        if idx is None:
            cols.append(np.zeros(P.shape[0]))
        else:
            cols.append(P[:, idx])
    return np.vstack(cols).T


def height_probs_from_Psub(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    height_nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
) -> np.ndarray:
    """Aggregate leaf probabilities into height-level probabilities."""
    sub_idx = {st: i for i, st in enumerate(subtypes)}
    P_sub = np.asarray(P_sub, dtype=float)
    rows = []
    for node in height_nodes:
        leaves = node_to_leaves.get(node, [])
        idx = [sub_idx[l] for l in leaves if l in sub_idx]
        if idx:
            rows.append(P_sub[:, idx].sum(axis=1))
        else:
            rows.append(np.zeros(P_sub.shape[0]))
    P_h = np.vstack(rows).T
    P_h = _normalize_rows(P_h)
    return P_h


def true_height_labels_from_subtypes(
    y_sub: Sequence[str],
    subtype_to_path: Dict[str, List[str]],
    H: int,
    height_nodes: Sequence[str],
) -> np.ndarray:
    """Map true subtype labels to height-node labels."""
    node_set = set(height_nodes)
    labels: List[str] = []
    for st in y_sub:
        path = subtype_to_path.get(st)
        if not path or H >= len(path):
            labels.append("UNKNOWN")
            continue
        node = "|".join(path[: H + 1])
        labels.append(node if node in node_set else "UNKNOWN")
    return np.array(labels, dtype=object)


def compute_metrics_from_probs(
    y_true: Sequence[str],
    P: np.ndarray,
    labels_all: Sequence[str],
    *,
    eval_labels: Optional[Sequence[str]] = None,
    drop_labels: Optional[Sequence[str]] = None,
    title: str = "",
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Compute overall + per-class metrics from probability matrix *P*."""
    y_true_arr = np.asarray(y_true, dtype=object)
    P = np.asarray(P, dtype=float)
    labels_all_list = list(labels_all)
    if eval_labels is None:
        eval_labels = labels_all_list
    eval_labels_list = list(eval_labels)

    mask = np.ones(len(y_true_arr), dtype=bool)
    if drop_labels:
        mask = ~np.isin(y_true_arr, list(drop_labels))
    y_true_arr = y_true_arr[mask]
    P = P[mask]

    if y_true_arr.size == 0:
        print(f"{title}: no samples after filtering.")
        return None

    pred_idx = P.argmax(axis=1)
    y_pred = np.array([labels_all_list[i] for i in pred_idx], dtype=object)

    acc = accuracy_score(y_true_arr, y_pred)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred)

    per_class_rows: List[Dict[str, Any]] = []
    for lab in eval_labels_list:
        y_true_bin = y_true_arr == lab
        y_pred_bin = y_pred == lab
        tp = int((y_true_bin & y_pred_bin).sum())
        fp = int((~y_true_bin & y_pred_bin).sum())
        fn = int((y_true_bin & ~y_pred_bin).sum())
        tn = int((~y_true_bin & ~y_pred_bin).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        ppv = precision

        auc_val: Optional[float] = None
        pr_auc: Optional[float] = None
        if lab in labels_all_list:
            lab_idx = labels_all_list.index(lab)
            if len(np.unique(y_true_bin)) > 1:
                try:
                    auc_val = roc_auc_score(y_true_bin, P[:, lab_idx])
                    pr_auc = average_precision_score(y_true_bin, P[:, lab_idx])
                except Exception:
                    auc_val = None
                    pr_auc = None
        per_class_rows.append({
            "label": lab,
            "support": int(y_true_bin.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ppv": ppv,
            "npv": npv,
            "roc_auc": auc_val,
            "pr_auc": pr_auc,
        })

    summary = pd.DataFrame({
        "n": [int(len(y_true_arr))],
        "accuracy": [acc],
        "balanced_accuracy": [bal_acc],
    })
    per_class = pd.DataFrame(per_class_rows)
    return summary, per_class


def show_table(
    df: pd.DataFrame,
    title: str,
    *,
    float_cols: Optional[List[str]] = None,
    head: Optional[int] = None,
) -> None:
    """Pretty-print a DataFrame with an optional title."""
    df = df.copy()
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].astype(float).round(3)
    if head is not None:
        df = df.head(head)
    print("\n" + title)
    try:
        from IPython.display import display  # type: ignore[import-untyped]
        display(df)
    except ImportError:
        print(df)


def show_metrics(
    title: str,
    y_true: Sequence[str],
    P: np.ndarray,
    labels_all: Sequence[str],
    eval_labels: Optional[Sequence[str]] = None,
    drop_labels: Optional[Sequence[str]] = None,
) -> None:
    """Compute and display metrics for *title*."""
    res = compute_metrics_from_probs(
        y_true, P, labels_all,
        eval_labels=eval_labels,
        drop_labels=drop_labels,
        title=title,
    )
    if res is None:
        return
    summary, per_class = res
    show_table(summary, f"[Metrics] {title}", float_cols=["accuracy", "balanced_accuracy"])
    show_table(
        per_class,
        f"[Per-class] {title}",
        float_cols=["precision", "recall", "f1", "ppv", "npv", "roc_auc", "pr_auc"],
    )


def report_prob_similarity(
    name_a: str,
    P_a: Optional[np.ndarray],
    name_b: str,
    P_b: Optional[np.ndarray],
    indices: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> None:
    """Warn if two probability matrices are identical."""
    if P_a is None or P_b is None:
        return
    if indices is not None:
        P_a = np.asarray(P_a)[indices]
        P_b = np.asarray(P_b)[indices]
    if P_a.shape != P_b.shape:
        print(f"{name_a} vs {name_b}: shapes differ {P_a.shape} vs {P_b.shape}")
        return
    same = np.allclose(P_a, P_b, atol=tol, rtol=0)
    if same:
        print(f"WARNING: {name_a} and {name_b} probabilities are identical within tol={tol}.")


def report_gad_presence(subtypes: Optional[Sequence[str]], label: str) -> bool:
    """Check whether any GAD subtype is present in *subtypes*."""
    if subtypes is None:
        return False
    has_gad = any(
        str(s).upper().startswith("GAD") or str(s).upper().startswith("GAD2")
        for s in subtypes
    )
    if not has_gad:
        print(f"NOTE: {label} has no GAD subtypes; P_sub-based predictions cannot be GAD.")
    return has_gad


def build_global_palette(
    subtypes: Sequence[str],
    families: Sequence[str] = ("CALB1", "SOX6", "GAD"),
    cmap_name: str = "tab20",
) -> Dict[str, Tuple[float, ...]]:
    """Build a consistent colour palette for all subtypes."""
    import matplotlib.pyplot as plt  # noqa: E402 – lazy import

    cmap = plt.get_cmap(cmap_name)

    family_subtypes: Dict[str, List[str]] = {fam: [] for fam in families}
    other_subtypes: List[str] = []

    for st in subtypes:
        st_upper = str(st).upper()
        if st_upper.startswith("CALB1") or ":CALB1" in st_upper:
            family_subtypes["CALB1"].append(st)
        elif st_upper.startswith("SOX6") or ":SOX6" in st_upper:
            family_subtypes["SOX6"].append(st)
        elif st_upper.startswith("GAD") or "GAD2" in st_upper:
            family_subtypes["GAD"].append(st)
        else:
            other_subtypes.append(st)

    for fam in families:
        family_subtypes[fam] = sorted(family_subtypes[fam])
    other_subtypes = sorted(other_subtypes)

    FAMILY_COLORS = {
        "CALB1": (0.12, 0.47, 0.71),
        "SOX6": (0.17, 0.63, 0.17),
        "GAD": (0.84, 0.15, 0.16),
    }

    palette: Dict[str, Tuple[float, ...]] = {}
    for fam in families:
        base = np.array(FAMILY_COLORS.get(fam, (0.5, 0.5, 0.5)))
        subtypes_in_fam = family_subtypes[fam]
        n = len(subtypes_in_fam)
        for i, st in enumerate(subtypes_in_fam):
            factor = 0.7 + 0.6 * (i / max(1, n - 1))
            color = np.clip(base * factor, 0, 1)
            palette[st] = tuple(color)

    for i, st in enumerate(other_subtypes):
        palette[st] = cmap(i / max(1, len(other_subtypes) - 1))[:3]

    return palette


def build_locked_palette_from_categories(
    categories: Sequence[str],
    cmap_name: str = "tab20",
) -> Dict[str, Any]:
    """Build a deterministic colour palette for a list of category labels."""
    import matplotlib.pyplot as plt  # noqa: E402 – lazy import

    labels: List[str] = []
    for c in categories:
        if c is None:
            continue
        s = str(c).strip()
        if not s or s.lower() == "nan":
            continue
        if s not in labels:
            labels.append(s)

    if not labels:
        return {}

    fam_palette = {
        "CALB1": "#1f77b4",
        "SOX6": "#2ca02c",
        "GAD": "#d62728",
        "LEF1": "#9467bd",
    }

    def _family_key(label: str) -> str:
        key = str(label).strip().upper()
        if key == "GAD2":
            key = "GAD"
        return key

    family_only = all(_family_key(lab) in fam_palette for lab in labels)
    if family_only:
        return {lab: fam_palette[_family_key(lab)] for lab in labels}

    cmap = plt.get_cmap(cmap_name)
    denom = max(1, len(labels) - 1)
    return {lab: cmap(i / denom) for i, lab in enumerate(labels)}


def attach_v4_family_from_Psub(
    adata: Any,
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    prefix: str = "pred_family",
) -> Any:
    """Attach predicted family labels (derived from subtype probs) to ``adata.obs``."""
    P_sub = np.asarray(P_sub)
    if P_sub.ndim != 2:
        raise ValueError(f"P_sub must be 2D, got shape={P_sub.shape}")
    if len(subtypes) != P_sub.shape[1]:
        raise ValueError("subtypes length must match P_sub columns")
    pred_idx = P_sub.argmax(axis=1)
    pred_sub = np.array([subtypes[i] for i in pred_idx])
    pred_fam = pd.Series(pred_sub).astype(str).map(_canon_fam).values
    adata.obs[prefix] = pred_fam
    adata.obs[prefix + "_pmax"] = P_sub.max(axis=1)
    return adata


def plot_umap_with_consistent_colors(
    ad: Any,
    labels_col: Optional[str],
    P_sub: Optional[np.ndarray],
    subtypes: Optional[Sequence[str]],
    title: str,
    *,
    s: int = 7,
    palette: Optional[Dict[str, Any]] = None,
    ax: Any = None,
    show_legend: bool = True,
    knobs: Optional[Dict[str, Any]] = None,
    apply_knobs: bool = True,
    y_true: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    legend_fontsize: Optional[int] = None,
    legend_markersize: Optional[int] = None,
    legend_ncol: Optional[int] = None,
    legend_loc: str = "lower center",
    legend_bbox: Tuple[float, float] = (0.5, -0.12),
    force_equal_aspect: bool = False,
    global_subtype_palette: Optional[Dict[str, Any]] = None,
) -> Any:
    """Plot UMAP with consistent subtype colours."""
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.lines import Line2D  # noqa: E402

    if palette is None:
        palette = global_subtype_palette if global_subtype_palette is not None else {}
    if knobs is None:
        knobs = default_plot_knobs()

    umap = ad.obsm["X_umap"]

    if labels_col is not None and labels_col in ad.obs:
        labels = ad.obs[labels_col].astype(str).values
        if apply_knobs and P_sub is not None and subtypes is not None:
            indices, _ = apply_knobs_to_umap_data(
                P_sub, subtypes, knobs, y_true=y_true,
                dataset_name=dataset_name or title, labels_override=labels,
            )
        else:
            indices = np.arange(len(labels))
            if apply_knobs and (knobs.get("smallest_on_top") or knobs.get("sort_by_confidence")):
                cat_counts = pd.Series(labels).value_counts()
                sort_keys = []
                for idx in indices:
                    cat = labels[idx]
                    count = cat_counts.get(cat, 0)
                    primary = -count if knobs.get("smallest_on_top") else count
                    sort_keys.append((primary, idx))
                sort_keys.sort()
                indices = np.array([k[1] for k in sort_keys])
    else:
        pred_idx = P_sub.argmax(axis=1)
        labels = np.array([subtypes[i] for i in pred_idx])
        if apply_knobs:
            indices, _ = apply_knobs_to_umap_data(
                P_sub, subtypes, knobs, y_true=y_true,
                dataset_name=dataset_name or title, labels_override=labels,
            )
        else:
            indices = np.arange(len(labels))

    if len(indices) == 0:
        print(f"Warning: no points after filtering for '{title}'. Plotting all points.")
        indices = np.arange(len(labels))

    umap = umap[indices]
    labels = labels[indices]

    palette_norm = {_normalize_label(k): v for k, v in palette.items()}
    auto_palette = build_locked_palette_from_categories(labels)
    for k, v in auto_palette.items():
        palette_norm.setdefault(_normalize_label(k), v)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = [palette_norm.get(_normalize_label(l), (0.5, 0.5, 0.5)) for l in labels]
    ax.scatter(
        umap[:, 0], umap[:, 1], c=colors, s=s,
        alpha=knobs.get("alpha", 0.7), edgecolors=knobs.get("edgecolor", "none"),
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    if force_equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if show_legend:
        unique_labels = sorted(set(labels))
        handles = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=palette_norm.get(_normalize_label(l), (0.5, 0.5, 0.5)),
                markersize=legend_markersize or 12, label=l,
            )
            for l in unique_labels
        ]
        ncol = legend_ncol or min(6, max(1, len(handles) // 2))
        ax.legend(
            handles=handles, loc=legend_loc, bbox_to_anchor=legend_bbox,
            fontsize=legend_fontsize or 12, frameon=False, ncol=ncol,
        )

    return ax


def plot_umap_with_continuous_values(
    ad: Any,
    values: np.ndarray,
    title: str,
    *,
    s: int = 7,
    ax: Any = None,
    cmap: str = "viridis",
    plot_indices: Optional[np.ndarray] = None,
    sort_by_value: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: Optional[str] = None,
    force_equal_aspect: bool = False,
) -> Any:
    """Plot UMAP coloured by a continuous value (e.g. confidence, entropy)."""
    import matplotlib.pyplot as plt  # noqa: E402

    values = np.asarray(values)
    umap = ad.obsm["X_umap"]

    if plot_indices is not None:
        if len(plot_indices) == 0:
            plot_indices = None
        else:
            umap = umap[plot_indices]
            values = values[plot_indices]

    if sort_by_value:
        order = np.argsort(values)
        umap = umap[order]
        values = values[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        umap[:, 0], umap[:, 1], c=values, s=s, cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolors="none",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    if force_equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    return ax


SAVE_FILES: Dict[str, str] = {
    "saved_family_subtype": "01_saved_family_subtype.png",
    "mouse_family_grid": "02_mouse_family_grid.png",
    "human_family_grid": "03_human_family_grid.png",
    "mouse_height_grid": "04_mouse_height_h6.png",
    "human_height_grid": "05_human_height_h6.png",
    "mouse_subtype_grid": "06_mouse_subtype.png",
    "human_subtype_grid": "07_human_subtype.png",
    "vneo_entropy": "08_vneo_entropy.png",
}


def _ensure_plot_dirs(
    cfg: Dict[str, Any],
) -> Tuple[Path, List[str]]:
    """Create plot output directories defined in *cfg*."""
    root = Path(cfg.get("root", "plots"))
    sets: List[str] = cfg.get("sets", ["backup", "current", "wip"])
    for s in sets:
        (root / s).mkdir(parents=True, exist_ok=True)
    return root, sets


def _enforce_fixed_files(
    dir_path: Path,
    expected: Sequence[str],
    purge: bool = False,
) -> None:
    """Warn (and optionally purge) unexpected files in *dir_path*."""
    expected_set = set(expected)
    existing = {p.name for p in dir_path.glob("*.png")}
    missing = sorted(expected_set - existing)
    extra = sorted(existing - expected_set)
    if missing:
        print(f"[save] Missing files in {dir_path}: {missing}")
    if extra:
        msg = f"[save] Extra files in {dir_path}: {extra}"
        if purge:
            for name in extra:
                (dir_path / name).unlink(missing_ok=True)
            print(msg + " (purged)")
        else:
            print(msg + " (not purged)")


def save_fig(
    fig: Any,
    key: str,
    save_cfg: Dict[str, Any],
) -> None:
    """Save *fig* to the directories specified in *save_cfg*."""
    if not save_cfg.get("enabled", False):
        return
    if key not in SAVE_FILES:
        print(f"[save] Unknown key: {key}")
        return
    root, sets = _ensure_plot_dirs(save_cfg)
    write_sets = save_cfg.get("write_sets", ["current"])
    dpi = save_cfg.get("dpi", 200)
    for s in write_sets:
        out_path = root / s / SAVE_FILES[key]
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if save_cfg.get("purge_extra", False):
        for s in sets:
            _enforce_fixed_files(root / s, SAVE_FILES.values(), purge=True)
    else:
        for s in sets:
            _enforce_fixed_files(root / s, SAVE_FILES.values(), purge=False)


def knn_smooth_probabilities(
    P: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
    weights: str = "distance",
) -> np.ndarray:
    """Smooth probability matrix using k-nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors

    P = np.asarray(P, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)

    if P.shape[0] != coords.shape[0]:
        raise ValueError(
            f"P rows ({P.shape[0]}) != coords rows ({coords.shape[0]})"
        )

    nn = NearestNeighbors(n_neighbors=min(k + 1, P.shape[0]), metric=metric)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    P_smooth = np.zeros_like(P)
    for i in range(P.shape[0]):
        neighbor_idx = indices[i]
        neighbor_dists = distances[i]

        if weights == "distance":
            w = 1.0 / (neighbor_dists + 1e-10)
        else:
            w = np.ones(len(neighbor_idx))

        w = w / w.sum()
        P_smooth[i] = (P[neighbor_idx] * w[:, None]).sum(axis=0)

    return _normalize_rows(P_smooth)


def knn_majority_vote_labels(
    labels: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
    weights: str = "distance",
) -> np.ndarray:
    """Apply k-NN majority voting to discrete labels."""
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    labels = np.asarray(labels)
    coords = np.asarray(coords, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(labels)), metric=metric)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    voted_labels = []
    for i in range(len(labels)):
        neighbor_idx = indices[i]
        neighbor_dists = distances[i]
        neighbor_labels = labels[neighbor_idx]

        if weights == "distance":
            w = 1.0 / (neighbor_dists + 1e-10)
        else:
            w = np.ones(len(neighbor_idx))

        vote_counts: Dict[Any, float] = {}
        for lbl, weight in zip(neighbor_labels, w):
            vote_counts[lbl] = vote_counts.get(lbl, 0.0) + weight

        winner = max(vote_counts, key=vote_counts.get)
        voted_labels.append(winner)

    return np.array(voted_labels, dtype=labels.dtype)


def apply_knn_voting_to_predictions(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    coords: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
    weights: str = "distance",
    method: str = "smooth_probs",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply k-NN voting to subtype predictions."""
    subtypes = list(subtypes)

    if method == "smooth_probs":
        P_smooth = knn_smooth_probabilities(P_sub, coords, k=k, metric=metric, weights=weights)
        pred_idx = P_smooth.argmax(axis=1)
        voted_labels = np.array([subtypes[i] for i in pred_idx])
        return P_smooth, voted_labels
    elif method == "vote_labels":
        pred_idx = P_sub.argmax(axis=1)
        raw_labels = np.array([subtypes[i] for i in pred_idx])
        voted_labels = knn_majority_vote_labels(raw_labels, coords, k=k, metric=metric, weights=weights)
        return P_sub.copy(), voted_labels
    else:
        raise ValueError(f"Unknown method: {method}. Use 'smooth_probs' or 'vote_labels'.")


def predict_height_labels(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    height_nodes: Sequence[str],
    node_to_leaves: Dict[str, List[str]],
    *,
    use_short_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict height-level labels from subtype probabilities."""
    P_height = _node_probs_for_nodes_local(P_sub, subtypes, height_nodes, node_to_leaves)
    pred_idx = P_height.argmax(axis=1)
    pred_labels = np.array([height_nodes[i] for i in pred_idx], dtype=object)

    if use_short_labels:
        node2short = {
            n: s for n, s in zip(
                height_nodes,
                _height_labels_from_nodes(height_nodes, node_to_leaves)
            )
        }
        pred_labels = np.array([node2short.get(l, l) for l in pred_labels], dtype=object)

    return pred_labels, P_height


def generate_flagship_figure(
    ad_human: Any,
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    *,
    height_nodes: Optional[Sequence[str]] = None,
    node_to_leaves: Optional[Dict[str, List[str]]] = None,
    output_dir: Optional[Path] = None,
    knn_k: int = 15,
    knn_weights: str = "distance",
    subtype_palette: Optional[Dict[str, str]] = None,
    family_palette: Optional[Dict[str, str]] = None,
    height_palette: Optional[Dict[str, str]] = None,
    figsize_subtype: Tuple[int, int] = (28, 9),
    figsize_height: Tuple[int, int] = (28, 9),
    dpi: int = 300,
    save: bool = True,
    model_label: str = "Improved + Family",
) -> Dict[str, Any]:
    """Generate flagship figures with k-NN voting for both subtype and height levels."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    subtypes = list(subtypes)
    umap = ad_human.obsm["X_umap"]

    if subtype_palette is None:
        subtype_palette = dict(COOLOR_SUBTYPE_PALETTE)
    if family_palette is None:
        family_palette = {"Sox6": "#2ca02c", "Calb1": "#1f77b4", "Gad2": "#d62728"}

    print(f"Applying k-NN voting (k={knn_k}, weights={knn_weights})...")
    P_smooth, pred_subtypes = apply_knn_voting_to_predictions(
        P_sub, subtypes, umap, k=knn_k, weights=knn_weights, method="smooth_probs"
    )

    fam_names = ["Sox6", "Calb1", "Gad2"]
    P_fam = np.zeros((P_smooth.shape[0], len(fam_names)))
    for j, st in enumerate(subtypes):
        fam = infer_family_from_subtype_local(st)
        fam_lower = {"CALB1": "Calb1", "SOX6": "Sox6", "GAD": "Gad2"}.get(fam, None)
        if fam_lower and fam_lower in fam_names:
            P_fam[:, fam_names.index(fam_lower)] += P_smooth[:, j]
    P_fam = _normalize_rows(P_fam)
    pred_families = np.array([fam_names[i] for i in P_fam.argmax(axis=1)])

    ad_human.obs["pred_subtype_knn"] = pred_subtypes
    ad_human.obs["pred_family_knn"] = pd.Categorical(pred_families, categories=fam_names)

    full_subtype_palette = dict(subtype_palette)
    for st in subtypes:
        if st not in full_subtype_palette:
            fam = infer_family_from_subtype_local(st)
            if fam == "SOX6":
                full_subtype_palette[st] = "#98df8a"
            elif fam == "CALB1":
                full_subtype_palette[st] = "#9edae5"
            else:
                full_subtype_palette[st] = "#f7b6d2"

    results = {
        "P_smooth": P_smooth,
        "pred_subtypes": pred_subtypes,
        "pred_families": pred_families,
        "figures": {},
    }

    def _get_human_ct_palette(cell_types):
        pal = {}
        for ct in cell_types:
            u = str(ct).upper()
            if u.startswith("SOX6"):
                pal[ct] = "#2ca02c"
            elif u.startswith("CALB1"):
                pal[ct] = "#1f77b4"
            else:
                pal[ct] = "#d62728"
        return pal

    human_ct_palette = _get_human_ct_palette(ad_human.obs["Cell_Type"].unique())

    print(f"Generating subtype flagship figure [{model_label}]...")
    fig_sub = plt.figure(figsize=figsize_subtype, dpi=dpi)

    ax1 = fig_sub.add_axes([0.02, 0.1, 0.28, 0.85])
    sc.pl.umap(ad_human, color="Cell_Type", ax=ax1, show=False,
               palette=human_ct_palette, s=5, alpha=0.8, frameon=True)
    ax1.set_title("A. True Human Cell Types", fontweight="bold", fontsize=16, pad=10)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    ax2 = fig_sub.add_axes([0.35, 0.1, 0.28, 0.85])
    sc.pl.umap(ad_human, color="pred_family_knn", ax=ax2, show=False,
               palette=family_palette, s=5, alpha=0.8, frameon=True)
    ax2.set_title(f"B. Predicted Family ({model_label})", fontweight="bold", fontsize=16, pad=10)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)

    ax3 = fig_sub.add_axes([0.68, 0.1, 0.28, 0.85])
    sc.pl.umap(ad_human, color="pred_subtype_knn", ax=ax3, show=False,
               palette=full_subtype_palette, s=5, alpha=0.8, frameon=True)
    ax3.set_title(f"C. Predicted Subtypes ({model_label})", fontweight="bold", fontsize=16, pad=10)
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7, ncol=2)

    if save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path_sub = output_dir / "human_flagship_subtype_knn.png"
        fig_sub.savefig(path_sub, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.3)
        path_sub_pdf = output_dir / "human_flagship_subtype_knn.pdf"
        fig_sub.savefig(path_sub_pdf, format='pdf', bbox_inches="tight", pad_inches=0.3)
        print(f"  Saved: {path_sub}")
        results["figures"]["subtype_png"] = str(path_sub)
        results["figures"]["subtype_pdf"] = str(path_sub_pdf)
    plt.close(fig_sub)

    if height_nodes is not None and node_to_leaves is not None:
        print(f"Generating height 5 flagship figure [{model_label}]...")

        pred_height_labels, P_height = predict_height_labels(
            P_smooth, subtypes, height_nodes, node_to_leaves, use_short_labels=True
        )
        ad_human.obs["pred_height5_knn"] = pred_height_labels

        if height_palette is None:
            height_palette = build_locked_palette_from_categories(pred_height_labels)

        fig_h5 = plt.figure(figsize=figsize_height, dpi=dpi)

        ax1 = fig_h5.add_axes([0.02, 0.1, 0.28, 0.85])
        sc.pl.umap(ad_human, color="Cell_Type", ax=ax1, show=False,
                   palette=human_ct_palette, s=5, alpha=0.8, frameon=True)
        ax1.set_title("A. True Human Cell Types", fontweight="bold", fontsize=16, pad=10)
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

        ax2 = fig_h5.add_axes([0.35, 0.1, 0.28, 0.85])
        sc.pl.umap(ad_human, color="pred_family_knn", ax=ax2, show=False,
                   palette=family_palette, s=5, alpha=0.8, frameon=True)
        ax2.set_title(f"B. Predicted Family ({model_label})", fontweight="bold", fontsize=16, pad=10)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)

        ax3 = fig_h5.add_axes([0.68, 0.1, 0.28, 0.85])
        sc.pl.umap(ad_human, color="pred_height5_knn", ax=ax3, show=False,
                   palette=height_palette, s=5, alpha=0.8, frameon=True)
        ax3.set_title(f"C. Predicted Height 5 ({model_label})", fontweight="bold", fontsize=16, pad=10)
        ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7, ncol=2)

        if save and output_dir:
            path_h5 = output_dir / "human_flagship_height5_knn.png"
            fig_h5.savefig(path_h5, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.3)
            path_h5_pdf = output_dir / "human_flagship_height5_knn.pdf"
            fig_h5.savefig(path_h5_pdf, format='pdf', bbox_inches="tight", pad_inches=0.3)
            print(f"  Saved: {path_h5}")
            results["figures"]["height5_png"] = str(path_h5)
            results["figures"]["height5_pdf"] = str(path_h5_pdf)
        plt.close(fig_h5)

        results["pred_height5"] = pred_height_labels
        results["P_height5"] = P_height

    from collections import Counter
    print("\n" + "=" * 70)
    print(f"FLAGSHIP SUMMARY [{model_label}]")
    print("=" * 70)

    fam_counts = Counter(pred_families)
    subtype_counts = Counter(pred_subtypes)

    print(f"\nPredicted family distribution:")
    for fam in fam_names:
        cnt = fam_counts.get(fam, 0)
        pct = 100 * cnt / len(pred_families)
        print(f"  {fam:6s}: {cnt:5d} cells ({pct:5.1f}%)")

    print(f"\nTop 10 predicted subtypes:")
    for st, cnt in subtype_counts.most_common(10):
        pct = 100 * cnt / len(pred_subtypes)
        print(f"  {st:25s}: {cnt:5d} ({pct:5.1f}%)")

    tafa1_count = subtype_counts.get("Sox6:Tafa1", 0)
    print(f"\n  Sox6:Tafa1 cells: {tafa1_count} ({100*tafa1_count/len(pred_subtypes):.1f}%)")

    if tafa1_count > 0:
        tafa1_mask = pred_subtypes == "Sox6:Tafa1"
        tafa1_umap = umap[tafa1_mask]
        print(f"  Tafa1 UMAP1 range: {tafa1_umap[:, 0].min():.2f} to {tafa1_umap[:, 0].max():.2f}")
        print(f"  Tafa1 UMAP2 range: {tafa1_umap[:, 1].min():.2f} to {tafa1_umap[:, 1].max():.2f}")
        print(f"  Tafa1 centroid: ({tafa1_umap[:, 0].mean():.2f}, {tafa1_umap[:, 1].mean():.2f})")

    return results
