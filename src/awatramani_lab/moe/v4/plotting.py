"""Plotting, rendering, and visualisation utilities for MoE v4 pipelines."""

from __future__ import annotations

import hashlib
import io
import math
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from .compat import predict_family_gate_safe
from .hierarchy import get_node, short_node_label_with_terminal


_DEFAULT_PLOT_KNOBS: Dict[str, Any] = {
    "smallest_on_top": False,
}


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return x / row_sums


_CANON_FAMS: List[str] = ["CALB1", "SOX6", "GAD"]


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


def _entropy(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-12, 1.0)
    return -np.sum(P * np.log(P), axis=1)


def _pmax(P: np.ndarray) -> np.ndarray:
    return np.asarray(P, dtype=float).max(axis=1)


def _pred_labels_from_P(P: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    labels = list(labels)
    idx = np.asarray(P).argmax(axis=1)
    return np.asarray([labels[i] for i in idx], dtype=object)


def _true_family_from_mouse(y_leaf: Sequence[str]) -> np.ndarray:
    return pd.Series(y_leaf).astype(str).map(_canon_fam).values


def _true_family_from_human(
    ad: Any,
    *,
    truth_family_fn: Optional[Callable[[str], str]] = None,
) -> np.ndarray:
    if "true_family" in ad.obs.columns:
        return ad.obs["true_family"].astype(str).map(_canon_fam).values
    if "Cell_Type" in ad.obs.columns and truth_family_fn is not None:
        return (
            ad.obs["Cell_Type"]
            .astype(str)
            .map(truth_family_fn)
            .astype(str)
            .map(_canon_fam)
            .values
        )
    return np.array(["UNKNOWN"] * ad.n_obs, dtype=object)


def _normalize_subtype_str(s: str) -> str:
    return s.replace(": ", ":").replace(" : ", ":").replace(" :", ":")


def infer_family_from_subtype_local(subtype: str) -> str:
    """Infer family from subtype string."""
    s = str(subtype).upper()
    if s.startswith("CALB1") or ":CALB1" in s:
        return "CALB1"
    if s.startswith("SOX6") or ":SOX6" in s:
        return "SOX6"
    if s.startswith("GAD") or ":GAD" in s or "GAD2" in s:
        return "GAD"
    return "UNKNOWN"


def format_height_label(node: str, node_to_leaves: Dict[str, List[str]]) -> str:
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
    nodes: List[str],
    node_to_leaves: Dict[str, List[str]],
) -> List[str]:
    """Produce short display labels for a list of height-partition nodes."""
    try:
        return [short_node_label_with_terminal(n, node_to_leaves) for n in nodes]
    except Exception:
        return [format_height_label(n, node_to_leaves) for n in nodes]


def build_locked_palette_from_categories(
    categories: Sequence[Any],
    cmap_name: str = "tab20",
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

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


def get_categorical_colors(
    categories: Sequence[Any],
    cmap_name: str = "tab20",
) -> Dict[str, Any]:
    """Creates a stable dictionary mapping categories to colours."""
    import matplotlib.pyplot as plt

    cats = sorted(list(set(str(c) for c in categories)))

    fam_palette = {
        "CALB1": "#1f77b4",
        "SOX6": "#2ca02c",
        "GAD": "#d62728",
    }

    if all(str(c).upper() in fam_palette for c in cats):
        return {c: fam_palette[str(c).upper()] for c in cats}

    cmap = plt.get_cmap(cmap_name)
    return {cat: cmap(i / max(1, len(cats) - 1)) for i, cat in enumerate(cats)}


def _get_legend_params(n_labels: int) -> Dict[str, Any]:
    if n_labels <= 5:
        return {"ncol": n_labels, "fs": 12}
    if n_labels <= 15:
        return {"ncol": 5, "fs": 10}
    return {"ncol": 7, "fs": 8}


def attach_v4_family_from_Psub(
    adata: Any,
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    prefix: str = "pred_family",
) -> Any:
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


def attach_saved_family_gate_to_obs_canonical(
    adata: Any,
    X: np.ndarray,
    safe_gate: Any,
    *,
    prefix: str = "saved_family_3",
) -> Tuple[str, np.ndarray]:
    """Writes ``adata.obs[prefix]`` in {CALB1, SOX6, GAD} and ``_pmax``."""
    X = np.asarray(X)
    assert X.shape[0] == adata.n_obs, (
        f"[saved gate attach] X rows ({X.shape[0]}) != adata.n_obs ({adata.n_obs}). "
        "Use the X matrix aligned with this AnnData."
    )
    P_fam, fams = predict_family_gate_safe(safe_gate, X)
    fam_to_i = {f: i for i, f in enumerate(fams)}

    P3 = np.zeros((P_fam.shape[0], 3), dtype=float)
    P3[:, 0] = P_fam[:, fam_to_i["Calb1"]]
    P3[:, 1] = P_fam[:, fam_to_i["Sox6"]]
    P3[:, 2] = P_fam[:, fam_to_i["Gad2"]]

    pred = np.asarray([_CANON_FAMS[i] for i in P3.argmax(axis=1)], dtype=object)
    adata.obs[prefix] = pred
    adata.obs[f"{prefix}_pmax"] = P3.max(axis=1).astype(float)
    return prefix, P3


def attach_saved_family_gate_preds(
    ad: Any,
    X: np.ndarray,
    saved_gate: Any,
    prefix: str = "pred_family_saved",
    *,
    attach_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Attach saved gate family predictions."""
    P_fam, fams = predict_family_gate_safe(saved_gate, X)
    if attach_fn is None:
        from .viz import attach_pred_columns_from_probs as attach_fn  # type: ignore[assignment]
    attach_fn(ad, P_fam, list(fams), prefix=prefix)  # type: ignore[misc]
    return P_fam, fams


def attach_v4_family_preds_from_Psub(
    ad: Any,
    P_sub: np.ndarray,
    leaf_names: Sequence[str],
    prefix: str,
    *,
    attach_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Attach family predictions derived by marginalising P_sub."""
    fams = ["Calb1", "Sox6", "Gad2"]
    leaf_names = list(leaf_names)

    idx: Dict[str, List[int]] = {f: [] for f in fams}
    for j, leaf in enumerate(leaf_names):
        if leaf.startswith("Calb1:"):
            idx["Calb1"].append(j)
        elif leaf.startswith("Sox6:"):
            idx["Sox6"].append(j)
        elif leaf.startswith("Gad2:"):
            idx["Gad2"].append(j)

    P_sub = np.asarray(P_sub, dtype=float)
    P_fam = np.vstack(
        [P_sub[:, idx[f]].sum(axis=1) if idx[f] else np.zeros(P_sub.shape[0]) for f in fams]
    ).T
    P_fam = P_fam / np.maximum(P_fam.sum(axis=1, keepdims=True), 1e-12)

    if attach_fn is None:
        from .viz import attach_pred_columns_from_probs as attach_fn  # type: ignore[assignment]
    attach_fn(ad, P_fam, fams, prefix=prefix)  # type: ignore[misc]
    return P_fam, fams


def ensure_true_family_human(
    ad: Any,
    *,
    celltype_col: str = "Cell_Type",
    out_col: str = "true_family",
) -> str:
    if out_col in ad.obs.columns:
        return out_col
    if celltype_col not in ad.obs.columns:
        ad.obs[out_col] = "UNKNOWN"
        return out_col
    ct = ad.obs[celltype_col].astype(str).str.upper()
    tf = np.where(
        ct.str.startswith("SOX6"),
        "SOX6",
        np.where(
            ct.str.startswith("CALB1"),
            "CALB1",
            np.where(ct.str.startswith("GAD"), "GAD", "UNKNOWN"),
        ),
    )
    ad.obs[out_col] = tf
    return out_col


def attach_saved_family_gate_legacy(
    adata: Any,
    X: np.ndarray,
    *,
    gate_improved: Any,
    family_names: Tuple[str, ...] = ("Calb1", "Sox6", "Gad2"),
    prefix: str = "pred_family_saved",
    predict_fn: Optional[Callable] = None,
    attach_fn: Optional[Callable] = None,
) -> Tuple[str, np.ndarray]:
    """Attach family gate predictions using the same path as training."""
    X = np.asarray(X)
    assert X.shape[0] == adata.n_obs, f"X rows {X.shape[0]} != adata.n_obs {adata.n_obs}"

    if predict_fn is None:
        predict_fn = _lazy_legacy_predict_family_classifier()
    P = predict_fn(gate_improved, X)

    if attach_fn is None:
        from .viz import attach_pred_columns_from_probs as attach_fn  # type: ignore[assignment]
    attach_fn(adata, P, list(family_names), prefix=prefix)  # type: ignore[misc]
    return prefix, P


def _lazy_legacy_predict_family_classifier() -> Callable:
    try:
        import MoE_helper_functions_v3 as moe_legacy  # type: ignore[import-untyped]

        return moe_legacy.predict_family_classifier
    except ImportError:
        raise ImportError(
            "Could not import legacy MoE_helper_functions_v3.  "
            "Pass predict_fn explicitly."
        )


def _get_default_render_fn() -> Callable:
    """Return ``moe.plot_umap_true_pred_pmax`` via lazy import."""
    try:
        import MoE_helper_functions_v3 as moe_legacy  # type: ignore[import-untyped]

        return moe_legacy.plot_umap_true_pred_pmax
    except ImportError:
        raise ImportError(
            "Could not import legacy MoE_helper_functions_v3.  "
            "Pass render_fn (plot_umap_true_pred_pmax) explicitly."
        )


def _fig_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = np.asarray(buf)[..., :3].copy()
        return rgb
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def _render_single_axis(
    fig: Any,
    ax: Any,
    dpi: int = 150,
    keep_legend: bool = False,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    if not keep_legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox)
    buf.seek(0)
    img = plt.imread(buf)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _render_axis_crop(fig: Any, ax: Any, dpi: int = 300) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=extent, pad_inches=0)
    buf.seek(0)
    return plt.imread(buf)


def _pad_to_same_size(
    imgs: List[np.ndarray],
    pad_value: float = 1.0,
) -> List[np.ndarray]:
    H = max(im.shape[0] for im in imgs)
    W = max(im.shape[1] for im in imgs)
    out = []
    for im in imgs:
        h, w = im.shape[:2]
        top = (H - h) // 2
        bot = (H - h) - top
        left = (W - w) // 2
        right = (W - w) - left
        out.append(
            np.pad(
                im,
                ((top, bot), (left, right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        )
    return out


def render_umap_panel_from_plot_umap(
    *,
    adata: Any,
    true_col: str,
    title: str,
    s: float,
    mode: str,
    pred_col: Optional[str] = None,
    pmax_col: Optional[str] = None,
    P_sub: Optional[np.ndarray] = None,
    subtypes: Optional[Sequence[str]] = None,
    variant: str = "improved",
    H: Optional[int] = None,
    groups_by_height: Optional[Dict[int, Any]] = None,
    node_to_leaves: Optional[Dict[str, List[str]]] = None,
    show_confidence: bool = False,
    ground_truth_height: bool = True,
    which_panel: str = "pred",
    dpi: int = 150,
    keep_legend: bool = False,
    render_fn: Optional[Callable] = None,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    if render_fn is None:
        render_fn = _get_default_render_fn()

    fig, axes = render_fn(
        adata,
        true_col=true_col,
        title=title,
        s=s,
        mode=mode,
        pred_col=pred_col,
        pmax_col=pmax_col,
        P_sub=P_sub,
        subtypes=subtypes,
        variant=variant,
        H=H,
        groups_by_height=groups_by_height,
        node_to_leaves=node_to_leaves,
        show_confidence=show_confidence,
        ground_truth_height=ground_truth_height,
        legend_outside=False,
        legend_mode="auto",
        legend_fontsize=7,
        dpi=dpi,
    )

    axes = np.atleast_1d(axes)
    if show_confidence:
        idx = {"conf": 0, "true": 1, "pred": 2}[which_panel]
    else:
        idx = {"true": 0, "pred": 1}[which_panel]

    img = _render_single_axis(fig, axes[idx], dpi=dpi, keep_legend=keep_legend)
    plt.close(fig)
    return img


def render_panel_no_legend(
    *,
    adata: Any,
    true_col: str,
    s: float,
    mode: str,
    pred_col: Optional[str] = None,
    pmax_col: Optional[str] = None,
    P_sub: Optional[np.ndarray] = None,
    subtypes: Optional[Sequence[str]] = None,
    variant: str = "",
    H: Optional[int] = None,
    groups_by_height: Optional[Dict[int, Any]] = None,
    node_to_leaves: Optional[Dict[str, List[str]]] = None,
    show_confidence: bool = False,
    ground_truth_height: bool = True,
    which_panel: str = "pred",
    dpi: int = 150,
    render_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    import matplotlib.pyplot as plt

    if render_fn is None:
        render_fn = _get_default_render_fn()

    fig, axes = render_fn(
        adata,
        true_col=true_col,
        s=s,
        mode=mode,
        pred_col=pred_col,
        pmax_col=pmax_col,
        P_sub=P_sub,
        subtypes=subtypes,
        variant=variant,
        H=H,
        groups_by_height=groups_by_height,
        node_to_leaves=node_to_leaves,
        show_confidence=show_confidence,
        ground_truth_height=ground_truth_height,
        legend_outside=False,
        dpi=dpi,
    )

    axes = np.atleast_1d(axes)
    if show_confidence:
        idx = {"conf": 0, "true": 1, "pred": 2}[which_panel]
    else:
        idx = {"true": 0, "pred": 1}[which_panel]

    ax = axes[idx]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("")

    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches=bbox)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)

    return img, ax.get_xlim(), ax.get_ylim()


def render_one_panel(
    *,
    adata: Any,
    true_col: str,
    s: float,
    mode: str,
    which_panel: str,
    pred_col: Optional[str] = None,
    pmax_col: Optional[str] = None,
    P_sub: Optional[np.ndarray] = None,
    subtypes: Optional[Sequence[str]] = None,
    variant: str = "",
    H: Optional[int] = None,
    groups_by_height: Optional[Dict[int, Any]] = None,
    node_to_leaves: Optional[Dict[str, List[str]]] = None,
    ground_truth_height: bool = True,
    show_confidence: bool = False,
    dpi: int = 150,
    keep_legend: bool = False,
    render_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Call render_fn but prevent display; return RGB image."""
    import matplotlib.pyplot as plt

    if render_fn is None:
        render_fn = _get_default_render_fn()

    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        fig, axes = render_fn(
            adata,
            true_col=true_col,
            s=s,
            mode=mode,
            pred_col=pred_col,
            pmax_col=pmax_col,
            P_sub=P_sub,
            subtypes=subtypes,
            variant=variant,
            H=H,
            groups_by_height=groups_by_height,
            node_to_leaves=node_to_leaves,
            ground_truth_height=ground_truth_height,
            show_confidence=show_confidence,
            legend_outside=False,
            dpi=dpi,
        )
        axes = np.atleast_1d(axes)

        if show_confidence:
            ax_conf, ax_true, ax_pred = axes[0], axes[1], axes[2]
        else:
            ax_true, ax_pred = axes[0], axes[1]
            ax_conf = None

        if which_panel == "conf":
            if ax_conf is None:
                raise ValueError("Requested conf panel but show_confidence=False.")
            ax_use = ax_conf
        elif which_panel == "true":
            ax_use = ax_true
        elif which_panel == "pred":
            ax_use = ax_pred
        else:
            raise ValueError("which_panel must be one of: conf/true/pred")

        ax_use.set_title("")
        if not keep_legend:
            leg = ax_use.get_legend()
            if leg is not None:
                leg.remove()

        img = _fig_to_rgb(fig)
        plt.close(fig)
    finally:
        if was_interactive:
            plt.ion()
    return img


def render_clean_axis(
    *,
    adata: Any,
    true_col: str,
    s: float,
    mode: str,
    which_panel: str,
    render_fn: Optional[Callable] = None,
    plot_knobs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    if render_fn is None:
        render_fn = _get_default_render_fn()
    if plot_knobs is None:
        plot_knobs = _DEFAULT_PLOT_KNOBS

    plt.ioff()
    dpi = kwargs.get("dpi", 150)
    keep_legend = kwargs.pop("keep_legend", False)
    plot_indices = kwargs.pop("plot_indices", None)
    title_text = kwargs.pop("title", None)
    show_conf = kwargs.get("show_confidence", False)

    if plot_indices is not None:
        if len(plot_indices) == 0:
            print(f"Warning: no points after filtering for '{title_text or which_panel}'. Plotting all points.")
            plot_indices = None
        else:
            adata = adata[plot_indices].copy()
            if "P_sub" in kwargs and kwargs["P_sub"] is not None:
                kwargs["P_sub"] = np.asarray(kwargs["P_sub"])[plot_indices]

    adata, kwargs = _maybe_sort_for_panel(
        adata, true_col, mode, which_panel, kwargs, plot_knobs=plot_knobs,
    )

    if keep_legend and "legend_mode" not in kwargs:
        kwargs["legend_mode"] = "right"

    fig, axes = render_fn(
        adata, true_col=true_col, s=s, mode=mode, legend_outside=False, **kwargs,
    )
    axes = np.atleast_1d(axes)

    if show_conf:
        idx_map = {"conf": 0, "true": 1, "pred": 2}
    else:
        idx_map = {"true": 0, "pred": 1}

    ax = axes[idx_map[which_panel]]
    ax.set_title("")

    _apply_smallest_on_top(ax, enabled=plot_knobs.get("smallest_on_top", False))

    if not keep_legend:
        if ax.get_legend():
            ax.get_legend().remove()

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.canvas.draw()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=extent, pad_inches=0)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img


def show_grid(
    images: List[np.ndarray],
    titles: List[str],
    nrows: int,
    ncols: int,
    *,
    figsize: Tuple[float, float] = (12, 8),
    suptitle: Optional[str] = None,
    shared_legend: Optional[List[Tuple[str, Any]]] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    images = _pad_to_same_size(images)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    axs = np.atleast_2d(axs)

    for i, (img, ttl) in enumerate(zip(images, titles)):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_title(ttl)
        ax.axis("off")

    for j in range(len(images), nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    if shared_legend:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=7,
                label=lab,
                markerfacecolor=col,
            )
            for lab, col in shared_legend
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(shared_legend),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97] if shared_legend else None)
    plt.show()


def make_family_grid(
    adata: Any,
    true_col: str,
    s: float,
    P_nb: np.ndarray,
    P_imp: np.ndarray,
    title: str,
    *,
    subtypes_nb: Sequence[str],
    subtypes_imp: Sequence[str],
    shared_family_legend: Optional[List[Tuple[str, Any]]] = None,
    dpi: int = 300,
    mode_label: str = "stacked",
    render_fn: Optional[Callable] = None,
) -> None:
    imgs: List[np.ndarray] = []
    titles: List[str] = []

    imgs.append(render_umap_panel_from_plot_umap(
        adata=adata, true_col=true_col, title=title, s=s,
        mode="cols", pred_col=true_col, pmax_col=None,
        show_confidence=False, ground_truth_height=True, which_panel="pred",
        dpi=dpi, keep_legend=False, render_fn=render_fn,
    ))
    titles.append("True Family")

    imgs.append(render_umap_panel_from_plot_umap(
        adata=adata, true_col=true_col, title=title, s=s,
        mode="cols", pred_col="pred_family_saved", pmax_col="pred_family_saved_pmax",
        show_confidence=False, ground_truth_height=True, which_panel="pred",
        dpi=dpi, keep_legend=False, render_fn=render_fn,
    ))
    titles.append("Pred Family (Saved Gate)")

    imgs.append(render_umap_panel_from_plot_umap(
        adata=adata, true_col=true_col, title=title, s=s,
        mode="family", P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        show_confidence=False, ground_truth_height=True, which_panel="pred",
        dpi=dpi, keep_legend=False, render_fn=render_fn,
    ))
    titles.append("Pred Family (Notebook)")

    imgs.append(render_umap_panel_from_plot_umap(
        adata=adata, true_col=true_col, title=title, s=s,
        mode="family", P_sub=P_imp, subtypes=subtypes_imp, variant="improved",
        show_confidence=False, ground_truth_height=True, which_panel="pred",
        dpi=dpi, keep_legend=False, render_fn=render_fn,
    ))
    titles.append(f"Pred Family (Improved/{mode_label})")

    show_grid(imgs, titles, 2, 2, figsize=(10, 8), suptitle=title,
              shared_legend=shared_family_legend)


def make_grid_figure(
    images: List[np.ndarray],
    titles: List[str],
    nrows: int,
    ncols: int,
    *,
    figsize: Tuple[float, float],
    suptitle: Optional[str] = None,
    shared_legend: Optional[List[Tuple[str, Any]]] = None,
    legend_ncol: int = 3,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    axes = np.atleast_2d(axes)

    for i, (img, ttl) in enumerate(zip(images, titles)):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_title(ttl)
        ax.axis("off")

    for j in range(len(images), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    if shared_legend:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color=col,
                label=lab,
                markersize=8,
            )
            for lab, col in shared_legend
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=legend_ncol,
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95] if shared_legend else None)
    plt.show()


def show_table(
    df: pd.DataFrame,
    title: str,
    *,
    float_cols: Optional[List[str]] = None,
    head: Optional[int] = None,
    display_fn: Optional[Callable] = None,
) -> None:
    """Print a title and display a DataFrame."""
    df = df.copy()
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].astype(float).round(3)
    if head is not None:
        df = df.head(head)
    print("\n" + title)
    if display_fn is None:
        try:
            from IPython.display import display as _display  # type: ignore[import-untyped]

            display_fn = _display
        except ImportError:
            display_fn = print
    display_fn(df)


def table_human_family_metrics(
    true_family: Sequence[str],
    pred_family: Sequence[str],
    title: str,
    *,
    display_fn: Optional[Callable] = None,
) -> Optional[Tuple[float, pd.DataFrame]]:
    tf = pd.Series(true_family).astype(str)
    pf = pd.Series(pred_family).astype(str)
    mask = tf.isin(["SOX6", "CALB1", "GAD", "LEF1"])
    if int(mask.sum()) == 0:
        print(f"\n[Human family] {title}: no usable true labels.")
        return None
    acc = float((tf[mask].values == pf[mask].values).mean())
    ctab = pd.crosstab(tf[mask], pf[mask], normalize="index").round(3)
    show_table(
        pd.DataFrame({"n_eval": [int(mask.sum())], "acc": [acc]}),
        f"[Table] Human family -- {title}",
        float_cols=["acc"],
        display_fn=display_fn,
    )
    show_table(
        ctab.reset_index(),
        f"[Table] Human family confusion -- {title}",
        display_fn=display_fn,
    )
    return acc, ctab


def _make_handles_for_categories(
    categories: Sequence[str],
    cmap_name: str = "tab20",
) -> Tuple[list, list]:
    import matplotlib.pyplot as plt

    cats = list(pd.Categorical(categories).categories)
    cmap = plt.get_cmap(cmap_name)

    handles = []
    for i, lab in enumerate(cats):
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=cmap(i / max(1, len(cats) - 1)),
                markeredgecolor="none",
                markersize=8,
                label=str(lab),
            )
        )
    return handles, cats


def _build_locked_legend_handles(
    locked_palette: Dict[str, Any],
    markersize: int = 32,
) -> list:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            label=label,
            markersize=markersize,
        )
        for label, color in locked_palette.items()
    ]


def _draw_grid(
    fig: Any,
    axs: np.ndarray,
    imgs: List[np.ndarray],
    titles: List[str],
    *,
    suptitle: str,
    legend: Optional[Dict[str, Any]] = None,
) -> None:
    for ax, img, ttl in zip(axs.ravel(), imgs, titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_title(ttl, fontsize=14)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=22)

    if legend is not None:
        fig.legend(
            legend["handles"],
            legend["labels"],
            loc=legend.get("loc", "lower center"),
            ncol=legend.get("ncol", 3),
            frameon=False,
            fontsize=12,
        )

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])


def plot_height_grid_2x3(
    *,
    adata: Any,
    true_sub_col: str,
    s: float,
    title: str,
    H: int,
    P_nb: np.ndarray,
    P_imp: np.ndarray,
    P_nb_fam: np.ndarray,
    P_imp_fam: np.ndarray,
    subtypes_nb: Sequence[str],
    subtypes_imp: Sequence[str],
    groups_nb: Dict[int, Any],
    groups_imp: Dict[int, Any],
    node2leaves_nb: Dict[str, List[str]],
    node2leaves_imp: Dict[str, List[str]],
    ground_truth_height: bool,
    dpi: int = 150,
    render_fn: Optional[Callable] = None,
) -> None:
    import matplotlib.pyplot as plt

    imgs: List[np.ndarray] = []
    titles = [
        "True",
        "Notebook",
        "Improved",
        "Confidence (pmax)",
        "Notebook + Saved Family",
        "Improved + Saved Family",
    ]

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        groups_by_height=groups_nb, node_to_leaves=node2leaves_nb,
        which_panel="true", show_confidence=False,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        groups_by_height=groups_nb, node_to_leaves=node2leaves_nb,
        which_panel="pred", show_confidence=False,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_imp, subtypes=subtypes_imp, variant="improved",
        groups_by_height=groups_imp, node_to_leaves=node2leaves_imp,
        which_panel="pred", show_confidence=False,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        groups_by_height=groups_nb, node_to_leaves=node2leaves_nb,
        which_panel="conf", show_confidence=True,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_nb_fam, subtypes=subtypes_nb, variant="notebook+family",
        groups_by_height=groups_nb, node_to_leaves=node2leaves_nb,
        which_panel="pred", show_confidence=False,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="height", H=H,
        P_sub=P_imp_fam, subtypes=subtypes_imp, variant="improved+family",
        groups_by_height=groups_imp, node_to_leaves=node2leaves_imp,
        which_panel="pred", show_confidence=False,
        ground_truth_height=ground_truth_height, dpi=dpi,
        render_fn=render_fn,
    ))

    nodesH = list(groups_nb[H])
    labelsH = [str(n).split("|")[-1] for n in nodesH]
    handles, labels = _make_handles_for_categories(labelsH, cmap_name="tab20")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi)
    _draw_grid(
        fig, axs, imgs, titles,
        suptitle=title,
        legend={"handles": handles, "labels": labels, "loc": "lower center", "ncol": 3},
    )
    plt.show()


def plot_subtype_grid_2x3(
    *,
    adata: Any,
    true_sub_col: str,
    s: float,
    title: str,
    P_nb: np.ndarray,
    P_imp: np.ndarray,
    P_nb_fam: np.ndarray,
    P_imp_fam: np.ndarray,
    subtypes_nb: Sequence[str],
    subtypes_imp: Sequence[str],
    dpi: int = 150,
    show_true_legend: bool = False,
    render_fn: Optional[Callable] = None,
) -> None:
    import matplotlib.pyplot as plt

    imgs: List[np.ndarray] = []
    titles = [
        "True Subtype",
        "Notebook",
        "Improved",
        "Confidence (pmax)",
        "Notebook + Saved Family",
        "Improved + Saved Family",
    ]

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="cols", pred_col=true_sub_col,
        which_panel="pred", show_confidence=False, dpi=dpi,
        keep_legend=show_true_legend, render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="subtype", P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        which_panel="pred", show_confidence=False, dpi=dpi,
        keep_legend=False, render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="subtype", P_sub=P_imp, subtypes=subtypes_imp, variant="improved",
        which_panel="pred", show_confidence=False, dpi=dpi,
        keep_legend=False, render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="subtype", P_sub=P_nb, subtypes=subtypes_nb, variant="notebook",
        which_panel="conf", show_confidence=True, dpi=dpi,
        keep_legend=False, render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="subtype", P_sub=P_nb_fam, subtypes=subtypes_nb, variant="notebook+family",
        which_panel="pred", show_confidence=False, dpi=dpi,
        keep_legend=False, render_fn=render_fn,
    ))

    imgs.append(render_one_panel(
        adata=adata, true_col=true_sub_col, s=s,
        mode="subtype", P_sub=P_imp_fam, subtypes=subtypes_imp, variant="improved+family",
        which_panel="pred", show_confidence=False, dpi=dpi,
        keep_legend=False, render_fn=render_fn,
    ))

    subtype_labels = list(subtypes_nb)
    handles, labels = _make_handles_for_categories(subtype_labels, cmap_name="tab20")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi)
    _draw_grid(
        fig, axs, imgs, titles,
        suptitle=title,
        legend={"handles": handles, "labels": labels, "loc": "lower center", "ncol": 6},
    )
    plt.show()


def build_groups_by_height_from_paths(
    bundle: Dict[str, Any],
) -> Dict[int, List[str]]:
    """Build ``{height: [node_path_str, ...]}`` from a v4 bundle."""
    st2path = bundle["subtype_to_path"]
    node_to_leaves = bundle["node_to_leaves"]

    by_h: Dict[int, set] = {}
    for leaf, path in st2path.items():
        for d in range(1, len(path)):
            node = "|".join(path[: d + 1])
            if node in node_to_leaves:
                by_h.setdefault(d, set()).add(node)

    return {h: sorted(list(nodes)) for h, nodes in by_h.items()}


def safe_accuracy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> float:
    y_true_arr = pd.Series(y_true).astype(str).to_numpy()
    y_pred_arr = pd.Series(y_pred).astype(str).to_numpy()
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: true={y_true_arr.shape[0]} pred={y_pred_arr.shape[0]}"
        )
    return float((y_true_arr == y_pred_arr).mean())


def compute_family_injected_outputs(
    bundle: Dict[str, Any],
    X: np.ndarray,
    *,
    mode: str = "soft",
    saved_gate_safe: Any = None,
    X_gate: Optional[np.ndarray] = None,
    predict_router_soft_fn: Optional[Callable] = None,
    predict_router_hard_fn: Optional[Callable] = None,
    predict_router_stacked_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Return v4-shaped predictions with the top two gates overridden by the saved family gate."""
    assert saved_gate_safe is not None, "saved_gate_safe must be provided"
    if X_gate is None:
        X_gate = X

    if predict_router_soft_fn is None:
        from . import pipelines as _v4_pipe

        predict_router_soft_fn = _v4_pipe.predict_router_soft
    if predict_router_hard_fn is None:
        from . import pipelines as _v4_pipe

        predict_router_hard_fn = _v4_pipe.predict_router_hard
    if predict_router_stacked_fn is None:
        from . import pipelines as _v4_pipe

        predict_router_stacked_fn = _v4_pipe.predict_router_stacked

    if mode == "soft":
        base = predict_router_soft_fn(bundle, X)
    elif mode == "hard":
        base = predict_router_hard_fn(bundle, X)
    elif mode == "stacked":
        base = predict_router_stacked_fn(bundle, X)
    else:
        raise ValueError("mode must be one of: soft, hard, stacked")

    tree = bundle["tree"]
    root_key = bundle["root_key"]
    leaves = list(bundle["leaves"])
    leaf_to_idx = {l: i for i, l in enumerate(leaves)}
    node_to_leaves = bundle["node_to_leaves"]
    excluded = set(bundle.get("excluded_leaves", []))

    gate_outputs = base.get("gate_outputs", None)
    if gate_outputs is None:
        gate_outputs = predict_router_soft_fn(bundle, X)["gate_outputs"]

    P_fam, fams = predict_family_gate_safe(saved_gate_safe, X_gate)
    fam_to_i = {f: i for i, f in enumerate(fams)}
    p_gad = P_fam[:, fam_to_i["Gad2"]].astype(float)
    p_sox = P_fam[:, fam_to_i["Sox6"]].astype(float)
    p_calb = P_fam[:, fam_to_i["Calb1"]].astype(float)

    ROOT_GATE = "node|rest_of_dendrogram"
    DDC_GATE = "node|rest_of_dendrogram|Ddc-high,Slc6a3-high"

    gad_leaves = {l for l in leaves if l.startswith("Gad2:")} - excluded
    sox_leaves = {l for l in leaves if l.startswith("Sox6:")} - excluded

    def _child_is_gad(child_full: str) -> bool:
        L = set(node_to_leaves.get(child_full, [])) - excluded
        return len(L & gad_leaves) > 0

    def _child_is_sox(child_full: str) -> bool:
        L = set(node_to_leaves.get(child_full, [])) - excluded
        return len(L & sox_leaves) > 0

    if (
        ROOT_GATE in gate_outputs
        and "nodes" in gate_outputs[ROOT_GATE]
        and len(gate_outputs[ROOT_GATE]["nodes"]) == 2
    ):
        n0, n1 = gate_outputs[ROOT_GATE]["nodes"]
        if _child_is_gad(n0):
            probs = np.vstack([p_gad, 1.0 - p_gad]).T
        else:
            probs = np.vstack([1.0 - p_gad, p_gad]).T
        gate_outputs[ROOT_GATE]["probs"] = _normalize_rows(probs)

    if (
        DDC_GATE in gate_outputs
        and "nodes" in gate_outputs[DDC_GATE]
        and len(gate_outputs[DDC_GATE]["nodes"]) == 2
    ):
        denom = np.clip(p_sox + p_calb, 1e-6, None)
        p_sox_cond = p_sox / denom
        n0, n1 = gate_outputs[DDC_GATE]["nodes"]
        if _child_is_sox(n0):
            probs = np.vstack([p_sox_cond, 1.0 - p_sox_cond]).T
        else:
            probs = np.vstack([1.0 - p_sox_cond, p_sox_cond]).T
        gate_outputs[DDC_GATE]["probs"] = _normalize_rows(probs)

    X_arr = np.asarray(X, dtype=float)
    n = X_arr.shape[0]
    P_leaf = np.zeros((n, len(leaves)), dtype=float)

    def _route(path: List[str], mass: np.ndarray) -> None:
        node_path = "|".join(path)
        sub = get_node(tree, path)

        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            if not child_keys:
                return
            child_full = [f"{node_path}|{k}" for k in child_keys]

            gate = gate_outputs.get(node_path)
            if gate is None or gate.get("probs") is None:
                probs = np.full((n, len(child_keys)), 1.0 / len(child_keys))
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != child_full:
                    try:
                        idx_list = [gate_nodes.index(ch) for ch in child_full]
                        probs = probs[:, idx_list]
                    except ValueError:
                        probs = np.full((n, len(child_keys)), 1.0 / len(child_keys))

            probs = _normalize_rows(probs)
            for j, child in enumerate(child_keys):
                _route(path + [child], mass * probs[:, j])

        elif isinstance(sub, list):
            leaf_labels = list(sub)
            if not leaf_labels:
                return
            gate = gate_outputs.get(node_path)
            if gate is None or gate.get("probs") is None or len(leaf_labels) == 1:
                probs = np.full((n, len(leaf_labels)), 1.0 / len(leaf_labels))
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes:
                    if gate_nodes == leaf_labels:
                        pass
                    else:
                        leaf_full = [f"{node_path}|{lab}" for lab in leaf_labels]
                        if gate_nodes == leaf_full:
                            pass
                        else:
                            try:
                                if all(ch in gate_nodes for ch in leaf_labels):
                                    idx_list = [gate_nodes.index(ch) for ch in leaf_labels]
                                else:
                                    idx_list = [gate_nodes.index(ch) for ch in leaf_full]
                                probs = probs[:, idx_list]
                            except Exception:
                                probs = np.full(
                                    (n, len(leaf_labels)), 1.0 / len(leaf_labels)
                                )

            probs = _normalize_rows(probs)
            for j, leaf in enumerate(leaf_labels):
                k = leaf_to_idx.get(leaf)
                if k is not None:
                    P_leaf[:, k] += mass * probs[:, j]
        else:
            return

    _route([root_key], np.ones(n, dtype=float))
    P_leaf = _normalize_rows(P_leaf)

    out = dict(base)
    out["P_sub"] = P_leaf
    out["leaves"] = leaves
    out["gate_outputs"] = gate_outputs
    return out


def _sort_order_by_label_counts(labels: Sequence[str]) -> np.ndarray:
    labels_s = pd.Series(labels).astype(str)
    counts = labels_s.value_counts()
    sort_key = labels_s.map(counts).to_numpy()
    return np.argsort(-sort_key, kind="stable")


def _node_probs_for_nodes_local(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    nodes: List[str],
    node_to_leaves: Dict[str, List[str]],
) -> np.ndarray:
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
    nodes: List[str],
    node_to_leaves: Dict[str, List[str]],
    use_short_labels: bool = True,
) -> np.ndarray:
    st2node: Dict[str, str] = {}
    for node in nodes:
        for st in node_to_leaves.get(node, []):
            st2node[str(st)] = node
    labels = np.array(
        [st2node.get(str(st), "Unknown") for st in true_labels], dtype=object,
    )
    if use_short_labels:
        node2short = {
            n: s
            for n, s in zip(nodes, _height_labels_from_nodes(nodes, node_to_leaves))
        }
        labels = np.array([node2short.get(l, l) for l in labels], dtype=object)
    return labels


def _pred_height_labels(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
    nodes: List[str],
    node_to_leaves: Dict[str, List[str]],
    use_short_labels: bool = True,
) -> np.ndarray:
    P_nodes = _node_probs_for_nodes_local(P_sub, subtypes, nodes, node_to_leaves)
    idx = np.asarray(P_nodes).argmax(axis=1)
    labels = np.array([nodes[i] for i in idx], dtype=object)
    if use_short_labels:
        node2short = {
            n: s
            for n, s in zip(nodes, _height_labels_from_nodes(nodes, node_to_leaves))
        }
        labels = np.array([node2short.get(l, l) for l in labels], dtype=object)
    return labels


def _pred_family_labels(
    P_sub: np.ndarray,
    subtypes: Sequence[str],
) -> np.ndarray:
    fams = [_canon_fam(s) for s in subtypes]
    fam_to_idx: Dict[str, List[int]] = {}
    for i, fam in enumerate(fams):
        fam_to_idx.setdefault(fam, []).append(i)
    fam_order = [f for f in _CANON_FAMS if f in fam_to_idx] + [
        f for f in fam_to_idx if f not in _CANON_FAMS
    ]
    P = np.vstack(
        [
            P_sub[:, fam_to_idx[f]].sum(axis=1)
            if fam_to_idx.get(f)
            else np.zeros(P_sub.shape[0])
            for f in fam_order
        ]
    ).T
    idx = np.asarray(P).argmax(axis=1)
    return np.array([fam_order[i] for i in idx], dtype=object)


def _labels_for_sorting(
    adata: Any,
    true_col: str,
    mode: str,
    which_panel: str,
    cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    if which_panel == "conf":
        return None
    use_short = cfg.get("use_short_labels", True)
    if which_panel == "true":
        if mode == "family":
            return np.array(
                [_canon_fam(x) for x in adata.obs[true_col].astype(str)], dtype=object,
            )
        if mode == "height" and cfg.get("ground_truth_height", True):
            nodes = cfg["groups_by_height"][cfg["H"]]
            return _true_height_labels_from_subtypes(
                adata.obs[true_col].astype(str).values,
                nodes,
                cfg["node_to_leaves"],
                use_short_labels=use_short,
            )
        return adata.obs[true_col].astype(str).values

    if mode == "cols":
        pred_col = cfg.get("pred_col")
        if pred_col and pred_col in adata.obs:
            return adata.obs[pred_col].astype(str).values
        return None
    if "P_sub" not in cfg or cfg["P_sub"] is None:
        return None
    if mode == "subtype":
        return _pred_labels_from_P(cfg["P_sub"], cfg["subtypes"])
    if mode == "family":
        return _pred_family_labels(cfg["P_sub"], cfg["subtypes"])
    if mode == "height":
        nodes = cfg["groups_by_height"][cfg["H"]]
        return _pred_height_labels(
            cfg["P_sub"],
            cfg["subtypes"],
            nodes,
            cfg["node_to_leaves"],
            use_short_labels=use_short,
        )
    return None


def _maybe_sort_for_panel(
    adata: Any,
    true_col: str,
    mode: str,
    which_panel: str,
    cfg: Dict[str, Any],
    *,
    plot_knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if plot_knobs is None:
        plot_knobs = _DEFAULT_PLOT_KNOBS
    if not plot_knobs.get("smallest_on_top", False):
        return adata, cfg
    labels = _labels_for_sorting(adata, true_col, mode, which_panel, cfg)
    if labels is None:
        return adata, cfg
    order = _sort_order_by_label_counts(labels)
    if len(order) != adata.n_obs:
        return adata, cfg
    adata = adata[order].copy()
    if "P_sub" in cfg and cfg["P_sub"] is not None:
        cfg["P_sub"] = np.asarray(cfg["P_sub"])[order]
    return adata, cfg


def _norm_label_key(label: Any) -> str:
    if label is None:
        return ""
    s = str(label).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _apply_locked_palette_to_axis(
    ax: Any,
    locked_palette: Optional[Dict[str, Any]],
) -> None:
    """Apply locked palette to categorical scatter."""
    import matplotlib.colors as mcolors

    if locked_palette is None:
        return

    norm_map = {_norm_label_key(k): v for k, v in locked_palette.items()}
    applied = False

    for coll in list(ax.collections):
        try:
            lab = coll.get_label()
        except Exception:
            lab = None
        if not lab or lab == "_nolegend_":
            continue
        key = _norm_label_key(lab)
        if key not in norm_map:
            continue
        color = norm_map[key]
        try:
            n = len(coll.get_offsets())
        except Exception:
            n = 0
        if n > 0:
            coll.set_facecolor(np.tile(color, (n, 1)))
            coll.set_edgecolor("none")
            applied = True

    if not applied and ax.collections:
        coll = ax.collections[0]
        leg = ax.get_legend()
        if leg is not None:
            labels = [t.get_text() for t in leg.get_texts()]
            colors = []
            for lab in labels:
                key = _norm_label_key(lab)
                if key in norm_map:
                    colors.append(norm_map[key])
                else:
                    colors.append(mcolors.to_rgba("gray"))
            cmap = mcolors.ListedColormap(colors)
            coll.set_cmap(cmap)
            try:
                coll.set_clim(-0.5, len(colors) - 0.5)
            except Exception:
                pass

            handles = getattr(leg, "legendHandles", None)
            if handles is None:
                handles = getattr(leg, "legend_handles", None)
            if handles is None:
                handles = []
            for handle, color in zip(handles, colors):
                if hasattr(handle, "set_markerfacecolor"):
                    handle.set_markerfacecolor(color)
                if hasattr(handle, "set_markeredgecolor"):
                    handle.set_markeredgecolor("none")


def _apply_smallest_on_top(ax: Any, enabled: bool = True) -> None:
    if not enabled:
        return
    cols = [c for c in ax.collections if hasattr(c, "get_offsets")]
    sizes = []
    for c in cols:
        try:
            n = len(c.get_offsets())
        except Exception:
            n = 0
        sizes.append((n, c))
    for rank, (_, coll) in enumerate(sorted(sizes, key=lambda x: x[0])):
        coll.set_zorder(rank + 1)


def _pick_axis_by_title(
    axes: np.ndarray,
    which_panel: str,
    show_conf: bool,
) -> Any:
    want_sets = {
        "conf": {
            "Confidence", "Pmax", "pmax", "Max Prob", "Max probability",
        },
        "true": {
            "True", "True (Height)", "True (Label)", "True (height)",
            "True (Subtype)", "True Subtype", "True Family",
        },
        "pred": {
            "Pred", "Prediction", "Predicted", "Notebook", "Improved",
            "Saved Gate",
        },
    }
    wants = want_sets.get(which_panel, set())
    for ax in axes:
        t = (ax.get_title() or "").strip()
        if t in wants:
            return ax

    idx_map = (
        {"conf": 0, "true": 1, "pred": 2}
        if show_conf
        else {"true": 0, "pred": 1}
    )
    return axes[idx_map[which_panel]]


def render_unified_panel(
    *,
    adata: Any,
    true_col: str,
    s: float,
    locked_palette: Optional[Dict[str, Any]],
    mode: str,
    which_panel: str,
    title_text: str,
    dpi: int = 300,
    render_fn: Optional[Callable] = None,
    plot_knobs: Optional[Dict[str, Any]] = None,
    **cfg: Any,
) -> Tuple[np.ndarray, str]:
    import matplotlib.pyplot as plt

    if render_fn is None:
        render_fn = _get_default_render_fn()
    if plot_knobs is None:
        plot_knobs = _DEFAULT_PLOT_KNOBS

    plt.ioff()

    plot_indices = cfg.pop("plot_indices", None)
    keep_legend = bool(cfg.pop("keep_legend", False))
    locked_palette_override = cfg.pop("locked_palette_override", None)
    cfg.pop("title", None)

    if plot_indices is not None:
        if len(plot_indices) == 0:
            print(
                f"Warning: no points after filtering for '{title_text}'. "
                "Plotting all points."
            )
            plot_indices = None
        else:
            adata = adata[plot_indices].copy()
            if "P_sub" in cfg and cfg["P_sub"] is not None:
                cfg["P_sub"] = np.asarray(cfg["P_sub"])[plot_indices]

    adata, cfg = _maybe_sort_for_panel(
        adata, true_col, mode, which_panel, cfg, plot_knobs=plot_knobs,
    )

    moe_kwargs: Dict[str, Any] = dict(
        true_col=true_col,
        s=s,
        mode=mode,
        title=None,
        legend_outside=False,
    )

    show_conf = bool(cfg.get("show_confidence", False))
    moe_kwargs["show_confidence"] = show_conf

    for kw in (
        "legend_fontsize", "legend_ncol", "legend_mode", "use_short_labels",
    ):
        if kw in cfg:
            moe_kwargs[kw] = cfg[kw]

    if keep_legend and "legend_mode" not in moe_kwargs:
        moe_kwargs["legend_mode"] = "right"

    if mode == "cols":
        moe_kwargs["pred_col"] = cfg["pred_col"]
        if cfg.get("pmax_col", None) is not None:
            moe_kwargs["pmax_col"] = cfg["pmax_col"]
    elif mode in {"family", "subtype"}:
        moe_kwargs["P_sub"] = cfg["P_sub"]
        moe_kwargs["subtypes"] = cfg["subtypes"]
    elif mode == "height":
        moe_kwargs["H"] = cfg["H"]
        moe_kwargs["P_sub"] = cfg["P_sub"]
        moe_kwargs["subtypes"] = cfg["subtypes"]
        moe_kwargs["groups_by_height"] = cfg["groups_by_height"]
        moe_kwargs["node_to_leaves"] = cfg["node_to_leaves"]
        moe_kwargs["ground_truth_height"] = cfg.get("ground_truth_height", True)
    else:
        raise ValueError(f"Unknown mode={mode}")

    fig, axes = render_fn(adata, **moe_kwargs)
    axes = np.atleast_1d(axes)

    ax = _pick_axis_by_title(axes, which_panel, show_conf)

    palette_to_use = locked_palette_override or locked_palette
    if palette_to_use is not None:
        _apply_locked_palette_to_axis(ax, palette_to_use)

    _apply_smallest_on_top(ax, enabled=plot_knobs.get("smallest_on_top", False))

    for a in axes:
        if not (keep_legend and a is ax):
            if a.get_legend() is not None:
                try:
                    a.get_legend().remove()
                except Exception:
                    pass
        a.set_title("")
        a.set_xlabel("")
        a.set_ylabel("")
        a.set_xticks([])
        a.set_yticks([])
        a.axis("off")

    if getattr(fig, "_suptitle", None) is not None:
        try:
            fig._suptitle.remove()
        except Exception:
            pass

    for t in list(fig.texts):
        try:
            t.remove()
        except Exception:
            pass

    img = _render_axis_crop(fig, ax, dpi=dpi)
    plt.close(fig)

    return img, title_text


def plot_unified_grid(
    adata: Any,
    true_col: str,
    s: float,
    title: str,
    configs: List[Dict[str, Any]],
    all_categories: Sequence[str],
    *,
    render_fn: Optional[Callable] = None,
    plot_knobs: Optional[Dict[str, Any]] = None,
    confirm_unique: bool = False,
    check_duplicates: bool = False,
    check_pred_only: bool = True,
    use_locked_palette: bool = False,
    locked_palette: Optional[Dict[str, Any]] = None,
    panel_w: float = 6,
    panel_h: float = 5,
    legend_pad: float = 1,
    legend_fontsize: Optional[float] = None,
    legend_markersize: float = 10,
    legend_ncol: Optional[int] = None,
    legend_y: float = 0.02,
    legend_loc: str = "lower center",
    legend_title: Optional[str] = None,
    legend_title_size: Optional[float] = None,
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
    return_fig: bool = False,
) -> Optional[Any]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if render_fn is None:
        render_fn = _get_default_render_fn()
    if plot_knobs is None:
        plot_knobs = _DEFAULT_PLOT_KNOBS
    if save_kwargs is None:
        save_kwargs = {}

    imgs: List[np.ndarray] = []
    color_map: Dict[str, Any]

    titles: List[str] = []
    for c in configs:
        ttl = c.get("title", "")
        if not ttl:
            ttl = f"{c.get('mode', '')}:{c.get('which_panel', '')}".strip(":")
        titles.append(ttl)

    if use_locked_palette:
        if locked_palette is None:
            locked_palette = build_locked_palette_from_categories(all_categories)
        color_map = locked_palette
    else:
        color_map = get_categorical_colors(all_categories)

    for conf, ttl in zip(configs, titles):
        if use_locked_palette:
            conf_local = dict(conf)
            mode = conf_local.pop("mode")
            which_panel = conf_local.pop("which_panel")
            img, _ = render_unified_panel(
                adata=adata,
                true_col=true_col,
                s=s,
                locked_palette=color_map,
                mode=mode,
                which_panel=which_panel,
                title_text=ttl,
                render_fn=render_fn,
                plot_knobs=plot_knobs,
                **conf_local,
            )
        else:
            img = render_clean_axis(
                adata=adata,
                true_col=true_col,
                s=s,
                render_fn=render_fn,
                plot_knobs=plot_knobs,
                **conf,
            )
        imgs.append(img)

    nrows = (len(configs) + 2) // 3 if len(configs) > 4 else 2
    ncols = 2 if len(configs) <= 4 else 3

    fig_h = panel_h * nrows + legend_pad
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, fig_h))

    for i, (ax, img, ttl) in enumerate(zip(axs.ravel(), imgs, titles)):
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_title(ttl, fontsize=14, fontweight="bold")
        ax.axis("off")

    for j in range(len(imgs), nrows * ncols):
        axs.ravel()[j].axis("off")

    if check_duplicates:
        hashes: Dict[str, List[str]] = {}
        for img, conf, ttl in zip(imgs, configs, titles):
            if check_pred_only and conf.get("which_panel") != "pred":
                continue
            h = hashlib.md5(img.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(ttl)
        dups = [v for v in hashes.values() if len(v) > 1]
        if dups:
            print(f"WARNING: duplicate panels detected in '{title}': {dups}")
        elif confirm_unique:
            print(f"No duplicate pred panels detected in '{title}'.")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=legend_markersize,
        )
        for label, color in color_map.items()
    ]

    bottom: float
    if handles:
        if legend_ncol is None:
            params = _get_legend_params(len(handles))
            ncol_leg = params["ncol"]
            fs = params["fs"] if legend_fontsize is None else legend_fontsize
        else:
            ncol_leg = legend_ncol
            fs = legend_fontsize or 9

        fig_leg = fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=ncol_leg,
            fontsize=fs,
            frameon=False,
            bbox_to_anchor=(0.5, legend_y),
            title=legend_title,
        )
        if legend_title is not None and legend_title_size is not None:
            fig_leg.set_title(legend_title, prop={"size": legend_title_size})

        legend_rows = max(1, math.ceil(len(handles) / max(1, ncol_leg)))
        legend_h_in = (fs / 72.0) * (legend_rows + 0.8)
        bottom = min(
            0.5,
            max(
                0.12,
                (legend_h_in / max(1e-6, fig_h))
                + (legend_pad / max(1e-6, fig_h)),
            ),
        )
    else:
        bottom = 0.05

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    if save_path:
        fig.savefig(save_path, **save_kwargs)
    plt.tight_layout(rect=[0, bottom, 1, 1])
    plt.show()
    if return_fig:
        return fig
    return None
