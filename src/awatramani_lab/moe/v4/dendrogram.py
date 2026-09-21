"""Horizontal hierarchy tree (dendrogram) plotting for MoE v4."""

from __future__ import annotations

import copy
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_MPL_IMPORTED = False


def _ensure_mpl():
    global _MPL_IMPORTED
    if not _MPL_IMPORTED:
        import matplotlib  # noqa: F401

        _MPL_IMPORTED = True


def default_dendrogram_config() -> Dict[str, Any]:
    """Return the default CONFIG dict for dendrogram plotting."""
    return {
        "colors": {
            "excluded_text": "#555555",
            "excluded_marker": "#999999",
            "edge_default": "#AAAAAA",
            "root_marker": "k",
            "root_bg": "white",
            "split_label_bg": "white",
            "no_score_fill": "#DDDDDD",
        },
        "spacing": {
            "min_leaf_dist": 0.07,
            "fam_gap": 0.18,
            "table_start": 0.02,
            "zone_x_start": 0.175,
            "zone_x_start_by_family": {},
            "fam_band_pad_by_family": {},
            "fam_band_pad_top_by_family": {},
            "fam_band_pad_bottom_by_family": {},
            "family_title_anchor_by_family": {},
            "table_end_pad": 0.10,
            "zone_x_end_pad": 0.0,
            "zone_x_end_min": 1.60,
            "header_y_offset": 0.15,
        },
        "table": {
            "columns": [
                {"key": "short", "header": "Subtype", "width": 0.22, "font_key": "row_main"},
                {"key": "id", "header": "Cluster #", "width": 0.12, "font_key": "row_main"},
                {"key": "genes", "header": "Signature", "width": 0.0, "font_key": "row_meta"},
            ],
        },
        "fonts": {
            "fam_label": 22,
            "root": 18,
            "split": 15,
            "header": 19,
            "row_main": 17,
            "row_meta": 14,
            "cbar": 15,
        },
        "bbox": {
            "split_pad": 0.4,
            "terminal_pad": 0.52,
            "root_pad": 0.15,
            "fam_label_pad": 0.25,
            "cbar_labelpad": 10,
        },
        "sizes": {
            "root_s": 275,
            "leaf_s": 130,
            "internal_s": 270,
            "edge_lw": 2.3,
            "edge_lw_min": 0.8,
            "edge_lw_max": 6.0,
            "edge_lw_gamma": 0.5,
            "table_lw": 0.9,
            "marker_lw": 1.6,
        },
        "positions": {
            "cbar": [0.79, 0.05, 0.17, 0.018],
            "cbar_orientation": "horizontal",
            "cbar_use_figure_coords": False,
            "axes_right": None,
            "subtype_header_x": None,
        },
        "fig": {"figsize": (32, 24)},
        "save": {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "facecolor": "white",
        },
        "y_layout": {
            "y_scale": 1.75,
            "ylim_pad_top": 0.25,
            "ylim_pad_bot": 0.30,
            "xlim_left": -0.16,
        },
        "x_layout": {
            "use_uniform_spacing": True,
            "uniform_dx": 0.14,
            "uniform_offset": 0.00,
            "manual_positions": {
                0: 0.00,
                1: 0.12,
                2: 0.26,
                3: 0.42,
                4: 0.60,
                5: 0.78,
                6: 0.92,
            },
            "max_internal_x": 0.92,
            "internal_leaf_gap": 0.04,
        },
        "edge_weighting": {
            "weigh_by": "mouse",
            "weigh_method": "cumulative",
            "scale": "log",
            "both_offset_dy": 0.010,
            "alpha": 0.80,
            "mouse_style": {
                "linestyle": "solid",
                "color": None,
                "alpha": 0.85,
                "zorder": 3,
            },
            "human_style": {
                "linestyle": (0, (7, 4)),
                "color": "#222222",
                "alpha": 0.55,
                "zorder": 2,
            },
            "show_both_legend": True,
            "legend_loc": "lower left",
            "debug_print": True,
            "debug_topk": 5,
        },
    }


CLUSTER_META: Dict[str, Dict[str, str]] = {
    "Gad2:Syndig1": {"id": "0", "genes": "Gad2, Chrm3, Zfpm2", "short": "Gad2: Syndig1"},
    "Sox6:Tmem132d": {"id": "1", "genes": "Sox6, Slc44a5, Kcnab1", "short": "Sox6: Tmem132d"},
    "Sox6:Arhgap28": {"id": "2", "genes": "Sox6, Cntnap5b, Nl1rapl2, Grid2", "short": "Sox6: Arhgap28"},
    "Sox6:March3": {"id": "3", "genes": "Sox6, Slc44a5, Pld5", "short": "Sox6: March3"},
    "Sox6:Tafa1": {"id": "4", "genes": "Sox6, Cntnap5b, Ptchd4", "short": "Sox6: Tafa1"},
    "Calb1:Pde11a": {"id": "5", "genes": "Calb1, Ntm, Atp8b1, Epha4", "short": "Calb1: Pde11a"},
    "Calb1:Kctd8": {"id": "6", "genes": "Calb1, Ntm, Kctd8", "short": "Calb1: Kctd8"},
    "Calb1:Ptprt": {"id": "7", "genes": "Calb1, Lepr, Ptprt", "short": "Calb1: Ptprt"},
    "Sox6:Kcnmb2": {"id": "8", "genes": "Sox6, Cntnap5b, Il1rapl2, Vcan, Kcnmb2", "short": "Sox6: Kcnmb2"},
    "Calb1:Sulf1": {"id": "9", "genes": "Calb1, Lepr, Cacna2d3, Zfp521, Sema5b", "short": "Calb1: Sulf1"},
    "Sox6:Vcan": {"id": "10", "genes": "Sox6, Cntnap5b, Il1rapl2, Vcan, Zfp804b", "short": "Sox6: Vcan"},
    "Calb1:Stac": {"id": "11", "genes": "Calb1, Ntm, Tmem132d, Pcdh17", "short": "Calb1: Stac"},
    "Calb1:Chrm2": {"id": "12", "genes": "Calb1, Ntm, Tmem132d, Col23a1", "short": "Calb1: Chrm2"},
    "Calb1:Sox6": {"id": "13", "genes": "Calb1, Lepr, Cacna2d3, Mob3b", "short": "Calb1: Sox6"},
    "Calb1:Lpar1": {"id": "14", "genes": "Calb1, Ntm, Rmst, Plpp4", "short": "Calb1: Lpar1"},
    "Gad2:Egfr": {"id": "15", "genes": "Gad2, Pde11a, Zeb2", "short": "Gad2: Egfr"},
    "Lef1": {"id": "17", "genes": "Lef1, Prkcd", "short": "Lef1"},
    "Calb1:Ccdc192": {"id": "18", "genes": "Calb1, Ntm, Atp8b1, Rmst, Cgnl1", "short": "Calb1: Ccdc192"},
    "Calb1:Gipr": {"id": "19", "genes": "Calb1, Lepr, Cacna2d3, Vip", "short": "Calb1: Gipr"},
    "Gad2:Ebf2": {"id": "20", "genes": "Gad2, Chrm3, Slc26a7", "short": "Gad2: Ebf2"},
}

FAMILY_PALETTE: Dict[str, Dict[str, str]] = {
    "Sox6": {"base": "#6A4C93", "light": "#E0D6EB"},
    "Calb1": {"base": "#1982C4", "light": "#D8EFFC"},
    "Gad2": {"base": "#FF9F1C", "light": "#FFF0D6"},
    "Lef1": {"base": "#FF595E", "light": "#FFE5E6"},
    "Ddc": {"base": "#8AC926", "light": "#EEF7DC"},
    "Other": {"base": "#888888", "light": "#F0F0F0"},
}


def get_fam_color(name: str) -> Dict[str, str]:
    """Return the family palette entry for a node/leaf name."""
    s = str(name)
    prefix = s.split(":", 1)[0].strip() if ":" in s else s.strip()
    if prefix in FAMILY_PALETTE:
        return FAMILY_PALETTE[prefix]
    if "Calb1" in s:
        return FAMILY_PALETTE["Calb1"]
    if "Sox6" in s:
        return FAMILY_PALETTE["Sox6"]
    if "Gad2" in s:
        return FAMILY_PALETTE["Gad2"]
    if "Lef1" in s:
        return FAMILY_PALETTE["Lef1"]
    if "Ddc" in s:
        return FAMILY_PALETTE["Ddc"]
    return FAMILY_PALETTE["Other"]


def _normalize_leaf_label(s: str) -> str:
    s = str(s).strip()
    return s.replace(": ", ":").replace(" : ", ":").replace(" :", ":")


def _clean_node_name(name: str) -> str:
    name = str(name)
    if "|" in name:
        name = name.split("|")[-1]
    parts = name.split(",")
    for p in parts:
        p = p.strip()
        if any(c in p for c in ["+", "-", "high"]):
            return p
    return name.strip()


def _compute_gate_targets(
    bundle: Dict[str, Any],
    gate_key: str,
    y_true_leaf: np.ndarray,
    excluded: set,
) -> Tuple[np.ndarray, np.ndarray]:
    out = bundle["_gate_outputs_ref"][gate_key]
    child0, child1 = out["nodes"][0], out["nodes"][1]
    node_to_leaves = bundle["node_to_leaves"]

    def leaves_for(child):
        s = set(node_to_leaves.get(child, []))
        if not s and isinstance(child, str) and ":" in child:
            s = {child}
        return s - excluded

    L1 = leaves_for(child1)
    keep = np.isin(y_true_leaf, list(leaves_for(child0))) | np.isin(
        y_true_leaf, list(L1)
    )
    return keep, np.isin(y_true_leaf, list(L1))[keep].astype(np.int8)


def build_gate_metric_map(
    results_v4_run: Dict[str, Any],
    y_sub_test: np.ndarray,
    mode: str = "soft",
    metric: str = "brier",
) -> Tuple[Dict[str, float], set]:
    """Compute per-gate metrics for colouring the dendrogram."""
    y_true = np.asarray(y_sub_test, dtype=str)
    excluded = set(
        results_v4_run.get("bundle", {}).get("excluded_leaves", [])
    )
    bundle = dict(results_v4_run["bundle"])
    gate_outputs = results_v4_run[mode]["gate_outputs"]
    bundle["_gate_outputs_ref"] = gate_outputs

    metric_map: Dict[str, float] = {}
    gate_nodes: set = set()
    for gate_key, out in gate_outputs.items():
        node_name = _clean_node_name(gate_key)
        keep, y = _compute_gate_targets(bundle, gate_key, y_true, excluded)
        if len(keep) == 0:
            continue
        p = np.asarray(out["probs"], float)[keep, 1]
        if metric == "brier":
            s = float(np.mean((p - y) ** 2))
        else:
            p = np.clip(p, 1e-12, 1 - 1e-12)
            s = float(
                -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            )
        metric_map[node_name] = s
        gate_nodes.add(node_name)

    return metric_map, gate_nodes


def build_gate_accuracy_map(
    results_v4_run: Dict[str, Any],
    y_sub_test: np.ndarray,
    mode: str = "soft",
) -> Tuple[Dict[str, float], set]:
    """Compute per-gate accuracy for colouring the dendrogram."""
    y_true = np.asarray(y_sub_test, dtype=str)
    excluded = set(
        results_v4_run.get("bundle", {}).get("excluded_leaves", [])
    )
    bundle = dict(results_v4_run["bundle"])
    gate_outputs = results_v4_run[mode]["gate_outputs"]
    bundle["_gate_outputs_ref"] = gate_outputs

    accuracy_map: Dict[str, float] = {}
    gate_nodes: set = set()
    for gate_key, out in gate_outputs.items():
        node_name = _clean_node_name(gate_key)
        keep, y = _compute_gate_targets(bundle, gate_key, y_true, excluded)
        if len(keep) == 0:
            continue
        probs = np.asarray(out["probs"], float)[keep]
        y_pred = np.argmax(probs, axis=1)
        acc = float(np.mean(y_pred == y))
        accuracy_map[node_name] = acc
        gate_nodes.add(node_name)

    return accuracy_map, gate_nodes


def build_gate_balanced_accuracy_map(
    results_v4_run: Dict[str, Any],
    y_sub_test: np.ndarray,
    mode: str = "soft",
) -> Tuple[Dict[str, float], set]:
    """Compute per-gate balanced accuracy for colouring the dendrogram."""
    y_true = np.asarray(y_sub_test, dtype=str)
    excluded = set(
        results_v4_run.get("bundle", {}).get("excluded_leaves", [])
    )
    bundle = dict(results_v4_run["bundle"])
    gate_outputs = results_v4_run[mode]["gate_outputs"]
    bundle["_gate_outputs_ref"] = gate_outputs

    bal_map: Dict[str, float] = {}
    gate_nodes: set = set()
    for gate_key, out in gate_outputs.items():
        node_name = _clean_node_name(gate_key)
        keep, y = _compute_gate_targets(bundle, gate_key, y_true, excluded)
        if len(keep) == 0:
            continue
        probs = np.asarray(out["probs"], float)[keep]
        y_pred = np.argmax(probs, axis=1)
        recalls = []
        for cls in (0, 1):
            mask = y == cls
            if mask.sum() > 0:
                recalls.append(float(np.mean(y_pred[mask] == cls)))
        bal_acc = float(np.mean(recalls)) if recalls else 0.0
        bal_map[node_name] = bal_acc
        gate_nodes.add(node_name)

    return bal_map, gate_nodes


def build_gate_roc_auc_map(
    results_v4_run: Dict[str, Any],
    y_sub_test: np.ndarray,
    mode: str = "soft",
) -> Tuple[Dict[str, float], set]:
    """Compute per-gate ROC-AUC for colouring the dendrogram."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_sub_test, dtype=str)
    excluded = set(
        results_v4_run.get("bundle", {}).get("excluded_leaves", [])
    )
    bundle = dict(results_v4_run["bundle"])
    gate_outputs = results_v4_run[mode]["gate_outputs"]
    bundle["_gate_outputs_ref"] = gate_outputs

    auc_map: Dict[str, float] = {}
    gate_nodes: set = set()
    for gate_key, out in gate_outputs.items():
        node_name = _clean_node_name(gate_key)
        keep, y = _compute_gate_targets(bundle, gate_key, y_true, excluded)
        if len(keep) == 0:
            continue
        if len(np.unique(y)) < 2:
            continue
        p = np.asarray(out["probs"], float)[keep, 1]
        auc_map[node_name] = float(roc_auc_score(y, p))
        gate_nodes.add(node_name)

    return auc_map, gate_nodes


def build_gate_cell_count_map(
    results_v4_run: Dict[str, Any],
    y_sub_test: np.ndarray,
    mode: str = "soft",
) -> Tuple[Dict[str, float], set]:
    """Compute per-gate cell count (log10) for colouring the dendrogram."""
    y_true = np.asarray(y_sub_test, dtype=str)
    excluded = set(
        results_v4_run.get("bundle", {}).get("excluded_leaves", [])
    )
    bundle = dict(results_v4_run["bundle"])
    gate_outputs = results_v4_run[mode]["gate_outputs"]
    bundle["_gate_outputs_ref"] = gate_outputs

    count_map: Dict[str, float] = {}
    gate_nodes: set = set()
    for gate_key, out in gate_outputs.items():
        node_name = _clean_node_name(gate_key)
        keep, y = _compute_gate_targets(bundle, gate_key, y_true, excluded)
        n = int(np.sum(keep))
        if n == 0:
            continue
        count_map[node_name] = float(np.log10(max(n, 1)))
        gate_nodes.add(node_name)

    return count_map, gate_nodes


def build_adj(
    tree: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
    """Build an adjacency dict from a hierarchy tree."""
    paths: Dict[str, List[str]] = {}

    def walk(n: Any, p: List[str] | None = None) -> None:
        if p is None:
            p = []
        if isinstance(n, dict):
            for k, v in n.items():
                walk(v, p + [k])
        elif isinstance(n, list):
            for leaf in n:
                paths[leaf] = p + [leaf]

    walk(tree)

    adj: Dict[str, List[str]] = {}
    nodes_set: set = set()
    leaves = [k for k in paths if k in CLUSTER_META]

    for l in leaves:
        p = paths[l]
        deduped_path = [p[0]]
        for x in p[1:]:
            if x != deduped_path[-1]:
                deduped_path.append(x)
        for i in range(len(deduped_path) - 1):
            u, v = deduped_path[i], deduped_path[i + 1]
            nodes_set.add(u)
            nodes_set.add(v)
            adj.setdefault(u, []).append(v)

    for u in adj:
        adj[u] = sorted(list(set(adj[u])))

    children = {v for u in adj for v in adj[u]}
    roots = list(nodes_set - children)
    if not roots and nodes_set:
        roots = [list(nodes_set)[0]]
    return adj, roots, list(nodes_set), leaves


def _x_for_depth(depth: int, max_d: int, xcfg: Dict[str, Any]) -> float:
    if xcfg["use_uniform_spacing"]:
        return depth * xcfg["uniform_dx"] + xcfg["uniform_offset"]
    mp = xcfg["manual_positions"]
    if depth in mp:
        return float(mp[depth])
    max_defined = max(mp.keys())
    return float(mp[max_defined]) + (depth - max_defined) * xcfg["uniform_dx"]


def _apply_y_scale(
    y_map: Dict[str, float],
    fam_bounds: Dict[str, Tuple[float, float]],
    ycfg: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    s = float(ycfg.get("y_scale", 1.0))
    if abs(s - 1.0) < 1e-9:
        return y_map, fam_bounds
    ys = list(y_map.values())
    y_mid = 0.5 * (min(ys) + max(ys))
    y_map2 = {k: (y_mid + (v - y_mid) * s) for k, v in y_map.items()}
    fam_bounds2 = {
        fam: (y_mid + (y0 - y_mid) * s, y_mid + (y1 - y_mid) * s)
        for fam, (y0, y1) in fam_bounds.items()
    }
    return y_map2, fam_bounds2


def compute_layout(
    adj: Dict[str, List[str]],
    roots: List[str],
    nodes: List[str],
    leaves: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]], float]:
    """Compute x/y positions for all nodes."""
    cfg = config or default_dendrogram_config()
    fam_order = ["Calb1", "Sox6", "Gad2", "Lef1"]
    fam_groups: Dict[str, List[str]] = {f: [] for f in fam_order}
    for l in leaves:
        hit = False
        for f in fam_order:
            if f in l or (f == "Lef1" and "Lef1" in l):
                fam_groups[f].append(l)
                hit = True
                break
        if not hit:
            fam_groups["Gad2"].append(l)

    dfs_order: List[str] = []

    def dfs(u: str, visited: set) -> None:
        if u in visited:
            return
        visited.add(u)
        if u in leaves:
            dfs_order.append(u)
        else:
            for v in adj.get(u, []):
                dfs(v, visited)

    visited_dfs: set = set()
    for r in roots:
        dfs(r, visited_dfs)

    y_map: Dict[str, float] = {}
    curr_y = 1.0
    min_d = cfg["spacing"]["min_leaf_dist"]
    fam_gap = cfg["spacing"]["fam_gap"]
    for fam in fam_order:
        if fam == "Sox6":
            curr_y += 0.05
        fs_leaves = [l for l in dfs_order if l in fam_groups[fam]]
        if not fs_leaves:
            continue
        for l in fs_leaves:
            y_map[l] = curr_y
            curr_y -= min_d
        curr_y -= fam_gap

    stack = list(roots)
    visited: set = set()
    post_order: List[str] = []
    while stack:
        u = stack[-1]
        if u not in visited:
            visited.add(u)
            for v in reversed(adj.get(u, [])):
                if v not in visited:
                    stack.append(v)
        else:
            post_order.append(stack.pop())

    for u in post_order:
        if u not in y_map:
            kids = [v for v in adj.get(u, []) if v in y_map]
            y_map[u] = (
                sum(y_map[v] for v in kids) / len(kids) if kids else 0.5
            )

    depths: Dict[str, int] = {r: 0 for r in roots}
    q = deque(roots)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)
    max_d_val = max(depths.values()) if depths else 1

    xcfg = cfg["x_layout"]
    x_map: Dict[str, float] = {}
    leaves_set = set(leaves)
    for n in nodes:
        d = depths.get(n, 0)
        if n in roots:
            x_map[n] = 0.0
        elif n in leaves_set:
            x_map[n] = np.nan
        else:
            x = _x_for_depth(int(d), int(max_d_val), xcfg)
            x = min(x, float(xcfg["max_internal_x"]))
            x_map[n] = x

    internal_xs = [
        x
        for x in x_map.values()
        if isinstance(x, (int, float)) and np.isfinite(x)
    ]
    leaf_x = (max(internal_xs) if internal_xs else 0.0) + float(
        xcfg["internal_leaf_gap"]
    )

    for n in nodes:
        if n in leaves_set:
            x_map[n] = leaf_x
    for r in roots:
        x_map[r] = 0.0

    fam_bounds: Dict[str, Tuple[float, float]] = {}
    band_pad = min_d * 0.8
    spacing_cfg = cfg.get("spacing", {})
    band_pad_by_family = spacing_cfg.get("fam_band_pad_by_family", {})
    band_pad_top_by_family = spacing_cfg.get("fam_band_pad_top_by_family", {})
    band_pad_bottom_by_family = spacing_cfg.get("fam_band_pad_bottom_by_family", {})
    for fam in fam_order:
        fs_leaves = fam_groups[fam]
        ys = [y_map[l] for l in fs_leaves if l in y_map]
        if ys:
            default_extra_pad = float(band_pad_by_family.get(fam, 0.0))
            top_extra_pad = float(
                band_pad_top_by_family.get(fam, default_extra_pad)
            )
            bottom_extra_pad = float(
                band_pad_bottom_by_family.get(fam, default_extra_pad)
            )
            y_min_b = min(ys) - band_pad - bottom_extra_pad
            y_max_b = max(ys) + band_pad + top_extra_pad
            fam_bounds[fam] = (y_min_b, y_max_b)

    y_map, fam_bounds = _apply_y_scale(y_map, fam_bounds, cfg["y_layout"])
    return x_map, y_map, fam_bounds, leaf_x


def _make_subtree_counts(
    adj: Dict[str, List[str]],
    roots: List[str],
    nodes: List[str],
    leaves_set: set,
    leaf_counts: Dict[str, float],
) -> Dict[str, float]:
    """Roll leaf counts up to internal nodes (subtree totals)."""
    subtree = {n: 0.0 for n in nodes}
    for l, c in leaf_counts.items():
        if l in subtree:
            subtree[l] = float(c)

    stack = list(roots)
    visited: set = set()
    post: List[str] = []
    while stack:
        u = stack[-1]
        if u not in visited:
            visited.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    stack.append(v)
        else:
            post.append(stack.pop())

    for u in post:
        if u not in leaves_set:
            subtree[u] = sum(subtree.get(v, 0.0) for v in adj.get(u, []))
    return subtree


def _count_to_lw(
    cnt: float,
    max_cnt: float,
    sizes_cfg: Dict[str, Any],
    ew_cfg: Dict[str, Any],
) -> float:
    lw_min = float(sizes_cfg.get("edge_lw_min", sizes_cfg.get("edge_lw", 2.0)))
    lw_max = float(sizes_cfg.get("edge_lw_max", sizes_cfg.get("edge_lw", 2.0)))
    gamma = float(sizes_cfg.get("edge_lw_gamma", 0.5))
    scale = str(ew_cfg.get("scale", "log")).lower()

    if max_cnt <= 0:
        t = 0.0
    elif scale == "linear":
        t = cnt / max_cnt
    else:
        t = np.log1p(cnt) / np.log1p(max_cnt)

    t = float(np.clip(t, 0.0, 1.0)) ** gamma
    return lw_min + (lw_max - lw_min) * t


def plot_hmoe_final(
    hierarchy_tree: Dict[str, Any],
    results_run: Dict[str, Any],
    *,
    y_sub_test: np.ndarray,
    filename: str = "HMoE_final_polished_v18.png",
    mode: str = "soft",
    metric: str = "brier",
    dpi: int = 140,
    human_pred_leaf: Optional[Sequence[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, FancyBboxPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.collections import PatchCollection, LineCollection
    from matplotlib.lines import Line2D

    def _bezier(p1, p2):
        xm = (p1[0] + p2[0]) / 2
        verts = [p1, (xm, p1[1]), (xm, p2[1]), p2]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        return MplPath(verts, codes)

    def _smooth_deep_curve(p1, p2, bottom_y):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        c1 = (x1 + dx * 0.1, bottom_y)
        c2 = (x2 - dx * 0.5, y2)
        verts = [p1, c1, c2, p2]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        return MplPath(verts, codes)

    def _offset_path(path, dy):
        verts = path.vertices.copy()
        verts[:, 1] += dy
        return MplPath(verts, path.codes)

    def parse_label(n):
        n_lower = n.lower()
        if "root" in n_lower or "node" in n_lower or "rest_of_dendrogram" in n_lower:
            return None
        if "Lef1" in n and "Lef1" not in n.split(":")[0] and _clean_node_name(n) == n:
            return "Lef1"
        return _clean_node_name(n)

    def get_terminal_label(v):
        if "Lef1" in v:
            return "Lef1+"
        meta = CLUSTER_META.get(v)
        if meta and "genes" in meta and meta["genes"] != "-":
            last_gene = meta["genes"].split(",")[-1].strip()
            return f"{last_gene}+"
        parts = v.split(":")
        if len(parts) > 1:
            return f"{parts[-1].strip()}+"
        return f"{v.strip()}+"

    cfg = config if config is not None else default_dendrogram_config()

    adj, roots, nodes, leaves = build_adj(hierarchy_tree)
    x_map, y_map, fam_bounds, leaf_x = compute_layout(
        adj, roots, nodes, leaves, config=cfg
    )

    leaves_set = set(leaves)
    excluded = set(results_run.get("bundle", {}).get("excluded_leaves", []))

    _higher_is_better_metrics = {"accuracy", "balanced_accuracy", "roc_auc", "cell_count"}

    if metric == "accuracy":
        metric_map, gate_nodes = build_gate_accuracy_map(
            results_run, y_sub_test, mode
        )
    elif metric == "balanced_accuracy":
        metric_map, gate_nodes = build_gate_balanced_accuracy_map(
            results_run, y_sub_test, mode
        )
    elif metric == "roc_auc":
        metric_map, gate_nodes = build_gate_roc_auc_map(
            results_run, y_sub_test, mode
        )
    elif metric == "cell_count":
        metric_map, gate_nodes = build_gate_cell_count_map(
            results_run, y_sub_test, mode
        )
    else:
        metric_map, gate_nodes = build_gate_metric_map(
            results_run, y_sub_test, mode, metric
        )

    if metric == "cell_count":
        vals = np.array(list(metric_map.values()), dtype=float)
        if len(vals) > 0:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = (vmax - vmin) * 0.05
            vmin -= pad
            vmax += pad
        else:
            vmin, vmax = 2.0, 5.0
        cmap_name = cfg.get("accuracy_colormap", {}).get("cmap", "YlOrRd")
        cmap = plt.get_cmap(cmap_name)
    elif metric in _higher_is_better_metrics:
        if "accuracy_colormap" in config:
            vmin = config["accuracy_colormap"].get("vmin", 0.50)
            vmax = config["accuracy_colormap"].get("vmax", 1.0)
            cmap_name = config["accuracy_colormap"].get("cmap", "RdYlGn")
        else:
            vals = np.array(list(metric_map.values()), dtype=float)
            if len(vals) > 0:
                data_min, data_max = float(np.nanmin(vals)), float(np.nanmax(vals))
                range_pad = (data_max - data_min) * 0.02
                vmin = max(0.0, data_min - range_pad)
                vmax = min(1.0, data_max + range_pad)
            else:
                vmin, vmax = 0.50, 1.0
            cmap_name = "RdYlGn"
        cmap = plt.get_cmap(cmap_name)
    else:
        vals = np.array(list(metric_map.values()), dtype=float)
        vmin, vmax = (
            (float(np.nanmin(vals)), float(np.nanmax(vals))) if len(vals) > 0 else (0, 1)
        )
        cmap = plt.get_cmap("RdYlGn_r")

    ew = cfg.get("edge_weighting", {})
    weigh_by = str(ew.get("weigh_by", "none")).lower()
    weigh_method = str(ew.get("weigh_method", "cumulative")).lower()
    both_offset_dy = float(ew.get("both_offset_dy", 0.010))
    debug_print = bool(ew.get("debug_print", True))
    debug_topk = int(ew.get("debug_topk", 5))

    ms = ew.get("mouse_style", {})
    hs = ew.get("human_style", {})
    default_alpha = float(ew.get("alpha", 0.80))

    if weigh_by in ("mouse", "both"):
        mouse_leaf_counts = Counter(
            _normalize_leaf_label(x)
            for x in np.asarray(y_sub_test, dtype=str)
        )
    else:
        mouse_leaf_counts = Counter()

    if weigh_by in ("human", "both"):
        if human_pred_leaf is None:
            raise ValueError(
                "Provide `human_pred_leaf=...` to weight edges by human."
            )
        human_leaf_counts = Counter(
            _normalize_leaf_label(x)
            for x in np.asarray(human_pred_leaf, dtype=str)
        )
    else:
        human_leaf_counts = Counter()

    for ex in excluded:
        mouse_leaf_counts.pop(ex, None)
        human_leaf_counts.pop(ex, None)

    if debug_print:
        print(
            "[edge_weighting] weigh_by:",
            weigh_by,
            "weigh_method:",
            weigh_method,
            "scale:",
            ew.get("scale"),
        )
        print(
            "[edge_weighting] mouse total/unique:",
            sum(mouse_leaf_counts.values()),
            len(mouse_leaf_counts),
        )
        print(
            "[edge_weighting] human total/unique:",
            sum(human_leaf_counts.values()),
            len(human_leaf_counts),
        )

    if weigh_method == "cumulative":
        mouse_subtree = (
            _make_subtree_counts(
                adj, roots, nodes, leaves_set, mouse_leaf_counts
            )
            if weigh_by in ("mouse", "both")
            else None
        )
        human_subtree = (
            _make_subtree_counts(
                adj, roots, nodes, leaves_set, human_leaf_counts
            )
            if weigh_by in ("human", "both")
            else None
        )
        mouse_max = max(mouse_subtree.values()) if mouse_subtree else 0.0
        human_max = max(human_subtree.values()) if human_subtree else 0.0
    elif weigh_method == "terminal":
        mouse_subtree = None
        human_subtree = None
        mouse_max = (
            max(mouse_leaf_counts.values()) if mouse_leaf_counts else 0.0
        )
        human_max = (
            max(human_leaf_counts.values()) if human_leaf_counts else 0.0
        )
    else:
        raise ValueError(
            f"Unknown weigh_method={weigh_method!r}. Use 'cumulative' or 'terminal'."
        )

    leaf_cache: Dict[str, str] = {}

    def first_leaf(n):
        if n in leaf_cache:
            return leaf_cache[n]
        cur = n
        while cur not in leaves_set and cur in adj and adj[cur]:
            cur = adj[cur][0]
        leaf_cache[n] = cur
        return cur

    node_col = {
        n: get_fam_color(n if n in leaves_set else first_leaf(n))["base"]
        for n in nodes
    }

    label_cfg = cfg.get("label_layout", {})

    split_label_collision_radius = float(
        label_cfg.get("split_label_collision_radius", 0.06)
    )
    split_label_v_sep_step = float(
        label_cfg.get("split_label_v_sep_step", 0.015)
    )
    split_label_max_tries = int(
        label_cfg.get("split_label_max_tries", 5)
    )

    table_columns = copy.deepcopy(
        cfg.get("table", {}).get("columns")
        or [
            {"key": "short", "header": "Subtype", "width": 0.22, "font_key": "row_main"},
            {"key": "id", "header": "Cluster #", "width": 0.12, "font_key": "row_main"},
            {"key": "genes", "header": "Signature", "width": 0.0, "font_key": "row_meta"},
        ]
    )
    table_xs: List[float] = []
    x_cursor = leaf_x + cfg["spacing"]["table_start"]
    for col in table_columns:
        table_xs.append(x_cursor)
        x_cursor += float(col.get("width", 0.0))
    table_end_x = x_cursor + cfg["spacing"]["table_end_pad"]

    zone_right = max(
        table_end_x + float(cfg["spacing"].get("zone_x_end_pad", 0.0)),
        float(cfg["spacing"].get("zone_x_end_min", table_end_x)),
    )

    fig, ax = plt.subplots(figsize=cfg["fig"]["figsize"], dpi=dpi)
    axes_right = cfg.get("positions", {}).get("axes_right")
    if axes_right is not None:
        fig.subplots_adjust(right=float(axes_right))

    ycfg = cfg["y_layout"]
    min_y = min(y_map.values()) if y_map else 0
    max_y = max(y_map.values()) if y_map else 1
    ax.set_ylim(
        min_y - float(ycfg["ylim_pad_bot"]),
        max_y + float(ycfg["ylim_pad_top"]),
    )
    ax.set_xlim(float(ycfg.get("xlim_left", -0.1)), zone_right)
    ax.axis("off")

    for fam, (y0, y1) in fam_bounds.items():
        c = FAMILY_PALETTE[fam]
        z_start = float(
            cfg["spacing"].get("zone_x_start_by_family", {}).get(
                fam, cfg["spacing"]["zone_x_start"]
            )
        )
        title_anchor = cfg["spacing"].get("family_title_anchor_by_family", {}).get(
            fam, {}
        )
        title_x = float(title_anchor.get("x", z_start + 0.02))
        title_y = float(title_anchor.get("y", y1 - 0.025))
        title_ha = str(title_anchor.get("ha", "left"))
        title_va = str(title_anchor.get("va", "top"))
        ax.add_patch(
            FancyBboxPatch(
                (z_start, y0),
                zone_right - z_start,
                y1 - y0 + 0.02,
                boxstyle="round,pad=0.01",
                fc=c["light"],
                ec="none",
                alpha=0.18,
                zorder=0,
            )
        )
        ax.text(
            title_x,
            title_y,
            f"{fam} Family",
            color=c["base"],
            fontweight="bold",
            fontsize=cfg["fonts"]["fam_label"],
            ha=title_ha,
            va=title_va,
            zorder=50,
        )

    patches, colors_e, lws = [], [], []
    patches_m, colors_m, lws_m = [], [], []
    patches_h, colors_h, lws_h = [], [], []
    labels_list: List[tuple] = []
    placed_internal_labels: List[Tuple[float, float]] = []

    def _edge_count_for(which, child_v, rep_leaf):
        if which == "mouse":
            if weigh_method == "cumulative":
                return float(mouse_subtree.get(child_v, 0.0)) if mouse_subtree is not None else 0.0
            return float(mouse_leaf_counts.get(rep_leaf, 0.0))
        if which == "human":
            if weigh_method == "cumulative":
                return float(human_subtree.get(child_v, 0.0)) if human_subtree is not None else 0.0
            return float(human_leaf_counts.get(rep_leaf, 0.0))
        return 0.0

    def _edge_lw_for(which, cnt):
        if which == "mouse":
            return _count_to_lw(cnt, mouse_max, cfg["sizes"], ew)
        if which == "human":
            return _count_to_lw(cnt, human_max, cfg["sizes"], ew)
        return float(cfg["sizes"]["edge_lw"])

    for u, kids in adj.items():
        for v in kids:
            rep = first_leaf(v)
            is_lef1_branch = "Lef1" in str(rep) and v in leaves_set
            is_terminal = v in leaves_set

            if is_lef1_branch:
                base_path = _smooth_deep_curve(
                    (x_map[u], y_map[u]),
                    (x_map[v], y_map[v]),
                    min_y - 0.08,
                )
            else:
                base_path = _bezier(
                    (x_map[u], y_map[u]), (x_map[v], y_map[v])
                )

            edge_color = node_col.get(v, cfg["colors"]["edge_default"])

            if weigh_by == "none":
                patches.append(PathPatch(base_path, fc="none"))
                colors_e.append(edge_color)
                lws.append(float(cfg["sizes"]["edge_lw"]))

            elif weigh_by in ("mouse", "human"):
                which = weigh_by
                cnt = _edge_count_for(which, v, rep)
                lw = _edge_lw_for(which, cnt)
                patches.append(PathPatch(base_path, fc="none"))
                style_key = f"{which}_style"
                override_color = ew.get(style_key, {}).get("color")
                patches_color = (
                    edge_color if override_color is None else override_color
                )
                colors_e.append(patches_color)
                lws.append(lw)

            elif weigh_by == "both":
                path_mouse = _offset_path(base_path, +both_offset_dy)
                path_human = _offset_path(base_path, -both_offset_dy)

                cnt_m = _edge_count_for("mouse", v, rep)
                cnt_h = _edge_count_for("human", v, rep)
                lw_m = _edge_lw_for("mouse", cnt_m)
                lw_h = _edge_lw_for("human", cnt_h)

                mouse_color = (
                    edge_color if ms.get("color") is None else ms["color"]
                )
                human_color = (
                    edge_color if hs.get("color") is None else hs["color"]
                )

                patches_m.append(PathPatch(path_mouse, fc="none"))
                colors_m.append(mouse_color)
                lws_m.append(lw_m)

                patches_h.append(PathPatch(path_human, fc="none"))
                colors_h.append(human_color)
                lws_h.append(lw_h)

            lbl_txt = None
            if is_terminal:
                lbl_txt = get_terminal_label(v)
                terminal_anchor_x = label_cfg.get(
                    "terminal_label_anchor_x", x_map[v] - 0.02
                )
                lx, ly = float(terminal_anchor_x), y_map[v]
                ha, va = "right", "center"
            else:
                lbl_txt = parse_label(v)
                if lbl_txt and v in adj and adj[v]:
                    for child in adj[v]:
                        if (
                            child in leaves_set
                            and get_terminal_label(child) == lbl_txt
                        ):
                            lbl_txt = None
                            break

                if lbl_txt:
                    lx = (x_map[u] + x_map[v]) / 2
                    ly = (y_map[u] + y_map[v]) / 2
                    ha = "center"
                    dy = y_map[v] - y_map[u]
                    nudge = float(
                        label_cfg.get("split_label_line_nudge", 0.028)
                    )
                    if dy > 0.01:
                        ly += nudge
                        va = "bottom"
                    elif dy < -0.01:
                        ly -= nudge
                        va = "top"
                    else:
                        ly += nudge
                        va = "bottom"

                    overlap = True
                    tries = 0
                    while overlap and tries < split_label_max_tries:
                        overlap = False
                        for plx, ply in placed_internal_labels:
                            if (
                                np.sqrt((lx - plx) ** 2 + (ly - ply) ** 2)
                                < split_label_collision_radius
                            ):
                                overlap = True
                                ly += (
                                    split_label_v_sep_step
                                    if ly > ply
                                    else -split_label_v_sep_step
                                )
                                tries += 1
                                break
                        if not overlap:
                            placed_internal_labels.append((lx, ly))

            if lbl_txt:
                if not is_terminal and "Lef1" in lbl_txt:
                    lx = x_map[u] + 0.6 * (x_map[v] - x_map[u])
                    ly = min_y - 0.045
                    ha, va = "center", "center"
                if is_terminal:
                    lx = lx + float(label_cfg.get("terminal_x_pad", 0.016))
                split_label_offsets = label_cfg.get("split_label_offsets", {})
                terminal_label_offsets = label_cfg.get(
                    "terminal_label_offsets", {}
                )
                offset_map = (
                    terminal_label_offsets if is_terminal else split_label_offsets
                )
                if lbl_txt in offset_map:
                    ndg = offset_map[lbl_txt]
                    lx += ndg.get("dx", 0.0)
                    ly += ndg.get("dy", 0.0)
                if is_terminal:
                    bbox_fill = get_fam_color(v)["light"]
                else:
                    semantic_fill = get_fam_color(lbl_txt)["light"]
                    if semantic_fill != FAMILY_PALETTE["Other"]["light"]:
                        bbox_fill = semantic_fill
                    else:
                        bbox_fill = get_fam_color(first_leaf(v))["light"]
                bbox_pad = cfg.get("bbox", {}).get(
                    "terminal_pad" if is_terminal else "split_pad",
                    cfg.get("bbox", {}).get("split_pad", 0.4),
                )
                bbox_style = dict(
                    fc=bbox_fill,
                    ec="none",
                    lw=cfg.get("bbox", {}).get("split_lw", 0.5),
                    alpha=0.96,
                    pad=bbox_pad,
                )
                labels_list.append(
                    (lx, ly, lbl_txt, edge_color, ha, va, bbox_style)
                )

    if weigh_by == "both":
        ax.add_collection(
            PatchCollection(
                patches_h,
                fc="none",
                ec=colors_h,
                lw=lws_h,
                alpha=float(hs.get("alpha", default_alpha)),
                zorder=int(hs.get("zorder", 2)),
            )
        )
        ax.add_collection(
            PatchCollection(
                patches_m,
                fc="none",
                ec=colors_m,
                lw=lws_m,
                alpha=float(ms.get("alpha", default_alpha)),
                zorder=int(ms.get("zorder", 3)),
            )
        )
        if bool(ew.get("show_both_legend", True)):
            handles = [
                Line2D(
                    [0],
                    [0],
                    color="#000000",
                    lw=4,
                    linestyle="solid",
                    label="Mouse (true counts)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=str(hs.get("color", "#222222")),
                    lw=4,
                    linestyle=hs.get("linestyle", (0, (7, 4))),
                    label="Human (pred counts)",
                ),
            ]
            ax.legend(
                handles=handles,
                loc=str(ew.get("legend_loc", "lower left")),
                frameon=True,
                framealpha=0.9,
            )
    else:
        ax.add_collection(
            PatchCollection(
                patches,
                fc="none",
                ec=colors_e,
                lw=lws,
                alpha=float(ew.get("alpha", 0.65)),
                zorder=2,
            )
        )

    for lx, ly, txt, c, ha, va, bb in labels_list:
        ax.text(
            lx,
            ly,
            txt,
            fontsize=cfg["fonts"]["split"],
            fontweight="bold",
            color=c,
            ha=ha,
            va=va,
            bbox=bb,
            zorder=20,
        )

    for r in roots:
        ax.scatter(
            [x_map[r]],
            [y_map[r]],
            c=cfg["colors"]["root_marker"],
            s=cfg["sizes"]["root_s"],
            zorder=10,
        )
        ax.text(
            x_map[r] + float(cfg.get("positions", {}).get("root_label_dx", -0.09)),
            y_map[r],
            "Root",
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=cfg["fonts"]["root"],
            zorder=11,
            bbox=dict(
                boxstyle=f"round,pad={cfg.get('bbox', {}).get('root_pad', 0.15)}",
                fc=cfg["colors"]["root_bg"],
                ec="none",
                alpha=0.9,
            ),
        )

    ln = [n for n in nodes if n in leaves_set]
    lc = [
        (
            cfg["colors"]["excluded_marker"]
            if n in excluded
            else node_col[n]
        )
        for n in ln
    ]
    ax.scatter(
        [x_map[n] for n in ln],
        [y_map[n] for n in ln],
        c=lc,
        s=cfg["sizes"]["leaf_s"],
        zorder=12,
        ec="white",
        lw=0.6,
    )

    true_gates = [
        n
        for n in nodes
        if n not in leaves_set
        and n not in roots
        and n in adj
        and len(adj[n]) > 1
    ]
    fc_list, ec_list = [], []
    for n in true_gates:
        ec_list.append(node_col.get(n, "#999"))
        clean_n = _clean_node_name(n)
        if clean_n in gate_nodes:
            t = (
                np.clip(
                    (metric_map[clean_n] - vmin) / (vmax - vmin), 0, 1
                )
                if vmax != vmin
                else 0.5
            )
            fc_list.append(cmap(t))
        else:
            fc_list.append(cfg["colors"]["no_score_fill"])

    ax.scatter(
        [x_map[n] for n in true_gates],
        [y_map[n] for n in true_gates],
        c=fc_list,
        edgecolors=ec_list,
        marker="D",
        s=cfg["sizes"]["internal_s"],
        lw=cfg["sizes"]["marker_lw"],
        zorder=13,
    )

    header_offset = float(cfg["spacing"].get("header_y_offset", 0.11))
    for x_col, col in zip(table_xs, table_columns):
        header_x = x_col
        if str(col.get("key", "")) == "short":
            subtype_header_x = cfg.get("positions", {}).get("subtype_header_x")
            if subtype_header_x is not None:
                header_x = float(subtype_header_x)
        ax.text(
            header_x,
            max_y + header_offset,
            str(col.get("header", "")),
            fontweight="bold",
            fontsize=cfg["fonts"]["header"],
            va="bottom",
            ha="left",
        )

    lines = []
    for l in leaves:
        y = y_map[l]
        meta = CLUSTER_META.get(l, {"id": "?", "genes": "-", "short": l})
        is_ex = l in excluded
        tc = cfg["colors"]["excluded_text"] if is_ex else node_col[l]
        alp = 0.7 if is_ex else 1.0
        if table_xs:
            lines.append([(leaf_x + 0.01, y), (table_xs[0] - 0.01, y)])
        for x_col, col in zip(table_xs, table_columns):
            key = str(col.get("key", ""))
            value = meta.get(key, l if key == "short" else "-")
            text_kwargs = {
                "x": x_col,
                "y": y,
                "s": value,
                "va": "center",
                "fontsize": cfg["fonts"][str(col.get("font_key", "row_main"))],
                "alpha": alp,
            }
            if key == "short":
                text_kwargs["fontweight"] = "bold"
                text_kwargs["color"] = tc
            elif key == "genes":
                text_kwargs["style"] = "italic"
                text_kwargs["color"] = "#444"
            ax.text(**text_kwargs)

    ax.add_collection(
        LineCollection(
            lines, colors="#EEE", lw=cfg["sizes"]["table_lw"], zorder=1
        )
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    if bool(cfg.get("positions", {}).get("cbar_use_figure_coords", False)):
        cbar_ax = fig.add_axes(cfg["positions"]["cbar"])
    else:
        cbar_ax = ax.inset_axes(cfg["positions"]["cbar"])
    cbar_orientation = str(
        cfg.get("positions", {}).get("cbar_orientation", "horizontal")
    )
    _metric_labels = {
        "accuracy": f"Gate Accuracy ({mode})",
        "balanced_accuracy": f"Gate Balanced Accuracy ({mode})",
        "roc_auc": f"Gate ROC-AUC ({mode})",
        "cell_count": "Cells Routed (log\u2081\u2080 scale)",
    }
    cbar_label = _metric_labels.get(
        metric, f"Gating {metric} ({mode}) lower=better"
    )
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation)
    cbar.set_label(
        cbar_label,
        fontsize=cfg["fonts"]["cbar"],
        labelpad=cfg.get("bbox", {}).get("cbar_labelpad", 10),
    )
    cbar.ax.tick_params(labelsize=max(cfg["fonts"]["cbar"] - 1, 6))
    if cbar_orientation == "vertical":
        cbar.ax.yaxis.set_label_position("right")

    save_cfg = cfg.get("save", {})
    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches=save_cfg.get("bbox_inches", "tight"),
        pad_inches=save_cfg.get("pad_inches", 0.02),
        facecolor=save_cfg.get("facecolor", "white"),
    )
    if str(plt.get_backend()).lower() != "agg":
        plt.show()
    plt.close(fig)
