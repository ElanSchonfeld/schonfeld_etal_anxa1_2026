"""Hierarchy utilities for MoE v4."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def walk_paths(tree: Dict[str, Any], path_prefix: List[str] | None = None) -> Dict[str, List[str]]:
    """Return a map of leaf subtype -> path list."""

    if path_prefix is None:
        path_prefix = []
    subtype_to_path: Dict[str, List[str]] = {}
    if isinstance(tree, list):
        for st in tree:
            subtype_to_path[st] = list(path_prefix)
        return subtype_to_path
    if isinstance(tree, dict):
        for node_label, child in tree.items():
            subtype_to_path.update(walk_paths(child, path_prefix + [node_label]))
        return subtype_to_path
    raise TypeError(f"Unexpected node type: {type(tree)} at path={path_prefix}")


def walk_terminal_paths(
    tree: Dict[str, Any], path: List[str] | None = None
) -> Tuple[Dict[str, List[str]], List[Tuple[List[str], List[str]]]]:
    """Return (subtype_to_path, terminal_buckets)."""

    if path is None:
        path = []

    subtype_to_path: Dict[str, List[str]] = {}
    terminal_buckets: List[Tuple[List[str], List[str]]] = []

    if isinstance(tree, list):
        terminal_buckets.append((path, tree))
        for st in tree:
            subtype_to_path[st] = list(path)
        return subtype_to_path, terminal_buckets

    if isinstance(tree, dict):
        for key, child in tree.items():
            child_sub2path, child_buckets = walk_terminal_paths(child, path + [key])
            subtype_to_path.update(child_sub2path)
            terminal_buckets.extend(child_buckets)
        return subtype_to_path, terminal_buckets

    raise TypeError(f"Unexpected node type: {type(tree)} at path={path}")


def get_node(tree: Dict[str, Any], path_list: List[str]) -> Any:
    """Return the subtree found at the given path list."""

    cur: Any = tree
    for key in path_list:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Path not found: {path_list}")
        cur = cur[key]
    return cur


def height_partitions(
    tree: Dict[str, Any], root_key: str = "node", max_heights: int = 50
) -> Iterable[List[str]]:
    """Compute dendrogram-cut partitions by iterative expansion."""

    root_path = [root_key]
    current = [root_path]
    partitions: List[List[str]] = []

    for _h in range(max_heights):
        partitions.append(["|".join(p) for p in current])

        next_nodes: List[List[str]] = []
        any_expanded = False
        for p in current:
            sub = get_node(tree, p)
            if isinstance(sub, dict):
                any_expanded = True
                for child_key in sub.keys():
                    next_nodes.append(p + [child_key])
            elif isinstance(sub, list):
                next_nodes.append(p)
            else:
                raise TypeError(f"Unexpected node type at {p}: {type(sub)}")

        current = next_nodes
        if not any_expanded:
            break

    partitions = [list(dict.fromkeys(part)) for part in partitions]
    return partitions


def build_node_to_leaves_from_subtype_paths(
    subtype_to_path: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Build node_path -> list of leaf subtypes for all prefix nodes."""

    node_to_leaves: Dict[str, set] = {}
    for st, path in subtype_to_path.items():
        for d in range(len(path)):
            node = "|".join(path[: d + 1])
            node_to_leaves.setdefault(node, set()).add(st)
    return {k: sorted(list(v)) for k, v in node_to_leaves.items()}


def node_to_leaves_from_subtype_paths(subtype_to_path: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return node_path -> list of leaf subtypes for all prefix nodes."""

    node_to_leaves: Dict[str, set] = {}
    for st, path in subtype_to_path.items():
        for d in range(len(path)):
            node = "|".join(path[: d + 1])
            node_to_leaves.setdefault(node, set()).add(st)
    return {k: sorted(list(v)) for k, v in node_to_leaves.items()}


def short_node_label_with_terminal(node_path: str, node_to_leaves: Dict[str, List[str]]) -> str:
    """Shorten a node path and optionally append the terminal leaf."""

    parts = node_path.split("|")
    last = parts[-1]

    fam = None
    for p in parts:
        if p.startswith("Calb1"):
            fam = "Calb1"
        elif p.startswith("Sox6"):
            fam = "Sox6"
        elif p.startswith("Gad2"):
            fam = "Gad2"
        elif p == "Lef1":
            fam = "Lef1"

    base = last if fam is None else (last if last.startswith(fam) else f"{fam} {last}")

    leaves = node_to_leaves.get(node_path, [])
    if len(leaves) == 1:
        return f"{base} ({leaves[0]})"
    return base


def node_probs_for_nodes(
    P_sub: np.ndarray,
    subtypes: List[str],
    nodes: List[str],
    node_to_leaves: Dict[str, List[str]],
) -> np.ndarray:
    """Sum leaf subtype probabilities under each node."""

    st2i = {s: i for i, s in enumerate(subtypes)}
    cols: List[np.ndarray] = []
    for node in nodes:
        leaves = node_to_leaves.get(node, [])
        idx = [st2i[s] for s in leaves if s in st2i]
        if len(idx) == 0:
            cols.append(np.zeros((P_sub.shape[0],), dtype=float))
        else:
            cols.append(P_sub[:, idx].sum(axis=1))
    return np.vstack(cols).T


if __name__ == "__main__":
    toy_tree = {
        "node": {
            "A": ["A1", "A2"],
            "B": {"B1": ["B1a"]},
        }
    }

    subtype_to_path = walk_paths(toy_tree)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    partitions = list(height_partitions(toy_tree, root_key="node"))

    subtypes = sorted(subtype_to_path.keys())
    P_sub = np.zeros((2, len(subtypes)), dtype=float)
    P_sub[0, subtypes.index("A1")] = 0.7
    P_sub[0, subtypes.index("A2")] = 0.3
    P_sub[1, subtypes.index("B1a")] = 1.0

    nodes = partitions[1]
    P_nodes = node_probs_for_nodes(P_sub, subtypes, nodes, node_to_leaves)

    print("subtype_to_path:", subtype_to_path)
    print("node_to_leaves:", node_to_leaves)
    print("partitions:", partitions)
    print("nodes:", nodes)
    print("P_nodes:", P_nodes)
    for n in nodes:
        print(n, "->", short_node_label_with_terminal(n, node_to_leaves))
