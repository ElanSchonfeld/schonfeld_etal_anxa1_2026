"""Generic tree utilities for MoE v4 routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .hierarchy import node_to_leaves_from_subtype_paths, walk_paths
from .routing_spec import EXCLUDED_LEAVES, filter_excluded_leaves


@dataclass(frozen=True)
class TreeIndex:
    root: str
    subtype_to_path: Dict[str, List[str]]
    node_to_children: Dict[str, List[str]]
    node_to_leaves: Dict[str, List[str]]
    leaves: List[str]

    def get_children(self, node: str) -> List[str]:
        return list(self.node_to_children.get(node, []))

    def get_leaves(self, node: str) -> List[str]:
        return list(self.node_to_leaves.get(node, []))

    def leaf_to_path(self, leaf: str) -> List[str]:
        path = self.subtype_to_path.get(leaf)
        if path is None:
            raise KeyError(f"Unknown leaf subtype: {leaf}")
        node_paths = ["|".join(path[: i + 1]) for i in range(len(path))]
        return node_paths + [leaf]

    def is_leaf(self, name: str) -> bool:
        return name in set(self.leaves)

    def node_depth(self, node: str) -> int:
        return max(0, len(node.split("|")) - 1)


def node_dir_name(node_name: str) -> str:
    return node_name.replace("|", "__")


def _append_child(node_to_children: Dict[str, List[str]], node: str, child: str) -> None:
    children = node_to_children.setdefault(node, [])
    if child not in children:
        children.append(child)


def _build_node_to_children(subtype_to_path: Dict[str, List[str]]) -> Dict[str, List[str]]:
    node_to_children: Dict[str, List[str]] = {}
    for leaf, path in subtype_to_path.items():
        if not path:
            continue
        for depth in range(len(path)):
            node = "|".join(path[: depth + 1])
            if depth + 1 < len(path):
                child = "|".join(path[: depth + 2])
            else:
                child = leaf
            _append_child(node_to_children, node, child)
    return node_to_children


def build_tree_index_from_subtype_paths(
    subtype_to_path: Dict[str, List[str]],
    *,
    root_key: str = "node",
    excluded_leaves: Optional[Iterable[str]] = None,
) -> TreeIndex:
    excluded = list(excluded_leaves) if excluded_leaves is not None else EXCLUDED_LEAVES
    leaves = list(subtype_to_path.keys())
    kept, _ = filter_excluded_leaves(leaves, excluded)
    subtype_to_path = {leaf: subtype_to_path[leaf] for leaf in kept}

    node_to_children = _build_node_to_children(subtype_to_path)
    node_to_leaves = node_to_leaves_from_subtype_paths(subtype_to_path)
    return TreeIndex(
        root=root_key,
        subtype_to_path=subtype_to_path,
        node_to_children=node_to_children,
        node_to_leaves=node_to_leaves,
        leaves=sorted(list(subtype_to_path.keys())),
    )


def build_tree_index_from_hierarchy(
    hierarchy_tree: Dict[str, object],
    *,
    root_key: str = "node",
    excluded_leaves: Optional[Iterable[str]] = None,
) -> TreeIndex:
    subtype_to_path = walk_paths(hierarchy_tree)
    return build_tree_index_from_subtype_paths(
        subtype_to_path,
        root_key=root_key,
        excluded_leaves=excluded_leaves,
    )


def child_to_leaves(tree: TreeIndex, child: str) -> List[str]:
    if child in tree.leaves:
        return [child]
    leaves = tree.node_to_leaves.get(child)
    if leaves is None:
        raise KeyError(f"Unknown child node or leaf: {child}")
    return list(leaves)


def internal_nodes(tree: TreeIndex) -> List[str]:
    return [node for node, children in tree.node_to_children.items() if children]


def leaf_edges(tree: TreeIndex, leaf: str) -> List[Tuple[str, str]]:
    """Return list of (parent_node, child_name) edges for a leaf."""

    path = tree.subtype_to_path.get(leaf)
    if path is None:
        raise KeyError(f"Unknown leaf subtype: {leaf}")
    edges: List[Tuple[str, str]] = []
    for depth in range(len(path)):
        parent = "|".join(path[: depth + 1])
        if depth + 1 < len(path):
            child = "|".join(path[: depth + 2])
        else:
            child = leaf
        edges.append((parent, child))
    return edges


def node_edges(tree: TreeIndex, node: str) -> List[Tuple[str, str]]:
    """Return list of (parent_node, child_name) edges for a node path."""

    parts = node.split("|")
    edges: List[Tuple[str, str]] = []
    for depth in range(len(parts) - 1):
        parent = "|".join(parts[: depth + 1])
        child = "|".join(parts[: depth + 2])
        edges.append((parent, child))
    return edges
