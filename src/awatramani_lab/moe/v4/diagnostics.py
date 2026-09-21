"""Diagnostic utilities for MoE v4 bundles."""

from __future__ import annotations

from typing import Any, Dict, List


def explain_missing_gates(
    bundle: Dict[str, Any],
    missing: List[str],
    max_show: int = 50,
) -> None:
    """Print detailed information about missing gates in a bundle."""
    node_to_leaves = bundle["node_to_leaves"]
    gates = bundle.get("gates", {})
    log_nodes = (bundle.get("log", {}) or {}).get("nodes", {})

    def leaves_count(k: str) -> int:
        return len(node_to_leaves.get(k, []))

    for g in missing[:max_show]:
        print("\n" + "=" * 100)
        print("MISSING:", g)
        print("leaves in this node:", leaves_count(g))

        if g in gates:
            kids = gates[g].get("children", [])
            print("children:", kids)
            for c in kids:
                print(
                    "  child:",
                    c,
                    "leaf_count:",
                    leaves_count(c),
                    "leaf_sample:",
                    node_to_leaves.get(c, [])[:3],
                )
        else:
            print(
                "not present in bundle['gates'] keys "
                "(so training never attempted or pruned earlier)"
            )

        ln = log_nodes.get(g)
        if ln is None:
            print("no log entry for this node")
        else:
            cs = ln.get("child_status", {})
            print("log child_status keys:", list(cs.keys()))
            for ck, info in cs.items():
                print(" ", ck, info)
