#!/usr/bin/env python3
"""Freeze the Height=5 tree partition mapping for Figure 2.

Run: python build_height5_partition.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
FROZEN = Path(__file__).resolve().parent.parent / "frozen"
PARTITION_HEIGHT = 5

from awatramani_lab.moe.v4.hierarchy import (  # noqa: E402
    height_partitions, walk_paths, node_to_leaves_from_subtype_paths,
    short_node_label_with_terminal,
)

meta = json.loads(
    (REPO_ROOT / "hmoe_annotate/model/v4_improved_unified/meta.json").read_text()
)
tree = meta["tree"]
partitions = list(height_partitions(tree, root_key="node"))
stp = walk_paths(tree)
n2l = node_to_leaves_from_subtype_paths(stp)

hp = partitions[PARTITION_HEIGHT]
leaf_to_hp = {}
for node_path in hp:
    leaves = n2l.get(node_path, [node_path])
    label = short_node_label_with_terminal(node_path, n2l)
    for leaf in leaves:
        leaf_to_hp[leaf] = label

out = {"height": PARTITION_HEIGHT, "hp_order": list(hp), "leaf_to_hp": leaf_to_hp}
FROZEN.mkdir(parents=True, exist_ok=True)
(FROZEN / "kamath_height5_partition.json").write_text(json.dumps(out, indent=2))
print(f"wrote kamath_height5_partition.json: {len(leaf_to_hp)} leaves -> "
      f"{len(set(leaf_to_hp.values()))} partitions (height {PARTITION_HEIGHT})")
