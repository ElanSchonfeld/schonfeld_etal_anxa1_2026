#!/usr/bin/env python3
"""Regenerate the frozen inputs for Figure 1 panel A from the unified model.

Run: python build_dendrogram_cache.py
"""
import json
import pickle
from os import environ
from pathlib import Path

import anndata as ad
import numpy as np

from awatramani_lab.moe.v4.hierarchy import (
    node_to_leaves_from_subtype_paths,
    walk_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(environ["M2H_SOURCE_ROOT"])
MODEL_META = REPO_ROOT / "hmoe_annotate" / "model" / "v4_improved_unified" / "meta.json"
MOUSE_H5AD = SOURCE_ROOT / "Data" / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad"

FROZEN = Path(__file__).resolve().parent.parent / "frozen"

model = json.loads(MODEL_META.read_text())
subtype_to_path = walk_paths(model["tree"])
excluded = set(model.get("excluded_leaves", []))
leaves = [leaf for leaf in sorted(subtype_to_path) if leaf not in excluded]
meta = {
    "tree": model["tree"],
    "leaves": leaves,
    "node_to_leaves": node_to_leaves_from_subtype_paths(subtype_to_path),
    "excluded_leaves": sorted(excluded),
}
FROZEN.mkdir(parents=True, exist_ok=True)
with (FROZEN / "hmoe_dendrogram_meta.pkl").open("wb") as handle:
    pickle.dump(meta, handle)
print(f"tree: {len(meta['leaves'])} leaves, {len(meta['node_to_leaves'])} nodes, excluded {meta['excluded_leaves']}")

mouse = ad.read_h5ad(MOUSE_H5AD)
y_sub = np.array(mouse.obs["subtype"].astype(str).values)
gate_keys = list(model["gate_nodes"])
save = {"y_sub": y_sub, "gate_keys": np.array(gate_keys, dtype=object)}
for gate_key in gate_keys:
    save[f"nodes_{gate_key}"] = np.array(model["gates"][gate_key], dtype=object)
np.savez_compressed(FROZEN / "dendrogram_cellcount_cache.npz", **save)
n = (FROZEN / "dendrogram_cellcount_cache.npz").stat().st_size
print(f"cached {len(gate_keys)} gate node-lists + {len(y_sub)} ground-truth labels "
      f"-> dendrogram_cellcount_cache.npz ({n/1024:.0f} KB)")
