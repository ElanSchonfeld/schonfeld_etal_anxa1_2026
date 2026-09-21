#!/usr/bin/env python3
"""Build unified HMoE model: raw-counts shallow gates + SCT deep gates."""

import gc
import os
import sys
import types
sys.modules["xgboost"] = types.ModuleType("xgboost")

import json
import shutil
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SOURCE_ROOT = Path(os.environ.get("M2H_SOURCE_ROOT", PROJECT_ROOT))
DATA_DIR = SOURCE_ROOT / "Data"
ARTIFACT_ROOT = Path(
    os.environ.get("HMOE_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "fig1_hmoe")
)
MODEL_TEMPLATE = (
    PROJECT_ROOT / "hmoe_annotate" / "model" / "v4_improved_unified" / "meta.json"
)
OUT_DIR = ARTIFACT_ROOT / "model" / "v4_improved_unified"
RAW_TMP = ARTIFACT_ROOT / "work" / "unified_raw"
SCT_TMP = ARTIFACT_ROOT / "work" / "unified_sct"

from awatramani_lab.moe.v4.pipelines import train_v4_improved
from awatramani_lab.moe.v4.child_models import predict_child_logits, load_child_model
from awatramani_lab.moe.v4.hierarchy import walk_paths


SHALLOW_DEPTH = 3


def gate_depth(node_key):
    return node_key.count("|")


def infer_family(st):
    prefix = str(st).split(":", 1)[0]
    if prefix in {"Sox6", "Calb1", "Gad2"}:
        return prefix
    return "Other"


def safe_id(value):
    return str(value).replace("|", "__").replace(":", "_").replace("/", "_")


def sct_transform(X_raw):
    """Compute SCT Pearson residuals from raw counts (numpy-only, no scanpy dependency at predict time)."""
    X = np.asarray(X_raw, dtype=np.float64)
    n_cells = X.shape[0]
    lib_sizes = X.sum(axis=1)
    total_lib = lib_sizes.sum()
    gene_rates = X.sum(axis=0) / total_lib
    mu = np.outer(lib_sizes, gene_rates)
    theta = 100.0
    denom = np.sqrt(mu + mu**2 / theta)
    denom[denom < 1e-12] = 1.0
    residuals = (X - mu) / denom
    clip_val = np.sqrt(n_cells)
    residuals = np.clip(residuals, -clip_val, clip_val)
    residuals[np.isnan(residuals)] = 0.0
    return residuals.astype(np.float32)


def predict_unified(bundle, X_raw):
    """Predict using unified model: raw counts for shallow gates, SCT for deep gates."""
    from awatramani_lab.moe.v4.pipelines import _softmax

    X_raw = np.asarray(X_raw, dtype=np.float32)

    X_sct = sct_transform(X_raw)

    tree = bundle["tree"]
    root_key = bundle["root_key"]
    leaves = list(bundle["leaves"])
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}
    gate_repr = bundle.get("gate_repr", {})
    raw_gates = bundle.get("raw_gates", {})
    sct_gates = bundle.get("sct_gates", {})

    n_samples = X_raw.shape[0]
    P_leaf = np.zeros((n_samples, len(leaves)), dtype=float)

    def _get_gate_output(node_path):
        """Get gate probabilities using the appropriate representation."""
        repr_type = gate_repr.get(node_path, "raw")
        if repr_type == "sct":
            gate = sct_gates.get(node_path)
            X = X_sct
        else:
            gate = raw_gates.get(node_path)
            X = X_raw

        if gate is None:
            return None

        children = list(gate.get("children", []))
        if not children:
            return None

        logits_list = []
        for child in children:
            model = gate.get("child_models", {}).get(child)
            if model is None:
                logits_list.append(np.zeros(n_samples))
            else:
                logits_list.append(predict_child_logits(model, X, gene_idx=None))

        logits = np.stack(logits_list, axis=1)
        probs = _softmax(logits)
        return {"probs": probs, "nodes": children}

    def _route(path, mass):
        node_path = "|".join(path)
        sub = _get_node(tree, path)

        if isinstance(sub, dict):
            child_keys = list(sub.keys())
            child_full = [f"{node_path}|{key}" for key in child_keys]
            gate = _get_gate_output(node_path)
            if gate is None:
                probs = np.full((n_samples, len(child_keys)), 1.0 / len(child_keys))
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != child_full:
                    idx = [gate_nodes.index(ch) for ch in child_full]
                    probs = probs[:, idx]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            for j, child in enumerate(child_keys):
                _route(path + [child], mass * probs[:, j])

        elif isinstance(sub, list):
            leaf_labels = list(sub)
            gate = _get_gate_output(node_path)
            if gate is None or len(leaf_labels) == 1:
                probs = np.full((n_samples, len(leaf_labels)), 1.0 / len(leaf_labels))
            else:
                probs = np.asarray(gate["probs"], dtype=float)
                gate_nodes = list(gate.get("nodes", []))
                if gate_nodes and gate_nodes != leaf_labels:
                    idx = [gate_nodes.index(ch) for ch in leaf_labels]
                    probs = probs[:, idx]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            for j, leaf in enumerate(leaf_labels):
                k = leaf_to_idx.get(leaf)
                if k is not None:
                    P_leaf[:, k] += mass * probs[:, j]

    _route([root_key], np.ones(n_samples, dtype=float))
    P_leaf = P_leaf / np.maximum(P_leaf.sum(axis=1, keepdims=True), 1e-12)

    return {"P_sub": P_leaf, "leaves": leaves}


def _get_node(tree, path):
    """Navigate tree by path."""
    sub = tree
    for key in path:
        if isinstance(sub, dict) and key in sub:
            sub = sub[key]
        else:
            return None
    return sub


def main():
    print("=" * 70)
    print("BUILD UNIFIED HMoE: RAW SHALLOW + SCT DEEP")
    print("=" * 70)

    print("\n1. Loading mouse data...")
    mouse_path = DATA_DIR / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad"
    if not mouse_path.is_file():
        raise SystemExit(
            f"Missing {mouse_path}. Set M2H_SOURCE_ROOT to the authorized source-data workspace."
        )
    mouse = ad.read_h5ad(mouse_path)
    y_sub = mouse.obs["subtype"].values.astype(str).copy()
    gene_names = list(mouse.var_names)
    X_raw = mouse.X
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)
    if np.nanmin(X_raw) < 0 or not np.allclose(X_raw[:500], np.round(X_raw[:500])):
        raise ValueError("Mouse HMoE training input must contain raw non-negative integer counts")
    print(f"   {X_raw.shape[0]} cells, {X_raw.shape[1]} genes")

    print("   Computing SCT Pearson residuals...")
    X_sct = sct_transform(X_raw)
    print(f"   SCT range: [{X_sct.min():.2f}, {X_sct.max():.2f}]")
    del mouse; gc.collect()

    meta = json.loads(MODEL_TEMPLATE.read_text())

    print(f"\n2. Training RAW gates (C=1.0, topk=200)...")
    train_v4_improved(
        X_raw, y_sub, meta["tree"], root_key=meta["root_key"],
        gene_names=gene_names, model_type="logreg", feature_method="f_classif",
        topk=200, min_pct=0.005, min_diff_pct=0.001, height_train=8,
        shared_features=False, seed=42,
        excluded_leaves=meta.get("excluded_leaves", []),
        out_dir=str(RAW_TMP),
        model_hyperparams={"C": 1.0},
    )

    print(f"\n3. Training SCT gates (C=0.1, topk=200)...")
    train_v4_improved(
        X_sct, y_sub, meta["tree"], root_key=meta["root_key"],
        gene_names=gene_names, model_type="logreg", feature_method="f_classif",
        topk=200, min_pct=0.005, min_diff_pct=0.001, height_train=8,
        shared_features=False, seed=42,
        excluded_leaves=meta.get("excluded_leaves", []),
        out_dir=str(SCT_TMP),
        model_hyperparams={"C": 0.1},
    )

    print(f"\n4. Assembling unified model (shallow≤{SHALLOW_DEPTH}=raw, deeper=sct)...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "child_models" / "raw").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "child_models" / "sct").mkdir(parents=True, exist_ok=True)

    gate_repr = {}
    for gate_key in meta["gate_nodes"]:
        depth = gate_depth(gate_key)
        if depth <= SHALLOW_DEPTH:
            repr_type = "raw"
            src_dir = RAW_TMP
        else:
            repr_type = "sct"
            src_dir = SCT_TMP
        gate_repr[gate_key] = repr_type

        safe_key = safe_id(gate_key)
        src_cm = src_dir / "child_models" / safe_key
        dst_cm = OUT_DIR / "child_models" / repr_type / safe_key
        if src_cm.exists():
            if dst_cm.exists():
                shutil.rmtree(dst_cm)
            shutil.copytree(src_cm, dst_cm)
        print(f"   {'RAW':4s} depth={depth}" if repr_type == "raw"
              else f"   {'SCT':4s} depth={depth}", gate_key.split("|")[-1])

    meta_out = meta.copy()
    meta_out["pipeline"] = "v4_improved_unified"
    meta_out["gate_repr"] = gate_repr
    meta_out["preprocessing"] = {
        "shallow": {"depth_max": SHALLOW_DEPTH, "repr": "raw_counts", "C": 1.0, "topk": 200},
        "deep": {"depth_min": SHALLOW_DEPTH + 1, "repr": "sct_pearson_residuals", "C": 0.1, "topk": 200},
        "note": "Users provide raw counts. Pipeline computes SCT internally for deep gates.",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta_out, indent=2))
    feature_dir = OUT_DIR.parent / "family_gate"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "features.json").write_text(json.dumps(gene_names))

    shutil.rmtree(RAW_TMP, ignore_errors=True)
    shutil.rmtree(SCT_TMP, ignore_errors=True)

    print("\n5. Loading unified model...")
    meta_loaded = json.loads((OUT_DIR / "meta.json").read_text())
    raw_gates = {}
    sct_gates = {}
    for gate_key in meta_loaded["gate_nodes"]:
        repr_type = meta_loaded["gate_repr"][gate_key]
        children = meta_loaded["gates"][gate_key]
        safe_key = safe_id(gate_key)
        child_models = {}
        for child in children:
            safe_child = safe_id(child)
            model_path = OUT_DIR / "child_models" / repr_type / safe_key / f"{safe_child}.safe.pt"
            child_models[child] = load_child_model(model_path)
        gate_data = {"children": children, "child_models": child_models}
        if repr_type == "raw":
            raw_gates[gate_key] = gate_data
        else:
            sct_gates[gate_key] = gate_data

    bundle = {
        "tree": meta_loaded["tree"],
        "root_key": meta_loaded["root_key"],
        "leaves": sorted(walk_paths(meta_loaded["tree"]).keys()),
        "gate_repr": meta_loaded["gate_repr"],
        "raw_gates": raw_gates,
        "sct_gates": sct_gates,
    }
    excluded = set(meta_loaded.get("excluded_leaves", []))
    bundle["leaves"] = [l for l in bundle["leaves"] if l not in excluded]
    leaves = bundle["leaves"]

    print("   Mouse resubstitution (unified model)...")
    result = predict_unified(bundle, X_raw)
    P_sub = result["P_sub"]
    preds = [leaves[i] for i in P_sub.argmax(axis=1)]
    ms = set(leaves); v = np.array([s in ms for s in y_sub])
    gf = np.array([infer_family(s) for s in y_sub[v]])
    pf = np.array([infer_family(s) for s in np.array(preds)[v]])
    print(f"   Subtype: {(y_sub[v]==np.array(preds)[v]).mean():.1%}, "
          f"Family: {(gf==pf).mean():.1%}")
    cp = Counter([p for p in preds if "Calb1" in p])
    cg = Counter([s for s in y_sub if "Calb1" in s])
    print(f"   Calb1 Gipr: GT={cg.get('Calb1:Gipr',0)} Pred={cp.get('Calb1:Gipr',0)}")
    print(f"   Calb1 Sox6: GT={cg.get('Calb1:Sox6',0)} Pred={cp.get('Calb1:Sox6',0)}")
    print(f"   Calb1 Sulf1: GT={cg.get('Calb1:Sulf1',0)} Pred={cp.get('Calb1:Sulf1',0)}")
    del X_raw, X_sct; gc.collect()

    print("\n" + "=" * 70)
    print("DONE: unified model saved to", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
