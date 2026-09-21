"""Self-contained HMoE v4 inference core (Gaertner DA subtypes)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

GAERTNER_SUBTYPES = [
    "Calb1:Ccdc192", "Calb1:Chrm2", "Calb1:Gipr", "Calb1:Kctd8",
    "Calb1:Lpar1", "Calb1:Pde11a", "Calb1:Ptprt", "Calb1:Sox6",
    "Calb1:Stac", "Calb1:Sulf1",
    "Gad2:Ebf2", "Gad2:Egfr",
    "Sox6:Arhgap28", "Sox6:Kcnmb2", "Sox6:March3", "Sox6:Tafa1",
    "Sox6:Tmem132d", "Sox6:Vcan",
]


def infer_family(subtype: str) -> str:
    """Map a Gaertner subtype label to its family."""
    s = str(subtype)
    if s.startswith("Sox6:"):
        return "Sox6"
    if s.startswith("Calb1:"):
        return "Calb1"
    if s.startswith("Gad2:"):
        return "Gad2"
    return "Other"


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.generic)) or value is None


def _validate_safe(obj: Any, _path: str = "root") -> None:
    """Reject checkpoints that smuggle in non-primitive python objects."""
    import torch

    if _is_primitive(obj) or isinstance(obj, np.ndarray):
        return
    if isinstance(obj, torch.Tensor):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Unsafe key type at {_path}: {type(k)}")
            _validate_safe(v, f"{_path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _validate_safe(v, f"{_path}[{i}]")
        return
    raise TypeError(f"Unsafe object at {_path}: {type(obj)}")


class _DummyModel:
    """Stand-in returned for a missing child checkpoint (emits -inf logits)."""

    def predict_child_logits(self, X: np.ndarray, *, gene_idx=None) -> np.ndarray:
        return np.full((int(np.asarray(X).shape[0]),), -1e9, dtype=float)


def load_child_model(path: str | Path) -> Any:
    """Load a single child-model bundle; return a dummy if the file is absent."""
    import torch

    path = Path(path)
    if not path.exists():
        return _DummyModel()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    _validate_safe(obj)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Expected a state_dict-only checkpoint: {path}")
    state = obj.get("state_dict", {})
    if isinstance(state, dict) and state.get("kind") is None and obj.get("meta", {}).get("kind"):
        state["kind"] = obj["meta"]["kind"]
    return state


def _subset_by_idx(X: np.ndarray, gene_idx: Optional[Sequence[int]]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if gene_idx is None:
        return X
    idx = np.asarray(list(gene_idx), dtype=int)
    if idx.size == 0:
        return X
    if idx.max(initial=-1) >= X.shape[1]:
        raise ValueError(f"gene_idx out of range for X with shape {X.shape}")
    return X[:, idx]


def _apply_scaler_if_present(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    scaler = model_bundle.get("scaler")
    if not scaler:
        return X
    if scaler.get("kind") == "standard":
        mu = np.asarray(scaler["mean"], dtype=np.float32)
        sigma = np.asarray(scaler["scale"], dtype=np.float32)
        return (X - mu) / sigma
    return X


def predict_child_logits(model_bundle: Any, X: np.ndarray, *, gene_idx=None) -> np.ndarray:
    """Predict binary logits for one child classifier (logreg/ffn/cnn)."""
    if isinstance(model_bundle, _DummyModel):
        return model_bundle.predict_child_logits(X, gene_idx=gene_idx)
    if isinstance(model_bundle, dict) and model_bundle.get("kind") == "dummy":
        n = int(np.asarray(X).shape[0])
        return np.full((n,), float(model_bundle.get("logit_value", -1e9)), dtype=float)
    if not isinstance(model_bundle, dict):
        raise TypeError("model_bundle must be a dict or dummy model")

    kind = str(model_bundle.get("kind", "")).lower()
    model_gene_idx = model_bundle.get("gene_idx") or []
    use_idx = list(gene_idx) if gene_idx is not None else model_gene_idx
    X_use = _subset_by_idx(X, use_idx)

    if kind == "logreg":
        X_use = _apply_scaler_if_present(model_bundle, X_use)
        weights = np.asarray(model_bundle.get("weights"), dtype=float).reshape(-1)
        bias = float(np.asarray(model_bundle.get("bias", 0.0)).reshape(-1)[0])
        if X_use.shape[1] != weights.shape[0]:
            raise ValueError("logreg weights do not match feature dimension")
        return X_use @ weights + bias

    if kind == "ffn":
        import torch
        import torch.nn as nn

        hidden_dim = int(model_bundle.get("hidden_dim", 64))
        dropout = float(model_bundle.get("dropout", 0.1))

        class BinaryFFN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(X_use.shape[1], hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(1)

        model = BinaryFFN()
        model.load_state_dict(model_bundle.get("state_dict", {}))
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X_use).float()).cpu().numpy().reshape(-1)

    if kind == "cnn":
        import torch
        import torch.nn as nn

        conv1 = int(model_bundle.get("conv1", 16))
        conv2 = int(model_bundle.get("conv2", 32))
        kernel = int(model_bundle.get("kernel", 5))

        class BinaryCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                pad = kernel // 2
                self.conv1 = nn.Conv1d(1, conv1, kernel, padding=pad)
                self.conv2 = nn.Conv1d(conv1, conv2, kernel, padding=pad)
                self.relu = nn.ReLU()
                self.out = nn.Linear(conv2, 1)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = x.mean(dim=2)
                return self.out(x).squeeze(1)

        model = BinaryCNN()
        model.load_state_dict(model_bundle.get("state_dict", {}))
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_use).float().unsqueeze(1)
            return model(xt).cpu().numpy().reshape(-1)

    raise ValueError(f"Unsupported child model kind: {kind}")


def walk_paths(tree: Any, prefix: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Map each leaf subtype to its node path through the dendrogram."""
    if prefix is None:
        prefix = []
    out: Dict[str, List[str]] = {}
    if isinstance(tree, list):
        for st in tree:
            out[st] = list(prefix)
        return out
    if isinstance(tree, dict):
        for label, child in tree.items():
            out.update(walk_paths(child, prefix + [label]))
        return out
    raise TypeError(f"Unexpected node type: {type(tree)}")


def sct_transform(X: np.ndarray) -> np.ndarray:
    """SCT Pearson residuals from raw counts (numpy-only), as in the pipeline."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    lib = X.sum(1)
    gr = X.sum(0) / lib.sum()
    mu = np.outer(lib, gr)
    d = np.sqrt(mu + mu**2 / 100.0)
    d[d < 1e-12] = 1.0
    r = np.clip((X - mu) / d, -np.sqrt(n), np.sqrt(n))
    r[np.isnan(r)] = 0.0
    return r.astype(np.float32)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def align_features(var_names: Sequence[str], X, gate_feats: Sequence[str]):
    """Align query genes onto the model's 21,604-gene training space."""
    import scipy.sparse as sp

    src_genes = {str(g).title(): i for i, g in enumerate(var_names)}
    ortho_fallback: Dict[str, str] = {}
    for g in var_names:
        m = re.match(r"^ZNF(\d+[A-Z]?)$", str(g).upper())
        if m:
            mouse_equiv = f"Zfp{m.group(1).lower()}".title()
            if mouse_equiv not in src_genes:
                ortho_fallback[mouse_equiv] = str(g).title()

    human_pos, mouse_pos, n_ortho = [], [], 0
    for m_pos, mg in enumerate(gate_feats):
        key = str(mg).title()
        if key in src_genes:
            human_pos.append(src_genes[key])
            mouse_pos.append(m_pos)
        elif key in ortho_fallback:
            human_key = ortho_fallback[key]
            if human_key in src_genes:
                human_pos.append(src_genes[human_key])
                mouse_pos.append(m_pos)
                n_ortho += 1

    n = X.shape[0]
    X_aligned = np.zeros((n, len(gate_feats)), dtype=np.float32)
    if human_pos:
        X_sub = X[:, human_pos]
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        X_aligned[:, mouse_pos] = np.asarray(X_sub, dtype=np.float32)
    return X_aligned, len(mouse_pos), n_ortho


def _safe_id(v: str) -> str:
    return str(v).replace("|", "__").replace(":", "_").replace("/", "_")


def load_model(model_dir: str | Path) -> Dict[str, Any]:
    """Load the unified HMoE model bundle from `model_dir/v4_improved_unified`."""
    model_dir = Path(model_dir)
    uni_dir = model_dir / "v4_improved_unified"
    meta = json.loads((uni_dir / "meta.json").read_text())

    raw_gates: Dict[str, Any] = {}
    sct_gates: Dict[str, Any] = {}
    for gk in meta["gate_nodes"]:
        rt = meta["gate_repr"][gk]
        children = meta["gates"][gk]
        sk = _safe_id(gk)
        child_models = {}
        for c in children:
            sc_ = _safe_id(c)
            child_models[c] = load_child_model(
                uni_dir / "child_models" / rt / sk / f"{sc_}.safe.pt"
            )
        target = sct_gates if rt == "sct" else raw_gates
        target[gk] = {"children": children, "child_models": child_models}

    excluded = set(meta.get("excluded_leaves", []))
    all_leaves = sorted(walk_paths(meta["tree"]).keys())
    leaves = [l for l in all_leaves if l not in excluded]

    feature_names = json.loads(
        (model_dir / "family_gate" / "features.json").read_text()
    )

    return {
        "tree": meta["tree"],
        "root_key": meta["root_key"],
        "leaves": leaves,
        "gate_repr": meta["gate_repr"],
        "raw_gates": raw_gates,
        "sct_gates": sct_gates,
        "feature_names": feature_names,
    }


def predict_proba(bundle: Dict[str, Any], X_raw: np.ndarray) -> np.ndarray:
    """Unified prediction: raw counts for shallow gates, SCT for deep gates."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    X_sct = sct_transform(X_raw)

    tree = bundle["tree"]
    rk = bundle["root_key"]
    leaves = bundle["leaves"]
    l2i = {l: i for i, l in enumerate(leaves)}
    gr = bundle.get("gate_repr", {})
    rg = bundle.get("raw_gates", {})
    sg = bundle.get("sct_gates", {})
    n = X_raw.shape[0]
    P = np.zeros((n, len(leaves)), dtype=float)

    def _gn(t, path):
        s = t
        for k in path:
            if isinstance(s, dict) and k in s:
                s = s[k]
            else:
                return None
        return s

    def _go(np_):
        rt = gr.get(np_, "raw")
        g = (sg if rt == "sct" else rg).get(np_)
        X = X_sct if rt == "sct" else X_raw
        if g is None:
            return None
        ch = list(g.get("children", []))
        if not ch:
            return None
        ll = []
        for c in ch:
            cm = g["child_models"].get(c)
            if cm is not None:
                ll.append(predict_child_logits(cm, X, gene_idx=None))
            else:
                ll.append(np.zeros(n))
        return {"probs": _softmax_np(np.stack(ll, axis=1)), "nodes": ch}

    def _rt(path, mass):
        np_ = "|".join(path)
        sub = _gn(tree, path)
        if isinstance(sub, dict):
            ck = list(sub.keys())
            cf = [f"{np_}|{k}" for k in ck]
            g = _go(np_)
            if g is None:
                pr = np.full((n, len(ck)), 1.0 / len(ck))
            else:
                pr = np.asarray(g["probs"], dtype=float)
                if g["nodes"] != cf:
                    pr = pr[:, [g["nodes"].index(c) for c in cf]]
            pr = pr / np.maximum(pr.sum(1, keepdims=True), 1e-12)
            for j, c in enumerate(ck):
                _rt(path + [c], mass * pr[:, j])
        elif isinstance(sub, list):
            ll = list(sub)
            g = _go(np_)
            if g is None or len(ll) == 1:
                pr = np.full((n, len(ll)), 1.0 / len(ll))
            else:
                pr = np.asarray(g["probs"], dtype=float)
                if g["nodes"] != ll:
                    pr = pr[:, [g["nodes"].index(c) for c in ll]]
            pr = pr / np.maximum(pr.sum(1, keepdims=True), 1e-12)
            for j, l in enumerate(ll):
                k = l2i.get(l)
                if k is not None:
                    P[:, k] += mass * pr[:, j]

    _rt([rk], np.ones(n, dtype=float))
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    return P
