#!/usr/bin/env python3
"""Held-out test metrics for the UNIFIED HMoE model (raw shallow + SCT deep)."""

import sys, json, gc, shutil
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split as skl_split
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef,
    average_precision_score, multilabel_confusion_matrix,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrain_hmoe_sct_hybrid import (
    gate_depth, safe_id, sct_transform, predict_unified, infer_family,
    SHALLOW_DEPTH, DATA_DIR, ARTIFACT_ROOT, MODEL_TEMPLATE,
)
from awatramani_lab.moe.v4.pipelines import train_v4_improved
from awatramani_lab.moe.v4.child_models import load_child_model
from awatramani_lab.moe.v4.hierarchy import walk_paths

SEED = 42
TEST_SIZE = 0.1
RAW_TMP = ARTIFACT_ROOT / "work" / "heldout_raw"
SCT_TMP = ARTIFACT_ROOT / "work" / "heldout_sct"
OUT_JSON = ARTIFACT_ROOT / "heldout_test_metrics_mouse.json"


def build_bundle(meta, raw_dir, sct_dir):
    """Assemble an in-memory unified bundle from freshly trained temp dirs."""
    raw_gates, sct_gates, gate_repr = {}, {}, {}
    for gk in meta["gate_nodes"]:
        rt = "raw" if gate_depth(gk) <= SHALLOW_DEPTH else "sct"
        gate_repr[gk] = rt
        src = raw_dir if rt == "raw" else sct_dir
        sk = safe_id(gk)
        cms = {c: load_child_model(src / "child_models" / sk / f"{safe_id(c)}.safe.pt")
               for c in meta["gates"][gk]}
        (raw_gates if rt == "raw" else sct_gates)[gk] = {
            "children": meta["gates"][gk], "child_models": cms}
    excluded = set(meta.get("excluded_leaves", []))
    leaves = [l for l in sorted(walk_paths(meta["tree"]).keys()) if l not in excluded]
    return {"tree": meta["tree"], "root_key": meta["root_key"], "leaves": leaves,
            "gate_repr": gate_repr, "raw_gates": raw_gates, "sct_gates": sct_gates}


def metrics(gt, P_sub, leaves):
    """Full multiclass metric suite (overall + macro + weighted + per-class)."""
    gt = np.asarray(gt)
    pred = np.array([leaves[i] for i in P_sub.argmax(1)])
    Y = label_binarize(gt, classes=leaves)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        gt, pred, labels=leaves, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        gt, pred, labels=leaves, average="weighted", zero_division=0)

    mcm = multilabel_confusion_matrix(gt, pred, labels=leaves)
    tn, fp = mcm[:, 0, 0].astype(float), mcm[:, 0, 1].astype(float)
    fn, tp = mcm[:, 1, 0].astype(float), mcm[:, 1, 1].astype(float)
    z = np.zeros(len(leaves))
    spec = np.divide(tn, tn + fp, out=z.copy(), where=(tn + fp) > 0)
    npv = np.divide(tn, tn + fn, out=z.copy(), where=(tn + fn) > 0)
    ppv = np.divide(tp, tp + fp, out=z.copy(), where=(tp + fp) > 0)
    sens = np.divide(tp, tp + fn, out=z.copy(), where=(tp + fn) > 0)
    f1c = np.divide(2 * ppv * sens, ppv + sens, out=z.copy(), where=(ppv + sens) > 0)
    support = tp + fn

    try:
        auc_c = np.atleast_1d(roc_auc_score(Y, P_sub, average=None))
    except Exception:
        auc_c = np.full(len(leaves), np.nan)
    try:
        ap_c = np.atleast_1d(average_precision_score(Y, P_sub, average=None))
    except Exception:
        ap_c = np.full(len(leaves), np.nan)

    def wmean(a):
        a = np.asarray(a, float); m = np.isfinite(a)
        return float(np.average(a[m], weights=support[m])) if m.any() and support[m].sum() > 0 else float("nan")

    def _fam(s):
        return s.split(":")[0] if ":" in s else infer_family(s)
    gt_fam = np.array([_fam(s) for s in gt])
    pr_fam = np.array([_fam(s) for s in pred])

    per_class = {leaves[i]: {
        "precision_ppv": float(ppv[i]), "recall_sensitivity": float(sens[i]),
        "specificity": float(spec[i]), "npv": float(npv[i]), "f1": float(f1c[i]),
        "roc_auc": float(auc_c[i]), "pr_auc": float(ap_c[i]), "support": int(support[i]),
    } for i in range(len(leaves))}

    return {
        "accuracy": float(accuracy_score(gt, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(gt, pred)),
        "precision_macro": float(p_mac), "recall_macro": float(r_mac), "f1_macro": float(f_mac),
        "precision_weighted": float(p_wt), "recall_weighted": float(r_wt), "f1_weighted": float(f_wt),
        "specificity_macro": float(np.nanmean(spec)), "specificity_weighted": wmean(spec),
        "npv_macro": float(np.nanmean(npv)), "npv_weighted": wmean(npv),
        "roc_auc_macro": float(np.nanmean(auc_c)), "roc_auc_weighted": wmean(auc_c),
        "pr_auc_macro": float(np.nanmean(ap_c)), "pr_auc_weighted": wmean(ap_c),
        "cohen_kappa": float(cohen_kappa_score(gt, pred, labels=leaves)),
        "mcc": float(matthews_corrcoef(gt, pred)),
        "family_accuracy": float((gt_fam == pr_fam).mean()),
        "mean_confidence": float(P_sub.max(1).mean()),
        "macro_f1": float(f_mac), "macro_roc_auc": float(np.nanmean(auc_c)),
        "n": int(len(gt)), "n_correct": int((gt == pred).sum()),
        "per_class": per_class,
    }


def main():
    print("=" * 70)
    print("HELD-OUT TEST METRICS: UNIFIED HMoE (raw shallow + SCT deep)")
    print("=" * 70)

    meta = json.loads(MODEL_TEMPLATE.read_text())
    EXCL = set(meta.get("excluded_leaves", []))

    print("\n1. Loading mouse LRRK2 (raw counts)...")
    mouse = ad.read_h5ad(DATA_DIR / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad")
    y = mouse.obs["subtype"].values.astype(str)
    gene_names = list(mouse.var_names)
    X = mouse.X.toarray() if sp.issparse(mouse.X) else np.asarray(mouse.X)
    X = np.asarray(X, dtype=np.float32)
    del mouse; gc.collect()
    print(f"   {X.shape[0]} cells x {X.shape[1]} genes, dtype check int={np.allclose(X[:100], np.round(X[:100]))}")

    valid_idx = np.where(~np.isin(y, list(EXCL)))[0]
    gt_valid = y[valid_idx]
    tr_loc, te_loc = skl_split(np.arange(len(valid_idx)), test_size=TEST_SIZE,
                               random_state=SEED, stratify=gt_valid)
    train_idx = valid_idx[tr_loc]
    test_idx = valid_idx[te_loc]
    print(f"\n2. Split: train={len(train_idx)}  test={len(test_idx)}  (canonical = 21977 / 2442)")
    assert len(train_idx) == 21977 and len(test_idx) == 2442, "split size mismatch!"

    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    for d in (RAW_TMP, SCT_TMP):
        shutil.rmtree(d, ignore_errors=True)
    common = dict(gene_names=gene_names, model_type="logreg", feature_method="f_classif",
                  topk=200, min_pct=0.005, min_diff_pct=0.001, height_train=8,
                  shared_features=False, seed=SEED,
                  excluded_leaves=meta.get("excluded_leaves", []),
                  root_key=meta["root_key"])
    print("\n3a. Training RAW gates (C=1.0) on train split...")
    train_v4_improved(Xtr, ytr, meta["tree"], out_dir=str(RAW_TMP),
                      model_hyperparams={"C": 1.0}, **common)
    print("3b. Computing SCT(train) and training SCT gates (C=0.1)...")
    Xtr_sct = sct_transform(Xtr)
    train_v4_improved(Xtr_sct, ytr, meta["tree"], out_dir=str(SCT_TMP),
                      model_hyperparams={"C": 0.1}, **common)
    del Xtr_sct; gc.collect()

    bundle = build_bundle(meta, RAW_TMP, SCT_TMP)
    leaves = bundle["leaves"]
    print(f"\n4. Assembled unified bundle: {len(leaves)} leaves, "
          f"{sum(v=='raw' for v in bundle['gate_repr'].values())} raw / "
          f"{sum(v=='sct' for v in bundle['gate_repr'].values())} sct gates")

    print("\n5. Predicting on HELD-OUT test cells (no Gad2 reassignment; mouse GT has Gad2 subtypes)...")
    P_test = predict_unified(bundle, Xte)["P_sub"]
    held = metrics(yte, P_test, leaves)

    P_train = predict_unified(bundle, Xtr)["P_sub"]
    resub = metrics(ytr, P_train, leaves)

    out = {
        "description": "Held-out test metrics for the UNIFIED HMoE architecture "
                       "(raw shallow + SCT deep), retrained on the canonical seed-42 "
                       "train split (21,977 cells) and evaluated on the held-out 2,442. "
                       "The PUBLISHED v4_improved_unified was trained on all cells "
                       "(resubstitution only); this reproduces its training recipe on "
                       "the train split for an independent held-out estimate.",
        "model": "v4_improved_unified (architecture; held-out retrain)",
        "split": {"seed": SEED, "test_size": TEST_SIZE, "n_train": int(len(train_idx)),
                  "n_test": int(len(test_idx)), "excluded_leaves": sorted(EXCL),
                  "matches_canonical_split": True},
        "unified_heldout_test": held,
        "unified_train_resubstitution": resub,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    def row(name, m):
        return (f"  {name:34s} sub={m['accuracy']*100:5.2f}%  fam={m['family_accuracy']*100:5.2f}%  "
                f"bal={m['balanced_accuracy']*100:5.2f}%  F1={m['macro_f1']:.3f}  "
                f"AUC={m['macro_roc_auc']:.4f}  conf={m['mean_confidence']:.3f}")
    print("\n" + "=" * 70)
    print("RESULTS (n_test = 2,442 held-out)")
    print("=" * 70)
    print(row("UNIFIED  held-out test", held))
    print(row("UNIFIED  train resubstitution", resub))
    print(f"\n  unified held-out: {held['n_correct']}/{held['n']} subtype correct")
    print(f"\nSaved -> {OUT_JSON}")

    shutil.rmtree(RAW_TMP, ignore_errors=True)
    shutil.rmtree(SCT_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
