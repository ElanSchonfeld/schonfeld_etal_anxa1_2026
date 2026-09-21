#!/usr/bin/env python3
"""5-fold stratified cross-validation of the UNIFIED HMoE on mouse LRRK2."""

import sys, json, gc, shutil
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrain_hmoe_sct_hybrid import (
    sct_transform, predict_unified, DATA_DIR, ARTIFACT_ROOT, MODEL_TEMPLATE,
)
from eval_unified_heldout_mouse import build_bundle, metrics
from awatramani_lab.moe.v4.pipelines import train_v4_improved

SEED = 42
N_SPLITS = 5
RAW_TMP = ARTIFACT_ROOT / "work" / "cv_raw"
SCT_TMP = ARTIFACT_ROOT / "work" / "cv_sct"
OUT_JSON = ARTIFACT_ROOT / "cv5_metrics_mouse.json"


def main():
    print("=" * 70)
    print("5-FOLD CV: UNIFIED HMoE (raw shallow + SCT deep), mouse LRRK2")
    print("=" * 70)

    meta = json.loads(MODEL_TEMPLATE.read_text())
    EXCL = set(meta.get("excluded_leaves", []))

    mouse = ad.read_h5ad(DATA_DIR / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad")
    y = mouse.obs["subtype"].values.astype(str)
    gene_names = list(mouse.var_names)
    X = mouse.X.toarray() if sp.issparse(mouse.X) else np.asarray(mouse.X)
    X = np.asarray(X, dtype=np.float32)
    del mouse; gc.collect()

    valid = np.where(~np.isin(y, list(EXCL)))[0]
    Xv, yv = X[valid].copy(), y[valid]
    del X; gc.collect()
    print(f"In-model cells: {len(yv)} (excluded {sorted(EXCL)})")

    common = dict(gene_names=gene_names, model_type="logreg", feature_method="f_classif",
                  topk=200, min_pct=0.005, min_diff_pct=0.001, height_train=8,
                  shared_features=False, seed=SEED,
                  excluded_leaves=meta.get("excluded_leaves", []), root_key=meta["root_key"])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = []
    leaves_ref = None
    oof_P = None

    for fi, (tr, te) in enumerate(skf.split(np.arange(len(yv)), yv), 1):
        print(f"\n=== Fold {fi}/{N_SPLITS}: train={len(tr)}  test={len(te)} ===", flush=True)
        for d in (RAW_TMP, SCT_TMP):
            shutil.rmtree(d, ignore_errors=True)
        train_v4_improved(Xv[tr], yv[tr], meta["tree"], out_dir=str(RAW_TMP),
                          model_hyperparams={"C": 1.0}, **common)
        Xsct = sct_transform(Xv[tr])
        train_v4_improved(Xsct, yv[tr], meta["tree"], out_dir=str(SCT_TMP),
                          model_hyperparams={"C": 0.1}, **common)
        del Xsct; gc.collect()

        bundle = build_bundle(meta, RAW_TMP, SCT_TMP)
        leaves = bundle["leaves"]
        if leaves_ref is None:
            leaves_ref = leaves
            oof_P = np.zeros((len(yv), len(leaves)), dtype=float)
        else:
            assert leaves == leaves_ref, "leaf order changed across folds!"
        P = predict_unified(bundle, Xv[te])["P_sub"]
        oof_P[te] = P
        m = metrics(yv[te], P, leaves)
        fold_metrics.append(m)
        print(f"  fold sub={m['accuracy']*100:.2f}%  fam={m['family_accuracy']*100:.2f}%  "
              f"bal={m['balanced_accuracy']*100:.2f}%  F1={m['f1_macro']:.3f}  "
              f"AUC={m['roc_auc_macro']:.4f}  PR-AUC={m['pr_auc_macro']:.4f}", flush=True)
        del bundle, P; gc.collect()

    scalar_keys = [k for k, v in fold_metrics[0].items()
                   if isinstance(v, (int, float)) and k not in ("n", "n_correct")]
    agg = {k: {"mean": float(np.mean([fm[k] for fm in fold_metrics])),
               "std": float(np.std([fm[k] for fm in fold_metrics]))} for k in scalar_keys}

    pc_keys = list(fold_metrics[0]["per_class"][leaves_ref[0]].keys())
    per_class_agg = {leaf: {pk: {
        "mean": float(np.mean([fm["per_class"][leaf][pk] for fm in fold_metrics])),
        "std": float(np.std([fm["per_class"][leaf][pk] for fm in fold_metrics])),
    } for pk in pc_keys} for leaf in leaves_ref}

    pooled = metrics(yv, oof_P, leaves_ref)
    pooled_per_class = pooled.pop("per_class")

    oof_pred = np.array([leaves_ref[i] for i in oof_P.argmax(1)])
    np.savez(OUT_JSON.with_name("cv5_oof_predictions.npz"),
             gt=yv.astype(str), pred=oof_pred.astype(str), leaves=np.array(leaves_ref))

    out = {
        "description": "5-fold stratified CV (seed 42) of the unified HMoE architecture "
                       "(raw shallow + SCT deep) on mouse LRRK2 in-model cells. Each fold "
                       "trains on 4/5 and evaluates on the held-out 1/5; every cell predicted "
                       "once out-of-fold. Same recipe as the deployed v4_improved_unified. "
                       "All rate metrics are one-vs-rest per leaf, macro or support-weighted.",
        "n_splits": N_SPLITS, "seed": SEED, "n_cells": int(len(yv)),
        "excluded_leaves": sorted(EXCL),
        "mean_std_across_folds": agg,
        "per_fold": fold_metrics,
        "per_class_mean_std": per_class_agg,
        "pooled_out_of_fold": pooled,
        "pooled_out_of_fold_per_class": pooled_per_class,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    def line(label, key, pct=True, prec=2):
        m, s = agg[key]["mean"], agg[key]["std"]
        if pct:
            return f"  {label:24s} {m*100:6.2f}% ± {s*100:.2f}%"
        return f"  {label:24s} {m:7.4f} ± {s:.4f}"

    print("\n" + "=" * 70)
    print(f"5-FOLD CV RESULTS  (mean ± std across folds; n={len(yv)} in-model cells)")
    print("=" * 70)
    print(line("Accuracy (subtype)", "accuracy"))
    print(line("Balanced accuracy", "balanced_accuracy"))
    print(line("Family accuracy", "family_accuracy"))
    print(line("Precision macro (PPV)", "precision_macro"))
    print(line("Recall macro (sens)", "recall_macro"))
    print(line("Specificity macro", "specificity_macro"))
    print(line("NPV macro", "npv_macro"))
    print(line("F1 macro", "f1_macro"))
    print(line("Precision weighted", "precision_weighted"))
    print(line("Recall weighted", "recall_weighted"))
    print(line("F1 weighted", "f1_weighted"))
    print(line("ROC-AUC macro", "roc_auc_macro", pct=False))
    print(line("ROC-AUC weighted", "roc_auc_weighted", pct=False))
    print(line("PR-AUC macro", "pr_auc_macro", pct=False))
    print(line("PR-AUC weighted", "pr_auc_weighted", pct=False))
    print(line("Cohen kappa", "cohen_kappa", pct=False))
    print(line("MCC", "mcc", pct=False))
    print(line("Mean confidence", "mean_confidence"))
    print(f"\n  -- pooled out-of-fold (each cell predicted once) --")
    print(f"  Subtype accuracy : {pooled['accuracy']*100:.2f}%  ({pooled['n_correct']}/{pooled['n']})")
    print(f"  Family accuracy  : {pooled['family_accuracy']*100:.2f}%")
    print(f"  F1 macro / AUC   : {pooled['f1_macro']:.3f} / {pooled['roc_auc_macro']:.4f}")
    print(f"\nSaved -> {OUT_JSON}")

    for d in (RAW_TMP, SCT_TMP):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
