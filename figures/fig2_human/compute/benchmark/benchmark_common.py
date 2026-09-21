#!/usr/bin/env python3
"""Shared data loading and family-level scoring for the Figure 2 method benchmark.

Run: imported by the run_*.py scripts in this directory
"""
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "42":
    os.environ["PYTHONHASHSEED"] = "42"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score, average_precision_score
from awatramani_lab.moe.v4.routing_spec import (
    infer_family_from_subtype,
    infer_family_from_human_celltype,
)

if "M2H_SOURCE_ROOT" not in os.environ:
    raise RuntimeError("set M2H_SOURCE_ROOT to the source workspace containing Data/")

SOURCE_ROOT = Path(os.environ["M2H_SOURCE_ROOT"])
MOUSE_H5AD = SOURCE_ROOT / "Data" / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad"
KAMATH_H5AD = SOURCE_ROOT / "Data" / "Human" / "Kamath" / "Kamath_DA_predicted.h5ad"
FROZEN = Path(__file__).resolve().parents[2] / "frozen"
RESULTS = Path(os.environ.get("BENCHMARK_RESULTS_DIR", FROZEN / "benchmark"))

FAMILIES = ["Sox6", "Calb1", "Gad2"]
EVAL_FAMILIES = ["Sox6", "Calb1"]
METRICS = ["ROC-AUC", "PR-AUC", "Accuracy", "Balanced Acc", "Macro F1"]
N_FEATURES = 2000
SEED = 42


def mouse_families(ad_mouse):
    return np.array([infer_family_from_subtype(str(s)) for s in ad_mouse.obs["subtype"].values])


def human_families(ad_human):
    return np.array([infer_family_from_human_celltype(ct) for ct in ad_human.obs["Cell_Type"]])


def load_features():
    ad_m = sc.read_h5ad(MOUSE_H5AD)
    ad_m.var_names = ad_m.var_names.str.title()
    ad_h = sc.read_h5ad(KAMATH_H5AD)
    ad_h.var_names = ad_h.var_names.str.title()

    shared = sorted(set(ad_m.var_names) & set(ad_h.var_names))
    Xs = ad_m[:, shared].X
    gene_var = np.asarray(Xs.power(2).mean(axis=0)).ravel() - np.asarray(Xs.mean(axis=0)).ravel() ** 2
    ranked = sorted(((gene_var[i], g) for i, g in enumerate(shared)), reverse=True)
    genes = sorted(g for _, g in ranked[:N_FEATURES])
    assert len(genes) == N_FEATURES

    y_m = mouse_families(ad_m)
    y_h = human_families(ad_h)
    train = np.isin(y_m, FAMILIES)
    test = np.isin(y_h, EVAL_FAMILIES)
    Xtr = ad_m[:, genes].X[train].toarray().astype(np.float32)
    Xte = ad_h[:, genes].X[test].toarray().astype(np.float32)
    return Xtr, y_m[train], Xte, y_h[test], genes


def reorder_proba(classes, proba, target_order=FAMILIES):
    classes = list(classes)
    P = np.zeros((proba.shape[0], len(target_order)), dtype=np.float32)
    for i, fam in enumerate(target_order):
        if fam in classes:
            P[:, i] = proba[:, classes.index(fam)]
    return P


def restrict_to_eval(P_full):
    idx = [FAMILIES.index(f) for f in EVAL_FAMILIES]
    P2 = P_full[:, idx].copy()
    return (P2 / np.maximum(P2.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def family_metrics(y_true, P):
    y_true = np.asarray(y_true)
    y_pred = np.array([EVAL_FAMILIES[j] for j in P.argmax(axis=1)], dtype=object)
    recalls, f1s, aucs, prs = [], [], [], []
    for i, fam in enumerate(EVAL_FAMILIES):
        tm, pm = y_true == fam, y_pred == fam
        tp, fp, fn = (tm & pm).sum(), (~tm & pm).sum(), (tm & ~pm).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(float(rec))
        f1s.append(float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
        aucs.append(float(roc_auc_score(tm.astype(int), P[:, i])))
        prs.append(float(average_precision_score(tm.astype(int), P[:, i])))
    return {
        "ROC-AUC": round(float(np.mean(aucs)), 4),
        "PR-AUC": round(float(np.mean(prs)), 4),
        "Accuracy": round(float((y_true == y_pred).mean()), 4),
        "Balanced Acc": round(float(np.mean(recalls)), 4),
        "Macro F1": round(float(np.mean(f1s)), 4),
    }


def write_results(filename, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["Method"] + METRICS]
    df.to_csv(RESULTS / filename, index=False)
    print(df.to_string(index=False))
