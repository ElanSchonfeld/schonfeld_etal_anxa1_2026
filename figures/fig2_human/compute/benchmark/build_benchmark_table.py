#!/usr/bin/env python3
"""Merge the per-method benchmark scores into benchmark_fair_accuracy.csv.

Run: python build_benchmark_table.py
"""
from os import environ
from pathlib import Path
import pandas as pd

FROZEN = Path(__file__).resolve().parents[2] / "frozen"
RESULTS = Path(environ.get("BENCHMARK_RESULTS_DIR", FROZEN / "benchmark"))
METRICS = ["ROC-AUC", "PR-AUC", "Accuracy", "Balanced Acc", "Macro F1"]

SOURCES = {
    "hmoe.csv": {
        "HMoE (raw)": ("HMoE", "inductive", "HMoE (Ours)", "raw counts"),
    },
    "classical.csv": {
        "Logistic Regression": ("LogReg", "inductive", "Classical ML", "log1p(CP10k)"),
        "Linear SVM": ("Linear SVM", "inductive", "Classical ML", "log1p(CP10k)"),
        "Random Forest": ("Random Forest", "inductive", "Classical ML", "log1p(CP10k)"),
        "Decision Tree (CART)": ("CART", "inductive", "Classical ML", "log1p(CP10k)"),
        "Gaussian NB": ("Naive Bayes", "inductive", "Classical ML", "log1p(CP10k)"),
        "MLP": ("MLP", "inductive", "Classical ML", "log1p(CP10k)"),
        "KNN (k=15, PCA, transductive)": ("KNN (PCA)", "transductive", "KNN Transfer", "log1p(CP10k)"),
        "Harmony + KNN (transductive)": ("Harmony+KNN", "transductive", "KNN Transfer", "log1p(CP10k)"),
    },
    "singlecellnet.csv": {
        "SingleCellNet": ("SingleCellNet", "inductive", "Bioinformatics methods", "log1p(CP10k)"),
    },
    "scvi_scanvi.csv": {
        "scANVI (transductive)": ("scANVI", "transductive", "Bioinformatics methods", "raw counts"),
        "scVI + KNN (transductive)": ("scVI+KNN", "transductive", "Bioinformatics methods", "raw counts"),
    },
    "scgpt.csv": {
        "scGPT + LR (inductive)": ("scGPT+LR", "inductive", "Foundation Model", "binned raw counts"),
        "scGPT + KNN (transductive)": ("scGPT+KNN", "transductive", "Foundation Model", "binned raw counts"),
    },
}

rows = []
for filename, methods in SOURCES.items():
    scores = pd.read_csv(RESULTS / filename).set_index("Method")
    for key, (display, protocol, category, preprocessing) in methods.items():
        rows.append({"Method": display,
                     **{m: float(scores.loc[key, m]) for m in METRICS},
                     "protocol": protocol, "category": category,
                     "preprocessing": preprocessing})

df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
assert len(df) == 14 and df["Method"].is_unique
df.to_csv(FROZEN / "benchmark_fair_accuracy.csv", index=False)
print(df.to_string(index=False))
