#!/usr/bin/env python3
"""Score the HMoE subtype posterior collapsed to families on Kamath cells and write hmoe.csv.

Run: python run_hmoe.py
"""
import benchmark_common as bc

import sys
from pathlib import Path
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from hmoe_annotate import hmoe_model as hm

bundle = hm.load_model(REPO_ROOT / "hmoe_annotate" / "model")
ad_h = sc.read_h5ad(bc.KAMATH_H5AD)
y_h = bc.human_families(ad_h)
test = np.isin(y_h, bc.EVAL_FAMILIES)
X, n_matched, _ = hm.align_features(ad_h.var_names, ad_h.X, bundle["feature_names"])
assert n_matched > 0
del ad_h

P_sub = hm.predict_proba(bundle, X)
del X
leaf_family = np.array([bc.infer_family_from_subtype(s) for s in bundle["leaves"]])
P2 = np.column_stack([P_sub[:, leaf_family == fam].sum(axis=1) for fam in bc.EVAL_FAMILIES]).astype(np.float64)
P2 = (P2 / np.maximum(P2.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)

bc.write_results("hmoe.csv", [{"Method": "HMoE (raw)", **bc.family_metrics(y_h[test], P2[test])}])
