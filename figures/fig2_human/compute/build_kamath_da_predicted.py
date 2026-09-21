#!/usr/bin/env python3
"""Build the HMoE-annotated Kamath AnnData read by the Figure 2 and Figure 3 compute scripts.

Run: python build_kamath_da_predicted.py
"""
import sys
from pathlib import Path
from os import environ
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
KAMATH = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath"
H5AD = KAMATH / "Human_Kamath_Dopamine.h5ad"
OUT = KAMATH / "Kamath_DA_predicted.h5ad"

from hmoe_annotate import hmoe_model as hm
from awatramani_lab.moe.v4.analysis import compute_prediction_uncertainty

bundle = hm.load_model(REPO_ROOT / "hmoe_annotate" / "model")
subtypes = list(bundle["leaves"])

ad = sc.read_h5ad(H5AD)
X, n_matched, _ = hm.align_features(ad.var_names, ad.X, bundle["feature_names"])
assert n_matched > 0
P_sub = hm.predict_proba(bundle, X)
del X

pred_subtype = np.array([subtypes[j] for j in P_sub.argmax(axis=1)])
pred_family = np.array([hm.infer_family(s) for s in pred_subtype])
gad2_excluded = pred_family == "Gad2"
gad2_leaves = [i for i, leaf in enumerate(subtypes) if "Gad2" in leaf]
for i in np.where(gad2_excluded)[0]:
    p = P_sub[i].copy()
    p[gad2_leaves] = 0
    pred_subtype[i] = subtypes[int(p.argmax())]
    pred_family[i] = hm.infer_family(pred_subtype[i])

sorted_probs = np.sort(P_sub, axis=1)
uncertain = compute_prediction_uncertainty(P_sub, method="confidence_gmm")["uncertain_mask"]

ad.obs["pred_subtype"] = pred_subtype
ad.obs["pred_family"] = pred_family
ad.obs["confidence"] = np.max(P_sub, axis=1)
ad.obs["pred_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
ad.obs["pred_confident"] = ~uncertain
ad.obs["gad2_excluded"] = gad2_excluded
ad.obsm["X_hmoe_P_sub"] = P_sub
ad.uns["hmoe_subtypes"] = list(subtypes)
ad.uns["hmoe_model"] = "v4_improved_unified"

assert len(subtypes) == 18, len(subtypes)
ad.write_h5ad(OUT)
print(f"wrote {OUT.name}: {ad.n_obs} cells x {len(subtypes)} subtypes, "
      f"{int((~uncertain).sum())} confident ({OUT.stat().st_size / 1024 ** 2:.0f} MB)")
