#!/usr/bin/env python3
"""Transfer mouse Sox6 and Calb1 subtype labels to Kamath cells with scANVI and freeze the predictions and posteriors.

Run: python build_scanvi_subtype_transfer.py
"""
from pathlib import Path
from os import environ
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import scvi

MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
MOUSE_H5AD = MANUSCRIPT_ROOT / "Data" / "Mouse" / "Mouse_LRRK2_Dopamine.h5ad"
KAMATH_H5AD = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath" / "Kamath_DA_predicted.h5ad"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"

N_FEATURES = 2000
SEED = 42

adm = sc.read_h5ad(MOUSE_H5AD)
adm.var_names = adm.var_names.str.title()
adh = sc.read_h5ad(KAMATH_H5AD)
adh.var_names = adh.var_names.str.title()

shared = sorted(set(adm.var_names) & set(adh.var_names))
Xs = adm[:, shared].X
gvar = np.asarray(Xs.power(2).mean(0)).ravel() - np.asarray(Xs.mean(0)).ravel() ** 2
ranked = sorted(((gvar[i], g) for i, g in enumerate(shared)), reverse=True)
feature_genes = sorted(g for _, g in ranked[:N_FEATURES])
assert len(feature_genes) == N_FEATURES

y_sub_all = adm.obs["subtype"].astype(str).values
keep_m = np.array([s.split(":", 1)[0] in ("Sox6", "Calb1") for s in y_sub_all])
y_mouse = y_sub_all[keep_m]

Xm = adm[keep_m, feature_genes].X.toarray().astype(np.float32)
Xh = adh[:, feature_genes].X.toarray().astype(np.float32)
n_mouse, n_human = Xm.shape[0], Xh.shape[0]
obs_names = np.asarray(adh.obs_names, dtype=object)
del adm, adh, Xs

ad_all = anndata.AnnData(X=np.vstack([Xm, Xh]))
ad_all.var_names = feature_genes
ad_all.obs["batch"] = ["mouse"] * n_mouse + ["human"] * n_human
ad_all.obs["labels"] = list(y_mouse) + ["unknown"] * n_human
ad_all.obs["batch"] = ad_all.obs["batch"].astype("category")
ad_all.obs["labels"] = ad_all.obs["labels"].astype("category")

scvi.settings.seed = SEED
torch.manual_seed(SEED)
np.random.seed(SEED)

scvi.model.SCVI.setup_anndata(ad_all, batch_key="batch")
scvi_model = scvi.model.SCVI(ad_all, n_latent=30, n_layers=2)
scvi_model.train(max_epochs=100, early_stopping=True, train_size=0.9,
                 enable_progress_bar=True, accelerator="cpu")

scanvi = scvi.model.SCANVI.from_scvi_model(
    scvi_model, unlabeled_category="unknown", labels_key="labels", adata=ad_all)
scanvi.train(max_epochs=50, enable_progress_bar=True, accelerator="cpu")

pred = np.asarray(scanvi.predict(), dtype=object)[n_mouse:]
soft = scanvi.predict(soft=True)

pred_df = pd.DataFrame({"obs_name": obs_names, "scanvi_pred_subtype": pred})
pred_df.to_csv(FROZEN / "scanvi_subtype_predictions.csv", index=False)

soft_df = pd.DataFrame(soft.values[n_mouse:], columns=[f"p_{c}" for c in soft.columns])
soft_df.insert(0, "obs_name", obs_names)
soft_df.to_csv(FROZEN / "scanvi_subtype_softprobs.csv", index=False)
print(f"wrote scanvi_subtype_predictions.csv and scanvi_subtype_softprobs.csv: "
      f"{n_human} cells x {soft.shape[1]} subtypes")
