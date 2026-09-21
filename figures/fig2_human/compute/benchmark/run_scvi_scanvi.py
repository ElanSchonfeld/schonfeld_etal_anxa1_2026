#!/usr/bin/env python3
"""Train scVI and scANVI on mouse plus Kamath cells, score scVI KNN and scANVI family calls, and write scvi_scanvi.csv.

Run: python run_scvi_scanvi.py
"""
import benchmark_common as bc

import numpy as np
import anndata
import torch
import scvi
from sklearn.neighbors import KNeighborsClassifier

Xtr, ytr, Xte, yte, genes = bc.load_features()
n_mouse, n_human = len(Xtr), len(Xte)

adata = anndata.AnnData(X=np.vstack([Xtr, Xte]).astype(np.float32, copy=False))
adata.obs["batch"] = ["mouse"] * n_mouse + ["human"] * n_human
adata.obs["labels"] = list(ytr) + ["unknown"] * n_human
adata.obs["batch"] = adata.obs["batch"].astype("category")
adata.obs["labels"] = adata.obs["labels"].astype("category")
adata.var_names = [str(g) for g in genes]

scvi.settings.seed = bc.SEED
torch.manual_seed(bc.SEED)
np.random.seed(bc.SEED)

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
scvi_model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)
scvi_model.train(max_epochs=100, early_stopping=True, train_size=0.9,
                 enable_progress_bar=True, accelerator="cpu")

latent = scvi_model.get_latent_representation()
Z_train = np.asarray(latent[:n_mouse], dtype=np.float32)
Z_test = np.asarray(latent[n_mouse:], dtype=np.float32)
knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean", weights="distance")
knn.fit(Z_train, ytr)
P_knn = bc.restrict_to_eval(bc.reorder_proba(knn.classes_, knn.predict_proba(Z_test)))
rows = [{"Method": "scVI + KNN (transductive)", **bc.family_metrics(yte, P_knn)}]

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
scanvi = scvi.model.SCANVI.from_scvi_model(
    scvi_model, unlabeled_category="unknown", labels_key="labels", adata=adata)
scanvi.train(max_epochs=50, enable_progress_bar=True, accelerator="cpu")

probs = scanvi.predict(soft=True)
P_all = probs.values[n_mouse:]
columns = list(probs.columns)
P = np.zeros((n_human, len(bc.EVAL_FAMILIES)), dtype=np.float32)
for i, fam in enumerate(bc.EVAL_FAMILIES):
    P[:, i] = P_all[:, columns.index(fam)]
P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
rows.append({"Method": "scANVI (transductive)", **bc.family_metrics(yte, P)})

bc.write_results("scvi_scanvi.csv", rows)
