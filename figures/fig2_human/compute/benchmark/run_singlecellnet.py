#!/usr/bin/env python3
"""Train SingleCellNet on mouse families, score Kamath cells, and write singlecellnet.csv.

Run: python run_singlecellnet.py
"""
import benchmark_common as bc

import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata
import pySingleCellNet as pySCN

Xtr, ytr, Xte, yte, genes = bc.load_features()

random.seed(bc.SEED)
np.random.seed(bc.SEED)


def build(X, labels=None):
    var = pd.DataFrame(index=pd.Index([str(g) for g in genes], name="gene"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    if labels is not None:
        obs["celltype"] = pd.Categorical(np.asarray(labels).astype(str))
    ad = anndata.AnnData(X=sp.csr_matrix(np.asarray(X, dtype=np.float32)), obs=obs, var=var)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.var["highly_variable"] = True
    return ad


ad_train = build(Xtr, ytr)
ad_query = build(Xte)

clf = pySCN.tl.train_classifier(ad_train, groupby="celltype", n_top_genes=30,
                                n_top_gene_pairs=40, n_trees=1000)
pySCN.tl.classify_anndata(ad_query, clf, nrand=0)

score = ad_query.obsm["SCN_score"]
P2 = np.column_stack([score[fam].to_numpy() for fam in bc.EVAL_FAMILIES]).astype(np.float64)
P = (P2 / np.maximum(P2.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)

bc.write_results("singlecellnet.csv", [{"Method": "SingleCellNet", **bc.family_metrics(yte, P)}])
