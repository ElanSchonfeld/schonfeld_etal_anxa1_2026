#!/usr/bin/env python3
"""Build Siletti per-cell metadata for Figure 2, Panel 9.

Run: python extract_siletti_percell.py
"""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

SEED = 42
HI = Path(__file__).resolve().parents[3] / "human_integration"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"

a = sc.read_h5ad(HI / "data" / "siletti_da_neurons.h5ad")
a.var_names_make_unique()
X0 = a.X[:50].toarray() if sp.issparse(a.X) else a.X[:50]
assert a.X.max() > 30 and np.allclose(X0, np.round(X0)), "X must be raw counts for Seurat LogNormalize"

sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)

def expr(sym):
    x = a[:, sym].X
    return (x.toarray() if sp.issparse(x) else np.asarray(x)).ravel()

markers = {f"expr_{g}": expr(g) for g in ["SOX6", "CALB1", "GAD2"]}

au = a.copy()
sc.pp.highly_variable_genes(au, n_top_genes=2000)
au = au[:, au.var.highly_variable].copy()
sc.pp.scale(au, max_value=10)
sc.pp.pca(au, n_comps=30, random_state=SEED)
sc.pp.neighbors(au, n_neighbors=15, random_state=SEED)
sc.tl.umap(au, random_state=SEED, min_dist=0.4)
U = au.obsm["X_umap"]

df = pd.DataFrame({
    "barcode": [f"{n}-Siletti" for n in a.obs_names.astype(str)],
    "UMAP1": U[:, 0], "UMAP2": U[:, 1],
    "dissection": a.obs["dissection"].astype(str).values,
    **markers,
})
uni = pd.read_csv(HI / "results" / "siletti_hmoe_unified.csv")
gt = pd.read_csv(FROZEN / "siletti_groundtruth.csv")
df = df.merge(uni, on="barcode", how="left").merge(gt, on="barcode", how="left")
assert df["hmoe_family"].notna().all() and df["siletti_subtype"].notna().all(), "join gap"

df.to_parquet(FROZEN / "siletti_percell.parquet", index=False)
print(f"wrote siletti_percell.parquet: {df.shape[0]} cells x {df.shape[1]} cols "
      f"({(FROZEN / 'siletti_percell.parquet').stat().st_size / 1024:.0f} KB)")
print("  HMoE family:", df.hmoe_family.value_counts().to_dict())
print("  GT subtypes:", df.siletti_subtype.nunique(), "| Novel_DA:", int((df.siletti_subtype == "Novel_DA").sum()))
