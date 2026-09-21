"""Step 00c - Recover Siletti DA-subtype labels for the 823 Siletti cells."""
import warnings; warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SEED, K, THR = 42, 15, 0.95
HERE = Path(__file__).resolve().parent.parent
DATA, RESULTS = HERE/"data", HERE/"results"
LABELED = Path(os.environ.get(
    "DOPABASE_SILETTI_REFERENCE_H5AD",
    HERE / "inputs" / "siletti" / "Siletti_DA_raw_counts.h5ad",
))
SUBMAP = {"1876": "Sox6_AGTR1", "1875": "Sox6_ASB4", "1874": "Calb1_CCKAR", "1873": "Calb1_DLK1",
          "1872": "Calb1_SCUBE1", "1871": "Calb1_CHRNB3", "1870": "Gad2_GRP", "1869": "Gad2_EBF2"}


def main():
    new = sc.read_h5ad(DATA/"siletti_da_neurons.h5ad")
    new.var_names = new.var["feature_name"].astype(str).values; new.var_names_make_unique()
    R = sc.read_h5ad(LABELED); R.var_names_make_unique()
    ylab = np.array([SUBMAP.get(s, "Siletti_other") for s in R.obs["subcluster_id"].astype(str)])

    shared = sorted(set(new.var_names) & set(R.var_names))
    nn = new[:, shared].copy(); rr = R[:, shared].copy()
    for a in (nn, rr):
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(rr, n_top_genes=2000)
    hv = rr.var_names[rr.var["highly_variable"]].tolist()
    A = nn[:, hv].X; A = A.toarray() if sp.issparse(A) else np.asarray(A)
    B = rr[:, hv].X; B = B.toarray() if sp.issparse(B) else np.asarray(B)
    mu, sd = B.mean(0), B.std(0) + 1e-8
    A = (A - mu)/sd; B = (B - mu)/sd
    pca = PCA(n_components=50, random_state=SEED).fit(B)
    Ap, Bp = pca.transform(A), pca.transform(B)

    knn = NearestNeighbors(n_neighbors=K).fit(Bp)
    dist, idx = knn.kneighbors(Ap)
    An = A/(np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B/(np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    best_corr = (An * Bn[idx[:, 0]]).sum(1)
    lab = []
    for i in range(len(Ap)):
        if best_corr[i] < THR:
            lab.append("Novel_DA"); continue
        votes = pd.Series(ylab[idx[i]]).value_counts()
        lab.append(votes.index[0])
    lab = np.array(lab)

    bc = np.array([f"{n}-Siletti" for n in new.obs_names.astype(str)])
    out = pd.DataFrame({"siletti_subtype": lab, "match_corr": np.round(best_corr, 3)}, index=bc)
    out.index.name = "barcode"
    out.to_csv(RESULTS/"siletti_labels.csv")
    print("Siletti label recovery (k=%d, corr>=%.2f):" % (K, THR))
    print(out["siletti_subtype"].value_counts().to_string())
    print(f"\nmatched {(lab != 'Novel_DA').sum()}/{len(lab)} to Siletti subtypes; "
          f"{(lab == 'Novel_DA').sum()} Novel_DA -> results/siletti_labels.csv")


if __name__ == "__main__":
    main()
