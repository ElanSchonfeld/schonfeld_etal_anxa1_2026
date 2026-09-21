"""Step 02 - Discovery-native clustering: over-cluster, then keep only the splits that are supported by de-novo differential expression."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp

SEED, NN, OVER_RES = 42, 15, 1.2
DE_LFC, DE_FDR = 1.0, 0.05
T_REPORT = [3, 5, 8, 10, 15, 20]
DEFAULT_T = 10
HERE = Path(__file__).resolve().parent.parent
DATA, RESULTS = HERE/"data", HERE/"results"


def main():
    emb = sc.read_h5ad(DATA/"sccraft_s42.h5ad")
    X = np.asarray(emb.obsm["X_scCRAFT"], np.float32)[:, :50]
    ad = sc.AnnData(np.zeros((X.shape[0], 1), np.float32)); ad.obsm["E"] = X
    sc.pp.neighbors(ad, use_rep="E", n_neighbors=NN, random_state=SEED)
    sc.tl.leiden(ad, resolution=OVER_RES, random_state=SEED, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    lab = ad.obs["leiden"].astype(int).values
    print(f"over-clustered at res {OVER_RES}: k0 = {len(set(lab))}")

    raw = sc.read_h5ad(DATA/"combined_raw.h5ad"); raw = raw[emb.obs_names].copy()
    raw.X = raw.layers["counts"].copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=2000); H = raw[:, raw.var.highly_variable].copy()
    E = H.X.tocsr() if sp.issparse(H.X) else sp.csr_matrix(H.X)
    Efull = sc.AnnData(E, var=pd.DataFrame(index=H.var_names))

    def distinct(a, b):
        m = np.isin(lab, [a, b]); sub = Efull[m].copy()
        sub.obs["g"] = np.where(lab[m] == a, "a", "b")
        sc.tl.rank_genes_groups(sub, "g", groups=["a"], reference="b", method="wilcoxon")
        r = sub.uns["rank_genes_groups"]; lfc = r["logfoldchanges"]["a"]; fdr = r["pvals_adj"]["a"]
        up = int(((lfc > DE_LFC) & (fdr < DE_FDR)).sum()); dn = int(((lfc < -DE_LFC) & (fdr < DE_FDR)).sum())
        return min(up, dn)

    def centroids(): return {c: X[lab == c].mean(0) for c in set(lab)}
    def nearest_pairs():
        cen = centroids(); cs = list(cen); pairs = set()
        for c in cs:
            o = min([x for x in cs if x != c], key=lambda x: np.linalg.norm(cen[c]-cen[x]))
            pairs.add(frozenset((c, o)))
        return [tuple(p) for p in pairs]

    cache = {}; trace = []
    while len(set(lab)) > 2:
        pairs = nearest_pairs()
        for p in pairs:
            key = frozenset(p)
            if key not in cache: cache[key] = distinct(*p)
        p_min = min(pairs, key=lambda p: cache[frozenset(p)]); d = cache[frozenset(p_min)]
        trace.append({"k": len(set(lab)), "merged_distinctness": d})
        if d >= max(T_REPORT): break
        a, b = p_min; lab[lab == b] = a
        cache = {k: v for k, v in cache.items() if a not in k and b not in k}
    tr = pd.DataFrame(trace); tr.to_csv(RESULTS/"merge_trace.csv", index=False)

    print("\ndistinctness bar T -> final k (each subtype >=T strong up & down markers vs nearest):")
    kT = {}
    for T in T_REPORT:
        below = tr[tr["merged_distinctness"] < T]
        kT[T] = int(below["k"].min()) - len(below) if len(below) else int(tr["k"].max())
        hit = tr[tr["merged_distinctness"] >= T]
        kT[T] = int(hit["k"].iloc[0]) if len(hit) else 2
        print(f"   T={T:2d} -> k={kT[T]}")

    lab = ad.obs["leiden"].astype(int).values; cache = {}
    while len(set(lab)) > kT[DEFAULT_T]:
        pairs = nearest_pairs()
        for p in pairs:
            key = frozenset(p)
            if key not in cache: cache[key] = distinct(*p)
        p_min = min(pairs, key=lambda p: cache[frozenset(p)])
        a, b = p_min; lab[lab == b] = a
        cache = {k: v for k, v in cache.items() if a not in k and b not in k}
    remap = {c: i for i, c in enumerate(sorted(set(lab), key=lambda c: -(lab == c).sum()))}
    lab = np.array([remap[x] for x in lab])
    print(f"\nFINAL: k={len(set(lab))} at T={DEFAULT_T}")
    print("cluster sizes:", dict(pd.Series(lab).value_counts().sort_index()))

    M = sp.csr_matrix(raw.layers["counts"]); vn = raw.var_names.astype(str)
    mt = np.array([g.upper().startswith(("MT-", "MT.")) for g in vn])
    tot = np.asarray(M.sum(1)).ravel(); ng = np.asarray((M > 0).sum(1)).ravel()
    pmt = 100*np.asarray(M[:, mt].sum(1)).ravel()/np.maximum(tot, 1)
    pd.DataFrame({"barcode": emb.obs_names, "leiden": lab, "pct_mito": pmt.round(2),
                  "n_genes": ng, "n_counts": tot}).to_csv(RESULTS/"clusters.csv", index=False)
    print(f"DONE -> {RESULTS/'clusters.csv'} (+ merge_trace.csv)")


if __name__ == "__main__":
    main()
