"""Step 09 - Finalize the human DA atlas: family, per-cluster stability, tiered hierarchy, and count-split (double-dip-free) markers."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED, NN, OVER_RES, NBOOT = 42, 15, 1.2, 30
STAB_ROBUST = 0.60
HERE = Path(__file__).resolve().parent.parent
DATA, FIG, RESULTS = HERE/"data", HERE/"figures", HERE/"results"
SKIP = {"GAPDH","TUBB2A","TUBA1A","TUBB","HSP90AB1","HSP90AA1","STMN2","UCHL1","SELENOW","SRP14",
        "UBB","UBC","ATP1B1","DDX24","PEBP1","ALDOA","CALM1","CALM2","MALAT1","NEAT1","ACTB",
        "ACTG1","FTH1","FTL","CLU","GNAS","SNCG","MEG3","CALY","FBXO2","TUBA1B","TUBB4B","MT3"}
def is_skip(g):
    g = g.upper(); return g in SKIP or g.startswith(("MT-","MT.","RPS","RPL","RNU","MIR","LINC","SNORD","RNR"))
np.random.seed(SEED)


def gmm_hi(x, n=3, seed=SEED):
    g = GaussianMixture(n, random_state=seed, n_init=5).fit(x.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel())); m = g.predict(x.reshape(-1, 1)) == hi
    return m, (x[m].min() if m.any() else np.nan)


def leiden(Xi, res, seed):
    a = sc.AnnData(np.zeros((Xi.shape[0], 1), np.float32)); a.obsm["E"] = Xi
    sc.pp.neighbors(a, use_rep="E", n_neighbors=NN, random_state=seed)
    sc.tl.leiden(a, resolution=res, random_state=seed, key_added="L", flavor="igraph", n_iterations=2, directed=False)
    return a.obs["L"].astype(int).values


def main():
    emb = sc.read_h5ad(DATA/"sccraft_s42.h5ad")
    X = np.ascontiguousarray(np.asarray(emb.obsm["X_scCRAFT"], np.float32)[:, :50])
    cl = pd.read_csv(RESULTS/"clusters.csv").set_index("barcode").reindex(emb.obs_names)
    lab = cl["leiden"].astype(int).values; clusters = sorted(set(lab))
    raw = sc.read_h5ad(DATA/"combined_raw.h5ad"); raw = raw[emb.obs_names].copy()
    study = raw.obs["study"].astype(str).values
    C = (raw.layers["counts"].tocsr() if sp.issparse(raw.layers["counts"]) else sp.csr_matrix(raw.layers["counts"]))
    ln = raw.copy(); ln.X = C.copy(); sc.pp.normalize_total(ln, target_sum=1e4); sc.pp.log1p(ln)
    def g(sym): x = ln[:, sym].X; return (x.toarray() if sp.issparse(x) else np.asarray(x)).ravel()

    S, Ca, G = g("SOX6"), g("CALB1"), g("GAD2")
    is_gad, thr_gad = gmm_hi(G, 3)
    zS = (S - S.mean())/S.std(); zC = (Ca - Ca.mean())/Ca.std()
    cellfam = np.where(is_gad, "Gad2", np.where(zS >= zC, "Sox6", "Calb1"))
    famof = {}
    for c in clusters:
        m = lab == c
        famof[c] = "Gad2" if pd.Series(cellfam[m]).value_counts().index[0] == "Gad2" \
            else ("Sox6" if S[m].mean() >= Ca[m].mean() else "Calb1")
    print("cell family:", dict(pd.Series(cellfam).value_counts()),
          "| cluster families:", dict(pd.Series([famof[c] for c in clusters]).value_counts()))

    rng = np.random.RandomState(SEED); n = X.shape[0]
    jac = {c: [] for c in clusters}
    for b in range(NBOOT):
        idx = np.sort(rng.choice(n, int(0.8*n), replace=False)); lb = leiden(X[idx], OVER_RES, SEED+b)
        sub_full = lab[idx]
        for c in clusters:
            a = sub_full == c
            if a.sum() == 0: jac[c].append(0.0); continue
            best = max((((a) & (lb == bc)).sum() / max(((a) | (lb == bc)).sum(), 1) for bc in set(lb)), default=0)
            jac[c].append(best)
    stability = {c: float(np.mean(jac[c])) for c in clusters}
    print("per-cluster stability (Jaccard):", {c: round(stability[c], 2) for c in clusters})

    cents_all = np.vstack([X[lab == c].mean(0) for c in clusters])
    Zg = linkage(cents_all, method="ward")
    shared_cut = float(Zg[np.argmax(np.diff(Zg[:, 2])), 2])
    tier1_of, gid = {}, 0
    for fam in ["Sox6", "Calb1", "Gad2"]:
        fcl = sorted(c for c in clusters if famof[c] == fam)
        if len(fcl) == 1:
            gid += 1; tier1_of[fcl[0]] = gid; continue
        Zf = linkage(np.vstack([X[lab == c].mean(0) for c in fcl]), method="ward")
        sub = fcluster(Zf, t=shared_cut, criterion="distance")
        for s in sorted(set(sub)):
            gid += 1
            for i, c in enumerate(fcl):
                if sub[i] == s: tier1_of[c] = gid
    n_tier1 = len(set(tier1_of.values()))
    print(f"tiers: Tier0 families={len(set(famof.values()))}, Tier1(within-family @ Ward {shared_cut:.1f})={n_tier1}, Tier2 fine={len(clusters)}")

    d = C.data.astype(np.int64); tr = rng.binomial(d, 0.5)
    Xte = sp.csr_matrix((d - tr, C.indices, C.indptr), shape=C.shape)
    te = sc.AnnData(Xte, var=pd.DataFrame(index=raw.var_names)); sc.pp.normalize_total(te, target_sum=1e4); sc.pp.log1p(te)
    te.obs["leiden"] = pd.Categorical(lab.astype(str))
    name = {}
    for fam in ["Sox6", "Calb1", "Gad2"]:
        cls = [c for c in clusters if famof[c] == fam]
        if not cls: continue
        used = set()
        def pick(top_list):
            for gn in top_list:
                if not is_skip(gn) and gn not in used: used.add(gn); return gn
            return "NA"
        if len(cls) == 1:
            sub = te.copy(); sub.obs["grp"] = np.where(lab == cls[0], str(cls[0]), "rest")
            sc.tl.rank_genes_groups(sub, "grp", groups=[str(cls[0])], method="wilcoxon")
            name[cls[0]] = f"{fam}_{pick(list(sub.uns['rank_genes_groups']['names'][str(cls[0])]))}"; continue
        sub = te[np.isin(lab, cls)].copy()
        sc.tl.rank_genes_groups(sub, "leiden", groups=[str(c) for c in cls], method="wilcoxon")
        nm = sub.uns["rank_genes_groups"]["names"]
        for c in sorted(cls, key=lambda c: -(lab == c).sum()):
            name[c] = f"{fam}_{pick(list(nm[str(c)]))}"

    rows = []
    for c in clusters:
        m = lab == c
        rows.append({"leiden": c, "name": name[c], "n": int(m.sum()), "family": famof[c],
                     "tier1": tier1_of[c], "stability": round(stability[c], 3),
                     "robust": stability[c] >= STAB_ROBUST,
                     "pct_kamath": round(100*(study[m] == "Kamath").mean(), 1),
                     "pct_siletti": round(100*(study[m] == "Siletti").mean(), 1)})
    out = pd.DataFrame(rows).sort_values(["tier1", "family", "leiden"])
    out.to_csv(RESULTS/"atlas_final.csv", index=False)
    pd.DataFrame([{"leiden": c, "family": famof[c], "name": name[c]} for c in clusters]).to_csv(
        RESULTS/"cluster_names.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nrobust (>= {STAB_ROBUST}): {int(out['robust'].sum())}/{len(out)} fine clusters")

    emb.obs["leiden"] = pd.Categorical(lab.astype(str))
    emb.obs["family"] = [famof[c] for c in lab]
    emb.obs["name"] = [name[c] for c in lab]; emb.obs["tier1"] = [tier1_of[c] for c in lab]
    emb.obs["cellfamily"] = cellfam
    emb.obs["stability"] = [stability[c] for c in lab]
    emb.obsm["X_emb50"] = X
    sc.pp.neighbors(emb, use_rep="X_emb50", n_neighbors=NN, random_state=SEED)
    sc.tl.paga(emb, groups="leiden"); sc.pl.paga(emb, plot=False)
    sc.tl.umap(emb, init_pos="paga", random_state=SEED, min_dist=0.5)
    print("canonical PAGA-init UMAP -> obsm['X_umap']")
    emb.write_h5ad(DATA/"human_da_atlas_annot.h5ad")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    o = out.sort_values("stability")
    cols = ["#2f8a62" if r else "#b0463c" for r in o["robust"]]
    ax.barh(range(len(o)), o["stability"], color=cols)
    ax.axvline(STAB_ROBUST, color="#555", ls="--", lw=1.2)
    ax.set_yticks(range(len(o))); ax.set_yticklabels(o["name"], fontsize=8)
    ax.set(xlabel="bootstrap stability (mean best-match Jaccard)", xlim=(0, 1),
           title=f"Per-cluster stability (clusterboot, {NBOOT} x 80% subsample) - green robust >= {STAB_ROBUST}")
    fig.tight_layout(); fig.savefig(FIG/"atlas_stability.png", dpi=150); plt.close(fig)
    print(f"DONE -> atlas_final.csv, human_da_atlas_annot.h5ad, atlas_stability.png")


if __name__ == "__main__":
    main()
