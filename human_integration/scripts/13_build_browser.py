"""Step 13 - Build the DopaBase Human browser object."""
import json, os, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp

SEED = 42
HERE = Path(__file__).resolve().parent.parent
DATA, RESULTS = HERE/"data", HERE/"results"
KAMATH_PRED = Path(os.environ.get(
    "DOPABASE_KAMATH_PRED_H5AD",
    HERE / "inputs" / "kamath" / "Kamath_DA_predicted.h5ad",
))
STAGE_OUT = DATA/"human_da_browser_new.h5ad"
_FAMNORM = {"SOX6": "Sox6", "CALB1": "Calb1", "GAD2": "Gad2", "LEF1": "Lef1"}


def std_subtype(s):
    s = str(s)
    if s in ("NA", "nan", "None", ""): return "NA"
    p = s.replace(":", "_").split("_", 1)
    fam = _FAMNORM.get(p[0].upper(), p[0].capitalize())
    return fam if len(p) == 1 else f"{fam}:" + "_".join(w.capitalize() for w in p[1].split("_"))


def strip(names, study):
    suf = f"-{study}"; return np.array([n[:-len(suf)] if n.endswith(suf) else n for n in names])


def _paga_layout_3d(conn, threshold=0.01):
    """3D layout of the PAGA cluster graph by classical MDS on PAGA graph distances."""
    from scipy.sparse.csgraph import shortest_path
    C = np.asarray(sp.csr_matrix(conn).todense(), float)
    C = 0.5 * (C + C.T)
    C[C < threshold] = 0.0
    n = C.shape[0]
    W = np.where(C > 0, 1.0 / np.maximum(C, 1e-12), 0.0)
    D = shortest_path(sp.csr_matrix(W), directed=False)
    ok = np.isfinite(D)
    if not ok.all():
        D = np.where(ok, D, D[ok].max() * 2.0)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:3]
    Y = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
    return Y - Y.mean(0)


def _paga_seed_3d(A, seed, min_dist=0.5, spread=1.0, n_epochs=200, jitter=0.08):
    """3D UMAP seeded from a 3D layout of the PAGA cluster graph, all three axes free."""
    from umap.umap_ import find_ab_params, make_epochs_per_sample
    from umap.layouts import optimize_layout_euclidean
    groups_key = A.uns["paga"]["groups"]
    nodes = _paga_layout_3d(A.uns["paga"]["connectivities"])
    codes = A.obs[groups_key].cat.codes.values.copy(); codes[codes < 0] = 0
    N = nodes - nodes.mean(0)
    N = N / (np.abs(N).max() + 1e-12) * 5.0
    rng = np.random.default_rng(seed)
    init = N[codes] + rng.normal(scale=jitter * float(np.abs(N).max()), size=(A.n_obs, 3))
    emb = np.ascontiguousarray(init.astype(np.float32))

    g = A.obsp["connectivities"].tocoo(); g.sum_duplicates()
    g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    g.eliminate_zeros()
    eps = make_epochs_per_sample(g.data, n_epochs)
    a, b = find_ab_params(spread, min_dist)
    rng_state = np.random.RandomState(seed).randint(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max, 3).astype(np.int64)
    Y = optimize_layout_euclidean(
        emb, emb, g.row.astype(np.int32), g.col.astype(np.int32), n_epochs, g.shape[1],
        eps, a, b, rng_state, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0,
        move_other=True, verbose=False)
    return np.asarray(Y, np.float32)


def method_layouts(latent, groups, spec2=None, spec3=None, seed=SEED):
    """Per-method spectral (2D+3D) + PAGA-init (2D+3D) UMAPs; PAGA groups = consensus clusters."""
    A = sc.AnnData(np.zeros((latent.shape[0], 1), np.float32)); A.obsm["E"] = np.asarray(latent, np.float32)
    A.obs["g"] = pd.Categorical(pd.Series(groups).astype(str).values)
    sc.pp.neighbors(A, use_rep="E", random_state=seed)
    if spec2 is None:
        sc.tl.umap(A, random_state=seed); spec2 = np.asarray(A.obsm["X_umap"], np.float32)
        sc.tl.umap(A, n_components=3, random_state=seed); spec3 = np.asarray(A.obsm["X_umap"], np.float32)
    sc.tl.paga(A, groups="g"); sc.pl.paga(A, plot=False)
    sc.tl.umap(A, init_pos="paga", random_state=seed); paga2 = np.asarray(A.obsm["X_umap"], np.float32)
    paga3 = _paga_seed_3d(A, seed)
    return spec2, spec3, paga2, paga3


def kamath_liger_hnorm(kp, k=25, value_lambda=10.0):
    """Reproduce Kamath's LIGER iNMF -> the aligned H.norm latent (n x k)."""
    import pyliger, anndata as ad, scipy.sparse as sp
    X = kp.X.tocsr() if sp.issparse(kp.X) else sp.csr_matrix(kp.X)
    donors = kp.obs["donor_id"].astype(str).values
    vc = pd.Series(donors).value_counts(); small = set(vc[vc < 50].index)
    batch = np.array(["small_pool" if d in small else d for d in donors])
    adata_list = []
    for d in np.unique(batch):
        m = batch == d
        sub = ad.AnnData(X=X[m].copy(), obs=kp.obs.loc[m, []].copy(), var=kp.var[[]].copy())
        sub.obs_names = kp.obs_names[m]; sub.var_names = kp.var_names
        sub.obs.index.name = "cell"; sub.var.index.name = "gene"
        sub.uns["sample_name"] = d; adata_list.append(sub)
    lig = pyliger.create_liger(adata_list)
    pyliger.normalize(lig); pyliger.select_genes(lig); pyliger.scale_not_center(lig)
    pyliger.optimize_ALS(lig, k=k, value_lambda=value_lambda, rand_seed=42)
    pyliger.quantile_norm(lig)
    hbybc = {}
    for a in lig.adata_list:
        for bc, row in zip(a.obs_names, np.asarray(a.obsm["H_norm"], np.float32)):
            hbybc[str(bc)] = row
    return np.vstack([hbybc[str(b)] for b in map(str, kp.obs_names)]).astype(np.float32)


def _declutter_3d(U, k=12, iters=4):
    """Relocate UMAP flyaways (isolated escapees) to the median of their nearest non-flyaway neighbours."""
    from sklearn.neighbors import NearestNeighbors
    U = np.asarray(U, np.float32).copy()
    for _ in range(iters):
        dk = NearestNeighbors(n_neighbors=7).fit(U).kneighbors(U, return_distance=True)[0][:, -1]
        med = np.median(dk); mad = np.median(np.abs(dk - med)) + 1e-9
        fly = dk > med + 5 * 1.4826 * mad
        if not fly.any():
            break
        nf = np.where(~fly)[0]; nb = NearestNeighbors(n_neighbors=k).fit(U[nf])
        for li in np.where(fly)[0]:
            _, ind = nb.kneighbors(U[li:li + 1]); U[li] = np.median(U[nf][ind[0]], axis=0)
    return U


def kamath_native(kp, seed=SEED):
    """Kamath native 3D built the USUAL way, on Kamath's LIGER latent."""
    canon = np.asarray(kp.obsm["X_umap"], np.float32)
    Hn = kamath_liger_hnorm(kp)
    A = sc.AnnData(np.zeros((Hn.shape[0], 1), np.float32)); A.obsm["H"] = Hn
    A.obs["g"] = pd.Categorical(kp.obs["pred_subtype"].astype(str).values)
    sc.pp.neighbors(A, use_rep="H", n_neighbors=30, random_state=seed)
    sc.tl.umap(A, n_components=3, min_dist=0.5, random_state=seed)
    def3 = _declutter_3d(np.asarray(A.obsm["X_umap"], np.float32))
    sc.tl.paga(A, groups="g"); sc.pl.paga(A, plot=False)
    sc.tl.umap(A, init_pos="paga", min_dist=0.5, random_state=seed)
    paga2 = np.asarray(A.obsm["X_umap"], np.float32)
    paga3 = _declutter_3d(_paga_seed_3d(A, seed))
    bc = list(map(str, kp.obs_names))
    d2 = dict(zip(bc, canon))
    return (d2, dict(zip(bc, def3)), dict(zip(bc, paga2)), dict(zip(bc, paga3)))


def siletti_native(barcodes_stripped):
    """Native Siletti-only UMAP (2D+3D spectral + PAGA) on the new 823 cells."""
    import bbknn
    s = sc.read_h5ad(DATA/"siletti_da_neurons.h5ad")
    s = s[[b for b in barcodes_stripped if b in set(map(str, s.obs_names))]].copy()
    s.var_names_make_unique()
    sc.pp.normalize_total(s, target_sum=1e4); sc.pp.log1p(s)
    sc.pp.highly_variable_genes(s, n_top_genes=2000)
    sh = s[:, s.var["highly_variable"]].copy(); sc.pp.scale(sh, max_value=10); sc.tl.pca(sh, n_comps=50)
    bbknn.bbknn(sh, batch_key="donor_id", n_pcs=30, neighbors_within_batch=8)
    sc.tl.leiden(sh, resolution=1.0, random_state=42, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(sh, random_state=42, min_dist=0.3); u2 = np.asarray(sh.obsm["X_umap"], np.float32)
    sc.tl.umap(sh, n_components=3, random_state=42, min_dist=0.3); u3 = np.asarray(sh.obsm["X_umap"], np.float32)
    sc.tl.paga(sh, groups="leiden"); sc.pl.paga(sh, plot=False)
    sc.tl.umap(sh, init_pos="paga", random_state=42, min_dist=0.3); p2 = np.asarray(sh.obsm["X_umap"], np.float32)
    p3 = _paga_seed_3d(sh, 42, min_dist=0.3)
    bc = list(map(str, s.obs_names))
    return (dict(zip(bc, u2)), dict(zip(bc, u3)), dict(zip(bc, p2)), dict(zip(bc, p3)))


def main():
    at = sc.read_h5ad(DATA/"human_da_atlas_annot.h5ad")
    raw = sc.read_h5ad(DATA/"combined_raw.h5ad"); raw = raw[at.obs_names].copy()
    rpca = sc.read_h5ad(DATA/"rpca_integrated.h5ad"); rpca = rpca[at.obs_names].copy()
    scvi = sc.read_h5ad(DATA/"scvi_integrated.h5ad"); scvi = scvi[at.obs_names].copy()
    fin = pd.read_csv(RESULTS/"atlas_final.csv").set_index("leiden")
    lab = at.obs["leiden"].astype(int).values
    names = {c: fin.loc[c, "name"] for c in sorted(set(lab))}
    consensus = np.array([names[c] for c in lab])

    C = raw.layers["counts"]; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    a = sc.AnnData(C.astype(np.float32), obs=pd.DataFrame(index=at.obs_names),
                   var=pd.DataFrame(index=raw.var_names))
    study = at.obs["study"].astype(str).values; is_kam = study == "Kamath"

    for c in ["study", "donor", "disease_status", "region", "study_src",
              "kamath_subtype", "siletti_subtype", "siletti_subcluster"]:
        a.obs[c] = at.obs[c].astype(str).values
    a.obs["family"] = pd.Categorical(at.obs["family"].astype(str).values, categories=["Sox6", "Calb1", "Gad2"])
    a.obs["consensus_subtype"] = pd.Categorical(consensus)
    a.obs["integrated_leiden"] = pd.Categorical(lab.astype(str))
    a.obs["tier1"] = pd.Categorical(at.obs["tier1"].astype(str).values)
    _t1 = {}
    for t, g in fin.reset_index().groupby("tier1"):
        rep = g.loc[g["n"].idxmax(), "name"]; _f0, _gn = rep.split("_", 1); _t1[str(t)] = f"{_f0}:{_gn}+"
    a.obs["tier1_named"] = pd.Categorical(a.obs["tier1"].astype(str).map(_t1))
    a.obs["stability"] = at.obs["stability"].astype(float).values
    a.obs["orig_subtype"] = pd.Categorical([std_subtype(s) for s in at.obs["orig_subtype"].astype(str)])
    slp = RESULTS/"siletti_labels.csv"
    if slp.exists():
        sl = pd.read_csv(slp).set_index("barcode")["siletti_subtype"]
        is_sil = study == "Siletti"
        rec = sl.reindex(a.obs_names[is_sil]).astype(str).map(
            lambda s: "Novel_DA" if s == "Novel_DA" else std_subtype(s)).values
        og = a.obs["orig_subtype"].astype(str).values; og[is_sil] = rec
        a.obs["orig_subtype"] = pd.Categorical(og)
        a.obs["siletti_subtype"] = pd.Categorical(
            np.where(is_sil, sl.reindex(a.obs_names).astype(str).values, a.obs["siletti_subtype"].astype(str)))
    a.obs["leiden_rpca"] = pd.Categorical(rpca.obs["leiden_rpca"].astype(str).values) if "leiden_rpca" in rpca.obs else "NA"
    a.obs["leiden_scvi"] = pd.Categorical(scvi.obs["leiden_scvi"].astype(str).values) if "leiden_scvi" in scvi.obs else "NA"

    kp = sc.read_h5ad(KAMATH_PRED)
    kp_sub, kp_fam, kp_conf = (kp.obs["pred_subtype"].astype(str).to_dict(),
                               kp.obs["pred_family"].astype(str).to_dict(),
                               kp.obs["confidence"].astype(float).to_dict())
    kam_bc = strip(np.array(a.obs_names), "Kamath")
    hs = np.array(["NA"]*a.n_obs, dtype=object); hf = np.array(["NA"]*a.n_obs, dtype=object)
    hc = np.full(a.n_obs, np.nan, np.float32)
    for i in np.where(is_kam)[0]:
        bc = kam_bc[i]
        if bc in kp_sub:
            hs[i] = kp_sub.get(bc, "NA"); hf[i] = kp_fam.get(bc, "NA"); hc[i] = kp_conf.get(bc, np.nan)
    shp = RESULTS/"siletti_hmoe_unified.csv"
    if shp.exists():
        sh = pd.read_csv(shp).set_index("barcode"); is_sil = study == "Siletti"
        v = sh.reindex(a.obs_names[is_sil])
        hs[is_sil] = v["hmoe_subtype"].astype(str).values
        hf[is_sil] = v["hmoe_family"].astype(str).values
        hc[is_sil] = v["hmoe_confidence"].astype(float).values
    a.obs["hmoe_subtype"] = pd.Categorical(hs); a.obs["hmoe_family"] = pd.Categorical(hf); a.obs["hmoe_confidence"] = hc
    print(f"HMoE: Kamath {int((hs[is_kam]!='NA').sum())}/{is_kam.sum()}; Siletti unified {int(shp.exists()) and int((study=='Siletti').sum())}")

    sil_bc = strip(np.array(a.obs_names), "Siletti")
    su2, su3, sp2, sp3 = siletti_native(sil_bc[~is_kam])
    def fill(dct, dim):
        M = np.full((a.n_obs, dim), np.nan, np.float32)
        for i in np.where(~is_kam)[0]:
            if sil_bc[i] in dct: M[i] = dct[sil_bc[i]]
        return M
    Xs2, Xs3, Xsp2, Xsp3 = fill(su2, 2), fill(su3, 3), fill(sp2, 2), fill(sp3, 3)
    print(f"Siletti native matched: {int((~np.isnan(Xs2[~is_kam, 0])).sum())}/{(~is_kam).sum()}")

    ku2, ku3, kp2, kp3 = kamath_native(kp)
    def _kfill(dct, dim):
        M = np.full((a.n_obs, dim), np.nan, np.float32)
        for i in np.where(is_kam)[0]:
            if kam_bc[i] in dct: M[i] = dct[kam_bc[i]]
        return M
    Xk2, Xk3, Xkp2, Xkp3 = _kfill(ku2, 2), _kfill(ku3, 3), _kfill(kp2, 2), _kfill(kp3, 3)

    X_sccraft = np.asarray(at.obsm["X_emb50"], np.float32)
    methods = {}
    sc_spec2, sc_spec3, sc_paga2, sc_paga3 = method_layouts(X_sccraft, lab)
    sc_paga2 = np.asarray(at.obsm["X_umap"], np.float32)
    methods["sccraft"] = (sc_spec2, sc_spec3, sc_paga2, sc_paga3)
    methods["rpca"] = method_layouts(rpca.obsm["X_rpca"], lab,
                                     np.asarray(rpca.obsm["X_umap_2d"], np.float32),
                                     np.asarray(rpca.obsm["X_umap_3d"], np.float32))
    methods["scvi"] = method_layouts(scvi.obsm["X_scVI"], lab,
                                     np.asarray(scvi.obsm["X_umap_2d"], np.float32),
                                     np.asarray(scvi.obsm["X_umap_3d"], np.float32))
    for m, (s2, s3, p2, p3) in methods.items():
        a.obsm[f"X_method_{m}_2d"] = s2; a.obsm[f"X_method_{m}_3d"] = s3
        a.obsm[f"X_method_{m}_paga_2d"] = p2; a.obsm[f"X_method_{m}_paga_3d"] = p3
        print(f"  method {m}: spectral + paga")

    a.obsm["X_umap_2d"] = methods["sccraft"][2]; a.obsm["X_umap_3d"] = methods["sccraft"][3]
    a.obsm["X_umap_2d_orig"] = methods["sccraft"][0]; a.obsm["X_umap_3d_orig"] = methods["sccraft"][1]
    a.obsm["X_umap"] = a.obsm["X_umap_2d"].copy()
    a.obsm["X_kamath_umap_2d"] = Xk2; a.obsm["X_kamath_umap_3d"] = Xk3
    a.obsm["X_kamath_umap_paga_2d"] = Xkp2; a.obsm["X_kamath_umap_paga_3d"] = Xkp3
    a.obsm["X_siletti_umap_2d"] = Xs2; a.obsm["X_siletti_umap_3d"] = Xs3
    a.obsm["X_siletti_umap_paga_2d"] = Xsp2; a.obsm["X_siletti_umap_paga_3d"] = Xsp3
    for _n, _u, _p in [("kamath_2d", Xk2, Xkp2), ("kamath_3d", Xk3, Xkp3),
                       ("siletti_2d", Xs2, Xsp2), ("siletti_3d", Xs3, Xsp3)]:
        _f = np.isfinite(_u).all(1) & np.isfinite(_p).all(1)
        assert _f.any() and not np.allclose(_u[_f], _p[_f]), \
            f"Default and PAGA embeddings are identical for {_n}"

    a.uns["integration"] = {"primary": "sccraft", "embedding": "X_scCRAFT"}
    a.uns["method_switch"] = json.dumps({"default": "scCRAFT", "methods": [
        {"label": "scCRAFT", "spec": "X_method_sccraft", "paga": "X_method_sccraft_paga"},
        {"label": "scVI", "spec": "X_method_scvi", "paga": "X_method_scvi_paga"},
        {"label": "RPCA", "spec": "X_method_rpca", "paga": "X_method_rpca_paga"}]})
    a.uns["dataset_switch"] = json.dumps({"col": "study", "modes": [
        {"label": "Integrated", "value": None, "emb": None, "emb_paga": None},
        {"label": "Kamath", "value": "Kamath", "emb": "X_kamath_umap", "emb_paga": "X_kamath_umap_paga"},
        {"label": "Siletti", "value": "Siletti", "emb": "X_siletti_umap", "emb_paga": "X_siletti_umap_paga"}]})
    a.uns["color_ui"] = json.dumps({"default": "Subtype", "modes": [
        {"label": "Subtype", "sources": [
            {"label": "Consensus", "col": "consensus_subtype"},
            {"label": "Ground truth", "col": "orig_subtype"},
            {"label": "HMoE", "col": "hmoe_subtype"}]},
        {"label": "Family", "col": "family"},
        {"label": "Disease", "col": "disease_status"},
        {"label": "Dataset", "col": "study"}]})
    a.uns["hover_fields"] = json.dumps([
        {"label": "Integrated subtype", "col": "consensus_subtype"},
        {"label": "HMoE subtype", "col": "hmoe_subtype"},
        {"label": "Subtype (ground truth)", "col": "orig_subtype"},
        {"label": "Dataset", "col": "study", "integrated_only": True}])

    Xc = a.X.tocsr(); Xc.data = np.rint(Xc.data).astype(np.int32); a.X = Xc
    a.write_h5ad(STAGE_OUT, compression="gzip", compression_opts=6)
    import os
    print(f"DONE -> {STAGE_OUT} ({os.path.getsize(STAGE_OUT)/1e6:.0f} MB)")
    print("obsm:", list(a.obsm.keys()))


if __name__ == "__main__":
    main()
