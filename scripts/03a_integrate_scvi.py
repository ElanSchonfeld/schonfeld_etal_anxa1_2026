"""Phase 3a: scVI integration of the combined Kamath+Siletti DA atlas."""
import os
from pathlib import Path
import numpy as np
import scanpy as sc
import scvi

SEED = 42
N_LATENT = 30
N_HVG = 3000

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DOPABASE_DATA_DIR", ROOT / "human_integration" / "data"))
IN = DATA / "combined_raw.h5ad"
OUT = DATA / "scvi_integrated.h5ad"


def main():
    scvi.settings.seed = SEED
    sc.settings.verbosity = 1
    a = sc.read_h5ad(IN)
    a.X = a.layers["counts"].copy()
    print("loaded", a.shape)

    sc.pp.filter_genes(a, min_cells=3)
    print("after gene filter:", a.shape)

    try:
        sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat_v3",
                                    batch_key="study", subset=False)
    except Exception as e:
        print("seurat_v3 failed, falling back to lognorm seurat:", e)
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, batch_key="study")
        a.X = a.layers["counts"].copy()
    ahvg = a[:, a.var["highly_variable"]].copy()
    ahvg.X = ahvg.layers["counts"].copy()
    print("HVG subset:", ahvg.shape)

    scvi.model.SCVI.setup_anndata(ahvg, layer="counts", batch_key="study")
    model = scvi.model.SCVI(ahvg, n_latent=N_LATENT)
    model.train()
    a.obsm["X_scVI"] = model.get_latent_representation()
    print("latent:", a.obsm["X_scVI"].shape)

    sc.pp.neighbors(a, use_rep="X_scVI", random_state=SEED)
    sc.tl.leiden(a, resolution=1.0, random_state=SEED, key_added="leiden_scvi")
    sc.tl.umap(a, random_state=SEED)
    a.obsm["X_umap_2d"] = a.obsm["X_umap"].copy()
    sc.tl.umap(a, n_components=3, random_state=SEED)
    a.obsm["X_umap_3d"] = a.obsm["X_umap"].copy()
    a.obsm["X_umap"] = a.obsm["X_umap_2d"]

    a.X = a.layers["counts"].copy()
    a.write_h5ad(OUT)
    print("DONE ->", OUT)
    print(a.obs["leiden_scvi"].value_counts())
    print("clusters x study:")
    print(a.obs.groupby("leiden_scvi")["study"].value_counts().unstack().fillna(0).astype(int))


if __name__ == "__main__":
    main()
