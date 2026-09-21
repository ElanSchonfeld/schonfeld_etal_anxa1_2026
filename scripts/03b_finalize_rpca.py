"""Phase 3b (finalize): attach the Seurat RPCA embedding to the combined AnnData, compute joint UMAP (2D + 3D) and Leiden, save standardized output."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DOPABASE_DATA_DIR", ROOT / "human_integration" / "data"))
WORK = Path(os.environ.get("DOPABASE_WORK_DIR", ROOT / "human_integration" / "work"))
IN = DATA / "combined_raw.h5ad"
RPCA_OUT = WORK / "rpca_out"
OUT = DATA / "rpca_integrated.h5ad"


def main():
    a = sc.read_h5ad(IN)
    emb = pd.read_csv(RPCA_OUT / "rpca_embedding.csv", index_col=0)
    emb = emb.loc[a.obs_names]
    a.obsm["X_rpca"] = emb.values.astype(np.float32)
    print("RPCA embedding:", a.obsm["X_rpca"].shape)

    clu = RPCA_OUT / "rpca_clusters.csv"
    if clu.exists():
        c = pd.read_csv(clu).set_index("cell").loc[a.obs_names]
        a.obs["leiden_rpca"] = pd.Categorical(c["leiden_rpca"].astype(str))

    sc.pp.neighbors(a, use_rep="X_rpca", random_state=SEED)
    if "leiden_rpca" not in a.obs:
        sc.tl.leiden(a, resolution=1.0, random_state=SEED, key_added="leiden_rpca")
    sc.tl.umap(a, random_state=SEED)
    a.obsm["X_umap_2d"] = a.obsm["X_umap"].copy()
    sc.tl.umap(a, n_components=3, random_state=SEED)
    a.obsm["X_umap_3d"] = a.obsm["X_umap"].copy()
    a.obsm["X_umap"] = a.obsm["X_umap_2d"]

    a.write_h5ad(OUT)
    print("DONE ->", OUT)
    print(a.obs.groupby("leiden_rpca")["study"].value_counts().unstack().fillna(0).astype(int))


if __name__ == "__main__":
    main()
