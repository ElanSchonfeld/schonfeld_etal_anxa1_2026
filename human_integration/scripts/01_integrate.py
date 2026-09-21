"""Step 01 - scCRAFT integration of the Kamath + Siletti human DA atlas."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, scanpy as sc, torch, anndata as ad
from sklearn.decomposition import PCA
from scCRAFT.model import (multi_resolution_cluster, train_integration_model,
                           obtain_embeddings)

SEED = 42
HERE = Path(__file__).resolve().parent.parent
DATA = HERE/"data"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    a = sc.read_h5ad(DATA/"combined_raw.h5ad")
    a.X = a.layers["counts"].copy()
    a.obs["batch"] = a.obs["study"].astype(str)
    sc.pp.filter_genes(a, min_cells=5)

    vn = a.var_names.astype(str)
    tech = np.array([g.upper().startswith(("MT-", "MT."))
                     or (g.upper().startswith(("RPS", "RPL"))
                         and not g.upper().startswith(("RPS6K", "RPSA"))) for g in vn])
    print(f"barring {int(tech.sum())} mito/ribosomal genes from HVG pool (kept in object)")

    sc.pp.normalize_per_cell(a, counts_per_cell_after=1e4)
    sc.pp.log1p(a)
    sub = a[:, ~tech].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=2000, batch_key="batch")
    ah = a[:, sub.var_names[sub.var["highly_variable"]].tolist()].copy()
    multi_resolution_cluster(ah, resolution1=0.5, method="Leiden")
    VAE = train_integration_model(ah, batch_key="batch", epochs=150)
    obtain_embeddings(ah, VAE, dim=50, pca=False, seed=SEED)

    Z = np.asarray(ah.obsm["X_scCRAFT"], np.float32)
    pca = PCA(n_components=min(256, Z.shape[1]), random_state=SEED).fit(Z)
    out = ad.AnnData(X=np.zeros((a.n_obs, 1), np.float32), obs=a.obs.copy())
    out.obsm["latent256"] = Z
    out.obsm["X_scCRAFT"] = pca.transform(Z)[:, :50].astype(np.float32)
    out.uns["pca_var_ratio"] = pca.explained_variance_ratio_.astype(np.float32)
    out.write_h5ad(DATA/"sccraft_s42.h5ad")
    print(f"DONE -> {DATA/'sccraft_s42.h5ad'}  "
          f"latent {Z.shape}, 50 PCs = {pca.explained_variance_ratio_[:50].sum():.3f} var")


if __name__ == "__main__":
    main()
