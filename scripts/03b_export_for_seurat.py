"""Phase 3b (export): write the combined raw-count matrix + metadata to disk for Seurat."""
import os
from pathlib import Path
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DOPABASE_DATA_DIR", ROOT / "human_integration" / "data"))
WORK = Path(os.environ.get("DOPABASE_WORK_DIR", ROOT / "human_integration" / "work"))
IN = DATA / "combined_raw.h5ad"
EXP = WORK / "seurat_export"


N_HVG = 3000


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    a = sc.read_h5ad(IN)
    a.X = a.layers["counts"].copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat_v3",
                                batch_key="study", subset=True)
    print("exporting HVG-only:", a.shape)
    X = a.layers["counts"]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    sio.mmwrite(str(EXP / "counts.mtx"), X.T.tocsr())
    (EXP / "genes.txt").write_text("\n".join(map(str, a.var_names)))
    (EXP / "cells.txt").write_text("\n".join(map(str, a.obs_names)))
    a.obs.to_csv(EXP / "metadata.csv")
    print("exported to", EXP)
    print("  counts (genes x cells):", X.T.shape)


if __name__ == "__main__":
    main()
