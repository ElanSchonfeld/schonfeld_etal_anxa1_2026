"""Phase 6: slim the browser h5ad for the DopaBase Human Hugging Face Space."""
import sys
import os
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DOPABASE_DATA_DIR", ROOT / "human_integration" / "data"))
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "human_da_browser_new.h5ad"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "human_da_browser_slim.h5ad"
KEEP_UNS = {"dataset_switch", "hover_fields", "integration",
            "color_ui", "method_switch"}


def main():
    a = ad.read_h5ad(IN)
    print("in:", a.shape, "X nnz", a.X.nnz)

    X = a.X.tocsr()
    X.data = np.rint(X.data).astype(np.int32)
    a.X = X
    a.layers.clear()
    del a.obsp
    a.uns = {k: v for k, v in a.uns.items() if k in KEEP_UNS}
    a.raw = None

    a.write_h5ad(OUT, compression="gzip", compression_opts=6)
    import os
    mb = os.path.getsize(OUT) / 1e6
    print(f"out: {OUT.name}  {mb:.0f} MB  (X dtype {a.X.dtype}, layers {list(a.layers)})")
    print("uns:", list(a.uns.keys()), "obsm:", list(a.obsm.keys()))


if __name__ == "__main__":
    main()
