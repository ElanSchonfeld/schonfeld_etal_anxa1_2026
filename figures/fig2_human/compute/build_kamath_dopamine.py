#!/usr/bin/env python3
"""Build the Kamath dopamine-neuron counts AnnData read by the Figure 2 and Figure 3 compute scripts.

Run: python build_kamath_dopamine.py
"""
from pathlib import Path
from os import environ
from hashlib import sha256
from urllib.request import urlopen
import numpy as np
import pandas as pd
import scanpy as sc

MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
KAMATH = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath"
SOURCE_URL = "https://datasets.cellxgene.cziscience.com/a41c9e65-1abd-428b-aa0a-1d11474bfbe7.h5ad"
SOURCE_SHA256 = "301766de347b40f91df2cc8fb84a9a8e7a9a9ea52a2e1f350c3e3aa626d5090f"
DOWNLOAD = KAMATH / "Human_Kamath_Dopamine_Neurons.h5ad"
CELL_METADATA = KAMATH / "Human_Kamath_Cell_Metadata.tsv"
OUT = KAMATH / "Human_Kamath_Dopamine.h5ad"
CHUNK = 1 << 22


def digest(path):
    h = sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


KAMATH.mkdir(parents=True, exist_ok=True)
if not DOWNLOAD.exists():
    with urlopen(SOURCE_URL) as src, open(DOWNLOAD, "wb") as dst:
        for block in iter(lambda: src.read(CHUNK), b""):
            dst.write(block)
assert digest(DOWNLOAD) == SOURCE_SHA256, DOWNLOAD

ad = sc.read_h5ad(DOWNLOAD)
umap = np.asarray(ad.obsm["X_umap"], dtype=np.float64)

ad.obs["barcode_raw"] = ad.obs_names.astype(str)
ad.obs["barcode_norm"] = ad.obs_names.str.rsplit("-", n=1).str[0]

meta = pd.concat([
    chunk[chunk.index.isin(ad.obs_names)]
    for chunk in pd.read_csv(CELL_METADATA, sep="\t", skiprows=[1], index_col="NAME",
                             dtype=str, chunksize=250_000)
]).loc[ad.obs_names]
meta["date"] = meta["date"].astype(np.int64)
meta["Donor_Age"] = meta["Donor_Age"].astype(np.float64)
meta["Donor_PMI"] = meta["Donor_PMI"].astype(np.float64)
ad.obs = ad.obs.join(meta, rsuffix="_meta_general")

ad.obs["X"] = umap[:, 0]
ad.obs["Y"] = umap[:, 1]
ad.obs["Cell_Type"] = ad.obs["author_cell_type"].values

symbol = ad.var["feature_name"].astype(str)
ad = ad[:, ~symbol.str.startswith("ENSG").to_numpy()].copy()
ad.var_names = ad.var["feature_name"].astype(str)
ad.var.index.name = None
ad.var_names_make_unique()

ad.var["mt"] = ad.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
ad.obsm["X_umap"] = umap.astype(np.float32)

assert ad.n_obs == 22048, ad.n_obs
assert not ad.obs[["Cell_Type", "Status", "Donor_Age", "Donor_PMI"]].isna().any().any()
ad.write_h5ad(OUT, compression="gzip")
print(f"wrote {OUT.name}: {ad.n_obs} cells x {ad.n_vars} genes "
      f"({OUT.stat().st_size / 1024 ** 2:.0f} MB)")
