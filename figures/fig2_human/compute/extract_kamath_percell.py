#!/usr/bin/env python3
"""Extract Kamath per-cell metadata for Figure 2.

Run: python extract_kamath_percell.py
"""
from pathlib import Path
from os import environ
import scanpy as sc
import pandas as pd

MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
H5AD = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath" / "Kamath_DA_predicted.h5ad"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"

ad = sc.read_h5ad(H5AD)
assert ad.uns.get("hmoe_model") == "v4_improved_unified", ad.uns.get("hmoe_model")

df = pd.DataFrame({
    "obs_name": ad.obs_names,
    "umap_x": ad.obsm["X_umap"][:, 0],
    "umap_y": ad.obsm["X_umap"][:, 1],
    "pred_subtype": ad.obs["pred_subtype"].astype(str).values,
    "pred_family": ad.obs["pred_family"].astype(str).values,
    "confidence": ad.obs["confidence"].astype(float).values,
    "pred_margin": ad.obs["pred_margin"].astype(float).values,
    "pred_confident": ad.obs["pred_confident"].astype(bool).values,
    "gad2_excluded": ad.obs["gad2_excluded"].astype(bool).values,
    "Cell_Type": ad.obs["Cell_Type"].astype(str).values,
})
df.to_parquet(FROZEN / "kamath_percell.parquet", index=False)
print(f"wrote kamath_percell.parquet: {df.shape[0]} cells x {df.shape[1]} cols "
      f"({(FROZEN / 'kamath_percell.parquet').stat().st_size / 1024:.0f} KB)")
print("  model:", ad.uns["hmoe_model"], "| pred_subtypes:", df.pred_subtype.nunique(),
      "| confident:", int(df.pred_confident.sum()))
