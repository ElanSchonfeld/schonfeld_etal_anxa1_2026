#!/usr/bin/env python3
"""Freeze the HMoE per-cell subtype posterior for Figure 2.

Run: python build_kamath_hmoe_softprobs.py
"""
from pathlib import Path
from os import environ
import numpy as np
import pandas as pd
import scanpy as sc

MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
H5AD = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath" / "Kamath_DA_predicted.h5ad"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"

ad = sc.read_h5ad(H5AD)
assert ad.uns.get("hmoe_model") == "v4_improved_unified", ad.uns.get("hmoe_model")
subtypes = [str(s) for s in ad.uns["hmoe_subtypes"]]
P = np.asarray(ad.obsm["X_hmoe_P_sub"], dtype=np.float32)

df = pd.DataFrame(P, columns=[f"p_{s}" for s in subtypes])
df.insert(0, "obs_name", ad.obs_names.values)
df.to_parquet(FROZEN / "kamath_hmoe_softprobs.parquet", index=False)
print(f"wrote kamath_hmoe_softprobs.parquet: {df.shape[0]} cells x {len(subtypes)} subtypes "
      f"({(FROZEN / 'kamath_hmoe_softprobs.parquet').stat().st_size / 1024:.0f} KB)")
