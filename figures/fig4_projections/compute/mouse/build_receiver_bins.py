#!/usr/bin/env python3
"""Build the mouse MERFISH receiver spatial bins and their mean expression under work/mouse/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python build_receiver_bins.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import source_root, work_dir

BINS_PER_REGION = 120
SEED = 42
AMYGDALA = {"BLAa": "BLA", "BLAp": "BLA", "BLAv": "BLA", "CEAm": "CEA", "CEAl": "CEA",
            "CEAc": "CEA", "MEA": "MEA", "LA": "LA"}


def assign_regions(obs):
    acr = obs["ccf_acronym"].values
    ml, dv, ap = obs["x"].values, obs["y"].values, obs["z"].values
    cp = acr == "CP"
    cp_terciles = np.percentile(ap[cp], [33, 67])
    cp_dv_med = np.median(dv[cp])
    acb_dv_med = np.median(dv[acr == "ACB"])
    ot_ml_med = np.median(ml[acr == "OT"])
    regions = []
    for i, a in enumerate(acr):
        if a == "CP":
            dv_label = "d" if dv[i] <= cp_dv_med else "v"
            ap_label = "r" if ap[i] < cp_terciles[0] else ("m" if ap[i] < cp_terciles[1] else "c")
            regions.append(f"CP-{ap_label}{dv_label}")
        elif a == "ACB":
            regions.append("ACB-core" if dv[i] <= acb_dv_med else "ACB-shell")
        elif a == "OT":
            regions.append("OT-med" if ml[i] <= ot_ml_med else "OT-lat")
        else:
            regions.append(AMYGDALA.get(a, a))
    return np.array(regions)


def main():
    out = work_dir("mouse")
    adata = ad.read_h5ad(source_root() / "cell2cell_final_v2/results/stage1/cache/wholebrain_target_structures.h5ad")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = np.asarray(X, dtype=np.float32)
    slab = assign_regions(adata.obs)
    acronym = adata.obs["ccf_acronym"].values
    coords = adata.obs[["x", "y", "z"]].values.astype(np.float64)
    bins, expr = [], []
    for region in sorted(set(slab)):
        mask = slab == region
        k = min(BINS_PER_REGION, int(mask.sum()))
        labels = (KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(coords[mask])
                  if k > 1 else np.zeros(int(mask.sum()), dtype=int))
        for b in range(k):
            sel = labels == b
            if sel.sum() == 0:
                continue
            centroid = coords[mask][sel].mean(axis=0)
            expr.append(np.asarray(X[mask][sel].mean(axis=0), dtype=np.float32).flatten())
            bins.append({"bin_id": len(bins), "receiver_region": region, "x_ccf": centroid[0],
                         "y_ccf": centroid[1], "z_ccf": centroid[2], "n_cells": int(sel.sum()),
                         "ccf_acronym": acronym[mask][0], "slab_region": region})
    pd.DataFrame(bins).to_csv(out / "receiver_bins_ccf.csv", index=False)
    np.savez_compressed(out / "recv_bin_expr.npz", recv_bin_expr=np.array(expr, dtype=np.float32))
    (out / "recv_bin_genes.json").write_text(json.dumps(list(adata.var_names)))


if __name__ == "__main__":
    main()
