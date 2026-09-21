#!/usr/bin/env python3
"""Build the Kraft Slide-tags sender pseudobulk, the 19-donor zone x cell-type receiver pseudobulk, and the ligand-receptor table under work/kraft/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace M2H_DATA_ROOT=/path/to/atlas-data python prepare_kraft_inputs.py
"""
import glob
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import data_root, source_root, work_dir

MSN = ["STRd D1 Matrix MSN", "STRd D1 Striosome MSN", "STRd D2 Matrix MSN", "STRd D2 Striosome MSN",
       "STRd D2 StrioMat Hybrid MSN", "STRv D1 MSN", "STRv D2 MSN", "STRv D1 NUDAP MSN",
       "STR D1D2 Hybrid MSN", "STR FS PTHLH-PVALB GABA", "STR SST-CHODL GABA",
       "STR TAC3-PLPP4 GABA"]
ZONES = ["1", "2", "3", "4", "5", "6"]
CT_COL, ZONE_COL = "Group_name", "Final_Zone_Assignments_No_Smooth"
MIN_CELLS = 50
MIN_SENDER_CELLS = 30


def kraft_paths():
    root = data_root() / "human/spatial/kraft_macosko"
    return [root / "kraft_master_lr_subset.h5ad"] + sorted(
        Path(p) for p in glob.glob(str(root / "reacquire_908genes/s*.h5ad")))


def senders(path):
    a = ad.read_h5ad(path)
    a = a[a.obs["pred_confident"].astype(str) == "True"].copy()
    a = a[a.obs["gad2_excluded"].astype(str) == "False"].copy()
    a = a[a.obs["Status"].astype(str) == "Ctrl"].copy()
    counts = a.obs["pred_subtype"].value_counts()
    subtypes = sorted(s for s, n in counts.items() if n >= MIN_SENDER_CELLS and isinstance(s, str))
    genes = list(a.var_names)
    pb = np.zeros((len(subtypes), len(genes)), dtype=np.float32)
    out_counts = {}
    for i, sub in enumerate(subtypes):
        mask = (a.obs["pred_subtype"] == sub).values
        out_counts[sub] = int(mask.sum())
        X = a.X[mask]
        pb[i] = np.asarray(X.mean(axis=0)).ravel() if sparse.issparse(X) else X.mean(axis=0)
    return pd.DataFrame(pb, index=subtypes, columns=genes), out_counts


def load_receiver(path):
    a = ad.read_h5ad(path)
    keep = a.obs[CT_COL].astype(str).isin(MSN) & a.obs[ZONE_COL].astype(str).isin(ZONES)
    return a[keep.values].copy()


def submatrix(a, genes):
    vmap = {g: i for i, g in enumerate(a.var_names)}
    cols = np.array([vmap.get(g, -1) for g in genes])
    ok = np.where(cols >= 0)[0]
    Xc = (a.X.tocsc() if sparse.issparse(a.X) else sparse.csc_matrix(a.X))[:, cols[ok]].tocoo()
    return sparse.csr_matrix((Xc.data.astype(np.float32), (Xc.row, ok[Xc.col])),
                             shape=(a.n_obs, len(genes)))


def main():
    out = work_dir("kraft")
    human = work_dir("human")
    kam = source_root() / "Data/Human/Kamath/Kamath_DA_predicted.h5ad"
    sender_df, sender_counts = senders(kam)
    sender_df.to_csv(out / "senders_pseudobulk.csv")
    (out / "sender_subtype_counts.json").write_text(json.dumps(sender_counts, indent=2))

    srcs = [load_receiver(p) for p in kraft_paths()]
    canon = pd.read_csv(human / "lr_pairs_filtered.csv")
    keep = canon["ligand_gene"].isin(set(sender_df.columns)) & \
        canon["receptor_gene"].isin(set(srcs[0].var_names))
    lr = canon[keep].reset_index(drop=True)
    lr.to_csv(out / "lr_pairs_filtered.csv", index=False)
    shared = set(srcs[0].var_names)
    for a in srcs[1:]:
        shared &= set(a.var_names)
    genes = sorted(shared | set(lr["receptor_gene"]))

    parts = [(submatrix(a, genes), a.obs[ZONE_COL].astype(str).values,
              a.obs[CT_COL].astype(str).values, a.obs["donor"].astype(str).values,
              a.obs["x"].to_numpy(float), a.obs["y"].to_numpy(float)) for a in srcs]
    X = sparse.vstack([p[0] for p in parts]).tocsr()
    zone = np.concatenate([p[1] for p in parts])
    ct = np.concatenate([p[2] for p in parts])
    donor = np.concatenate([p[3] for p in parts])
    xx = np.concatenate([p[4] for p in parts])
    yy = np.concatenate([p[5] for p in parts])

    rows, bins = [], []
    for z in ZONES:
        for c in MSN:
            m = (zone == z) & (ct == c)
            if m.sum() >= MIN_CELLS:
                rows.append(np.asarray(X[m].mean(axis=0)).ravel())
                bins.append({"bin_id": len(bins), "zone": z, "celltype": c, "region": f"Z{z}",
                             "group_name": c, "n_cells": int(m.sum()),
                             "x_mean": float(xx[m].mean()), "y_mean": float(yy[m].mean())})
    pd.DataFrame(rows, columns=genes).to_csv(out / "receivers_pseudobulk.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "bin_metadata.csv", index=False)
    (out / "receiver_genes.json").write_text(json.dumps(genes))


if __name__ == "__main__":
    main()
