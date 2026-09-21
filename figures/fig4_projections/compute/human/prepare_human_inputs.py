#!/usr/bin/env python3
"""Build the Kamath sender pseudobulk and the HMBA-BG human region x cell-type receiver pseudobulk under work/human/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace M2H_DATA_ROOT=/path/to/atlas-data python prepare_human_inputs.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import data_root, source_root, work_dir

RECEIVER_GROUPS = [
    "STRd D1 Matrix MSN", "STRd D1 Striosome MSN", "STRd D2 Matrix MSN", "STRd D2 Striosome MSN",
    "STRd D2 StrioMat Hybrid MSN", "STRv D1 MSN", "STRv D2 MSN", "STRv D1 NUDAP MSN",
    "STR D1D2 Hybrid MSN", "OT D1 ICj",
    "STR FS PTHLH-PVALB GABA", "STR TAC3-PLPP4 GABA", "STR SST-CHODL GABA", "STR SST-ADARB2 GABA",
    "STR SST-RSPO2 GABA", "STR LYPD6-RSPO2 GABA", "STRd Cholinergic GABA", "STR Cholinergic GABA",
]
RECEIVER_REGIONS = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "NAC", "VeP"]
MIN_CELLS = 50
CHUNK_SIZE = 5000


def receiver_metadata(allen_dir):
    cell_cluster = pd.read_csv(allen_dir / "cell_to_cluster_membership.csv",
                               usecols=["cell_label", "cluster_alias"])
    annot = pd.read_csv(allen_dir / "cluster_to_cluster_annotation_membership.csv")
    group_map = annot[annot["cluster_annotation_term_set_name"] == "Group"][
        ["cluster_alias", "cluster_annotation_term_name"]
    ].rename(columns={"cluster_annotation_term_name": "group_name"})
    cell_meta = pd.read_csv(allen_dir / "cell_metadata.csv", usecols=["cell_label", "library_label"])
    lib_meta = pd.read_csv(allen_dir / "library_metadata.csv",
                           usecols=["library_label", "region_of_interest_label"]
                           ).rename(columns={"region_of_interest_label": "region"})
    meta = (cell_cluster.merge(group_map, on="cluster_alias", how="left")
            .merge(cell_meta, on="cell_label", how="left")
            .merge(lib_meta, on="library_label", how="left"))
    return meta[meta["group_name"].isin(RECEIVER_GROUPS) & meta["region"].isin(RECEIVER_REGIONS)].copy()


def receiver_bins(meta):
    bins = []
    for region in RECEIVER_REGIONS:
        for grp in RECEIVER_GROUPS:
            cells = meta.loc[(meta["region"] == region) & (meta["group_name"] == grp), "cell_label"].values
            if len(cells) >= MIN_CELLS:
                bins.append({"bin_id": len(bins), "region": region, "group_name": grp,
                             "n_cells": len(cells), "cell_labels": cells})
    return bins


def bin_pseudobulk(adata, cell_labels, bins):
    label_to_idx = {label: i for i, label in enumerate(cell_labels)}
    rows, cols = [], []
    counts = np.zeros(len(bins), dtype=np.int64)
    for i, b in enumerate(bins):
        idx = [label_to_idx[c] for c in b["cell_labels"] if c in label_to_idx]
        rows.extend(idx)
        cols.extend([i] * len(idx))
        counts[i] = len(idx)
    member = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(adata.n_obs, len(bins)))
    all_idxs = np.unique(rows)
    sums = np.zeros((len(bins), adata.n_vars), dtype=np.float64)
    for start in range(0, len(all_idxs), CHUNK_SIZE):
        chunk = all_idxs[start:start + CHUNK_SIZE]
        X = sparse.csr_matrix(adata.X[chunk]).astype(np.float64)
        sums += (member[chunk].T @ X).toarray()
    out = np.zeros(sums.shape, dtype=np.float32)
    for i in range(len(bins)):
        if counts[i] > 0:
            out[i] = (sums[i] / counts[i]).astype(np.float32)
    return out


def sender_pseudobulk(path):
    adata = ad.read_h5ad(path)
    subtypes = sorted(adata.obs["pred_subtype"].unique())
    expr = np.zeros((len(subtypes), adata.n_vars), dtype=np.float32)
    counts = {}
    for i, sub in enumerate(subtypes):
        mask = (adata.obs["pred_subtype"] == sub).values
        counts[sub] = int(mask.sum())
        X = adata.X[mask]
        expr[i] = np.asarray(X.mean(axis=0)).ravel() if sparse.issparse(X) else X.mean(axis=0)
    return pd.DataFrame(expr, index=subtypes, columns=adata.var_names.values), counts


def main():
    out = work_dir("human")
    allen_dir = data_root() / "human/atlas/allen/basal_ganglia/multiome"
    bins = receiver_bins(receiver_metadata(allen_dir))
    adata = ad.read_h5ad(allen_dir / "HMBA-10xMultiome-BG-Human-raw.h5ad", backed="r")
    recv_genes = (adata.var["gene_symbol"].values if "gene_symbol" in adata.var.columns
                  else adata.var_names.values)
    cell_labels = (adata.obs["cell_label"].values if "cell_label" in adata.obs.columns
                   else adata.obs_names.values)
    recv = bin_pseudobulk(adata, cell_labels, bins)
    senders, counts = sender_pseudobulk(source_root() / "Data/Human/Kamath/Kamath_DA_predicted.h5ad")

    senders.to_csv(out / "senders_pseudobulk.csv")
    pd.DataFrame(recv, columns=recv_genes).to_csv(out / "receivers_pseudobulk.csv", index=False)
    pd.DataFrame([{k: v for k, v in b.items() if k != "cell_labels"} for b in bins]
                 ).to_csv(out / "bin_metadata.csv", index=False)
    (out / "sender_subtype_counts.json").write_text(json.dumps(counts, indent=2))
    (out / "receiver_genes.json").write_text(json.dumps(list(recv_genes)))


if __name__ == "__main__":
    main()
