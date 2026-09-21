#!/usr/bin/env python3
"""Select the LIANA consensus ligand-receptor pairs measurable and variable in a prepared sender and receiver set, written as work/<name>/lr_pairs_filtered.csv.

Run: python select_lr_pairs.py human
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import work_dir

MIN_CV = 0.3


def liana_consensus():
    from liana.resource import select_resource
    lr = select_resource("consensus").rename(columns={"ligand": "ligand_gene", "receptor": "receptor_gene"})
    lr["source"] = "liana_consensus"
    lr = lr[["ligand_gene", "receptor_gene", "source"]].drop_duplicates()
    return lr.drop_duplicates(subset=["ligand_gene", "receptor_gene"])


def log_cv(values):
    vals = np.log1p(values)
    mean = vals.mean()
    return vals.std() / mean if mean > 1e-6 else 0.0


def main(name):
    d = work_dir(name)
    sender_df = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    sender_expr = sender_df.values
    sender_idx = {g: i for i, g in enumerate(sender_df.columns)}
    recv_expr = pd.read_csv(d / "receivers_pseudobulk.csv").values.astype(np.float32)
    recv_genes = json.loads((d / "receiver_genes.json").read_text())
    recv_idx = {g: i for i, g in enumerate(recv_genes)}
    bin_meta = pd.read_csv(d / "bin_metadata.csv")

    lr = liana_consensus()
    lr = lr[lr["ligand_gene"].isin(set(sender_df.columns)) & lr["receptor_gene"].isin(set(recv_genes))].copy()
    lr["ligand_cv"] = [log_cv(sender_expr[:, sender_idx[g]]) for g in lr["ligand_gene"]]
    lr = lr[lr["ligand_cv"] >= MIN_CV].copy()

    regions = sorted(bin_meta["region"].unique())
    region_means = np.zeros((len(regions), recv_expr.shape[1]), dtype=np.float64)
    for j, reg in enumerate(regions):
        region_means[j] = recv_expr[(bin_meta["region"] == reg).values].mean(axis=0)
    lr["receptor_cv"] = [log_cv(region_means[:, recv_idx[g]]) for g in lr["receptor_gene"]]
    lr = lr[lr["receptor_cv"] >= MIN_CV].copy()
    lr.to_csv(d / "lr_pairs_filtered.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
