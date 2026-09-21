#!/usr/bin/env python3
"""Fit the rhesus macaque projection on the donor-stratified receiver bins and write it under work/macaque/refit/.

Run: python build_rhesus_refit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.parafac_pipeline import aggregate_to_regions, build_tensor, run_parafac_pipeline
from lib.paths import work_dir

SEED, N_FACTORS, L1_RECEIVER = 42, 20, 0.05


def main():
    d = work_dir("macaque")
    out = work_dir("macaque/refit")
    sd = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    donor = d / "donor"
    recv = pd.read_csv(donor / "receivers_pseudobulk.csv").values.astype(np.float32)
    recv_genes = json.loads((donor / "receiver_genes.json").read_text())
    bm = pd.read_csv(donor / "bin_metadata.csv")
    lr = pd.read_csv(d / "lr_pairs_filtered.csv")
    lr = lr[lr.source == "liana_consensus"][["ligand_gene", "receptor_gene"]].reset_index(drop=True)
    subs = list(sd.index)
    T = build_tensor(sd.values.astype(np.float32), list(sd.columns), recv, recv_genes, lr)
    proj, _ = run_parafac_pipeline(T, subs, n_factors=N_FACTORS, l1_receiver=L1_RECEIVER, seed=SEED)
    pd.DataFrame(proj, index=subs, columns=[f"bin_{i}" for i in range(proj.shape[1])]
                 ).to_csv(out / "projection_bin_level.tsv", sep="\t")
    region, _ = aggregate_to_regions(proj, subs, bm)
    region.to_csv(out / "projection_matrix_macaque.tsv", sep="\t")


if __name__ == "__main__":
    main()
