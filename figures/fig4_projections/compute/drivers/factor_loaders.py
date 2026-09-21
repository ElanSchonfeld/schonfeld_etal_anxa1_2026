"""PARAFAC factors and ligand-receptor pair order for each of the five projection datasets."""
import json

import numpy as np
import pandas as pd

from lib.parafac_pipeline import build_tensor, fit_factors
from lib.paths import work_dir

SEED, N_FACTORS, L1_RECEIVER = 42, 20, 0.05


def _csv_inputs(name):
    d = work_dir(name)
    sd = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    recv = pd.read_csv(d / "receivers_pseudobulk.csv").values.astype(np.float32)
    genes = json.loads((d / "receiver_genes.json").read_text())
    lr = pd.read_csv(d / "lr_pairs_filtered.csv")
    lr = lr[lr.source == "liana_consensus"][["ligand_gene", "receptor_gene"]].reset_index(drop=True)
    return sd, recv, genes, lr


def fit_from_csv(name):
    sd, recv, genes, lr = _csv_inputs(name)
    T = build_tensor(sd.values.astype(np.float32), list(sd.columns), recv, genes, lr)
    S, R, W, A = fit_factors(T, N_FACTORS, L1_RECEIVER, SEED)
    return dict(S=S, R=R, W=W, A=A, pairs=list(zip(lr.ligand_gene, lr.receptor_gene)))


def load_human():
    return fit_from_csv("human")


def load_macaque():
    return fit_from_csv("macaque")


def load_kraft():
    z = np.load(work_dir("kraft") / "liana_foldin_factors.npz", allow_pickle=True)
    return dict(S=z["S"], R=z["R"], W=z["W"], A=z["LR"],
                pairs=list(zip(z["ligand"], z["receptor"])))
