#!/usr/bin/env python3
"""Write the per-pair reconstruction contribution and the per-family communication magnitude for the five projection datasets under work/drivers/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python recon_and_magnitude.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivers.factor_loaders import load_human, load_kraft, load_macaque
from lib.paths import source_root, work_dir
from lib.tensor_projection import tensor_factor_cp_projection
from mouse.build_mouse_projection import (DROP, LRRK2_H5AD, SALMANI_H5AD, SENDER_GENES_H5AD,
                                          STRIATAL, CV_THRESHOLD, N_FACTORS, SEED, lr_pairs)
from mouse.senders import lrrk2_pseudobulk, salmani_pseudobulk

FAMS = ["Anxa1", "Sox6", "Calb1"]
ANXA1 = {"Sox6:Tafa1", "Sox6:Vcan"}
CSV_DATASETS = {"human": load_human, "kraft": load_kraft, "macaque": load_macaque}


def fam_of(s):
    if s in ANXA1:
        return "Anxa1"
    if str(s).startswith("Sox6:"):
        return "Sox6"
    if str(s).startswith("Calb1:"):
        return "Calb1"
    return None


def recon_factor_mean(d):
    S, W, R, A = (np.asarray(d[k], float) for k in ("S", "W", "R", "A"))
    recon = (A * (W * S.mean(0) * R.mean(0))).sum(1)
    L = [str(a).upper() for a, _ in d["pairs"]]
    Rr = [str(b).upper() for _, b in d["pairs"]]
    return pd.DataFrame({"ligand": L, "receptor": Rr, "recon": recon,
                         "recon_pct": rankdata(recon) / len(recon) * 100})


def recon_slice_norm(factors):
    LR, S, R, W = (np.asarray(factors[k]) for k in ("lr", "sender", "receiver", "weights"))
    SR = np.einsum("sr,br->rsb", S, R)
    norm = np.array([np.linalg.norm((W * LR[k])[:, None, None] * SR) for k in range(LR.shape[0])])
    lrp = factors["lr_pairs"]
    return pd.DataFrame({"ligand": lrp.ligand_complex.astype(str).str.upper(),
                         "receptor": lrp.receptor_complex.astype(str).str.upper(),
                         "recon": norm,
                         "recon_pct": pd.Series(norm).rank(pct=True).to_numpy() * 100.0})


def perfam_rows(name, L, R, sender_df, recnorm):
    rows = []
    subs = list(sender_df.index)
    for fam in FAMS:
        mem = [s for s in subs if fam_of(s) == fam]
        if not mem:
            continue
        sub = sender_df.loc[mem]
        lign = np.array([np.linalg.norm(np.log1p(sub[g].values)) / np.sqrt(len(mem))
                         if g in sub.columns else 0.0 for g in L])
        mag = lign * recnorm
        rank = rankdata(-mag, method="ordinal").astype(int)
        for k in range(len(L)):
            rows.append(dict(dataset=name, family=fam, ligand=L[k], receptor=R[k],
                             pair=f"{L[k]}->{R[k]}", mag=float(mag[k]), rank=int(rank[k])))
    return rows


def csv_magnitude(name):
    d = work_dir(name)
    lr = pd.read_csv(d / "lr_pairs_filtered.csv")
    lr = lr[lr.source == "liana_consensus"].reset_index(drop=True)
    L = lr.ligand_gene.str.upper().values
    R = lr.receptor_gene.str.upper().values
    sd = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    sd.columns = [c.upper() for c in sd.columns]
    rv = pd.read_csv(d / "receivers_pseudobulk.csv", low_memory=False)
    rv.columns = [c.upper() for c in rv.columns]
    recnorm = np.array([np.linalg.norm(np.log1p(rv[g].values)) if g in rv.columns else 0.0 for g in R])
    return perfam_rows(name, L, R, sd, recnorm)


def mouse_magnitude(name, sender_df, recv_expr, recv_genes, L, R):
    rgm = {g.upper(): i for i, g in enumerate(recv_genes)}
    re = np.asarray(recv_expr, float)
    recn = np.array([np.linalg.norm(np.log1p(re[:, rgm[g]])) if g in rgm else 0.0 for g in R])
    return perfam_rows(name, L, R, sender_df, recn)


def mouse_inputs():
    d = work_dir("mouse")
    root = source_root()
    sender_genes = list(ad.read_h5ad(root / SENDER_GENES_H5AD, backed="r").var_names)
    bins = pd.read_csv(d / "receiver_bins_ccf.csv")
    keep = bins.slab_region.astype(str).isin(STRIATAL).to_numpy()
    recv = np.load(d / "recv_bin_expr.npz")["recv_bin_expr"].astype(np.float32)[keep]
    recv_genes = json.loads((d / "recv_bin_genes.json").read_text())
    LR = lr_pairs(sender_genes, recv_genes)
    lrrk2 = root / LRRK2_H5AD
    subtypes = sorted(s for s in ad.read_h5ad(lrrk2, backed="r").obs["subtype"].astype(str).unique()
                      if s not in DROP)
    senders = {"mouse_LRRK2": pd.DataFrame(
        lrrk2_pseudobulk(lrrk2, subtypes, sender_genes, set(LR.ligand_complex)),
        index=subtypes, columns=sender_genes)}
    mat, subs = salmani_pseudobulk(root / SALMANI_H5AD, "pred_subtype_unified", sender_genes)
    sal = pd.DataFrame(mat, index=subs, columns=sender_genes)
    senders["salmani"] = sal.loc[[x for x in sal.index if x not in DROP]]
    return sender_genes, recv, recv_genes, LR, senders


def main():
    out = work_dir("drivers")
    rows = []
    for name, loader in CSV_DATASETS.items():
        recon_factor_mean(loader()).to_csv(out / f"{name}_recon.csv", index=False)
        rows += csv_magnitude(name)

    sender_genes, recv, recv_genes, LR, senders = mouse_inputs()
    L = LR.ligand_complex.astype(str).str.upper().values
    R = LR.receptor_complex.astype(str).str.upper().values
    for name, sdf in senders.items():
        up = sdf.copy()
        up.columns = [c.upper() for c in up.columns]
        need = sorted(set(L))
        sd = pd.DataFrame({g: (up[g].to_numpy(float) if g in up.columns else np.zeros(len(up)))
                           for g in need}, index=list(up.index))
        rows += mouse_magnitude(name, sd, recv, recv_genes, L, R)
        s = sdf.loc[[x for x in sdf.index if not str(x).startswith("Gad2:")]]
        _, factors = tensor_factor_cp_projection(
            s.to_numpy(np.float32), sender_genes, recv, recv_genes, LR, list(s.index),
            min_lig_subtype_cv=CV_THRESHOLD, n_factors=N_FACTORS, seed=SEED, force_method="top3")
        recon_slice_norm(factors).to_csv(out / f"{name}_recon.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "perfamily_magnitude.csv", index=False)


if __name__ == "__main__":
    main()
