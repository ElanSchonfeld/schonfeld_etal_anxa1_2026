#!/usr/bin/env python3
"""Build the bin-level striatal projections of the Gaertner and Salmani mouse senders under work/mouse/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python build_mouse_projection.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import source_root, work_dir
from lib.tensor_projection import tensor_factor_cp_projection
from mouse.senders import lrrk2_pseudobulk, salmani_pseudobulk

SEED, N_FACTORS, CV_THRESHOLD = 42, 20, 0.3
STRIATAL = [f"CP-{z}" for z in ("rd", "rv", "md", "mv", "cd", "cv")] + \
    ["ACB-core", "ACB-shell", "OT-med", "OT-lat"]
DROP = {"Gad2:Syndig1", "Lef1"}
SENDER_GENES_H5AD = "cell2cell_final_v2/data/senders/lrrk2_da_senders_spatial_fixed.h5ad"
LRRK2_H5AD = "Data/Mouse/Mouse_LRRK2_Dopamine_RAW.h5ad"
SALMANI_H5AD = "Data/Mouse/YaghmaeianSalmani/YaghmaeianSalmani_authorDA_predicted.h5ad"


def lr_pairs(sender_genes, recv_genes):
    from liana.resource import select_resource
    r = select_resource("mouseconsensus").rename(
        columns={"ligand": "ligand_complex", "receptor": "receptor_complex"})
    sci = {g.upper(): g for g in sender_genes}
    rci = {g.upper(): g for g in recv_genes}
    r["ligand_complex"] = r.ligand_complex.map(lambda g: sci.get(str(g).upper()))
    r["receptor_complex"] = r.receptor_complex.map(lambda g: rci.get(str(g).upper()))
    return r.dropna(subset=["ligand_complex", "receptor_complex"]).drop_duplicates()[
        ["ligand_complex", "receptor_complex"]].reset_index(drop=True)


def main():
    d = work_dir("mouse")
    root = source_root()
    sender_genes = list(ad.read_h5ad(root / SENDER_GENES_H5AD, backed="r").var_names)
    bins = pd.read_csv(d / "receiver_bins_ccf.csv")
    keep = bins.slab_region.astype(str).isin(STRIATAL).to_numpy()
    recv = np.load(d / "recv_bin_expr.npz")["recv_bin_expr"].astype(np.float32)[keep]
    recv_genes = json.loads((d / "recv_bin_genes.json").read_text())
    bins = bins.loc[keep].reset_index(drop=True)
    LR = lr_pairs(sender_genes, recv_genes)

    lrrk2 = root / LRRK2_H5AD
    subtypes = sorted(s for s in ad.read_h5ad(lrrk2, backed="r").obs["subtype"].astype(str).unique()
                      if s not in DROP)
    senders = {"LRRK2": pd.DataFrame(
        lrrk2_pseudobulk(lrrk2, subtypes, sender_genes, set(LR.ligand_complex)),
        index=subtypes, columns=sender_genes)}
    mat, subs = salmani_pseudobulk(root / SALMANI_H5AD, "pred_subtype_unified", sender_genes)
    salmani = pd.DataFrame(mat, index=subs, columns=sender_genes)
    senders["Salmani"] = salmani.loc[[x for x in salmani.index if x not in DROP]]

    for tag, df in senders.items():
        sdf = df.loc[[x for x in df.index if not str(x).startswith("Gad2:")]]
        proj, _ = tensor_factor_cp_projection(
            sdf.to_numpy(np.float32), sender_genes, recv, recv_genes, LR, list(sdf.index),
            min_lig_subtype_cv=CV_THRESHOLD, n_factors=N_FACTORS, seed=SEED, force_method="top3")
        pd.DataFrame(proj, index=list(sdf.index), columns=[f"bin_{i}" for i in bins.bin_id]
                     ).to_csv(d / f"allenFinal_bin_level_{tag}.tsv", sep="\t")
    bins.to_csv(d / "allenFinal_receiver_bins.csv", index=False)


if __name__ == "__main__":
    main()
