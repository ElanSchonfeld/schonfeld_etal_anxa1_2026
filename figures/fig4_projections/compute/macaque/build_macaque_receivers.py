#!/usr/bin/env python3
"""Build the rhesus HMBA-BG receiver pseudobulks for region x cell-type bins and for region x cell-type x donor bins under work/macaque/.

Run: M2H_DATA_ROOT=/path/to/atlas-data python build_macaque_receivers.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import data_root, work_dir

RECEIVER_GROUPS = [
    "STRd D1 Matrix MSN", "STRd D1 Striosome MSN", "STRd D2 Matrix MSN", "STRd D2 Striosome MSN",
    "STRd D2 StrioMat Hybrid MSN", "STRv D1 MSN", "STRv D2 MSN", "STRv D1 NUDAP MSN",
    "STR D1D2 Hybrid MSN", "OT D1 ICj",
    "STR FS PTHLH-PVALB GABA", "STR TAC3-PLPP4 GABA", "STR SST-CHODL GABA", "STR SST-ADARB2 GABA",
    "STR SST-RSPO2 GABA", "STR LYPD6-RSPO2 GABA", "STRd Cholinergic GABA", "STR Cholinergic GABA",
]
REGIONS = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NAC"]
MIN_CELLS = 50
RHESUS = "Macaca mulatta"


def sidecars():
    return data_root() / "human/atlas/allen/basal_ganglia/multiome"


def macaque_dir():
    return data_root() / "macaque/allen"


def rhesus_donors():
    d = pd.read_csv(sidecars() / "donor_metadata.csv")
    return set(d.loc[d["species_scientific_name"] == RHESUS, "donor_label"])


def receiver_metadata():
    side = sidecars()
    cc = pd.read_csv(side / "cell_to_cluster_membership.csv", usecols=["cell_label", "cluster_alias"])
    ca = pd.read_csv(side / "cluster_to_cluster_annotation_membership.csv")
    gmap = ca[ca["cluster_annotation_term_set_name"] == "Group"][
        ["cluster_alias", "cluster_annotation_term_name"]
    ].rename(columns={"cluster_annotation_term_name": "group_name"})
    cell_meta = pd.read_csv(macaque_dir() / "cell_metadata.csv", usecols=["cell_label", "library_label"])
    lib_meta = pd.read_csv(side / "library_metadata.csv",
                           usecols=["library_label", "region_of_interest_label", "donor_label"]
                           ).rename(columns={"region_of_interest_label": "region"})
    return (cc.merge(gmap, on="cluster_alias", how="left")
            .merge(cell_meta, on="cell_label", how="inner")
            .merge(lib_meta, on="library_label", how="left"))


def region_bins(meta, regions):
    bins = []
    for region in regions:
        for grp in RECEIVER_GROUPS:
            cells = meta.loc[(meta["region"] == region) & (meta["group_name"] == grp), "cell_label"].values
            if len(cells) >= MIN_CELLS:
                bins.append({"bin_id": len(bins), "region": region, "group_name": grp,
                             "n_cells": len(cells), "cell_labels": cells})
    return bins


def donor_bins(meta, regions):
    bins = []
    for region in regions:
        for grp in RECEIVER_GROUPS:
            for donor in sorted(meta["donor_label"].dropna().unique()):
                cells = meta.loc[(meta["region"] == region) & (meta["group_name"] == grp)
                                 & (meta["donor_label"] == donor), "cell_label"].values
                if len(cells) >= MIN_CELLS:
                    bins.append({"bin_id": len(bins), "region": region, "group_name": grp,
                                 "donor": donor, "n_cells": len(cells), "cell_labels": cells})
    return bins


def pseudobulk(adata, bins):
    label_to_idx = {label: i for i, label in enumerate(adata.obs_names.values)}
    out = np.zeros((len(bins), adata.n_vars), dtype=np.float32)
    for i, b in enumerate(bins):
        idx = [label_to_idx[c] for c in b["cell_labels"] if c in label_to_idx]
        if not idx:
            continue
        X = adata.X[idx]
        out[i] = (np.asarray(X.mean(axis=0)).ravel() if sparse.issparse(X) else X.mean(axis=0))
    return out


def write(out, bins, expr, genes):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(expr, columns=genes).to_csv(out / "receivers_pseudobulk.csv", index=False)
    pd.DataFrame([{k: v for k, v in b.items() if k != "cell_labels"} for b in bins]
                 ).to_csv(out / "bin_metadata.csv", index=False)
    (out / "receiver_genes.json").write_text(json.dumps(list(genes)))


def main():
    out = work_dir("macaque")
    meta = receiver_metadata()
    meta = meta[meta["group_name"].isin(RECEIVER_GROUPS)
                & meta["donor_label"].isin(rhesus_donors())].copy()
    adata = ad.read_h5ad(macaque_dir() / "HMBA-10xMultiome-BG-Macaque-raw.h5ad", backed="r")
    genes = (adata.var["gene_symbol"].values if "gene_symbol" in adata.var.columns
             else adata.var_names.values)
    meta = meta[meta["cell_label"].isin(set(adata.obs_names.values))].copy()
    rb = region_bins(meta[meta["region"].isin(REGIONS)], REGIONS)
    write(out, rb, pseudobulk(adata, rb), genes)
    db = donor_bins(meta[meta["region"].isin(REGIONS)], REGIONS)
    write(out / "donor", db, pseudobulk(adata, db), genes)


if __name__ == "__main__":
    main()
