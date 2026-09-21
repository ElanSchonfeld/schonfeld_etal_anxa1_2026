#!/usr/bin/env python3
"""Build the rhesus HMBA-BG dopaminergic sender pseudobulk and its subtype cell counts under work/macaque/.

Run: M2H_DATA_ROOT=/path/to/atlas-data python build_macaque_senders.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import FIG, work_dir
from macaque.build_macaque_receivers import macaque_dir, rhesus_donors, sidecars

sys.path.insert(0, str(FIG.parents[1]))
import hmoe_annotate.hmoe_model as hm

DA_GROUPS = ["SN SOX6 Dopa", "SN-VTR CALB1 Dopa", "SN-VTR GAD2 Dopa"]
MIN_CELLS = 20
MODEL_DIR = FIG.parents[1] / "hmoe_annotate" / "model"


def da_cells():
    side = sidecars()
    cc = pd.read_csv(side / "cell_to_cluster_membership.csv", usecols=["cell_label", "cluster_alias"])
    ca = pd.read_csv(side / "cluster_to_cluster_annotation_membership.csv")
    gmap = ca[ca["cluster_annotation_term_set_name"] == "Group"][
        ["cluster_alias", "cluster_annotation_term_name"]].rename(
        columns={"cluster_annotation_term_name": "grp"})
    meta = pd.read_csv(macaque_dir() / "cell_metadata.csv",
                       usecols=["cell_label", "library_label", "donor_label"])
    meta = meta[meta["donor_label"].isin(rhesus_donors())]
    m = cc.merge(gmap, on="cluster_alias").merge(meta, on="cell_label", how="inner")
    return set(m.loc[m.grp.isin(DA_GROUPS), "cell_label"])


def collapse_symbols(X, symbols):
    total = X.sum(0)
    keep = {}
    for j, g in enumerate(symbols):
        if g not in keep or total[j] > total[keep[g]]:
            keep[g] = j
    cols = sorted(keep.values())
    return X[:, cols].astype(np.float32), symbols[cols]


def main():
    out = work_dir("macaque")
    wanted = da_cells()
    A = ad.read_h5ad(macaque_dir() / "HMBA-10xMultiome-BG-Macaque-raw.h5ad", backed="r")
    order = {c: i for i, c in enumerate(A.obs_names)}
    idx = sorted(order[c] for c in wanted if c in order)
    X = A.X[idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    Xc, genes = collapse_symbols(X, A.var["gene_symbol"].astype(str).values)

    bundle = hm.load_model(MODEL_DIR)
    leaves = list(bundle["leaves"])
    X_aligned, _, _ = hm.align_features(genes, sp.csr_matrix(Xc), bundle["feature_names"])
    P = hm.predict_proba(bundle, X_aligned)
    hard = np.array(leaves)[P.argmax(1)]

    rows, kept, counts = [], [], {}
    for leaf in leaves:
        mask = hard == leaf
        if mask.sum() < MIN_CELLS:
            continue
        rows.append(Xc[mask].mean(0))
        kept.append(leaf)
        counts[leaf] = int(mask.sum())
    pd.DataFrame(np.vstack(rows).astype(np.float32), index=kept, columns=genes
                 ).to_csv(out / "senders_pseudobulk.csv")
    (out / "sender_subtype_counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
