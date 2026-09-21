#!/usr/bin/env python3
"""Freeze the per-cell table and donor-level statistics for the Gad2 panel.

Run: python freeze_fig2_gad2.py
"""
import json
from os import environ
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.stats import wilcoxon

MAN = Path(environ["M2H_SOURCE_ROOT"])
H5AD = MAN / "Data/Human/Kamath/Kamath_DA_predicted.h5ad"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"; FROZEN.mkdir(exist_ok=True)

DA_MARKERS = ["TH", "SLC6A3", "SLC18A2", "DDC", "NR4A2"]
NONDA_MARKERS = ["GAD1", "GAD2", "SLC32A1", "AQP4"]
DA_PROGRAM = ["SLC6A3", "SLC18A2", "DDC", "NR4A2"]
MARKERS = DA_MARKERS + NONDA_MARKERS


def _col(ad, gene, tot):
    i = list(ad.var_names).index(gene); x = ad.X[:, i]
    c = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return np.log1p(c / np.maximum(tot, 1) * 1e4)


def donor_paired_p(v, gad, donor, donors_use):
    dk = np.array([v[(donor == d) & ~gad].mean() for d in donors_use])
    de = np.array([v[(donor == d) & gad].mean() for d in donors_use])
    if np.allclose(dk, de): return 1.0
    try: return float(wilcoxon(de, dk).pvalue)
    except ValueError: return 1.0


def fold(v, gad):
    r = np.median(v[gad]) / max(np.median(v[~gad]), 1e-9)
    return f"{1/r:.1f}" + r"$\times\!\downarrow$" if r < 1 else f"{r:.1f}" + r"$\times\!\uparrow$"


def main():
    ad = sc.read_h5ad(H5AD)
    gad = ad.obs["gad2_excluded"].values.astype(bool)
    donor = ad.obs["donor_id"].astype(str).values
    tot = np.asarray(ad.X.sum(1)).ravel()
    umis = ad.obs["total_counts"].values.astype(float)
    genes = ad.obs["n_genes_by_counts"].values.astype(float)
    mt = ad.obs["pct_counts_mt"].values.astype(float)
    ribo_mask = np.array([g.upper().startswith(("RPS", "RPL")) for g in ad.var_names])
    ribo = np.asarray(ad.X[:, ribo_mask].sum(1)).ravel() / np.maximum(tot, 1) * 100
    xy = ad.obsm["X_umap"]

    df = pd.DataFrame({
        "umap_x": xy[:, 0], "umap_y": xy[:, 1], "gad2_excluded": gad, "donor_id": donor,
        "umis": umis, "genes": genes, "mt": mt, "ribo": ribo,
    })
    for g in MARKERS:
        df[g] = _col(ad, g, tot)
    df.to_parquet(FROZEN / "kamath_gad2_percell.parquet", index=False)

    donors_use = [d for d in np.unique(donor)
                  if ((donor == d) & ~gad).sum() >= 10 and ((donor == d) & gad).sum() >= 10]
    Z = [(df[g].values - df[g].values.mean()) / (df[g].values.std() + 1e-9) for g in DA_PROGRAM]
    da_score = np.mean(Z, axis=0)
    slope = [{"donor": d, "kept": float(da_score[(donor == d) & ~gad].mean()),
              "excl": float(da_score[(donor == d) & gad].mean())} for d in donors_use]

    stats = {
        "n_kept": int((~gad).sum()), "n_excl": int(gad.sum()), "nd": len(donors_use),
        "pct_excl": round(100 * gad.mean(), 1),
        "qc": {k: {"p": donor_paired_p(v, gad, donor, donors_use), "fold": fold(v, gad)}
               for k, v in [("umis", umis), ("genes", genes), ("mt", mt), ("ribo", ribo)]},
        "markers": {g: donor_paired_p(df[g].values, gad, donor, donors_use) for g in MARKERS},
        "da_score_p": donor_paired_p(da_score, gad, donor, donors_use),
        "slopegraph": slope,
    }
    json.dump(stats, open(FROZEN / "kamath_gad2_stats.json", "w"), indent=1)
    print(f"  kamath_gad2_percell.parquet: {len(df)} cells x {df.shape[1]} cols")
    print(f"  stats: kept {stats['n_kept']:,} / excl {stats['n_excl']:,} ({stats['pct_excl']}%); {stats['nd']} donors; DA-score p={stats['da_score_p']:.2e}")


if __name__ == "__main__":
    main()
