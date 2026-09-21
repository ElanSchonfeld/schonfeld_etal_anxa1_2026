#!/usr/bin/env python3
"""Recompute panel e's matrix from the h5ad and check the capture is not stale."""
import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
import yaml
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "config" / "scdrs_traits.yaml"

CONTROL_VALUES = ("Ctrl", "Control")
GROUPING = "hmoe_subtype"
_MISSING = ("nan", "NaN", "None", "", "NA", "<NA>")
_FAM_RANK = {"Sox6": 0, "Calb1": 1, "Gad2": 2}


def _family_prefix(c):
    for fam in ("Sox6", "Calb1", "Gad2"):
        if str(c).startswith(fam):
            return fam
    return None


def _bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    out = np.empty(n)
    out[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(out, 0.0, 1.0)


def _star(f):
    return "***" if f < 0.001 else "**" if f < 0.01 else "*" if f < 0.05 else ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the DopaBase Human scDRS heatmap values."
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=os.environ.get("DOPABASE_BROWSER_H5AD"),
        help="deployed browser H5AD, or set DOPABASE_BROWSER_H5AD",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="trait-order YAML (default: the checked-in release config)",
    )
    args = parser.parse_args()
    if args.h5ad is None:
        parser.error("provide --h5ad or set DOPABASE_BROWSER_H5AD")
    return args


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    display = {t["trait_id"]: t["display_name"] for t in cfg["scdrs"]["traits"]}
    obs = ad.read_h5ad(args.h5ad, backed="r").obs
    print(f"config {args.config.name}: {len(display)} traits | h5ad {len(obs):,} cells")

    mask = obs["disease_status"].astype(str).isin(CONTROL_VALUES).to_numpy()
    labels = obs[GROUPING].astype(str).to_numpy()[mask]
    keep = ~np.isin(labels, _MISSING)
    keep &= ~np.array([_family_prefix(x) == "Gad2" for x in labels], dtype=bool)
    labels = labels[keep]
    inner = mask.copy()
    inner[mask] = keep
    mask = inner

    cats = sorted(pd.unique(labels))
    gidx = np.array([cats.index(c) for c in labels])
    present = [t for t in display if f"scdrs_{t}_norm_score" in obs.columns]
    missing = [t for t in display if t not in present]
    if missing:
        print(f"WARNING: config lists traits absent from the h5ad: {missing}")

    zval = np.full((len(present), len(cats)), np.nan)
    pmat = np.ones((len(present), len(cats)))
    for i, t in enumerate(present):
        ns = obs[f"scdrs_{t}_norm_score"].values[mask].astype(float)
        ok = np.isfinite(ns)
        gmean, gstd = float(ns[ok].mean()), float(ns[ok].std(ddof=1))
        for j in range(len(cats)):
            inm = (gidx == j) & ok
            if not inm.any():
                continue
            sub, rest = ns[inm], ns[(gidx != j) & ok]
            zval[i, j] = (sub.mean() - gmean) / gstd
            pmat[i, j] = stats.mannwhitneyu(sub, rest, alternative="greater").pvalue

    zc = np.nan_to_num(zval, nan=0.0)
    zsd = zc.std(axis=1, ddof=1, keepdims=True)
    zc = (zc - zc.mean(axis=1, keepdims=True)) / np.where(zsd > 0, zsd, 1.0)
    alpha = sorted(range(len(cats)),
                   key=lambda i: (_FAM_RANK.get(_family_prefix(cats[i]), 9), str(cats[i])))
    order = []
    for fam in ("Sox6", "Calb1"):
        block = [i for i in alpha if _family_prefix(cats[i]) == fam]
        if len(block) <= 2:
            order.extend(block)
            continue
        link = linkage(np.nan_to_num(pdist(zc[:, block].T, metric="cosine"), nan=0.0),
                       method="average")
        order.extend(block[j] for j in leaves_list(link))
    cats = [cats[i] for i in order]
    zval, pmat = zval[:, order], pmat[:, order]
    fdr = np.vstack([_bh_fdr(pmat[i]) for i in range(pmat.shape[0])])

    print(f"\nEXPECT IN CAPTURE: "
          f"{len(present)} traits x {len(cats)} subtypes, {int(mask.sum()):,} cells\n")

    w = max(len(display[t]) for t in present)
    print(" " * (w + 2) + " ".join(f"{c.split(':')[1][:9]:>10s}" for c in cats))
    for i, t in enumerate(present):
        row = " ".join(f"{zval[i, j]:+.2f}{_star(fdr[i, j])}".rjust(10)
                       for j in range(len(cats)))
        print(f"{display[t]:>{w}s}  {row}")
    print("\nRows are in config order and columns are clustered within family.")


if __name__ == "__main__":
    main()
