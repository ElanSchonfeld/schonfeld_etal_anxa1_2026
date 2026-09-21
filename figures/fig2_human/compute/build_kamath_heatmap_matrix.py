#!/usr/bin/env python3
"""Freeze the plotted matrix for the Kamath subtype heatmap.

Run: python build_kamath_heatmap_matrix.py
"""
import json
from os import environ
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")
MANUSCRIPT_ROOT = Path(environ["M2H_SOURCE_ROOT"])
H5AD = MANUSCRIPT_ROOT / "Data" / "Human" / "Kamath" / "Kamath_DA_predicted.h5ad"
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
sys.path.insert(0, str(HERE.parent / "code"))
from kamath_common import FAMILY_COLORS, SUBTYPE_COLORS, infer_family  # noqa: E402

ANX = ["Sox6:Tafa1", "Sox6:Vcan"]
SEED = 42
MIN_SUBTYPE = 50
MIN_CELLS_DONOR = 10
TOPK = 3
ANXA_N = 8
CAP = 300
ANXA_ALWAYS = ["TAFA1", "ALDH1A1"]
GRP_ORDER = ["Anxa1+", "Sox6+", "Calb1+"]
GRP_COL = {"Anxa1+": "#08519c", "Sox6+": "#4292c6", "Calb1+": FAMILY_COLORS["Calb1"]}


def lognorm_layer(ad):
    a = ad.copy(); a.X = a.X.astype(np.float32)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    return a


def _auroc(X, pos, neg, gi):
    """Rank-based single-gene AUROC (positives=pos) over gene indices gi."""
    from scipy.stats import rankdata
    idx = np.where(pos | neg)[0]; y = pos[idx].astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    sub = X[idx][:, gi]; sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    out = np.empty(len(gi))
    for k in range(sub.shape[1]):
        r = rankdata(sub[:, k]); out[k] = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def family_worst2_score(a, pos, samefam, min_cells=10):
    """Family-aware worst-of-2 AUROC specificity for one subtype (pos vs rest-of-own-family AND pos vs other-families, min of the two)."""
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    within = samefam & ~pos; cross = ~samefam
    worst = np.full(a.n_vars, -np.inf)
    det = np.asarray((X[pos] > 0).mean(0)).ravel()
    if int(within.sum()) < min_cells or int(cross.sum()) < min_cells:
        return worst, det
    mP = np.asarray(X[pos].mean(0)).ravel()
    mW = np.asarray(X[within].mean(0)).ravel(); mC = np.asarray(X[cross].mean(0)).ravel()
    keep = np.where((mP > mW) & (mP > mC) & (det > 0.25))[0]
    if len(keep):
        worst[keep] = np.minimum(_auroc(X, pos, within, keep), _auroc(X, pos, cross, keep))
    return worst, det


def anxa_pooled_markers(a, n):
    """Pooled Anxa1+ markers by worst-of-2 Wilcoxon AUROC (Anxa1 vs other-Sox6 AND vs Calb1)."""
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    pr = a.obs["pred"].astype(str).values
    anx = np.isin(pr, ANX); sox6 = pd.Series(pr).str.startswith("Sox6:").values
    oS, cal = sox6 & ~anx, ~sox6
    mA = np.asarray(X[anx].mean(0)).ravel(); mO = np.asarray(X[oS].mean(0)).ravel()
    mC = np.asarray(X[cal].mean(0)).ravel(); dA = np.asarray((X[anx] > 0).mean(0)).ravel()
    keep = np.where((mA > mO) & (mA > mC) & (dA > 0.25))[0]
    worst = np.minimum(_auroc(X, anx, oS, keep), _auroc(X, anx, cal, keep))
    var = np.asarray(a.var_names)
    sc_ = dict(zip(var[keep], worst))
    ranked = [var[i] for i in keep[np.argsort(-worst)]]
    always = [g for g in ANXA_ALWAYS if g in a.var_names]
    block = (always + [g for g in ranked if g not in always])[:n]
    return block, {g: float(sc_.get(g, float("nan"))) for g in block}


def main():
    ad = sc.read_h5ad(H5AD)
    ad = ad[~ad.obs["gad2_excluded"].values.astype(bool)].copy()
    ad.X = sp.csr_matrix(ad.X) if not sp.issparse(ad.X) else ad.X.tocsr()
    rng = np.random.RandomState(SEED)
    a = lognorm_layer(ad)
    a.obs["pred"] = a.obs["pred_subtype"].astype(str)
    pred = a.obs["pred"].values
    fam_rank = {"Sox6": 0, "Calb1": 1, "Gad2": 2}
    subs = [s for s in pd.unique(pred) if (pred == s).sum() >= MIN_SUBTYPE]
    subs = sorted(subs, key=lambda s: (fam_rank.get(infer_family(s), 3), s))

    anx = np.isin(pred, ANX)
    fam = np.array([infer_family(s) for s in pred])
    grp_masks = {"Anxa1+": anx, "Sox6+": fam == "Sox6", "Calb1+": fam == "Calb1"}

    anx_genes, anx_sc = anxa_pooled_markers(a, ANXA_N)
    G = a.n_vars; SPEC = np.full((G, len(subs)), -np.inf)
    for j, S in enumerate(subs):
        score, det = family_worst2_score(a, pred == S, fam == infer_family(S))
        score = score.copy(); score[(score <= 0.5) | (det < 0.25)] = -np.inf
        SPEC[:, j] = score
    var = np.asarray(a.var_names)
    anx_set = set(anx_genes)
    for i, g in enumerate(var):
        if g in anx_set:
            SPEC[i, :] = -np.inf
    best = np.argmax(SPEC, axis=1); bestval = SPEC[np.arange(G), best]
    sub_genes, owner, scores = [], {}, {}
    for g in anx_genes:
        owner[g] = "Anxa1+"; scores[g] = anx_sc[g]
    for j, S in enumerate(subs):
        idx = np.where((best == j) & np.isfinite(bestval))[0]
        idx = idx[np.argsort(-SPEC[idx, j])]
        for i in idx[:TOPK]:
            g = var[i]; sub_genes.append(g); owner[g] = S; scores[g] = float(SPEC[i, j])
    genes = anx_genes + sub_genes

    gidx = [list(var).index(g) for g in genes]
    cols, main_col_subtype, cur = [], [], 0
    for s in subs:
        ii = np.where(pred == s)[0]
        if len(ii) > CAP:
            ii = rng.choice(ii, CAP, replace=False)
        cols.append(ii); main_col_subtype += [s] * len(ii); cur += len(ii)
    order = np.concatenate(cols)
    Xm = a.X[order][:, gidx]
    M = (Xm.toarray() if sp.issparse(Xm) else np.asarray(Xm)).T
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True) + 1e-9
    Z = (M - mu) / sd

    agg_cols, agg_col_group = [], []
    for Gn in GRP_ORDER:
        ii = np.where(grp_masks[Gn])[0]
        if len(ii) > CAP:
            ii = rng.choice(ii, CAP, replace=False)
        agg_cols.append(ii); agg_col_group += [Gn] * len(ii)
    aorder = np.concatenate(agg_cols)
    Xa = a.X[aorder][:, gidx]
    Ma = (Xa.toarray() if sp.issparse(Xa) else np.asarray(Xa)).T
    Za = (Ma - mu) / sd

    VLO, VHI = np.percentile(np.concatenate([Z.ravel(), Za.ravel()]), [5, 95])

    pd.DataFrame(Z, index=genes, columns=[f"c{i:04d}" for i in range(Z.shape[1])]).to_parquet(
        FROZEN / "kamath_heatmap_matrix.parquet")
    pd.DataFrame(Za, index=genes, columns=[f"c{i:04d}" for i in range(Za.shape[1])]).to_parquet(
        FROZEN / "kamath_heatmap_agg.parquet")
    meta = {
        "genes": genes, "owner": owner, "subs": subs, "blocks": ["Anxa1+"] + subs,
        "main_col_subtype": main_col_subtype, "agg_col_group": agg_col_group,
        "GRP_ORDER": GRP_ORDER, "GRP_COL": GRP_COL,
        "SUBTYPE_COLORS": {s: SUBTYPE_COLORS.get(s, "#bbb") for s in subs},
        "VLO": float(VLO), "VHI": float(VHI),
    }
    (FROZEN / "kamath_heatmap_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote kamath_heatmap_matrix.parquet ({len(genes)} genes x {Z.shape[1]} cells), "
          f"agg ({Za.shape[1]} cells), meta ({len(subs)} subtypes)")
    print("  Anxa1+ block:", ", ".join(anx_genes))


if __name__ == "__main__":
    main()
