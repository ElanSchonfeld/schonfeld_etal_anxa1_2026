#!/usr/bin/env python3
"""Refit the Kraft projection on the LIANA-consensus pairs, fold the Slide-tags spatial sub-bins onto it, and write the zone display matrix and the anchored sub-bin masses under work/kraft/.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace M2H_DATA_ROOT=/path/to/atlas-data python kraft_spatialbins.py
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kraft.prepare_kraft_inputs import CT_COL, MSN, ZONE_COL, ZONES, kraft_paths
from lib.parafac_pipeline import build_tensor, fit_factors, hierarchical_recon
from lib.paths import work_dir

BINCELLS = 50
SEED, N_FACTORS, L1_RECEIVER = 42, 20, 0.05
ZC = [f"Z{z}" for z in ZONES]


def cell_receptors(receptors):
    u2i = {r: i for i, r in enumerate(receptors)}
    parts = []
    for path in kraft_paths():
        a = ad.read_h5ad(path)
        keep = a.obs[CT_COL].astype(str).isin(MSN) & a.obs[ZONE_COL].astype(str).isin(ZONES)
        a = a[keep.values]
        Xc = (a.X.tocsc() if sparse.issparse(a.X) else sparse.csc_matrix(a.X))
        vmap = {g: i for i, g in enumerate(a.var_names)}
        cols = np.array([vmap.get(r, -1) for r in receptors])
        ok = np.where(cols >= 0)[0]
        sm = Xc[:, cols[ok]].tocoo()
        parts.append((sparse.csr_matrix((sm.data.astype(np.float32), (sm.row, ok[sm.col])),
                                        shape=(a.n_obs, len(receptors))),
                      a.obs[ZONE_COL].astype(str).values, a.obs[CT_COL].astype(str).values,
                      a.obs["donor"].astype(str).values,
                      a.obs["x"].to_numpy(float), a.obs["y"].to_numpy(float)))
    return (sparse.vstack([p[0] for p in parts]).tocsr(),
            *[np.concatenate([p[i] for p in parts]) for i in range(1, 6)])


def spatial_subbins(REC, zone, ct, donor, xx, yy, pair_cols):
    rows, bz, ncl = [], [], []
    for z in ZONES:
        for c in MSN:
            for d in np.unique(donor):
                idx = np.where((zone == z) & (ct == c) & (donor == d))[0]
                if len(idx) < BINCELLS:
                    continue
                k = max(1, len(idx) // BINCELLS)
                side = int(np.ceil(k ** 0.5))
                gx = np.clip(((xx[idx] - xx[idx].min()) / (np.ptp(xx[idx]) + 1e-9) * side).astype(int), 0, side - 1)
                gy = np.clip(((yy[idx] - yy[idx].min()) / (np.ptp(yy[idx]) + 1e-9) * side).astype(int), 0, side - 1)
                for key in np.unique(gx * side + gy):
                    ci = idx[gx * side + gy == key]
                    if len(ci) >= BINCELLS:
                        rows.append(np.log1p(np.asarray(REC[ci][:, pair_cols].mean(0)).ravel()))
                        bz.append(f"Z{z}")
                        ncl.append(len(ci))
    return np.vstack(rows), np.array(bz), np.array(ncl, dtype=float)


def anchored_mass(proj, bz, ncl):
    zone_proj = pd.DataFrame({z: proj.iloc[:, np.where(bz == z)[0]].mean(1)
                              for z in ZC if (bz == z).any()}).reindex(columns=ZC)
    mass = pd.DataFrame(0.0, index=proj.index, columns=proj.columns)
    for z in ZC:
        vs = np.where(bz == z)[0]
        if len(vs) == 0:
            continue
        nc = ncl[vs]
        sh = proj.iloc[:, vs].to_numpy()
        wmean = (sh * nc[None, :]).sum(1) / nc.sum()
        score = np.divide(zone_proj[z].to_numpy()[:, None] * sh, wmean[:, None],
                          out=np.zeros_like(sh), where=wmean[:, None] > 0)
        mass.iloc[:, vs] = score * (nc / nc.sum())[None, :]
    return zone_proj, mass


def main():
    d = work_dir("kraft")
    sdf = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    sub = list(sdf.index)
    lr = pd.read_csv(d / "lr_pairs_filtered.csv")
    rcv = pd.read_csv(d / "receivers_pseudobulk.csv")
    recv_genes = json.loads((d / "receiver_genes.json").read_text())

    T = build_tensor(sdf.values.astype(np.float32), list(sdf.columns),
                     rcv.values.astype(np.float32), recv_genes, lr).astype(np.float64)
    S, R, W, LR = fit_factors(T, N_FACTORS, L1_RECEIVER, SEED)

    sg = {g: i for i, g in enumerate(sdf.columns)}
    sl = np.zeros((len(sub), len(lr)))
    for k, lig in enumerate(lr["ligand_gene"]):
        if lig in sg:
            sl[:, k] = np.log1p(sdf.values[:, sg[lig]])
    rgi = {g: i for i, g in enumerate(rcv.columns)}
    ridx = np.array([rgi.get(r, -1) for r in lr["receptor_gene"]])
    RRc = np.zeros((len(rcv), len(lr)))
    ok = ridx >= 0
    RRc[:, ok] = np.log1p(rcv.values[:, ridx[ok]])
    norm_k = np.linalg.norm(sl, axis=0) * np.linalg.norm(RRc, axis=0)
    norm_k[norm_k < 1e-12] = 1e-12

    receptors = sorted(set(lr["receptor_gene"]))
    REC, zone, ct, donor, xx, yy = cell_receptors(receptors)
    u2i = {r: i for i, r in enumerate(receptors)}
    pair_cols = np.array([u2i[r] for r in lr["receptor_gene"]])
    rows, bz, ncl = spatial_subbins(REC, zone, ct, donor, xx, yy, pair_cols)

    Kn, Sn, Rn = len(lr), len(sub), W.shape[0]
    A = (W[None, None, :] * LR[:, None, :] * S[None, :, :]).reshape(Kn * Sn, Rn)
    Ap = np.linalg.pinv(A)
    folded = np.clip(((rows / norm_k)[:, :, None] * sl.T[None, :, :]
                      ).reshape(rows.shape[0], Kn * Sn) @ Ap.T, 0, None)
    np.savez(d / "liana_foldin_factors.npz", R=folded, S=S, LR=LR, W=W, bz=bz,
             ncl=ncl, ligand=lr["ligand_gene"].to_numpy(),
             receptor=lr["receptor_gene"].to_numpy(), subtypes=np.array(sub))
    sw = S * W
    top3 = np.zeros((Sn, folded.shape[0]))
    for i in range(Sn):
        for fi in np.argsort(sw[i])[-3:]:
            top3[i] += sw[i, fi] * folded[:, fi]
    top3 = top3 / np.maximum(top3.sum(1, keepdims=True), 1e-10)

    Tsub = np.einsum("ik,vk->kiv", sl, rows).astype(np.float32)
    for k in range(Tsub.shape[0]):
        nz = np.linalg.norm(Tsub[k])
        if nz > 1e-8:
            Tsub[k] /= nz
    P = hierarchical_recon(Tsub, top3, sub, SEED)
    proj = pd.DataFrame(P, index=sub, columns=[f"bin_{i}" for i in range(len(bz))])
    zone_proj, mass = anchored_mass(proj, bz, ncl)
    zone_proj.to_csv(d / "projection_matrix_spatialbins_liana_hier.tsv", sep="\t")
    mass.to_csv(d / "anchored_mass_liana_hier.tsv", sep="\t")
    pd.DataFrame({"zone": bz, "ncells": ncl.astype(int)}).to_csv(d / "anchored_bins_liana.csv", index=False)


if __name__ == "__main__":
    main()
