"""Recompute the HUMAN brain-wide selectivity axis using ALL DA donors.

Run:
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release" / "development"
OUT = ROOT / "results" / "brainwide_all_da_donors"
DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
DRIVE = DATA / "derived/m2h_fda_scdrs_atlas_v2"
DA_PB = DRIVE / "pseudobulk" / "human_integrated_control_da_leaf"
WHB = DRIVE / "human_whb_region_pseudobulk"

ANXA1_LEAVES = {"Sox6:Tafa1", "Sox6:Vcan"}


def bh(p):
    p = np.asarray(p, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]; m = pv.size; o = np.argsort(pv); r = pv[o] * m / (np.arange(m) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty(m); q[o] = np.minimum(r, 1); out[ok] = q
    return out


def build_populations(leaves):
    pops = {l: {l} for l in leaves}
    for fam in ("Sox6", "Calb1", "Gad2"):
        mem = {l for l in leaves if l.startswith(fam + ":")}
        if mem:
            pops[fam] = mem
    pops["Anxa1"] = {l for l in leaves if l in ANXA1_LEAVES}
    return {k: v for k, v in pops.items() if v}


def load_da():
    feats = pd.read_parquet(str(DA_PB) + ".features.parquet")
    meta = pd.read_parquet(str(DA_PB) + ".metadata.parquet").reset_index(drop=True)
    C = sp.load_npz(str(DA_PB) + ".counts.npz")
    detected = sp.load_npz(str(DA_PB) + ".detected_cells.npz")
    if C.shape[0] != len(meta):
        C = C.T
    if detected.shape[0] != len(meta):
        detected = detected.T
    return (
        np.asarray(C.todense(), float),
        np.asarray(detected.todense(), float),
        feats["feature_name"].astype(str).str.upper().to_numpy(),
        meta,
    )


def load_whb_reference():
    """One pooled reference profile per anatomy (Neurons + Nonneurons, all WHB donors)."""
    parts = []
    for p in ("Neurons", "Nonneurons"):
        md = pd.read_parquet(WHB / f"{p}.region_metadata.parquet").reset_index(drop=True)
        ft = pd.read_parquet(WHB / f"{p}.region_features.parquet")
        z = np.load(WHB / f"{p}.region_counts.npz")
        C = np.asarray(z["counts"], dtype=float)
        if C.shape[0] != len(md):
            C = C.T
        parts.append((C, ft["feature_name"].astype(str).str.upper().to_numpy(), md))
    genes = parts[0][1]
    assert (parts[1][1] == genes).all(), "WHB partitions have different feature order"
    anatomies = sorted(set(parts[0][2]["anatomy"]) | set(parts[1][2]["anatomy"]))
    ref_counts, ref_lib, ref_detected, ref_cells = {}, {}, {}, {}
    for a in anatomies:
        c = np.zeros(len(genes)); det = np.zeros(len(genes)); L = 0.0; cells = 0
        for Cd, _g, md in parts:
            m = md["anatomy"].eq(a).to_numpy()
            if m.any():
                partition = "Neurons" if md is parts[0][2] else "Nonneurons"
                z = np.load(WHB / f"{partition}.region_counts.npz")
                detected = np.asarray(z["detected"], dtype=float)
                if detected.shape[0] != len(md):
                    detected = detected.T
                c += Cd[m].sum(0)
                det += detected[m].sum(0)
                L += float(md.loc[m, "library_size"].sum())
                cells += int(md.loc[m, "n_cells"].sum())
        if L > 0:
            ref_counts[a] = c; ref_lib[a] = L
            ref_detected[a] = det; ref_cells[a] = cells
    return ref_counts, ref_lib, ref_detected, ref_cells, genes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Cda, Dda, da_genes, da_meta = load_da()
    ref_counts, ref_lib, ref_detected, ref_cells, whb_genes = load_whb_reference()
    sel = pd.read_parquet(REL / "target_population_selectivity.parquet")
    fda = set(sel[sel.species.eq("human")].human_gene.dropna().str.upper())

    common = sorted(set(da_genes) & set(whb_genes) & fda)
    print(f"gene space: DA {len(set(da_genes))}, WHB {len(set(whb_genes))}, FDA {len(fda)} -> common {len(common)}")
    di = {g: i for i, g in enumerate(da_genes)}
    wi = {g: i for i, g in enumerate(whb_genes)}
    dcols = np.array([di[g] for g in common]); wcols = np.array([wi[g] for g in common])

    leaves = set(da_meta["population_id"].unique())
    pops = build_populations(leaves)
    donors = sorted(da_meta["donor_id"].unique())
    print(f"DA donors used: {len(donors)} -> {donors}")
    anatomies = sorted(ref_counts)
    ref_l2 = {a: np.log2((ref_counts[a][wcols] + 0.5) / ref_lib[a] * 1e6) for a in anatomies}
    ref_detection = {a: ref_detected[a][wcols] / ref_cells[a] for a in anatomies}

    rows = []
    for pop, members in pops.items():
        foc = {}; focal_detection = {}; focal_cells = {}
        for d in donors:
            m = (da_meta["donor_id"].eq(d) & da_meta["population_id"].isin(members)).to_numpy()
            if not m.any():
                continue
            L = float(da_meta.loc[m, "library_size"].sum())
            n_cells = int(da_meta.loc[m, "n_cells"].sum())
            if L <= 0:
                continue
            foc[d] = np.log2((Cda[np.ix_(m, dcols)].sum(0) + 0.5) / L * 1e6)
            focal_detection[d] = Dda[np.ix_(m, dcols)].sum(0) / n_cells
            focal_cells[d] = n_cells
        if len(foc) < 3:
            continue
        F = np.vstack([foc[d] for d in sorted(foc)])
        FD = np.vstack([focal_detection[d] for d in sorted(foc)])
        n = F.shape[0]
        eff_all, se_all = [], []
        for a in anatomies:
            D = F - ref_l2[a][None, :]
            eff_all.append(D.mean(0))
            se_all.append(D.std(0, ddof=1) / np.sqrt(n))
        E = np.vstack(eff_all); S = np.vstack(se_all)
        k = np.argmin(E, axis=0)
        eff = E[k, np.arange(E.shape[1])]
        se = S[k, np.arange(S.shape[1])]
        focal_det = FD.mean(0)
        detdiff = focal_det - np.vstack([ref_detection[a] for a in anatomies])[
            k, np.arange(E.shape[1])
        ]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(se > 0, eff / se, np.nan)
        p = stats.t.sf(t, df=n - 1)
        rows.append(pd.DataFrame({
            "population_id": pop, "human_gene": common,
            "bw_effect_alldonors": eff, "bw_se_alldonors": se, "bw_p_alldonors": p,
            "bw_competitor_alldonors": [anatomies[i] for i in k],
            "bw_n_donors_alldonors": n,
            "bw_focal_n_cells_alldonors": sum(focal_cells.values()),
            "bw_focal_detection_alldonors": focal_det,
            "bw_detection_difference_alldonors": detdiff,
            "bw_common_donor_ids_alldonors": " | ".join(sorted(foc)),
            "bw_n_eligible_comparators_alldonors": len(anatomies),
        }))
    out = pd.concat(rows, ignore_index=True)
    out["bw_q_fda_alldonors"] = np.nan
    for pop, idx in out.groupby("population_id").groups.items():
        out.loc[idx, "bw_q_fda_alldonors"] = bh(out.loc[idx, "bw_p_alldonors"].to_numpy())
    dest = OUT / "brainwide_all_da_donors.parquet"
    out.to_parquet(dest, index=False)
    print(f"\nwrote {dest} ({len(out):,} rows, {out.population_id.nunique()} populations, n={out.bw_n_donors_alldonors.max()} donors)")
    return out


if __name__ == "__main__":
    main()
