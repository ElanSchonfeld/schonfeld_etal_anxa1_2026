"""Composite per-family driver tables: reconstruction percentile times communication magnitude, gated on reconstruction contribution."""
import numpy as np
import pandas as pd

from lib.paths import work_dir

GROUPS = ["Anxa1", "Sox6", "Calb1"]
DATASETS = [("mouse_LRRK2", True), ("salmani", True), ("macaque", False),
            ("human", False), ("kraft", False)]
INTRACELLULAR_LIGANDS = {
    "GNAI1", "GNAI2", "GNAI3", "GNAO1", "GNAS", "GNAQ", "GNA11", "GNA12",
    "GNA13", "GNAL", "GNAT1", "GNAT2", "GNB1", "GNB2", "GNB3", "GNG2",
    "ARF1", "ARF3", "ARF4", "ARF5", "ARF6",
    "HSP90AA1", "HSP90AB1", "HSPA5", "HSPA8", "CALR", "CALM1", "CALM2",
    "CALM3", "VCL", "LIN7C", "SHANK1", "SHANK2", "SHANK3", "ACTB", "ACTG1",
    "YWHAZ", "PRKACA", "RAC1", "RHOA", "CDC42",
}
INTRACELLULAR_RECEPTORS = {
    "ADCY1", "ADCY2", "ADCY3", "ADCY4", "ADCY5", "ADCY6", "ADCY7", "ADCY8",
    "ADCY9", "ADCY10", "GNAI2", "GNAS", "PRKACA",
}


def otsu_threshold(x):
    v = np.log10(np.clip(np.asarray(x, float), 1e-12, None))
    h, e = np.histogram(v, 256)
    p = h / h.sum()
    c = np.cumsum(p)
    mids = (e[:-1] + e[1:]) / 2
    mu = np.cumsum(p * mids)
    sb = (mu[-1] * c - mu) ** 2 / np.clip(c * (1 - c), 1e-12, None)
    return float(10 ** mids[np.argmax(sb)])


def bona_fide(lig, rec):
    return (str(lig).upper() not in INTRACELLULAR_LIGANDS
            and str(rec).upper() not in INTRACELLULAR_RECEPTORS)


def driver_table(key, mouse_case):
    d = work_dir("drivers")
    src = pd.read_csv(d / "perfamily_magnitude.csv")
    t = src[src.dataset == key].copy()
    rc = pd.read_csv(d / f"{key}_recon.csv")
    recon = {(str(l).upper(), str(r).upper()): p for l, r, p in zip(rc.ligand, rc.receptor, rc.recon_pct)}
    raw = {(str(l).upper(), str(r).upper()): v for l, r, v in zip(rc.ligand, rc.receptor, rc.recon)}
    keys = list(zip(t.ligand.astype(str).str.upper(), t.receptor.astype(str).str.upper()))
    t["recon_pct"] = [recon.get(k, np.nan) for k in keys]
    t["recon_raw"] = [raw.get(k, np.nan) for k in keys]
    t = t.dropna(subset=["recon_pct", "recon_raw"])
    t = t[t.recon_raw >= otsu_threshold(t.recon_raw.to_numpy())]
    t["composite"] = t.recon_pct / 100.0 * t["mag"]
    piv = t.pivot_table(index=["ligand", "receptor"], columns="family",
                        values="composite", fill_value=0.0).reset_index()
    for g in GROUPS:
        if g not in piv.columns:
            piv[g] = 0.0
    piv = piv[[bona_fide(l, r) for l, r in zip(piv.ligand, piv.receptor)]].reset_index(drop=True)
    if mouse_case:
        piv["ligand"] = piv.ligand.str.capitalize()
        piv["receptor"] = piv.receptor.str.capitalize()
    return piv[["ligand", "receptor"] + GROUPS].assign(dataset=key)
