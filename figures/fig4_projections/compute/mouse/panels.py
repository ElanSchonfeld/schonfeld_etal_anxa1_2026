"""Display matrix, super-bin group masses, and dorsolateral-ventromedial centres of mass for one mouse sender dataset."""
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from lib.display import GROUPS, members
from lib.paths import source_root, work_dir

REGIONS = ["CP", "ACB", "OT"]
C3 = {f"CP-{a}{b}": "CP" for a in "rmc" for b in "dv"}
C3.update({"ACB-core": "ACB", "ACB-shell": "ACB", "OT-med": "OT", "OT-lat": "OT"})
N_SUPERBIN, SEED = 97, 42
DATASETS = {
    "LRRK2": ("Data/Mouse/Mouse_LRRK2_Dopamine_RAW.h5ad", "subtype"),
    "Salmani": ("Data/Mouse/YaghmaeianSalmani/YaghmaeianSalmani_authorDA_predicted.h5ad",
                "pred_subtype_unified"),
}


def load(tag):
    d = work_dir("mouse")
    h5, col = DATASETS[tag]
    bl = pd.read_csv(d / f"allenFinal_bin_level_{tag}.tsv", sep="\t", index_col=0)
    bins = pd.read_csv(d / "allenFinal_receiver_bins.csv")
    counts = ad.read_h5ad(source_root() / h5, backed="r").obs[col].astype(str).value_counts().to_dict()
    if len(bl.index) != 16:
        raise ValueError(f"{tag}: {len(bl.index)} subtypes")
    return bl, bins, pd.Series({x: counts.get(x, 1.0) for x in bl.index}, dtype=float)


def display_matrix(bl, bins, w):
    sl = bins.slab_region.astype(str)
    reg = sl.map(C3)
    R = pd.DataFrame({c: np.column_stack([bl.loc[:, (sl == s).to_numpy()].mean(1)
                                          for s in sorted(sl[reg == c].unique())]).sum(1)
                      for c in REGIONS}, index=list(bl.index))
    fr = R.div(R.sum(1), axis=0)
    return pd.DataFrame({g: fr.loc[ms].mul(w.loc[ms], axis=0).sum(0) / w.loc[ms].sum()
                         for g, ms in members(bl.index).items()}).T[REGIONS]


def superbins(bl, bins, w):
    sl = bins.slab_region.astype(str)
    reg = sl.map(C3)
    keep = reg.notna().to_numpy()
    mass = pd.DataFrame({g: (bl.loc[ms].mul(w.loc[ms], axis=0).sum(0) / w.loc[ms].sum()).to_numpy()
                         for g, ms in members(bl.index).items()})
    mass["region"] = reg.to_numpy()
    mass["slab"] = sl.to_numpy()
    mass = mass[keep].reset_index(drop=True)
    xyz = bins.loc[keep, ["x_ccf", "y_ccf", "z_ccf"]].to_numpy()
    lab = np.full(len(mass), -1)
    nxt = 0
    for s in mass.slab.unique():
        idx = np.where(mass.slab.to_numpy() == s)[0]
        k = min(max(2, int(round(N_SUPERBIN * len(idx) / len(mass)))), len(idx))
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(xyz[idx])
        lab[idx] = km + nxt
        nxt += km.max() + 1
    mass["u"] = lab
    return mass.groupby(["u", "region"], as_index=False)[GROUPS].mean()


def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return np.zeros_like(v, float) if hi <= lo else (v - lo) / (hi - lo)


def dlvm_com(bl, bins):
    cp = bins.slab_region.astype(str).str.startswith("CP-").to_numpy()
    b = bins.loc[cp]
    dl = 0.5 * minmax(b.x_ccf.to_numpy()) + 0.5 * minmax(-b.y_ccf.to_numpy())
    S = bl.loc[:, [f"bin_{i}" for i in b.bin_id]].to_numpy()
    E = np.clip(S - S.mean(0, keepdims=True), 0, None)
    subs = list(bl.index)
    com = pd.DataFrame({"subtype": subs, "dlvm_com": (E @ dl) / E.sum(1)})
    com["group"] = np.where(com.subtype.isin(["Sox6:Tafa1", "Sox6:Vcan"]), "Anxa1+",
                            np.where(com.subtype.str.startswith("Sox6:"), "Sox6+ core", "Calb1+"))
    return com
