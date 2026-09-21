"""Step 00b - Build the Kamath + Siletti raw-count merge for integration."""
import warnings; warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc

HERE = Path(__file__).resolve().parent.parent
KAMATH = Path(os.environ.get(
    "DOPABASE_KAMATH_H5AD",
    HERE / "inputs" / "kamath" / "Human_Kamath_Dopamine.h5ad",
))
SILETTI = HERE / "data" / "siletti_da_neurons.h5ad"
OUT = HERE / "data" / "combined_raw.h5ad"
KEEP = ["study", "donor", "disease_status", "region", "family",
        "orig_subtype", "kamath_subtype", "siletti_subtype", "siletti_subcluster"]


def infer_family(s):
    u = str(s).upper()
    if "CALB1" in u: return "Calb1"
    if "SOX6" in u: return "Sox6"
    if "GAD" in u: return "Gad2"
    return "Other"


def collapse_dupe_symbols(a):
    names = a.var_names.astype(str)
    if names.is_unique: return a
    X = a.X.tocsc() if sp.issparse(a.X) else sp.csc_matrix(a.X)
    uniq = pd.Index(names).unique()
    cols = [X[:, np.where(names == g)[0]].sum(axis=1) if (names == g).sum() > 1
            else X[:, np.where(names == g)[0][0]] for g in uniq]
    out = ad.AnnData(X=sp.hstack([sp.csc_matrix(c) for c in cols]).tocsr(),
                     obs=a.obs.copy(), var=pd.DataFrame(index=uniq))
    print(f"      collapsed {a.n_vars} -> {out.n_vars} unique symbols")
    return out


def as_counts(a):
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    samp = X[:50].toarray()
    if not np.allclose(samp, np.round(samp)):
        raise ValueError("X is not integer counts; refusing to mix representations")
    a.X = X; return a


def main():
    print("[1/4] load Kamath")
    k = as_counts(ad.read_h5ad(KAMATH)); k = collapse_dupe_symbols(k)
    k.var_names = k.var_names.astype(str).str.upper()
    k.obs["study"] = "Kamath"
    k.obs["donor"] = k.obs["donor_id"].astype(str)
    k.obs["disease_status"] = k.obs["Status"].astype(str) if "Status" in k.obs else "Unknown"
    k.obs["region"] = "Midbrain (SNc/VTA, FACS DA)"
    k.obs["family"] = [infer_family(x) for x in k.obs["Cell_Type"].astype(str)]
    k.obs["orig_subtype"] = k.obs["Cell_Type"].astype(str)
    k.obs["kamath_subtype"] = k.obs["Cell_Type"].astype(str)
    k.obs["siletti_subtype"] = "NA"; k.obs["siletti_subcluster"] = "NA"
    print(f"      Kamath {k.shape}, subtypes={k.obs['kamath_subtype'].nunique()}")

    print("[2/4] load clean 823 Siletti DA (machinery-selected)")
    s = as_counts(ad.read_h5ad(SILETTI))
    s.var_names = s.var_names.astype(str).str.upper(); s = collapse_dupe_symbols(s)
    tmp = s.copy(); sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    def lv(g): return (tmp[:, g].X.toarray().ravel() if g in tmp.var_names else np.zeros(tmp.n_obs))
    M = np.column_stack([lv("SOX6"), lv("CALB1"), lv("GAD2")])
    fam = np.array(["Sox6", "Calb1", "Gad2"])[M.argmax(1)]; fam[M.max(1) == 0] = "Other"
    s.obs["study"] = "Siletti"
    s.obs["donor"] = s.obs["donor_id"].astype(str)
    s.obs["disease_status"] = "Control"
    s.obs["region"] = s.obs["dissection"].astype(str)
    s.obs["family"] = fam
    s.obs["orig_subtype"] = "Siletti_DA"
    s.obs["kamath_subtype"] = "NA"
    s.obs["siletti_subtype"] = "Siletti_DA"
    s.obs["siletti_subcluster"] = "NA"
    print(f"      Siletti {s.shape}; provisional family: {dict(pd.Series(fam).value_counts())}")
    print(f"      region (dissection): {dict(s.obs['region'].value_counts())}")

    print("[3/4] intersect shared symbols + concat")
    shared = k.var_names.intersection(s.var_names)
    print(f"      shared genes: {len(shared)} (Kamath {k.n_vars}, Siletti {s.n_vars})")
    k = k[:, shared].copy(); s = s[:, shared].copy()
    k.obs = k.obs[KEEP].copy(); s.obs = s.obs[KEEP].copy()
    comb = ad.concat([k, s], join="inner", label="study_src", keys=["Kamath", "Siletti"], index_unique="-")
    comb.layers["counts"] = comb.X.copy()
    comb.obs["study"] = comb.obs["study"].astype("category")
    comb.obs["donor"] = comb.obs["donor"].astype("category")
    print(f"      combined {comb.shape}; {dict(comb.obs['study'].value_counts())}; donors={comb.obs['donor'].nunique()}")

    print(f"[4/4] write {OUT}")
    comb.write_h5ad(OUT); print("DONE")


if __name__ == "__main__":
    main()
