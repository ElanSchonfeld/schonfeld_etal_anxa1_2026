"""Raw-count sender pseudobulks per dopaminergic subtype for the two mouse datasets."""
import anndata as ad
import numpy as np
import scipy.sparse as sp


def lrrk2_pseudobulk(path, subtypes, sender_genes, ligands):
    data = ad.read_h5ad(path, backed="r")
    lookup = {str(g).upper(): i for i, g in enumerate(data.var_names)}
    wanted = [g for g in sender_genes if g in ligands and g.upper() in lookup]
    values = data[:, [lookup[g.upper()] for g in wanted]].X
    values = values.to_memory() if hasattr(values, "to_memory") else values
    values = values.toarray() if sp.issparse(values) else np.asarray(values)
    labels = data.obs["subtype"].astype(str).to_numpy()
    col = {g: i for i, g in enumerate(sender_genes)}
    result = np.zeros((len(subtypes), len(sender_genes)), dtype=np.float32)
    for i, subtype in enumerate(subtypes):
        mask = labels == subtype
        if mask.any():
            result[i, [col[g] for g in wanted]] = values[mask].mean(axis=0)
    data.file.close()
    return result


def salmani_pseudobulk(path, obs_col, sender_genes):
    a = ad.read_h5ad(path)
    names = {g.upper(): g for g in a.var_names}
    keep = [g for g in sender_genes if g.upper() in names]
    col = {g: j for j, g in enumerate(sender_genes)}
    X = a[:, [names[g.upper()] for g in keep]].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X, np.float32)
    pred = a.obs[obs_col].astype(str).values
    subs = sorted(set(pred))
    mat = np.zeros((len(subs), len(sender_genes)), np.float32)
    for i, s in enumerate(subs):
        pb = X[pred == s].mean(0)
        for j, g in enumerate(keep):
            mat[i, col[g]] = pb[j]
    return mat, subs
