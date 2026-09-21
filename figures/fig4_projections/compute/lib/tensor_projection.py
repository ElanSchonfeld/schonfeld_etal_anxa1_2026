"""Non-negative PARAFAC projection of sender subtypes onto spatial receiver bins with a ligand-variability filter."""
import numpy as np

from lib.parafac_pipeline import blend_family, fit_factors, reconstruction_candidates


def intra_family_r(projection, subtypes):
    families = {"Sox6": [], "Calb1": [], "Gad2": [], "Lef1": []}
    for i, s in enumerate(subtypes):
        for fam in families:
            if s.startswith(fam):
                families[fam].append(i)
                break
    corr = np.corrcoef(projection)
    fam_rs = []
    for idxs in families.values():
        if len(idxs) >= 2:
            fam_corr = corr[np.ix_(idxs, idxs)]
            upper = fam_corr[np.triu_indices_from(fam_corr, k=1)]
            upper = upper[~np.isnan(upper)]
            if len(upper) > 0:
                fam_rs.append(upper.mean())
    return np.mean(fam_rs) if fam_rs else 0.0


def tensor_factor_cp_projection(subtype_expr, sender_genes, recv_bin_expr, recv_genes, lr_pairs,
                                subtypes, min_lig_subtype_cv=0.3, n_factors=20, l1_receiver=0.05,
                                seed=42, force_method="top3"):
    S, K, M = subtype_expr.shape[0], len(lr_pairs), recv_bin_expr.shape[0]
    sender_idx = {g: i for i, g in enumerate(sender_genes)}
    recv_idx = {g: i for i, g in enumerate(recv_genes)}
    sender_lig = np.zeros((S, K), dtype=np.float32)
    recv_rec = np.zeros((M, K), dtype=np.float32)
    for i in range(K):
        sender_lig[:, i] = np.log1p(subtype_expr[:, sender_idx[lr_pairs.iloc[i]["ligand_complex"]]])
        recv_rec[:, i] = np.log1p(recv_bin_expr[:, recv_idx[lr_pairs.iloc[i]["receptor_complex"]]])

    lig_cv = np.zeros(K)
    for j in range(K):
        vals = sender_lig[:, j]
        if vals.mean() > 1e-6:
            lig_cv[j] = vals.std() / vals.mean()
    specific = lig_cv >= min_lig_subtype_cv
    sender_spec = sender_lig[:, specific]
    recv_spec = recv_rec[:, specific]
    spec_lr = lr_pairs.iloc[np.where(specific)[0]].reset_index(drop=True)
    Kp = int(specific.sum())

    T = np.zeros((Kp, S, M), dtype=np.float32)
    for k in range(Kp):
        T[k] = np.outer(sender_spec[:, k], recv_spec[:, k])
    for k in range(Kp):
        norm = np.linalg.norm(T[k])
        if norm > 1e-8:
            T[k] /= norm

    S_mat, R_mat, W, LR_mat = fit_factors(T, n_factors, l1_receiver, seed)
    candidates, factor_cvs = reconstruction_candidates(S_mat, W, R_mat, subtypes, intra_family_r)
    forced = [c for c in candidates if c[0] == force_method]
    best_name, _, best_proj = forced[0] if forced else candidates[0]
    projection_norm = best_proj / np.maximum(best_proj.sum(axis=1, keepdims=True), 1e-10)

    families = {"Sox6": [], "Calb1": [], "Gad2": []}
    for i, s in enumerate(subtypes):
        for fam in families:
            if s.startswith(fam):
                families[fam].append(i)
                break
    for fam, idxs in sorted(families.items()):
        n_fam = len(idxs)
        if n_fam < 2:
            continue
        fam_sender = sender_spec[idxs]
        fam_cv = np.zeros(Kp)
        for k in range(Kp):
            vals = fam_sender[:, k]
            if vals.mean() > 1e-6:
                fam_cv[k] = vals.std() / vals.mean()
        nonzero = fam_cv[fam_cv > 0.01]
        if len(nonzero) == 0:
            continue
        fam_lr_idx = np.where(fam_cv >= np.percentile(nonzero, 50))[0]
        fam_tensor = np.zeros((len(fam_lr_idx), n_fam, M), dtype=np.float32)
        for ki, k in enumerate(fam_lr_idx):
            fam_tensor[ki] = np.outer(fam_sender[:, k], recv_spec[:, k])
            norm = np.linalg.norm(fam_tensor[ki])
            if norm > 1e-8:
                fam_tensor[ki] /= norm
        blend_family(projection_norm, idxs, fam_tensor, n_fam, seed)

    projection_norm = projection_norm / np.maximum(projection_norm.sum(axis=1, keepdims=True), 1e-10)
    factors = {"sender": S_mat, "receiver": R_mat, "lr": LR_mat, "weights": W,
               "factor_cvs": factor_cvs, "lr_pairs": spec_lr, "method": best_name + "+hierarchical"}
    return projection_norm, factors
