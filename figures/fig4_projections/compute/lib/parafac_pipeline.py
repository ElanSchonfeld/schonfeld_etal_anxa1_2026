"""Tensor construction, non-negative PARAFAC projection, and bin-to-region aggregation for pseudobulk senders and receiver bins."""
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac


def build_tensor(sender_expr, sender_genes, recv_expr, recv_genes, lr_pairs):
    S, M, K = sender_expr.shape[0], recv_expr.shape[0], len(lr_pairs)
    sender_idx = {g: i for i, g in enumerate(sender_genes)}
    recv_idx = {g: i for i, g in enumerate(recv_genes)}
    sender_lig = np.zeros((S, K), dtype=np.float32)
    recv_rec = np.zeros((M, K), dtype=np.float32)
    for k in range(K):
        lig = lr_pairs.iloc[k]["ligand_gene"]
        rec = lr_pairs.iloc[k]["receptor_gene"]
        if lig in sender_idx:
            sender_lig[:, k] = np.log1p(sender_expr[:, sender_idx[lig]])
        if rec in recv_idx:
            recv_rec[:, k] = np.log1p(recv_expr[:, recv_idx[rec]])
    T = np.zeros((K, S, M), dtype=np.float32)
    for k in range(K):
        T[k] = np.outer(sender_lig[:, k], recv_rec[:, k])
    for k in range(K):
        norm = np.linalg.norm(T[k])
        if norm > 1e-8:
            T[k] /= norm
    return T


def family_of(subtype):
    for fam in ["Sox6", "Calb1", "Gad2"]:
        if subtype.startswith(fam):
            return fam
    return "Unknown"


def intra_family_r(projection, subtypes):
    families = {"Sox6": [], "Calb1": []}
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


def reconstruction_candidates(S_mat, W, R_mat, subtypes, intra_r=intra_family_r):
    n_factors = S_mat.shape[1]
    S, M = S_mat.shape[0], R_mat.shape[0]
    factor_cvs = np.zeros(n_factors)
    for r in range(n_factors):
        vals = S_mat[:, r]
        if vals.mean() > 1e-8:
            factor_cvs[r] = vals.std() / vals.mean()
    candidates = []
    proj_full = S_mat @ np.diag(W) @ R_mat.T
    candidates.append(("full", intra_r(proj_full, subtypes), proj_full))
    nonzero_cvs = factor_cvs[factor_cvs > 0.01]
    if len(nonzero_cvs) > 0:
        diff_mask = factor_cvs >= np.median(nonzero_cvs)
        proj_diff = np.zeros((S, M))
        for r in range(n_factors):
            if diff_mask[r]:
                proj_diff += W[r] * np.outer(S_mat[:, r], R_mat[:, r])
        candidates.append((f"differential ({int(diff_mask.sum())} factors)",
                           intra_r(proj_diff, subtypes), proj_diff))
    for alpha in [2, 3]:
        proj_sharp = ((S_mat * W[None, :]) ** alpha) @ R_mat.T
        candidates.append((f"sharp_a{alpha}", intra_r(proj_sharp, subtypes), proj_sharp))
    for topk in [3, 5]:
        proj_topk = np.zeros((S, M))
        for i in range(S):
            sw = S_mat[i, :] * W
            for fi in np.argsort(sw)[-topk:][::-1]:
                proj_topk[i] += sw[fi] * R_mat[:, fi]
        candidates.append((f"top{topk}", intra_r(proj_topk, subtypes), proj_topk))
    candidates.sort(key=lambda x: x[1])
    return candidates, factor_cvs


def blend_family(projection_norm, idxs, fam_tensor, fam_rank, seed):
    n_fam = len(idxs)
    M = projection_norm.shape[1]
    cp_fam, _ = non_negative_parafac(
        tl.tensor(fam_tensor), rank=fam_rank, init="random", random_state=seed,
        n_iter_max=200, return_errors=True)
    w_f, (_, s_f, r_f) = cp_fam
    s_f, r_f, w_f = np.array(s_f), np.array(r_f), np.array(w_f)
    fam_profiles = ((s_f * w_f[None, :]) ** 2) @ r_f.T
    fam_profiles_norm = fam_profiles / np.maximum(fam_profiles.sum(axis=1, keepdims=True), 1e-10)
    best_alpha, best_fam_ir = 0.5, 1.0
    for alpha in [0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
        blended = np.zeros((n_fam, M))
        for fi, gi in enumerate(idxs):
            blended[fi] = alpha * projection_norm[gi] + (1 - alpha) * fam_profiles_norm[fi]
        upper = np.corrcoef(blended)[np.triu_indices(n_fam, k=1)]
        upper = upper[~np.isnan(upper)]
        ir = float(upper.mean()) if len(upper) > 0 else 0.0
        if ir < best_fam_ir:
            best_fam_ir, best_alpha = ir, alpha
    for fi, gi in enumerate(idxs):
        projection_norm[gi] = best_alpha * projection_norm[gi] + (1 - best_alpha) * fam_profiles_norm[fi]


def hierarchical_recon(T, projection_norm, subtypes, seed=42):
    T = np.asarray(T)
    K, S, M = T.shape
    projection_norm = projection_norm.copy()
    families = {"Sox6": [], "Calb1": []}
    for i, s in enumerate(subtypes):
        fam = family_of(s)
        if fam in families:
            families[fam].append(i)
    for fam, idxs in sorted(families.items()):
        n_fam = len(idxs)
        if n_fam < 2:
            continue
        fam_T = T[:, idxs, :]
        fam_sender_means = np.zeros((n_fam, K))
        for k in range(K):
            fam_sender_means[:, k] = fam_T[k].sum(axis=1)
        fam_cv = np.zeros(K)
        for k in range(K):
            vals = fam_sender_means[:, k]
            if vals.mean() > 1e-6:
                fam_cv[k] = vals.std() / vals.mean()
        nonzero_fam_cv = fam_cv[fam_cv > 0.01]
        if len(nonzero_fam_cv) == 0:
            continue
        fam_lr_idx = np.where(fam_cv >= np.percentile(nonzero_fam_cv, 50))[0]
        fam_tensor = np.zeros((len(fam_lr_idx), n_fam, M), dtype=np.float32)
        for ki, k in enumerate(fam_lr_idx):
            fam_tensor[ki] = fam_T[k, :, :]
            norm = np.linalg.norm(fam_tensor[ki])
            if norm > 1e-8:
                fam_tensor[ki] /= norm
        fam_rank = min(n_fam, len(fam_lr_idx))
        if fam_rank < 2:
            continue
        blend_family(projection_norm, idxs, fam_tensor, fam_rank, seed)
    return projection_norm / np.maximum(projection_norm.sum(axis=1, keepdims=True), 1e-10)


def fit_factors(T, n_factors, l1_receiver, seed):
    cp_result, _ = non_negative_parafac(
        tl.tensor(T), rank=n_factors, init="random", random_state=seed,
        n_iter_max=300, return_errors=True)
    weights_cp, (lr_factors, sender_factors, recv_factors) = cp_result
    if l1_receiver > 0:
        threshold = l1_receiver * float(np.max(recv_factors))
        recv_factors = np.maximum(np.array(recv_factors) - threshold, 0)
    return np.array(sender_factors), np.array(recv_factors), np.array(weights_cp), np.array(lr_factors)


def run_parafac_pipeline(T, subtypes, n_factors=20, l1_receiver=0.05, seed=42):
    S_mat, R_mat, W, LR_mat = fit_factors(T, n_factors, l1_receiver, seed)
    candidates, factor_cvs = reconstruction_candidates(S_mat, W, R_mat, subtypes)
    best_name, _, best_proj = candidates[0]
    projection_norm = best_proj / np.maximum(best_proj.sum(axis=1, keepdims=True), 1e-10)
    projection_norm = hierarchical_recon(T, projection_norm, subtypes, seed)
    factors = {"sender": S_mat, "receiver": R_mat, "lr": LR_mat, "weights": W,
               "factor_cvs": factor_cvs, "method": best_name + "+hierarchical"}
    return projection_norm, factors


def aggregate_to_regions(projection_norm, subtypes, bin_meta):
    regions = bin_meta["region"].values
    unique_regions = sorted(set(regions))
    region_proj = np.zeros((len(subtypes), len(unique_regions)), dtype=np.float64)
    for j, reg in enumerate(unique_regions):
        region_proj[:, j] = projection_norm[:, regions == reg].mean(axis=1)
    region_norm = region_proj / np.maximum(region_proj.sum(axis=1, keepdims=True), 1e-10)
    region_df = pd.DataFrame(region_norm, index=subtypes, columns=unique_regions)
    row_means = region_norm.mean(axis=1, keepdims=True)
    row_stds = np.maximum(region_norm.std(axis=1, keepdims=True), 1e-10)
    region_z = pd.DataFrame((region_norm - row_means) / row_stds, index=subtypes, columns=unique_regions)
    return region_df, region_z
