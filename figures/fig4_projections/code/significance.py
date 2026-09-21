#!/usr/bin/env python3
"""Significance test for the projection panels."""
import numpy as np
import pandas as pd

N_PERM, SEED = 5000, 42
N_BOOT, BOOT_SEED = 2000, 43
GROUPS = ["Anxa1-associated", "Sox6 family", "Calb1 family"]


def _deltas(mass, region_codes, n_regions):
    """Mass (3 x n_bins), region_codes (n_bins,) -> delta (3 x n_regions)."""
    ng = mass.shape[0]
    M = np.vstack([np.bincount(region_codes, weights=np.asarray(mass[f], dtype=float),
                               minlength=n_regions) for f in range(ng)])
    tot = M.sum(1, keepdims=True)
    frac = np.zeros_like(M, dtype=float)
    np.divide(M, tot, out=frac, where=tot > 0)
    others = (frac.sum(0, keepdims=True) - frac) / (ng - 1)
    return frac - others


def _bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def compute_table(bins, regions):
    """Bins: DataFrame with `region` + the three GROUPS mass columns. regions: ordered region list."""
    b = bins[bins["region"].isin(regions)].reset_index(drop=True)
    mass = b[GROUPS].to_numpy(dtype=float).T
    codes = pd.Categorical(b["region"], categories=regions).codes.astype(int)
    nt, (ng, nb) = len(regions), mass.shape
    obs = _deltas(mass, codes, nt)

    rng = np.random.default_rng(SEED)
    more = np.zeros_like(obs, dtype=int)
    cols = np.arange(nb)
    for _ in range(N_PERM):
        perm = np.array([rng.permutation(ng) for _ in range(nb)], dtype=int).T
        nd = _deltas(mass[perm, cols], codes, nt)
        more += np.where(obs >= 0, nd >= obs - 1e-12, nd <= obs + 1e-12)
    p_perm = (more + 1) / (N_PERM + 1)

    brng = np.random.default_rng(BOOT_SEED)
    boot = np.empty((N_BOOT, ng, nt))
    idx = np.arange(nb)
    for i in range(N_BOOT):
        bi = brng.choice(idx, size=nb, replace=True)
        boot[i] = _deltas(mass[:, bi], codes[bi], nt)
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)

    rows = []
    for f, group in enumerate(GROUPS):
        q = _bh(p_perm[f])
        for t, region in enumerate(regions):
            rows.append({"group": group, "region": region, "delta": obs[f, t],
                         "p_perm": p_perm[f, t], "q": q[t],
                         "ci_lo": lo[f, t], "ci_hi": hi[f, t],
                         "significant": bool(q[t] < 0.05)})
    return pd.DataFrame(rows)


def compute(bins, regions):
    """Bins: DataFrame with `region` + the three GROUPS mass columns. regions: ordered region list."""
    d = compute_table(bins, regions)
    return {(r.group, r.region): bool(r.significant) for r in d.itertuples()}
