#!/usr/bin/env python3
"""Compute the Figure 4 mouse statistics from the frozen tables in frozen/stats/.

Run: python figures/fig4_projections/code/manuscript_stats.py
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np, pandas as pd

import significance as SIG

FROZEN = Path(__file__).resolve().parent.parent / "frozen"
STATS = FROZEN / "stats"
REGIONS = ["CP", "ACB", "OT"]
GROUPS = SIG.GROUPS
PAIR = ["Sox6 family", "Calb1 family"]
ORDER = ["Calb1+", "Sox6+ core", "Anxa1+"]
PANEL = {"LRRK2": "panelB_mouse", "Salmani": "panelD_salmani"}


def gradient(tag):
    com = pd.read_csv(STATS / f"{tag}_dlvm_com.csv")
    v = com.dlvm_com.to_numpy()
    idx = {g: np.where(com.group.to_numpy() == g)[0] for g in ORDER}
    sizes = [len(idx[g]) for g in ORDER]

    M = (v[None, :] > v[:, None]).astype(float) + 0.5 * (v[None, :] == v[:, None])
    np.fill_diagonal(M, 0.0)

    def jt(g0, g1, g2):
        return (M[np.ix_(g0, g1)].sum() + M[np.ix_(g0, g2)].sum() + M[np.ix_(g1, g2)].sum())

    obs = jt(idx[ORDER[0]], idx[ORDER[1]], idx[ORDER[2]])
    jt_max = sizes[0] * sizes[1] + sizes[0] * sizes[2] + sizes[1] * sizes[2]

    all_i = np.arange(len(v))
    n_ge = n_tot = 0
    for c0 in itertools.combinations(all_i, sizes[0]):
        rest = np.setdiff1d(all_i, c0, assume_unique=True)
        for c1 in itertools.combinations(rest, sizes[1]):
            c2 = np.setdiff1d(rest, c1, assume_unique=True)
            n_tot += 1
            if jt(np.array(c0), np.array(c1), c2) >= obs - 1e-12:
                n_ge += 1
    means = {g: float(v[idx[g]].mean()) for g in ORDER}
    return dict(means=means, jt=obs, jt_max=jt_max, p=n_ge / n_tot, n_perm=n_tot,
                monotone=means["Anxa1+"] > means["Sox6+ core"] > means["Calb1+"])


def _fracs(mass, codes, nt):
    M = np.vstack([np.bincount(codes, weights=mass[f], minlength=nt) for f in range(mass.shape[0])])
    tot = M.sum(1, keepdims=True)
    fr = np.zeros_like(M, float)
    np.divide(M, tot, out=fr, where=tot > 0)
    return fr


def targets(tag):
    agg = pd.read_csv(STATS / f"{tag}_superbins.csv")
    b = agg[agg.region.isin(REGIONS)].reset_index(drop=True)
    codes = pd.Categorical(b.region, categories=REGIONS).codes.astype(int)
    nt, nb = len(REGIONS), len(b)

    mass = b[PAIR].to_numpy(float).T
    obs = _fracs(mass, codes, nt)
    d_obs = obs[0] - obs[1]

    rng = np.random.default_rng(SIG.SEED)
    more = np.zeros(nt, int)
    for _ in range(SIG.N_PERM):
        swap = rng.random(nb) < 0.5
        mp = mass.copy()
        mp[0, swap], mp[1, swap] = mass[1, swap], mass[0, swap]
        f = _fracs(mp, codes, nt)
        d = f[0] - f[1]
        more += np.where(d_obs >= 0, d >= d_obs - 1e-12, d <= d_obs + 1e-12)
    p = (more + 1) / (SIG.N_PERM + 1)

    brng = np.random.default_rng(SIG.SEED + 1)
    boot = np.empty((SIG.N_BOOT, nt))
    for i in range(SIG.N_BOOT):
        bi = brng.choice(np.arange(nb), size=nb, replace=True)
        f = _fracs(mass[:, bi], codes[bi], nt)
        boot[i] = f[0] - f[1]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    q = SIG._bh(p)
    return pd.DataFrame(dict(region=REGIONS, sox6_share=obs[0], calb1_share=obs[1],
                             delta=d_obs, pp=d_obs * 100, fold=obs[0] / obs[1],
                             p=p, q=q, ci_lo=lo, ci_hi=hi,
                             z=d_obs / ((hi - lo) / (2 * 1.959964)),
                             significant=((lo > 0) | (hi < 0)) & (q < 0.05))), agg


def effect_sizes(tag):
    """Reconcile the raw cell2cell shares with the heatmap's +/-1 display scale."""
    mat = pd.read_csv(FROZEN / f"{PANEL[tag]}_matrix.csv", index_col=0).loc[GROUPS, REGIONS]
    centred = mat - mat.mean(0)
    vmax = float(np.abs(centred.to_numpy()).max())
    display = centred / vmax
    dorsal_bias = mat["CP"] / (mat["ACB"] + mat["OT"])
    return dict(shares=mat, centred_pp=centred * 100, vmax_pp=vmax * 100,
                display=display, dorsal_bias=dorsal_bias)


def check_frozen(tag, agg):
    sig = SIG.compute(agg[["region"] + GROUPS], REGIONS)
    fz = pd.read_csv(FROZEN / f"{PANEL[tag]}_sig.csv")
    bad = [(r.group, r.region) for r in fz.itertuples()
           if bool(r.significant) != sig[(r.group, r.region)]]
    return len(fz) - len(bad), len(fz), bad


def main():
    for tag in ("LRRK2", "Salmani"):
        role = "Figure 4a/b" if tag == "LRRK2" else "Supplementary Figure 4a"
        print("=" * 84)
        print(f"MOUSE {tag}   ({role})")
        print("=" * 84)

        g = gradient(tag)
        print("\n1. DORSOLATERAL-TO-VENTROMEDIAL GRADIENT  (Jonckheere-Terpstra, exact)")
        print("   DL-VM centre of mass, 1 = dorsolateral / 0 = ventromedial "
              "(axes min-max normalised within CP):")
        for grp in ("Anxa1+", "Sox6+ core", "Calb1+"):
            print(f"     {grp:12s} {g['means'][grp]:.4f}")
        print(f"   monotone Anxa1+ > Sox6+ core > Calb1+ : {g['monotone']}")
        print(f"   JT = {g['jt']:.0f}/{g['jt_max']}   "
              f"EXACT P = {g['p']:.4f}   ({g['n_perm']:,} assignments enumerated)")

        t, agg = targets(tag)
        print("\n2. TARGET CONTRASTS  (Sox6 family vs Calb1 family, ~100 super-bins)")
        print(t[["region", "sox6_share", "calb1_share", "pp", "fold", "p", "q",
                 "significant"]].round(4).to_string(index=False))

        e = effect_sizes(tag)
        print("\n3. EFFECT-SIZE LADDER")
        print("   a) share of each family's projection mass (raw cell2cell output, %):")
        print((e["shares"] * 100).round(2).to_string())
        print("\n   b) column-centred family bias, PERCENTAGE POINTS  <- recommended for text:")
        print(e["centred_pp"].round(2).to_string())
        print(f"\n   c) heatmap display value (= b / {e['vmax_pp']:.2f} pp; +/-1.00 by construction):")
        print(e["display"].round(2).to_string())
        print("\n   d) dorsal:ventral projection bias, CP / (ACB + OT):")
        for grp in GROUPS:
            print(f"     {grp:20s} {e['dorsal_bias'][grp]:.3f}")
        r = e["dorsal_bias"]["Sox6 family"] / e["dorsal_bias"]["Calb1 family"]
        print(f"     Sox6 : Calb1 ratio of ratios = {r:.3f}x")

        ok, n, bad = check_frozen(tag, agg)
        print(f"\n   self-check vs frozen {PANEL[tag]}_sig.csv: {ok}/{n} match"
              + (f"  MISMATCH {bad}" if bad else "  OK"))
        print()


if __name__ == "__main__":
    main()
