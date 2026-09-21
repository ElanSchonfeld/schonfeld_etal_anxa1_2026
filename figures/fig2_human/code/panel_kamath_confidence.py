#!/usr/bin/env python3
"""Figure 2 (supp) - HMoE per-cell confidence on Kamath DA neurons (3 panels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                    # noqa: E402
from kamath_common import SUBTYPE_COLORS, infer_family   # noqa: E402

OUT = HERE.parent / "output" / "panels"
FROZEN = HERE.parent / "frozen"
ANX = ["Sox6:Tafa1", "Sox6:Vcan"]


def norm_entropy(P):
    P = np.clip(P / np.clip(P.sum(1, keepdims=True), 1e-12, None), 1e-12, 1.0)
    H = -(P * np.log(P)).sum(1)
    return H / np.log(P.shape[1])


def family_conf_pred(P_sub, subtypes):
    subs = np.array([str(s) for s in subtypes])
    sox = np.array([s.startswith("Sox6") for s in subs])
    cal = np.array([s.startswith("Calb1") for s in subs])
    p_sox = P_sub[:, sox].sum(1); p_cal = P_sub[:, cal].sum(1)
    tot = np.clip(p_sox + p_cal, 1e-12, None)
    p_sox, p_cal = p_sox / tot, p_cal / tot
    conf = np.maximum(p_sox, p_cal)
    pred = np.where(p_sox >= p_cal, "Sox6", "Calb1")
    return conf, pred


def reliability(conf, correct, nbins=12):
    edges = np.linspace(0.5, 1.0, nbins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if m.sum() >= 15:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


def clean(ax):
    ax.set_box_aspect(0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def roman_label(ax, txt):
    ax.text(-0.04, 1.06, txt, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def main():
    df = pd.read_parquet(FROZEN / "kamath_percell.parquet")
    df = df[~df["gad2_excluded"].values.astype(bool)].reset_index(drop=True)
    xy = df[["umap_x", "umap_y"]].to_numpy()
    conf = df["confidence"].to_numpy()
    sub = df["pred_subtype"].astype(str).to_numpy()
    obs_names = df["obs_name"].to_numpy()
    n = len(df)

    soft = pd.read_parquet(FROZEN / "kamath_hmoe_softprobs.parquet").set_index("obs_name")
    soft = soft.reindex(obs_names)
    subtypes = [c[2:] for c in soft.columns]
    P_sub = soft.to_numpy(dtype=float)
    ent = norm_entropy(P_sub)

    fig = plt.figure(figsize=(17.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, 0.95], wspace=0.28,
                          left=0.02, right=0.975, top=0.90, bottom=0.20)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = float(np.percentile(ent, 5)), float(np.percentile(ent, 95))
    rng = np.random.RandomState(42)
    o = rng.permutation(len(ent))
    sm = ax.scatter(xy[o, 0], xy[o, 1], c=ent[o], cmap="viridis", vmin=vmin, vmax=vmax,
                    s=6, rasterized=True, edgecolors="none")
    cax = ax.inset_axes([0.60, 0.96, 0.37, 0.03])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, vmax]); cb.set_ticklabels(["certain", "uncertain"])
    cax.tick_params(labelsize=8, pad=1)
    cb.set_label("HMoE uncertainty (norm. entropy, p5–p95)", fontsize=8, labelpad=2)
    clean(ax)
    ax.set_title("HMoE prediction uncertainty", fontweight="bold", fontsize=13)
    roman_label(ax, "i.")

    ax = fig.add_subplot(gs[0, 1])
    subs = sorted(set(sub), key=lambda s: (infer_family(s), s))
    data = [conf[sub == s] for s in subs]
    parts = ax.violinplot(data, positions=range(len(subs)), showmedians=True, widths=0.85)
    for b, s in zip(parts["bodies"], subs):
        b.set_facecolor(SUBTYPE_COLORS.get(s, "#bbb"))
        b.set_edgecolor("#08306b" if s in ANX else "none")
        b.set_linewidth(2.0 if s in ANX else 0); b.set_alpha(0.9)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("grey"); parts[key].set_linewidth(1.0)
    ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, rotation=90, fontsize=8)
    for tick, s in zip(ax.get_xticklabels(), subs):
        if s in ANX:
            tick.set_fontweight("bold"); tick.set_color("#08306b")
    ax.set_ylabel("HMoE confidence (max posterior)", fontsize=11)
    ax.set_ylim(0.25, 1.03)
    ax.set_title("Confidence by predicted subtype (Anxa1 outlined/bold)", fontweight="bold", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    roman_label(ax, "ii.")

    ax = fig.add_subplot(gs[0, 2])
    gt_fam = np.array([infer_family(c) for c in df["Cell_Type"].astype(str).to_numpy()])
    evalm = np.isin(gt_fam, ["Sox6", "Calb1"])

    hconf, hpred = family_conf_pred(P_sub, subtypes)
    hx, hy = reliability(hconf[evalm], (hpred[evalm] == gt_fam[evalm]).astype(float))
    ax.plot(hx, hy, "-o", color="#08519c", lw=2, ms=5, label="HMoE", zorder=3)

    sv = pd.read_csv(FROZEN / "scanvi_subtype_softprobs.csv").set_index("obs_name").reindex(obs_names)
    scols = [c for c in sv.columns if c.startswith("p_")]
    sv_subs = [c[2:] for c in scols]
    P_sv = sv[scols].to_numpy(dtype=float)
    sconf, spred = family_conf_pred(P_sv, sv_subs)
    ok = ~np.isnan(sconf) & evalm
    sx, sy = reliability(sconf[ok], (spred[ok] == gt_fam[ok]).astype(float))
    ax.plot(sx, sy, "-s", color="#e6550d", lw=2, ms=5, label="scANVI", zorder=3)

    ax.plot([0.5, 1.0], [0.5, 1.0], ":", color="grey", lw=1.2, label="perfect calibration")
    ax.set_xlim(0.5, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("family confidence (max prob)", fontsize=10.5)
    ax.set_ylabel("family accuracy (vs Cell_Type)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Reliability: is the confidence meaningful?", fontweight="bold", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    roman_label(ax, "iii.")

    figio.save_panel(fig, OUT / "kamath_hmoe_confidence")
    print(f"  -> kamath_hmoe_confidence  ({n:,} cells)")


if __name__ == "__main__":
    main()
