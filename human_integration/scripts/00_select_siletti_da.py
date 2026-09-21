"""Step 00 - Select dopaminergic neurons from the Siletti et al. 2023 human brain atlas, for the Kamath + Siletti human DA atlas.

Run:  python scripts/00_select_siletti_da.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
import cellxgene_census
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

SEED = 42
CENSUS_VERSION = "2023-12-15"
HERE = Path(__file__).resolve().parent.parent
DATA, FIG = HERE/"data", HERE/"figures"
DA_PROGRAM = ["TH", "DDC", "SLC6A3", "SLC18A2"]
np.random.seed(SEED)


def gmm_threshold(x):
    """2-component GMM on 1-D x -> crossover into the higher-mean component."""
    g = GaussianMixture(2, random_state=SEED, n_init=10).fit(x.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel()))
    grid = np.linspace(x.min(), x.max(), 4000)
    pr = g.predict(grid.reshape(-1, 1))
    thr = grid[np.where(pr == hi)[0][0]]
    return thr, g


def main():
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    mb = ds[ds.dataset_title.str.contains("Midbrain", na=False, regex=False)
            & ds.dataset_title.str.contains("Dissection", na=False, regex=False)
            & ds.dataset_title.str.contains("Substantia Nigra|Periaqueductal|Red Nucleus",
                                            na=False, regex=True)]
    ids = mb.dataset_id.tolist()
    id2t = {r.dataset_id: r.dataset_title.split(" - ")[-1] for _, r in mb.iterrows()}
    print(f"Census {CENSUS_VERSION}; midbrain DA-relevant dissections: {list(id2t.values())}")
    allobs = cellxgene_census.get_obs(census, "Homo sapiens",
        value_filter="dataset_id in " + str(ids), column_names=["cell_type"])
    allobs = allobs.to_pandas() if hasattr(allobs, "to_pandas") else pd.DataFrame(allobs)
    n_all = len(allobs); n_neu = int((allobs["cell_type"] == "neuron").sum())
    print(f"all cells={n_all:,}; neurons={n_neu:,} ({100*n_neu/n_all:.0f}%); glia/other={n_all-n_neu:,}")
    ad = cellxgene_census.get_anndata(
        census, "Homo sapiens",
        obs_value_filter="dataset_id in " + str(ids) + " and cell_type == 'neuron'",
        column_names={"obs": ["dataset_id", "cell_type", "raw_sum", "donor_id"]})
    census.close()
    ad.var_names = ad.var["feature_name"].astype(str).values; ad.var_names_make_unique()
    ad.obs["dissection"] = ad.obs["dataset_id"].map(id2t).astype(str)
    print(f"fetched {ad.n_obs:,} midbrain neurons x {ad.n_vars:,} genes")

    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)

    def gene(g): return ad[:, g].X.toarray().ravel() if g in ad.var_names else np.zeros(ad.n_obs)
    dbh, tph2, th, vmat = gene("DBH"), gene("TPH2"), gene("TH"), gene("SLC18A2")

    prog = [g for g in DA_PROGRAM if g in ad.var_names]
    M = np.column_stack([gene(g) for g in prog])
    s = M.mean(1)
    nmach = (M > 0).sum(1)
    ad.obs["DA_score"] = s

    thr, gm = gmm_threshold(s[s > 0])
    da_scored = (s > thr) & (nmach >= 2)

    tp = tph2[da_scored & (tph2 > 0)]
    tph2_thr = gmm_threshold(tp)[0] if len(tp) > 20 else 1.0
    dbh_thr = 1.0
    is_5ht = tph2 > tph2_thr; is_na = dbh > dbh_thr
    DA = da_scored & ~is_5ht & ~is_na
    strict = (th > 0.25) & (vmat > 0.25); recovered = DA & ~strict
    ad.obs["DA_score_pass"] = da_scored; ad.obs["is_DA"] = DA
    print(f"\nDA_score GMM thr={thr:.3f}; TPH2 thr={tph2_thr:.2f}")
    print(f"DA-scored={da_scored.sum()}  -NA({int((da_scored&is_na).sum())})  "
          f"-5HT({int((da_scored&is_5ht).sum())})  => FINAL DA = {DA.sum()}")
    print(f"  strict(TH+&VMAT2+)={int((DA&strict).sum())}  dropout-recovered={int(recovered.sum())}")
    print(f"  by dissection: {dict(ad.obs['dissection'][DA].value_counts())}")

    th_raw = ad.layers["counts"][:, ad.var_names.get_loc("TH")]
    th_raw = th_raw.toarray().ravel() if sp.issparse(th_raw) else np.asarray(th_raw).ravel()
    th_pos = th > 0
    th_no_mach = th_pos & ~da_scored
    th_5ht = th_pos & da_scored & is_5ht
    th_na  = th_pos & da_scored & (~is_5ht) & is_na
    th_da  = th_pos & DA
    da_th_drop = DA & ~th_pos
    cat = np.where(DA, "DA", np.where(da_scored & is_5ht, "5HT",
                   np.where(da_scored & is_na, "NA", "no-mach")))
    decomp = dict(n_all=n_all, n_neu=n_neu, th_pos=int(th_pos.sum()), th_hi=int((th_raw >= 3).sum()),
                  th_no_mach=int(th_no_mach.sum()), th_5ht=int(th_5ht.sum()), th_na=int(th_na.sum()),
                  th_da=int(th_da.sum()), da_th_drop=int(da_th_drop.sum()), cat=cat)
    print(f"TH+ ('~1800' candidate)={decomp['th_pos']} (TH>=3 counts, non-ambient={decomp['th_hi']}): "
          f"-no-machinery({decomp['th_no_mach']}) -5HT({decomp['th_5ht']}) -NA({decomp['th_na']}) "
          f"= TH+&DA({decomp['th_da']}); +TH-dropout-recovered({decomp['da_th_drop']}) => FINAL {DA.sum()}")

    out = ad[DA].copy(); out.X = out.layers["counts"].copy()
    for k in ["DA_score", "dissection", "donor_id"]: out.obs[k] = ad.obs[k].values[DA]
    del out.layers["counts"]
    out.write_h5ad(DATA/"siletti_da_neurons.h5ad")
    print(f"saved {out.n_obs} DA neurons -> {DATA/'siletti_da_neurons.h5ad'}")

    tf_genes = [g for g in ["NR4A2", "FOXA2", "LMX1A", "LMX1B", "EN1"] if g in ad.var_names]
    n_da_tf = (np.column_stack([gene(g) for g in tf_genes]) > 0).sum(1)
    tier = np.where(DA & strict, "1_strict_DA",
            np.where(DA & ~strict, "2_recovered_DA",
             np.where(cat == "5HT", "4_excl_serotonergic",
              np.where(cat == "NA", "5_excl_noradrenergic", "3_excl_no_machinery"))))
    thset = ad[th_pos].copy(); thset.X = thset.layers["counts"].copy()
    if "counts" in thset.layers: del thset.layers["counts"]
    for k, v in [("DA_score", s), ("nmach", nmach), ("n_DA_TF", n_da_tf), ("is_DA", DA),
                 ("tier", tier), ("TH_counts", th_raw), ("dissection", ad.obs["dissection"].values),
                 ("donor_id", ad.obs["donor_id"].values), ("raw_sum", ad.obs["raw_sum"].values)]:
        thset.obs[k] = np.asarray(v)[th_pos]
    thset.uns["machinery_score_threshold"] = float(thr)
    thset.uns["tph2_threshold"] = float(tph2_thr)
    thset.write_h5ad(DATA/"siletti_th_all.h5ad")
    print(f"saved {thset.n_obs} TH+ neurons (full, tiered) -> {DATA/'siletti_th_all.h5ad'}")
    print("  tier counts:", dict(pd.Series(tier[th_pos]).value_counts().sort_index()))

    au = ad.copy()
    sc.pp.highly_variable_genes(au, n_top_genes=2000)
    au = au[:, au.var.highly_variable].copy()
    sc.pp.scale(au, max_value=10); sc.pp.pca(au, n_comps=50, random_state=SEED)
    sc.pp.neighbors(au, n_neighbors=15, random_state=SEED); sc.tl.umap(au, random_state=SEED)
    U = au.obsm["X_umap"]

    make_figure(ad, s, gm, thr, tph2, tph2_thr, dbh, dbh_thr, da_scored, is_5ht, is_na,
                DA, strict, recovered, th, vmat, U, prog, decomp)
    print("DONE")


def make_figure(ad, s, gm, thr, tph2, tph2_thr, dbh, dbh_thr, da_scored, is_5ht,
                is_na, DA, strict, recovered, th, vmat, U, prog, decomp):
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10.5, "figure.dpi": 100})
    depth = ad.obs["raw_sum"].values.astype(float); diss = ad.obs["dissection"].values
    N = ad.n_obs; cat = decomp["cat"]; th_pos = th > 0
    tp, nm, h5, na = decomp["th_pos"], decomp["th_no_mach"], decomp["th_5ht"], decomp["th_na"]
    dr, fin = decomp["da_th_drop"], int(DA.sum())
    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("a  Provenance funnel", fontweight="bold", loc="left")
    steps = [("midbrain dissections (SN, PAG,\nRN) - all cell types", decomp["n_all"], "#3b6ea5"),
             (f"NEURONS  ({100*decomp['n_neu']/decomp['n_all']:.0f}%;\nrest = glia)", decomp["n_neu"], "#5a9bd4"),
             ("TH-expressing neurons\n(TH>0)  =  the “~1,800”", tp, "#e8934a"),
             (f"{fin} DOPAMINERGIC NEURONS\n(full dopamine phenotype)", fin, "#7a1f14")]
    notes = [f"−{decomp['n_all']-decomp['n_neu']:,}  glia / non-neuronal",
             f"−{decomp['n_neu']-tp:,}  no TH detected",
             f"−{tp-fin:,}  TH+ but not DA\n(ambient TH / no VMAT2·DAT / raphe / NA)"]
    y = 0.96
    for i, (lab, n, col) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((0.04, y-0.135), 0.72, 0.16, boxstyle="round,pad=0.008",
                                    fc=col, ec="none", alpha=.94, transform=ax.transAxes))
        ax.text(0.40, y-0.055, f"{lab}\n{n:,}", ha="center", va="center", fontsize=8.1,
                transform=ax.transAxes, color="white", fontweight="bold")
        if i < 3:
            ax.annotate("", xy=(0.40, y-0.20), xytext=(0.40, y-0.135),
                        arrowprops=dict(arrowstyle="-|>", color="#333"), transform=ax.transAxes)
            ax.text(0.80, y-0.165, notes[i], ha="left", va="center", fontsize=6.8,
                    color="#666", style="italic", transform=ax.transAxes)
        y -= 0.255

    ax = fig.add_subplot(gs[0, 1]); ax.set_title("b  dopamine-machinery score (bimodal)", fontweight="bold", loc="left")
    _, bins, _ = ax.hist(s, bins=150, color="#d9d9d9", label="all neurons")
    ax.hist(s[DA], bins=bins, color="#c0392b", alpha=.85, label=f"DA (n={fin})")
    ax.axvline(thr, color="k", ls="--", lw=1.8, label=f"GMM threshold {thr:.2f}  (+ ≥2 genes)")
    ax.set(xlabel="mean log-norm (TH, DDC, SLC6A3, VMAT2)", ylabel="neurons", yscale="log", ylim=(0.5, N*1.5))
    ax.legend(fontsize=7.5, frameon=False)
    ax.text(0.98, 0.55, "zero-spike =\nnon-expressers", transform=ax.transAxes, ha="right", fontsize=7, color="#888")

    ax = fig.add_subplot(gs[0, 2]); ax.set_title("c  Why 823, not the “~1,800”?", fontweight="bold", loc="left")
    r1, r2, r3 = tp-nm, tp-nm-h5, tp-nm-h5-na
    bars = [("TH+\nneurons", 0, tp, "#e8934a", f"{tp:,}"),
            ("− no DA\nmachinery", r1, nm, "#c9c9c9", f"−{nm:,}"),
            ("− sero-\ntonergic", r2, h5, "#72b7b2", f"−{h5:,}"),
            ("− nor-\nadren.", r3, na, "#b07aa1", f"−{na:,}"),
            ("+ TH-drop\nrecovered", r3, dr, "#4c78a8", f"+{dr:,}"),
            ("final\nDA", 0, fin, "#7a1f14", f"{fin:,}")]
    for i, (lab, bot, h, col, txt) in enumerate(bars):
        ax.bar(i, h, bottom=bot, color=col, edgecolor="#555", lw=.4, width=0.72)
        ax.text(i, bot+h+tp*0.012, txt, ha="center", va="bottom", fontsize=7.2,
                fontweight="bold" if i in (0, 5) else "normal")
    conn = [tp, r1, r2, r3, fin]
    for i in range(5):
        ax.plot([i+0.36, i+1-0.36], [conn[i], conn[i]], color="#999", lw=.7, ls="--")
    ax.plot([-0.42, 1.42], [decomp["th_hi"], decomp["th_hi"]], color="#444", ls=":", lw=1.1)
    ax.text(-0.38, decomp["th_hi"]+tp*0.02, f"TH≥3 counts (non-ambient) ≈ {decomp['th_hi']:,}",
            fontsize=6.4, color="#444", ha="left")
    ax.set_xticks(range(6)); ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set(ylabel="neurons"); ax.set_ylim(0, tp*1.16)
    ax.text(0.97, 0.90, "TH is necessary,\nnot sufficient", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.8, color="#7a1f14", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0]); ax.set_title("d  TH+ is not enough: must package (VMAT2)", fontweight="bold", loc="left")
    colmap = {"no-mach": "#c9c9c9", "5HT": "#72b7b2", "NA": "#b07aa1", "DA": "#c0392b"}
    labmap = {"no-mach": "TH+ no machinery", "5HT": "serotonergic", "NA": "noradrenergic", "DA": "DA (kept)"}
    jit = np.random.uniform(-0.03, 0.03, ad.n_obs)
    for gg in ["no-mach", "5HT", "NA", "DA"]:
        m = th_pos & (cat == gg)
        if not m.any(): continue
        ax.scatter(th[m]+jit[m], vmat[m], s=7, c=colmap[gg], lw=0,
                   alpha=.45 if gg == "no-mach" else .85, label=f"{labmap[gg]} ({int(m.sum())})")
    ax.set(xlabel="TH (log-norm)", ylabel="VMAT2 / SLC18A2 (log-norm)")
    ax.legend(frameon=False, fontsize=7, markerscale=1.6, loc="lower right")
    low = th_pos & (cat == "no-mach")
    pct_low = 100*np.mean(vmat[low] < 0.25) if low.any() else 0
    ax.text(0.03, 0.97, f"{pct_low:.0f}% of excluded TH+ have\nVMAT2≈ 0 → cannot package\ndopamine → not a DA neuron",
            transform=ax.transAxes, va="top", fontsize=7.3, color="#555")

    ax = fig.add_subplot(gs[1, 1]); ax.set_title("e  marker specificity", fontweight="bold", loc="left")
    grp = np.where(DA, "DA", np.where(is_5ht & da_scored, "5HT (rm)", "non-DA"))
    genes = prog + ["TPH2", "DBH", "GAD2"]
    order = ["non-DA", "DA", "5HT (rm)"]
    md = np.zeros((len(order), len(genes))); frac = np.zeros_like(md)
    for gi, g in enumerate(genes):
        v = ad[:, g].X.toarray().ravel() if g in ad.var_names else np.zeros(N)
        for oi, gg in enumerate(order):
            m = grp == gg; md[oi, gi] = v[m].mean() if m.any() else 0; frac[oi, gi] = (v[m] > 0).mean() if m.any() else 0
    mdn = md / (md.max(0) + 1e-9)
    for oi in range(len(order)):
        for gi in range(len(genes)):
            ax.scatter(gi, oi, s=frac[oi, gi]*150+5, c=[[plt.cm.Reds(mdn[oi, gi])]], edgecolor="#999", lw=.3)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5)
    ax.axvline(len(prog)-0.5, color="#bbb", ls=":"); ax.text(len(prog)-0.4, 2.4, "controls", fontsize=7, color="#888")
    ax.set_ylim(-0.6, 2.9); ax.margins(x=.04)
    ax.text(0.5, 1.04, "dot = % detected · colour = mean expr", transform=ax.transAxes, ha="center", fontsize=6.8, color="#888")

    ax = fig.add_subplot(gs[1, 2]); ax.set_title("f  selected DA neurons (coherent island)", fontweight="bold", loc="left")
    ax.scatter(U[~DA, 0], U[~DA, 1], c="#dddddd", s=3, lw=0, label=f"other neurons")
    ax.scatter(U[DA, 0], U[DA, 1], c="#c0392b", s=6, lw=0, label=f"DA (n={fin})")
    ax.set(xticks=[], yticks=[], xlabel="UMAP1", ylabel="UMAP2"); ax.legend(frameon=False, markerscale=2, fontsize=8)

    ax = fig.add_subplot(gs[2, 0]); ax.set_title("g  serotonergic removal (data-driven)", fontweight="bold", loc="left")
    ax.hist(tph2[da_scored], bins=55, color="#72b7b2"); ax.axvline(tph2_thr, color="red", ls="--", lw=1.8, label=f"antimode {tph2_thr:.2f}")
    ax.set(xlabel="TPH2 (log-norm) in DA-scored", ylabel="neurons", yscale="log"); ax.legend(fontsize=8, frameon=False)
    ax.text(0.4, 0.6, f"distinct 5HT mode\n= {int((da_scored&is_5ht).sum())} dorsal-raphe\nremoved", transform=ax.transAxes, fontsize=8, color="#2a6")

    ax = fig.add_subplot(gs[2, 1]); ax.set_title("h  dropout: recovered DA are low-depth", fontweight="bold", loc="left")
    bp = ax.boxplot([depth[DA & strict], depth[recovered], depth[~DA]], labels=["strict\nDA", "recovered\nDA", "non-DA"],
                    patch_artist=True, showfliers=False, widths=.6)
    for p, cl in zip(bp["boxes"], ["#4c78a8", "#f2a640", "#bbb"]): p.set_facecolor(cl)
    ax.set(ylabel="counts / cell", yscale="log")
    ax.text(0.02, 0.03, f"+{int(recovered.sum())} DA recovered\nfrom dropout (≥2 machinery\ngenes despite TH/other loss)", transform=ax.transAxes, fontsize=7.3, color="#c0392b")

    ax = fig.add_subplot(gs[2, 2]); ax.set_title(f"i  final DA neurons by dissection (n={fin})", fontweight="bold", loc="left")
    vc = pd.Series(diss[DA]).value_counts()[::-1]; ax.barh(range(len(vc)), vc.values, color="#4c78a8")
    ax.set_yticks(range(len(vc))); ax.set_yticklabels(vc.index, fontsize=8.5)
    for i, v in enumerate(vc.values): ax.text(v, i, f" {v}", va="center", fontsize=8.5)
    ax.set(xlabel="DA neurons"); ax.text(.97, .05, "SN-RN includes VTA", transform=ax.transAxes, fontsize=7.5, ha="right", color="#777")

    fig.suptitle(f"Selecting {fin} dopaminergic neurons from the Siletti human midbrain  "
                 f"(not the “~1,800” TH+ neurons: TH is necessary, not sufficient)",
                 fontsize=14.5, fontweight="bold", y=0.997)
    fig.savefig(FIG/"siletti_DA_selection.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"figure -> {FIG/'siletti_DA_selection.png'}")


if __name__ == "__main__":
    main()
