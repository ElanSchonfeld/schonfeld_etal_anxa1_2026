#!/usr/bin/env python3
"""Freeze the SMALL derived inputs that the drive-free code/ render scripts read.

Run: PPMI_ROOT=/path/to/authorized/ppmi python freeze_fig5_inputs.py
"""
import os, sys, json, shutil, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection_fields import (
    GROUP_ORDER,
    build_hmba_projection_frame,
    resample_group_fields_to_fsl,
)

FROZEN = Path(__file__).resolve().parent.parent / "frozen"
SMASH_CSV = FROZEN / "bilateral_variogram_smash.csv"
GORD = list(GROUP_ORDER)


def _required_ppmi_root():
    value = os.environ.get("PPMI_ROOT")
    if not value:
        raise RuntimeError("PPMI_ROOT must point to an authorized local PPMI data root")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PPMI_ROOT is not a directory: {root}")
    return root


def _weighted_depletion(frame, group, depletion, mask):
    weights = np.clip(frame.centered_fields[group], 0, None) * mask
    selected = (weights > 0) & np.isfinite(depletion)
    return np.sum(weights[selected] * depletion[selected]) / np.sum(weights[selected])


MIN_STAGE_N = 3


def _load_axes(npz_path, frame):
    cz = np.load(npz_path, allow_pickle=True); axes = {}
    for sk in ["time", "hy", "updrs"]:
        deps = {l.split('|')[0]: frame.to_h(d) for l, d in cz[f"deps_{sk}"]
                if int(l.split('n=')[1]) >= MIN_STAGE_N}
        order = [s for s in cz[f"stages_{sk}"] if s in deps]
        axes[sk] = (order, deps)
    return axes


def _rainbow_frame(frame, axes, rows):
    """Rows = list of (rowname, mask)."""
    allv = [_weighted_depletion(frame, g, axes[sk][1][s], mask) for _, mask in rows for sk in ["time", "hy", "updrs"]
            for g in GORD for s in axes[sk][0]]
    vmin, vmax = float(np.percentile(allv, 5)), float(np.percentile(allv, 95))
    recs = []
    for rowname, mask in rows:
        for sk in ["time", "hy", "updrs"]:
            order, deps = axes[sk]
            for g in GORD:
                for s in order:
                    recs.append(dict(row=rowname, group=g, scheme=sk, stage=s,
                                     value=float(_weighted_depletion(frame, g, deps[s], mask))))
    return pd.DataFrame(recs), vmin, vmax


def freeze_rainbows(frame, ds):
    putamen, caudate = frame.putamen, frame.caudate
    dorsal = putamen | caudate
    size_x = frame.striatum.shape[0]
    x_mm = frame.affine[0, 0] * np.arange(size_x)[:, None, None] + frame.affine[0, 3]
    put_left, put_right = putamen & (x_mm < 0), putamen & (x_mm > 0)
    caud_left, caud_right = caudate & (x_mm < 0), caudate & (x_mm > 0)

    ax_bi = _load_axes(ds / "fig12_staging_all_depmaps.npz", frame)
    df, vmin, vmax = _rainbow_frame(
        frame, ax_bi,
        [("Overall (dorsal)", dorsal), ("Putamen", putamen), ("Caudate", caudate)],
    )
    df[df.row == "Overall (dorsal)"].to_csv(FROZEN / "staging_rainbow_matrix.csv", index=False)
    json.dump({"vmin": vmin, "vmax": vmax, "groups": GORD}, open(FROZEN / "staging_rainbow_scale.json", "w"), indent=0)
    print(f"  staging_rainbow: {df.row.eq('Overall (dorsal)').sum()} cells  scale [{vmin:.0f},{vmax:.0f}]")

    ax_h = _load_axes(ds / "fig12_staging_all_depmaps_hemi.npz", frame)
    rows_h = [("MA Overall\n(dorsal)", put_left | caud_left),
              ("LA Overall\n(dorsal)", put_right | caud_right),
              ("MA Putamen", put_left), ("LA Putamen", put_right),
              ("MA Caudate", caud_left), ("LA Caudate", caud_right)]
    dfh, vminh, vmaxh = _rainbow_frame(frame, ax_h, rows_h)
    dfh.to_csv(FROZEN / "hemi_rainbow_matrix.csv", index=False)
    json.dump({"vmin": vminh, "vmax": vmaxh, "groups": GORD,
               "row_order": [r for r, _ in rows_h]}, open(FROZEN / "hemi_rainbow_scale.json", "w"), indent=0)
    print(f"  hemi_rainbow: {len(dfh)} cells  scale [{vminh:.0f},{vmaxh:.0f}]")


def _block_boots_mm(mm, nb=500, seed=42):
    """6mm-block bootstrap indices from explicit mm coordinates (grid-agnostic: 1mm H or FSL 2mm)."""
    bk = np.floor(mm / 6.0).astype(np.int64); bkey = bk[:, 0] * 100003 + bk[:, 1] * 317 + bk[:, 2]
    ublk = np.unique(bkey); idxblk = [np.where(bkey == b)[0] for b in ublk]; rng = np.random.default_rng(seed)
    return [np.concatenate([idxblk[p] for p in rng.integers(0, len(ublk), len(ublk))]) for _ in range(nb)]


def _block_boots(strv, affine, nb=500, seed=42):
    ijk = np.array(np.where(strv)).T; mm = (affine[:3, :3] @ ijk.T + affine[:3, 3:]).T
    return _block_boots_mm(mm, nb, seed)


def _reduced_curves(frame, strv, dep_full, pvals):
    """Reduced curves on the 1mm H frame (used by the hemi scatter)."""
    return _curves_from_arrays(
        dep_full[strv],
        {g: frame.noncentered_fields[g][strv] for g in GORD},
        _block_boots(strv, frame.affine),
        pvals,
    )


def _curves_from_arrays(dv, ev, boots, pvals):
    """Return dict of per-group reduced curves (fit endpoints, band xs/lo/hi, decile x/y, rho, p)."""
    xs = np.linspace(0, 100, 50); out = {"band_xs": xs}
    for g in GORD:
        e = ev[g]; ok = np.isfinite(e) & np.isfinite(dv); e, d = e[ok], dv[ok]
        o = np.argsort(e); de = d[o]; pct = 100 * np.arange(len(e)) / max(1, len(e) - 1)
        zf = np.polyfit(pct, de, 1)
        L = []
        for s in boots:
            ee = ev[g][s]; dd = dv[s]; okk = np.isfinite(ee) & np.isfinite(dd); ee, dd = ee[okk], dd[okk]
            if len(ee) < 30: continue
            oo = np.argsort(ee); L.append(np.polyval(np.polyfit(100 * np.arange(len(ee)) / max(1, len(ee) - 1), dd[oo], 1), xs))
        L = np.array(L)
        edg = np.linspace(0, len(e), 11).astype(int)
        dx = np.array([100 * (edg[k] + edg[k + 1]) / 2 / len(e) for k in range(10)])
        dy = np.array([de[edg[k]:edg[k + 1]].mean() for k in range(10)])
        out[f"{g}_fit"] = np.polyval(zf, [0, 100])
        out[f"{g}_blo"] = np.percentile(L, 2.5, 0); out[f"{g}_bhi"] = np.percentile(L, 97.5, 0)
        out[f"{g}_dx"] = dx; out[f"{g}_dy"] = dy
        out[f"{g}_rho"] = np.float64(spearmanr(e, d)[0]); out[f"{g}_p"] = np.float64(pvals[g])
    return out


def _load_smash_pvalues(path):
    if not path.is_file():
        null_inputs = FROZEN / "brainsmash_inputs"
        raise FileNotFoundError(
            "BrainSMASH results are required before scatter_bilateral.npz can be frozen. "
            f"The NIfTI inputs were written to {null_inputs}. Run "
            "compute/brainsmash_map_correspondence.py with those target, mask, and four field "
            f"files, write {path}, then rerun this freezer."
        )
    smash = pd.read_csv(path)
    required = {"group", "p_smash"}
    missing_columns = required - set(smash.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    if smash["group"].duplicated().any():
        duplicates = sorted(smash.loc[smash["group"].duplicated(False), "group"].astype(str).unique())
        raise ValueError(f"{path} has duplicate group rows: {duplicates}")
    by_group = smash.set_index("group")
    missing_groups = [group for group in GORD if group not in by_group.index]
    if missing_groups:
        raise ValueError(f"{path} is missing groups: {missing_groups}")
    pvalues = pd.to_numeric(by_group.loc[GORD, "p_smash"], errors="coerce")
    if not np.isfinite(pvalues).all() or not ((pvalues >= 0) & (pvalues <= 1)).all():
        raise ValueError(f"{path} contains non-finite or out-of-range p_smash values")
    return {group: float(pvalues.loc[group]) for group in GORD}


def freeze_scatter_bilateral(frame, ds):
    import nibabel as nib
    fsl_fields = resample_group_fields_to_fsl(frame)
    fsl = fsl_fields.reference
    cz = np.load(ds / "fig12_staging_all_depmaps.npz", allow_pickle=True)
    depF = np.nanmean(np.stack([d for _, d in cz["deps_time"]]), 0)
    STRf = fsl_fields.striatum
    ENRf = fsl_fields.fields
    strv = STRf & np.isfinite(depF)
    null_inputs = FROZEN / "brainsmash_inputs"; null_inputs.mkdir(exist_ok=True)
    def save_null_input(name, values, dtype=np.float32):
        header = fsl.header.copy(); header.set_data_dtype(dtype)
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=dtype), fsl.affine, header), null_inputs / name)
    save_null_input("target_stage_averaged_depletion_mni2mm.nii.gz", depF)
    save_null_input("mask_hmba_striatum_mni2mm.nii.gz", strv, np.uint8)
    for g in GORD:
        safe = {"Anxa1": "anxa1", "Sox6-all": "sox6_all", "Sox6-rest": "sox6_non_anxa1", "Calb1": "calb1"}[g]
        save_null_input(f"field_{safe}_signed_noncentered_mni2mm.nii.gz", ENRf[g])
    ijk = np.array(np.where(strv)).T; mm = ijk * np.array(fsl.header.get_zooms()[:3], float)
    pv = _load_smash_pvalues(SMASH_CSV)
    cur = _curves_from_arrays(depF[strv], {g: ENRf[g][strv] for g in GORD}, _block_boots_mm(mm), pv)
    np.savez_compressed(FROZEN / "scatter_bilateral.npz", groups=np.array(GORD), **cur)
    print(f"  scatter_bilateral (FSL 2mm, n={int(strv.sum())} vox):",
          {g: f"rho={cur[f'{g}_rho']:+.3f} p={cur[f'{g}_p']:.4f}" for g in GORD})


def freeze_delta_passthrough(ds):
    src = ds / "fig12_group_stats.csv"
    shutil.copy2(src, FROZEN / "delta_stats.csv")
    nsub = int(pd.read_csv(src).n.iloc[0])
    json.dump({"n_baselines": nsub}, open(FROZEN / "delta_meta.json", "w"))
    print(f"  delta_stats.csv (n={nsub})")


def _stamp(frame, ds):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
    import freeze_provenance as fp
    fp.stamp(FROZEN, "fig5_inputs", [
        Path(__file__).resolve().parent / "projection_fields.py",
        frame.locked_bins_path,
        frame.annotation_path,
        frame.fsl_2mm_path,
        SMASH_CSV,
        ds / "fig12_staging_all_depmaps.npz",
        ds / "fig12_staging_all_depmaps_hemi.npz",
        ds / "fig12_group_stats.csv",
    ])


if __name__ == "__main__":
    FROZEN.mkdir(parents=True, exist_ok=True)
    ppmi_root = _required_ppmi_root()
    ds = ppmi_root / "derived/datscan_spatial"
    print("[frame] build HMBA fields from shared Figure 4 arm2 locked bins ...", flush=True)
    frame = build_hmba_projection_frame(ppmi_root)
    freeze_rainbows(frame, ds)
    freeze_scatter_bilateral(frame, ds)
    freeze_delta_passthrough(ds)
    _stamp(frame, ds)
    print("done (projection group).")
