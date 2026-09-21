#!/usr/bin/env python3
"""Splat the per-bin mouse projection enrichment onto the Allen mouse reference template and freeze the coronal fields, masks, backgrounds, group titles, structure order, and display scales.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python mouse/build_enrichment_fields.py
"""
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import FROZEN, source_root, work_dir

ATLAS = "allen_mouse_25um"
PANEL_B_MATRIX = "cell2cell_final_v2/results/stage2_finalized_nonspatial/parafac_projection_matrix_full18.tsv"
STRUCTURES = OrderedDict([("CP", (672,)), ("ACB", (56,)), ("OT", (754,))])
STRUCT_PREFIX = {"CP": ("CP-",), "ACB": ("ACB",), "OT": ("OT-",)}
VENTRAL_STRUCTURES = ("ACB", "OT")
AP_LEVELS = [4.2, 4.95, 5.7, 6.45, 7.2]
AP_WINDOW = 0.6
AP_WINDOW_BY_LEVEL = {6.8: 0.8}
SPLAT_SIGMA_CP_MM = 0.35
SPLAT_SIGMA_OTHER_MM = 0.50
SPLAT_SIGMA_ML_FACTOR = 2.0
PANEL_PCTILE = 97.0
ACB_LOCAL_BASELINE_QUANTILE = 0.20
SIGMA = 0.5
ANXA = ("Sox6:Tafa1", "Sox6:Vcan")
LABELS = ["Anxa1⁺\n(Tafa1/Vcan)", "Sox6\n(non-Anxa core)", "Calb1\n(all 10 subtypes)"]
PANEL_A_TO_B = {LABELS[0]: "Anxa1-associated", LABELS[1]: "Sox6 family", LABELS[2]: "Calb1 family"}
GROUPED_MOUSE_COLUMNS = OrderedDict([
    ("CP", ["CP-cd", "CP-rd", "CP-md", "CP-mv", "CP-rv", "CP-cv"]),
    ("ACB", ["ACB-core", "ACB-shell"]),
    ("OT", ["OT-lat", "OT-med"]),
    ("Amygdala", ["BLA", "CEA", "LA", "MEA"]),
])
MOUSE_COUNTS = {
    "Sox6:Tmem132d": 2466, "Sox6:Arhgap28": 2298, "Sox6:March3": 2203,
    "Sox6:Tafa1": 1987, "Sox6:Kcnmb2": 1552, "Sox6:Vcan": 1438,
    "Calb1:Pde11a": 1801, "Calb1:Kctd8": 1697, "Calb1:Ptprt": 1617,
    "Calb1:Sulf1": 1533, "Calb1:Stac": 1361, "Calb1:Chrm2": 930,
    "Calb1:Sox6": 901, "Calb1:Lpar1": 894, "Calb1:Ccdc192": 409,
    "Calb1:Gipr": 342,
}


def load_receiver_bins(path):
    bins = pd.read_csv(path).copy()
    bins = bins.rename(columns={"x_ccf": "center_ml", "y_ccf": "center_dv", "z_ccf": "center_ap"})
    if "region" not in bins.columns:
        bins["region"] = bins["slab_region"]
    return bins


def build_groups(index):
    sox6 = sorted(s for s in index if str(s).startswith("Sox6:"))
    calb1 = sorted(s for s in index if str(s).startswith("Calb1:") and s in MOUSE_COUNTS)
    return OrderedDict([
        (LABELS[0], [s for s in ANXA if s in sox6]),
        (LABELS[1], [s for s in sox6 if s not in ANXA]),
        (LABELS[2], calb1),
    ])


def add_enrichment(bins, score_df, groups):
    labels = list(groups.keys())
    cols = [f"bin_{int(b)}" for b in bins["bin_id"]]
    present = [c for c in cols if c in score_df.columns]
    raw = {}
    for lab in labels:
        members = [s for s in groups[lab] if s in score_df.index]
        if members:
            weights = np.array([MOUSE_COUNTS.get(s, 1.0) for s in members], dtype=float)
            weights = weights / weights.sum()
            values = score_df.loc[members, present].to_numpy()
            pooled = pd.Series((values * weights[:, None]).sum(axis=0), index=present)
            raw[lab] = pooled.reindex(cols).fillna(0.0).to_numpy()
        else:
            raw[lab] = np.zeros(len(bins))
    base = np.mean(np.vstack([raw[l] for l in labels]), axis=0)
    pos = base[base > 0]
    eps = float(np.median(pos)) * 0.1 if pos.size else 1e-9
    region = bins["region"].astype(str).to_numpy()
    is_ventral_striatum = np.array([r.startswith(("ACB", "OT-")) for r in region])
    enr_map = {}
    for i, lab in enumerate(labels):
        enr = np.log2((raw[lab] + eps) / (base + eps))
        centered = enr.copy()
        cp_like = ~is_ventral_striatum
        if cp_like.any():
            centered[cp_like] = centered[cp_like] - centered[cp_like].mean()
        ec = f"enr__{i}"
        bins[f"proj__{i}"] = np.nan_to_num(raw[lab], nan=0.0, posinf=0.0, neginf=0.0)
        bins[ec] = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
        enr_map[lab] = ec
    return enr_map


def load_panel_b_matrix(path):
    proj_raw_df = pd.read_csv(path, sep="\t", index_col=0)
    comp = pd.DataFrame(index=proj_raw_df.index)
    for new_col, old_cols in GROUPED_MOUSE_COLUMNS.items():
        present = [c for c in old_cols if c in proj_raw_df.columns]
        comp[new_col] = proj_raw_df[present].sum(axis=1)
    keep = [s for s in comp.index if str(s).split(":")[0] in ("Sox6", "Calb1")]
    comp = comp.loc[keep]
    frac = comp.div(comp.sum(axis=1), axis=0)
    subs = list(frac.index)
    w = np.array([MOUSE_COUNTS.get(s, 1.0) for s in subs], dtype=float)
    mu = frac.mul(w, axis=0).sum(axis=0) / w.sum()
    sd = np.sqrt(((frac - mu) ** 2).mul(w, axis=0).sum(axis=0) / w.sum())
    z = ((frac - mu) / sd.replace(0, np.nan)).fillna(0.0)
    groups = OrderedDict([
        ("Anxa1-associated", [s for s in ANXA if s in z.index]),
        ("Sox6 family", [s for s in z.index if str(s).startswith("Sox6:")]),
        ("Calb1 family", [s for s in z.index if str(s).startswith("Calb1:")]),
    ])
    rows = []
    for label, members in groups.items():
        idx = [subs.index(s) for s in members]
        ww = pd.Series(w, index=subs).loc[members].to_numpy(dtype=float)
        rows.append(pd.Series(z.iloc[idx].mul(ww, axis=0).sum(axis=0) / ww.sum(), name=label))
    return pd.DataFrame(rows).fillna(0.0)


def lock_to_panel_b(field, mask, lab, panel_b, structure):
    broad_value = float(panel_b.loc[PANEL_A_TO_B[lab], structure])
    if abs(broad_value) <= 1e-12:
        return field
    values = field[mask]
    locked = field.copy()
    if broad_value > 0 and values.size:
        oriented = values - float(np.quantile(values, ACB_LOCAL_BASELINE_QUANTILE))
    else:
        oriented = np.zeros_like(values)
    locked[mask] = oriented
    return locked


def ap_window(ap_mm):
    return AP_WINDOW_BY_LEVEL.get(round(float(ap_mm), 1), AP_WINDOW)


def family_ap_locations(score_df, raw_bins, groups):
    labels = list(groups)
    fam = {s: l for l, ms in groups.items() for s in ms}
    cols = [f"bin_{int(i)}" for i in raw_bins.bin_id]
    keep = [c in score_df.columns for c in cols]
    cols = [c for c, k in zip(cols, keep) if k]
    bb = raw_bins[keep].reset_index(drop=True)
    reg = bb.slab_region.astype(str).to_numpy()
    ap = bb.z_ccf.to_numpy(float)
    dv = np.array([s[4] if s.startswith("CP") and len(s) > 4 else "" for s in reg])
    m = np.array([s.startswith("CP") for s in reg]) & (dv == "d")
    subs = [s for s in score_df.index if s in fam]
    X = score_df.loc[subs, cols].to_numpy(float)[:, m]
    apm = ap[m]
    L = np.log(np.clip(X, 1e-12, None))
    Y = L - L.mean(0, keepdims=True)
    loc = []
    for i in range(len(subs)):
        v = Y[i]
        sel = v >= np.quantile(v, 0.9)
        vv = np.clip(v[sel], 0, None)
        aa = apm[sel]
        o = np.argsort(aa)
        c = np.cumsum(vv[o]) / vv.sum()
        loc.append(float(aa[o][np.searchsorted(c, 0.5)]))
    loc = np.array(loc)
    return {l: float(np.median([loc[i] for i, s in enumerate(subs) if fam[s] == l])) for l in labels}


def ap_kernel(labels, locations):
    G = {}
    for lvl in AP_LEVELS:
        raw = {l: float(np.exp(-((lvl - locations[l]) ** 2) / (2 * SIGMA ** 2))) for l in labels}
        mx = max(raw.values())
        for l in labels:
            G[(l, lvl)] = raw[l] / mx
    return G


def build_context(atlas, ap_mm):
    ann = atlas.annotation
    ref = atlas.reference
    res = atlas.resolution[0]
    ap_vox = int(ap_mm * 1000 / res)
    if ap_vox >= ann.shape[0]:
        return None
    coronal = ann[ap_vox]
    ref_slice = ref[ap_vox]
    midline = coronal.shape[1] // 2
    masks_full = {}
    for name, sids in STRUCTURES.items():
        m = np.isin(coronal, list(sids))
        m[:, midline:] = False
        masks_full[name] = m
    union = np.zeros_like(coronal, dtype=bool)
    for m in masks_full.values():
        union |= m
    if not union.any():
        return None
    coords = np.argwhere(union)
    dv0, ml0 = coords.min(axis=0)
    dv1, ml1 = coords.max(axis=0)
    dv0 = max(0, dv0 - 30)
    dv1 = min(coronal.shape[0], dv1 + 30)
    ml0 = max(0, ml0 - 45)
    ml1 = min(midline, ml1 + 45)
    brain = coronal > 0
    brain[:, midline:] = False
    ref_crop = ref_slice[dv0:dv1, ml0:ml1].astype(float)
    ref_norm = ref_crop / max(ref_crop.max(), 1.0)
    brain_crop = brain[dv0:dv1, ml0:ml1]
    dapi = np.ones((*ref_crop.shape, 3))
    g = 0.92 - ref_norm[brain_crop] * 0.25
    for c in range(3):
        dapi[brain_crop, c] = g
    ext = [ml0 * res / 1000, ml1 * res / 1000, dv1 * res / 1000, dv0 * res / 1000]
    return {"dapi": dapi, "masks": {n: m[dv0:dv1, ml0:ml1] for n, m in masks_full.items()},
            "ext": ext, "res": res, "dv0": dv0, "ml0": ml0, "shape": ref_crop.shape}


def splat(shape, row, col, ev, nc, mask, sigma, center=True):
    rr = np.rint(row).astype(int)
    cc = np.rint(col).astype(int)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1]) & np.isfinite(ev) & np.isfinite(nc)
    field = np.zeros(shape)
    if not ok.any():
        return field
    ncn = np.clip(nc[ok], 0, None)
    smooth_sigma = (float(sigma), float(sigma) * SPLAT_SIGMA_ML_FACTOR)
    dens = np.zeros(shape)
    np.add.at(dens, (rr[ok], cc[ok]), ncn)
    dens = gaussian_filter(dens, smooth_sigma)
    num = np.zeros(shape)
    np.add.at(num, (rr[ok], cc[ok]), ev[ok] * ncn)
    num = gaussian_filter(num, smooth_sigma)
    valid = mask & (dens > 0.05 * dens.max()) if dens.max() > 0 else np.zeros(shape, bool)
    if not valid.any():
        return field
    field[valid] = num[valid] / dens[valid]
    fill = mask & ~valid
    if fill.any():
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        nearest = field[tuple(idx)]
        field[fill] = nearest[fill]
    if center:
        field[mask] -= field[mask].mean()
    field[~mask] = 0.0
    return field


def main():
    from brainglobe_atlasapi import BrainGlobeAtlas

    d = work_dir("mouse")
    score_df = pd.read_csv(d / "allenFinal_bin_level_LRRK2.tsv", sep="\t", index_col=0)
    raw_bins = pd.read_csv(d / "allenFinal_receiver_bins.csv")
    allbins = load_receiver_bins(d / "allenFinal_receiver_bins.csv")
    keep = allbins["region"].astype(str).str.startswith(("CP-", "ACB", "OT-"))
    bins = allbins[keep].copy().reset_index(drop=True)
    groups = build_groups(score_df.index)
    labels = list(groups.keys())
    enr_map = add_enrichment(bins, score_df, groups)
    panel_b = load_panel_b_matrix(source_root() / PANEL_B_MATRIX)
    G = ap_kernel(labels, family_ap_locations(score_df, raw_bins, groups))

    atlas = BrainGlobeAtlas(ATLAS, check_latest=False)
    contexts = {ap: build_context(atlas, ap) for ap in AP_LEVELS}

    base_fields = {}
    for ap in AP_LEVELS:
        ctx = contexts.get(ap)
        if ctx is None:
            continue
        slab = bins[np.abs(bins["center_ap"] - ap) <= ap_window(ap)]
        res = ctx["res"]
        W = ctx["shape"][1]
        for lab in labels:
            for sname in STRUCTURES:
                mask = ctx["masks"][sname]
                if not mask.any():
                    continue
                sb = slab[slab["region"].astype(str).str.startswith(STRUCT_PREFIX[sname])]
                if len(sb) < 4:
                    continue
                row = sb["center_dv"].to_numpy() * 1000 / res - ctx["dv0"]
                col = (W - 1) - (sb["center_ml"].to_numpy() * 1000 / res - ctx["ml0"])
                sig_mm = SPLAT_SIGMA_CP_MM if sname == "CP" else SPLAT_SIGMA_OTHER_MM
                value_col = enr_map[lab]
                if sname in VENTRAL_STRUCTURES:
                    value_col = value_col.replace("enr__", "proj__")
                fe = splat(ctx["shape"], row, col, sb[value_col].to_numpy(),
                           sb["n_cells"].to_numpy(), mask, sig_mm * 1000.0 / res,
                           center=(sname == "CP"))
                if sname in VENTRAL_STRUCTURES:
                    fe = lock_to_panel_b(fe, mask, lab, panel_b, sname)
                base_fields[(ap, lab, sname)] = fe

    render_fields = {}
    scale_values = {}
    for (ap, lab, sname), fe_base in base_fields.items():
        fe = fe_base * (G.get((lab, float(ap)), 1.0) if sname == "CP" else 1.0)
        render_fields[(ap, lab, sname)] = fe
        ev = np.abs(fe[contexts[ap]["masks"][sname]])
        ev = ev[ev > 0]
        if ev.size:
            scale_values.setdefault((lab, sname), []).append(ev)
    render_scales = {}
    for key, chunks in scale_values.items():
        vals = np.concatenate(chunks) if chunks else np.array([])
        render_scales[key] = float(np.percentile(vals, PANEL_PCTILE)) if vals.size else 0.0

    structs = list(STRUCTURES.keys())
    arrays = {"ap_mm": np.asarray(AP_LEVELS, dtype=float)}
    for i, ap in enumerate(AP_LEVELS):
        ctx = contexts.get(ap)
        if ctx is None:
            continue
        arrays[f"dapi_{i}"] = np.asarray(ctx["dapi"], dtype=np.float32)
        arrays[f"ext_{i}"] = np.asarray(ctx["ext"], dtype=np.float32)
        for s in structs:
            m = ctx["masks"].get(s)
            if m is not None and m.any():
                arrays[f"mask_{i}_{s}"] = np.asarray(m, dtype=bool)
        for gi, lab in enumerate(labels):
            for s in structs:
                fe = render_fields.get((ap, lab, s))
                if fe is not None:
                    arrays[f"field_{i}_{gi}_{s}"] = np.asarray(fe, dtype=np.float32)
    FROZEN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FROZEN / "mouse_enrichment_fields.npz", **arrays)

    titles = [(lab.partition("\n")[0] if "⁺" in lab.partition("\n")[0]
               else lab.partition("\n")[0] + "⁺") for lab in labels]
    pd.DataFrame({"group_idx": range(len(labels)), "title": titles, "group_key": labels}
                 ).to_csv(FROZEN / "mouse_enrichment_meta.csv", index=False)
    pd.DataFrame({"structure": structs}).to_csv(FROZEN / "mouse_enrichment_structures.csv", index=False)
    pd.DataFrame([{"group_idx": gi, "structure": s, "scale": render_scales.get((lab, s), 0.0)}
                  for gi, lab in enumerate(labels) for s in structs]
                 ).to_csv(FROZEN / "mouse_enrichment_scales.csv", index=False)


if __name__ == "__main__":
    main()
