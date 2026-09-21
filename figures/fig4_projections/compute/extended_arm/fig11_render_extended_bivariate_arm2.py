#!/usr/bin/env python3
"""BIVARIATE variant of the arm2 extended projection figure: encode BOTH components of the map at once."""
import importlib.util, sys, os, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
GSPATH = SCRIPT_DIR / "fig11_render_extended_globalscale_arm2.py"
SLAB = float(os.environ.get("BV_SLAB", "4.5"))
POSTFILL = float(os.environ.get("BV_POSTFILL", "0.0"))
NORM = float(os.environ.get("BV_NORM", "75"))
FLOOR = float(os.environ.get("BV_FLOOR", "0.50"))
ARM = os.environ.get("BV_ARM", "arm1_matched")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

GS = load("gs_driver", GSPATH)
ENR = GS.ENR

def _smooth_in_mask(f, mask, sig):
    """Gaussian-smooth f only within mask (weight-normalized) -> blends piecewise-constant nearest-neighbor fill seams."""
    if sig <= 0 or not mask.any(): return f
    fm = gaussian_filter(np.where(mask, f, 0.0), sig); wm = gaussian_filter(mask.astype(float), sig)
    return np.where(mask, fm / np.maximum(wm, 1e-9), f)


def add_nc_positive_columns(bins, groups):
    """Non-centered POSITIVE relative enrichment (log2(raw/base) clipped at 0) -> alpha/magnitude source."""
    labels = list(groups.keys())
    raw = {lab: ENR._group_bin_score(bins, groups[lab]) for lab in labels}
    base = np.mean(np.vstack([raw[lab] for lab in labels]), axis=0)
    pos = base[base > 0]; eps = float(np.median(pos)) * 0.1 if pos.size else 1e-9
    mapping = {}
    for i, lab in enumerate(labels):
        enr = np.log2((raw[lab] + eps) / (base + eps))
        col = f"ncpos__{i}"
        bins[col] = np.nan_to_num(np.clip(enr, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)
        mapping[lab] = col
    return mapping


def make_bivariate_build_fields(R, cen_map, nc_map, mag_scale, floor=0.30, sig_pu=2.3, sig_nonput=3.6, norm_pctile=90.0, postfill_sig=0.0):
    def build_fields(atlas, bins, panel, target):
        slice_list = R._build_slice_list(atlas); slices = {key: sdata for key, _l, sdata in slice_list}
        fields = {}; intensity_fields = {}; qc = []
        for key, level_mm, sdata in slice_list:
            axis = sdata.get("axis", "coronal"); coord = "y_ccf_mm" if axis == "coronal" else "z_ccf_mm"
            slab = bins[np.abs(bins[coord] - level_mm) <= R.SLAB_HALF_MM].copy()
            lc = int(sdata["masks"].get("left_col_cut", sdata["shape"][1] // 2))
            gross = {}
            for gk in R.GROSS_TO_DIRECT_REGIONS:
                gm = sdata["masks"].get(gk)
                if gm is None: continue
                gm = gm.copy(); gm[:, lc:] = False; gross[gk] = gm
            sdata["direct_masks"] = gross
            for group_name in R.DISPLAY_COLUMNS:
                cen_col = cen_map.get(group_name); nc_col = nc_map.get(group_name)
                gmag = float(mag_scale.get(group_name, 1.0))
                field = np.zeros(sdata["shape"], dtype=float); intensity = np.zeros(sdata["shape"], dtype=float)
                for gk, regions in R.GROSS_TO_DIRECT_REGIONS.items():
                    mask = gross.get(gk)
                    if mask is None or not mask.any(): continue
                    gdf = slab[slab["region"].astype(str).isin(regions)]
                    if gdf.empty or cen_col is None or cen_col not in gdf.columns: continue
                    o = {}
                    for a in ("x", "y", "z"):
                        lo, hi = gdf[f"{a}_ccf_mm"].quantile([0.02, 0.98]); o[f"{a}_lo"], o[f"{a}_hi"] = float(lo), float(hi)
                    row, col = R.map_bins_to_mask(gdf, mask, o, flip_ml=R.FLIP_WITHIN_REGION_ML, axis=axis)
                    ncell = gdf["n_cells"].to_numpy(dtype=float)
                    sig = sig_pu if gk == "Pu" else sig_nonput
                    fcol = ENR._splat(sdata["shape"], row, col, gdf[cen_col].to_numpy(dtype=float), ncell, mask, sig)
                    fv = np.abs(fcol[mask]); fv = fv[np.isfinite(fv) & (fv > 0)]
                    sc = float(np.percentile(fv, norm_pctile)) if fv.size else 0.0
                    if sc > 1e-9: fcol = np.clip(fcol / sc, -1.0, 1.0)
                    fcol = _smooth_in_mask(fcol, mask, postfill_sig)
                    fmag = GS._splat_nocenter(sdata["shape"], row, col, gdf[nc_col].to_numpy(dtype=float), ncell, mask, sig)
                    fmag = _smooth_in_mask(fmag, mask, postfill_sig)
                    w = floor + (1.0 - floor) * np.clip(fmag / gmag, 0.0, 1.0)
                    fcol = fcol * w
                    field += fcol
                    intensity += np.abs(fcol)
                left = sdata["masks"].get("signal_left", sdata["masks"]["signal"])
                field[~left] = 0.0; intensity[~left] = 0.0
                if R.MIRROR_LEFT_TO_RIGHT_HEMI:
                    field = R._mirror_field_to_right(field, lc); intensity = R._mirror_field_to_right(intensity, lc)
                fields[(key, group_name)] = field; intensity_fields[(key, group_name)] = intensity
            if R.MIRROR_LEFT_TO_RIGHT_HEMI:
                sdata["direct_masks_display"] = {g: R._mirror_mask_to_right(m, lc) for g, m in gross.items()}
                sdata["masks"]["signal_display"] = R._mirror_mask_to_right(sdata["masks"]["signal"], lc)
            else:
                sdata["direct_masks_display"] = gross; sdata["masks"]["signal_display"] = sdata["masks"]["signal"]
            qc.append({"slice_key": key, "axis": axis, "ccf_level_mm": float(level_mm), "render_mode": "bivariate_texture_x_magnitude"})
        return slices, fields, intensity_fields, pd.DataFrame(qc)
    return build_fields


def render():
    B = load("build_final_human_projection_figure", GS.BUILD)
    bins_path = GS.METH / f"{ARM}_panelA_locked_bins.parquet"
    B.RPCA_BINS = bins_path
    B.OUT_DIR = GS.OUTROOT / f"{ARM}_bivariate"; B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== rendering BIVARIATE (color=texture, saturation=shared magnitude) for {ARM} -> {B.OUT_DIR} ===", flush=True)
    B.configure_matplotlib()
    display, accepted_raw, rpca_raw, bins, kraft, kraft_meta = B.load_inputs()
    row_order = B.ordered_subtypes(display)
    calb1 = B.cluster_family("Calb1", bins, display, rpca_raw)
    groups = B.selected_spatial_display_columns(calb1, row_order)
    cen_map = ENR.add_enrichment_columns(bins, groups)
    nc_map = add_nc_positive_columns(bins, groups)
    allv = np.concatenate([v[v > 0] for v in (bins[c].to_numpy(dtype=float) for c in nc_map.values())])
    _G = float(np.percentile(allv, 85)) if allv.size else 1.0
    MAG_SCALE = {g: _G for g in nc_map}
    print(f"SHARED magnitude scale (all columns) = {_G:.3f}", flush=True)

    _orig = B.load_module
    def _patched(name, path):
        mod = _orig(name, path)
        if "constrained_spatial_transfer" in str(path):
            mod.build_fields = make_bivariate_build_fields(mod, cen_map, nc_map, MAG_SCALE, floor=FLOOR, norm_pctile=NORM, postfill_sig=POSTFILL)
            mod.SLAB_HALF_MM = SLAB
            mod.CORONAL_Y_LEVELS_CCF_MM = (-8.0, -4.0, 0.0, 4.0, 8.0)
            mod.AXIAL_Z_LEVELS_CCF_MM = (-8.0, -4.0, 0.0, 4.0, 8.0)
            mod.DISPLAY_SCALE_MODE = "per_column"
            mod.EXACT_FIELD_PASSTHROUGH = True
            mod.SHOW_STRUCTURE_LABELS = True
            mod.STRUCTURE_LABEL_NAMES = {"Caudate": "Caudate", "Pu": "Putamen", "NAC": "NAc"}
            mod.DIRECT_REGIONS = tuple(r for r in mod.DIRECT_REGIONS if r != "VeP")
            mod.GROSS_TO_DIRECT_REGIONS = {k: v for k, v in mod.GROSS_TO_DIRECT_REGIONS.items() if k != "GP"}
            mod.EXTRA_LEFT_MARGIN_IN = 2.0
            mod.STRUCT_LABEL_AXES_X = -0.30; mod.STRUCT_LABEL_FONTSIZE = 12
            mod.ROW_LABEL_AXES_X = -0.46; mod.SIDE_BLOCK_LABEL_FIG_X = 0.022
            mod.CBAR_RECT = [0.40, 0.030, 0.32, 0.020]; mod.CBAR_LABEL_FONTSIZE = 25
        return mod
    B.load_module = _patched
    out = B.render_selected_cluster_spatial_atlas(bins, rpca_raw, calb1, row_order, display_target_mode="panel_a_column_z")
    print(f"[{ARM}_bivariate] wrote -> {out}", flush=True)


if __name__ == "__main__":
    render()
