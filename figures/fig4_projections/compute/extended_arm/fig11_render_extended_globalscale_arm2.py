#!/usr/bin/env python3
"""Duplicate of the arm2 extended projection figure, COLUMN-Z ANCHORED so between-family and between-region magnitude is visible."""
import importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt

SCRIPT_DIR = Path(__file__).resolve().parent
FIG4_DIR = SCRIPT_DIR.parent.parent
CODE = SCRIPT_DIR
RENDER = CODE / "render_extended_remap_enrichment.py"
BUILD = CODE / "build_final_human_projection_figure.py"
METH = FIG4_DIR / "frozen" / "extended_arm"
OUTROOT = FIG4_DIR / "output" / "extended_arm"
ARM = "arm2_matched"
SIGQC = OUTROOT / ARM / "figure11_human_projection_final_extended_signal_qc.csv"
TEXTURE_ALPHA = 4.0
VMAX = 0.65

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

ENR = load("render_enr", RENDER)


def _splat_nocenter(shape, row, col, ev, nc, mask, sigma):
    """N_cells-weighted, density-normalized splat that PRESERVES the mean (no centering), then nearest-fills the mask so the field reaches the outline."""
    rr = np.rint(row).astype(int); cc = np.rint(col).astype(int)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1]) & np.isfinite(ev) & np.isfinite(nc)
    field = np.zeros(shape, dtype=float)
    if not ok.any(): return field
    nc_ok = np.clip(nc[ok], 0, None)
    dens = np.zeros(shape); np.add.at(dens, (rr[ok], cc[ok]), nc_ok); dens = gaussian_filter(dens, sigma)
    num = np.zeros(shape); np.add.at(num, (rr[ok], cc[ok]), ev[ok] * nc_ok); num = gaussian_filter(num, sigma)
    valid = mask & (dens > 0.05 * dens.max()) if dens.max() > 0 else np.zeros(shape, bool)
    if not valid.any(): return field
    field[valid] = num[valid] / dens[valid]
    fill = mask & ~valid
    if fill.any():
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        field[fill] = field[tuple(idx)][fill]
    field[~mask] = 0.0
    return field


def make_anchored_build_fields(mod, anch_map, vmax):
    def build_fields(atlas, bins, panel, target):
        slice_list = mod._build_slice_list(atlas)
        slices = {key: sdata for key, _lvl, sdata in slice_list}
        fields = {}; intensity_fields = {}; qc = []
        for key, level_mm, sdata in slice_list:
            axis = sdata.get("axis", "coronal"); coord = "y_ccf_mm" if axis == "coronal" else "z_ccf_mm"
            slab = bins[np.abs(bins[coord] - level_mm) <= mod.SLAB_HALF_MM].copy()
            lc = int(sdata["masks"].get("left_col_cut", sdata["shape"][1] // 2))
            gross_masks = {}
            for gk in mod.GROSS_TO_DIRECT_REGIONS:
                gm = sdata["masks"].get(gk)
                if gm is None: continue
                gm = gm.copy(); gm[:, lc:] = False; gross_masks[gk] = gm
            sdata["direct_masks"] = gross_masks
            for group_name in mod.DISPLAY_COLUMNS:
                anch_col = anch_map.get(group_name)
                field = np.zeros(sdata["shape"], dtype=float); intensity = np.zeros(sdata["shape"], dtype=float)
                for gk, regions in mod.GROSS_TO_DIRECT_REGIONS.items():
                    mask = gross_masks.get(gk)
                    if mask is None or not mask.any(): continue
                    gdf = slab[slab["region"].astype(str).isin(regions)]
                    if gdf.empty or anch_col is None or anch_col not in gdf.columns: continue
                    o = {}
                    for a in ("x", "y", "z"):
                        lo, hi = gdf[f"{a}_ccf_mm"].quantile([0.02, 0.98]); o[f"{a}_lo"], o[f"{a}_hi"] = float(lo), float(hi)
                    row, col = mod.map_bins_to_mask(gdf, mask, o, flip_ml=mod.FLIP_WITHIN_REGION_ML, axis=axis)
                    ev = gdf[anch_col].to_numpy(dtype=float); nc = gdf["n_cells"].to_numpy(dtype=float)
                    sig = ENR.SPLAT_SIGMA if gk == "Pu" else ENR.NONPUT_SPLAT_SIGMA
                    f = _splat_nocenter(sdata["shape"], row, col, ev, nc, mask, sig)
                    f = np.clip(f / vmax, -1.0, 1.0)
                    field = np.where(mask, f, field); intensity = np.where(mask, np.abs(f), intensity)
                left_signal = sdata["masks"].get("signal_left", sdata["masks"]["signal"])
                field[~left_signal] = 0.0; intensity[~left_signal] = 0.0
                if mod.MIRROR_LEFT_TO_RIGHT_HEMI:
                    field = mod._mirror_field_to_right(field, lc); intensity = mod._mirror_field_to_right(intensity, lc)
                fields[(key, group_name)] = field; intensity_fields[(key, group_name)] = intensity
            if mod.MIRROR_LEFT_TO_RIGHT_HEMI:
                sdata["direct_masks_display"] = {g: mod._mirror_mask_to_right(m, lc) for g, m in gross_masks.items()}
                sdata["masks"]["signal_display"] = mod._mirror_mask_to_right(sdata["masks"]["signal"], lc)
            else:
                sdata["direct_masks_display"] = gross_masks; sdata["masks"]["signal_display"] = sdata["masks"]["signal"]
            qc.append({"slice_key": key, "axis": axis, "ccf_level_mm": float(level_mm),
                       "n_spatial_bins_in_slab": int(len(slab)), "render_mode": "continuous_columnz_anchored_globalscale"})
        return slices, fields, intensity_fields, pd.DataFrame(qc)
    return build_fields


def render():
    bins_path = METH / f"{ARM}_panelA_locked_bins.parquet"
    print(f"=== rendering COLUMN-Z ANCHORED extended figure for {ARM} ===", flush=True)
    B = load("build_final_human_projection_figure", BUILD)
    B.RPCA_BINS = bins_path
    B.OUT_DIR = OUTROOT / f"{ARM}_globalscale"; B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    B.configure_matplotlib()
    display, accepted_raw, rpca_raw, bins, kraft, kraft_meta = B.load_inputs()
    row_order = B.ordered_subtypes(display)
    calb1 = B.cluster_family("Calb1", bins, display, rpca_raw)
    groups = B.selected_spatial_display_columns(calb1, row_order)
    enr_map = ENR.add_enrichment_columns(bins, groups)

    _orig = B.load_module
    def _patched(name, path):
        mod = _orig(name, path)
        if "constrained_spatial_transfer" in str(path):
            mod.build_fields = ENR.make_continuous_enrichment_build_fields(mod, enr_map)
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
    print(f"[{ARM}_globalscale] wrote -> {out}", flush=True)


if __name__ == "__main__":
    render()
