"""Render the extended human atlas with bin-level centered enrichment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
FIG4_DIR = SCRIPT_DIR.parent.parent
BUILD_FINAL = SCRIPT_DIR / "build_final_human_projection_figure.py"
REMAP_BINS = FIG4_DIR / "frozen" / "extended_arm" / "arm2_matched_panelA_locked_bins.parquet"

KAMATH = {
    "Sox6:Tafa1": 5068, "Sox6:Arhgap28": 3386, "Sox6:Vcan": 2438, "Sox6:Tmem132d": 2233,
    "Sox6:Kcnmb2": 674, "Sox6:March3": 336,
    "Calb1:Ptprt": 1893, "Calb1:Sulf1": 1418, "Calb1:Sox6": 1123, "Calb1:Pde11a": 1101,
    "Calb1:Kctd8": 699, "Calb1:Gipr": 684, "Calb1:Ccdc192": 424, "Calb1:Lpar1": 359,
    "Calb1:Stac": 144, "Calb1:Chrm2": 68,
}
REGION_GROUPS = {
    "Putamen": {"PuR", "PuC", "PuPV"},
    "Caudate": {"CaH", "CaB", "CaT"},
    "NAc": {"NACc", "NACs"},
    "VeP": {"VeP"},
}
SPLAT_SIGMA = 2.4
NONPUT_SPLAT_SIGMA = 4.2
PANEL_NORM_PCTILE = 97.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe(subtype: str) -> str:
    return subtype.replace(":", "_")


def _group_bin_score(bins: pd.DataFrame, members: list[str]) -> np.ndarray:
    cols, w = [], []
    for m in members:
        col = f"score_{_safe(m)}"
        if col in bins.columns and m in KAMATH:
            cols.append(col)
            w.append(float(KAMATH[m]))
    if not cols:
        return np.zeros(len(bins), dtype=float)
    w = np.asarray(w, dtype=float)
    return (bins[cols].to_numpy() * w).sum(axis=1) / w.sum()


def add_enrichment_columns(bins: pd.DataFrame, groups: "dict[str, list[str]]") -> dict[str, str]:
    """Attach enr__<i> centered cross-column enrichment columns to `bins`, computed from the RAW remap scores."""
    labels = list(groups.keys())
    raw = {lab: _group_bin_score(bins, groups[lab]) for lab in labels}
    base = np.mean(np.vstack([raw[lab] for lab in labels]), axis=0)
    pos = base[base > 0]
    eps = float(np.median(pos)) * 0.1 if pos.size else 1e-9
    region = bins["region"].astype(str).to_numpy()
    mapping: dict[str, str] = {}
    for i, lab in enumerate(labels):
        enr = np.log2((raw[lab] + eps) / (base + eps))
        centered = enr.copy()
        for members in REGION_GROUPS.values():
            m = np.isin(region, list(members))
            if m.any():
                centered[m] = centered[m] - centered[m].mean()
        col = f"enr__{i}"
        bins[col] = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
        mapping[lab] = col
    return mapping


def _splat(shape, row, col, ev, nc, mask, sigma):
    """N_cells-weighted, density-normalized splat of a signed per-bin value into a mask, then mean-centered over the mask."""
    rr = np.rint(row).astype(int)
    cc = np.rint(col).astype(int)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1]) & np.isfinite(ev) & np.isfinite(nc)
    field = np.zeros(shape, dtype=float)
    if not ok.any():
        return field
    nc_ok = np.clip(nc[ok], 0, None)
    dens = np.zeros(shape, dtype=float)
    np.add.at(dens, (rr[ok], cc[ok]), nc_ok)
    dens = gaussian_filter(dens, sigma)
    num = np.zeros(shape, dtype=float)
    np.add.at(num, (rr[ok], cc[ok]), ev[ok] * nc_ok)
    num = gaussian_filter(num, sigma)
    valid = mask & (dens > 0.05 * dens.max()) if dens.max() > 0 else np.zeros(shape, dtype=bool)
    if not valid.any():
        return field
    field[valid] = num[valid] / dens[valid]
    fill_targets = mask & ~valid
    if fill_targets.any():
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        nearest = field[tuple(idx)]
        field[fill_targets] = nearest[fill_targets]
    field[mask] -= field[mask].mean()
    field[~mask] = 0.0
    return field


def make_continuous_enrichment_build_fields(R, enr_map: dict[str, str]):
    """Continuous gross-structure splat of the per-bin centered enrichment."""

    def build_fields(atlas, bins, panel, target):
        slice_list = R._build_slice_list(atlas)
        slices = {key: sdata for key, _level, sdata in slice_list}
        fields: dict = {}
        intensity_fields: dict = {}
        qc_rows = []
        for key, level_mm, sdata in slice_list:
            axis = sdata.get("axis", "coronal")
            coord = "y_ccf_mm" if axis == "coronal" else "z_ccf_mm"
            slab = bins[np.abs(bins[coord] - level_mm) <= R.SLAB_HALF_MM].copy()
            lc = int(sdata["masks"].get("left_col_cut", sdata["shape"][1] // 2))
            gross_masks: dict = {}
            for gk in R.GROSS_TO_DIRECT_REGIONS:
                gm = sdata["masks"].get(gk)
                if gm is None:
                    continue
                gm = gm.copy()
                gm[:, lc:] = False
                gross_masks[gk] = gm
            sdata["direct_masks"] = gross_masks

            for group_name in R.DISPLAY_COLUMNS:
                enr_col = enr_map.get(group_name)
                field = np.zeros(sdata["shape"], dtype=float)
                intensity = np.zeros(sdata["shape"], dtype=float)
                for gk, regions in R.GROSS_TO_DIRECT_REGIONS.items():
                    mask = gross_masks.get(gk)
                    if mask is None or not mask.any():
                        continue
                    gdf = slab[slab["region"].astype(str).isin(regions)]
                    if gdf.empty or enr_col is None or enr_col not in gdf.columns:
                        continue
                    o = {}
                    for a in ("x", "y", "z"):
                        lo, hi = gdf[f"{a}_ccf_mm"].quantile([0.02, 0.98])
                        o[f"{a}_lo"], o[f"{a}_hi"] = float(lo), float(hi)
                    row, col = R.map_bins_to_mask(
                        gdf, mask, o, flip_ml=R.FLIP_WITHIN_REGION_ML, axis=axis,
                    )
                    ev = gdf[enr_col].to_numpy(dtype=float)
                    nc = gdf["n_cells"].to_numpy(dtype=float)
                    sig = SPLAT_SIGMA if gk == "Pu" else NONPUT_SPLAT_SIGMA
                    f = _splat(sdata["shape"], row, col, ev, nc, mask, sig)
                    fv = np.abs(f[mask])
                    fv = fv[np.isfinite(fv) & (fv > 0)]
                    sc = float(np.percentile(fv, PANEL_NORM_PCTILE)) if fv.size else 0.0
                    if sc > 1e-9:
                        f = np.clip(f / sc, -1.0, 1.0)
                    field += f
                    intensity += np.abs(f)
                left_signal = sdata["masks"].get("signal_left", sdata["masks"]["signal"])
                field[~left_signal] = 0.0
                intensity[~left_signal] = 0.0
                if R.MIRROR_LEFT_TO_RIGHT_HEMI:
                    field = R._mirror_field_to_right(field, lc)
                    intensity = R._mirror_field_to_right(intensity, lc)
                fields[(key, group_name)] = field
                intensity_fields[(key, group_name)] = intensity

            if R.MIRROR_LEFT_TO_RIGHT_HEMI:
                sdata["direct_masks_display"] = {
                    g: R._mirror_mask_to_right(m, lc) for g, m in gross_masks.items()
                }
                sdata["masks"]["signal_display"] = R._mirror_mask_to_right(
                    sdata["masks"]["signal"], lc
                )
            else:
                sdata["direct_masks_display"] = gross_masks
                sdata["masks"]["signal_display"] = sdata["masks"]["signal"]

            qc_rows.append({
                "slice_key": key,
                "axis": axis,
                "ccf_level_mm": float(level_mm),
                "n_spatial_bins_in_slab": int(len(slab)),
                "n_gross_masks": int(sum(m.any() for m in gross_masks.values())),
                "render_mode": "continuous_enrichment_no_subdivisions",
            })
        return slices, fields, intensity_fields, pd.DataFrame(qc_rows)

    return build_fields


def _leaveout_columns(B, row_order: list[str]) -> "dict[str, list[str]]":
    from collections import OrderedDict
    anxa = [s for s in B.ANXA_ANCHORS if s in row_order]
    all_sox6 = [s for s in row_order if s.startswith("Sox6:")]
    all_calb1 = [s for s in row_order if s.startswith("Calb1:")]
    anxa_neg = [s for s in all_sox6 if s not in {"Sox6:Tafa1", "Sox6:Vcan"}]
    aldh_neg = [s for s in all_sox6 if s not in {"Sox6:Tafa1", "Sox6:Vcan", "Sox6:Kcnmb2"}]
    cols: "OrderedDict[str, list[str]]" = OrderedDict()
    if anxa:
        cols["Anxa1+\n(Tafa1/Vcan)"] = anxa
    if all_sox6:
        cols["Sox6+\n(all 6 subtypes)"] = all_sox6
    if all_calb1:
        cols["Calb1+\n(all subtypes)"] = all_calb1
    if anxa_neg:
        cols["Sox6+/Anxa1-\n(drop Tafa1/Vcan)"] = anxa_neg
    if aldh_neg:
        cols["Sox6+/Aldh1a1-\n(drop +Kcnmb2)"] = aldh_neg
    return cols


def main() -> None:
    if not REMAP_BINS.exists():
        raise FileNotFoundError(f"Missing remap bins: {REMAP_BINS}")
    leaveouts = "--leaveouts" in sys.argv
    B = _load("build_final_human_projection_figure", BUILD_FINAL)
    B.RPCA_BINS = REMAP_BINS

    B.configure_matplotlib()
    B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    display, accepted_raw, rpca_raw, bins, kraft, kraft_meta = B.load_inputs()
    row_order = B.ordered_subtypes(display)
    calb1 = B.cluster_family("Calb1", bins, display, rpca_raw)

    fig_stem = None
    if leaveouts:
        B.selected_spatial_display_columns = lambda calb1, row_order: _leaveout_columns(B, row_order)
        fig_stem = "figure11_human_projection_final_extended_leaveouts"

    groups = B.selected_spatial_display_columns(calb1, row_order)
    enr_map = add_enrichment_columns(bins, groups)

    _orig_load = B.load_module

    def _patched_load(name, path):
        mod = _orig_load(name, path)
        if "constrained_spatial_transfer" in str(path):
            mod.build_fields = make_continuous_enrichment_build_fields(mod, enr_map)
            mod.DISPLAY_SCALE_MODE = "per_column"
            mod.HORIZONTAL_BLOCKS = True
            mod.EXACT_FIELD_PASSTHROUGH = True
            mod.SHOW_STRUCTURE_LABELS = True
            mod.STRUCTURE_LABEL_NAMES = {
                "Caudate": "Caudate", "Pu": "Putamen", "NAC": "NAc",
            }
            mod.DIRECT_REGIONS = tuple(r for r in mod.DIRECT_REGIONS if r != "VeP")
            mod.GROSS_TO_DIRECT_REGIONS = {
                k: v for k, v in mod.GROSS_TO_DIRECT_REGIONS.items() if k != "GP"
            }
            mod.EXTRA_LEFT_MARGIN_IN = 2.0
            mod.STRUCT_LABEL_AXES_X = -0.30
            mod.STRUCT_LABEL_FONTSIZE = 12
            mod.ROW_LABEL_AXES_X = -0.46
            mod.SIDE_BLOCK_LABEL_FIG_X = 0.022
            mod.CBAR_RECT = [0.40, 0.090, 0.20, 0.013]
            mod.CBAR_LABEL_FONTSIZE = 22
        return mod

    B.load_module = _patched_load

    out_png = B.render_selected_cluster_spatial_atlas(
        bins,
        rpca_raw,
        calb1,
        row_order,
        fig_stem=fig_stem,
        display_target_mode="panel_a_column_z",
    )
    print(f"wrote extended (continuous bin-enrichment) figure: {out_png}")


if __name__ == "__main__":
    main()
