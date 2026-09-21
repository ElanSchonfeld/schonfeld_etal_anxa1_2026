#!/usr/bin/env python3
"""Render the horizontal bivariate extended human projection atlas for one arm.

Run: BV_ARM=arm2_matched python figures/fig4_projections/compute/render_extended_arm.py
"""
import os, sys, shutil, importlib.util
from pathlib import Path
import numpy as np

ARM = os.environ.setdefault("BV_ARM", "arm2_matched")
HERE = Path(__file__).resolve().parent
FIG4 = HERE.parent
RUNTIME = HERE / "extended_arm"
DATA = FIG4 / "frozen" / "extended_arm"
BVPATH = RUNTIME / "fig11_render_extended_bivariate_arm2.py"
OUT = FIG4 / "output" / "panels"
STEM = {"arm2_matched": "extended_arm2_horizontal", "arm1_matched": "extended_arm1_horizontal"}[ARM]
REQUIRED_INPUTS = (
    "fig11b_panelA_constrained_display_targets.tsv",
    "fig11b_panelA_constrained_targets.tsv",
    "fig11b_method_tangram_uniform_density_targets.tsv",
    "projection_matrix_human.tsv",
    "bin_metadata.csv",
    f"{ARM}_panelA_locked_bins.parquet",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def main():
    missing = [DATA / name for name in REQUIRED_INPUTS if not (DATA / name).is_file()]
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Figure 4 extended-arm inputs are missing; expected:\n"
            f"{listed}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    BV = _load("bivariate_arm", BVPATH)
    GS, ENR = BV.GS, BV.ENR
    B = _load("build_final_human_projection_figure", GS.BUILD)
    B.RPCA_BINS = DATA / f"{ARM}_panelA_locked_bins.parquet"
    tmp = OUT.parent / f"_extended_{ARM}_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    B.OUT_DIR = tmp
    B.configure_matplotlib()

    display, accepted_raw, rpca_raw, bins, kraft, kraft_meta = B.load_inputs()
    row_order = B.ordered_subtypes(display)
    calb1 = B.cluster_family("Calb1", bins, display, rpca_raw)
    groups = B.selected_spatial_display_columns(calb1, row_order)
    cen_map = ENR.add_enrichment_columns(bins, groups)
    nc_map = BV.add_nc_positive_columns(bins, groups)
    allv = np.concatenate([v[v > 0] for v in (bins[c].to_numpy(dtype=float) for c in nc_map.values())])
    _G = float(np.percentile(allv, 85)) if allv.size else 1.0
    MAG_SCALE = {g: _G for g in nc_map}

    _orig = B.load_module
    def _patched(name, path):
        mod = _orig(name, path)
        if "constrained_spatial_transfer" in str(path):
            mod.build_fields = BV.make_bivariate_build_fields(
                mod, cen_map, nc_map, MAG_SCALE, floor=BV.FLOOR, norm_pctile=BV.NORM, postfill_sig=BV.POSTFILL)
            mod.SLAB_HALF_MM = BV.SLAB
            mod.CORONAL_Y_LEVELS_CCF_MM = (-8.0, -4.0, 0.0, 4.0, 8.0)
            mod.AXIAL_Z_LEVELS_CCF_MM = (-8.0, -4.0, 0.0, 4.0, 8.0)
            mod.DISPLAY_SCALE_MODE = "per_column"
            mod.EXACT_FIELD_PASSTHROUGH = True
            mod.SHOW_STRUCTURE_LABELS = True
            mod.STRUCTURE_LABEL_NAMES = {"Caudate": "Caudate", "Pu": "Putamen", "NAC": "NAc"}
            mod.DIRECT_REGIONS = tuple(r for r in mod.DIRECT_REGIONS if r != "VeP")
            mod.GROSS_TO_DIRECT_REGIONS = {k: v for k, v in mod.GROSS_TO_DIRECT_REGIONS.items() if k != "GP"}
            mod.HORIZONTAL_BLOCKS = True
            mod.EXTRA_LEFT_MARGIN_IN = 2.0
            mod.STRUCT_LABEL_AXES_X = -0.30; mod.STRUCT_LABEL_FONTSIZE = 12
            mod.ROW_LABEL_AXES_X = -0.46; mod.SIDE_BLOCK_LABEL_FIG_X = 0.022
            mod.CBAR_RECT = [0.40, 0.090, 0.20, 0.013]; mod.CBAR_LABEL_FONTSIZE = 22
        return mod
    B.load_module = _patched

    B.render_selected_cluster_spatial_atlas(
        bins, rpca_raw, calb1, row_order, display_target_mode="panel_a_column_z")

    src_png = tmp / "figure11_human_projection_final_extended.png"
    src_pdf = tmp / "figure11_human_projection_final_extended.pdf"
    shutil.copy2(src_png, OUT / f"{STEM}.png")
    if src_pdf.exists():
        shutil.copy2(src_pdf, OUT / f"{STEM}.pdf")
    shutil.rmtree(tmp)
    print(f"wrote {OUT/STEM}.png + .pdf  (arm={ARM})")


if __name__ == "__main__":
    main()
