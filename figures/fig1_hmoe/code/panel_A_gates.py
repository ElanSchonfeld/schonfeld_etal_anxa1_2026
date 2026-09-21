#!/usr/bin/env python3
"""Figure 1 - HMoE architecture dendrogram (Panel A)."""
import sys
import pickle
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                   # noqa: E402
from awatramani_lab.moe.v4.dendrogram import (plot_hmoe_final, default_dendrogram_config,
                                              FAMILY_PALETTE)

FROZEN = HERE.parent / "frozen"
OUT = HERE.parent / "output" / "panels"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "HMoE_cell_count_colored_gates"

meta = pickle.load(open(FROZEN / "hmoe_dendrogram_meta.pkl", "rb"))
cache = np.load(FROZEN / "dendrogram_cellcount_cache.npz", allow_pickle=True)
y_sub = cache["y_sub"]
gate_outputs = {gk: {"nodes": list(cache[f"nodes_{gk}"])} for gk in cache["gate_keys"]}
results_v4 = {
    "bundle": {"node_to_leaves": meta["node_to_leaves"], "excluded_leaves": meta["excluded_leaves"]},
    "soft": {"gate_outputs": gate_outputs, "leaves": meta["leaves"]},
}

CONFIG = default_dendrogram_config()

FAMILY_PALETTE["Sox6"]  = {"base": "#0173B2", "light": "#E2EFF6"}
FAMILY_PALETTE["Calb1"] = {"base": "#DE8F05", "light": "#FBF2E2"}
FAMILY_PALETTE["Gad2"]  = {"base": "#2ca02c", "light": "#E7F4E7"}

CONFIG["fig"]["figsize"] = (13.8, 10.6)

CONFIG["spacing"]["min_leaf_dist"]   = 0.14
CONFIG["spacing"]["fam_gap"]         = 0.18
CONFIG["spacing"]["table_start"]     = 0.12
CONFIG["spacing"]["table_end_pad"]   = 0.02
CONFIG["spacing"]["zone_x_start"]    = 0.30
CONFIG["spacing"]["zone_x_start_by_family"] = {
    "Calb1": 0.30,
    "Sox6": 0.58,
    "Gad2": 0.58,
}
CONFIG["spacing"]["fam_band_pad_top_by_family"] = {
    "Calb1": 0.00,
    "Sox6": 0.03,
    "Gad2": 0.26,
}
CONFIG["spacing"]["fam_band_pad_bottom_by_family"] = {
    "Calb1": 0.00,
    "Sox6": 0.02,
    "Gad2": 0.02,
}
CONFIG["spacing"]["family_title_anchor_by_family"] = {
    "Calb1": {"x": 0.34, "y": 5.03, "ha": "left", "va": "top"},
    "Sox6": {"x": 0.66, "y": -0.66, "ha": "left", "va": "top"},
    "Gad2": {"x": 0.66, "y": -4.16, "ha": "left", "va": "top"},
}
CONFIG["spacing"]["zone_x_end_min"]  = 2.70
CONFIG["spacing"]["header_y_offset"] = 0.11

CONFIG["y_layout"]["y_scale"]       = 3.75
CONFIG["y_layout"]["ylim_pad_top"]  = 0.24
CONFIG["y_layout"]["ylim_pad_bot"]  = 0.08
CONFIG["y_layout"]["xlim_left"]     = -0.36

CONFIG["x_layout"]["uniform_dx"]    = 0.26
CONFIG["x_layout"]["manual_positions"] = {
    0: 0.00, 1: 0.22, 2: 0.50, 3: 0.82, 4: 1.16, 5: 1.48, 6: 1.78,
}
CONFIG["x_layout"]["max_internal_x"] = 1.78

CONFIG["label_layout"] = {
    "split_label_collision_radius": 0.20,
    "split_label_v_sep_step":       0.060,
    "split_label_max_tries":        32,
    "split_label_line_nudge":       0.070,
    "terminal_x_pad":               0.0,
    "terminal_label_anchor_x":      1.76,
    "split_label_offsets": {
        "Gad2+": {"dx": 0.02, "dy": -0.10},
        "Ddc-high": {"dx": -0.09, "dy": 0.82},
        "Calb1+": {"dx": -0.04, "dy": -0.10},
        "Sox6+": {"dx": -0.04, "dy": -0.12},
        "Lepr+": {"dx": -0.07, "dy": -0.04},
        "Cacna2d3+": {"dx": -0.09, "dy": 0.08},
        "Cacna2d3-": {"dx": -0.01, "dy": 0.02},
        "Ntm+": {"dx": -0.05, "dy": -0.03},
        "Atp8b1+": {"dx": -0.08, "dy": 0.08},
        "Tmem132d+": {"dx": -0.08, "dy": -0.06},
        "Rmst+": {"dx": -0.05, "dy": 0.08},
        "Rmst-": {"dx": -0.04, "dy": 0.02},
        "Eph4a+": {"dx": -0.07, "dy": 0.08},
        "Eph4a-": {"dx": -0.10, "dy": -0.02},
        "Il1rapl2+": {"dx": -0.09, "dy": 0.08},
        "Il1rapl2-": {"dx": -0.05, "dy": 0.02},
        "Cntnap5b+": {"dx": -0.09, "dy": 0.04},
        "Slc44a5+": {"dx": -0.08, "dy": -0.06},
        "Vcan+": {"dx": -0.05, "dy": 0.09},
        "Vcan-": {"dx": 0.00, "dy": 0.02},
        "Zfp521+": {"dx": -0.06, "dy": 0.06},
        "Zfp521-": {"dx": 0.00, "dy": 0.02},
        "Chrm3+": {"dx": -0.08, "dy": 0.08},
        "Chrm3-": {"dx": -0.08, "dy": -0.04},
    },
    "terminal_label_offsets": {
        "Cgnl1+": {"dx": 0.02, "dy": 0.06},
        "Plpp4+": {"dx": 0.02, "dy": 0.06},
        "Epha4+": {"dx": 0.02, "dy": 0.05},
        "Pcdh17+": {"dx": 0.02, "dy": -0.02},
        "Kcnmb2+": {"dx": 0.02, "dy": 0.05},
        "Zfp804b+": {"dx": 0.02, "dy": 0.05},
    },
}

CONFIG["fonts"]["split"]      = 13
CONFIG["fonts"]["fam_label"]  = 16
CONFIG["fonts"]["header"]     = 14
CONFIG["fonts"]["row_main"]   = 13
CONFIG["fonts"]["row_meta"]   = 14
CONFIG["fonts"]["cbar"]       = 13
CONFIG["fonts"]["root"]       = 14

CONFIG["bbox"]["split_pad"]   = 0.55
CONFIG["bbox"]["terminal_pad"] = 0.70
CONFIG["bbox"]["cbar_labelpad"] = 8

CONFIG["sizes"]["leaf_s"]        = 110
CONFIG["sizes"]["internal_s"]    = 190
CONFIG["sizes"]["edge_lw_min"]   = 0.6
CONFIG["sizes"]["edge_lw_max"]   = 6.0
CONFIG["sizes"]["edge_lw_gamma"] = 0.45

CONFIG["positions"]["axes_right"] = 0.86
CONFIG["positions"]["cbar"] = [0.89, 0.64, 0.012, 0.23]
CONFIG["positions"]["cbar_orientation"] = "vertical"
CONFIG["positions"]["cbar_use_figure_coords"] = True
CONFIG["positions"]["root_label_dx"] = -0.17
CONFIG["positions"]["subtype_header_x"] = 1.94

CONFIG["table"] = {
    "columns": [
        {"key": "short", "header": "Subtype", "width": 0.22, "font_key": "row_main"},
    ],
}

CONFIG["accuracy_colormap"] = {"vmin": 0.50, "vmax": 1.00, "cmap": "YlOrRd"}

CONFIG["edge_weighting"]["weigh_by"]     = "mouse"
CONFIG["edge_weighting"]["weigh_method"] = "cumulative"
CONFIG["edge_weighting"]["scale"]        = "log"

for ext in ("png", "pdf"):
    plot_hmoe_final(meta["tree"], results_v4, y_sub_test=y_sub,
                    filename=f"{STEM}.{ext}", mode="soft", metric="cell_count",
                    dpi=600, config=CONFIG)
Path(f"{STEM}.pdf").rename(f"{STEM}.ai")
print(f"  wrote {STEM.name}.{{png,ai}}")
