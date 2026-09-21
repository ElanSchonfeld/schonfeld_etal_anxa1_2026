#!/usr/bin/env python3
"""Panel A constrained spatial transfer for Fig. 11b."""
from __future__ import annotations

import json
import hashlib
import io
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

def _ensure_helvetica_bold_for_matplotlib():
    """Extract Helvetica Regular+Bold from .ttc and register with mpl."""
    import os
    from pathlib import Path
    src_ttc = Path("/System/Library/Fonts/Helvetica.ttc")
    if not src_ttc.exists():
        return None
    cache_dir = Path(os.path.expanduser("~/.cache/matplotlib_helvetica"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        0: cache_dir / "Helvetica.ttf",
        1: cache_dir / "Helvetica-Bold.ttf",
    }
    needs_extract = any(not p.exists() for p in targets.values())
    if needs_extract:
        try:
            from fontTools.ttLib import TTCollection
        except ImportError:
            return None
        ttc = TTCollection(str(src_ttc))
        for face_idx, out_path in targets.items():
            if face_idx < len(ttc.fonts) and not out_path.exists():
                ttc.fonts[face_idx].save(str(out_path))
    import matplotlib.font_manager as _fm
    for out_path in targets.values():
        if out_path.exists():
            _fm.fontManager.addfont(str(out_path))
    return cache_dir


_ensure_helvetica_bold_for_matplotlib()
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom as ndzoom
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
FIG4_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = FIG4_DIR.parent.parent
DATA_DIR = FIG4_DIR / "frozen" / "extended_arm"
sys.path.insert(0, str(PROJECT_ROOT))


PANEL_A_BIN_META = DATA_DIR / "panelA_bin_metadata.csv"
PANEL_A_SNAPSHOT_COMMIT = "f9be8ce6b753affca9e2a23c8146d3504f298087"
PANEL_A_SNAPSHOT_ROOT = "figures/fig4_projections/frozen/extended_arm/panelA_snapshot"
PANEL_A_SNAPSHOT_DIR = DATA_DIR / "panelA_snapshot"
PANEL_A_CANONICAL_SHA256_PREFIX = "5735892f68964dd2"
PANEL_A_BIN_CANONICAL_SHA256_PREFIX = "957717900e510846"
H4_BIN_METADATA = DATA_DIR / "h4_bin_435_metadata.csv"
VE_P_COORDS = DATA_DIR / "snrna_ccf_coords_cluster.parquet"
CCF_CELLS = VE_P_COORDS

OUT_DIR = FIG4_DIR / "output" / "extended_arm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ATLAS_NAME = "allen_human_500um"
VOXEL_SIZE_MM = 2.0
MIN_CELLS_PER_BIN = 35
VOXELS_PER_REGION = 50
RANDOM_SEED = 42
RENDER_DPI = 600

CORONAL_Y_LEVELS_CCF_MM = (-4.0, 0.0, 4.0, 8.0)
AXIAL_Z_LEVELS_CCF_MM = (-4.0, 0.0, 4.0, 8.0)
AP_LEVELS_MM = (78.0, 82.0, 86.0, 90.0, 92.0)

SLAB_HALF_MM = 5.0
SMOOTH_SIGMA_PX = 0.85
GAMMA = 0.50
DISPLAY_TARGET_MODE = "panel_a_column_z"
DISPLAY_ABS_PERCENTILE = 80.0
MIN_COLOR_ALPHA = 0.28
RAW_COLOR_ALPHA = 0.88
SIGNED_COLOR_ALPHA = 0.90
INTENSITY_COLOR_FLOOR = 0.55
DISPLAY_SCALE_MODE = "shared_robust"
SHAPE_RATIO_CLIP = (0.70, 1.45)
TEXTURE_RESIDUAL_DISPLAY = False
TEXTURE_RESIDUAL_GAIN = 0.55
EXACT_FIELD_PASSTHROUGH = False
SHOW_STRUCTURE_LABELS = False
STRUCTURE_LABEL_NAMES: dict[str, str] = {}
ROW_LABEL_AXES_X = -0.08
SIDE_BLOCK_LABEL_FIG_X = 0.018
HORIZONTAL_BLOCKS = False
EXTRA_LEFT_MARGIN_IN = 0.0
STRUCT_LABEL_AXES_X = -0.34
STRUCT_LABEL_FONTSIZE = 11
CBAR_RECT = None
CBAR_LABEL_FONTSIZE = None
FLIP_WITHIN_REGION_ML = False
MIRROR_LEFT_TO_RIGHT_HEMI = True
BG_CROP_PAD_VOX = 36
MIN_BINS_FOR_REGION_MASK = 2
REGION_MASK_ASSIGN_SIGMA_PX = 7.0
INTERP_NEIGHBORS = 1
CALB1_MOJENA_K = 1.0
CALB1_K_RANGE = (2, 3, 4, 5, 6)
CALB1_CONSENSUS_REGION_COLUMNS = (
    "CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "VeP",
)

DIRECT_REGIONS = ("CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "VeP")
NAC_FINE_REGIONS = ("NACc", "NACs")
PUT_REGIONS = ("PuR", "PuC", "PuPV")

ATLAS_IDS = {
    "CaH": (10335,),
    "CaB": (10336,),
    "CaT": (10337,),
    "Pu": (10338,),
    "NAC": (10339,),
    "GP": (10342, 10343, 10344),
    "LV": (10596, 10597, 10598, 10599, 10600, 266441657),
}
SIGNAL_IDS = (
    ATLAS_IDS["CaH"] + ATLAS_IDS["CaB"] + ATLAS_IDS["CaT"]
    + ATLAS_IDS["Pu"] + ATLAS_IDS["NAC"] + ATLAS_IDS["GP"]
)
STR_IDS = (
    ATLAS_IDS["CaH"] + ATLAS_IDS["CaB"] + ATLAS_IDS["CaT"]
    + ATLAS_IDS["Pu"] + ATLAS_IDS["NAC"]
)

REGION_TO_ATLAS_MASK = {
    "CaH": "CaH",
    "CaB": "CaB",
    "CaT": "CaT",
    "PuR": "Pu",
    "PuC": "Pu",
    "PuPV": "Pu",
    "NACc": "NAC",
    "NACs": "NAC",
    "VeP": "GP",
}

GROSS_TO_DIRECT_REGIONS = OrderedDict([
    ("Caudate", ("CaH", "CaB", "CaT")),
    ("Pu", PUT_REGIONS),
    ("NAC", NAC_FINE_REGIONS),
    ("GP", ("VeP",)),
])

DISPLAY_COLUMNS = OrderedDict([
    ("Sox6:Tafa1", ["Sox6:Tafa1"]),
    ("Sox6:Vcan", ["Sox6:Vcan"]),
    ("Sox6:Tmem132d", ["Sox6:Tmem132d"]),
    ("Sox6:Kcnmb2", ["Sox6:Kcnmb2"]),
])
DISPLAY_LABELS: dict[str, str] = {}
CALB1_SPLIT_PAYLOAD: dict = {}

DIVERGING_CMAP = plt.get_cmap("coolwarm")
RAW_CMAP = DIVERGING_CMAP

DIVERGING_PER_COLUMN_ALPHA = False

SUPPORT_ALPHA_MASKS: "dict | None" = None

INFERNO_VISIBLE_FLOOR = LinearSegmentedColormap.from_list(
    "inferno_visible_floor",
    plt.get_cmap("inferno")(np.linspace(0.18, 1.0, 256)),
)

STYLE_MODE = "signed_diverging"

WARM_CMAP = LinearSegmentedColormap.from_list(
    "warm_white",
    [
        (0.00, (1.0, 1.0, 1.0)),
        (0.10, (1.0, 0.85, 0.70)),
        (0.30, (0.95, 0.55, 0.15)),
        (0.55, (0.85, 0.25, 0.05)),
        (0.80, (0.65, 0.05, 0.05)),
        (1.00, (0.40, 0.00, 0.00)),
    ],
)

BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#222222"
LABEL_COLOR = "#555555"
OUTLINE_COLOR = "#333333"
CROSSHAIR_COLOR = "#888888"
Q_COLOR = "#666666"
VENT_COLOR = "#7098BB"


def safe_col(name: str) -> str:
    return (
        name.replace(":", "_")
        .replace("+", "pos")
        .replace("-", "neg")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _pad_to_canvas(arr: np.ndarray, target_h: int, target_w: int, fill_value=0):
    """Center-pad a 2D/3D array to (target_h, target_w) with constant fill."""
    if not isinstance(arr, np.ndarray):
        return arr
    h, w = arr.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    pad_lef = (target_w - w) // 2
    pad_rig = target_w - w - pad_lef
    if pad_top == 0 and pad_bot == 0 and pad_lef == 0 and pad_rig == 0:
        return arr
    if arr.ndim == 2:
        return np.pad(
            arr,
            ((pad_top, pad_bot), (pad_lef, pad_rig)),
            mode="constant",
            constant_values=fill_value,
        )
    if arr.ndim == 3:
        return np.pad(
            arr,
            ((pad_top, pad_bot), (pad_lef, pad_rig), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )
    raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def equalize_slice_canvases(
    slices: dict,
    fields: dict,
    intensity_fields: dict,
    display_columns_iter,
) -> None:
    """Pad slices (dapi, masks, fields) to a per-block canvas in-place."""
    display_columns_iter = list(display_columns_iter)

    def _group_slices_by_axis():
        coronal = {}
        axial = {}
        for skey, sdata in slices.items():
            axis = sdata.get("axis", "coronal")
            if axis == "axial":
                axial[skey] = sdata
            else:
                coronal[skey] = sdata
        return coronal, axial

    def _equalize_block(block_slices: dict) -> None:
        if not block_slices:
            return
        max_h = max(sd["shape"][0] for sd in block_slices.values())
        max_w = max(sd["shape"][1] for sd in block_slices.values())
        for skey, sdata in block_slices.items():
            res = float(sdata.get("res", 0.5))
            sdata["dapi"] = _pad_to_canvas(sdata["dapi"], max_h, max_w, fill_value=1.0)
            new_masks = {}
            for name, m in sdata.get("masks", {}).items():
                if isinstance(m, np.ndarray):
                    fv = False if m.dtype == bool else 0
                    new_masks[name] = _pad_to_canvas(m, max_h, max_w, fill_value=fv)
                else:
                    new_masks[name] = m
            sdata["masks"] = new_masks
            for key in ("direct_masks", "direct_masks_display"):
                block = sdata.get(key, {}) or {}
                new_block = {}
                for region, m in block.items():
                    if isinstance(m, np.ndarray):
                        new_block[region] = _pad_to_canvas(m, max_h, max_w, fill_value=False)
                    else:
                        new_block[region] = m
                sdata[key] = new_block
            sdata["shape"] = (max_h, max_w)
            sdata["extent"] = (0, max_w * res, max_h * res, 0)
            if SUPPORT_ALPHA_MASKS is not None and skey in SUPPORT_ALPHA_MASKS:
                SUPPORT_ALPHA_MASKS[skey] = _pad_to_canvas(
                    SUPPORT_ALPHA_MASKS[skey], max_h, max_w, fill_value=False
                )
            for group_name in display_columns_iter:
                key = (skey, group_name)
                if key in fields:
                    fields[key] = _pad_to_canvas(fields[key], max_h, max_w, fill_value=0.0)
                if key in intensity_fields:
                    intensity_fields[key] = _pad_to_canvas(
                        intensity_fields[key], max_h, max_w, fill_value=0.0
                    )

    all_slices = {**_group_slices_by_axis()[0], **_group_slices_by_axis()[1]}
    _equalize_block(all_slices)

    for skey, sdata in all_slices.items():
        masks_dict = sdata.get("masks", {})
        brain = masks_dict.get("brain")
        striatum = None
        for key in ("Caudate", "Pu", "NAC"):
            m = masks_dict.get(key)
            if isinstance(m, np.ndarray) and m.any():
                striatum = m if striatum is None else (striatum | m)
        signal = masks_dict.get("signal")
        lv = masks_dict.get("vent")
        if striatum is not None:
            anchor = striatum
        elif signal is not None and lv is not None and isinstance(signal, np.ndarray):
            anchor = signal | lv
        elif signal is not None:
            anchor = signal
        elif brain is not None:
            anchor = brain
        else:
            anchor = None
        res = float(sdata.get("res", 0.5))
        h, w = sdata["shape"]

        def _bbox_from_mask(mask):
            if mask is None or not isinstance(mask, np.ndarray) or not mask.any():
                return None
            coords = np.argwhere(mask)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            x0, x1 = c0 * res, (c1 + 1) * res
            y0, y1 = r0 * res, (r1 + 1) * res
            return (x0, x1, y1, y0), ((x0 + x1) / 2.0, (y0 + y1) / 2.0), (x1 - x0, y1 - y0)

        brain_info = _bbox_from_mask(brain)
        if brain_info is None:
            sdata["brain_bbox_mm"] = (0.0, w * res, h * res, 0.0)
            sdata["brain_centroid_mm"] = (w * res / 2.0, h * res / 2.0)
            sdata["brain_extent_mm"] = (w * res, h * res)
        else:
            sdata["brain_bbox_mm"], sdata["brain_centroid_mm"], sdata["brain_extent_mm"] = brain_info

        signal_info = _bbox_from_mask(anchor)
        if signal_info is None:
            sdata["signal_bbox_mm"] = sdata["brain_bbox_mm"]
            sdata["signal_centroid_mm"] = sdata["brain_centroid_mm"]
            sdata["signal_extent_mm"] = sdata["brain_extent_mm"]
        else:
            sdata["signal_bbox_mm"], sdata["signal_centroid_mm"], sdata["signal_extent_mm"] = signal_info


def read_panel_a_snapshot(name: str, **kwargs) -> tuple[pd.DataFrame, str]:
    data = (PANEL_A_SNAPSHOT_DIR / name).read_bytes()
    return pd.read_csv(io.BytesIO(data), **kwargs), hashlib.sha256(data).hexdigest()


def load_panel_a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    panel, panel_hash = read_panel_a_snapshot(
        "projection_matrix_human.tsv", sep="\t", index_col=0,
    )
    bin_level, bin_hash = read_panel_a_snapshot(
        "projection_bin_level.tsv", sep="\t", index_col=0,
    )
    if not panel_hash.startswith(PANEL_A_CANONICAL_SHA256_PREFIX):
        raise RuntimeError(
            "Loaded Panel A snapshot does not match canonical Figure 11 hash"
        )
    if not bin_hash.startswith(PANEL_A_BIN_CANONICAL_SHA256_PREFIX):
        raise RuntimeError(
            "Loaded Panel A bin-level snapshot does not match canonical Figure 11 hash"
        )
    bin_meta = pd.read_csv(PANEL_A_BIN_META)
    hashes = {
        "panel_a_sha256": panel_hash,
        "panel_a_bin_level_sha256": bin_hash,
    }
    return panel, bin_level, bin_meta, hashes


def build_group_score_lookup(
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, pd.Series]]]:
    group_names = sorted(bin_meta["group_name"].astype(str).unique())
    lookup: dict[str, dict[str, pd.Series]] = {}
    for subtype in bin_level.index:
        lookup[subtype] = {}
        for region, rows in bin_meta.groupby("region"):
            scores = pd.Series(0.0, index=group_names, dtype=float)
            for row in rows.itertuples(index=False):
                col = f"bin_{int(row.bin_id)}"
                scores.loc[str(row.group_name)] = float(bin_level.loc[subtype, col])
            lookup[subtype][str(region)] = scores
    return group_names, lookup


def partition_metrics(
    dist_sq: np.ndarray,
    labels: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    sizes = pd.Series(labels).value_counts().sort_index().astype(int).tolist()
    sample_sil = silhouette_samples(dist_sq, labels, metric="precomputed")
    return {
        "silhouette": float(silhouette_score(dist_sq, labels, metric="precomputed")),
        "min_silhouette": float(np.min(sample_sil)),
        "n_negative_silhouettes": int((sample_sil < 0).sum()),
        "sizes": sizes,
        "per_sample_silhouette": {
            str(names[i]): float(sample_sil[i]) for i in range(len(names))
        },
    }


def column_z_values(frame: pd.DataFrame) -> np.ndarray:
    values = frame.astype(float).to_numpy()
    mean = np.nanmean(values, axis=0, keepdims=True)
    sd = np.nanstd(values, axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return np.nan_to_num((values - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_euclidean_distance(values: np.ndarray) -> np.ndarray:
    distances = pdist(values, metric="euclidean")
    if not np.all(np.isfinite(distances)) or np.nanmax(distances) <= 0:
        return np.zeros((values.shape[0], values.shape[0]), dtype=float)
    distance = squareform(distances)
    distance /= float(distance.max())
    np.fill_diagonal(distance, 0.0)
    return distance


def combined_calb1_distance(
    calb1: pd.DataFrame,
    spatial_shape: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Single transparent Calb1 distance: 50 percent regional, 50 percent spatial."""
    regional_values = column_z_values(calb1)
    spatial_values = column_z_values(spatial_shape)
    regional_distance = normalized_euclidean_distance(regional_values)
    spatial_distance = normalized_euclidean_distance(spatial_values)
    distance = 0.5 * regional_distance + 0.5 * spatial_distance
    np.fill_diagonal(distance, 0.0)
    return distance, {
        "regional_weight": 0.5,
        "spatial_weight": 0.5,
        "regional_features": list(calb1.columns),
        "spatial_features": int(spatial_shape.shape[1]),
        "regional_transform": "column z-score across Calb1 subtypes",
        "spatial_transform": "column z-score across Calb1 subtypes",
        "distance": "average of max-normalized Euclidean distance matrices",
    }


def build_calb1_spatial_texture(
    calb1: pd.DataFrame,
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Build Calb1 H4-bin texture used only for within-region variation."""
    _group_names, lookup = build_group_score_lookup(bin_level, bin_meta)
    meta = pd.read_csv(H4_BIN_METADATA)
    h4_regions = [
        region for region in CALB1_CONSENSUS_REGION_COLUMNS
        if region in set(meta["region"].astype(str))
    ]
    meta = meta[meta["region"].astype(str).isin(h4_regions)].copy()
    meta = meta.sort_values(["region", "bin_id"]).reset_index(drop=True)
    feature_ids = [
        f"h4_{int(row.bin_id):04d}_{str(row.region)}"
        for row in meta.itertuples(index=False)
    ]
    feature_regions = pd.Series(
        meta["region"].astype(str).to_numpy(),
        index=feature_ids,
        name="region",
    )
    weights = pd.Series(
        meta["n_cells"].astype(float).to_numpy(),
        index=feature_ids,
        name="n_cells",
    )
    raw = pd.DataFrame(0.0, index=calb1.index, columns=feature_ids)
    for subtype in calb1.index:
        vals = []
        for row in meta.itertuples(index=False):
            scores = lookup.get(subtype, {}).get(str(row.region))
            vals.append(0.0 if scores is None else float(scores.get(str(row.group_name), 0.0)))
        raw.loc[subtype] = np.maximum(np.asarray(vals, dtype=float), 0.0)

    shape = pd.DataFrame(1.0, index=calb1.index, columns=feature_ids)
    for subtype in calb1.index:
        for region in h4_regions:
            cols = list(feature_regions.index[feature_regions.eq(region)])
            local = raw.loc[subtype, cols].to_numpy(dtype=float)
            local_weights = weights.loc[cols].to_numpy(dtype=float)
            if np.any(local > 0):
                mean_shape = float(np.sum(local * local_weights) / np.sum(local_weights))
                if np.isfinite(mean_shape) and mean_shape > 0:
                    ratio = local / mean_shape
                    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                    mean_ratio = float(np.sum(ratio * local_weights) / np.sum(local_weights))
                    if np.isfinite(mean_ratio) and mean_ratio > 0:
                        ratio = ratio / mean_ratio
                    else:
                        ratio = np.ones(len(cols), dtype=float)
                else:
                    ratio = np.ones(len(cols), dtype=float)
            else:
                ratio = np.ones(len(cols), dtype=float)
            shape.loc[subtype, cols] = ratio

    summary = {
        "h4_spatial_regions": h4_regions,
        "h4_spatial_bins": int(len(meta)),
        "h4_spatial_bins_by_region": {
            str(k): int(v) for k, v in meta.groupby("region").size().items()
        },
        "spatial_shape_ratio_clip": "none for clustering",
        "spatial_texture_source": (
            "canonical projection_bin_level mapped by region and group_name "
            "onto H4 spatial bins"
        ),
        "spatial_texture_definition": "unit-mean within each H4 region",
    }
    return shape, feature_regions, summary


def calb1_simple_partition(
    distance: np.ndarray,
    names: list[str],
    k: int,
) -> tuple[np.ndarray, dict[str, object]] | None:
    if np.nanmax(distance) <= 0:
        return None
    labels = fcluster(
        linkage(squareform(distance), method="average"),
        t=k,
        criterion="maxclust",
    )
    labels = np.asarray(labels, dtype=int)
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    metrics = partition_metrics(distance, labels, names)
    metrics["sizes"] = pd.Series(labels).value_counts().sort_index().astype(int).tolist()
    metrics["has_singleton_cluster"] = bool(pd.Series(labels).value_counts().min() == 1)
    return labels, metrics


def normalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    mapping = {
        old: new_i + 1
        for new_i, old in enumerate(
            sorted(set(labels), key=lambda x: np.where(labels == x)[0][0])
        )
    }
    return np.asarray([mapping[int(x)] for x in labels], dtype=int)


def calb1_cluster_region_qc(
    calb1: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    qc: dict[str, dict[str, float]] = {}
    for ci in sorted(set(labels)):
        members = [calb1.index[j] for j in range(len(calb1)) if labels[j] == ci]
        mean_profile = calb1.loc[members].mean(axis=0)
        qc[f"Calb1 cluster {ci}"] = {
            str(region): float(value) for region, value in mean_profile.items()
        }
    return qc


def calb1_pairwise_distance_qc(
    distance: np.ndarray,
    names: list[str],
) -> dict[str, dict[str, float]]:
    return {
        str(names[i]): {
            str(names[j]): float(distance[i, j])
            for j in range(len(names))
        }
        for i in range(len(names))
    }


def compute_calb1_split(
    panel: pd.DataFrame,
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
) -> dict:
    available_cols = [c for c in CALB1_CONSENSUS_REGION_COLUMNS if c in panel.columns]
    calb1 = panel.loc[
        [r for r in panel.index if r.startswith("Calb1:")],
        available_cols,
    ].copy()
    if len(calb1) < 2:
        return {
            "K": int(len(calb1)),
            "method": "not clustered, fewer than two Calb1 subtypes",
            "clusters": {"Calb1": list(calb1.index)},
        }

    names = list(calb1.index)
    spatial_shape, _spatial_regions, spatial_summary = build_calb1_spatial_texture(
        calb1, bin_level, bin_meta,
    )
    distance, distance_summary = combined_calb1_distance(calb1, spatial_shape)
    Z = linkage(squareform(distance), method="average")
    merge_distances = Z[:, 2]
    threshold = float(
        merge_distances.mean() + CALB1_MOJENA_K * merge_distances.std(ddof=1)
    )
    n_keep = int((merge_distances <= threshold).sum())
    mojena_k = max(len(calb1) - n_keep, 2)
    metrics: dict[str, object] = {
        "mojena_pick": int(mojena_k),
        "mojena_threshold": threshold,
        "merge_distances": [float(x) for x in merge_distances],
        "distance_input": distance_summary,
        "spatial_input": spatial_summary,
        "pairwise_distance": calb1_pairwise_distance_qc(distance, names),
    }
    solutions: dict[int, tuple[np.ndarray, dict[str, object]]] = {}
    for k in CALB1_K_RANGE:
        result = calb1_simple_partition(distance, names, k)
        if result is None:
            continue
        labels, one_metrics = result
        labels = normalize_cluster_labels(labels)
        one_metrics["partition_source"] = "average_linkage_on_single_combined_distance"
        one_metrics["selection_score"] = float(one_metrics["silhouette"])
        one_metrics["cluster_region_means"] = calb1_cluster_region_qc(calb1, labels)
        solutions[k] = (np.asarray(labels, dtype=int), one_metrics)
        metrics[f"K={k}"] = one_metrics

    if not solutions:
        raise RuntimeError("No Calb1 K produced a non-degenerate partition")
    chosen_k = max(
        solutions,
        key=lambda k: (
            float(solutions[k][1]["silhouette"]),
            -int(k),
        ),
    )

    chosen_labels = solutions[chosen_k][0]
    clusters: OrderedDict[str, list[str]] = OrderedDict()
    for ci in sorted(set(chosen_labels)):
        members = [calb1.index[j] for j in range(len(calb1)) if chosen_labels[j] == ci]
        clusters[f"Calb1 cluster {ci}"] = members

    metrics["chosen_K"] = int(chosen_k)
    metrics["candidate_K"] = [int(k) for k in sorted(solutions)]
    metrics["chosen_has_singleton_cluster"] = bool(
        metrics[f"K={chosen_k}"]["has_singleton_cluster"]
    )
    return {
        "K": int(chosen_k),
        "method": (
            "Simple unsupervised Calb1 clustering from one equal-weight "
            "Panel A regional distance and one equal-weight H4 within-region "
            "spatial texture distance"
        ),
        "selection_criterion": (
            "Column-z-score the Panel A Calb1 regional matrix and the H4 "
            "unit-mean within-region texture matrix. Compute Euclidean "
            "distances within each block, max-normalize each distance matrix, "
            "average them 50/50, run average-linkage clustering, and choose "
            "the K with the highest silhouette on that final distance. "
            "Cluster labels are neutral and are not named by target regions."
        ),
        "input": "loaded Panel A matrix plus canonical H4 within-region bin texture",
        "metrics": metrics,
        "clusters": clusters,
    }


def load_display_columns(
    panel: pd.DataFrame,
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
) -> OrderedDict[str, list[str]]:
    columns: OrderedDict[str, list[str]] = OrderedDict()
    for subtype in ("Sox6:Tafa1", "Sox6:Vcan", "Sox6:Kcnmb2"):
        columns[subtype] = [subtype]
    columns["Rest Sox6"] = [
        s for s in panel.index
        if s.startswith("Sox6:")
        and s not in {"Sox6:Tafa1", "Sox6:Vcan", "Sox6:Kcnmb2"}
    ]
    payload = compute_calb1_split(panel, bin_level, bin_meta)
    CALB1_SPLIT_PAYLOAD.clear()
    CALB1_SPLIT_PAYLOAD.update(payload)
    for name, members in payload["clusters"].items():
        columns[name] = [m for m in members if m in panel.index]
    return columns


def label_for_group(group_name: str, members: list[str]) -> str:
    genes = [m.split(":", 1)[1] if ":" in m else m for m in members]
    def _wrap_genes(gs: list[str]) -> str:
        if len(gs) <= 2:
            return "/".join(gs)
        mid = (len(gs) + 1) // 2
        return "/".join(gs[:mid]) + "/\n" + "/".join(gs[mid:])

    if group_name == "Sox6:Tafa1":
        return "Anxa1+\n(Tafa1)"
    if group_name == "Sox6:Vcan":
        return "Anxa1+\n(Vcan)"
    if group_name == "Sox6:Kcnmb2":
        return "Sox6+ Aldh1a1+\n(Kcnmb2)"
    if group_name == "Rest Sox6":
        return f"Sox6+ Aldh1a1-\n({_wrap_genes(genes)})"
    if group_name.startswith("Calb1 cluster"):
        heading = group_name.replace("Calb1", "Calb1+")
        return f"{heading}\n({_wrap_genes(genes)})"
    return group_name


def refresh_display_labels() -> None:
    DISPLAY_LABELS.clear()
    for group_name, members in DISPLAY_COLUMNS.items():
        DISPLAY_LABELS[group_name] = label_for_group(group_name, members)


def verify_panel_a_rebuild(
    panel: pd.DataFrame,
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
) -> float:
    rebuilt = {}
    for region, rows in bin_meta.groupby("region"):
        cols = [f"bin_{int(i)}" for i in rows["bin_id"]]
        rebuilt[region] = bin_level[cols].mean(axis=1)
    rebuilt_df = pd.DataFrame(rebuilt)[panel.columns]
    rebuilt_df = rebuilt_df.div(rebuilt_df.sum(axis=1), axis=0)
    return float((rebuilt_df - panel).abs().max().max())


def atlas_left_striatum_centroid_ap(atlas: BrainGlobeAtlas) -> float:
    ann = atlas.annotation
    res = atlas.resolution[0] / 1000.0
    midline = ann.shape[2] // 2
    vox = np.argwhere(np.isin(ann, STR_IDS))
    vox = vox[vox[:, 2] >= midline]
    return float(vox[:, 0].mean() * res)


def atlas_left_striatum_centroid_dv(atlas: BrainGlobeAtlas) -> float:
    """DV centroid (mm) of the left-striatum mass in atlas voxel space."""
    ann = atlas.annotation
    res = atlas.resolution[1] / 1000.0
    midline = ann.shape[2] // 2
    vox = np.argwhere(np.isin(ann, STR_IDS))
    vox = vox[vox[:, 2] >= midline]
    return float(vox[:, 1].mean() * res)


def calibrate_dv_sign(atlas: BrainGlobeAtlas, cells_z_mean: float, dv_centroid_mm: float) -> int:
    """Return +1 if positive z_ccf_mm is INFERIOR in atlas, -1 if SUPERIOR."""
    return -1


def finalize_spatial_bins(bins: pd.DataFrame, ac_ap_mm: float) -> pd.DataFrame:
    for axis, col in (("x", "x_ccf_mm"), ("y", "y_ccf_mm"), ("z", "z_ccf_mm")):
        if f"{axis}_bin" not in bins.columns:
            bins[f"{axis}_bin"] = np.floor(bins[col] / VOXEL_SIZE_MM).astype(np.int32)
    bins["atlas_ap_mm"] = ac_ap_mm - bins["y_ccf_mm"].astype(float)
    return bins


def load_h4_texture_bins(
    score_cols: list[str],
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
    ac_ap_mm: float,
) -> pd.DataFrame:
    _group_names, lookup = build_group_score_lookup(bin_level, bin_meta)
    meta = pd.read_csv(H4_BIN_METADATA)
    meta = meta[meta["region"].isin([r for r in DIRECT_REGIONS if r != "VeP"])].copy()
    meta = meta.rename(columns={
        "x_ccf_mean": "x_ccf_mm",
        "y_ccf_mean": "y_ccf_mm",
        "z_ccf_mean": "z_ccf_mm",
    })
    bins = meta[[
        "bin_id", "region", "group_name", "n_cells", "x_ccf_mm", "y_ccf_mm", "z_ccf_mm",
    ]].copy()
    bins["bin_id"] = bins["bin_id"].astype(int)
    for col in ("x_ccf_mm", "y_ccf_mm", "z_ccf_mm"):
        bins[col] = pd.to_numeric(bins[col], errors="coerce")
    bins = bins.dropna(subset=["x_ccf_mm", "y_ccf_mm", "z_ccf_mm"])
    for subtype in score_cols:
        values = []
        for row in bins.itertuples(index=False):
            scores = lookup.get(subtype, {}).get(str(row.region))
            if scores is None:
                values.append(1.0)
            else:
                values.append(float(scores.get(str(row.group_name), 0.0)))
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        bins[subtype] = np.maximum(values, 0.0)
    bins["source_texture"] = "panel_a_bin_level_on_h4_spatial_bins"
    return finalize_spatial_bins(bins, ac_ap_mm)


def load_vep_neutral_bins(score_cols: list[str], ac_ap_mm: float) -> pd.DataFrame:
    cols = ["region", "hemisphere", "x_ccf_mm", "y_ccf_mm", "z_ccf_mm"]
    cells = pd.read_parquet(VE_P_COORDS, columns=cols)
    cells = cells[
        cells["hemisphere"].astype(str).str.lower().eq("left")
        & cells["region"].astype(str).eq("VeP")
    ].copy()
    for axis, col in (("x", "x_ccf_mm"), ("y", "y_ccf_mm"), ("z", "z_ccf_mm")):
        cells[f"{axis}_bin"] = np.floor(cells[col] / VOXEL_SIZE_MM).astype(np.int32)
    bins = (
        cells.groupby(["region", "x_bin", "y_bin", "z_bin"], observed=True)
        .agg(
            n_cells=("region", "size"),
            x_ccf_mm=("x_ccf_mm", "mean"),
            y_ccf_mm=("y_ccf_mm", "mean"),
            z_ccf_mm=("z_ccf_mm", "mean"),
        )
        .reset_index()
    )
    bins = bins[bins["n_cells"] >= MIN_CELLS_PER_BIN].copy()
    bins["bin_id"] = -1
    for subtype in score_cols:
        bins[subtype] = 1.0
    bins["source_texture"] = "neutral_vep"
    return finalize_spatial_bins(bins, ac_ap_mm)


def cluster_cells_to_region_voxels(
    cells: pd.DataFrame,
    region: str,
    group_names: list[str],
) -> pd.DataFrame:
    region_cells = cells[cells["region"].astype(str).eq(region)].copy()
    if region_cells.empty:
        return pd.DataFrame()
    coords = region_cells[["x_ccf_mm", "y_ccf_mm", "z_ccf_mm"]].to_numpy(dtype=float)
    n_clusters = min(VOXELS_PER_REGION, len(region_cells))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        batch_size=min(8192, len(region_cells)),
        n_init=10,
    )
    region_cells["region_voxel"] = kmeans.fit_predict(coords)
    meta = (
        region_cells.groupby("region_voxel", observed=True)
        .agg(
            n_cells=("region", "size"),
            x_ccf_mm=("x_ccf_mm", "mean"),
            y_ccf_mm=("y_ccf_mm", "mean"),
            z_ccf_mm=("z_ccf_mm", "mean"),
        )
        .reset_index()
    )
    counts = (
        region_cells.groupby(["region_voxel", "group_name"], observed=True)
        .size()
        .unstack("group_name", fill_value=0)
        .reindex(columns=group_names, fill_value=0)
        .reset_index()
    )
    out = meta.merge(counts, on="region_voxel", how="left")
    out.insert(0, "region", region)
    out = out.sort_values(["y_ccf_mm", "x_ccf_mm", "z_ccf_mm"]).reset_index(drop=True)
    out["region_voxel"] = np.arange(len(out), dtype=int)
    return out


def add_panel_a_local_texture(
    bins: pd.DataFrame,
    panel: pd.DataFrame,
    group_names: list[str],
    lookup: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    out = bins.copy()
    group_counts = out[group_names].to_numpy(dtype=float)
    n_cells = out["n_cells"].to_numpy(dtype=float)
    props = group_counts / np.maximum(n_cells[:, None], 1.0)
    region_arr = out["region"].astype(str).to_numpy()

    for subtype in panel.index:
        shape = np.ones(len(out), dtype=float)
        for region in DIRECT_REGIONS:
            idx = np.where(region_arr == region)[0]
            if len(idx) == 0:
                continue
            scores = lookup.get(subtype, {}).get(region)
            if scores is None:
                scores = pd.Series(1.0, index=group_names, dtype=float)
            score_vec = scores.reindex(group_names).fillna(0.0).to_numpy(dtype=float)
            local = props[idx].dot(np.maximum(score_vec, 0.0))
            local = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.any(local > 0):
                local = np.ones(len(idx), dtype=float)
            shape[idx] = np.maximum(local, 0.0)
        out[subtype] = shape
    return out


def load_spatial_bins(
    panel: pd.DataFrame,
    bin_level: pd.DataFrame,
    bin_meta: pd.DataFrame,
    ac_ap_mm: float,
) -> pd.DataFrame:
    score_cols = list(panel.index)
    bins = pd.concat(
        [
            load_h4_texture_bins(score_cols, bin_level, bin_meta, ac_ap_mm),
            load_vep_neutral_bins(score_cols, ac_ap_mm),
        ],
        ignore_index=True,
        sort=False,
    )
    bins = bins.sort_values(["region", "y_ccf_mm", "x_ccf_mm", "z_ccf_mm"]).reset_index(drop=True)
    bins.insert(0, "spatial_bin_id", [f"pact_{i:04d}" for i in range(len(bins))])
    bins["region_voxel"] = bins.groupby("region", observed=True).cumcount().astype(int)
    return bins


def target_region_totals(panel: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    target = panel.reindex(columns=list(DIRECT_REGIONS)).copy()
    fine_counts = bins.groupby("region")["n_cells"].sum()
    denom = float(fine_counts.reindex(NAC_FINE_REGIONS).fillna(0).sum())
    if "NAC" in panel.columns and denom > 0:
        for region in NAC_FINE_REGIONS:
            target[region] = target[region] + panel["NAC"] * float(fine_counts.get(region, 0)) / denom
    return target


def panel_a_display_targets(target: pd.DataFrame) -> pd.DataFrame:
    if DISPLAY_TARGET_MODE == "panel_a_raw":
        display = target.copy()
    elif DISPLAY_TARGET_MODE == "panel_a_row_log_ratio":
        row_mean = target.mean(axis=1).replace(0, np.nan)
        ratio = target.div(row_mean, axis=0).replace(0, np.nan)
        display = np.log2(ratio).fillna(0.0)
    elif DISPLAY_TARGET_MODE == "panel_a_excess_uniform":
        row_total = target.sum(axis=1).replace(0, np.nan)
        expected = row_total / target.shape[1]
        display = target.sub(expected, axis=0)
    elif DISPLAY_TARGET_MODE == "panel_a_column_z":
        sd = target.std(axis=0, ddof=1).replace(0, np.nan)
        display = (target - target.mean(axis=0)) / sd
    elif DISPLAY_TARGET_MODE == "panel_a_row_z":
        sd = target.std(axis=1, ddof=1).replace(0, np.nan)
        display = target.sub(target.mean(axis=1), axis=0).div(sd, axis=0)
    elif DISPLAY_TARGET_MODE == "panel_a_family_z":
        display = pd.DataFrame(0.0, index=target.index, columns=target.columns)
        families = pd.Series({idx: display_family(str(idx)) for idx in target.index})
        for family, rows in families.groupby(families).groups.items():
            fam = target.loc[list(rows)]
            if len(fam) < 2:
                display.loc[list(rows)] = 0.0
                continue
            sd = fam.std(axis=0, ddof=1).replace(0, np.nan)
            display.loc[list(rows)] = fam.sub(fam.mean(axis=0), axis=1).div(sd, axis=1)
    else:
        raise ValueError(f"Unknown display target mode: {DISPLAY_TARGET_MODE}")
    return display.fillna(0.0)


def display_family(name: str) -> str:
    if name == "Rest Sox6" or name.startswith("Sox6:"):
        return "Sox6"
    if name.startswith("Calb1"):
        return "Calb1"
    return name.split(":", 1)[0]


def display_targets_for_columns(
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subtype_display_target = panel_a_display_targets(target)
    group_target = {}
    group_display_target = {}
    for group_name, members in DISPLAY_COLUMNS.items():
        valid = [member for member in members if member in target.index]
        if valid:
            group_target[group_name] = target.loc[valid].mean(axis=0)
            group_display_target[group_name] = subtype_display_target.loc[valid].mean(axis=0)
        else:
            group_target[group_name] = pd.Series(0.0, index=target.columns)
            group_display_target[group_name] = pd.Series(0.0, index=target.columns)
    return pd.DataFrame(group_target).T, pd.DataFrame(group_display_target).T


def panel_a_concordance_table(
    target: pd.DataFrame,
) -> pd.DataFrame:
    subtype_display_target = panel_a_display_targets(target)
    group_target, group_display_target = display_targets_for_columns(target)
    rows = []
    for group_name, members in DISPLAY_COLUMNS.items():
        valid = [member for member in members if member in target.index]
        if not valid:
            continue
        expected_display = subtype_display_target.loc[valid].mean(axis=0)
        expected_raw = target.loc[valid].mean(axis=0)
        for region in group_target.columns:
            rows.append({
                "group_name": group_name,
                "region": region,
                "n_member_subtypes": len(valid),
                "member_subtypes": ";".join(valid),
                "group_raw_target": float(group_target.loc[group_name, region]),
                "expected_raw_from_panel_a": float(expected_raw.loc[region]),
                "raw_diff": float(group_target.loc[group_name, region] - expected_raw.loc[region]),
                "group_display_target": float(group_display_target.loc[group_name, region]),
                "expected_display_from_panel_a": float(expected_display.loc[region]),
                "display_diff": float(
                    group_display_target.loc[group_name, region] - expected_display.loc[region]
                ),
            })
    out = pd.DataFrame(rows)
    max_raw = float(out["raw_diff"].abs().max()) if len(out) else 0.0
    max_display = float(out["display_diff"].abs().max()) if len(out) else 0.0
    if max(max_raw, max_display) > 1e-12:
        raise RuntimeError(
            "Panel A concordance failed: grouped targets no longer equal "
            "the mean of member subtype Panel A values"
        )
    return out


def unit_mean_shape(shape: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_shape = float(np.sum(shape * weights) / np.sum(weights))
    if not np.isfinite(mean_shape) or mean_shape <= 0:
        return np.ones(len(shape), dtype=float)
    ratio = shape / mean_shape
    ratio = np.clip(ratio, SHAPE_RATIO_CLIP[0], SHAPE_RATIO_CLIP[1])
    mean_ratio = float(np.sum(ratio * weights) / np.sum(weights))
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        return np.ones(len(shape), dtype=float)
    return ratio / mean_ratio


def constrain_to_panel_a(
    panel: pd.DataFrame,
    bins: pd.DataFrame,
    score_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = bins.copy()
    target = target_region_totals(panel, out)
    display_target = panel_a_display_targets(target)
    n_cells = out["n_cells"].to_numpy(dtype=float)
    region_arr = out["region"].astype(str).to_numpy()

    for subtype in panel.index:
        shape_col = subtype if subtype in score_cols else None
        if shape_col is None:
            shape = np.ones(len(out), dtype=float)
        else:
            shape = out[shape_col].to_numpy(dtype=float)
            shape = np.maximum(shape, 0.0)
        mass = np.zeros(len(out), dtype=float)
        display = np.zeros(len(out), dtype=float)
        for region in DIRECT_REGIONS:
            idx = np.where(region_arr == region)[0]
            if len(idx) == 0:
                continue
            total = float(target.loc[subtype, region])
            display_total = float(display_target.loc[subtype, region])
            region_cells = n_cells[idx]
            shape_ratio = unit_mean_shape(shape[idx], region_cells)
            local_score = total * shape_ratio
            local_display = display_total * shape_ratio
            mass[idx] = local_score * region_cells / np.sum(region_cells)
            display[idx] = local_display
            out.loc[out.index[idx], f"score_{safe_col(subtype)}"] = local_score
            out.loc[out.index[idx], f"display_{safe_col(subtype)}"] = local_display
        out[f"mass_{safe_col(subtype)}"] = mass
        out[f"density_{safe_col(subtype)}"] = mass / np.maximum(n_cells, 1.0)
        out[f"display_{safe_col(subtype)}"] = display
    return out, target, display_target


def group_value(bins: pd.DataFrame, members: list[str], prefix: str) -> np.ndarray:
    cols = [f"{prefix}_{safe_col(m)}" for m in members if f"{prefix}_{safe_col(m)}" in bins.columns]
    if not cols:
        return np.zeros(len(bins), dtype=float)
    return bins[cols].mean(axis=1).to_numpy(dtype=float)


def _mri_dapi(ref_slice: np.ndarray, brain_mask: np.ndarray, vent_mask: np.ndarray) -> np.ndarray:
    """Return an MRI-look RGB canvas from the Allen Human T1 reference."""
    ref_crop = ref_slice.astype(float)
    canvas = np.ones((*ref_crop.shape, 3), dtype=float)
    positive = ref_crop[brain_mask & (ref_crop > 0)]
    if len(positive):
        lo, hi = np.percentile(positive, [1, 99])
        ref_norm = np.clip((ref_crop - lo) / max(hi - lo, 1e-9), 0, 1)
    else:
        ref_norm = np.zeros_like(ref_crop, dtype=float)
    gray = 0.10 + 0.85 * ref_norm[brain_mask]
    canvas[brain_mask] = np.stack([gray, gray, gray], axis=-1)
    if vent_mask is not None and vent_mask.any():
        canvas[vent_mask] = np.array([0.04, 0.04, 0.04])
    return canvas


def _full_hemisphere_crop_bounds(brain_full: np.ndarray, pad_vox: int) -> tuple[int, int, int, int]:
    """Bounding box of the whole-brain mask, padded, clipped to image."""
    coords = np.argwhere(brain_full)
    if len(coords) == 0:
        return 0, brain_full.shape[0], 0, brain_full.shape[1]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - pad_vox)
    r1 = min(brain_full.shape[0], int(r1) + pad_vox + 1)
    c0 = max(0, int(c0) - pad_vox)
    c1 = min(brain_full.shape[1], int(c1) + pad_vox + 1)
    return r0, r1, c0, c1


def _basal_ganglia_crop_bounds(
    brain_full: np.ndarray,
    anchor_mask: np.ndarray,
    pad_vox: int,
) -> tuple[int, int, int, int]:
    """Square-ish bbox around the anchor (signal + ventricles), padded and clipped to the brain mask."""
    if not anchor_mask.any():
        return _full_hemisphere_crop_bounds(brain_full, pad_vox)
    coords = np.argwhere(anchor_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    midline_col = brain_full.shape[1] // 2
    c_left = min(int(c0), 2 * midline_col - int(c1))
    c_right = max(int(c1), 2 * midline_col - int(c0))
    r0 = max(0, int(r0) - pad_vox)
    r1 = min(brain_full.shape[0], int(r1) + pad_vox + 1)
    c0 = max(0, c_left - pad_vox)
    c1 = min(brain_full.shape[1], c_right + pad_vox + 1)
    return r0, r1, c0, c1


def extract_slice(atlas: BrainGlobeAtlas, ap_mm: float) -> dict:
    """Full-hemisphere coronal slice at atlas-native AP mm (publication MRI grade)."""
    ann = atlas.annotation
    ref = atlas.reference
    res = atlas.resolution[0] / 1000.0
    midline = ann.shape[2] // 2
    ap_vox = int(round(ap_mm / res))
    ap_vox = max(0, min(ann.shape[0] - 1, ap_vox))
    coronal = ann[ap_vox, :, :]
    ref_slice = ref[ap_vox, :, :]

    masks_full = {name: np.isin(coronal, ids) for name, ids in ATLAS_IDS.items()}
    signal_full = np.isin(coronal, SIGNAL_IDS)
    brain_full = coronal > 0
    anchor = signal_full | masks_full["LV"]

    r0, r1, c0, c1 = _basal_ganglia_crop_bounds(brain_full, anchor, BG_CROP_PAD_VOX)

    def crop(arr: np.ndarray) -> np.ndarray:
        return arr[r0:r1, c0:c1]

    brain = crop(brain_full)
    vent = crop(masks_full["LV"])
    dapi = _mri_dapi(crop(ref_slice), brain, vent)

    masks = {name: crop(m) for name, m in masks_full.items()}
    masks["Caudate"] = masks["CaH"] | masks["CaB"] | masks["CaT"]
    masks["signal"] = crop(signal_full)
    masks["brain"] = brain
    masks["vent"] = vent
    left_col_cut = max(0, midline - c0)
    left_only_signal = masks["signal"].copy()
    left_only_signal[:, left_col_cut:] = False
    masks["signal_left"] = left_only_signal
    masks["left_col_cut"] = left_col_cut

    return {
        "ap_mm": float(ap_mm),
        "res": float(res),
        "dapi": dapi,
        "masks": masks,
        "shape": dapi.shape[:2],
        "extent": (0, dapi.shape[1] * res, dapi.shape[0] * res, 0),
        "axis": "coronal",
        "left_col_cut": left_col_cut,
    }


def extract_axial_slice(atlas: BrainGlobeAtlas, dv_mm: float) -> dict:
    """Full-brain axial slice at atlas-native DV mm."""
    ann = atlas.annotation
    ref = atlas.reference
    res = atlas.resolution[1] / 1000.0
    midline = ann.shape[2] // 2
    dv_vox = int(round(dv_mm / res))
    dv_vox = max(0, min(ann.shape[1] - 1, dv_vox))
    axial = ann[:, dv_vox, :]
    ref_slice = ref[:, dv_vox, :]

    masks_full = {name: np.isin(axial, ids) for name, ids in ATLAS_IDS.items()}
    signal_full = np.isin(axial, SIGNAL_IDS)
    brain_full = axial > 0
    anchor = signal_full | masks_full["LV"]

    r0, r1, c0, c1 = _basal_ganglia_crop_bounds(brain_full, anchor, BG_CROP_PAD_VOX)

    def crop(arr: np.ndarray) -> np.ndarray:
        return arr[r0:r1, c0:c1]

    brain = crop(brain_full)
    vent = crop(masks_full["LV"])
    dapi = _mri_dapi(crop(ref_slice), brain, vent)

    masks = {name: crop(m) for name, m in masks_full.items()}
    masks["Caudate"] = masks["CaH"] | masks["CaB"] | masks["CaT"]
    masks["signal"] = crop(signal_full)
    masks["brain"] = brain
    masks["vent"] = vent
    left_col_cut = max(0, midline - c0)
    left_only_signal = masks["signal"].copy()
    left_only_signal[:, left_col_cut:] = False
    masks["signal_left"] = left_only_signal
    masks["left_col_cut"] = left_col_cut

    return {
        "dv_mm": float(dv_mm),
        "res": float(res),
        "dapi": dapi,
        "masks": masks,
        "shape": dapi.shape[:2],
        "extent": (0, dapi.shape[1] * res, dapi.shape[0] * res, 0),
        "axis": "axial",
        "left_col_cut": left_col_cut,
    }


def region_source_limits(bins: pd.DataFrame) -> dict[str, dict[str, float]]:
    limits = {}
    for atlas_key, regions in GROSS_TO_DIRECT_REGIONS.items():
        df = bins[bins["region"].isin(regions)]
        if df.empty:
            continue
        x_lo, x_hi = df["x_ccf_mm"].quantile([0.02, 0.98])
        y_lo, y_hi = df["y_ccf_mm"].quantile([0.02, 0.98])
        z_lo, z_hi = df["z_ccf_mm"].quantile([0.02, 0.98])
        if x_hi <= x_lo:
            x_lo, x_hi = df["x_ccf_mm"].min() - 1, df["x_ccf_mm"].max() + 1
        if y_hi <= y_lo:
            y_lo, y_hi = df["y_ccf_mm"].min() - 1, df["y_ccf_mm"].max() + 1
        if z_hi <= z_lo:
            z_lo, z_hi = df["z_ccf_mm"].min() - 1, df["z_ccf_mm"].max() + 1
        one = {
            "x_lo": float(x_lo),
            "x_hi": float(x_hi),
            "y_lo": float(y_lo),
            "y_hi": float(y_hi),
            "z_lo": float(z_lo),
            "z_hi": float(z_hi),
        }
        limits[atlas_key] = one
        for region in regions:
            limits[str(region)] = one
    return limits


def snap_to_mask(row: np.ndarray, col: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pix = np.argwhere(mask)
    if len(pix) == 0 or len(row) == 0:
        return row, col
    tree = cKDTree(pix.astype(float))
    query = np.column_stack([row, col])
    _dist, idx = tree.query(query, k=1)
    snapped = pix[idx]
    return snapped[:, 0].astype(float), snapped[:, 1].astype(float)


def map_bins_to_mask(
    df: pd.DataFrame,
    mask: np.ndarray,
    limits: dict[str, float],
    *,
    flip_ml: bool = False,
    axis: str = "coronal",
) -> tuple[np.ndarray, np.ndarray]:
    """Project CCF-mm bin coordinates onto pixel (row, col) within a 2D mask."""
    coords = np.argwhere(mask)
    if len(coords) == 0 or len(df) == 0:
        return np.array([]), np.array([])
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)

    x = df["x_ccf_mm"].to_numpy(dtype=float)
    nx = (x - limits["x_lo"]) / max(limits["x_hi"] - limits["x_lo"], 1e-9)
    nx = np.clip(nx, 0, 1)

    if axis == "axial":
        y = df["y_ccf_mm"].to_numpy(dtype=float)
        ny = (limits["y_hi"] - y) / max(limits["y_hi"] - limits["y_lo"], 1e-9)
        ny = np.clip(ny, 0, 1)
        row_frac = ny
    else:
        z = df["z_ccf_mm"].to_numpy(dtype=float)
        nz = (limits["z_hi"] - z) / max(limits["z_hi"] - limits["z_lo"], 1e-9)
        nz = np.clip(nz, 0, 1)
        row_frac = nz

    row = rmin + row_frac * max(rmax - rmin, 1)
    if flip_ml:
        col = cmax - nx * max(cmax - cmin, 1)
    else:
        col = cmin + nx * max(cmax - cmin, 1)
    return snap_to_mask(row, col, mask)


def build_direct_region_masks(
    sdata: dict,
    slab: pd.DataFrame,
    limits: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    axis = sdata.get("axis", "coronal")
    left_col_cut = sdata.get("left_col_cut", sdata["shape"][1] // 2)
    out = {
        region: np.zeros(sdata["shape"], dtype=bool)
        for region in DIRECT_REGIONS
    }
    for atlas_key, regions in GROSS_TO_DIRECT_REGIONS.items():
        gross_full = sdata["masks"][atlas_key]
        if not gross_full.any() or atlas_key not in limits:
            continue
        gross = gross_full.copy()
        gross[:, left_col_cut:] = False
        if not gross.any():
            continue
        counts = {
            region: int(slab["region"].astype(str).eq(region).sum())
            for region in regions
        }
        present = [region for region, n in counts.items() if n >= MIN_BINS_FOR_REGION_MASK]
        if not present and any(n > 0 for n in counts.values()):
            present = [max(counts, key=counts.get)]
        if not present:
            continue
        if len(present) == 1:
            out[present[0]] = gross.copy()
            continue

        supports = []
        support_regions = []
        for region in present:
            region_df = slab[slab["region"].astype(str).eq(region)]
            row, col = map_bins_to_mask(region_df, gross, limits[atlas_key], axis=axis)
            if len(row) == 0:
                continue
            img = np.zeros(sdata["shape"], dtype=float)
            rr = np.rint(row).astype(int)
            cc = np.rint(col).astype(int)
            weights = region_df["n_cells"].to_numpy(dtype=float)
            ok = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
            np.add.at(img, (rr[ok], cc[ok]), weights[ok])
            img = gaussian_filter(img, sigma=REGION_MASK_ASSIGN_SIGMA_PX)
            img[~gross] = 0.0
            supports.append(img)
            support_regions.append(region)
        if not supports:
            continue

        pix = np.argwhere(gross)
        stack = np.stack([img[pix[:, 0], pix[:, 1]] for img in supports], axis=1)
        winners = np.argmax(stack, axis=1)
        for i, region in enumerate(support_regions):
            mask = np.zeros(sdata["shape"], dtype=bool)
            take = winners == i
            mask[pix[take, 0], pix[take, 1]] = True
            out[region] = mask
    return out


def splat(shape: tuple[int, int], row: np.ndarray, col: np.ndarray, weight: np.ndarray) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), dtype=float)
    if len(row) == 0:
        return img
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    dr = row - r0
    dc = col - c0
    for rr, cc, ww in (
        (r0, c0, (1 - dr) * (1 - dc)),
        (r0 + 1, c0, dr * (1 - dc)),
        (r0, c0 + 1, (1 - dr) * dc),
        (r0 + 1, c0 + 1, dr * dc),
    ):
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w) & np.isfinite(weight)
        np.add.at(img, (rr[ok], cc[ok]), weight[ok] * ww[ok])
    return img


def interpolated_region_field(
    shape: tuple[int, int],
    row: np.ndarray,
    col: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill a region mask from spatial bins and preserve the bin mass sum."""
    img = np.zeros(shape, dtype=float)
    total = float(np.nansum(weight))
    if total <= 0 or not mask.any():
        return img

    ok = np.isfinite(row) & np.isfinite(col) & np.isfinite(weight) & (weight > 0)
    row = row[ok]
    col = col[ok]
    weight = weight[ok]
    pix = np.argwhere(mask)
    if len(row) == 0 or len(pix) == 0:
        img[mask] = total / float(mask.sum())
        return img

    points = np.column_stack([row, col])
    if len(points) == 1:
        values = np.full(len(pix), float(weight[0]), dtype=float)
    else:
        k = min(INTERP_NEIGHBORS, len(points))
        tree = cKDTree(points)
        dist, idx = tree.query(pix.astype(float), k=k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if dist.shape[0] == 1 and len(pix) > 1:
            dist = dist.T
            idx = idx.T
        kernel = 1.0 / np.maximum(dist, 0.75) ** 2
        values = (kernel * weight[idx]).sum(axis=1) / kernel.sum(axis=1)
    img[pix[:, 0], pix[:, 1]] = values
    img = gaussian_filter(img, sigma=SMOOTH_SIGMA_PX)
    img[~mask] = 0.0
    after = float(img[mask].sum())
    if after > 0:
        img *= total / after
    else:
        img[mask] = total / float(mask.sum())
    return img


def interpolated_score_field(
    shape: tuple[int, int],
    row: np.ndarray,
    col: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    """Fill a region mask from signed bin values and preserve the mean."""
    img = np.zeros(shape, dtype=float)
    if not np.isfinite(target_mean) or not mask.any():
        return img

    ok = np.isfinite(row) & np.isfinite(col) & np.isfinite(value)
    row = row[ok]
    col = col[ok]
    value = value[ok]
    pix = np.argwhere(mask)
    if len(row) == 0 or len(pix) == 0:
        img[mask] = target_mean
        return img

    points = np.column_stack([row, col])
    if len(points) == 1:
        values = np.full(len(pix), float(value[0]), dtype=float)
    else:
        k = min(INTERP_NEIGHBORS, len(points))
        tree = cKDTree(points)
        dist, idx = tree.query(pix.astype(float), k=k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if dist.shape[0] == 1 and len(pix) > 1:
            dist = dist.T
            idx = idx.T
        kernel = 1.0 / np.maximum(dist, 0.75) ** 2
        values = (kernel * value[idx]).sum(axis=1) / kernel.sum(axis=1)
    img[pix[:, 0], pix[:, 1]] = values
    img = gaussian_filter(img, sigma=SMOOTH_SIGMA_PX)
    img[~mask] = 0.0
    after = float(img[mask].mean())
    if abs(after) > 1e-12:
        img *= target_mean / after
    else:
        img[mask] = target_mean
    return img


def _mirror_field_to_right(field: np.ndarray, left_col_cut: int) -> np.ndarray:
    """Mirror left-hemisphere field values onto the right hemisphere."""
    if left_col_cut <= 0 or left_col_cut >= field.shape[1]:
        return field
    out = field.copy()
    left = field[:, :left_col_cut]
    mirrored = np.fliplr(left)
    n_right_cols = field.shape[1] - left_col_cut
    if n_right_cols <= 0:
        return out
    width = min(mirrored.shape[1], n_right_cols)
    out[:, left_col_cut:left_col_cut + width] = mirrored[:, :width]
    return out


def _mirror_mask_to_right(mask: np.ndarray, left_col_cut: int) -> np.ndarray:
    if left_col_cut <= 0 or left_col_cut >= mask.shape[1]:
        return mask
    out = mask.copy()
    left = mask[:, :left_col_cut]
    mirrored = np.fliplr(left)
    n_right_cols = mask.shape[1] - left_col_cut
    width = min(mirrored.shape[1], n_right_cols)
    out[:, left_col_cut:left_col_cut + width] = mirrored[:, :width] | out[:, left_col_cut:left_col_cut + width]
    return out


def _build_slice_list(atlas: BrainGlobeAtlas) -> list[tuple[str, float, dict]]:
    """Return [(key, axis_level_ccf_mm, sdata), ...] in render order."""
    ac_ap_mm = atlas_left_striatum_centroid_ap(atlas)
    ac_dv_mm = atlas_left_striatum_centroid_dv(atlas)
    dv_sign = calibrate_dv_sign(atlas, 0.0, ac_dv_mm)

    out: list[tuple[str, float, dict]] = []
    for y_ccf in sorted(CORONAL_Y_LEVELS_CCF_MM, reverse=True):
        atlas_ap_mm = float(ac_ap_mm - y_ccf)
        sdata = extract_slice(atlas, atlas_ap_mm)
        sdata["ccf_level_mm"] = float(y_ccf)
        sdata["row_label"] = f"y = {y_ccf:+.0f} mm"
        out.append((f"coronal_y{y_ccf:+.0f}", float(y_ccf), sdata))
    for z_ccf in sorted(AXIAL_Z_LEVELS_CCF_MM, reverse=True):
        atlas_dv_mm = float(ac_dv_mm + dv_sign * z_ccf)
        sdata = extract_axial_slice(atlas, atlas_dv_mm)
        sdata["ccf_level_mm"] = float(z_ccf)
        sdata["row_label"] = f"z = {z_ccf:+.0f} mm"
        out.append((f"axial_z{z_ccf:+.0f}", float(z_ccf), sdata))
    return out


def build_fields(
    atlas: BrainGlobeAtlas,
    bins: pd.DataFrame,
    panel: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[
    dict[str, dict],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
]:
    limits = region_source_limits(bins)
    slice_list = _build_slice_list(atlas)
    slices: dict[str, dict] = {key: sdata for key, _level, sdata in slice_list}
    fields: dict[tuple[str, str], np.ndarray] = {}
    intensity_fields: dict[tuple[str, str], np.ndarray] = {}
    qc_rows = []
    group_target, group_display_target = display_targets_for_columns(target)

    for key, level_mm, sdata in slice_list:
        axis = sdata.get("axis", "coronal")
        if axis == "coronal":
            slab = bins[np.abs(bins["y_ccf_mm"] - level_mm) <= SLAB_HALF_MM].copy()
        else:
            slab = bins[np.abs(bins["z_ccf_mm"] - level_mm) <= SLAB_HALF_MM].copy()
        direct_masks = build_direct_region_masks(sdata, slab, limits)
        sdata["direct_masks"] = direct_masks
        for group_name, members in DISPLAY_COLUMNS.items():
            field = np.zeros(sdata["shape"], dtype=float)
            intensity_field = np.zeros(sdata["shape"], dtype=float)
            group_scores = group_value(slab, members, "score")
            for region in DIRECT_REGIONS:
                region_df = slab[slab["region"].astype(str).eq(region)]
                if region_df.empty:
                    continue
                mask = direct_masks[region]
                if not mask.any():
                    continue
                idx = region_df.index.to_numpy()
                pos = slab.index.get_indexer(idx)
                row, col = map_bins_to_mask(
                    region_df,
                    mask,
                    limits[region],
                    flip_ml=FLIP_WITHIN_REGION_ML,
                    axis=axis,
                )
                raw_mean = float(group_target.loc[group_name, region])
                display_mean = float(group_display_target.loc[group_name, region])
                region_cells = region_df["n_cells"].to_numpy(dtype=float)
                cells_sum = float(np.sum(region_cells))
                raw_target_mean = (
                    float(np.sum(group_scores[pos] * region_cells) / cells_sum)
                    if cells_sum > 0
                    else 0.0
                )
                if TEXTURE_RESIDUAL_DISPLAY and raw_target_mean > 1e-12:
                    texture_ratio = group_scores[pos] / raw_target_mean
                    region_scores = display_mean + TEXTURE_RESIDUAL_GAIN * (texture_ratio - 1.0)
                elif abs(raw_mean) > 1e-12:
                    region_scores = display_mean * group_scores[pos] / raw_mean
                else:
                    region_scores = np.full(len(region_df), display_mean, dtype=float)
                target_mean = (
                    float(np.sum(region_scores * region_cells) / cells_sum)
                    if cells_sum > 0
                    else display_mean
                )
                field += interpolated_score_field(
                    sdata["shape"], row, col, region_scores, mask, target_mean,
                )
                intensity_field += interpolated_score_field(
                    sdata["shape"], row, col, group_scores[pos], mask, raw_target_mean,
                )
            left_signal_mask = sdata["masks"].get("signal_left", sdata["masks"]["signal"])
            field[~left_signal_mask] = 0.0
            intensity_field[~left_signal_mask] = 0.0
            if MIRROR_LEFT_TO_RIGHT_HEMI:
                left_col_cut = int(sdata.get("left_col_cut", sdata["shape"][1] // 2))
                field = _mirror_field_to_right(field, left_col_cut)
                intensity_field = _mirror_field_to_right(intensity_field, left_col_cut)
            fields[(key, group_name)] = field
            intensity_fields[(key, group_name)] = intensity_field

        if MIRROR_LEFT_TO_RIGHT_HEMI:
            left_col_cut = int(sdata.get("left_col_cut", sdata["shape"][1] // 2))
            mirrored_direct = {}
            for region, region_mask in direct_masks.items():
                mirrored_direct[region] = _mirror_mask_to_right(region_mask, left_col_cut)
            sdata["direct_masks_display"] = mirrored_direct
            sdata["masks"]["signal_display"] = _mirror_mask_to_right(
                sdata["masks"]["signal"], left_col_cut
            )
        else:
            sdata["direct_masks_display"] = direct_masks
            sdata["masks"]["signal_display"] = sdata["masks"]["signal"]

        qc_rows.append({
            "slice_key": key,
            "axis": axis,
            "ccf_level_mm": float(level_mm),
            "n_spatial_bins_in_slab": int(len(slab)),
            "n_regions_in_slab": int(slab["region"].nunique()) if len(slab) else 0,
            "n_direct_region_masks": int(sum(m.any() for m in direct_masks.values())),
            "direct_region_mask_names": ";".join(
                region for region, mask in direct_masks.items() if mask.any()
            ),
        })
    return slices, fields, intensity_fields, pd.DataFrame(qc_rows)


def summarize_rendered_signal(
    slices: dict[str, dict],
    fields: dict[tuple[str, str], np.ndarray],
    target: pd.DataFrame,
) -> pd.DataFrame:
    group_target, group_display_target = display_targets_for_columns(target)
    raw_rank = group_target.rank(axis=1, ascending=False, method="min")
    display_rank = group_display_target.rank(axis=1, ascending=False, method="min")
    rows = []
    for group_name in DISPLAY_COLUMNS:
        for region in DIRECT_REGIONS:
            pix_total = 0
            mean_total = 0.0
            max_abs = 0.0
            keys_seen = []
            for slice_key, sdata in slices.items():
                mask = sdata.get("direct_masks", {}).get(region)
                if mask is None or not mask.any():
                    continue
                values = fields[(slice_key, group_name)][mask]
                n_pix = int(mask.sum())
                pix_total += n_pix
                mean_total += float(np.mean(values)) * n_pix
                max_abs = max(max_abs, float(np.max(np.abs(values))))
                keys_seen.append(str(slice_key))
            rendered_area_mean = mean_total / pix_total if pix_total else np.nan
            rows.append({
                "group_name": group_name,
                "region": region,
                "canonical_raw_target": float(group_target.loc[group_name, region]),
                "canonical_raw_rank": int(raw_rank.loc[group_name, region]),
                "canonical_display_target": float(group_display_target.loc[group_name, region]),
                "canonical_display_rank": int(display_rank.loc[group_name, region]),
                "rendered_area_mean": rendered_area_mean,
                "rendered_max_abs": max_abs,
                "rendered_pixels": pix_total,
                "slices_visible": ";".join(keys_seen),
            })
    return pd.DataFrame(rows)


def render(
    slices: dict[str, dict],
    fields: dict[tuple[str, str], np.ndarray],
    intensity_fields: dict[tuple[str, str], np.ndarray],
    target: pd.DataFrame,
) -> None:
    slice_keys = list(slices.keys())
    n_rows = len(slice_keys)
    n_cols = len(DISPLAY_COLUMNS)
    coronal_rows = [k for k in slice_keys if slices[k].get("axis") == "coronal"]
    axial_rows = [k for k in slice_keys if slices[k].get("axis") == "axial"]
    n_coronal = len(coronal_rows)
    n_axial = len(axial_rows)
    _horizontal = False
    _cor_span: tuple | None = None
    _ax_span: tuple | None = None

    if STYLE_MODE == "panel_b_replica":
        equalize_slice_canvases(slices, fields, intensity_fields, list(DISPLAY_COLUMNS))
        any_sdata = next(iter(slices.values()))
        canvas_h, canvas_w = any_sdata["shape"]
        coronal_slices = [s for s in slices.values() if s.get("axis") == "coronal"]
        all_sig_extents = [s.get("signal_extent_mm", (0.0, 0.0)) for s in slices.values()]
        coronal_sig_extents = [s.get("signal_extent_mm", (0.0, 0.0)) for s in coronal_slices]
        max_striatum_w = max((w for (w, _) in all_sig_extents), default=canvas_w * float(any_sdata.get("res", 0.5)))
        if coronal_sig_extents:
            max_striatum_h = max(h for (_, h) in coronal_sig_extents)
        else:
            max_striatum_h = max((h for (_, h) in all_sig_extents), default=canvas_h * float(any_sdata.get("res", 0.5)))
        VIS_PAD_MM_LR = 22.0
        STRIATUM_FILL_FRACTION = 0.50
        coronal_brain_h = [
            float(s.get("brain_extent_mm", (0.0, 999.0))[1])
            for s in coronal_slices
        ]
        if coronal_brain_h:
            brain_h_cap = min(coronal_brain_h)
            brain_h_cap = max(brain_h_cap - 1.0, 1.0)
        else:
            brain_h_cap = 999.0
        win_w_mm = float(max_striatum_w) + 2.0 * VIS_PAD_MM_LR
        desired_win_h = float(max_striatum_h) / STRIATUM_FILL_FRACTION
        win_h_mm = min(desired_win_h, brain_h_cap)
        unified_visible_window_mm = (win_w_mm, win_h_mm)
        panel_w_in = 4.5
        panel_h_in = panel_w_in * (win_h_mm / max(win_w_mm, 1e-6))
        _horizontal = HORIZONTAL_BLOCKS and n_coronal > 0 and n_axial > 0
        _cor_span = _ax_span = None
        if _horizontal:
            n_block_rows = max(n_coronal, n_axial)
            _bottom_band_in = 2.2
            fig_h = panel_h_in * n_block_rows + 3.0 + _bottom_band_in
            block_w_in = panel_w_in * n_cols
            _label_reach_in = max(
                abs(ROW_LABEL_AXES_X), abs(STRUCT_LABEL_AXES_X)
            ) * panel_w_in
            gutter_in = max(_label_reach_in + 0.9, EXTRA_LEFT_MARGIN_IN, 1.8)
            right_margin_in = 0.5
            _fig_w = gutter_in + block_w_in + gutter_in + block_w_in + right_margin_in
            top_frac = 1.0 - 3.0 / fig_h
            bottom_frac = _bottom_band_in / fig_h
            panel_frac_h = (top_frac - bottom_frac) / n_block_rows
            cor_left = gutter_in / _fig_w
            cor_right = (gutter_in + block_w_in) / _fig_w
            ax_left = (gutter_in + block_w_in + gutter_in) / _fig_w
            ax_right = (gutter_in + block_w_in + gutter_in + block_w_in) / _fig_w
            fig = plt.figure(figsize=(_fig_w, fig_h), dpi=RENDER_DPI)
            fig.patch.set_facecolor(BG_COLOR)
            gs_cor = fig.add_gridspec(
                n_coronal, n_cols, left=cor_left, right=cor_right,
                top=top_frac, bottom=top_frac - n_coronal * panel_frac_h,
                wspace=0.10, hspace=0.22,
            )
            gs_ax = fig.add_gridspec(
                n_axial, n_cols, left=ax_left, right=ax_right,
                top=top_frac, bottom=top_frac - n_axial * panel_frac_h,
                wspace=0.10, hspace=0.22,
            )
            _cor_span = (cor_left, cor_right)
            _ax_span = (ax_left, ax_right)
            axes = np.empty((n_rows, n_cols), dtype=object)
            for r in range(n_coronal):
                for c in range(n_cols):
                    axes[r, c] = fig.add_subplot(gs_cor[r, c])
            for r in range(n_axial):
                for c in range(n_cols):
                    axes[n_coronal + r, c] = fig.add_subplot(gs_ax[r, c])
        else:
            fig_h = panel_h_in * n_rows + 3.0 + 0.6 + 1.4 + 0.30 * n_rows
            _fig_w_base = panel_w_in * n_cols
            _left_abs = 0.080 * _fig_w_base + EXTRA_LEFT_MARGIN_IN
            _band_abs = 0.910 * _fig_w_base
            _fig_w = _fig_w_base + EXTRA_LEFT_MARGIN_IN
            _gs_left = _left_abs / _fig_w
            _gs_right = (_left_abs + _band_abs) / _fig_w
            fig = plt.figure(
                figsize=(_fig_w, fig_h), dpi=RENDER_DPI,
            )
            fig.patch.set_facecolor(BG_COLOR)
            n_coronal_safe = max(n_coronal, 1)
            n_axial_safe = max(n_axial, 1)
            outer = fig.add_gridspec(
                2, 1,
                height_ratios=[n_coronal_safe, n_axial_safe],
                left=_gs_left, right=_gs_right, top=0.910, bottom=0.085,
                hspace=0.20,
            )
            if n_coronal > 0 and n_axial > 0:
                gs_cor = outer[0].subgridspec(
                    n_coronal, n_cols, wspace=0.10, hspace=0.22,
                )
                gs_ax = outer[1].subgridspec(
                    n_axial, n_cols, wspace=0.10, hspace=0.22,
                )
            elif n_coronal > 0:
                gs_cor = outer[0].subgridspec(n_coronal, n_cols, wspace=0.10, hspace=0.22)
                gs_ax = None
            else:
                gs_cor = None
                gs_ax = outer[1].subgridspec(n_axial, n_cols, wspace=0.10, hspace=0.22)
            axes = np.empty((n_rows, n_cols), dtype=object)
            for r in range(n_coronal):
                for c in range(n_cols):
                    axes[r, c] = fig.add_subplot(gs_cor[r, c])
            for r in range(n_axial):
                for c in range(n_cols):
                    axes[n_coronal + r, c] = fig.add_subplot(gs_ax[r, c])
    else:
        fig = plt.figure(figsize=(2.55 * n_cols, 2.65 * n_rows + 0.6), dpi=RENDER_DPI)
        fig.patch.set_facecolor(BG_COLOR)
        gs = fig.add_gridspec(
            n_rows, n_cols,
            left=0.075, right=0.985, top=0.965, bottom=0.06,
            wspace=0.03,
            hspace=0.06,
            height_ratios=[1.0] * n_rows,
        )
        axes = np.empty((n_rows, n_cols), dtype=object)
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r, c] = fig.add_subplot(gs[r, c])

    group_target, group_display_target = display_targets_for_columns(target)
    global_color_vmin = 0.0
    column_alpha_vmin: dict[str, float] = {}
    if STYLE_MODE == "panel_b_replica":
        all_pos_vals = []
        for group_name in DISPLAY_COLUMNS:
            group_fields = [fields[(k, group_name)] for k in slice_keys]
            pos = [np.maximum(v, 0.0) for v in group_fields]
            pos = [v[v > 0] for v in pos if np.any(v > 0)]
            if pos:
                pooled = np.concatenate(pos)
                all_pos_vals.append(pooled)
                column_alpha_vmin[group_name] = float(np.percentile(pooled, 7.5))
            else:
                column_alpha_vmin[group_name] = 0.0
        if all_pos_vals:
            global_color_vmin = float(np.percentile(np.concatenate(all_pos_vals), 7.5))
    if DISPLAY_SCALE_MODE == "shared_robust":
        scale_source = group_target if DISPLAY_TARGET_MODE == "panel_a_raw" else group_display_target
        values = scale_source.to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        values = values[values > 0] if DISPLAY_TARGET_MODE == "panel_a_raw" else np.abs(values)
        values = values[values > 0]
        global_scale = float(np.percentile(values, DISPLAY_ABS_PERCENTILE)) if len(values) else 1.0
    else:
        scale_values = []
        for group_name in DISPLAY_COLUMNS:
            group_fields = [fields[(k, group_name)] for k in slice_keys]
            if DISPLAY_TARGET_MODE == "panel_a_raw":
                nonzero = [v[v > 0] for v in group_fields if np.any(v > 0)]
            else:
                nonzero = [np.abs(v[np.abs(v) > 0]) for v in group_fields if np.any(np.abs(v) > 0)]
            if nonzero:
                scale_values.append(float(np.percentile(np.concatenate(nonzero), DISPLAY_ABS_PERCENTILE)))
        global_scale = max(max(scale_values) if scale_values else 1.0, 1e-12)
    global_scale = max(global_scale, 1e-12)
    column_scale: dict[str, float] = {}
    column_intensity_scale: dict[str, float] = {}
    for group_name in DISPLAY_COLUMNS:
        group_fields = [fields[(k, group_name)] for k in slice_keys]
        if DISPLAY_TARGET_MODE == "panel_a_raw":
            nonzero = [v[v > 0] for v in group_fields if np.any(v > 0)]
        else:
            nonzero = [np.abs(v[np.abs(v) > 0]) for v in group_fields if np.any(np.abs(v) > 0)]
        if DISPLAY_SCALE_MODE == "per_column" and nonzero:
            scale = float(np.percentile(np.concatenate(nonzero), DISPLAY_ABS_PERCENTILE))
        else:
            scale = global_scale
        column_scale[group_name] = max(scale, 1e-12)

        group_intensity = [intensity_fields[(k, group_name)] for k in slice_keys]
        positive = [v[v > 0] for v in group_intensity if np.any(v > 0)]
        if DISPLAY_SCALE_MODE == "per_column" and positive:
            intensity_scale = float(np.percentile(np.concatenate(positive), DISPLAY_ABS_PERCENTILE))
        elif positive:
            intensity_scale = float(np.percentile(np.concatenate(positive), DISPLAY_ABS_PERCENTILE))
        else:
            intensity_scale = 1.0
        column_intensity_scale[group_name] = max(intensity_scale, 1e-12)

    _labeled_structures: set = set()
    _label_axis: str | None = None
    for row_i, slice_key in enumerate(slice_keys):
        sdata = slices[slice_key]
        masks = sdata["masks"]
        extent = sdata["extent"]
        display_direct = sdata.get("direct_masks_display", sdata.get("direct_masks", {}))
        render_mask = np.zeros(sdata["shape"], dtype=bool)
        for region_mask in display_direct.values():
            render_mask |= region_mask
        signal_display = masks.get("signal_display", masks["signal"])
        for col_i, group_name in enumerate(DISPLAY_COLUMNS):
            ax = axes[row_i, col_i]
            ax.set_facecolor(BG_COLOR)
            field = fields[(slice_key, group_name)]
            intensity_field = intensity_fields[(slice_key, group_name)]
            scale = column_scale[group_name]
            intensity_scale = column_intensity_scale[group_name]
            if STYLE_MODE == "panel_b_replica" and not EXACT_FIELD_PASSTHROUGH:
                field_smooth = field.copy()
                for _rname, _rmask in display_direct.items():
                    if not isinstance(_rmask, np.ndarray) or not _rmask.any():
                        continue
                    field_in_region = np.where(_rmask, field, 0.0)
                    f_blur = gaussian_filter(field_in_region, sigma=1.4)
                    m_blur = gaussian_filter(_rmask.astype(float), sigma=1.4)
                    safe = m_blur > 0.05
                    f_blur_norm = np.zeros_like(f_blur)
                    f_blur_norm[safe] = f_blur[safe] / m_blur[safe]
                    field_smooth[_rmask] = f_blur_norm[_rmask]
                field = field_smooth
            if STYLE_MODE == "panel_b_replica" and DISPLAY_TARGET_MODE == "panel_a_raw":
                norm = Normalize(vmin=global_color_vmin, vmax=global_scale)
                plot_field = np.clip(field, global_color_vmin, global_scale)
                rgb = RAW_CMAP(norm(plot_field))[:, :, :3]
            elif STYLE_MODE == "panel_b_replica":
                scale_signed = 1.0 if EXACT_FIELD_PASSTHROUGH else max(scale, 1e-12)
                signed_unit = np.clip(field / scale_signed, -1, 1)
                plot_field = signed_unit
                norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                rgb = DIVERGING_CMAP(norm(plot_field))[:, :, :3]
            elif DISPLAY_TARGET_MODE == "panel_a_raw":
                plot_field = np.clip(field, 0.0, scale)
                norm = Normalize(vmin=0.0, vmax=scale)
                rgb = RAW_CMAP(norm(plot_field))[:, :, :3]
            else:
                signed_unit = np.clip(field / scale, -1, 1)
                intensity_unit = np.clip(intensity_field / intensity_scale, 0, 1) ** GAMMA
                color_gain = INTENSITY_COLOR_FLOOR + (1.0 - INTENSITY_COLOR_FLOOR) * intensity_unit
                plot_field = np.clip(signed_unit * color_gain, -1, 1)
                norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                rgb = DIVERGING_CMAP(norm(plot_field))[:, :, :3]
            composite = sdata["dapi"].copy()
            alpha = np.zeros_like(field, dtype=float)
            if STYLE_MODE == "panel_b_replica" and DISPLAY_TARGET_MODE == "panel_a_raw":
                col_vmin = column_alpha_vmin.get(group_name, 0.0)
                col_vmax = max(scale, col_vmin + 1e-9)
                norm_intensity = np.clip(
                    (field - col_vmin) / (col_vmax - col_vmin), 0.0, 1.0,
                )
                alpha[render_mask] = np.clip(
                    0.20 + 0.75 * norm_intensity[render_mask], 0.0, 0.95,
                )
            elif STYLE_MODE == "panel_b_replica":
                if DIVERGING_PER_COLUMN_ALPHA:
                    col_scale_alpha = max(scale, 1e-9)
                    norm_intensity = np.clip(np.abs(field) / col_scale_alpha, 0.0, 1.0)
                    alpha[render_mask] = np.clip(
                        0.20 + 0.75 * norm_intensity[render_mask], 0.0, 0.95,
                    )
                else:
                    alpha[render_mask] = 0.95
            elif DISPLAY_TARGET_MODE == "panel_a_raw":
                alpha[render_mask] = RAW_COLOR_ALPHA
            else:
                alpha[render_mask] = SIGNED_COLOR_ALPHA
            if SUPPORT_ALPHA_MASKS is not None:
                _sup = SUPPORT_ALPHA_MASKS.get(slice_key)
                if _sup is not None:
                    alpha[~_sup] = 0.0
            for c in range(3):
                composite[render_mask, c] = (
                    composite[render_mask, c] * (1.0 - alpha[render_mask])
                    + rgb[render_mask, c] * alpha[render_mask]
                )
            interp = "lanczos" if STYLE_MODE == "panel_b_replica" else "bilinear"
            ax.imshow(
                composite, extent=extent, aspect="equal",
                interpolation=interp,
                resample=True,
                interpolation_stage="rgba",
            )

            if STYLE_MODE == "panel_b_replica":
                for _rname, _rmask in display_direct.items():
                    if not isinstance(_rmask, np.ndarray) or not _rmask.any():
                        continue
                    _outline = gaussian_filter(_rmask.astype(float), sigma=1.6)
                    ax.contour(
                        _outline, extent=extent, levels=[0.5],
                        colors="#222222", linewidths=1.0, origin="upper",
                    )
                if SHOW_STRUCTURE_LABELS and col_i == 0:
                    _axis_now = sdata.get("axis", "coronal")
                    if _axis_now != _label_axis:
                        _labeled_structures = set()
                        _label_axis = _axis_now
                    _left_masks = sdata.get("direct_masks", {})
                    _callouts = []
                    for _rname in display_direct:
                        _name = STRUCTURE_LABEL_NAMES.get(_rname)
                        if not _name or _name in _labeled_structures:
                            continue
                        _lm = _left_masks.get(_rname)
                        if _lm is None or not _lm.any():
                            continue
                        _dist = distance_transform_edt(_lm)
                        _ri, _ci = np.unravel_index(int(np.argmax(_dist)), _lm.shape)
                        _H, _W = _lm.shape
                        _l, _r, _b, _t = extent
                        _sx = _l + (_ci + 0.5) / _W * (_r - _l)
                        _sy = _t - (_ri + 0.5) / _H * (_t - _b)
                        _callouts.append((_name, _sx, _sy))
                        _labeled_structures.add(_name)
                    if _callouts:
                        _callouts.sort(key=lambda c: c[2])
                        _n = len(_callouts)
                        _slots = (
                            np.linspace(0.78, 0.22, _n) if _n > 1 else np.array([0.5])
                        )
                        for (_name, _sx, _sy), _slot in zip(_callouts, _slots):
                            ax.annotate(
                                _name,
                                xy=(_sx, _sy), xycoords="data",
                                xytext=(STRUCT_LABEL_AXES_X, float(_slot)),
                                textcoords="axes fraction",
                                fontsize=STRUCT_LABEL_FONTSIZE, fontweight="bold",
                                color=TEXT_COLOR, ha="center", va="center",
                                annotation_clip=False, zorder=7,
                                arrowprops=dict(
                                    arrowstyle="-", lw=0.8, color="#333333",
                                    shrinkA=2, shrinkB=3,
                                    connectionstyle="arc3,rad=0.0",
                                ),
                            )
            else:
                outline = gaussian_filter(signal_display.astype(float), 0.6)
                ax.contour(outline, extent=extent, levels=[0.5], colors=OUTLINE_COLOR,
                           linewidths=1.0, origin="upper")
                for region_mask in display_direct.values():
                    if region_mask.any():
                        boundary = gaussian_filter(region_mask.astype(float), 0.45)
                        ax.contour(boundary, extent=extent, levels=[0.5],
                                   colors="#555555", linewidths=0.3,
                                   origin="upper", alpha=0.55)
                brain_outline = gaussian_filter(masks["brain"].astype(float), 0.7)
                ax.contour(brain_outline, extent=extent, levels=[0.5], colors="#444444",
                           linewidths=0.5, origin="upper")
                if masks["vent"].any():
                    vent_outline = gaussian_filter(masks["vent"].astype(float), 0.6)
                    ax.contour(vent_outline, extent=extent, levels=[0.5], colors=VENT_COLOR,
                               linewidths=0.45, origin="upper")

            if row_i == 0 or (_horizontal and row_i == n_coronal):
                if STYLE_MODE == "panel_b_replica":
                    ax.set_title(
                        DISPLAY_LABELS.get(group_name, group_name),
                        fontsize=28, fontweight="bold",
                        color=TEXT_COLOR, pad=16,
                        linespacing=1.15,
                    )
                else:
                    ax.set_title(DISPLAY_LABELS.get(group_name, group_name),
                                 fontsize=7.2, fontweight="bold",
                                 color=TEXT_COLOR, pad=8)
            if STYLE_MODE == "panel_b_replica":
                ax.set_xticks([])
                ax.set_yticks([])
                sc = sdata.get("signal_centroid_mm")
                bc = sdata.get("brain_centroid_mm")
                if sc is not None and bc is not None:
                    cx_mm = 0.6 * sc[0] + 0.4 * bc[0]
                    cy_mm = 0.6 * sc[1] + 0.4 * bc[1]
                elif sc is not None:
                    cx_mm, cy_mm = sc
                elif bc is not None:
                    cx_mm, cy_mm = bc
                else:
                    cx_mm = canvas_w * float(sdata.get("res", 0.5)) / 2.0
                    cy_mm = canvas_h * float(sdata.get("res", 0.5)) / 2.0
                if sdata.get("axis") == "axial":
                    cy_mm -= 0.25 * unified_visible_window_mm[1]
                half_w = unified_visible_window_mm[0] / 2.0
                half_h = unified_visible_window_mm[1] / 2.0
                x_lo = cx_mm - half_w
                x_hi = cx_mm + half_w
                y_lo = cy_mm - half_h
                y_hi = cy_mm + half_h
                brain_bbox = sdata.get("brain_bbox_mm")
                if brain_bbox is not None:
                    b_left, b_right, b_bottom, b_top = brain_bbox
                    if x_lo < b_left:
                        x_hi += (b_left - x_lo)
                        x_lo = b_left
                    if x_hi > b_right:
                        x_lo -= (x_hi - b_right)
                        x_hi = b_right
                    x_lo = max(x_lo, b_left)
                    x_hi = min(x_hi, b_right)
                    if y_lo < b_top:
                        y_hi += (b_top - y_lo)
                        y_lo = b_top
                    if y_hi > b_bottom:
                        y_lo -= (y_hi - b_bottom)
                        y_hi = b_bottom
                    y_lo = max(y_lo, b_top)
                    y_hi = min(y_hi, b_bottom)
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_hi, y_lo)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#222222")
                    spine.set_linewidth(1.8)
            else:
                ax.axis("off")

        row_label = sdata.get("row_label", slice_key)
        if STYLE_MODE == "panel_b_replica":
            axes[row_i, 0].text(
                ROW_LABEL_AXES_X, 0.5, row_label,
                transform=axes[row_i, 0].transAxes,
                fontsize=34, color=TEXT_COLOR,
                va="center", ha="right", rotation=90, fontweight="bold",
            )
        else:
            axes[row_i, 0].text(
                -0.10, 0.5, row_label,
                transform=axes[row_i, 0].transAxes,
                fontsize=10.5, color=LABEL_COLOR,
                va="center", ha="right", rotation=90, fontweight="bold",
            )

    sidelabel_fs = 34 if STYLE_MODE == "panel_b_replica" else 13
    sidelabel_color = TEXT_COLOR if STYLE_MODE == "panel_b_replica" else LABEL_COLOR
    sidelabel_x = SIDE_BLOCK_LABEL_FIG_X if STYLE_MODE == "panel_b_replica" else 0.018
    if _horizontal:
        block_label_y = 1.0 - 0.5 / fig_h
        if n_coronal > 0 and _cor_span is not None:
            fig.text(sum(_cor_span) / 2.0, block_label_y, "Coronal (y)",
                     fontsize=sidelabel_fs, color=sidelabel_color, fontweight="bold",
                     va="center", ha="center")
        if n_axial > 0 and _ax_span is not None:
            fig.text(sum(_ax_span) / 2.0, block_label_y, "Axial (z)",
                     fontsize=sidelabel_fs, color=sidelabel_color, fontweight="bold",
                     va="center", ha="center")
    else:
        if n_coronal > 0:
            coronal_top = axes[0, 0].get_position()
            coronal_bot = axes[n_coronal - 1, 0].get_position()
            coronal_mid_y = (coronal_top.y1 + coronal_bot.y0) / 2.0
            fig.text(sidelabel_x, coronal_mid_y, "Coronal (y)",
                     fontsize=sidelabel_fs, color=sidelabel_color, fontweight="bold",
                     va="center", ha="center", rotation=90)
        if n_axial > 0:
            axial_top = axes[n_coronal, 0].get_position()
            axial_bot = axes[n_rows - 1, 0].get_position()
            axial_mid_y = (axial_top.y1 + axial_bot.y0) / 2.0
            fig.text(sidelabel_x, axial_mid_y, "Axial (z)",
                     fontsize=sidelabel_fs, color=sidelabel_color, fontweight="bold",
                     va="center", ha="center", rotation=90)

    lr_fs = 36 if STYLE_MODE == "panel_b_replica" else 11
    lr_y_off = 0.045 if STYLE_MODE == "panel_b_replica" else 0.012
    lr_color = TEXT_COLOR if STYLE_MODE == "panel_b_replica" else LABEL_COLOR

    def _draw_lr(bl):
        fig.text(bl.x0 + bl.width * 0.30, bl.y0 - lr_y_off, "L",
                 fontsize=lr_fs, color=lr_color, fontweight="bold", va="top", ha="center")
        fig.text(bl.x0 + bl.width * 0.50, bl.y0 - lr_y_off, "|",
                 fontsize=lr_fs, color=lr_color, fontweight="bold", va="top", ha="center")
        fig.text(bl.x0 + bl.width * 0.70, bl.y0 - lr_y_off, "R",
                 fontsize=lr_fs, color=lr_color, fontweight="bold", va="top", ha="center")

    if _horizontal:
        _mid = n_cols // 2
        _draw_lr(axes[n_coronal - 1, _mid].get_position())
        _draw_lr(axes[n_rows - 1, _mid].get_position())
    else:
        _draw_lr(axes[-1, 0].get_position())

    if STYLE_MODE != "panel_b_replica":
        scalebar = AnchoredSizeBar(
            axes[-1, 0].transData,
            5.0,
            "5 mm",
            "lower left",
            pad=0.35,
            color=TEXT_COLOR,
            frameon=False,
            size_vertical=0.12,
            fontproperties=fm.FontProperties(size=12, weight="bold"),
        )
        axes[-1, 0].add_artist(scalebar)

    if STYLE_MODE == "panel_b_replica":
        cbar_ax = fig.add_axes(CBAR_RECT if CBAR_RECT is not None else [0.58, 0.030, 0.38, 0.020])
        if DISPLAY_TARGET_MODE == "panel_a_raw":
            cbar_cmap = RAW_CMAP
            cbar_norm = Normalize(vmin=0.0, vmax=1.0)
            ticks = [0.0, 1.0]
            ticklabels = ["Low", "High"]
            label = "Projection density"
        else:
            cbar_cmap = DIVERGING_CMAP
            cbar_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            ticks = [-1.0, 0.0, 1.0]
            ticklabels = ["Depleted (-1)", "0", "Enriched (+1)"]
            label = "Subtype projection relative to other subtypes (column z-score)"
    else:
        cbar_ax = fig.add_axes([0.70, 0.018, 0.25, 0.010])
        cbar_cmap = RAW_CMAP if DISPLAY_TARGET_MODE == "panel_a_raw" else DIVERGING_CMAP
        cbar_norm = (
            Normalize(vmin=0.0, vmax=1.0)
            if DISPLAY_TARGET_MODE == "panel_a_raw"
            else TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        )
        if DISPLAY_TARGET_MODE == "panel_a_raw":
            ticks = [0.0, 1.0]
            ticklabels = ["Low", "High"]
            label = "Projection density, per column"
        else:
            ticks = [-1.0, 0, 1.0]
            ticklabels = ["Low", "0", "High"]
            if DISPLAY_TARGET_MODE == "panel_a_row_z":
                label = "Panel A row Z, per column"
            elif DISPLAY_TARGET_MODE == "panel_a_excess_uniform":
                label = "Panel A raw excess, per column"
            elif DISPLAY_TARGET_MODE == "panel_a_column_z":
                label = "relative projection enrichment"
            elif DISPLAY_TARGET_MODE == "panel_a_family_z":
                label = "Projection probability x intensity"
            else:
                label = "Panel A Z, per column"
    sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=ticks)
    cb_tick_fs = 30 if STYLE_MODE == "panel_b_replica" else 10
    cb_label_fs = (
        CBAR_LABEL_FONTSIZE if CBAR_LABEL_FONTSIZE is not None
        else (30 if STYLE_MODE == "panel_b_replica" else 10)
    )
    cb_color = TEXT_COLOR if STYLE_MODE == "panel_b_replica" else LABEL_COLOR
    cb.ax.set_xticklabels(ticklabels, fontsize=cb_tick_fs,
                          fontweight="bold", color=cb_color)
    cb.set_label(label, fontsize=cb_label_fs,
                 color=cb_color, fontweight="bold")
    cb.outline.set_edgecolor("#999999")
    cb.outline.set_linewidth(0.5)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig11b_panelA_constrained_spatial_transfer.{ext}",
                    dpi=RENDER_DPI, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)


def write_validation(
    panel: pd.DataFrame,
    bins: pd.DataFrame,
    target: pd.DataFrame,
    display_target: pd.DataFrame,
    panel_rebuild_max_diff: float,
    qc: pd.DataFrame,
    signal_qc: pd.DataFrame,
    panel_a_hashes: dict[str, str],
    elapsed_s: float,
) -> None:
    mass_cols = [c for c in bins.columns if c.startswith("mass_")]
    score_cols = [c for c in bins.columns if c.startswith("score_")]
    density_cols = [c for c in bins.columns if c.startswith("density_")]
    display_cols = [c for c in bins.columns if c.startswith("display_")]
    keep_cols = [
        "spatial_bin_id", "bin_id", "region", "n_cells", "x_ccf_mm",
        "y_ccf_mm", "z_ccf_mm", "atlas_ap_mm", "x_bin", "y_bin",
        "z_bin", "region_voxel", "group_name", "source_texture",
    ]
    keep_cols = [c for c in keep_cols if c in bins.columns]
    bins[keep_cols + score_cols + mass_cols + density_cols + display_cols].to_csv(
        OUT_DIR / "fig11b_panelA_constrained_spatial_bins.csv",
        index=False,
    )

    rows = []
    for subtype in panel.index:
        mass_col = f"mass_{safe_col(subtype)}"
        score_col = f"score_{safe_col(subtype)}"
        got_mass = bins.groupby("region")[mass_col].sum().reindex(DIRECT_REGIONS).fillna(0.0)
        got_mean = {}
        for region, df in bins.groupby("region"):
            weights = df["n_cells"].to_numpy(dtype=float)
            scores = df[score_col].to_numpy(dtype=float)
            got_mean[str(region)] = float(np.sum(scores * weights) / np.sum(weights))
        got_mean = pd.Series(got_mean).reindex(DIRECT_REGIONS).fillna(0.0)
        expected = target.loc[subtype].reindex(DIRECT_REGIONS).fillna(0.0)
        for region in DIRECT_REGIONS:
            rows.append({
                "subtype": subtype,
                "region": region,
                "spatial_weighted_mean": float(got_mean.loc[region]),
                "spatial_mass_sum": float(got_mass.loc[region]),
                "panelA_target": float(expected.loc[region]),
                "mean_diff": float(got_mean.loc[region] - expected.loc[region]),
                "mass_diff": float(got_mass.loc[region] - expected.loc[region]),
                "diff": float(got_mean.loc[region] - expected.loc[region]),
            })
        if "NAC" in panel.columns:
            panel_nac_total = float(panel.loc[subtype, ["NAC", "NACc", "NACs"]].sum())
            spatial_nac_mass = float(got_mass.reindex(NAC_FINE_REGIONS).sum())
            rows.append({
                "subtype": subtype,
                "region": "NAC_total",
                "spatial_weighted_mean": np.nan,
                "spatial_mass_sum": spatial_nac_mass,
                "panelA_target": panel_nac_total,
                "mean_diff": np.nan,
                "mass_diff": spatial_nac_mass - panel_nac_total,
                "diff": spatial_nac_mass - panel_nac_total,
            })
    validation = pd.DataFrame(rows)
    validation.to_csv(OUT_DIR / "fig11b_panelA_constrained_validation.csv", index=False)
    target.to_csv(OUT_DIR / "fig11b_panelA_constrained_targets.tsv", sep="\t")
    display_target.to_csv(OUT_DIR / "fig11b_panelA_constrained_display_targets.tsv", sep="\t")
    group_target, group_display_target = display_targets_for_columns(target)
    concordance = panel_a_concordance_table(target)
    group_target.to_csv(OUT_DIR / "fig11b_panelA_constrained_group_targets.tsv", sep="\t")
    group_display_target.to_csv(
        OUT_DIR / "fig11b_panelA_constrained_group_display_targets.tsv",
        sep="\t",
    )
    concordance.to_csv(
        OUT_DIR / "fig11b_panelA_constrained_panelA_concordance.csv",
        index=False,
    )
    qc.to_csv(OUT_DIR / "fig11b_panelA_constrained_render_qc.csv", index=False)
    signal_qc.to_csv(
        OUT_DIR / "fig11b_panelA_constrained_render_signal_qc.csv",
        index=False,
    )
    (OUT_DIR / "fig11b_panelA_constrained_calb1_split.json").write_text(
        json.dumps(CALB1_SPLIT_PAYLOAD, indent=2) + "\n"
    )

    meta = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "method": "Canonical Figure 11 Panel A mean-constrained within-region spatial transfer",
        "inputs": {
            "panel_a_snapshot_commit": PANEL_A_SNAPSHOT_COMMIT,
            "panel_a_snapshot_root": PANEL_A_SNAPSHOT_ROOT,
            "panel_a": f"{PANEL_A_SNAPSHOT_ROOT}/projection_matrix_human.tsv",
            "panel_a_sha256": panel_a_hashes["panel_a_sha256"],
            "panel_a_bin_level": f"{PANEL_A_SNAPSHOT_ROOT}/projection_bin_level.tsv",
            "panel_a_bin_level_sha256": panel_a_hashes["panel_a_bin_level_sha256"],
            "within_region_texture_bin_level": f"{PANEL_A_SNAPSHOT_ROOT}/projection_bin_level.tsv",
            "panel_a_bin_metadata": str(PANEL_A_BIN_META.relative_to(PROJECT_ROOT)),
            "h4_spatial_metadata": str(H4_BIN_METADATA.relative_to(PROJECT_ROOT)),
            "ccf_cells": str(CCF_CELLS.relative_to(PROJECT_ROOT)),
            "calb1_unsupervised_split": (
                "computed from the loaded Panel A matrix and canonical H4 "
                "within-region spatial texture"
            ),
            "atlas": ATLAS_NAME,
        },
        "parameters": {
            "voxel_size_mm": VOXEL_SIZE_MM,
            "min_cells_per_bin": MIN_CELLS_PER_BIN,
            "random_seed": RANDOM_SEED,
            "ap_levels_mm": list(AP_LEVELS_MM),
            "slab_half_mm": SLAB_HALF_MM,
            "smooth_sigma_px": SMOOTH_SIGMA_PX,
            "gamma": GAMMA,
            "min_color_alpha": MIN_COLOR_ALPHA,
            "raw_color_alpha": RAW_COLOR_ALPHA,
            "signed_color_alpha": SIGNED_COLOR_ALPHA,
            "intensity_color_floor": INTENSITY_COLOR_FLOOR,
            "display_target_mode": DISPLAY_TARGET_MODE,
            "display_scale_mode": DISPLAY_SCALE_MODE,
            "display_abs_percentile": DISPLAY_ABS_PERCENTILE,
            "shape_ratio_clip": list(SHAPE_RATIO_CLIP),
            "flip_within_region_ml": FLIP_WITHIN_REGION_ML,
            "interp_neighbors": INTERP_NEIGHBORS,
            "panel_a_constraint": "N-cell weighted mean score within each region equals the Panel A regional score.",
            "mass_column": "Mass columns multiply score by regional n-cell share, so they also sum to the Panel A regional score.",
            "nac_allocation": "Panel A NAC is allocated to NACc and NACs by constrained spatial-bin cell counts.",
            "voxelization": "Existing canonical H4 CCF bins are plotted directly for striatal regions. VeP uses neutral 2 mm CCF bins because the H4 texture table has no VeP bins.",
            "within_region_texture": "Within-region texture comes from the canonical Figure 11 Panel A bin-level score table mapped onto H4 spatial bins by region and group_name, then normalized within each subregion. Canonical Figure 11 Panel A region scores remain the only source for regional amplitude.",
            "rendering": "AP-local region voxels define each fine-region territory and rendered shading. Raw Panel A projection density is conserved as the regional amount. Display color uses the exact canonical Figure 11 Panel A column-Z value for each member subtype, averaged only after subtype-level Panel A values are computed for grouped columns. Raw Panel A density modulates saturation with a color floor so low or depleted regions remain blue rather than atlas gray. Within-region texture is bounded so it cannot dominate between-region differences. The ML coordinate is flipped only when placing bins inside each already defined fine-region mask; this fixes the within-region lateral-medial display orientation without changing region boundaries or Panel A amplitudes.",
            "render_signal_qc": "For each display group and fine region, render_signal_qc reports the canonical target rank and the area-mean value in the rendered AP slices.",
            "panel_a_concordance": "fig11b_panelA_constrained_panelA_concordance.csv must have zero raw_diff and display_diff. This prevents re-Z-scoring grouped columns after subtype aggregation.",
            "display_columns": dict(DISPLAY_COLUMNS),
            "display_labels": dict(DISPLAY_LABELS),
        },
        "validation": {
            "panel_a_source_rebuild_max_abs_diff": panel_rebuild_max_diff,
            "constrained_spatial_mean_roundtrip_max_abs_diff": float(validation["mean_diff"].abs().max()),
            "constrained_spatial_mass_roundtrip_max_abs_diff": float(validation["mass_diff"].abs().max()),
            "panel_a_group_concordance_max_abs_raw_diff": float(concordance["raw_diff"].abs().max()),
            "panel_a_group_concordance_max_abs_display_diff": float(concordance["display_diff"].abs().max()),
        },
        "elapsed_seconds": elapsed_s,
    }
    (OUT_DIR / "fig11b_panelA_constrained_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


def main() -> None:
    t0 = time.time()
    print("Loading Panel A", flush=True)
    panel, bin_level, bin_meta, panel_a_hashes = load_panel_a()
    panel_rebuild_max_diff = verify_panel_a_rebuild(panel, bin_level, bin_meta)
    print(f"Panel A source rebuild max abs diff: {panel_rebuild_max_diff:.3e}", flush=True)
    DISPLAY_COLUMNS.clear()
    DISPLAY_COLUMNS.update(load_display_columns(panel, bin_level, bin_meta))
    refresh_display_labels()
    print("Display columns:", ", ".join(DISPLAY_COLUMNS.keys()), flush=True)

    score_cols = list(panel.index)

    print("Loading Allen atlas", flush=True)
    atlas = BrainGlobeAtlas(ATLAS_NAME, check_latest=False)
    ac_ap_mm = atlas_left_striatum_centroid_ap(atlas)
    print(f"Allen left striatum centroid AP: {ac_ap_mm:.3f} mm", flush=True)

    print("Building constrained spatial bins", flush=True)
    texture_bin_level = bin_level
    bins = load_spatial_bins(panel, texture_bin_level, bin_meta, ac_ap_mm)
    print(f"Spatial bins: {len(bins):,}", flush=True)
    print(bins.groupby("region")["n_cells"].agg(["count", "sum"]).to_string(), flush=True)

    bins, target, display_target = constrain_to_panel_a(panel, bins, score_cols)

    print("Rendering Panel A constrained enrichment fields", flush=True)
    slices, fields, intensity_fields, qc = build_fields(atlas, bins, panel, target)
    signal_qc = summarize_rendered_signal(slices, fields, target)
    print(qc.to_string(index=False), flush=True)
    render(slices, fields, intensity_fields, target)

    elapsed_s = time.time() - t0
    write_validation(
        panel, bins, target, display_target, panel_rebuild_max_diff,
        qc, signal_qc, panel_a_hashes, elapsed_s,
    )
    print(f"Wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
