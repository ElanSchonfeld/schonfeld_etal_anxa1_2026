#!/usr/bin/env python3
"""Build the finalized human DA projection figure."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


SCRIPT_DIR = Path(__file__).resolve().parent
FIG4_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = FIG4_DIR.parent.parent
DATA_DIR = FIG4_DIR / "frozen" / "extended_arm"
FIG11_DIR = DATA_DIR
OUT_DIR = FIG4_DIR / "output" / "extended_arm"
COMPOSITE_BROAD_TITLE = "Broad-region projection matrix"
COMPOSITE_SPATIAL_TITLE = "Tangram (uniform density prior) spatial projection atlas"
STANDALONE_BROAD_TITLE = "Canonical broad-region projection matrix"
DPI = 600

PANEL_A_DISPLAY = DATA_DIR / "fig11b_panelA_constrained_display_targets.tsv"
PANEL_A_RAW = DATA_DIR / "fig11b_panelA_constrained_targets.tsv"
RPCA_METHOD_RAW = DATA_DIR / "fig11b_method_tangram_uniform_density_targets.tsv"
RPCA_BINS = DATA_DIR / "arm2_matched_panelA_locked_bins.parquet"
RPCA_FIG = (
    FIG11_DIR
    / "spatial_method_benchmark"
    / "fig11b_method_comparison"
    / "fig11b_method_tangram.png"
)
RPCA_SELECTED_CLUSTER_FIG_STEM = "figure11_human_projection_final_extended"
RPCA_SUPPLEMENTARY_FIG_STEM = "figure11_human_projection_final_supplementary"
ANXA_FOCUS_PANEL_B = (
    FIG11_DIR
    / "anxa_focus"
    / "fig11_anxa_focus_inferno_p075_925_notitle.png"
)
SPATIAL_RENDERER_SCRIPT = SCRIPT_DIR / "figure11b_panelA_constrained_spatial_transfer.py"
KRAFT_ZONE_MATRIX = DATA_DIR / "projection_matrix_human.tsv"
KRAFT_BIN_METADATA = DATA_DIR / "bin_metadata.csv"
SOFT_TRANSFER_DIR = (
    FIG11_DIR
    / "spatial_method_benchmark"
    / "soft_voxel_agreement"
)
SOFT_TRANSFER_PAIRWISE = SOFT_TRANSFER_DIR / "soft_voxel_pairwise.tsv"
SOFT_TRANSFER_SUMMARY = SOFT_TRANSFER_DIR / "soft_voxel_summary.tsv"
SOFT_TRANSFER_MULTISCALE = SOFT_TRANSFER_DIR / "soft_voxel_multiscale.tsv"
FINE_SCAFFOLD_PAIRWISE = SOFT_TRANSFER_DIR / "fine_scaffold_pairwise.tsv"
FINE_SCAFFOLD_MULTISCALE = SOFT_TRANSFER_DIR / "fine_scaffold_multiscale.tsv"
FINE_SCAFFOLD_REGISTRATION = SOFT_TRANSFER_DIR / "fine_scaffold_region_registration.tsv"
METHOD_COMPARISON_DIR = FIG11_DIR / "spatial_method_benchmark" / "fig11b_method_comparison"

REGION_ORDER = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "VeP"]
BROAD_HEATMAP_VLIM: float | None = None
GROUPED_REGION_ORDER = [r for r in REGION_ORDER if r != "VeP"]
GROUPED_HEATMAP_VLIM = 1.0
GROUPED_GROUP_COLORS = {
    "Anxa1-associated": "#08306b",
    "Sox6 family": "#4292c6",
    "Calb1 family": "#DE8F05",
}
ANXA_ANCHORS = ("Sox6:Tafa1", "Sox6:Vcan")
ANXA_LIKE_CORR_THRESHOLD = 0.9
EXTENDED_TEXTURE_RATIO_CLIP = (0.25, 4.0)
KRAFT_ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
KRAFT_UNIFORM = 1.0 / len(KRAFT_ZONE_ORDER)
KRAFT_ZONE_TICK_LABELS = [
    "Z1\nSM/assoc",
    "Z2\nassoc",
    "Z3\nassoc",
    "Z4\nlimbic",
    "Z5\nsensorimotor",
    "Z6\nsensorimotor",
]
KRAFT_ZONE_COLORS = {
    "Z1": "#8C564B",
    "Z2": "#1F77B4",
    "Z3": "#6BAED6",
    "Z4": "#2CA25F",
    "Z5": "#E6550D",
    "Z6": "#A63603",
}

SOX6_ORDER = [
    "Sox6:Tafa1",
    "Sox6:Vcan",
    "Sox6:Kcnmb2",
    "Sox6:Arhgap28",
    "Sox6:March3",
    "Sox6:Tmem132d",
]
CALB1_CLUSTER_ORDER_HINT = [
    "Calb1:Chrm2",
    "Calb1:Kctd8",
    "Calb1:Sox6",
    "Calb1:Sulf1",
    "Calb1:Ccdc192",
    "Calb1:Lpar1",
    "Calb1:Pde11a",
    "Calb1:Gipr",
    "Calb1:Ptprt",
    "Calb1:Stac",
]

FAMILY_COLORS = {
    "Sox6": "#2F6FAD",
    "Calb1": "#CC7A22",
}
TRANSFER_METHODS = OrderedDict(
    [
        ("seurat_cca_k25", "CCA k=25"),
        ("seurat_cca_k100", "CCA k=100"),
        ("seurat_cca_k200", "CCA k=200"),
    ]
)
TRANSFER_METHOD_COLORS = {
    "seurat_cca_k25": "#9ECAE1",
    "seurat_cca_k100": "#4E79A7",
    "seurat_cca_k200": "#08519C",
}
ZONE_DOMAIN_LABELS = {
    "Z1": "sensorimotor/associative overlap",
    "Z2": "associative caudate",
    "Z3": "associative caudate",
    "Z4": "limbic accumbens",
    "Z5": "sensorimotor putamen",
    "Z6": "sensorimotor putamen",
}
KRAFT_ZONE_INTERPRETATION_SOURCE = (
    "Kraft and Lee et al. 2026 bioRxiv 10.64898/2026.03.04.709715: "
    "text states that Zones 2/3 correspond broadly to associative striatum "
    "in caudate, Zone 4 to high limbic input in accumbens, Zones 5/6 to "
    "sensorimotor putamen, and Zone 1 overlaps both sensorimotor and associative."
)


@dataclass
class FamilyClusterResult:
    family: str
    members: list[str]
    metrics: pd.DataFrame
    all_memberships: pd.DataFrame
    selected_k: int
    selected_clusters: dict[str, list[str]]
    linkage_matrix: np.ndarray


@dataclass
class AnxaSimilarityAudit:
    table: pd.DataFrame
    flagged_calb1: list[str]


def _ensure_helvetica_bold_for_matplotlib():
    """Extract Helvetica Regular + Bold from `Helvetica.ttc` into standalone `.ttf` files in a cache dir and register them with matplotlib's font_manager."""
    import os
    from pathlib import Path
    src_ttc = Path("/System/Library/Fonts/Helvetica.ttc")
    if not src_ttc.exists():
        return
    cache_dir = Path(os.path.expanduser("~/.cache/matplotlib_helvetica"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        0: cache_dir / "Helvetica.ttf",
        1: cache_dir / "Helvetica-Bold.ttf",
    }
    if any(not p.exists() for p in targets.values()):
        try:
            from fontTools.ttLib import TTCollection
        except ImportError:
            return
        ttc = TTCollection(str(src_ttc))
        for face_idx, out_path in targets.items():
            if face_idx < len(ttc.fonts) and not out_path.exists():
                ttc.fonts[face_idx].save(str(out_path))
    import matplotlib.font_manager as _fm
    for out_path in targets.values():
        if out_path.exists():
            _fm.fontManager.addfont(str(out_path))


def configure_matplotlib() -> None:
    _ensure_helvetica_bold_for_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_col(name: str) -> str:
    return (
        name.replace(":", "_")
        .replace("+", "pos")
        .replace("-", "neg")
        .replace(" ", "_")
        .replace("/", "_")
    )


def family_of(subtype: str) -> str:
    return subtype.split(":", 1)[0]


def short_name(subtype: str) -> str:
    fam, gene = subtype.split(":", 1)
    if subtype == "Sox6:Tafa1":
        return "Anxa1+ Tafa1"
    if subtype == "Sox6:Vcan":
        return "Anxa1+ Vcan"
    if subtype == "Sox6:Kcnmb2":
        return "Sox6+ Aldh1a1+ Kcnmb2"
    if fam == "Sox6":
        return f"Sox6+ Aldh1a1- {gene}"
    return f"Calb1+ {gene}"


def compact_name(subtype: str) -> str:
    return subtype.split(":", 1)[1]


def cluster_short_name(label: str, members: list[str]) -> str:
    genes = "/".join(m.split(":", 1)[1] for m in members)
    return f"{label} ({genes})"


def save_all(fig: plt.Figure, stem: Path, dpi: int = DPI) -> None:
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04, "dpi": dpi}
    fig.savefig(stem.with_suffix(".png"), **save_kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)


def save_composite(fig: plt.Figure, stem: Path, dpi: int = DPI) -> None:
    save_kwargs = {"dpi": dpi, "facecolor": "white", "edgecolor": "none"}
    fig.savefig(stem.with_suffix(".png"), **save_kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)


def cleanup_stale_outputs() -> None:
    stale_names = [
        "rpca_spatial_nmf_territory_loadings.tsv",
        "rpca_spatial_nmf_territory_bin_weights.parquet",
        "rpca_spatial_nmf_territory_region_summary.tsv",
        "fig11_final_panel_d_transfer_method_robustness_summary.tsv",
        "fig11_final_panel_e_rpca_cca_agreement_summary.tsv",
    ]
    stale_stems = [
        "fig11_final_panel_d_e_spatial_territories",
        "fig11_final_panel_e_transfer_method_robustness",
        "fig11_final_panel_d_e_anxa_similarity_audit",
    ]
    for name in stale_names:
        (OUT_DIR / name).unlink(missing_ok=True)
    for stem in stale_stems:
        for suffix in (".png", ".pdf", ".svg"):
            (OUT_DIR / f"{stem}{suffix}").unlink(missing_ok=True)


def crop_white_margin(rgb: np.ndarray, threshold: int = 248, pad: int = 18) -> np.ndarray:
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return rgb
    y, x = np.where(mask)
    y0 = max(int(y.min()) - pad, 0)
    y1 = min(int(y.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(x.min()) - pad, 0)
    x1 = min(int(x.max()) + pad + 1, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    display = pd.read_csv(PANEL_A_DISPLAY, sep="\t", index_col=0).reindex(columns=REGION_ORDER)
    accepted_raw = pd.read_csv(PANEL_A_RAW, sep="\t", index_col=0).reindex(columns=REGION_ORDER)
    rpca_raw = pd.read_csv(RPCA_METHOD_RAW, sep="\t", index_col=0).reindex(columns=REGION_ORDER)
    bins = pd.read_parquet(RPCA_BINS)
    kraft = pd.read_csv(KRAFT_ZONE_MATRIX, sep="\t", index_col=0)
    kraft = kraft.rename(columns={str(i): f"Z{i}" for i in range(1, 7)})
    kraft = kraft.reindex(columns=KRAFT_ZONE_ORDER)
    kraft_meta = pd.read_csv(KRAFT_BIN_METADATA)
    return display, accepted_raw, rpca_raw, bins, kraft, kraft_meta


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ordered_subtypes(display: pd.DataFrame) -> list[str]:
    calb_available = [s for s in CALB1_CLUSTER_ORDER_HINT if s in display.index]
    leftovers = [
        s
        for s in display.index
        if s.startswith("Calb1:") and s not in calb_available
    ]
    return [s for s in SOX6_ORDER if s in display.index] + calb_available + sorted(leftovers)


def spatial_display_matrix(
    bins: pd.DataFrame,
    subtypes: list[str],
    display_targets: pd.DataFrame,
    raw_targets: pd.DataFrame,
) -> np.ndarray:
    region = bins["region"].astype(str)
    rows = []
    for subtype in subtypes:
        score_col = f"score_{safe_col(subtype)}"
        if score_col not in bins.columns:
            raise RuntimeError(f"Missing RPCA score column: {score_col}")
        score = bins[score_col].to_numpy(dtype=float)
        region_raw = region.map(raw_targets.loc[subtype]).to_numpy(dtype=float)
        region_display = region.map(display_targets.loc[subtype]).to_numpy(dtype=float)
        ratio = np.ones_like(score)
        ok = np.abs(region_raw) > 1e-12
        ratio[ok] = score[ok] / region_raw[ok]
        rows.append(region_display * ratio)
    return np.vstack(rows)


def cluster_family(
    family: str,
    bins: pd.DataFrame,
    display_targets: pd.DataFrame,
    raw_targets: pd.DataFrame,
) -> FamilyClusterResult:
    members = sorted([idx for idx in raw_targets.index if str(idx).startswith(f"{family}:")])
    features = spatial_display_matrix(bins, members, display_targets, raw_targets)
    weights = np.sqrt(bins["n_cells"].to_numpy(dtype=float))[None, :]
    scaled = normalize(features * weights)
    distances = pdist(scaled, metric="cosine")
    linkage_matrix = linkage(distances, method="average")
    distance_matrix = squareform(distances)

    metrics_rows = []
    member_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    max_k = min(8, len(members))
    for k in range(2, max_k):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        n_clusters = len(set(labels))
        if n_clusters < 2 or n_clusters >= len(members):
            continue
        labels_by_k[k] = labels
        sizes = pd.Series(labels).value_counts().sort_index()
        silhouette = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
        metrics_rows.append(
            {
                "family": family,
                "method": "seurat_rpca",
                "feature_space": "panel_a_locked_spatial_display_map_row_l2_cosine",
                "k": int(k),
                "silhouette": silhouette,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "selected": False,
            }
        )
        for cluster_id in sorted(set(labels)):
            cluster_members = [
                members[i] for i in range(len(members)) if labels[i] == cluster_id
            ]
            for member in cluster_members:
                member_rows.append(
                    {
                        "family": family,
                        "method": "seurat_rpca",
                        "feature_space": "panel_a_locked_spatial_display_map_row_l2_cosine",
                        "k": int(k),
                        "cluster_id": int(cluster_id),
                        "member": member,
                        "cluster_members": "/".join(cluster_members),
                    }
                )

    if not metrics_rows:
        raise RuntimeError(f"{family} spatial clustering produced no candidate partitions")

    metrics = pd.DataFrame(metrics_rows)
    selectable = metrics[metrics["min_cluster_size"] > 1]
    if selectable.empty:
        selectable = metrics
    best = float(selectable["silhouette"].max())
    near_best = selectable[selectable["silhouette"] >= 0.95 * best]
    selected_k = int(near_best.sort_values("k").iloc[-1]["k"])
    metrics.loc[metrics["k"] == selected_k, "selected"] = True

    selected_labels = labels_by_k[selected_k]
    selected_clusters: dict[str, list[str]] = {}
    for i, cluster_id in enumerate(sorted(set(selected_labels)), start=1):
        selected_clusters[f"{family} cluster {i}"] = [
            members[j] for j in range(len(members)) if selected_labels[j] == cluster_id
        ]

    return FamilyClusterResult(
        family=family,
        members=members,
        metrics=metrics,
        all_memberships=pd.DataFrame(member_rows),
        selected_k=selected_k,
        selected_clusters=selected_clusters,
        linkage_matrix=linkage_matrix,
    )


def build_cluster_tables(
    clusters: list[FamilyClusterResult],
    display_targets: pd.DataFrame,
    raw_targets: pd.DataFrame,
    kraft: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_rows = []
    raw_region_rows = []
    kraft_rows = []
    missing_rows = []

    for result in clusters:
        for label, members in result.selected_clusters.items():
            available_panel = [m for m in members if m in display_targets.index]
            region_rows.append(
                pd.Series(
                    display_targets.loc[available_panel].mean(axis=0),
                    name=cluster_short_name(label, members),
                )
            )
            raw_region_rows.append(
                pd.Series(
                    raw_targets.loc[available_panel].mean(axis=0),
                    name=cluster_short_name(label, members),
                )
            )

            available_kraft = [m for m in members if m in kraft.index]
            missing_kraft = [m for m in members if m not in kraft.index]
            if available_kraft:
                kraft_rows.append(
                    pd.Series(
                        kraft.loc[available_kraft].mean(axis=0),
                        name=cluster_short_name(label, members),
                    )
                )
            else:
                kraft_rows.append(
                    pd.Series(
                        [np.nan] * len(KRAFT_ZONE_ORDER),
                        index=KRAFT_ZONE_ORDER,
                        name=cluster_short_name(label, members),
                    )
                )
            missing_rows.append(
                {
                    "cluster": cluster_short_name(label, members),
                    "n_members": len(members),
                    "n_kraft_available": len(available_kraft),
                    "missing_from_native_kraft": ";".join(missing_kraft),
                }
            )

    cluster_region_display = pd.DataFrame(region_rows)
    cluster_region_raw = pd.DataFrame(raw_region_rows)
    cluster_kraft = pd.DataFrame(kraft_rows)
    missing = pd.DataFrame(missing_rows)
    return cluster_region_display, cluster_region_raw, cluster_kraft, missing


def selected_spatial_display_columns(calb1: FamilyClusterResult, row_order: list[str]) -> OrderedDict[str, list[str]]:
    """Three-column extended-figure layout: 1."""
    columns: OrderedDict[str, list[str]] = OrderedDict()
    anxa = [s for s in ANXA_ANCHORS if s in row_order]
    if anxa:
        columns["Anxa1+\n(Tafa1/Vcan)"] = anxa
    all_sox6 = [s for s in row_order if s.startswith("Sox6:")]
    if all_sox6:
        columns["Sox6+\n(all 6 subtypes)"] = all_sox6
    all_calb1 = [s for s in row_order if s.startswith("Calb1:")]
    if all_calb1:
        columns["Calb1+\n(all subtypes)"] = all_calb1
    return columns


def ungrouped_spatial_display_columns(row_order: list[str]) -> OrderedDict[str, list[str]]:
    """One column per subtype (no Sox6/Calb1 clustering)."""
    columns: OrderedDict[str, list[str]] = OrderedDict()
    for subtype in row_order:
        columns[subtype] = [subtype]
    return columns


def ungrouped_display_labels(row_order: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for subtype in row_order:
        fam, gene = subtype.split(":", 1)
        if subtype == "Sox6:Tafa1":
            labels[subtype] = "Anxa1+\n(Tafa1)"
        elif subtype == "Sox6:Vcan":
            labels[subtype] = "Anxa1+\n(Vcan)"
        elif subtype == "Sox6:Kcnmb2":
            labels[subtype] = "Sox6+ Aldh1a1+\n(Kcnmb2)"
        elif fam == "Sox6":
            labels[subtype] = f"Sox6+ Aldh1a1-\n({gene})"
        else:
            labels[subtype] = f"Calb1+\n({gene})"
    return labels


def unit_mean_shape_with_clip(
    shape: np.ndarray,
    weights: np.ndarray,
    ratio_clip: tuple[float, float] | None,
) -> np.ndarray:
    mean_shape = float(np.sum(shape * weights) / np.sum(weights))
    if not np.isfinite(mean_shape) or mean_shape <= 0:
        return np.ones(len(shape), dtype=float)
    ratio = shape / mean_shape
    if ratio_clip is not None:
        ratio = np.clip(ratio, ratio_clip[0], ratio_clip[1])
    mean_ratio = float(np.sum(ratio * weights) / np.sum(weights))
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        return np.ones(len(shape), dtype=float)
    return ratio / mean_ratio


def rebuild_score_columns_from_shape(
    bins: pd.DataFrame,
    target: pd.DataFrame,
    ratio_clip: tuple[float, float] | None,
) -> pd.DataFrame:
    out = bins.copy()
    n_cells = out["n_cells"].to_numpy(dtype=float)
    region_arr = out["region"].astype(str).to_numpy()
    for subtype in target.index:
        shape_col = f"shape_{safe_col(subtype)}"
        score_col = f"score_{safe_col(subtype)}"
        if shape_col not in out.columns or score_col not in out.columns:
            continue
        shape = np.maximum(out[shape_col].to_numpy(dtype=float), 0.0)
        score = out[score_col].to_numpy(dtype=float).copy()
        for region in REGION_ORDER:
            idx = np.where(region_arr == region)[0]
            if len(idx) == 0:
                continue
            weights = n_cells[idx]
            ratio = unit_mean_shape_with_clip(shape[idx], weights, ratio_clip)
            score[idx] = float(target.loc[subtype, region]) * ratio
        out[score_col] = score
    return out


SUBREGION_GROUPS = {
    "Pu": ("PuR", "PuC", "PuPV"),
    "Caudate": ("CaH", "CaB", "CaT"),
    "NAC": ("NACc", "NACs"),
}


KAMATH_SUBTYPE_COUNTS = {
    "Sox6:Tafa1": 5068,
    "Sox6:Arhgap28": 3386,
    "Sox6:Vcan": 2438,
    "Sox6:Tmem132d": 2233,
    "Calb1:Ptprt": 1893,
    "Calb1:Sulf1": 1418,
    "Calb1:Sox6": 1123,
    "Calb1:Pde11a": 1101,
    "Calb1:Kctd8": 699,
    "Calb1:Gipr": 684,
    "Sox6:Kcnmb2": 674,
    "Calb1:Ccdc192": 424,
    "Calb1:Lpar1": 359,
    "Sox6:March3": 336,
    "Calb1:Stac": 144,
    "Calb1:Chrm2": 68,
}


def rebuild_score_columns_with_collapsed_regions(
    bins: pd.DataFrame,
    target: pd.DataFrame,
    scale_by_cell_count: bool = True,
) -> pd.DataFrame:
    """Recompute score_<subtype> columns using MERGED Pu / Caudate / NAC targets (weighted mean of sub-region targets weighted by bin counts)."""
    out = bins.copy()
    n_cells = out["n_cells"].to_numpy(dtype=float)
    region_arr = out["region"].astype(str).to_numpy()
    if scale_by_cell_count:
        counts = np.array(
            [KAMATH_SUBTYPE_COUNTS.get(s, 0.0) for s in target.index],
            dtype=float,
        )
        mean_count = float(np.mean(counts[counts > 0])) if (counts > 0).any() else 1.0
        cell_weight = {
            s: float(np.sqrt(max(KAMATH_SUBTYPE_COUNTS.get(s, mean_count), 1.0) / mean_count))
            for s in target.index
        }
    else:
        cell_weight = {s: 1.0 for s in target.index}
    for subtype in target.index:
        shape_col = f"shape_{safe_col(subtype)}"
        score_col = f"score_{safe_col(subtype)}"
        if shape_col not in out.columns or score_col not in out.columns:
            continue
        shape = np.maximum(out[shape_col].to_numpy(dtype=float), 0.0)
        score = out[score_col].to_numpy(dtype=float).copy()
        cw = cell_weight.get(subtype, 1.0)

        handled = set()
        for merged_name, sub_regions in SUBREGION_GROUPS.items():
            avail = [sr for sr in sub_regions if sr in target.columns]
            if not avail:
                continue
            mask = np.isin(region_arr, avail)
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            handled.update(avail)
            sub_totals = {
                sr: float(n_cells[region_arr == sr].sum()) for sr in avail
            }
            total = sum(sub_totals.values())
            if total <= 0:
                continue
            merged_target = cw * sum(
                float(target.loc[subtype, sr]) * sub_totals[sr] / total
                for sr in avail
            )
            shape_block = shape[idx]
            weights = n_cells[idx]
            denom = float(weights.sum())
            if denom <= 0:
                ratio = np.ones(len(idx))
            else:
                mean_shape = float(np.sum(shape_block * weights) / denom)
                ratio = (
                    np.ones(len(idx))
                    if mean_shape <= 0
                    else shape_block / mean_shape
                )
            score[idx] = merged_target * ratio

        for region in REGION_ORDER:
            if region in handled:
                continue
            idx = np.where(region_arr == region)[0]
            if len(idx) == 0:
                continue
            weights = n_cells[idx]
            denom = float(weights.sum())
            if denom <= 0:
                ratio = np.ones(len(idx))
            else:
                mean_shape = float(np.sum(shape[idx] * weights) / denom)
                ratio = (
                    np.ones(len(idx))
                    if mean_shape <= 0
                    else shape[idx] / mean_shape
                )
            if region in target.columns:
                score[idx] = cw * float(target.loc[subtype, region]) * ratio
        out[score_col] = score
    return out


def write_texture_similarity_qc(
    bins: pd.DataFrame,
    display_columns: OrderedDict[str, list[str]],
    ratio_clip: tuple[float, float] | None,
    fig_stem: str = RPCA_SELECTED_CLUSTER_FIG_STEM,
) -> None:
    rows = []
    for region in ("PuR", "PuC", "PuPV"):
        df = bins[bins["region"].astype(str).eq(region)]
        if df.empty:
            continue
        weights = df["n_cells"].to_numpy(dtype=float)

        def group_shape(members: list[str]) -> np.ndarray | None:
            ratios = []
            for subtype in members:
                shape_col = f"shape_{safe_col(subtype)}"
                if shape_col not in df.columns:
                    continue
                shape = np.maximum(df[shape_col].to_numpy(dtype=float), 0.0)
                ratios.append(unit_mean_shape_with_clip(shape, weights, ratio_clip))
            if not ratios:
                return None
            return np.mean(ratios, axis=0)

        anxa = group_shape(list(ANXA_ANCHORS))
        if anxa is None:
            continue
        for group_name, members in display_columns.items():
            values = group_shape(members)
            if values is None:
                continue
            corr = float(np.corrcoef(values, anxa)[0, 1]) if np.std(values) and np.std(anxa) else np.nan
            rows.append({
                "region": region,
                "group_name": group_name,
                "member_subtypes": ";".join(members),
                "ratio_clip_low": np.nan if ratio_clip is None else ratio_clip[0],
                "ratio_clip_high": np.nan if ratio_clip is None else ratio_clip[1],
                "corr_to_merged_anxa_shape": corr,
                "shape_ratio_min": float(np.min(values)),
                "shape_ratio_max": float(np.max(values)),
                "shape_ratio_sd": float(np.std(values)),
            })
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"{fig_stem}_texture_similarity_qc.csv",
        index=False,
    )


def render_selected_cluster_spatial_atlas(
    bins: pd.DataFrame,
    target: pd.DataFrame,
    calb1: FamilyClusterResult,
    row_order: list[str],
    display_columns: "OrderedDict[str, list[str]] | None" = None,
    fig_stem: str | None = None,
    display_label_overrides: dict[str, str] | None = None,
    display_target_mode: str = "panel_a_column_z",
    cell_counts: dict[str, int] | None = None,
) -> Path:
    """Render the panel-B-style spatial atlas figure."""
    renderer = load_module("fig11b_selected_cluster_renderer", SPATIAL_RENDERER_SCRIPT)
    renderer.DISPLAY_COLUMNS.clear()
    if display_columns is None:
        display_columns = selected_spatial_display_columns(calb1, row_order)
    if fig_stem is None:
        fig_stem = RPCA_SELECTED_CLUSTER_FIG_STEM
    renderer.DISPLAY_COLUMNS.update(display_columns)
    renderer.refresh_display_labels()
    if display_label_overrides:
        renderer.DISPLAY_LABELS.update(display_label_overrides)

    atlas = renderer.BrainGlobeAtlas(renderer.ATLAS_NAME, check_latest=False)
    render_bins = bins.copy()
    if "atlas_ap_mm" not in render_bins.columns:
        render_bins = renderer.finalize_spatial_bins(
            render_bins,
            renderer.atlas_left_striatum_centroid_ap(atlas),
        )
    render_bins = rebuild_score_columns_from_shape(
        render_bins,
        target,
        EXTENDED_TEXTURE_RATIO_CLIP,
    )
    write_texture_similarity_qc(
        render_bins,
        renderer.DISPLAY_COLUMNS,
        EXTENDED_TEXTURE_RATIO_CLIP,
        fig_stem=fig_stem,
    )

    if cell_counts is not None:
        synth_columns: OrderedDict[str, list[str]] = OrderedDict()
        for group_name, members in renderer.DISPLAY_COLUMNS.items():
            valid = [m for m in members if m in target.index]
            if len(valid) <= 1:
                synth_columns[group_name] = members
                continue
            weights = np.array([float(cell_counts.get(m, 1.0)) for m in valid], dtype=float)
            wsum = float(weights.sum())
            if wsum <= 0:
                synth_columns[group_name] = members
                continue
            p = weights / wsum
            syn_id = f"FAM_{renderer.safe_col(group_name)}"
            target.loc[syn_id] = (p @ target.loc[valid].to_numpy())
            for prefix in ("score", "shape"):
                cols = [f"{prefix}_{renderer.safe_col(m)}" for m in valid]
                cols = [c for c in cols if c in render_bins.columns]
                if len(cols) == len(valid):
                    render_bins[f"{prefix}_{renderer.safe_col(syn_id)}"] = (
                        render_bins[cols].to_numpy() @ p
                    )
            synth_columns[group_name] = [syn_id]
        renderer.DISPLAY_COLUMNS.clear()
        renderer.DISPLAY_COLUMNS.update(synth_columns)

    old_out = renderer.OUT_DIR
    old_style_mode = getattr(renderer, "STYLE_MODE", "signed_diverging")
    old_raw_cmap = renderer.RAW_CMAP
    old_crop_pad = renderer.BG_CROP_PAD_VOX
    old_abs_percentile = renderer.DISPLAY_ABS_PERCENTILE
    renderer.STYLE_MODE = "panel_b_replica"
    renderer.BG_CROP_PAD_VOX = 160
    renderer.DISPLAY_ABS_PERCENTILE = (
        92.5
        if display_target_mode == "panel_a_raw"
        else 95.0
        if cell_counts is not None
        else 85.0
    )
    old_display_target = renderer.DISPLAY_TARGET_MODE
    renderer.DISPLAY_TARGET_MODE = display_target_mode
    old_texture_residual = getattr(renderer, "TEXTURE_RESIDUAL_DISPLAY", False)
    old_texture_residual_gain = getattr(renderer, "TEXTURE_RESIDUAL_GAIN", 0.55)
    renderer.TEXTURE_RESIDUAL_DISPLAY = (
        display_target_mode == "panel_a_column_z"
        and len(renderer.DISPLAY_COLUMNS) <= 3
    )
    renderer.TEXTURE_RESIDUAL_GAIN = 0.55
    old_div_alpha = getattr(renderer, "DIVERGING_PER_COLUMN_ALPHA", False)
    old_scale_mode = getattr(renderer, "DISPLAY_SCALE_MODE", "shared_robust")
    if cell_counts is not None:
        renderer.DISPLAY_SCALE_MODE = "per_column"

    tmp_dir = OUT_DIR / "_tmp_selected_cluster_spatial_render"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    renderer.OUT_DIR = tmp_dir
    try:
        slices, fields, intensity_fields, render_qc = renderer.build_fields(
            atlas,
            render_bins,
            target,
            target,
        )
        signal_qc = renderer.summarize_rendered_signal(slices, fields, target)
        renderer.render(slices, fields, intensity_fields, target)
    finally:
        renderer.OUT_DIR = old_out
        renderer.STYLE_MODE = old_style_mode
        renderer.RAW_CMAP = old_raw_cmap
        renderer.BG_CROP_PAD_VOX = old_crop_pad
        renderer.DISPLAY_ABS_PERCENTILE = old_abs_percentile
        renderer.DISPLAY_TARGET_MODE = old_display_target
        renderer.TEXTURE_RESIDUAL_DISPLAY = old_texture_residual
        renderer.TEXTURE_RESIDUAL_GAIN = old_texture_residual_gain
        renderer.DIVERGING_PER_COLUMN_ALPHA = old_div_alpha
        renderer.DISPLAY_SCALE_MODE = old_scale_mode

    outputs = {}
    for ext in ("png", "pdf", "svg"):
        src = tmp_dir / f"fig11b_panelA_constrained_spatial_transfer.{ext}"
        dst = OUT_DIR / f"{fig_stem}.{ext}"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        outputs[ext] = dst
    shutil.rmtree(tmp_dir)
    render_qc.to_csv(OUT_DIR / f"{fig_stem}_render_qc.csv", index=False)
    signal_qc.to_csv(OUT_DIR / f"{fig_stem}_signal_qc.csv", index=False)
    return outputs["png"]


def spatial_shape_ratio_matrix(
    bins: pd.DataFrame,
    subtypes: list[str],
    raw_targets: pd.DataFrame,
) -> np.ndarray:
    region = bins["region"].astype(str).to_numpy()
    rows = []
    for subtype in subtypes:
        score_col = f"score_{safe_col(subtype)}"
        if score_col not in bins.columns:
            raise RuntimeError(f"Missing RPCA score column: {score_col}")
        score = bins[score_col].to_numpy(dtype=float)
        region_raw = np.array([raw_targets.loc[subtype, r] for r in region], dtype=float)
        ratio = np.ones_like(score)
        ok = np.abs(region_raw) > 1e-12
        ratio[ok] = score[ok] / region_raw[ok]
        rows.append(ratio)
    return np.vstack(rows)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlations_to_anxa(features: np.ndarray, members: list[str]) -> dict[str, float]:
    anchors = [members.index(anchor) for anchor in ANXA_ANCHORS if anchor in members]
    if not anchors:
        raise RuntimeError("Anxa anchors are missing from the spatial audit")
    anchor_vec = np.nanmean(features[anchors, :], axis=0)
    return {
        member: safe_pearson(features[i, :], anchor_vec)
        for i, member in enumerate(members)
    }


def build_anxa_similarity_audit(
    bins: pd.DataFrame,
    row_order: list[str],
    display_targets: pd.DataFrame,
    raw_targets: pd.DataFrame,
) -> AnxaSimilarityAudit:
    members = [
        subtype
        for subtype in row_order
        if subtype in display_targets.index and f"score_{safe_col(subtype)}" in bins.columns
    ]
    spatial = spatial_display_matrix(bins, members, display_targets, raw_targets)
    texture = spatial_shape_ratio_matrix(bins, members, raw_targets)
    positive = np.maximum(spatial, 0.0)
    raw_region = raw_targets.loc[members, REGION_ORDER].to_numpy(dtype=float)
    display_region = display_targets.loc[members, REGION_ORDER].to_numpy(dtype=float)

    signed_corr = correlations_to_anxa(spatial, members)
    positive_corr = correlations_to_anxa(positive, members)
    texture_corr = correlations_to_anxa(texture, members)
    raw_corr = correlations_to_anxa(raw_region, members)
    display_corr = correlations_to_anxa(display_region, members)

    rows = []
    for subtype in members:
        rows.append(
            {
                "subtype": subtype,
                "label": short_name(subtype),
                "family": family_of(subtype),
                "signed_spatial_corr_to_anxa": signed_corr[subtype],
                "positive_spatial_corr_to_anxa": positive_corr[subtype],
                "within_region_texture_corr_to_anxa": texture_corr[subtype],
                "raw_region_corr_to_anxa": raw_corr[subtype],
                "display_region_corr_to_anxa": display_corr[subtype],
            }
        )
    table = pd.DataFrame(rows)
    table["anxa_like_flag"] = (
        table["family"].eq("Calb1")
        & (
            table["signed_spatial_corr_to_anxa"].ge(ANXA_LIKE_CORR_THRESHOLD)
            | table["positive_spatial_corr_to_anxa"].ge(ANXA_LIKE_CORR_THRESHOLD)
            | table["raw_region_corr_to_anxa"].ge(ANXA_LIKE_CORR_THRESHOLD)
        )
    )
    flagged = table.loc[table["anxa_like_flag"], "subtype"].tolist()
    return AnxaSimilarityAudit(table=table, flagged_calb1=flagged)


def add_heatmap_grid(ax: plt.Axes, n_rows: int, n_cols: int, color: str = "white") -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=color, linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_broad_heatmap(
    ax: plt.Axes,
    display: pd.DataFrame,
    row_order: list[str],
    title: str = "Canonical broad-region projection matrix",
    row_labeler=short_name,
    title_fontsize: float = 8,
    colorbar_label: str = "relative projection enrichment",
    xtick_fontsize: float = 6,
    ytick_fontsize: float = 5.3,
    cbar_label_fontsize: float = 5.8,
    cbar_tick_fontsize: float = 5,
) -> None:
    mat = display.loc[row_order, REGION_ORDER]
    if BROAD_HEATMAP_VLIM is None:
        vlim = max(1.5, float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))))
    else:
        vlim = float(BROAD_HEATMAP_VLIM)
    im = ax.imshow(
        mat.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
    )
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_ORDER, rotation=45, ha="right", fontsize=xtick_fontsize)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([row_labeler(s) for s in row_order], fontsize=ytick_fontsize)
    for tick, subtype in zip(ax.get_yticklabels(), row_order):
        tick.set_color(FAMILY_COLORS[family_of(subtype)])
    add_heatmap_grid(ax, len(row_order), len(REGION_ORDER))
    ax.set_title(title, fontsize=title_fontsize, pad=4, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(colorbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)


def build_group_matrix(display: "pd.DataFrame | None" = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Collapse the per-subtype projection into the three driver-reporting groups using CELL-WEIGHTED COLUMN-Z (locked-in method, 2026-06-12)."""
    raw = pd.read_csv(PANEL_A_RAW, sep="\t", index_col=0).reindex(columns=GROUPED_REGION_ORDER)
    frac = raw.div(raw.sum(axis=1), axis=0)
    subs = list(frac.index)
    w = np.array([KAMATH_SUBTYPE_COUNTS.get(s, 1.0) for s in subs], dtype=float)
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    anxa = [s for s in ANXA_ANCHORS if s in frac.index]
    if anxa:
        groups["Anxa1-associated"] = anxa
    sox6 = [s for s in frac.index if str(s).startswith("Sox6:")]
    if sox6:
        groups["Sox6 family"] = sox6
    calb1 = [s for s in frac.index if str(s).startswith("Calb1:")]
    if calb1:
        groups["Calb1 family"] = calb1
    rows = []
    row_color: dict[str, str] = {}
    for label, members in groups.items():
        idx = [subs.index(s) for s in members]
        ww = w[idx]
        rows.append(pd.Series(frac.iloc[idx].mul(ww, axis=0).sum(axis=0) / ww.sum(), name=label))
        row_color[label] = GROUPED_GROUP_COLORS[label]
    return pd.DataFrame(rows).fillna(0.0), row_color


def draw_sig_marks(ax, mat: pd.DataFrame, sig: dict, note: "str | None" = None,
                   star_fontsize: float = 9.0, note_xy=(1.0, 1.035),
                   note_ha: str = "right", note_va: str = "bottom") -> None:
    for yi, r in enumerate(mat.index):
        for xi, c in enumerate(mat.columns):
            if (r, c) not in sig:
                continue
            if sig[(r, c)]:
                val = float(mat.iloc[yi, xi])
                ax.text(xi + 0.30, yi - 0.30, "*", ha="center", va="center",
                        fontsize=star_fontsize, fontweight="bold",
                        color="white" if abs(val) >= 0.55 else "black", zorder=10)
            else:
                ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1.0, 1.0, fill=False,
                             facecolor="none", hatch="////", edgecolor="0.28",
                             linewidth=0.0, zorder=6))
    if note:
        ax.text(note_xy[0], note_xy[1], note, transform=ax.transAxes, ha=note_ha,
                va=note_va, fontsize=5.6, color="#333333", clip_on=False)


def build_group_significance(n_boot: int = 2000, seed: int = 20240624,
                             delta0: float = 0.01) -> dict:
    sb = PANEL_A_RAW.parent / "fig11b_panelA_constrained_spatial_bins.csv"
    df = pd.read_csv(sb)
    regions = list(GROUPED_REGION_ORDER)
    df = df[df["region"].isin(regions)].reset_index(drop=True)
    mass_cols = [c for c in df.columns if c.startswith("mass_")]
    ren = {c: c.replace("mass_", "").replace("_", ":", 1) for c in mass_cols}
    proj = df[mass_cols].rename(columns=ren)
    groups = OrderedDict()
    groups["Anxa1-associated"] = [s for s in ANXA_ANCHORS if s in proj.columns]
    groups["Sox6 family"] = [s for s in proj.columns if str(s).startswith("Sox6:")]
    groups["Calb1 family"] = [s for s in proj.columns if str(s).startswith("Calb1:")]
    code = pd.Categorical(df["region"], categories=regions).codes
    nt, ng = len(regions), len(groups)
    fam = np.zeros((len(df), ng))
    for gi, (g, mem) in enumerate(groups.items()):
        w = np.array([KAMATH_SUBTYPE_COUNTS.get(s, 1.0) for s in mem], float)
        w = w / w.sum() if w.sum() else w
        fam[:, gi] = (proj[mem].to_numpy() * w[None, :]).sum(1)

    def deltas(idx):
        c, fm = code[idx], fam[idx]
        M = np.vstack([np.bincount(c, weights=fm[:, f], minlength=nt) for f in range(ng)])
        tot = M.sum(1, keepdims=True)
        frac = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
        others = (frac.sum(0, keepdims=True) - frac) / (ng - 1)
        return frac - others

    n = len(df)
    obs = deltas(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.array([deltas(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    sig = {}
    for gi, g in enumerate(groups):
        for ti, t in enumerate(regions):
            ci_excl0 = (lo[gi, ti] > 0) or (hi[gi, ti] < 0)
            sig[(g, t)] = bool(ci_excl0 and abs(obs[gi, ti]) >= delta0)
    return sig


def _group_display_label(label: str) -> str:
    """Render the marker-positive superscript plus for the Sox6/Calb1 family rows (Sox6 family -> Sox6⁺ family)."""
    if label == "Sox6 family":
        return "Sox6⁺ family"
    if label == "Calb1 family":
        return "Calb1⁺ family"
    return label


def _territory_of(col: str) -> str:
    """Map a broad-region column to its striatal territory."""
    if col.startswith("Ca"):
        return "Caudate"
    if col.startswith("Pu"):
        return "Putamen"
    if col.startswith("NAC"):
        return "Nucleus accumbens"
    return col


def _territory_spans(cols: list[str], territory_fn=None) -> list[tuple[str, int, int]]:
    fn = territory_fn or _territory_of
    spans: list[tuple[str, int, int]] = []
    cur, start = None, 0
    for i, c in enumerate(cols):
        t = fn(c)
        if cur is None:
            cur, start = t, i
        elif t != cur:
            spans.append((cur, start, i - 1))
            cur, start = t, i
    if cur is not None:
        spans.append((cur, start, len(cols) - 1))
    return spans


def plot_grouped_broad_heatmap(
    ax: plt.Axes,
    mat: pd.DataFrame,
    row_color: dict[str, str],
    title: str = "Grouped broad-region projection matrix",
    title_fontsize: float = 8,
    colorbar_label: str = "Projection enrichment",
    xtick_fontsize: float = 6,
    ytick_fontsize: float = 6.5,
    cbar_label_fontsize: float = 5.8,
    cbar_tick_fontsize: float = 5,
    vlim: float | None = None,
    show_territories: bool = True,
    territory_fontsize: float = 7,
    territory_fn=None,
    ytick_labels: "list | None" = None,
    sig: "dict | None" = None,
    sig_note: "str | None" = None,
) -> None:
    cols = list(mat.columns)
    n_rows, n_cols = len(mat.index), len(cols)

    C = mat.sub(mat.mean(axis=0), axis=1)
    vmax = float(np.abs(C.to_numpy()).max()) or 1.0
    dispm = (C / vmax)
    im = ax.imshow(
        dispm.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(dispm.iat[i, j])
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=max(xtick_fontsize - 0.5, 5.5), zorder=8,
                    color="white" if abs(v) > 0.7 else "black")

    for i in range(1, n_rows):
        ax.plot([-0.5, n_cols - 0.5], [i - 0.5, i - 0.5], color="black",
                lw=0.9, zorder=3, solid_capstyle="butt")

    sb_w = 0.42
    sb_gap = 0.16
    sb_right = -0.5 - sb_gap
    sb_left = sb_right - sb_w
    for i, label in enumerate(mat.index):
        ax.add_patch(
            Rectangle(
                (sb_left, i - 0.5), sb_w, 1.0,
                facecolor=row_color[label], edgecolor="white", lw=1.0,
                zorder=4, clip_on=False,
            )
        )
    ax.set_xlim(sb_left - 0.05, n_cols - 0.5)

    ax.add_patch(
        Rectangle((-0.5, -0.5), n_cols, n_rows, fill=False,
                  edgecolor="black", lw=0.8, zorder=5, clip_on=False)
    )

    title_pad = 6.0
    if show_territories:
        spans = _territory_spans(cols, territory_fn)
        for _, _, _e in spans[:-1]:
            ax.plot([_e + 0.5, _e + 0.5], [-0.5, n_rows - 0.5], color="black",
                    lw=0.9, zorder=3, solid_capstyle="butt")
        for label, _s, _e in spans:
            ax.plot([_s - 0.4, _e + 0.4], [-0.64, -0.64], color="black", lw=1.0,
                    zorder=6, clip_on=False, solid_capstyle="butt")
            disp = "Nucleus\naccumbens" if label == "Nucleus accumbens" else label
            ax.text((_s + _e) / 2.0, -0.74, disp, ha="center", va="bottom",
                    fontsize=territory_fontsize, fontweight="bold", color="black",
                    clip_on=False, linespacing=0.95)
        title_pad = min(title_fontsize * 2.5, 22.0)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=xtick_fontsize)
    ax.set_yticks(np.arange(n_rows))
    _ylabels = (ytick_labels if ytick_labels is not None
                else [_group_display_label(s) for s in mat.index])
    ax.set_yticklabels(_ylabels, fontsize=ytick_fontsize, fontweight="bold")
    for tick, label in zip(ax.get_yticklabels(), mat.index):
        tick.set_color(row_color[label])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if sig is not None:
        if show_territories:
            _nxy, _nha, _nva = (0.5, -0.24), "center", "top"
        else:
            _nxy, _nha, _nva = (1.0, 1.035), "right", "bottom"
        draw_sig_marks(ax, dispm, sig, note=sig_note,
                       star_fontsize=max(ytick_fontsize + 1.0, 8.0),
                       note_xy=_nxy, note_ha=_nha, note_va=_nva)

    ax.set_title(title, fontsize=title_fontsize, pad=title_pad, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label(colorbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize, length=2)
    cb.outline.set_linewidth(0.5)


def plot_cluster_heatmap(
    ax: plt.Axes,
    cluster_region_display: pd.DataFrame,
    title: str,
) -> None:
    mat = cluster_region_display.reindex(columns=REGION_ORDER)
    vlim = max(1.5, float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))))
    im = ax.imshow(
        mat.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
    )
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_ORDER, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=5)
    add_heatmap_grid(ax, mat.shape[0], mat.shape[1])
    ax.set_title(title, fontsize=7.5, pad=4, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("cluster mean Panel A Z", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)


def plot_kraft_heatmap(
    ax: plt.Axes,
    kraft: pd.DataFrame,
    rows: list[str],
    title: str,
    row_labeler=short_name,
) -> None:
    mat = kraft.reindex(index=rows, columns=KRAFT_ZONE_ORDER)
    centered = mat - KRAFT_UNIFORM
    vlim = max(0.03, float(np.nanmax(np.abs(centered.to_numpy(dtype=float)))))
    im = ax.imshow(
        centered.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
    )
    ax.set_xticks(np.arange(len(KRAFT_ZONE_ORDER)))
    ax.set_xticklabels(KRAFT_ZONE_TICK_LABELS, fontsize=5.4)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row_labeler(r) for r in rows], fontsize=5.2)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if ":" in row:
            tick.set_color(FAMILY_COLORS[family_of(row)])
    add_heatmap_grid(ax, len(rows), len(KRAFT_ZONE_ORDER))
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("zone share minus 1/6", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)


def plot_kraft_cluster_heatmap(ax: plt.Axes, cluster_kraft: pd.DataFrame, title: str) -> None:
    centered = cluster_kraft.reindex(columns=KRAFT_ZONE_ORDER) - KRAFT_UNIFORM
    vlim = max(0.03, float(np.nanmax(np.abs(centered.to_numpy(dtype=float)))))
    im = ax.imshow(
        centered.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
    )
    ax.set_xticks(np.arange(len(KRAFT_ZONE_ORDER)))
    ax.set_xticklabels(KRAFT_ZONE_TICK_LABELS, fontsize=5.4)
    ax.set_yticks(np.arange(centered.shape[0]))
    ax.set_yticklabels(centered.index.tolist(), fontsize=5)
    add_heatmap_grid(ax, centered.shape[0], centered.shape[1])
    ax.set_title(title, fontsize=7.5, pad=4, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("cluster zone share minus 1/6", fontsize=5.3)
    cb.ax.tick_params(labelsize=5)


def plot_zone_domain_legend(ax: plt.Axes) -> None:
    ax.axis("off")
    text = "Native Kraft zones: " + " | ".join(
        f"{zone} {ZONE_DOMAIN_LABELS[zone]}" for zone in KRAFT_ZONE_ORDER
    )
    ax.text(0, 0.5, text, ha="left", va="center", fontsize=6)


def kraft_zone_shares(kraft: pd.DataFrame) -> pd.DataFrame:
    zone = kraft.reindex(columns=KRAFT_ZONE_ORDER).copy()
    return zone.div(zone.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def plot_kraft_zone_bars(
    ax: plt.Axes,
    kraft: pd.DataFrame,
    row_order: list[str],
    title: str = "D. Native Kraft Zone 1-6 composition",
) -> None:
    order = [s for s in row_order if s in kraft.index]
    zones = kraft_zone_shares(kraft).loc[order]
    y = np.arange(len(order))
    left = np.zeros(len(order), dtype=float)
    for zone in KRAFT_ZONE_ORDER:
        vals = zones[zone].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.74,
            color=KRAFT_ZONE_COLORS[zone],
            label=f"{zone}: {ZONE_DOMAIN_LABELS[zone]}",
            zorder=2,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(s) for s in order], fontsize=5.2)
    for tick, subtype in zip(ax.get_yticklabels(), order):
        tick.set_color(FAMILY_COLORS[family_of(subtype)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Projection share", fontsize=6)
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    ax.tick_params(labelsize=5)
    n_sox6 = sum(1 for s in order if family_of(s) == "Sox6")
    ax.axhline(n_sox6 - 0.5, color="#CCCCCC", lw=0.8, ls="--")
    ax.legend(
        fontsize=4.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=2,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        handletextpad=0.3,
    )


def plot_kraft_cluster_zone_bars(
    ax: plt.Axes,
    cluster_kraft: pd.DataFrame,
    title: str = "E. Kraft Zone 1-6 by RPCA cluster",
) -> None:
    zones = kraft_zone_shares(cluster_kraft)
    compact = zones.copy()
    compact.index = [
        idx.replace("Sox6 cluster ", "Sox6 ")
        .replace("Calb1 cluster ", "Calb1 ")
        .replace(" (", "\n(")
        for idx in compact.index
    ]
    y = np.arange(compact.shape[0])
    left = np.zeros(compact.shape[0], dtype=float)
    for zone in KRAFT_ZONE_ORDER:
        vals = compact[zone].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.72,
            color=KRAFT_ZONE_COLORS[zone],
            label=f"{zone}: {ZONE_DOMAIN_LABELS[zone]}",
            zorder=2,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(compact.index.tolist(), fontsize=4.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean projection share", fontsize=6)
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    ax.tick_params(labelsize=5)
    ax.axhline(sox6_cluster_count(cluster_kraft) - 0.5, color="#CCCCCC", lw=0.8, ls="--")
    ax.legend(
        fontsize=4.3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=2,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        handletextpad=0.3,
    )


def sox6_cluster_count(cluster_kraft: pd.DataFrame) -> int:
    return sum(1 for idx in cluster_kraft.index if str(idx).startswith("Sox6 cluster"))


def plot_family_clustering_panel(
    sox6: FamilyClusterResult,
    calb1: FamilyClusterResult,
    cluster_region_display: pd.DataFrame,
    out_stem: Path,
) -> None:
    fig = plt.figure(figsize=(10.6, 3.4), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.4],
        wspace=0.34,
    )
    ax_s_d = fig.add_subplot(gs[0])
    ax_c_d = fig.add_subplot(gs[1])
    dendrogram(
        sox6.linkage_matrix,
        labels=[m.split(":", 1)[1] for m in sox6.members],
        leaf_rotation=45,
        leaf_font_size=7,
        ax=ax_s_d,
    )
    ax_s_d.set_title(f"Sox6 spatial clustering, K={sox6.selected_k}", fontsize=8, fontweight="bold")
    ax_s_d.set_ylabel("cosine distance", fontsize=6)
    ax_s_d.tick_params(labelsize=6)

    dendrogram(
        calb1.linkage_matrix,
        labels=[m.split(":", 1)[1] for m in calb1.members],
        leaf_rotation=45,
        leaf_font_size=7,
        ax=ax_c_d,
    )
    ax_c_d.set_title(f"Calb1 spatial clustering, K={calb1.selected_k}", fontsize=8, fontweight="bold")
    ax_c_d.set_ylabel("cosine distance", fontsize=6)
    ax_c_d.tick_params(labelsize=6)

    fig.suptitle(
        "Unsupervised clustering from RPCA Panel-A-locked spatial maps",
        y=0.995,
        fontsize=10,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.22, wspace=0.30)
    save_all(fig, out_stem)
    plt.close(fig)


def plot_cluster_dendrogram(
    ax: plt.Axes,
    result: FamilyClusterResult,
    title: str,
    *,
    title_fontsize: float = 6.8,
    leaf_font_size: float = 6,
    ylabel_fontsize: float = 5.2,
    tick_fontsize: float = 5.2,
) -> None:
    dendrogram(
        result.linkage_matrix,
        labels=[m.split(":", 1)[1] for m in result.members],
        leaf_rotation=45,
        leaf_font_size=leaf_font_size,
        ax=ax,
    )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=3)
    ax.set_ylabel("cosine distance", fontsize=ylabel_fontsize)
    ax.tick_params(labelsize=tick_fontsize)


def plot_anxa_similarity_bars(
    ax: plt.Axes,
    audit: AnxaSimilarityAudit,
    row_order: list[str],
    metric: str,
    title: str,
) -> None:
    order = [row for row in row_order if row in set(audit.table["subtype"])]
    table = audit.table.set_index("subtype").loc[order].sort_values(metric, ascending=True)
    y = np.arange(len(table))
    colors = [FAMILY_COLORS[family] for family in table["family"]]
    ax.barh(y, table[metric].to_numpy(dtype=float), color=colors, height=0.72, zorder=2)
    ax.axvline(0, color="#666666", lw=0.7, zorder=1)
    ax.axvline(ANXA_LIKE_CORR_THRESHOLD, color="#AA0000", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-1, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"].tolist(), fontsize=5.0)
    for tick, family in zip(ax.get_yticklabels(), table["family"]):
        tick.set_color(FAMILY_COLORS[family])
    ax.set_xlabel("Pearson r to Anxa mean", fontsize=5.8)
    ax.set_title(title, fontsize=8, pad=4, fontweight="bold")
    ax.tick_params(labelsize=5.3)
    ax.grid(axis="x", color="#DDDDDD", lw=0.45, zorder=0)


def plot_anxa_similarity_panel(
    audit: AnxaSimilarityAudit,
    row_order: list[str],
    out_stem: Path,
) -> None:
    fig = plt.figure(figsize=(10.6, 5.4), constrained_layout=False)
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.34,
    )
    ax_d = fig.add_subplot(gs[0, 0])
    plot_anxa_similarity_bars(
        ax_d,
        audit,
        row_order,
        "signed_spatial_corr_to_anxa",
        "Anxa similarity in RPCA spatial atlas",
    )
    add_panel_label(ax_d, "d", x=-0.24, y=1.06)

    ax_e = fig.add_subplot(gs[0, 1])
    plot_anxa_similarity_bars(
        ax_e,
        audit,
        row_order,
        "within_region_texture_corr_to_anxa",
        "Anxa similarity in within-region texture",
    )
    add_panel_label(ax_e, "e", x=-0.24, y=1.06)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.20, wspace=0.34)
    save_all(fig, out_stem)
    plt.close(fig)


def transfer_method_summary_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise = pd.read_csv(FINE_SCAFFOLD_PAIRWISE, sep="\t")
    multiscale = pd.read_csv(FINE_SCAFFOLD_MULTISCALE, sep="\t")
    registration = pd.read_csv(FINE_SCAFFOLD_REGISTRATION, sep="\t")
    support_label = "seurat_cca_fine_translate"
    one = pairwise[pairwise["support_label"].eq(support_label)].copy()
    if one.empty:
        raise RuntimeError(f"Missing fine scaffold support label: {support_label}")
    one_registration = registration[registration["support_label"].eq(support_label)].copy()
    if one_registration.empty:
        raise RuntimeError(f"Missing fine scaffold registration rows: {support_label}")
    summary = pd.DataFrame(
        [
            {
                "support_label": support_label,
                "method_label": "CCA fixed translate",
                "n_subtype_region_pairs": int(len(one)),
                "median_voxel_cosine_vs_rpca": float(np.nanmedian(one["voxel_cosine_vs_rpca"])),
                "p10_voxel_cosine_vs_rpca": float(np.nanpercentile(one["voxel_cosine_vs_rpca"], 10)),
                "median_centroid_shift_mm_vs_rpca": float(
                    np.nanmedian(one["centroid_shift_mm_vs_rpca"])
                ),
                "p90_centroid_shift_mm_vs_rpca": float(
                    np.nanpercentile(one["centroid_shift_mm_vs_rpca"], 90)
                ),
                "median_geometry_shift_mm_vs_rpca": float(
                    np.nanmedian(one_registration["geometry_shift_mm_vs_rpca"])
                ),
                "p90_geometry_shift_mm_vs_rpca": float(
                    np.nanpercentile(one_registration["geometry_shift_mm_vs_rpca"], 90)
                ),
                "puc_geometry_shift_mm_vs_rpca": float(
                    one_registration.loc[
                        one_registration["region"].eq("PuC"),
                        "geometry_shift_mm_vs_rpca",
                    ].iloc[0]
                ),
                "puc_median_centroid_shift_mm_vs_rpca": float(
                    np.nanmedian(
                        one.loc[
                            one["region"].eq("PuC"),
                            "centroid_shift_mm_vs_rpca",
                        ]
                    )
                ),
                "puc_p90_centroid_shift_mm_vs_rpca": float(
                    np.nanpercentile(
                        one.loc[
                            one["region"].eq("PuC"),
                            "centroid_shift_mm_vs_rpca",
                        ],
                        90,
                    )
                ),
            }
        ]
    )
    summary.to_csv(
        OUT_DIR / "fig11_final_panel_e_rpca_cca_agreement_summary.tsv",
        sep="\t",
        index=False,
    )
    return one, multiscale, one_registration


def plot_rpca_cca_agreement(ax: plt.Axes) -> None:
    pairwise, _, registration = transfer_method_summary_table()
    map_cos = pairwise["voxel_cosine_vs_rpca"].to_numpy(dtype=float)
    centroid_shift = pairwise["centroid_shift_mm_vs_rpca"].to_numpy(dtype=float)
    n_maps = int(len(pairwise))
    region_order = [region for region in REGION_ORDER if region in set(registration["region"])]
    reg = registration.set_index("region").loc[region_order].reset_index()
    puc_shift = float(reg.loc[reg["region"].eq("PuC"), "geometry_shift_mm_vs_rpca"].iloc[0])
    puc_local = pairwise.loc[pairwise["region"].eq("PuC"), "centroid_shift_mm_vs_rpca"].to_numpy(dtype=float)
    ax.set_axis_off()
    ax.set_title("RPCA/CCA fixed-scaffold agreement", fontsize=8, fontweight="bold", pad=4)
    ax.text(
        0.5,
        0.94,
        f"CCA registered to fixed RPCA fine regions, Panel-A locked, n = {n_maps} maps",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color="#333333",
    )

    map_ax = ax.inset_axes([0.045, 0.16, 0.27, 0.64])
    map_ax.set_facecolor("#FAFAFA")
    map_ax.axvspan(0.98, 1.0, color="#DDECDC", alpha=0.9, zorder=0)
    map_ax.hist(
        map_cos[np.isfinite(map_cos)],
        bins=np.linspace(0.94, 1.0, 16),
        color="#2F6F9F",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.45,
    )
    map_median = float(np.nanmedian(map_cos))
    map_p10 = float(np.nanpercentile(map_cos, 10))
    map_ax.axvline(map_median, color="#222222", linewidth=1.0)
    map_ax.set_xlim(0.94, 1.0)
    map_ax.set_title("local maps", fontsize=6.6, fontweight="bold", pad=3)
    map_ax.set_xlabel("cosine", fontsize=5.3, labelpad=1.0)
    map_ax.set_yticks([])
    map_ax.tick_params(axis="x", labelsize=4.9, length=2)
    map_ax.text(
        0.05,
        0.93,
        f"median {map_median:.3f}\np10 {map_p10:.3f}",
        transform=map_ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3},
    )

    reg_ax = ax.inset_axes([0.37, 0.16, 0.26, 0.64])
    reg_ax.set_facecolor("#FAFAFA")
    y = np.arange(len(reg))
    colors = ["#C97931" if region == "PuC" else "#B9B9B9" for region in reg["region"]]
    reg_ax.barh(y, reg["geometry_shift_mm_vs_rpca"].to_numpy(dtype=float), color=colors, height=0.72)
    reg_ax.set_yticks(y)
    reg_ax.set_yticklabels(reg["region"], fontsize=4.9)
    reg_ax.invert_yaxis()
    reg_ax.set_xlim(0, max(4.1, float(reg["geometry_shift_mm_vs_rpca"].max()) * 1.08))
    reg_ax.set_xlabel("mm", fontsize=5.3, labelpad=1.0)
    reg_ax.set_title("region offset", fontsize=6.6, fontweight="bold", pad=3)
    reg_ax.tick_params(axis="x", labelsize=4.9, length=2)
    reg_ax.grid(axis="x", color="#DDDDDD", linewidth=0.35)
    reg_ax.text(
        0.95,
        0.92,
        f"PuC {puc_shift:.2f} mm",
        transform=reg_ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3},
    )

    res_ax = ax.inset_axes([0.705, 0.16, 0.255, 0.64])
    res_ax.set_facecolor("#FAFAFA")
    res_ax.axvspan(0.0, 0.35, color="#DDECDC", alpha=0.9, zorder=0)
    res_ax.hist(
        centroid_shift[np.isfinite(centroid_shift)],
        bins=np.linspace(0.0, 0.55, 16),
        color="#2D7D46",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.45,
    )
    res_median = float(np.nanmedian(centroid_shift))
    res_p90 = float(np.nanpercentile(centroid_shift, 90))
    puc_res = float(np.nanmedian(puc_local))
    res_ax.axvline(res_median, color="#222222", linewidth=1.0)
    res_ax.axvline(res_p90, color="#222222", linewidth=0.9, linestyle=":")
    res_ax.set_xlim(0.0, 0.55)
    res_ax.set_title("registered residual", fontsize=6.6, fontweight="bold", pad=3)
    res_ax.set_xlabel("centroid shift, mm", fontsize=5.3, labelpad=1.0)
    res_ax.set_yticks([])
    res_ax.tick_params(axis="x", labelsize=4.9, length=2)
    res_ax.text(
        0.05,
        0.93,
        f"median {res_median:.2f} mm\nPuC {puc_res:.2f} mm",
        transform=res_ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3},
    )
    for sub_ax in (map_ax, reg_ax, res_ax):
        sub_ax.spines["top"].set_visible(False)
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["left"].set_visible(False)
        sub_ax.spines["bottom"].set_linewidth(0.5)


def plot_kraft_panel(
    kraft: pd.DataFrame,
    row_order: list[str],
    cluster_kraft: pd.DataFrame,
    out_stem: Path,
) -> None:
    fig = plt.figure(figsize=(10.4, 5.6), constrained_layout=False)
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.45, 0.9],
        wspace=0.44,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    plot_kraft_zone_bars(
        ax_bars,
        kraft,
        row_order,
        "Native Kraft Zone 1-6 projection composition",
    )
    ax_cluster = fig.add_subplot(gs[0, 1])
    plot_kraft_cluster_zone_bars(
        ax_cluster,
        cluster_kraft,
        "Kraft Zone 1-6 by RPCA cluster",
    )
    fig.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.22, wspace=0.45)
    save_all(fig, out_stem)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04, fontsize: float = 13) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def build_composite_figure(
    display: pd.DataFrame,
    row_order: list[str],
    spatial_png: Path,
    sox6: FamilyClusterResult,
    calb1: FamilyClusterResult,
) -> None:
    fig_w = 18.0
    panel_b_aspect_w_over_h = 1.95
    panel_b_h_in = (fig_w - 0.4) / panel_b_aspect_w_over_h
    top_row_h_in = 5.0
    fig_h = top_row_h_in + panel_b_h_in + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[top_row_h_in, panel_b_h_in],
        width_ratios=[1.0, 1.35],
        hspace=0.18,
        wspace=0.20,
        left=0.08,
        right=0.985,
        top=0.96,
        bottom=0.03,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    group_mat, group_color = build_group_matrix(display)
    plot_grouped_broad_heatmap(
        ax_a, group_mat, group_color,
        COMPOSITE_BROAD_TITLE,
        title_fontsize=15,
        xtick_fontsize=12,
        ytick_fontsize=12,
        cbar_label_fontsize=12,
        cbar_tick_fontsize=11,
        territory_fontsize=12,
    )
    add_panel_label(ax_a, "a", x=-0.22, y=1.06, fontsize=28)

    gs_c = gs[0, 1].subgridspec(1, 2, wspace=0.28)
    ax_s = fig.add_subplot(gs_c[0, 0])
    plot_cluster_dendrogram(
        ax_s, sox6,
        f"Sox6 spatial clustering, K={sox6.selected_k}",
        title_fontsize=14, leaf_font_size=12, ylabel_fontsize=12, tick_fontsize=11,
    )
    add_panel_label(ax_s, "c", x=-0.14, y=1.05, fontsize=28)

    ax_c = fig.add_subplot(gs_c[0, 1])
    plot_cluster_dendrogram(
        ax_c, calb1,
        f"Calb1 spatial clustering, K={calb1.selected_k}",
        title_fontsize=14, leaf_font_size=12, ylabel_fontsize=12, tick_fontsize=11,
    )

    ax_b = fig.add_subplot(gs[1, :])
    anxa_img = crop_white_margin(np.asarray(Image.open(ANXA_FOCUS_PANEL_B).convert("RGB")))
    ax_b.imshow(anxa_img, interpolation="none", aspect="auto")
    ax_b.axis("off")
    ax_b.set_title(
        "Anxa+ DA (Sox6:Tafa1 + Sox6:Vcan) projection - putamen only",
        fontsize=16, fontweight="bold", pad=8,
    )
    add_panel_label(ax_b, "b", x=0.002, y=1.02, fontsize=28)

    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.035, hspace=0.08, wspace=0.28)
    pos_b = ax_b.get_position()
    ax_b.set_position([0.012, pos_b.y0, 0.978, pos_b.height])
    save_composite(fig, OUT_DIR / "figure11_human_projection_final")
    plt.close(fig)


def write_outputs(
    display: pd.DataFrame,
    accepted_raw: pd.DataFrame,
    rpca_raw: pd.DataFrame,
    bins: pd.DataFrame,
    kraft: pd.DataFrame,
    kraft_meta: pd.DataFrame,
    sox6: FamilyClusterResult,
    calb1: FamilyClusterResult,
    anxa_audit: AnxaSimilarityAudit,
    cluster_region_display: pd.DataFrame,
    cluster_region_raw: pd.DataFrame,
    cluster_kraft: pd.DataFrame,
    missing_kraft: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_outputs()

    pd.concat([sox6.metrics, calb1.metrics], ignore_index=True).to_csv(
        OUT_DIR / "rpca_spatial_family_cluster_metrics.csv",
        index=False,
    )
    pd.concat([sox6.all_memberships, calb1.all_memberships], ignore_index=True).to_csv(
        OUT_DIR / "rpca_spatial_family_cluster_multiresolution.csv",
        index=False,
    )
    cluster_region_display.to_csv(
        OUT_DIR / "rpca_spatial_cluster_region_display_targets.tsv",
        sep="\t",
    )
    cluster_region_raw.to_csv(
        OUT_DIR / "rpca_spatial_cluster_region_raw_targets.tsv",
        sep="\t",
    )
    (rpca_raw - accepted_raw).to_csv(
        OUT_DIR / "rpca_method_target_minus_accepted_target.tsv",
        sep="\t",
    )
    cluster_kraft.to_csv(
        OUT_DIR / "rpca_spatial_cluster_native_kraft_zone_means.tsv",
        sep="\t",
    )
    anxa_audit.table.to_csv(
        OUT_DIR / "rpca_spatial_anxa_similarity_audit.tsv",
        sep="\t",
        index=False,
    )
    kraft_zone_shares(kraft).to_csv(
        OUT_DIR / "native_kraft_zone_composition.tsv",
        sep="\t",
    )
    (kraft - KRAFT_UNIFORM).to_csv(
        OUT_DIR / "native_kraft_zone_share_minus_uniform.tsv",
        sep="\t",
    )
    missing_kraft.to_csv(
        OUT_DIR / "native_kraft_missing_subtypes_by_spatial_cluster.csv",
        index=False,
    )

    selected = {
        "seurat_rpca_spatial_clustering": {
            "feature_space": "panel_a_locked_spatial_display_map_row_l2_cosine",
            "selection_rule": "highest K within 95 percent of the best selectable silhouette; require non-singleton clusters when any such partition exists",
            "sox6": {
                "selected_k": sox6.selected_k,
                "clusters": sox6.selected_clusters,
            },
            "calb1": {
                "selected_k": calb1.selected_k,
                "clusters": calb1.selected_clusters,
            },
        }
    }
    (OUT_DIR / "rpca_spatial_family_selected_clusters.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )

    roundtrip_rows = []
    for subtype in rpca_raw.index:
        for region in REGION_ORDER:
            score_col = f"score_{safe_col(subtype)}"
            mass_col = f"mass_{safe_col(subtype)}"
            if score_col not in bins.columns or mass_col not in bins.columns:
                continue
            mask = bins["region"].astype(str) == region
            n = bins.loc[mask, "n_cells"].to_numpy(dtype=float)
            score = bins.loc[mask, score_col].to_numpy(dtype=float)
            mass = bins.loc[mask, mass_col].to_numpy(dtype=float)
            weighted_score = float(np.sum(score * n) / np.sum(n)) if np.sum(n) else np.nan
            roundtrip_rows.append(
                {
                    "subtype": subtype,
                    "region": region,
                    "target_raw": float(rpca_raw.loc[subtype, region]),
                    "weighted_score": weighted_score,
                    "abs_raw_roundtrip_error": abs(weighted_score - float(rpca_raw.loc[subtype, region])),
                    "target_mass": float(rpca_raw.loc[subtype, region]),
                    "mass_sum": float(np.sum(mass)),
                    "abs_mass_roundtrip_error": abs(float(np.sum(mass)) - float(rpca_raw.loc[subtype, region])),
                }
            )
    roundtrip = pd.DataFrame(roundtrip_rows)
    roundtrip.to_csv(OUT_DIR / "rpca_panel_a_roundtrip_qc.csv", index=False)

    qc = {
        "max_abs_raw_roundtrip_error": float(roundtrip["abs_raw_roundtrip_error"].max()),
        "max_abs_mass_roundtrip_error": float(roundtrip["abs_mass_roundtrip_error"].max()),
        "n_rpca_spatial_bins": int(bins.shape[0]),
        "n_panel_a_subtypes": int(display.shape[0]),
        "extended_spatial_atlas_render_texture_ratio_clip": list(EXTENDED_TEXTURE_RATIO_CLIP),
        "max_abs_rpca_method_target_minus_accepted_target": float(
            np.nanmax(np.abs((rpca_raw - accepted_raw).to_numpy(dtype=float)))
        ),
        "n_native_kraft_subtypes": int(kraft.shape[0]),
        "anxa_like_correlation_threshold": ANXA_LIKE_CORR_THRESHOLD,
        "anxa_like_calb1_rows": anxa_audit.flagged_calb1,
        "max_calb1_signed_spatial_corr_to_anxa": float(
            anxa_audit.table.loc[
                anxa_audit.table["family"].eq("Calb1"),
                "signed_spatial_corr_to_anxa",
            ].max()
        ),
        "native_kraft_missing_panel_a_subtypes": sorted(
            [idx for idx in display.index if idx not in kraft.index]
        ),
        "native_kraft_zones": KRAFT_ZONE_ORDER,
        "native_kraft_zone_domains": ZONE_DOMAIN_LABELS,
        "native_kraft_zone_interpretation_source": KRAFT_ZONE_INTERPRETATION_SOURCE,
        "kraft_zone_bin_counts": {
            f"Z{zone}": int(count)
            for zone, count in kraft_meta["zone"].astype(str).value_counts().sort_index().items()
        },
    }
    (OUT_DIR / "figure11_human_projection_final_qc.json").write_text(
        json.dumps(qc, indent=2) + "\n"
    )

    input_paths = {
        "panel_a_display_targets": PANEL_A_DISPLAY,
        "panel_a_raw_targets": PANEL_A_RAW,
        "rpca_method_raw_targets": RPCA_METHOD_RAW,
        "rpca_panel_a_locked_bins": RPCA_BINS,
        "spatial_renderer_script": SPATIAL_RENDERER_SCRIPT,
        "native_kraft_zone_matrix": KRAFT_ZONE_MATRIX,
        "native_kraft_bin_metadata": KRAFT_BIN_METADATA,
        "soft_transfer_pairwise": SOFT_TRANSFER_PAIRWISE,
        "soft_transfer_summary": SOFT_TRANSFER_SUMMARY,
        "soft_transfer_multiscale": SOFT_TRANSFER_MULTISCALE,
    }
    manifest = {
        "figure": "figure11_human_projection_final",
        "purpose": "human DA projection topography with Panel-A broad regions, RPCA spatial atlas, and RPCA spatial clustering",
        "outputs": {
            "composite": "figure11_human_projection_final.svg/png/pdf",
            "broad_heatmap": "fig11_final_panel_a_broad_region_heatmap.svg/png/pdf",
            "spatial_atlas": f"{RPCA_SELECTED_CLUSTER_FIG_STEM}.svg/png/pdf",
            "spatial_atlas_texture_similarity_qc": f"{RPCA_SELECTED_CLUSTER_FIG_STEM}_texture_similarity_qc.csv",
            "family_clustering": "fig11_final_panel_c_rpca_spatial_family_clustering.svg/png/pdf",
            "anxa_similarity_audit_table": "rpca_spatial_anxa_similarity_audit.tsv",
        },
        "inputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(path),
            }
            for name, path in input_paths.items()
        },
        "methods": {
            "broad_region_heatmap": "frozen Panel-A display targets from the accepted Panel-A constrained spatial-transfer output",
            "spatial_atlas": "seurat_rpca Panel-A-locked bins rendered with the current Calb1 unsupervised cluster columns and the fixed Sox6 display columns; for rendering only, subtype score columns are rebuilt from raw shape maps with a looser 0.25-4.0 unit-mean ratio clip so within-region texture is not collapsed by the original 0.70-1.45 panel-A-locking clip; raw spatial panels use constant opacity over the rendered mask so low-but-real signal remains visible as dark inferno color",
            "spatial_clustering": "average-linkage hierarchical clustering with cosine distance on row-L2-normalized RPCA Panel-A-locked spatial display maps; bins weighted by sqrt(n_cells); RPCA method raw targets are used for exact atlas round-trip",
            "anxa_similarity_audit": "Pearson correlation of each subtype to the mean Anxa1+ Tafa1/Vcan spatial map, computed on the signed RPCA Panel-A-locked display atlas; a region-normalized within-region texture correlation is retained in the audit table but not shown in the final composite",
            "kraft_validation": "native Kraft/Macosko Zone 1-6 receiver matrix from Final_Zone_Assignments_No_Smooth; the figure shows raw Zone 1-6 composition bars by subtype and by RPCA spatial cluster without rank-transforming subtypes or collapsing zones into inferred domain weights",
            "kraft_zone_labels": KRAFT_ZONE_INTERPRETATION_SOURCE,
        },
        "anxa_similarity_audit": {
            "anchors": list(ANXA_ANCHORS),
            "threshold": ANXA_LIKE_CORR_THRESHOLD,
            "flagged_calb1_rows": anxa_audit.flagged_calb1,
        },
        "selected_clusters": selected["seurat_rpca_spatial_clustering"],
        "qc": qc,
    }
    (OUT_DIR / "figure11_human_projection_final_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    display, accepted_raw, rpca_raw, bins, kraft, kraft_meta = load_inputs()
    row_order = ordered_subtypes(display)

    sox6 = cluster_family("Sox6", bins, display, rpca_raw)
    calb1 = cluster_family("Calb1", bins, display, rpca_raw)
    anxa_audit = build_anxa_similarity_audit(bins, row_order, display, rpca_raw)
    cluster_region_display, cluster_region_raw, cluster_kraft, missing_kraft = build_cluster_tables(
        [sox6, calb1],
        display,
        rpca_raw,
        kraft,
    )

    write_outputs(
        display,
        accepted_raw,
        rpca_raw,
        bins,
        kraft,
        kraft_meta,
        sox6,
        calb1,
        anxa_audit,
        cluster_region_display,
        cluster_region_raw,
        cluster_kraft,
        missing_kraft,
    )

    spatial_png = render_selected_cluster_spatial_atlas(
        bins,
        rpca_raw,
        calb1,
        row_order,
        cell_counts=KAMATH_SUBTYPE_COUNTS,
    )

    render_selected_cluster_spatial_atlas(
        bins,
        rpca_raw,
        calb1,
        row_order,
        display_columns=ungrouped_spatial_display_columns(row_order),
        fig_stem=RPCA_SUPPLEMENTARY_FIG_STEM,
        display_label_overrides=ungrouped_display_labels(row_order),
    )

    group_mat, group_color = build_group_matrix(display)
    group_mat.to_csv(OUT_DIR / "fig11_final_panel_a_broad_region_heatmap.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    plot_grouped_broad_heatmap(
        ax, group_mat, group_color, STANDALONE_BROAD_TITLE,
        xtick_fontsize=6.5, ytick_fontsize=7.5, territory_fontsize=7.5,
    )
    fig.subplots_adjust(left=0.26, right=0.9, top=0.78, bottom=0.28)
    save_all(fig, OUT_DIR / "fig11_final_panel_a_broad_region_heatmap")
    plt.close(fig)

    plot_family_clustering_panel(
        sox6,
        calb1,
        cluster_region_display,
        OUT_DIR / "fig11_final_panel_c_rpca_spatial_family_clustering",
    )
    build_composite_figure(
        display,
        row_order,
        spatial_png,
        sox6,
        calb1,
    )

    print(f"wrote {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Sox6 clusters: K={sox6.selected_k} {sox6.selected_clusters}")
    print(f"Calb1 clusters: K={calb1.selected_k} {calb1.selected_clusters}")


if __name__ == "__main__":
    main()
