#!/usr/bin/env python3
"""Build the Figure 5 HMBA-BG projection fields from the shared Figure 4 substrate."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import pandas as pd
from nilearn import image
from scipy.ndimage import distance_transform_edt, gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCKED_BINS = (
    PROJECT_ROOT
    / "figures/fig4_projections/frozen/extended_arm/arm2_matched_panelA_locked_bins.parquet"
)
DEFAULT_ANNOTATION = (
    PROJECT_ROOT
    / "figures/fig5_datscan_depletion/frozen/annotations_compressed_700.nii.gz"
)
FIG4_SCORING = (
    PROJECT_ROOT
    / "figures/fig4_projections/compute/extended_arm/render_extended_remap_enrichment.py"
)

GROUP_ORDER = ("Anxa1", "Sox6-all", "Sox6-rest", "Calb1")
GROUP_MEMBERS = {
    "Anxa1": ["Sox6:Tafa1", "Sox6:Vcan"],
    "Sox6-all": [
        "Sox6:Tafa1",
        "Sox6:Vcan",
        "Sox6:Kcnmb2",
        "Sox6:Arhgap28",
        "Sox6:March3",
        "Sox6:Tmem132d",
    ],
    "Sox6-rest": ["Sox6:Kcnmb2", "Sox6:Arhgap28", "Sox6:March3", "Sox6:Tmem132d"],
    "Calb1": [
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
    ],
}
LABELS = {
    "CaH": 121,
    "CaB": 122,
    "CaT": 123,
    "PuR": 125,
    "PuC": 126,
    "PuPV": 127,
    "NACc": 129,
    "NACs": 130,
}
GROSS_REGIONS = {
    "Ca": {"CaH", "CaB", "CaT"},
    "Pu": {"PuR", "PuC", "PuPV"},
    "NAC": {"NACc", "NACs"},
}
REGION_TO_GROSS = {region: gross for gross, regions in GROSS_REGIONS.items() for region in regions}
SPLAT_SIGMA = 2.6


@dataclass(frozen=True)
class HmbaProjectionFrame:
    reference: nib.spatialimages.SpatialImage
    annotation: np.ndarray
    affine: np.ndarray
    striatum: np.ndarray
    putamen: np.ndarray
    caudate: np.ndarray
    centered_fields: dict[str, np.ndarray]
    noncentered_fields: dict[str, np.ndarray]
    fsl_2mm_path: Path
    locked_bins_path: Path
    annotation_path: Path

    def to_h(self, values: np.ndarray) -> np.ndarray:
        """Resample one FSL 2 mm array to the 1 mm HMBA frame and striatal mask."""
        fsl = nib.load(self.fsl_2mm_path)
        source = nib.Nifti1Image(np.nan_to_num(values, nan=0).astype(np.float32), fsl.affine)
        resampled = image.resample_to_img(source, self.reference, interpolation="linear").get_fdata()
        return np.where(self.striatum, np.clip(resampled, 0, None), np.nan)


@dataclass(frozen=True)
class FslProjectionFields:
    reference: nib.spatialimages.SpatialImage
    striatum: np.ndarray
    fields: dict[str, np.ndarray]


def _required_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _load_fig4_scoring():
    path = _required_file(FIG4_SCORING, "Figure 4 projection scoring module")
    spec = importlib.util.spec_from_file_location("fig4_projection_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Figure 4 projection scoring module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_hmba_projection_frame(
    ppmi_root: str | Path,
    *,
    locked_bins_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
) -> HmbaProjectionFrame:
    ppmi_root = Path(ppmi_root).expanduser().resolve()
    fsl_2mm = _required_file(
        ppmi_root / "derived/av133_spatial/mni152_fsl_2mm.nii.gz",
        "PPMI FSL 2 mm reference",
    )
    locked_bins = _required_file(
        Path(locked_bins_path) if locked_bins_path is not None else DEFAULT_LOCKED_BINS,
        "Figure 4 arm2 locked-bin overlay",
    )
    annotation_file = _required_file(
        Path(annotation_path) if annotation_path is not None else DEFAULT_ANNOTATION,
        "HMBA-BG annotation overlay",
    )

    annotation_image = nib_processing.resample_to_output(
        nib.load(annotation_file), [1, 1, 1], order=0
    )
    annotation = np.asarray(annotation_image.dataobj).astype(int)
    affine = annotation_image.affine
    inverse_affine = np.linalg.inv(affine)
    shape = annotation.shape

    striatum = np.isin(annotation, list(LABELS.values()))
    putamen = np.isin(annotation, [LABELS[name] for name in ("PuR", "PuC", "PuPV")])
    caudate = np.isin(annotation, [LABELS[name] for name in ("CaH", "CaB", "CaT")])

    scoring = _load_fig4_scoring()
    bins = pd.read_parquet(locked_bins)
    raw = {
        group: scoring._group_bin_score(bins, members)
        for group, members in GROUP_MEMBERS.items()
    }
    base = np.mean([raw["Anxa1"], raw["Sox6-all"], raw["Calb1"]], axis=0)
    positive_base = base[base > 0]
    epsilon = float(np.median(positive_base)) * 0.1 if positive_base.size else 1e-9
    region = bins["region"].astype(str).to_numpy()
    for group in GROUP_ORDER:
        enrichment = np.log2((raw[group] + epsilon) / (base + epsilon))
        bins[f"enrnc_{group}"] = np.nan_to_num(
            enrichment.copy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        for members in GROSS_REGIONS.values():
            selected = np.isin(region, list(members))
            if selected.any():
                enrichment[selected] -= enrichment[selected].mean()
        bins[f"enr_{group}"] = np.nan_to_num(
            enrichment, nan=0.0, posinf=0.0, neginf=0.0
        )

    def left_extent(label_id: int) -> np.ndarray:
        voxels = np.argwhere(annotation == label_id)
        millimetres = (affine[:3, :3] @ voxels.T + affine[:3, 3:]).T
        return millimetres[millimetres[:, 0] < 0]

    atlas_extents = {
        region_name: np.percentile(left_extent(label_id), [2, 98], axis=0)
        for region_name, label_id in LABELS.items()
    }
    mapped = np.full((len(bins), 3), np.nan)
    for region_name in LABELS:
        selected = (bins["region"] == region_name).to_numpy()
        if not selected.any():
            continue
        coordinates = bins.loc[selected, ["x_ccf_mm", "y_ccf_mm", "z_ccf_mm"]].to_numpy()
        source_low, source_high = np.percentile(coordinates, [2, 98], axis=0)
        normalized = np.clip(
            (coordinates - source_low)
            / np.where(source_high - source_low > 1e-6, source_high - source_low, 1),
            0,
            1,
        )
        atlas_low, atlas_high = atlas_extents[region_name]
        mapped[selected] = atlas_low + normalized * (atlas_high - atlas_low)

    voxels = (
        np.column_stack([mapped, np.ones(len(bins), dtype=float)]) @ inverse_affine.T
    )[:, :3]
    indices = np.full((len(bins), 3), -1, dtype=int)
    finite_coordinates = np.isfinite(voxels).all(axis=1)
    indices[finite_coordinates] = np.rint(voxels[finite_coordinates]).astype(int)

    def splat(column: str) -> np.ndarray:
        density = np.zeros(shape)
        numerator = np.zeros(shape)
        i, j, k = indices.T
        cell_counts = bins["n_cells"].to_numpy(dtype=float)
        values = bins[column].to_numpy(dtype=float)
        valid_index = (
            (i >= 0)
            & (i < shape[0])
            & (j >= 0)
            & (j < shape[1])
            & (k >= 0)
            & (k < shape[2])
            & np.isfinite(values)
        )
        np.add.at(density, (i[valid_index], j[valid_index], k[valid_index]), cell_counts[valid_index])
        np.add.at(
            numerator,
            (i[valid_index], j[valid_index], k[valid_index]),
            values[valid_index] * cell_counts[valid_index],
        )
        density = gaussian_filter(density, SPLAT_SIGMA)
        numerator = gaussian_filter(numerator, SPLAT_SIGMA)
        field = np.full(shape, np.nan)
        for gross_region in ("Ca", "Pu", "NAC"):
            gross_mask = striatum & np.isin(
                annotation,
                [LABELS[name] for name in LABELS if REGION_TO_GROSS[name] == gross_region],
            )
            maximum_density = density[gross_mask].max() if gross_mask.any() else 0
            valid_field = (
                gross_mask & (density > 0.05 * maximum_density)
                if maximum_density > 0
                else np.zeros(shape, dtype=bool)
            )
            if not valid_field.any():
                continue
            field[valid_field] = numerator[valid_field] / density[valid_field]
            fill = gross_mask & ~valid_field
            nearest = distance_transform_edt(
                ~valid_field, return_distances=False, return_indices=True
            )
            field[fill] = field[tuple(nearest)][fill]
        return field

    midline_index = -affine[0, 3] / affine[0, 0]
    i_grid = np.arange(shape[0])[:, None, None]
    right = striatum & ((affine[0, 0] * i_grid + affine[0, 3]) > 0)

    def mirror(field: np.ndarray) -> np.ndarray:
        source_i = np.clip(
            np.rint(2 * midline_index - np.arange(shape[0])).astype(int), 0, shape[0] - 1
        )
        bilateral = field.copy()
        bilateral[right] = field[source_i, :, :][right]
        return bilateral

    centered = {group: mirror(splat(f"enr_{group}")) for group in GROUP_ORDER}
    noncentered = {group: mirror(splat(f"enrnc_{group}")) for group in GROUP_ORDER}
    return HmbaProjectionFrame(
        reference=annotation_image,
        annotation=annotation,
        affine=affine,
        striatum=striatum,
        putamen=putamen,
        caudate=caudate,
        centered_fields=centered,
        noncentered_fields=noncentered,
        fsl_2mm_path=fsl_2mm,
        locked_bins_path=locked_bins,
        annotation_path=annotation_file,
    )


def resample_group_fields_to_fsl(
    frame: HmbaProjectionFrame,
    fsl_path: str | Path | None = None,
) -> FslProjectionFields:
    """Resample the non-centered group fields and striatal mask to FSL 2 mm space."""
    reference_path = _required_file(
        Path(fsl_path) if fsl_path is not None else frame.fsl_2mm_path,
        "FSL 2 mm reference",
    )
    reference = nib.load(reference_path)

    def to_fsl(values: np.ndarray, interpolation: str) -> np.ndarray:
        source = nib.Nifti1Image(np.nan_to_num(values).astype(np.float32), frame.affine)
        return image.resample_to_img(source, reference, interpolation=interpolation).get_fdata()

    striatum = to_fsl(frame.striatum.astype(float), "nearest") > 0.5
    fields = {
        group: to_fsl(frame.noncentered_fields[group], "linear") for group in GROUP_ORDER
    }
    return FslProjectionFields(reference=reference, striatum=striatum, fields=fields)
