#!/usr/bin/env python3
"""Build the verified source-dataset registry for atlas v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fda_atlas_v2.contracts import validate_table  # noqa: E402


DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
MANUSCRIPT = Path(
    os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent)
).expanduser()
RELEASE = ROOT / "release/development"
HUMAN_WHB = DATA / "human/atlas/allen/whole_human_brain/WHB-10Xv3/20240330"
MOUSE_DA = MANUSCRIPT / "scDRS/input/lrrk2finaldataset_da_filtered.h5ad"
RHESUS_HMBA = DATA / "macaque/allen/HMBA-10xMultiome-BG-Macaque-raw.h5ad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def h5ad_shape(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        shape = matrix.attrs["shape"] if "shape" in matrix.attrs else matrix.shape
    return int(shape[0]), int(shape[1])


def require_shape(path: Path, expected: tuple[int, int]) -> None:
    observed = h5ad_shape(path)
    if observed != expected:
        raise ValueError(f"unexpected H5AD shape for {path}: {observed} != {expected}")


def paths_json(paths: list[Path]) -> str:
    return json.dumps([str(path.resolve()) for path in paths], separators=(",", ":"))


def manifest_fields(path: Path) -> dict:
    return {
        "source_manifest_path": str(path.resolve()),
        "source_manifest_sha256": sha256(path),
    }


def build(release_id: str, release_dir: Path) -> pd.DataFrame:
    human_manifest_path = release_dir / "human_integrated_browser_equivalence_manifest.json"
    human_manifest = read_json(human_manifest_path)
    human = human_manifest["scientific_object"]
    human_path = Path(human["path"])
    require_shape(human_path, (22_871, 25_037))
    if human.get("shape") != [22_871, 25_037] or not human_manifest["checks"].get(
        "X_nonnegative_integer_counts"
    ):
        raise ValueError("human integrated manifest does not verify the frozen raw counts")

    scdrs_manifest_path = release_dir / "scdrs_population_effects_manifest.json"
    scdrs_manifest = read_json(scdrs_manifest_path)
    require_shape(MOUSE_DA, (27_621, 32_285))
    mouse_input = scdrs_manifest["inputs"].get(str(MOUSE_DA))
    if not mouse_input or len(str(mouse_input.get("sha256", ""))) != 64:
        raise ValueError("mouse DA input lacks its recorded content checksum")

    human_whb_paths = [
        HUMAN_WHB / "WHB-10Xv3-Neurons-raw.h5ad",
        HUMAN_WHB / "WHB-10Xv3-Nonneurons-raw.h5ad",
    ]
    human_whb_shapes = [(2_480_956, 59_357), (888_263, 59_357)]
    for path, shape in zip(human_whb_paths, human_whb_shapes):
        require_shape(path, shape)
    human_whb_manifest_path = release_dir / "human_wholebrain_da_exclusion_manifest.json"
    human_whb_manifest = read_json(human_whb_manifest_path)
    if human_whb_manifest["counts"].get("registry_cells") != 2_026:
        raise ValueError("human WHB manifest does not contain the 2,026-cell DA exclusion")

    mouse_config_path = ROOT / "configs/mouse_whb_partitions.yaml"
    mouse_config = yaml.safe_load(mouse_config_path.read_text())
    if len(mouse_config) != 13:
        raise ValueError("mouse whole-brain registry requires all 13 partitions")
    mouse_whb_paths = []
    for partition, record in mouse_config.items():
        path = Path(
            str(record["raw_h5ad"]).replace("${DOPABASE_DATA_ROOT}", str(DATA))
        ).expanduser().resolve()
        require_shape(path, (int(record["raw_rows"]), 32_285))
        mouse_whb_paths.append(path)
    if sum(int(record["raw_rows"]) for record in mouse_config.values()) != 2_349_544:
        raise ValueError("mouse whole-brain raw rows do not sum to 2,349,544")
    mouse_whb_manifest_path = release_dir / "mouse_wholebrain_da_exclusion_manifest.json"
    mouse_whb_manifest = read_json(mouse_whb_manifest_path)
    if (
        mouse_whb_manifest["counts"].get("partitions") != 13
        or mouse_whb_manifest["counts"].get("total_comparator_exclusions") != 13_217
    ):
        raise ValueError("mouse whole-brain exclusion provenance is incomplete")

    hmba_manifest_path = release_dir / "rhesus_hmba_input_manifest.json"
    hmba_manifest = read_json(hmba_manifest_path)
    hmba_input = hmba_manifest.get("inputs", {}).get(str(RHESUS_HMBA.resolve()), {})
    if (
        hmba_manifest.get("species_scientific_name") != "Macaca mulatta"
        or hmba_manifest.get("taxon") != "NCBITaxon:9544"
        or hmba_manifest.get("counts", {}).get("cells") != 3_368
        or hmba_manifest.get("counts", {}).get("donors") != 4
        or not RHESUS_HMBA.is_file()
        or h5ad_shape(RHESUS_HMBA) != (839_102, 35_219)
        or hmba_input.get("sha256") != sha256(RHESUS_HMBA)
    ):
        raise ValueError("Allen HMBA strict rhesus source is not provenance locked")
    rhesus_manifest_path = release_dir / "rhesus_chiou_input_manifest.json"
    rhesus_manifest = read_json(rhesus_manifest_path)
    rhesus_input = rhesus_manifest.get("input", {})
    rhesus_matrix = rhesus_manifest.get("matrix", {})
    rhesus_da = rhesus_manifest.get("da_membership", {})
    rhesus_path = Path(str(rhesus_input.get("path", "")))
    rhesus_membership = Path(str(rhesus_da.get("membership_path", "")))
    if (
        rhesus_manifest.get("release_id") != release_id
        or not rhesus_path.is_file()
        or rhesus_path.stat().st_size != int(rhesus_input.get("size_bytes", -1))
        or not rhesus_membership.is_file()
        or sha256(rhesus_membership) != rhesus_da.get("membership_sha256")
        or rhesus_matrix.get("cells") != 2_583_967
        or rhesus_matrix.get("genes") != 29_060
        or rhesus_matrix.get("all_nonnegative_integers") is not True
        or rhesus_da.get("cells") != 3_645
        or rhesus_da.get("donors") != 5
    ):
        raise ValueError("Chiou rhesus primary atlas is not provenance locked")

    rows = [
        {
            "release_id": release_id,
            "dataset_id": "dopabase_human_integrated_da_v2",
            "species": "human",
            "dataset_role": "integrated_da_atlas",
            "assay": "single-nucleus RNA-seq",
            "anatomy_scope": "Kamath substantia nigra DA plus exact Siletti DA bridge",
            "source_name": "DopaBase Human corrected Kamath-Siletti integration",
            "source_release": "atlas v2 frozen corrected integration",
            "source_paths_json": paths_json([human_path]),
            **manifest_fields(human_manifest_path),
            "matrix_representation": "raw integer counts for quantification; integrated labels and embeddings for identity and display",
            "normalization_or_transformation": "raw counts summed by study, donor, and HMoE population, then full-library log2 CPM with study retained as a stratum",
            "n_cells": 22_871,
            "n_features": 25_037,
            "rank_cohort_n_cells": 16_507,
            "rank_cohort_n_donors": 11,
            "availability_status": "available_verified",
            "rank_role": "primary_within_da",
            "identity_role": "Kamath-grounded population universe with Kamath and Siletti as primary integrated study strata; Siletti is also the Allen bridge",
            "sha256": human["sha256"],
            "checksum_status": "manifest_recorded_content_sha256",
        },
        {
            "release_id": release_id,
            "dataset_id": "allen_human_whb_10xv3_20240330",
            "species": "human",
            "dataset_role": "whole_brain_comparator",
            "assay": "single-nucleus RNA-seq",
            "anatomy_scope": "whole human brain, neuronal plus non-neuronal partitions",
            "source_name": "Allen-hosted Siletti WHB-10Xv3",
            "source_release": "WHB-10Xv3/20240330 expression; 20241115 metadata",
            "source_paths_json": paths_json(human_whb_paths),
            **manifest_fields(human_whb_manifest_path),
            "matrix_representation": "partitioned nonnegative integer raw counts",
            "normalization_or_transformation": "source-matched donor-population raw pseudobulks, then full-library log2 CPM within Allen WHB",
            "n_cells": 3_369_219,
            "n_features": 59_357,
            "rank_cohort_n_cells": pd.NA,
            "rank_cohort_n_donors": pd.NA,
            "availability_status": "available_verified",
            "rank_role": "primary_brainwide",
            "identity_role": "exact 823-cell Kamath-grounded focal bridge plus frozen 2,026-cell DA exclusion; all eligible neuronal and non-neuronal cells form the comparator",
            "sha256": "",
            "checksum_status": "component_content_sha256_not_recomputed",
        },
        {
            "release_id": release_id,
            "dataset_id": "dopabase_mouse_da_lrrk2",
            "species": "mouse",
            "dataset_role": "integrated_da_atlas",
            "assay": "single-nucleus RNA-seq",
            "anatomy_scope": "midbrain DA focal atlas",
            "source_name": "DopaBase mouse control and LRRK2 DA atlas",
            "source_release": "frozen manuscript scDRS input",
            "source_paths_json": paths_json([MOUSE_DA]),
            **manifest_fields(scdrs_manifest_path),
            "matrix_representation": "nonnegative integer raw counts with frozen integrated subtype labels",
            "normalization_or_transformation": "raw counts collapsed to canonical symbols, summed by donor and HMoE population, then full-library log2 CPM within mouse DA",
            "n_cells": 27_621,
            "n_features": 32_285,
            "rank_cohort_n_cells": 24_419,
            "rank_cohort_n_donors": 4,
            "availability_status": "available_verified",
            "rank_role": "primary_within_da",
            "identity_role": "frozen integrated DopaBase mouse HMoE labels across two control and two LRRK2 donors",
            "sha256": mouse_input["sha256"],
            "checksum_status": "manifest_recorded_content_sha256",
        },
        {
            "release_id": release_id,
            "dataset_id": "allen_mouse_wmb_10xv3_20230630",
            "species": "mouse",
            "dataset_role": "whole_brain_comparator",
            "assay": "single-nucleus RNA-seq",
            "anatomy_scope": "whole mouse brain across all 13 WMB divisions",
            "source_name": "Allen WMB-10Xv3",
            "source_release": "WMB-10Xv3/20230630 expression; WMB-10X/20241115 metadata",
            "source_paths_json": paths_json(mouse_whb_paths),
            **manifest_fields(mouse_whb_manifest_path),
            "matrix_representation": "13-partition nonnegative integer raw counts",
            "normalization_or_transformation": "source-matched donor-population raw pseudobulks, then full-library log2 CPM within Allen WMB",
            "n_cells": 2_349_544,
            "n_features": 32_285,
            "rank_cohort_n_cells": pd.NA,
            "rank_cohort_n_donors": pd.NA,
            "availability_status": "available_verified",
            "rank_role": "primary_brainwide",
            "identity_role": "all 13 raw partitions with 8,409 Allen Dopa-taxonomy cells, 4,808 ambiguous annotations, and all raw-only rows excluded from non-DA comparison",
            "sha256": "",
            "checksum_status": "component_content_sha256_not_recomputed",
        },
        {
            "release_id": release_id,
            "dataset_id": "allen_hmba_rhesus_bg_raw_counts",
            "species": "macaque",
            "dataset_role": "focal_da_atlas",
            "assay": "single-nucleus RNA-seq",
            "anatomy_scope": "Allen HMBA basal ganglia taxonomy-defined midbrain DA",
            "source_name": "Allen HMBA rhesus basal ganglia multiome RNA",
            "source_release": "HMBA-10xMultiome-BG/20250630",
            "source_paths_json": paths_json([RHESUS_HMBA]),
            **manifest_fields(hmba_manifest_path),
            "matrix_representation": "nonnegative integer raw counts",
            "normalization_or_transformation": "strict NCBITaxon:9544 filter, taxonomy-defined DA selection, HMoE assignment, duplicate symbols summed, donor-population raw pseudobulk, then full-library log2 CPM",
            "n_cells": 839_102,
            "n_features": 35_219,
            "rank_cohort_n_cells": 3_368,
            "rank_cohort_n_donors": 4,
            "availability_status": "available_verified",
            "rank_role": "primary_within_da",
            "identity_role": "Allen HMBA NCBITaxon:9544 Macaca mulatta taxonomy-defined DA cells and branch-free HMoE populations drive donor-aware within-DA inference; Macaca nemestrina is excluded",
            "sha256": hmba_input["sha256"],
            "checksum_status": "manifest_recorded_content_sha256",
        },
        {
            "release_id": release_id,
            "dataset_id": "chiou_rhesus_brainwide_scrnaseq3",
            "species": "macaque",
            "dataset_role": "whole_brain_comparator",
            "assay": "sci-RNA-seq3",
            "anatomy_scope": "30 sampled rhesus brain regions, neuronal plus non-neuronal",
            "source_name": "Chiou adult rhesus macaque brain atlas",
            "source_release": "CELLxGENE dataset version 5bd7d724-1f53-474f-8391-57aadfc24864",
            "source_paths_json": paths_json([rhesus_path]),
            **manifest_fields(rhesus_manifest_path),
            "matrix_representation": "raw.X CSR nonnegative integer raw counts",
            "normalization_or_transformation": "exact source DA exclusion, donor-region raw pseudobulk, full-library log2 CPM, then equal donor weighting within each Chiou region",
            "n_cells": 2_583_967,
            "n_features": 29_060,
            "rank_cohort_n_cells": 2_580_322,
            "rank_cohort_n_donors": 5,
            "availability_status": "available_verified",
            "rank_role": "primary_brainwide",
            "identity_role": "cross-cohort shared reference only; all 3,645 exact source CL:0000700 dopaminergic cells are excluded across every source region before donor-balanced non-DA aggregation",
            "sha256": rhesus_input["sha256"],
            "checksum_status": "manifest_recorded_content_sha256",
        },
    ]
    frame = pd.DataFrame(rows)
    validate_table("dataset_registry", frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--release-dir", type=Path, default=RELEASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else release_dir / "dataset_registry.parquet"
    )
    frame = build(args.release_id, release_dir)
    partial = output.with_suffix(".parquet.partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(output)
    manifest = {
        "release_id": args.release_id,
        "scope": "source datasets defining DA identity, scDRS, and both selectivity axes",
        "rows": len(frame),
        "species": frame["species"].value_counts().sort_index().to_dict(),
        "availability": frame["availability_status"].value_counts().sort_index().to_dict(),
        "output": {"path": str(output), "sha256": sha256(output)},
        "pipeline": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/contracts.py"): sha256(
                SRC / "fda_atlas_v2/contracts.py"
            ),
            str(SRC / "fda_atlas_v2/dataset_registry.py"): sha256(
                SRC / "fda_atlas_v2/dataset_registry.py"
            ),
        },
    }
    manifest_path = release_dir / "dataset_registry_manifest.json"
    manifest_partial = manifest_path.with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
