#!/usr/bin/env python3
"""Rebuild the strict 533-target, no-branch selectivity release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fda_atlas_v2.contracts import TRAITS, validate_table  # noqa: E402
from fda_atlas_v2.direct_target_universe import (  # noqa: E402
    add_direct_label_edges,
    build_direct_label_edges,
    direct_target_pairs,
)
from fda_atlas_v2.disk_pseudobulk import (  # noqa: E402
    combine_disk_pseudobulk_partitions,
    load_disk_pseudobulk_checkpoint,
)
from fda_atlas_v2.pseudobulk import (  # noqa: E402
    DiskPseudobulkResult,
    SparsePseudobulkResult,
    load_sparse_pseudobulk_checkpoint,
    save_sparse_pseudobulk_checkpoint,
)
from fda_atlas_v2.selectivity_pipeline import (  # noqa: E402
    chunked_strongest_competitor_selectivity,
)
from fda_atlas_v2.rhesus_hmba import population_members  # noqa: E402
from fda_atlas_v2.target_selectivity import (  # noqa: E402
    assemble_target_population_selectivity,
    build_target_gene_registry,
)
from rebuild_within_da_rest import (  # noqa: E402
    ANXA1_LEAVES,
    build_pooled_rest_pseudobulk,
    load_checkpoints,
    population_levels,
)
from recompute_brainwide_all_da_donors import (  # noqa: E402
    bh,
    build_populations,
    load_da,
    load_whb_reference,
)


RELEASE = ROOT / "release/development"
OUTPUT = ROOT / "results/direct_target_rebuild"
DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
MANUSCRIPT = Path(
    os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent)
).expanduser()
MOUSE_ORTHOLOGS = MANUSCRIPT / "Data/Mouse_Human_Gene_Orthologs.tsv"
MACAQUE_ORTHOLOGS = DATA / "cache/ensembl/116/human_macaca_mulatta_orthologs.tsv"
NEW_TARGETS = {"IL23R", "SOD1", "SMN2", "LDHA", "APOB"}
MOUSE_CACHE = OUTPUT / "mouse_brainwide_target_checkpoints"
MOUSE_AXIS_CACHE = OUTPUT / "mouse_brainwide_new_axes"
MACAQUE_AXIS_CACHE = OUTPUT / "macaque_axes"
MOUSE_REGION = DATA / "derived/m2h_fda_scdrs_atlas_v2/mouse_whb_region_pseudobulk_pooled"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def active_populations() -> dict[str, list[str]]:
    registry = pd.read_parquet(RELEASE / "population_registry.parquet")
    active = registry[
        registry["release_availability"].astype(str).eq("available")
        & ~registry["hierarchy_level"].astype(str).eq("branch")
        & ~registry["population_id"].astype(str).str.contains("/", regex=False)
    ].copy()
    output = {
        species: sorted(
            active.loc[active["species"].astype(str).eq(species), "population_id"]
            .astype(str)
            .unique()
        )
        for species in ("human", "mouse", "macaque")
    }
    for species in ("human", "mouse"):
        if "Anxa1" not in output[species]:
            output[species].append("Anxa1")
            output[species].sort()
    output["macaque"] = sorted(population_members())
    expected = {"human": 22, "mouse": 22, "macaque": 22}
    observed = {species: len(values) for species, values in output.items()}
    if observed != expected:
        raise ValueError(f"no-branch population grid differs: {observed}")
    return output


def build_universe() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    original_edges = pd.read_parquet(RELEASE / "drug_target_edges.parquet")
    labels = build_direct_label_edges(pd.read_parquet(RELEASE / "active_moieties.parquet"))
    edges = add_direct_label_edges(original_edges, labels)
    substance = pd.read_parquet(RELEASE / "regulatory_substance_target_evidence.parquet")
    targets = direct_target_pairs(edges, substance)
    registries = {
        "human": build_target_gene_registry(targets, species="human"),
        "mouse": build_target_gene_registry(
            targets,
            species="mouse",
            orthologs=pd.read_csv(MOUSE_ORTHOLOGS, sep="\t", dtype=str),
        ),
        "macaque": build_target_gene_registry(
            targets,
            species="macaque",
            orthologs=pd.read_csv(MACAQUE_ORTHOLOGS, sep="\t", dtype=str),
        ),
    }
    for species, registry in registries.items():
        registry.insert(0, "release_id", "development")
        if len(registry) != 533:
            raise ValueError(f"{species} target registry is not complete")
    return edges, registries, labels


def human_brainwide_axes(
    target_registry: pd.DataFrame,
    populations: list[str],
) -> dict[str, pd.DataFrame]:
    counts, detected, da_genes, da_meta = load_da()
    ref_counts, ref_lib, ref_detected, ref_cells, whb_genes = load_whb_reference()
    da_index = {gene: index for index, gene in enumerate(da_genes)}
    whb_index = {gene: index for index, gene in enumerate(whb_genes)}
    targets = set(target_registry["species_gene"].astype(str).str.upper())
    common = sorted(targets & set(da_genes) & set(whb_genes))
    dcols = np.asarray([da_index[gene] for gene in common])
    wcols = np.asarray([whb_index[gene] for gene in common])
    donors = sorted(da_meta["donor_id"].astype(str).unique())
    anatomies = sorted(ref_counts)
    reference = {
        anatomy: np.log2((ref_counts[anatomy][wcols] + 0.5) / ref_lib[anatomy] * 1e6)
        for anatomy in anatomies
    }
    reference_detection = {
        anatomy: ref_detected[anatomy][wcols] / ref_cells[anatomy]
        for anatomy in anatomies
    }
    memberships = build_populations(set(da_meta["population_id"].astype(str)))
    if set(populations) != set(memberships):
        raise ValueError("human brain-wide population membership differs from no-branch grid")
    output = {}
    for population in populations:
        focal = {}
        focal_detection = {}
        focal_cells = {}
        for donor in donors:
            mask = (
                da_meta["donor_id"].astype(str).eq(donor)
                & da_meta["population_id"].astype(str).isin(memberships[population])
            ).to_numpy()
            if not mask.any():
                continue
            library = float(da_meta.loc[mask, "library_size"].sum())
            cells = int(da_meta.loc[mask, "n_cells"].sum())
            if library <= 0 or cells <= 0:
                continue
            focal[donor] = np.log2(
                (counts[np.ix_(mask, dcols)].sum(axis=0) + 0.5) / library * 1e6
            )
            focal_detection[donor] = detected[np.ix_(mask, dcols)].sum(axis=0) / cells
            focal_cells[donor] = cells
        ordered = sorted(focal)
        if len(ordered) < 3:
            output[population] = pd.DataFrame(
                {
                    "gene_id": common,
                    "effect": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "detection_difference": np.nan,
                    "competitor_id": None,
                    "n_common_donors": len(ordered),
                    "focal_n_cells": sum(focal_cells.values()),
                    "focal_detection": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "competitor_loo_selection_fraction": np.nan,
                    "common_donor_ids": " | ".join(ordered),
                    "n_eligible_comparators": 0,
                    "status": "insufficient_comparator_support",
                }
            )
            print(f"  human brainwide {population} insufficient donors={len(ordered)}")
            continue
        matrix = np.vstack([focal[donor] for donor in ordered])
        detection_matrix = np.vstack([focal_detection[donor] for donor in ordered])
        n_donors = len(ordered)
        effects = []
        errors = []
        for anatomy in anatomies:
            differences = matrix - reference[anatomy][None, :]
            effects.append(differences.mean(axis=0))
            errors.append(differences.std(axis=0, ddof=1) / np.sqrt(n_donors))
        effect_matrix = np.vstack(effects)
        error_matrix = np.vstack(errors)
        strongest = np.argmin(effect_matrix, axis=0)
        positions = np.arange(len(common))
        effect = effect_matrix[strongest, positions]
        standard_error = error_matrix[strongest, positions]
        focal_det = detection_matrix.mean(axis=0)
        comparator_det = np.vstack(
            [reference_detection[anatomy] for anatomy in anatomies]
        )[strongest, positions]
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic = np.where(standard_error > 0, effect / standard_error, np.nan)
        p_value = stats.t.sf(statistic, df=n_donors - 1)
        output[population] = pd.DataFrame(
            {
                "gene_id": common,
                "effect": effect,
                "ci_low": effect - stats.t.ppf(0.975, df=n_donors - 1) * standard_error,
                "ci_high": effect + stats.t.ppf(0.975, df=n_donors - 1) * standard_error,
                "detection_difference": focal_det - comparator_det,
                "competitor_id": [anatomies[index] for index in strongest],
                "n_common_donors": n_donors,
                "focal_n_cells": sum(focal_cells.values()),
                "focal_detection": focal_det,
                "p_value": p_value,
                "q_value": bh(p_value),
                "competitor_loo_selection_fraction": np.nan,
                "common_donor_ids": " | ".join(ordered),
                "n_eligible_comparators": len(anatomies),
                "status": "estimated",
            }
        )
        print(f"  human brainwide {population}")
    return output


def load_checkpoint(path: Path):
    representation = json.loads(path.read_text()).get("matrix_representation")
    if representation == "disk_backed_raw_count_sums_by_donor_population":
        return load_disk_pseudobulk_checkpoint(path, verify_hashes=False)
    if representation == "raw_count_sums_by_donor_population":
        return load_sparse_pseudobulk_checkpoint(path)
    raise ValueError(f"unsupported pseudobulk checkpoint: {path}")


def subset_checkpoint(result, names, wanted: set[str]) -> tuple[SparsePseudobulkResult, tuple[str, ...]]:
    positions = [index for index, name in enumerate(names) if str(name) in wanted]
    selected_names = tuple(str(names[index]) for index in positions)
    if not selected_names:
        raise ValueError("no requested targets occur in a pseudobulk checkpoint")
    counts = result.counts[:, positions]
    detected = result.detected_cells[:, positions]
    return (
        SparsePseudobulkResult(
            counts=sp.csr_matrix(np.asarray(counts) if not sp.issparse(counts) else counts),
            detected_cells=sp.csr_matrix(
                np.asarray(detected) if not sp.issparse(detected) else detected
            ),
            metadata=result.metadata.copy(),
            feature_indices=np.arange(len(positions), dtype=np.int64),
        ),
        selected_names,
    )


def compute_mouse_brainwide_new_axes(
    target_registry: pd.DataFrame,
    populations: list[str],
) -> dict[str, pd.DataFrame]:
    plan = pd.read_parquet(RELEASE / "genomewide_selectivity_job_plan.parquet")
    template = plan[
        plan["species"].astype(str).eq("mouse")
        & plan["axis_id"].astype(str).eq("brainwide")
    ].iloc[0]
    template_paths = mouse_brainwide_source_paths(template)
    comparator_paths = [Path(value) for value in json.loads(template["comparator_tables_json"])]
    comparator_populations = sorted(
        {
            value
            for path in comparator_paths
            for value in pd.read_parquet(path)["population_id"].dropna().astype(str)
        }
    )
    eligible = target_registry[
        target_registry["human_gene"].astype(str).isin(NEW_TARGETS)
        & target_registry["gene_mapping_status"].astype(str).eq(
            "reciprocal_one_to_one_mouse_ortholog"
        )
    ]
    wanted = set(eligible["species_gene"].astype(str))
    nonfocal_paths = template_paths[2:]
    nonfocal = []
    nonfocal_names = []
    for index, _path in enumerate(nonfocal_paths, start=2):
        selected, selected_names = load_sparse_pseudobulk_checkpoint(
            MOUSE_CACHE / f"mouse_bw_targets_{index}.manifest.json"
        )
        nonfocal.append(selected)
        nonfocal_names.append(selected_names)
    if len(set(nonfocal_names)) != 1:
        raise ValueError("mouse whole-brain target feature order differs by partition")
    output = {}
    for level in ("family", "leaf"):
        focal_index = 0 if level == "family" else 1
        focal, focal_names = load_sparse_pseudobulk_checkpoint(
            MOUSE_CACHE / f"mouse_bw_targets_{focal_index}.manifest.json"
        )
        combined, names = combine_disk_pseudobulk_partitions(
            [focal, *nonfocal], [focal_names, *nonfocal_names]
        )
        level_populations = [
            population
            for population in populations
            if (":" not in population) == (level == "family")
        ]
        for population in level_populations:
            axis = chunked_strongest_competitor_selectivity(
                combined,
                names,
                focal_population=population,
                comparator_populations=comparator_populations,
                gene_chunk_size=100,
                minimum_common_donors=3,
                minimum_cells_per_donor_population=10,
                minimum_focal_detection=0.05,
            )
            output[population] = axis
            print(f"  mouse brainwide additions {population}")
    return output


def mouse_brainwide_new_axes(
    target_registry: pd.DataFrame,
    populations: list[str],
) -> dict[str, pd.DataFrame]:
    output = {}
    missing = []
    for population in populations:
        path = MOUSE_AXIS_CACHE / f"{population.replace(':', '__')}.parquet"
        if path.is_file():
            output[population] = pd.read_parquet(path)
        else:
            missing.append(population)
    if missing:
        raise FileNotFoundError(f"mouse brain-wide axis caches are missing: {missing}")
    return output


def mouse_brainwide_source_paths(template: pd.Series | None = None) -> list[Path]:
    if template is None:
        plan = pd.read_parquet(RELEASE / "genomewide_selectivity_job_plan.parquet")
        template = plan[
            plan["species"].astype(str).eq("mouse")
            & plan["axis_id"].astype(str).eq("brainwide")
        ].iloc[0]
    source = [Path(value) for value in json.loads(template["checkpoint_manifests_json"])]
    focal_root = DATA / "derived/m2h_fda_scdrs_atlas_v2/mouse_whb_pseudobulk/focal"
    return [
        focal_root / "mouse_whb_focal_family.manifest.json",
        focal_root / "mouse_whb_focal_leaf.manifest.json",
        *source[1:],
    ]


def cache_mouse_partition(index: int) -> None:
    paths = mouse_brainwide_source_paths()
    if index < 0 or index >= len(paths):
        raise ValueError(f"mouse cache index must be 0 to {len(paths) - 1}")
    _edges, registries, _labels = build_universe()
    registry = registries["mouse"]
    wanted = set(
        registry.loc[
            registry["human_gene"].astype(str).isin(NEW_TARGETS)
            & registry["gene_mapping_status"].astype(str).eq(
                "reciprocal_one_to_one_mouse_ortholog"
            ),
            "species_gene",
        ].astype(str)
    )
    result, names = load_checkpoint(paths[index])
    subset, selected_names = subset_checkpoint(result, names, wanted)
    MOUSE_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = save_sparse_pseudobulk_checkpoint(
        subset,
        selected_names,
        MOUSE_CACHE,
        checkpoint_id=f"mouse_bw_targets_{index}",
        overwrite=True,
    )
    print(json.dumps({"index": index, "source": str(paths[index]), "cache": str(manifest)}))


def cache_mouse_axis(index: int) -> None:
    _edges, registries, _labels = build_universe()
    populations = active_populations()["mouse"]
    if index < 0 or index >= len(populations):
        raise ValueError(f"mouse axis index must be 0 to {len(populations) - 1}")
    population = populations[index]
    axis = compute_mouse_brainwide_new_axes(registries["mouse"], [population])[population]
    path = MOUSE_AXIS_CACHE / f"{population.replace(':', '__')}.parquet"
    atomic_parquet(axis, path)
    print(json.dumps({"index": index, "population": population, "path": str(path)}))


def mouse_brainwide_axes(
    target_registry: pd.DataFrame,
    populations: list[str],
    checkpoints: dict,
) -> dict[str, pd.DataFrame]:
    metadata = pd.read_parquet(MOUSE_REGION / "mouse.region_metadata.parquet")
    features = pd.read_parquet(MOUSE_REGION / "mouse.region_features.parquet")
    region = np.load(MOUSE_REGION / "mouse.region_counts.npz")
    region_counts = np.asarray(region["counts"], dtype=float)
    region_detected = np.asarray(region["detected"], dtype=float)
    region_genes = tuple(features["feature_name"].astype(str))
    target_genes = set(
        target_registry["species_gene"].dropna().astype(str).str.strip()
    )
    common = sorted(target_genes & set(region_genes))
    region_index = {gene: index for index, gene in enumerate(region_genes)}
    region_columns = np.asarray([region_index[gene] for gene in common])
    anatomies = sorted(metadata["anatomy"].astype(str).unique())
    reference = {}
    reference_detection = {}
    for anatomy in anatomies:
        mask = metadata["anatomy"].astype(str).eq(anatomy).to_numpy()
        libraries = metadata.loc[mask, "library_size"].to_numpy(dtype=float)
        cells = metadata.loc[mask, "n_cells"].to_numpy(dtype=float)
        reference[anatomy] = np.log2(
            (region_counts[np.ix_(mask, region_columns)] + 0.5)
            / libraries[:, None]
            * 1e6
        ).mean(axis=0)
        reference_detection[anatomy] = (
            region_detected[np.ix_(mask, region_columns)] / cells[:, None]
        ).mean(axis=0)

    levels = population_levels(checkpoints)
    checkpoint_by_level = {}
    for level in ("family", "leaf"):
        result, names = checkpoints[("mouse", level)]
        name_index = {str(name): index for index, name in enumerate(names)}
        measured = [gene for gene in common if gene in name_index]
        if measured != common:
            missing = sorted(set(common) - set(measured))
            raise ValueError(f"mouse brain-wide DA checkpoint lacks targets: {missing}")
        checkpoint_by_level[level] = (
            result,
            np.asarray([name_index[gene] for gene in common]),
        )

    output = {}
    for population in populations:
        if population == "Anxa1":
            level = "leaf"
            members = set(ANXA1_LEAVES)
        else:
            level = levels[("mouse", population)]
            members = {population}
        result, columns = checkpoint_by_level[level]
        da_metadata = result.metadata.reset_index(drop=True)
        donors = sorted(da_metadata["donor_id"].astype(str).unique())
        focal = []
        focal_detection = []
        focal_cells = []
        used_donors = []
        for donor in donors:
            mask = (
                da_metadata["donor_id"].astype(str).eq(donor)
                & da_metadata["population_id"].astype(str).isin(members)
            ).to_numpy()
            if not mask.any():
                continue
            library = float(da_metadata.loc[mask, "library_size"].sum())
            cells = int(da_metadata.loc[mask, "n_cells"].sum())
            if library <= 0 or cells < 10:
                continue
            counts = np.asarray(result.counts[mask][:, columns].sum(axis=0)).reshape(-1)
            detected = np.asarray(
                result.detected_cells[mask][:, columns].sum(axis=0)
            ).reshape(-1)
            focal.append(np.log2((counts + 0.5) / library * 1e6))
            focal_detection.append(detected / cells)
            focal_cells.append(cells)
            used_donors.append(donor)
        if len(focal) < 3:
            raise ValueError(f"mouse brain-wide {population} has insufficient DA donors")
        focal_matrix = np.vstack(focal)
        focal_detection_matrix = np.vstack(focal_detection)
        effect_matrix = np.vstack(
            [focal_matrix.mean(axis=0) - reference[value] for value in anatomies]
        )
        strongest = np.argmin(effect_matrix, axis=0)
        positions = np.arange(len(common))
        effect = effect_matrix[strongest, positions]
        standard_error = focal_matrix.std(axis=0, ddof=1) / np.sqrt(len(focal))
        focal_det = focal_detection_matrix.mean(axis=0)
        comparator_det = np.vstack(
            [reference_detection[value] for value in anatomies]
        )[strongest, positions]
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic = np.where(standard_error > 0, effect / standard_error, np.nan)
        p_value = stats.t.sf(statistic, df=len(focal) - 1)
        output[population] = pd.DataFrame(
            {
                "gene_id": common,
                "effect": effect,
                "ci_low": effect
                - stats.t.ppf(0.975, df=len(focal) - 1) * standard_error,
                "ci_high": effect
                + stats.t.ppf(0.975, df=len(focal) - 1) * standard_error,
                "detection_difference": focal_det - comparator_det,
                "competitor_id": [anatomies[index] for index in strongest],
                "n_common_donors": len(focal),
                "focal_n_cells": sum(focal_cells),
                "focal_detection": focal_det,
                "p_value": p_value,
                "q_value": bh(p_value),
                "competitor_loo_selection_fraction": np.nan,
                "common_donor_ids": " | ".join(used_donors),
                "n_eligible_comparators": len(anatomies),
                "status": "estimated",
            }
        )
        print(f"  mouse brainwide {population}")
    return output


def within_axes(
    species: str,
    target_registry: pd.DataFrame,
    populations: list[str],
    checkpoints: dict,
) -> dict[str, pd.DataFrame]:
    """Reuse verified pooled-rest rows and compute only the five new targets."""

    existing = pd.read_parquet(RELEASE / "target_population_selectivity.rest_da.parquet")
    existing = existing[
        existing["species"].astype(str).eq(species)
        & existing["trait_id"].astype(str).eq("pd")
        & existing["population_id"].astype(str).isin(populations)
    ]
    rename = {
        "species_gene": "gene_id",
        "within_da_effect": "effect",
        "within_da_ci_low": "ci_low",
        "within_da_ci_high": "ci_high",
        "within_da_detection_difference": "detection_difference",
        "within_da_competitor_id": "competitor_id",
        "within_da_n_common_donors": "n_common_donors",
        "within_da_focal_n_cells": "focal_n_cells",
        "within_da_focal_detection": "focal_detection",
        "within_da_p_value": "p_value",
        "within_da_q_value": "q_value",
        "within_da_competitor_loo_selection_fraction": "competitor_loo_selection_fraction",
        "within_da_common_donor_ids": "common_donor_ids",
        "within_da_n_eligible_comparators": "n_eligible_comparators",
        "within_da_status": "status",
    }
    columns = list(rename.values())
    eligible = target_registry[
        target_registry["human_gene"].astype(str).isin(NEW_TARGETS)
        & target_registry["gene_mapping_status"].astype(str).eq(
            "human_direct" if species == "human" else "reciprocal_one_to_one_mouse_ortholog"
        )
    ]
    wanted = set(eligible["species_gene"].astype(str))
    levels = population_levels(checkpoints)
    subset_by_level = {}
    for level in ("family", "leaf"):
        result, names = checkpoints[(species, level)]
        subset_by_level[level] = subset_checkpoint(result, names, wanted)
    output = {}
    for population in populations:
        if population == "Anxa1":
            level = "leaf"
            members = set(ANXA1_LEAVES)
        else:
            level = levels[(species, population)]
            members = {population}
        result, names = subset_by_level[level]
        pooled = build_pooled_rest_pseudobulk(result, members, population)
        additions = chunked_strongest_competitor_selectivity(
            pooled,
            names,
            focal_population=population,
            comparator_populations=["rest_DA"],
            gene_chunk_size=100,
            minimum_common_donors=3,
            minimum_cells_per_donor_population=10,
            minimum_focal_detection=0.05,
        )
        shared = existing[
            existing["population_id"].astype(str).eq(population)
        ][list(rename)].rename(columns=rename)
        shared = shared[shared["gene_id"].fillna("").astype(str).str.strip().ne("")]
        covered = set(shared["gene_id"].astype(str))
        remaining = additions[~additions["gene_id"].astype(str).isin(covered)]
        axis = pd.concat([shared, remaining[columns]], ignore_index=True)
        if axis["gene_id"].astype(str).duplicated().any():
            raise ValueError(f"within-DA target duplication: {species} {population}")
        output[population] = axis
        print(f"  {species} within DA {population}")
    return output


def macaque_axes(
    target_registry: pd.DataFrame,
    populations: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    del target_registry
    return load_macaque_axes(populations)


def cache_macaque_axes() -> None:
    raise RuntimeError(
        "use scripts/build_hmba_rhesus_fda.py axes for strict rhesus caches"
    )


def cache_macaque_axis(index: int) -> None:
    del index
    raise RuntimeError(
        "use scripts/build_hmba_rhesus_fda.py axes for strict rhesus caches"
    )


def load_macaque_axes(populations: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    within = {}
    brainwide = {}
    for population in populations:
        prefix = population.replace(":", "__")
        within_path = MACAQUE_AXIS_CACHE / f"{prefix}.within_da.parquet"
        brainwide_path = MACAQUE_AXIS_CACHE / f"{prefix}.brainwide.parquet"
        if not within_path.is_file() or not brainwide_path.is_file():
            raise FileNotFoundError(f"macaque axis cache is missing: {population}")
        within[population] = pd.read_parquet(within_path)
        brainwide[population] = pd.read_parquet(brainwide_path)
    return within, brainwide


def assemble_species(
    species: str,
    registry: pd.DataFrame,
    populations: list[str],
    within: dict[str, pd.DataFrame],
    brainwide: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    registry = registry.drop(columns=["release_id"], errors="ignore")
    population_parts = []
    for population in populations:
        base = assemble_target_population_selectivity(
            within[population],
            brainwide[population],
            registry,
            release_id="development",
            species=species,
            trait_id="pd",
            population_id=population,
        )
        population_parts.extend(
            [base.assign(trait_id=trait) for trait in TRAITS]
        )
    result = pd.concat(population_parts, ignore_index=True)
    validate_table("target_population_selectivity", result)
    return result


def build() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    edges, registries, labels = build_universe()
    populations = active_populations()
    checkpoints = load_checkpoints()
    within_human = within_axes(
        "human", registries["human"], populations["human"], checkpoints
    )
    within_mouse = within_axes(
        "mouse", registries["mouse"], populations["mouse"], checkpoints
    )
    brain_human = human_brainwide_axes(registries["human"], populations["human"])
    brain_mouse = mouse_brainwide_axes(
        registries["mouse"], populations["mouse"], checkpoints
    )
    within_macaque, brain_macaque = load_macaque_axes(populations["macaque"])
    result = pd.concat(
        [
            assemble_species(
                "human", registries["human"], populations["human"], within_human, brain_human
            ),
            assemble_species(
                "mouse", registries["mouse"], populations["mouse"], within_mouse, brain_mouse
            ),
            assemble_species(
                "macaque",
                registries["macaque"],
                populations["macaque"],
                within_macaque,
                brain_macaque,
            ),
        ],
        ignore_index=True,
    )
    validate_table("target_population_selectivity", result)
    expected_rows = sum(map(len, populations.values())) * len(TRAITS) * 533
    if len(result) != expected_rows:
        raise ValueError(f"selectivity grid expected {expected_rows}, observed {len(result)}")
    if result["population_id"].astype(str).str.contains("/", regex=False).any():
        raise ValueError("branch population leaked into rebuilt selectivity")
    if result.groupby(["species", "trait_id", "population_id"])["target_id"].nunique().ne(533).any():
        raise ValueError("one or more analysis slices do not contain 533 targets")
    atomic_parquet(result, OUTPUT / "target_population_selectivity.parquet")
    atomic_parquet(edges, OUTPUT / "drug_target_edges.parquet")
    atomic_parquet(labels, OUTPUT / "direct_label_target_evidence.parquet")
    for species, registry in registries.items():
        atomic_parquet(registry, OUTPUT / f"{species}_target_gene_registry.parquet")
    payload = {
        "contract_version": "strict_direct_single_gene_533_v1",
        "release_id": "development",
        "scope": "strict direct single-gene FDA target selectivity",
        "parameters": {
            "targets": 533,
            "family_group_components_included": False,
            "secondary_pharmacology_included": False,
            "branch_populations_included": False,
            "macaque_species_scientific_name": "Macaca mulatta",
            "macaque_focal_dataset": "Allen HMBA rhesus-only basal-ganglia raw RNA",
            "macaque_brainwide_comparator": (
                "Allen HMBA focal DA donors versus donor-balanced Chiou non-DA "
                "regional reference after exact exclusion of all source DA cells"
            ),
            "brainwide_axes_are_donor_matched": False,
            "within_da_comparator": "pooled rest of DA",
            "human_brainwide_comparator": (
                "integrated control DA donors versus pooled raw-count non-DA anatomy"
            ),
            "mouse_brainwide_comparator": (
                "all-condition DA donors versus mean donor-level non-DA anatomy"
            ),
            "mouse_brainwide_targets_use_one_estimator": True,
            "minimum_focal_detection": 0.05,
            "detection_difference_is_gate": False,
            "confidence_interval_is_gate": False,
            "minimum_common_donors_upstream": 3,
            "traits": list(TRAITS),
        },
        "counts": {
            "rows": len(result),
            "targets": result["target_id"].nunique(),
            "populations": {species: len(values) for species, values in populations.items()},
            "traits": result["trait_id"].nunique(),
            "direct_label_edges": len(labels),
            "new_targets": sorted(NEW_TARGETS),
        },
        "outputs": {},
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(SRC / "fda_atlas_v2/direct_target_universe.py"): sha256(
                SRC / "fda_atlas_v2/direct_target_universe.py"
            ),
            str(SRC / "fda_atlas_v2/target_selectivity.py"): sha256(
                SRC / "fda_atlas_v2/target_selectivity.py"
            ),
            str(SRC / "fda_atlas_v2/contracts.py"): sha256(
                SRC / "fda_atlas_v2/contracts.py"
            ),
        },
    }
    for path in sorted(OUTPUT.glob("*.parquet")):
        payload["outputs"][path.name] = {
            "path": str(path.resolve()),
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "sha256": sha256(path),
        }
    atomic_json(payload, OUTPUT / "target_population_selectivity_manifest.json")
    return result, registries, edges, labels


def apply() -> None:
    required = [
        "target_population_selectivity.parquet",
        "target_population_selectivity_manifest.json",
        "drug_target_edges.parquet",
        "direct_label_target_evidence.parquet",
        "human_target_gene_registry.parquet",
        "mouse_target_gene_registry.parquet",
        "macaque_target_gene_registry.parquet",
    ]
    for name in required:
        source = OUTPUT / name
        if not source.is_file():
            raise FileNotFoundError(source)
    selectivity = pd.read_parquet(OUTPUT / required[0])
    validate_table("target_population_selectivity", selectivity)
    for name in required:
        source = OUTPUT / name
        destination = RELEASE / name
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.write_bytes(source.read_bytes())
        os.replace(partial, destination)
    manifest = json.loads((RELEASE / "target_population_selectivity_manifest.json").read_text())
    manifest["contract_version"] = "strict_direct_single_gene_533_v1"
    manifest["outputs"] = {
        name: {
            "path": str((RELEASE / name).resolve()),
            "rows": pq.ParquetFile(RELEASE / name).metadata.num_rows,
            "sha256": sha256(RELEASE / name),
        }
        for name in required
        if name.endswith(".parquet")
    }
    manifest["pipeline_files"] = {
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        str(SRC / "fda_atlas_v2/direct_target_universe.py"): sha256(
            SRC / "fda_atlas_v2/direct_target_universe.py"
        ),
        str(SRC / "fda_atlas_v2/target_selectivity.py"): sha256(
            SRC / "fda_atlas_v2/target_selectivity.py"
        ),
        str(SRC / "fda_atlas_v2/contracts.py"): sha256(
            SRC / "fda_atlas_v2/contracts.py"
        ),
    }
    atomic_json(manifest, RELEASE / "target_population_selectivity_manifest.json")
    print(json.dumps({"applied": required, "rows": len(selectivity)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--cache-mouse-partition", type=int)
    parser.add_argument("--cache-mouse-axis", type=int)
    parser.add_argument("--cache-macaque-axes", action="store_true")
    parser.add_argument("--cache-macaque-axis", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.cache_mouse_partition is not None:
        cache_mouse_partition(arguments.cache_mouse_partition)
    elif arguments.cache_mouse_axis is not None:
        cache_mouse_axis(arguments.cache_mouse_axis)
    elif arguments.cache_macaque_axes:
        cache_macaque_axes()
    elif arguments.cache_macaque_axis is not None:
        cache_macaque_axis(arguments.cache_macaque_axis)
    elif arguments.apply:
        apply()
    else:
        build()
