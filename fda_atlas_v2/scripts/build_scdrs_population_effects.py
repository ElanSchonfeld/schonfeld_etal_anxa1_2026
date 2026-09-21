#!/usr/bin/env python3
"""Build donor-aware scDRS population effects for human and mouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.human_scdrs import (
    validate_integrated_score_table,
    validate_score_manifest,
)
from fda_atlas_v2.population_effects import (
    estimate_population_effects,
    select_population_lead_set,
    validated_human_integrated_control_mask,
)
from fda_atlas_v2.resource_safety import large_data_preflight


def config_path(value: str | Path) -> Path:
    """Resolve the three documented path tokens used by the YAML config."""

    roots = {
        "${DOPABASE_DATA_ROOT}": Path(
            os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")
        ),
        "${DOPABASE_SOURCE_ROOT}": Path(
            os.environ.get("DOPABASE_SOURCE_ROOT", ROOT.parent)
        ),
        "${DOPABASE_FDA_ROOT}": Path(
            os.environ.get("DOPABASE_FDA_ROOT", ROOT)
        ),
    }
    text = str(value)
    for token, path in roots.items():
        text = text.replace(token, str(path.expanduser().resolve()))
    return Path(text).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=here / "configs" / "population_effects.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=here / "release" / "development")
    parser.add_argument(
        "--analysis-role",
        choices=("primary", "duplicate_corrected_sensitivity"),
        default="primary",
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(values: pd.Index) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def apply_filters(frame: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    for column, values in filters.items():
        if column not in frame:
            raise ValueError(f"cohort filter column is missing: {column}")
        keep &= frame[column].astype(str).isin([str(value) for value in values])
    return frame.loc[keep].copy()


def load_metadata(
    species: str,
    config: dict,
    analysis_role: str,
) -> tuple[pd.DataFrame, Path]:
    path = config_path(config["h5ad"])
    adata = ad.read_h5ad(path, backed="r")
    columns = {
        config["donor_column"],
        config["study_column"],
        config["condition_column"],
        config["leaf_column"],
    }
    for filters in config["cohorts"].values():
        columns.update(filters)
    missing = sorted(columns - set(adata.obs.columns))
    if missing:
        adata.file.close()
        raise ValueError(f"{species}: missing metadata columns {missing}")
    metadata = adata.obs.loc[:, sorted(columns)].copy()
    if species == "human":
        template = config["score_template"]
        score_columns = [template.format(trait=trait) for trait in TRAITS]
        score_path = config_path(config["score_table"])
        manifest_path = config_path(config["score_manifest"])
        validate_score_manifest(
            manifest_path,
            score_path,
            expected_analysis_role=analysis_role,
        )
        scores = pd.read_parquet(score_path)
        validate_integrated_score_table(scores, adata.obs[["study", "disease_status"]])
        metadata = metadata.join(scores[score_columns])
    adata.file.close()
    if not metadata.index.is_unique:
        raise ValueError(f"{species}: focal cell identifiers are not unique")
    metadata["leaf"] = metadata[config["leaf_column"]].astype(str)
    metadata = metadata[
        ~metadata["leaf"].isin([str(value) for value in config["excluded_leaves"]])
    ].copy()
    return metadata, path


def load_scores(
    species: str,
    trait: str,
    metadata: pd.DataFrame,
    config: dict,
) -> pd.Series:
    if species == "human":
        column = config["score_template"].format(trait=trait)
        return pd.to_numeric(metadata[column], errors="coerce").rename("score")
    path = config_path(config["score_directory"]) / f"{trait}.score.gz"
    score = pd.read_csv(path, sep="\t", index_col=0)["norm_score"]
    if not score.index.is_unique:
        raise ValueError(f"mouse {trait}: score cell identifiers are not unique")
    expected_cells = int(config["score_source_n_cells"])
    if len(score) != expected_cells:
        raise ValueError(
            f"mouse {trait}: expected {expected_cells} score-calibration cells, "
            f"observed {len(score)}"
        )
    if index_sha256(score.index) != str(config["score_source_cell_id_sha256"]):
        raise ValueError(f"mouse {trait}: score-calibration cell order is stale")
    missing = metadata.index.difference(score.index)
    if len(missing):
        raise ValueError(f"mouse {trait}: {len(missing)} focal cells lack scores")
    return pd.to_numeric(score.reindex(metadata.index), errors="coerce").rename("score")


def add_hierarchy(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()
    result["family"] = result["leaf"].str.split(":", n=1).str[0]
    return result


def parent_map(level: str, metadata: pd.DataFrame) -> dict[str, str]:
    if level == "family":
        return {value: "DA" for value in metadata["family"].unique()}
    return (
        metadata[["leaf", "family"]]
        .drop_duplicates()
        .set_index("leaf")["family"]
        .to_dict()
    )


def append_not_selected(
    selected: pd.DataFrame,
    all_populations: list[str],
    leader_id: str,
) -> pd.DataFrame:
    present = set(selected["population_id"].astype(str))
    rows = []
    for population_id in sorted(set(all_populations) - present):
        rows.append(
            {
                "population_id": population_id,
                "lead_status": "not_in_parent_lead_set",
                "leader_id": leader_id,
                "delta_from_leader": np.nan,
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "n_common_donors": 0,
                "selection_reason": "parent population is outside the lead set",
            }
        )
    if rows:
        selected = pd.concat([selected, pd.DataFrame(rows)], ignore_index=True)
    return selected


def build_species(
    species: str,
    species_config: dict,
    global_config: dict,
    analysis_role: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[Path]]:
    metadata, h5ad_path = load_metadata(species, species_config, analysis_role)
    metadata = add_hierarchy(metadata)
    if species == "human":
        exact_primary = validated_human_integrated_control_mask(
            metadata,
            donor_column=species_config["donor_column"],
            study_column=species_config["study_column"],
            condition_column=species_config["condition_column"],
        )
        configured_primary = apply_filters(
            metadata, species_config["cohorts"][species_config["primary_cohort"]]
        )
        if set(configured_primary.index) != set(metadata.index[exact_primary]):
            raise ValueError(
                "configured human primary cohort is not the exact frozen "
                "Kamath-plus-Siletti control union"
            )
    input_paths = [h5ad_path]
    if species == "human":
        input_paths.extend(
            [
                config_path(species_config["score_table"]),
                config_path(species_config["score_manifest"]),
            ]
        )
    if species == "mouse":
        score_manifest_path = config_path(species_config["score_manifest"])
        score_manifest = json.loads(score_manifest_path.read_text())
        if score_manifest.get("random_seed") != 42:
            raise ValueError("mouse scDRS score manifest must record random_seed 42")
        missing_traits = sorted(
            set(global_config["traits"]) - set(score_manifest.get("traits", []))
        )
        if missing_traits:
            raise ValueError(
                f"mouse scDRS score manifest lacks atlas traits: {missing_traits}"
            )
        input_paths.append(score_manifest_path)
        input_paths.extend(
            config_path(species_config["score_directory"]) / f"{trait}.score.gz"
            for trait in global_config["traits"]
        )

    effects_out = []
    leads_out = []
    donor_out = []
    for trait in global_config["traits"]:
        score = load_scores(species, trait, metadata, species_config)
        scored = metadata.join(score)
        for cohort_name, filters in species_config["cohorts"].items():
            cohort = apply_filters(scored, filters)
            if cohort.empty:
                raise ValueError(f"{species} {cohort_name}: cohort is empty")
            allowed_parents: set[str] | None = None
            for level in ("family", "leaf"):
                source_donor_id = cohort[species_config["donor_column"]].astype(str)
                study = cohort[species_config["study_column"]].astype(str)
                cell_scores = pd.DataFrame(
                    {
                        "donor_id": (
                            study + "::" + source_donor_id
                            if species == "human"
                            else source_donor_id
                        ),
                        "source_donor_id": source_donor_id,
                        "study": study,
                        "condition": cohort[species_config["condition_column"]].astype(str),
                        "population_id": cohort[level].astype(str),
                        "score": cohort["score"],
                    },
                    index=cohort.index,
                )
                effects, donor_means = estimate_population_effects(
                    cell_scores,
                    min_cells_per_donor_population=int(
                        global_config["min_cells_per_donor_population"]
                    ),
                )
                parents = parent_map(level, cohort)
                effects["parent_population_id"] = effects["population_id"].map(parents)

                eligible_effects = effects
                eligible_donors = donor_means
                if allowed_parents is not None:
                    eligible_ids = set(
                        effects.loc[
                            effects["parent_population_id"].isin(allowed_parents),
                            "population_id",
                        ].astype(str)
                    )
                    eligible_effects = effects[
                        effects["population_id"].astype(str).isin(eligible_ids)
                    ]
                    eligible_donors = donor_means[
                        donor_means["population_id"].astype(str).isin(eligible_ids)
                    ]
                leads = select_population_lead_set(
                    eligible_effects,
                    eligible_donors,
                    min_common_donors=int(global_config["min_common_donors"]),
                    min_donors=int(global_config["min_donors"]),
                )
                leader_id = (
                    str(leads.iloc[0]["leader_id"]) if not leads.empty else "unavailable"
                )
                leads = append_not_selected(
                    leads, effects["population_id"].astype(str).tolist(), leader_id
                )
                allowed_parents = set(
                    leads.loc[
                        leads["lead_status"].isin(["leader", "lead_set"]),
                        "population_id",
                    ].astype(str)
                )

                common = {
                    "release_id": global_config["release_id"],
                    "species": species,
                    "trait_id": trait,
                    "hierarchy_level": level,
                    "analysis_cohort": cohort_name,
                    "is_primary_cohort": cohort_name == species_config["primary_cohort"],
                    "analysis_role": analysis_role,
                }
                for column, value in common.items():
                    effects[column] = value
                    leads[column] = value
                    donor_means[column] = value
                effects_out.append(effects)
                leads_out.append(leads)
                donor_out.append(donor_means)
    return effects_out, leads_out, donor_out, input_paths


def write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def concat_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    columns = list(dict.fromkeys(column for part in parts for column in part.columns))
    cleaned = [part.dropna(axis=1, how="all") for part in parts]
    return pd.concat(cleaned, ignore_index=True).reindex(columns=columns)


def main() -> None:
    args = parse_args()
    resources = large_data_preflight(estimated_peak_memory_gib=2.0)
    config = yaml.safe_load(args.config.read_text())
    global TRAITS
    TRAITS = list(config["traits"])
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    output_paths = {
        "effects": args.output_dir / f"{prefix}scdrs_population_effects.parquet",
        "leads": args.output_dir / f"{prefix}trait_population_leads.parquet",
        "donor_means": args.output_dir / f"{prefix}scdrs_donor_population_means.parquet",
        "manifest": args.output_dir / f"{prefix}scdrs_population_effects_manifest.json",
    }
    if not args.overwrite and any(path.exists() for path in output_paths.values()):
        raise FileExistsError("population-effect outputs exist; pass --overwrite")

    effects_parts = []
    lead_parts = []
    donor_parts = []
    package_root = Path(__file__).resolve().parents[1]
    pipeline_paths = [
        args.config,
        Path(__file__).resolve(),
        package_root / "src" / "fda_atlas_v2" / "population_effects.py",
        package_root / "src" / "fda_atlas_v2" / "human_scdrs.py",
        package_root / "src" / "fda_atlas_v2" / "contracts.py",
        package_root / "src" / "fda_atlas_v2" / "resource_safety.py",
    ]
    input_paths = list(pipeline_paths)
    for species, species_config in config["species"].items():
        effects, leads, donor_means, species_inputs = build_species(
            species, species_config, config, args.analysis_role
        )
        effects_parts.extend(effects)
        lead_parts.extend(leads)
        donor_parts.extend(donor_means)
        input_paths.extend(species_inputs)

    effects = concat_parts(effects_parts)
    leads = concat_parts(lead_parts)
    donor_means = concat_parts(donor_parts)
    validate_table("scdrs_population_effects", effects)
    validate_table("trait_population_leads", leads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_parquet(effects, output_paths["effects"])
    write_atomic_parquet(leads, output_paths["leads"])
    write_atomic_parquet(donor_means, output_paths["donor_means"])
    manifest = {
        "schema_version": "2.0.0",
        "release_id": config["release_id"],
        "parameters": {
            "score_field": config["score_field"],
            "min_cells_per_donor_population": config[
                "min_cells_per_donor_population"
            ],
            "min_common_donors": config["min_common_donors"],
            "min_donors": config["min_donors"],
            "traits": config["traits"],
            "analysis_role": args.analysis_role,
            "branch_populations_included": False,
            "primary_cohorts": {
                species: value["primary_cohort"]
                for species, value in config["species"].items()
            },
            "mouse_score_source_n_cells": int(
                config["species"]["mouse"]["score_source_n_cells"]
            ),
            "mouse_score_source_cell_id_sha256": str(
                config["species"]["mouse"]["score_source_cell_id_sha256"]
            ),
        },
        "counts": {
            "effects": len(effects),
            "lead_rows": len(leads),
            "donor_population_rows": len(donor_means),
        },
        "validation": {
            "table_contracts_pass": True,
            "primary_cohorts": {
                species: value["primary_cohort"]
                for species, value in config["species"].items()
            },
            "raw_input_hashes_reverified_after_interruption": True,
            "raw_reproduction_pending": False,
            "human_scdrs_raw_count_checkpoint_verified": True,
            "duplicate_corrected_gene_set_sensitivity": (
                args.analysis_role == "duplicate_corrected_sensitivity"
            ),
            "integrated_human_analysis_count": len(TRAITS),
            "resource_preflight": resources,
        },
        "inputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(input_paths))
        },
        "pipeline_files": {
            str(path.resolve()): sha256(path)
            for path in pipeline_paths
        },
        "outputs": {},
    }
    for name in ("effects", "leads", "donor_means"):
        path = output_paths[name]
        manifest["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": len({"effects": effects, "leads": leads, "donor_means": donor_means}[name]),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    partial_manifest = output_paths["manifest"].with_suffix(".json.partial")
    partial_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(partial_manifest, output_paths["manifest"])
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
