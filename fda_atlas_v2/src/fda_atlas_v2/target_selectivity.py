"""Post-genome-wide FDA target joins for the two selectivity axes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .contracts import TRAITS, validate_table
from .ranking import compute_dual_selectivity_rank


ORTHOLOG_GENE_COLUMNS = {
    "mouse": "Mouse Gene",
    "macaque": "Macaque Gene",
}


def primary_gene_mapping_status(species: str) -> str:
    """Return the only mapping status eligible for quantitative ranking."""

    if species == "human":
        return "human_direct"
    if species not in ORTHOLOG_GENE_COLUMNS:
        raise ValueError(f"unsupported atlas species: {species}")
    return f"reciprocal_one_to_one_{species}_ortholog"


def combine_target_evidence_universe(
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> pd.DataFrame:
    required = {"target_id", "human_gene"}
    frames = []
    for label, source in (("moiety", moiety_edges), ("substance", substance_edges)):
        if missing := sorted(required - set(source.columns)):
            raise ValueError(f"{label} target evidence is missing columns: {missing}")
        frame = source[["target_id", "human_gene"]].drop_duplicates().copy()
        frame["evidence_grain"] = label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.groupby("target_id")["human_gene"].nunique()
    if (conflicts > 1).any():
        raise ValueError("one target ID maps to multiple human gene symbols")
    grains = (
        combined.groupby(["target_id", "human_gene"], sort=True)["evidence_grain"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("target_evidence_grains")
        .reset_index()
    )
    return grains.sort_values("target_id", kind="stable").reset_index(drop=True)


def validate_axis_identity(
    frame: pd.DataFrame,
    *,
    release_id: str,
    species: str,
    population_id: str,
    axis_id: str,
    dataset_id: str,
) -> None:
    """Require every genome-wide row to carry the expected immutable axis identity."""

    expected = {
        "release_id": release_id,
        "species": species,
        "focal_population_id": population_id,
        "axis_id": axis_id,
        "dataset_id": dataset_id,
    }
    if missing := sorted(set(expected) - set(frame.columns)):
        raise ValueError(f"axis table is missing identity columns: {missing}")
    for column, value in expected.items():
        observed = set(frame[column].dropna().astype(str))
        if observed != {str(value)} or frame[column].isna().any():
            raise ValueError(
                f"axis identity mismatch for {column}: expected {value}, observed {sorted(observed)}"
            )


def validate_selected_trait_populations(
    candidates: pd.DataFrame, *, species: str
) -> pd.DataFrame:
    """Require one supported selected level and one leader per frozen trait."""

    required = {
        "trait_id", "hierarchy_level", "population_id",
        "selected_resolution", "bridge_available", "default_leader",
    }
    if missing := sorted(required - set(candidates.columns)):
        raise ValueError(f"{species} trait candidates are missing columns: {missing}")
    for column in ("selected_resolution", "bridge_available"):
        if not pd.api.types.is_bool_dtype(candidates[column]) or candidates[column].isna().any():
            raise ValueError(f"{species} trait candidate {column} must be nonnull boolean")
    selected_rows = candidates[
        candidates["selected_resolution"].astype(bool)
        & candidates["bridge_available"].astype(bool)
    ][["trait_id", "hierarchy_level", "population_id", "default_leader"]].copy()
    if selected_rows.duplicated().any():
        raise ValueError(f"{species} selected trait-population rows are duplicated")
    if not pd.api.types.is_bool_dtype(selected_rows["default_leader"]):
        raise ValueError(f"{species} default_leader must be boolean")
    selected = selected_rows.copy()
    observed_traits = set(selected["trait_id"].astype(str))
    if observed_traits != set(TRAITS):
        raise ValueError(
            f"{species} selected trait set differs: {sorted(observed_traits)}"
        )
    if (selected.groupby("trait_id")["hierarchy_level"].nunique() != 1).any():
        raise ValueError(f"{species} must select exactly one hierarchy level per trait")
    if (selected.groupby("trait_id")["default_leader"].sum() != 1).any():
        raise ValueError(f"{species} must select exactly one default leader per trait")
    if selected["population_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{species} selected population ID is blank")
    if not set(selected["hierarchy_level"].astype(str)).issubset(
        {"family", "branch", "leaf"}
    ):
        raise ValueError(f"{species} selected hierarchy level is invalid")
    return selected.sort_values(
        ["trait_id", "default_leader", "population_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_target_gene_registry(
    drug_target_edges: pd.DataFrame,
    *,
    species: str,
    orthologs: pd.DataFrame | None = None,
    species_gene_column: str | None = None,
) -> pd.DataFrame:
    """Resolve each HGNC target to one quantitative species gene or a reason."""

    required = {"target_id", "human_gene"}
    if missing := sorted(required - set(drug_target_edges.columns)):
        raise ValueError(f"drug target edges are missing columns: {missing}")
    targets = drug_target_edges[["target_id", "human_gene"]].drop_duplicates().copy()
    if targets["target_id"].duplicated().any() or targets["human_gene"].astype(str).eq("").any():
        raise ValueError("drug target IDs do not map one-to-one to nonblank human genes")
    if species == "human":
        targets["species_gene"] = targets["human_gene"].astype(str)
        targets["species_gene_identifier_type"] = "gene_symbol"
        targets["ortholog_count"] = 1
        targets["orthologs_json"] = targets["species_gene"].map(
            lambda value: json.dumps([value])
        )
        targets["gene_mapping_status"] = "human_direct"
        return targets.sort_values("target_id", kind="stable").reset_index(drop=True)
    if species not in ORTHOLOG_GENE_COLUMNS:
        raise ValueError(f"target gene registry is not implemented for species: {species}")
    if orthologs is None:
        raise ValueError(f"{species} target registry requires a pinned ortholog table")
    species_column = species_gene_column or ORTHOLOG_GENE_COLUMNS[species]
    required_ortholog = {species_column, "Human Gene"}
    if missing := sorted(required_ortholog - set(orthologs.columns)):
        raise ValueError(f"{species} ortholog table is missing columns: {missing}")
    evidence_columns = {"Orthology Type", "Orthology Confidence"}
    present_evidence = evidence_columns & set(orthologs.columns)
    if present_evidence and present_evidence != evidence_columns:
        raise ValueError(
            f"{species} ortholog table must provide both Orthology Type and "
            "Orthology Confidence when either is present"
        )
    mapping_columns = list(required_ortholog) + sorted(present_evidence)
    mapping = orthologs[mapping_columns].fillna("").astype(str)
    mapping = mapping[
        mapping[species_column].str.strip().ne("")
        & mapping["Human Gene"].str.strip().ne("")
    ].drop_duplicates()
    grouped = mapping.groupby("Human Gene", sort=True)[species_column].agg(
        lambda values: tuple(sorted(set(values)))
    )
    if present_evidence:
        type_grouped = mapping.groupby("Human Gene", sort=True)["Orthology Type"].agg(
            lambda values: tuple(sorted(set(values)))
        )
        confidence_grouped = mapping.groupby("Human Gene", sort=True)[
            "Orthology Confidence"
        ].agg(lambda values: tuple(sorted(set(values))))
    else:
        type_grouped = pd.Series(dtype=object)
        confidence_grouped = pd.Series(dtype=object)
    species_human_counts = mapping.groupby(species_column, sort=True)["Human Gene"].nunique()
    targets = targets.merge(
        grouped.rename("species_orthologs"),
        left_on="human_gene",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    targets["ortholog_count"] = targets["species_orthologs"].map(
        lambda value: len(value) if isinstance(value, tuple) else 0
    )
    targets["species_gene"] = targets["species_orthologs"].map(
        lambda value: value[0] if isinstance(value, tuple) and len(value) == 1 else ""
    )
    targets["species_gene_identifier_type"] = (
        "ensembl_gene_id" if species_column.endswith("Ensembl Gene ID") else "gene_symbol"
    )
    reverse_count_column = f"{species}_gene_human_ortholog_count"
    targets[reverse_count_column] = targets["species_gene"].map(
        species_human_counts
    ).fillna(0).astype(int)
    targets["orthologs_json"] = targets["species_orthologs"].map(
        lambda value: json.dumps(list(value) if isinstance(value, tuple) else [])
    )
    targets["orthology_types_json"] = targets["human_gene"].map(type_grouped).map(
        lambda value: json.dumps(list(value) if isinstance(value, tuple) else [])
    )
    targets["orthology_confidences_json"] = targets["human_gene"].map(
        confidence_grouped
    ).map(lambda value: json.dumps(list(value) if isinstance(value, tuple) else []))
    source_one_to_one = (
        targets["human_gene"].map(type_grouped).map(
            lambda value: value == ("ortholog_one2one",)
        )
        if present_evidence else pd.Series(True, index=targets.index)
    )
    source_high_confidence = (
        targets["human_gene"].map(confidence_grouped).map(
            lambda value: value == ("1",)
        )
        if present_evidence else pd.Series(True, index=targets.index)
    )
    targets["gene_mapping_status"] = np.select(
        [
            targets["ortholog_count"].eq(1)
            & targets[reverse_count_column].eq(1)
            & source_one_to_one
            & source_high_confidence,
            targets["ortholog_count"].eq(1) & ~source_one_to_one,
            targets["ortholog_count"].eq(1) & ~source_high_confidence,
            targets["ortholog_count"].eq(1),
            targets["ortholog_count"].eq(0),
        ],
        [
            primary_gene_mapping_status(species),
            f"non_one_to_one_{species}_ortholog",
            f"low_confidence_{species}_ortholog",
            f"nonreciprocal_{species}_ortholog",
            f"no_{species}_ortholog",
        ],
        default=f"multiple_{species}_orthologs",
    )
    return targets.drop(columns="species_orthologs").sort_values(
        "target_id", kind="stable"
    ).reset_index(drop=True)


def _axis_table(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {
        "gene_id",
        "effect",
        "ci_low",
        "ci_high",
        "detection_difference",
        "competitor_id",
        "n_common_donors",
        "focal_n_cells",
        "focal_detection",
        "p_value",
        "q_value",
        "competitor_loo_selection_fraction",
        "common_donor_ids",
        "n_eligible_comparators",
        "status",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{prefix} axis table is missing columns: {missing}")
    if frame["gene_id"].astype(str).duplicated().any():
        raise ValueError(f"{prefix} axis has duplicate quantitative genes")
    columns = {
        "gene_id": "species_gene",
        "effect": f"{prefix}_effect",
        "ci_low": f"{prefix}_ci_low",
        "ci_high": f"{prefix}_ci_high",
        "detection_difference": f"{prefix}_detection_difference",
        "competitor_id": f"{prefix}_competitor_id",
        "n_common_donors": f"{prefix}_n_common_donors",
        "focal_n_cells": f"{prefix}_focal_n_cells",
        "focal_detection": f"{prefix}_focal_detection",
        "p_value": f"{prefix}_p_value",
        "q_value": f"{prefix}_q_value",
        "competitor_loo_selection_fraction": (
            f"{prefix}_competitor_loo_selection_fraction"
        ),
        "common_donor_ids": f"{prefix}_common_donor_ids",
        "n_eligible_comparators": f"{prefix}_n_eligible_comparators",
        "status": f"{prefix}_status",
    }
    return frame[list(columns)].rename(columns=columns)


def assemble_target_population_selectivity(
    within_da: pd.DataFrame,
    brainwide: pd.DataFrame,
    target_registry: pd.DataFrame,
    *,
    release_id: str,
    species: str,
    trait_id: str,
    population_id: str,
) -> pd.DataFrame:
    """Join both genome-wide axes, then rank only adequately mapped FDA targets."""

    required_registry = {
        "target_id",
        "human_gene",
        "species_gene",
        "ortholog_count",
        "orthologs_json",
        "gene_mapping_status",
    }
    if missing := sorted(required_registry - set(target_registry.columns)):
        raise ValueError(f"target registry is missing columns: {missing}")
    if target_registry["target_id"].duplicated().any():
        raise ValueError("target registry has duplicate target IDs")
    result = target_registry.copy().merge(
        _axis_table(within_da, "within_da"),
        on="species_gene",
        how="left",
        validate="many_to_one",
    ).merge(
        _axis_table(brainwide, "brainwide"),
        on="species_gene",
        how="left",
        validate="many_to_one",
    )
    result.insert(0, "release_id", release_id)
    result.insert(1, "species", species)
    result.insert(2, "trait_id", trait_id)
    result.insert(3, "population_id", population_id)
    result["eligible"] = result["gene_mapping_status"].eq(
        primary_gene_mapping_status(species)
    )
    for prefix in ("within_da", "brainwide"):
        missing_status = result[f"{prefix}_status"].isna()
        result.loc[
            missing_status & result["eligible"], f"{prefix}_status"
        ] = "gene_not_measured_in_axis"
        result.loc[
            missing_status & ~result["eligible"], f"{prefix}_status"
        ] = "not_evaluated_ineligible_mapping"
    result["focal_n_donors"] = result[
        ["within_da_n_common_donors", "brainwide_n_common_donors"]
    ].min(axis=1, skipna=False)
    result["focal_n_cells"] = result[
        ["within_da_focal_n_cells", "brainwide_focal_n_cells"]
    ].min(axis=1, skipna=False)
    result["expression_support"] = result[
        ["within_da_focal_detection", "brainwide_focal_detection"]
    ].min(axis=1, skipna=False)
    result = compute_dual_selectivity_rank(result)
    validate_table("target_population_selectivity", result)
    return result
