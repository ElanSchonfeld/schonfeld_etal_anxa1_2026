"""Assemble same-target cross-species evidence without comparing absolute expression."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd

from .contracts import validate_table


SPECIES_ORDER = {"human": 0, "mouse": 1, "macaque": 2}


def effect_availability(source_effect: object, target_effect: object) -> str:
    source_available = pd.notna(source_effect) and np.isfinite(float(source_effect))
    target_available = pd.notna(target_effect) and np.isfinite(float(target_effect))
    if source_available and target_available:
        return "both_available"
    if not source_available and not target_available:
        return "both_unavailable"
    return "source_unavailable" if not source_available else "target_unavailable"


def effect_direction(source_effect: object, target_effect: object, availability: str) -> str:
    if availability != "both_available":
        return "unavailable"
    source = float(source_effect)
    target = float(target_effect)
    if not np.isfinite(source) or not np.isfinite(target):
        raise ValueError("ranked cross-species effects must be finite")
    if source == 0 or target == 0:
        return "includes_zero"
    if np.sign(source) != np.sign(target):
        return "opposite_direction"
    return "same_positive" if source > 0 else "same_negative"


def assemble_cross_species_effects(
    selectivity: pd.DataFrame,
    *,
    release_id: str,
    shared_populations_only: bool = False,
) -> pd.DataFrame:
    """Pair each species within a trait and target without hiding population mismatch."""

    required = {
        "species", "trait_id", "population_id", "target_id", "rank_status",
        "gene_mapping_status",
        "within_da_effect", "within_da_ci_low", "within_da_ci_high",
        "within_da_percentile", "within_da_q_value", "within_da_competitor_id",
        "brainwide_effect", "brainwide_ci_low", "brainwide_ci_high",
        "brainwide_percentile", "brainwide_q_value", "brainwide_competitor_id",
    }
    if missing := sorted(required - set(selectivity.columns)):
        raise ValueError(f"target selectivity is missing cross-species fields: {missing}")
    key = ["species", "trait_id", "target_id", "population_id"]
    if selectivity.duplicated(key).any():
        raise ValueError("target selectivity has duplicate cross-species input rows")
    unknown_species = sorted(set(selectivity["species"].astype(str)) - set(SPECIES_ORDER))
    if unknown_species:
        raise ValueError(f"unknown cross-species input species: {unknown_species}")

    rows = []
    group_key = ["trait_id", "target_id"]
    for (trait_id, target_id), group in selectivity.groupby(
        group_key, sort=True, dropna=False
    ):
        records_by_species = {
            str(species): species_group.sort_values("population_id", kind="stable").to_dict("records")
            for species, species_group in group.groupby("species", sort=False)
        }
        ordered_species = sorted(records_by_species, key=SPECIES_ORDER.__getitem__)
        for source_species, target_species in combinations(ordered_species, 2):
            for source, target in product(
                records_by_species[source_species], records_by_species[target_species]
            ):
                source_population = str(source["population_id"])
                target_population = str(target["population_id"])
                if not source_population or not target_population:
                    raise ValueError("cross-species input has a blank population ID")
                shared_population = source_population == target_population
                if shared_populations_only and not shared_population:
                    continue
                mapping_status = (
                    "same_human_target_and_shared_hmoe_population"
                    if shared_population
                    else "same_human_target_but_different_trait_relevant_hmoe_populations"
                )
                for contrast, prefix in (("within_da", "within_da"), ("brainwide", "brainwide")):
                    availability = effect_availability(
                        source[f"{prefix}_effect"], target[f"{prefix}_effect"]
                    )
                    rows.append(
                        {
                        "release_id": release_id,
                        "trait_id": str(trait_id),
                        "target_id": str(target_id),
                        "source_species": str(source["species"]),
                        "target_species": str(target["species"]),
                        "source_population_id": source_population,
                        "target_population_id": target_population,
                        "contrast": contrast,
                        "source_effect": source[f"{prefix}_effect"],
                        "source_ci_low": source[f"{prefix}_ci_low"],
                        "source_ci_high": source[f"{prefix}_ci_high"],
                        "source_percentile": source[f"{prefix}_percentile"],
                        "source_q_value": source[f"{prefix}_q_value"],
                        "source_competitor_id": source[f"{prefix}_competitor_id"],
                        "source_rank_status": str(source["rank_status"]),
                        "source_gene_mapping_status": str(
                            source["gene_mapping_status"]
                        ),
                        "target_effect": target[f"{prefix}_effect"],
                        "target_ci_low": target[f"{prefix}_ci_low"],
                        "target_ci_high": target[f"{prefix}_ci_high"],
                        "target_percentile": target[f"{prefix}_percentile"],
                        "target_q_value": target[f"{prefix}_q_value"],
                        "target_competitor_id": target[f"{prefix}_competitor_id"],
                        "target_rank_status": str(target["rank_status"]),
                        "target_gene_mapping_status": str(
                            target["gene_mapping_status"]
                        ),
                        "availability_status": availability,
                        "direction_status": (
                            "unavailable"
                            if availability != "both_available"
                            else (
                                effect_direction(
                                    source[f"{prefix}_effect"],
                                    target[f"{prefix}_effect"],
                                    availability,
                                )
                                if shared_population
                                else "different_population_not_comparable"
                            )
                        ),
                        "mapping_status": mapping_status,
                        "magnitude_comparison_status": (
                            "within_species_effects_only_no_cross_species_subtraction"
                        ),
                        "affects_dual_selectivity_rank": False,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no same-trait, same-target cross-species pairs are available")
    result = result.sort_values(
        ["trait_id", "target_id", "source_population_id", "source_species",
         "target_species", "contrast"],
        kind="stable",
    ).reset_index(drop=True)
    validate_table("cross_species_effects", result)
    return result
