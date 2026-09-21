"""Build frozen trait and DA-population registries for the public release."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import TRAITS, validate_table


TRAIT_LABELS = {
    "pd": "Parkinson disease",
    "scz": "Schizophrenia",
    "adhd": "Attention-deficit/hyperactivity disorder",
    "nicotine_dep": "Nicotine dependence",
    "bipolar": "Bipolar disorder",
    "chronic_pain": "Chronic pain",
    "mdd": "Major depressive disorder",
    "oud": "Opioid use disorder",
    "ocd": "Obsessive-compulsive disorder",
    "anxiety_any": "Anxiety",
    "fibromyalgia": "Fibromyalgia",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_scdrs_gene_set(path: Path, expected_trait: str) -> list[tuple[str, float]]:
    """Parse one weighted scDRS gene-set row without losing gene identifiers."""

    lines = path.read_text().splitlines()
    if len(lines) != 2 or lines[0].split("\t") != ["TRAIT", "GENESET"]:
        raise ValueError(f"{path} is not a one-trait scDRS gene-set file")
    trait, encoded = lines[1].split("\t", 1)
    if trait != expected_trait:
        raise ValueError(f"{path} contains trait {trait}, expected {expected_trait}")
    records = []
    for token in encoded.split(","):
        gene, weight = token.rsplit(":", 1)
        gene = gene.strip()
        numeric = float(weight)
        if not gene or not np.isfinite(numeric):
            raise ValueError(f"invalid gene-set token in {path}: {token}")
        records.append((gene, numeric))
    return records


def build_gene_set_duplicate_audit(
    release_id: str,
    gene_set_paths: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for trait, path in sorted(gene_set_paths.items()):
        grouped = defaultdict(list)
        for gene, weight in parse_scdrs_gene_set(path, trait):
            grouped[gene].append(weight)
        for gene, weights in sorted(grouped.items()):
            if len(weights) < 2:
                continue
            rows.append(
                {
                    "release_id": release_id,
                    "trait_id": trait,
                    "gene": gene,
                    "source_entry_count": len(weights),
                    "source_weights_json": json.dumps(weights),
                    "scdrs_effective_weight": weights[-1],
                    "resolution_method": "last occurrence retained by scdrs.util.load_gs",
                    "weights_disagree": len(set(weights)) > 1,
                }
            )
    return pd.DataFrame(rows)


def collapse_gene_set_duplicate_weights(
    records: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], pd.DataFrame]:
    """Collapse duplicate gene symbols by an order-invariant arithmetic mean."""

    grouped: dict[str, list[float]] = defaultdict(list)
    order = []
    for gene, weight in records:
        if gene not in grouped:
            order.append(gene)
        grouped[gene].append(float(weight))
    collapsed = [(gene, float(np.mean(grouped[gene]))) for gene in order]
    changes = []
    for gene in order:
        weights = grouped[gene]
        if len(weights) < 2:
            continue
        corrected = float(np.mean(weights))
        changes.append(
            {
                "gene": gene,
                "source_entry_count": len(weights),
                "source_weights_json": json.dumps(weights),
                "original_last_occurrence_weight": weights[-1],
                "corrected_mean_weight": corrected,
                "absolute_weight_change": abs(corrected - weights[-1]),
                "relative_weight_change": (
                    abs(corrected - weights[-1]) / abs(weights[-1])
                    if weights[-1] != 0 else np.nan
                ),
                "correction_method": "arithmetic mean per duplicate gene symbol",
            }
        )
    return collapsed, pd.DataFrame(changes)


def build_trait_registry(
    release_id: str,
    gene_set_paths: dict[str, Path],
    gwas_manifest: dict,
    *,
    magma_version: str,
) -> pd.DataFrame:
    if set(gene_set_paths) != set(TRAITS):
        raise ValueError(
            f"trait gene-set paths do not match the frozen {len(TRAITS)} traits"
        )
    missing_labels = sorted(set(TRAITS) - set(TRAIT_LABELS))
    if missing_labels:
        raise ValueError(f"TRAIT_LABELS is missing display names for: {missing_labels}")
    gwas_traits = gwas_manifest.get("traits", {})
    rows = []
    for trait in sorted(TRAITS):
        path = gene_set_paths[trait]
        records = parse_scdrs_gene_set(path, trait)
        effective = dict(records)
        source = gwas_traits.get(trait)
        if not isinstance(source, dict):
            raise ValueError(f"GWAS provenance is missing for {trait}")
        digest = sha256(path)
        rows.append(
            {
                "release_id": release_id,
                "trait_id": trait,
                "display_name": TRAIT_LABELS[trait],
                "gene_set_version": f"{magma_version}:sha256:{digest[:16]}",
                "n_genes": len(effective),
                "n_source_gene_entries": len(records),
                "n_duplicate_gene_entries": len(records) - len(effective),
                "gene_set_sha256": digest,
                "gene_set_path": str(path),
                "weight_type": "MAGMA gene z score",
                "gwas_study": source.get("study"),
                "gwas_title": source.get("title"),
                "gwas_pmid": source.get("pmid"),
                "gwas_source": source.get("source"),
                "gwas_population": source.get("population"),
                "gwas_genome_build": source.get("genome_build"),
                "gwas_sample_size": source.get(
                    "n_total", source.get("n_effective", source.get("fixed_n"))
                ),
                "gene_set_method": "top weighted genes from the pinned MAGMA pipeline",
                "duplicate_resolution": "last occurrence retained by scdrs.util.load_gs",
            }
        )
    registry = pd.DataFrame(rows)
    validate_table("trait_registry", registry)
    return registry


def build_population_registry(
    release_id: str,
    population_effects: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "species", "hierarchy_level", "population_id", "parent_population_id"
    }
    if missing := sorted(required - set(population_effects.columns)):
        raise ValueError(f"population effects are missing columns: {missing}")
    source = population_effects[
        ["species", "hierarchy_level", "population_id", "parent_population_id"]
    ].drop_duplicates()
    population = source["population_id"].astype(str)
    source = source[
        ~population.eq("Sox6_nonAnxa")
        & ~source["hierarchy_level"].astype(str).eq("branch")
    ].copy()
    if source.duplicated(["species", "population_id"]).any():
        raise ValueError("population has multiple hierarchy definitions")
    rows = []
    for row in source.sort_values(
        ["species", "hierarchy_level", "population_id"], kind="stable"
    ).itertuples(index=False):
        parent_population_id = (
            str(row.population_id).split(":", 1)[0]
            if str(row.hierarchy_level) == "leaf"
            else row.parent_population_id
        )
        definitions = {
            "human": (
                "HMoE taxonomy applied to the integrated Kamath-plus-Siletti "
                "DopaBase Human object; Kamath defines subtype ground truth and "
                "Siletti contributes the independent integrated study stratum"
            ),
            "mouse": "frozen mouse HMoE DA taxonomy",
            "macaque": (
                "HMoE predictions on taxonomy-defined Allen HMBA Macaca mulatta "
                "midbrain DA cells from four contributing donors; Anxa1 is "
                "Sox6:Tafa1 plus Sox6:Vcan; brain-wide selectivity uses the "
                "DA-excluded Chiou rhesus cross-cohort reference"
            ),
        }
        if row.species not in definitions:
            raise ValueError(f"unknown population-registry species: {row.species}")
        definition = definitions[row.species]
        rows.append(
            {
                "release_id": release_id,
                "species": row.species,
                "population_id": row.population_id,
                "population_name": row.population_id,
                "parent_population_id": parent_population_id,
                "da_status": "da",
                "definition_source": definition,
                "hierarchy_level": row.hierarchy_level,
                "release_availability": "available",
            }
        )
    registry = pd.DataFrame(rows)
    validate_table("population_registry", registry)
    return registry
