#!/usr/bin/env python3
"""Append the merged Anxa1 focal population to the donor-aware scDRS effects.

Run:  python scripts/append_anxa1_population_effects.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for _p in (str(SRC), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fda_atlas_v2.contracts import validate_table  # noqa: E402
from fda_atlas_v2.population_effects import estimate_population_effects  # noqa: E402
import build_scdrs_population_effects as B  # noqa: E402

RELEASE = ROOT / "release" / "development"
CONFIG = ROOT / "configs" / "population_effects.yaml"
ANXA1_LEAVES = ("Sox6:Tafa1", "Sox6:Vcan")
SPECIES = ("human", "mouse")
LEVEL = "family"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _member(kind, leaf, family):
    if kind == "Anxa1":
        return leaf in ANXA1_LEAVES
    if kind == "Sox6_nonAnxa":
        return family == "Sox6" and leaf not in ANXA1_LEAVES
    return False


def merged_rows(species: str, species_config: dict, config: dict):
    """Return (effects_rows, donor_rows) for the merged Anxa1 population."""
    metadata, _ = B.load_metadata(species, species_config, "primary")
    metadata = B.add_hierarchy(metadata)
    primary = species_config["primary_cohort"]
    filters = species_config["cohorts"][primary]
    donor_col = species_config["donor_column"]
    study_col = species_config["study_column"]
    cond_col = species_config["condition_column"]

    effects_out, donor_out = [], []
    for trait in config["traits"]:
        score = B.load_scores(species, trait, metadata, species_config)
        scored = metadata.join(score)
        cohort = B.apply_filters(scored, filters)
        if cohort.empty:
            raise ValueError(f"{species} {primary}: primary cohort empty")
        source_donor = cohort[donor_col].astype(str)
        study = cohort[study_col].astype(str)
        donor_id = (study + "::" + source_donor if species == "human" else source_donor)
        for kind in ("Anxa1", "Sox6_nonAnxa"):
            pop = pd.Series(
                [kind if _member(kind, leaf, fam) else "rest_DA"
                 for leaf, fam in zip(cohort["leaf"], cohort["family"])],
                index=cohort.index,
            )
            cell_scores = pd.DataFrame(
                {
                    "donor_id": donor_id,
                    "source_donor_id": source_donor,
                    "study": study,
                    "condition": cohort[cond_col].astype(str),
                    "population_id": pop,
                    "score": cohort["score"],
                },
                index=cohort.index,
            )
            effects, donor_means = estimate_population_effects(
                cell_scores,
                min_cells_per_donor_population=int(config["min_cells_per_donor_population"]),
            )
            eff = effects[effects["population_id"].eq(kind)].copy()
            if eff.empty:
                raise ValueError(f"{species} {trait} {kind}: not estimable")
            eff["parent_population_id"] = "DA"
            dm = donor_means[donor_means["population_id"].isin([kind, "rest_DA"])].copy()
            common = {
                "release_id": config["release_id"],
                "species": species,
                "trait_id": trait,
                "hierarchy_level": LEVEL,
                "analysis_cohort": primary,
                "is_primary_cohort": True,
                "analysis_role": "primary",
            }
            for col, val in common.items():
                eff[col] = val
                dm[col] = val
            effects_out.append(eff)
            donor_out.append(dm)
    return effects_out, donor_out


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    B.TRAITS = list(config["traits"])

    effect_parts, donor_parts = [], []
    for species in SPECIES:
        e, d = merged_rows(species, config["species"][species], config)
        effect_parts.extend(e)
        donor_parts.extend(d)
    add_effects = pd.concat(effect_parts, ignore_index=True)
    add_donors = pd.concat(donor_parts, ignore_index=True)

    paths = {
        "effects": RELEASE / "scdrs_population_effects.parquet",
        "donor_means": RELEASE / "scdrs_donor_population_means.parquet",
    }
    additions = {"effects": add_effects, "donor_means": add_donors}
    combined = {}
    for name, path in paths.items():
        existing = pd.read_parquet(path)
        population = existing["population_id"].astype(str)
        replace = (
            existing["species"].astype(str).isin(SPECIES)
            & population.isin(["Anxa1", "Sox6_nonAnxa", "rest_DA"])
        )
        existing = existing[~replace]
        columns = list(dict.fromkeys([*existing.columns, *additions[name].columns]))
        combined[name] = pd.concat(
            [existing.reindex(columns=columns), additions[name].reindex(columns=columns)],
            ignore_index=True,
        )
    validate_table("scdrs_population_effects", combined["effects"])
    for name, path in paths.items():
        atomic_parquet(combined[name], path)

    manifest_path = RELEASE / "scdrs_population_effects_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("anxa1_addendum", {})
    manifest["anxa1_addendum"] = {
        "leaves": list(ANXA1_LEAVES),
        "hierarchy_level": LEVEL,
        "contrast": "Anxa1 vs pooled rest_DA (two-population relative_effect)",
        "species": list(SPECIES),
        "rows_added": int(len(add_effects)),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest["counts"]["effects"] = int(len(combined["effects"]))
    manifest["counts"]["donor_population_rows"] = int(len(combined["donor_means"]))
    for name, path in paths.items():
        manifest["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": int(len(combined[name])),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest.setdefault("pipeline_files", {})[str(Path(__file__).resolve())] = sha256(
        Path(__file__).resolve()
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps({
        "anxa1_effect_rows_added": int(len(add_effects)),
        "anxa1_donor_rows_added": int(len(add_donors)),
        "effects_total": int(len(combined["effects"])),
    }, indent=2))


if __name__ == "__main__":
    main()
