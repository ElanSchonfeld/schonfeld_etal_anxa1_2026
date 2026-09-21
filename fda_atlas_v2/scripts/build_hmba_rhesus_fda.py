#!/usr/bin/env python3
"""Build the FDA Atlas rhesus HMBA focal-DA evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp
import scdrs


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (REPOSITORY_ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fda_atlas_v2.contracts import TRAITS, validate_table  # noqa: E402
from fda_atlas_v2.cross_cohort_brainwide import (  # noqa: E402
    cross_cohort_brainwide_selectivity,
)
from fda_atlas_v2.direct_target_universe import (  # noqa: E402
    add_direct_label_edges,
    build_direct_label_edges,
    direct_target_pairs,
)
from fda_atlas_v2.population_effects import (  # noqa: E402
    estimate_population_effects,
    select_population_lead_set,
)
from fda_atlas_v2.pseudobulk import (  # noqa: E402
    aggregate_sparse_pseudobulk,
    save_sparse_pseudobulk_checkpoint,
)
from fda_atlas_v2.rhesus_hmba import (  # noqa: E402
    ANXA1_LEAVES,
    CANONICAL_LEAVES,
    HMBA_DA_GROUPS,
    population_members,
    select_rhesus_da_cells,
)
from fda_atlas_v2.selectivity_pipeline import (  # noqa: E402
    chunked_strongest_competitor_selectivity,
)
from fda_atlas_v2.target_selectivity import build_target_gene_registry  # noqa: E402
from rebuild_within_da_rest import build_pooled_rest_pseudobulk  # noqa: E402
from hmoe_annotate.hmoe_model import (  # noqa: E402
    align_features as align_hmoe_features,
    load_model as load_hmoe_model,
    predict_proba as predict_hmoe_probabilities,
)


DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
RAW_H5AD = DATA / "macaque/allen/HMBA-10xMultiome-BG-Macaque-raw.h5ad"
SIDECAR_ROOT = DATA / "human/atlas/allen/basal_ganglia/multiome"
CELL_MEMBERSHIP = SIDECAR_ROOT / "cell_to_cluster_membership.csv"
CLUSTER_HIERARCHY = SIDECAR_ROOT / "cluster_to_cluster_annotation_membership.csv"
CELL_METADATA = DATA / "macaque/allen/cell_metadata.csv"
DONORS = ROOT / "configs/hmba_macaque_donors.csv"
ORTHOLOGS = DATA / "cache/ensembl/116/human_macaca_mulatta_orthologs.tsv"
GENE_SETS = ROOT / "derived/rhesus_scdrs_gene_sets/primary/symbol"
RELEASE = ROOT / "release/development"
DERIVED = DATA / "derived/m2h_fda_scdrs_atlas_v2/rhesus_hmba"
SCT_ROOT = DATA / "derived/m2h_scdrs_sct_2026-07-30/rhesus"
PSEUDOBULK = DATA / "derived/m2h_fda_scdrs_atlas_v2/pseudobulk"
AXIS_CACHE = ROOT / "results/direct_target_rebuild/macaque_axes"
CHIOU_REFERENCE = (
    DATA
    / "derived/m2h_fda_scdrs_atlas_v2/"
    "rhesus_chiou_brainwide_reference"
)
CHIOU_REFERENCE_MANIFEST = (
    RELEASE / "rhesus_chiou_brainwide_reference_manifest.json"
)
MACAQUE_REGION_CONTRASTS = (
    ROOT / "results/direct_target_rebuild/macaque_brainwide_region_contrasts.parquet"
)
HMOE_ROOT = Path(
    os.environ.get("DOPABASE_HMOE_ROOT", REPOSITORY_ROOT / "hmoe_annotate")
).expanduser()
HMOE_MODEL = HMOE_ROOT / "model"
HMOE_MODEL_SOURCE = HMOE_ROOT / "hmoe_model.py"
EXPECTED_DA_CELLS = 3_368
EXPECTED_DA_DONORS = 4
SEED = 42


def pending_brainwide_axis(genes: list[str]) -> pd.DataFrame:
    """Return explicit unavailable rows when Allen focal-donor support is too low."""

    return pd.DataFrame(
        {
            "gene_id": genes,
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "detection_difference": np.nan,
            "competitor_id": None,
            "n_common_donors": 0,
            "focal_n_cells": 0,
            "competitor_n_cells": 0,
            "focal_detection": np.nan,
            "raw_effect": np.nan,
            "raw_standard_error": np.nan,
            "loo_raw_effect_min": np.nan,
            "loo_raw_effect_max": np.nan,
            "matrix_representation": (
                "Allen focal donor log2 CPM versus donor-balanced Chiou non-DA "
                "cross-cohort reference"
            ),
            "strongest_competitor_rule": (
                "unavailable: fewer than three Allen focal donors have ten cells"
            ),
            "interval_method": "unavailable: insufficient focal donors",
            "selection_uncertainty_status": "insufficient_focal_donor_support",
            "status": "insufficient_comparator_support",
            "p_value": np.nan,
            "q_value": np.nan,
            "competitor_loo_selection_fraction": np.nan,
            "common_donor_ids": "",
            "n_eligible_comparators": 0,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=index)
    os.replace(partial, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def load_cell_selection() -> pd.DataFrame:
    selected = select_rhesus_da_cells(
        pd.read_csv(CELL_MEMBERSHIP),
        pd.read_csv(CLUSTER_HIERARCHY),
        pd.read_csv(CELL_METADATA),
        pd.read_csv(DONORS),
    )
    if len(selected) != EXPECTED_DA_CELLS:
        raise ValueError(
            f"strict rhesus HMBA DA cells differ: {len(selected)} != {EXPECTED_DA_CELLS}"
        )
    if selected["donor_label"].nunique() != EXPECTED_DA_DONORS:
        raise ValueError("strict rhesus HMBA DA donor count differs from four")
    return selected


def load_selected_counts(selected: pd.DataFrame) -> ad.AnnData:
    source = ad.read_h5ad(RAW_H5AD, backed="r")
    try:
        positions = source.obs_names.get_indexer(selected["cell_label"].astype(str))
        if (positions < 0).any() or len(set(positions.tolist())) != len(positions):
            raise ValueError("strict rhesus cell identities do not match the HMBA raw H5AD")
        data = source[positions].to_memory()
    finally:
        source.file.close()
    if data.shape != (EXPECTED_DA_CELLS, 35_219) or not sp.issparse(data.X):
        raise ValueError(f"unexpected selected HMBA raw matrix: {data.shape}")
    values = data.X.data
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values, np.rint(values), rtol=0, atol=1e-8)
    ):
        raise ValueError("HMBA X is not sparse nonnegative integer raw counts")
    data.obs = selected.set_index("cell_label").reindex(data.obs_names).copy()
    return data


def predict_hmoe(data: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HMOE_MODEL_SOURCE.is_file():
        raise FileNotFoundError(HMOE_MODEL_SOURCE)
    model = load_hmoe_model(HMOE_MODEL)
    gene_symbols = data.var["gene_symbol"].astype(object).fillna("").astype(str)
    aligned, matched_genes, _ = align_hmoe_features(
        gene_symbols,
        data.X,
        model["feature_names"],
    )
    if matched_genes == 0:
        raise ValueError("no HMBA gene symbols matched the HMoE training features")
    probabilities = predict_hmoe_probabilities(model, aligned)
    leaves = np.asarray(model["leaves"], dtype=object)
    predicted = leaves[probabilities.argmax(axis=1)].astype(str)
    if not set(predicted).issubset(CANONICAL_LEAVES):
        raise ValueError(f"HMoE produced noncanonical leaves: {sorted(set(predicted) - set(CANONICAL_LEAVES))}")
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    return predicted, confidence, margin


def collapse_gene_symbols(data: ad.AnnData) -> tuple[sp.csr_matrix, list[str], sp.csr_matrix]:
    symbols = (
        data.var["gene_symbol"].astype(object).fillna("").astype(str).str.strip().tolist()
    )
    names = []
    position = {}
    codes = []
    for index, symbol in enumerate(symbols):
        name = symbol if symbol else f"__unmapped_feature_{index}"
        if name not in position:
            position[name] = len(names)
            names.append(name)
        codes.append(position[name])
    collapse = sp.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float64),
            (np.arange(len(codes), dtype=int), np.asarray(codes, dtype=int)),
        ),
        shape=(len(codes), len(names)),
    )
    collapsed = (data.X.tocsr() @ collapse).tocsr()
    return collapsed, names, collapse


def build_membership_and_pseudobulk() -> tuple[pd.DataFrame, Path]:
    selected = load_cell_selection()
    data = load_selected_counts(selected)
    predicted, confidence, margin = predict_hmoe(data)
    selected = data.obs.copy()
    selected["leaf"] = predicted
    selected["family"] = selected["leaf"].str.split(":", n=1).str[0]
    selected["hmoe_confidence"] = confidence
    selected["hmoe_margin"] = margin
    selected.insert(0, "cell_id", selected.index.astype(str))
    selected["species"] = "macaque"
    selected["species_scientific_name"] = "Macaca mulatta"
    selected["donor_id"] = selected["donor_label"].astype(str)
    selected["taxonomy_da_group"] = selected[
        "cluster_annotation_term_name"
    ].astype(str)
    if set(selected["taxonomy_da_group"]) != set(HMBA_DA_GROUPS):
        raise ValueError("one or more exact HMBA DA taxonomy groups are absent")
    if not set(ANXA1_LEAVES).issubset(set(selected["leaf"])):
        raise ValueError("the two canonical Anxa1 leaves are not both present")
    anxa = selected[selected["leaf"].isin(ANXA1_LEAVES)]
    if anxa["donor_id"].nunique() != EXPECTED_DA_DONORS:
        raise ValueError("Anxa1 is not represented in all four rhesus DA donors")

    collapsed, names, collapse = collapse_gene_symbols(data)
    observations = selected[["donor_id", "leaf"]].rename(
        columns={"leaf": "population_id"}
    )
    pseudobulk = aggregate_sparse_pseudobulk(
        data.X,
        observations,
        group_columns=["donor_id", "population_id"],
        feature_collapse=collapse,
        minimum_cells=1,
    )
    checkpoint = save_sparse_pseudobulk_checkpoint(
        pseudobulk,
        names,
        PSEUDOBULK,
        checkpoint_id="rhesus_hmba_da_leaf",
        overwrite=True,
    )
    cell_path = RELEASE / "rhesus_hmba_focal_cell_registry.parquet"
    atomic_parquet(selected.reset_index(drop=True), cell_path)
    matrix_path = DERIVED / "rhesus_hmba_da_collapsed_raw.h5ad"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    score_data = ad.AnnData(
        X=collapsed.astype(np.float32),
        obs=selected.set_index("cell_id")[["donor_id", "leaf", "family"]].copy(),
        var=pd.DataFrame(index=pd.Index(names, name="gene_symbol")),
    )
    partial_matrix = matrix_path.with_suffix(".partial.h5ad")
    score_data.write_h5ad(partial_matrix, compression="gzip")
    os.replace(partial_matrix, matrix_path)
    manifest = {
        "release_id": "development",
        "species_key": "macaque",
        "species_scientific_name": "Macaca mulatta",
        "taxon": "NCBITaxon:9544",
        "excluded_species": ["Macaca nemestrina"],
        "taxonomy_da_groups": list(HMBA_DA_GROUPS),
        "population_definition": {
            "branches_included": False,
            "Anxa1": list(ANXA1_LEAVES),
            "HMoE_assignment": "hard argmax without confidence or detection-difference gate",
        },
        "counts": {
            "cells": len(selected),
            "donors": selected["donor_id"].nunique(),
            "cells_by_donor": selected["donor_id"].value_counts().sort_index().to_dict(),
            "cells_by_family": selected["family"].value_counts().sort_index().to_dict(),
            "cells_by_leaf": selected["leaf"].value_counts().sort_index().to_dict(),
            "anxa1_cells": len(anxa),
            "anxa1_donors": anxa["donor_id"].nunique(),
            "unique_gene_symbols": len(names),
        },
        "transformations": [
            "raw integer counts retained",
            "duplicate source features summed by exact gene_symbol",
            "raw counts summed by donor and HMoE leaf before full-library log2 CPM",
        ],
        "inputs": {
            str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (RAW_H5AD, CELL_MEMBERSHIP, CLUSTER_HIERARCHY, CELL_METADATA, DONORS)
        },
        "outputs": {
            "cell_registry": str(cell_path.resolve()),
            "collapsed_raw_h5ad": str(matrix_path.resolve()),
            "pseudobulk_manifest": str(checkpoint.resolve()),
        },
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str((SRC / "fda_atlas_v2/rhesus_hmba.py").resolve()): sha256(
                SRC / "fda_atlas_v2/rhesus_hmba.py"
            ),
            str(HMOE_MODEL_SOURCE.resolve()): sha256(HMOE_MODEL_SOURCE),
            str((HMOE_MODEL / "v4_improved_unified/meta.json").resolve()): sha256(
                HMOE_MODEL / "v4_improved_unified/meta.json"
            ),
        },
    }
    atomic_json(manifest, RELEASE / "rhesus_hmba_input_manifest.json")
    return selected, checkpoint


def load_gene_set(path: Path, var_names: pd.Index) -> tuple[list[str], list[float]]:
    loaded = scdrs.util.load_gs(
        str(path), src_species="human", dst_species="human", to_intersect=list(var_names)
    )
    if len(loaded) != 1:
        raise ValueError(f"expected one gene set in {path}")
    genes, weights = next(iter(loaded.values()))
    if len(genes) < 100:
        raise ValueError(f"{path.stem}: fewer than 100 rhesus genes matched")
    return list(genes), list(weights)


def score_traits() -> pd.DataFrame:
    if getattr(scdrs, "__version__", "") != "1.0.2":
        raise ValueError(f"expected scDRS 1.0.2, observed {scdrs.__version__}")
    matrix_path = SCT_ROOT / "sct.mtx"
    raw_path = DERIVED / "rhesus_hmba_da_collapsed_raw.h5ad"
    sct = sio.mmread(matrix_path).T.tocsr()
    sct_genes = [line.strip() for line in (SCT_ROOT / "sct_genes.txt").read_text().splitlines() if line.strip()]
    sct_cells = [line.strip() for line in (SCT_ROOT / "sct_cells.txt").read_text().splitlines() if line.strip()]
    if sct.shape != (len(sct_cells), len(sct_genes)):
        raise ValueError(f"rhesus SCT orientation {sct.shape} vs {len(sct_cells)}x{len(sct_genes)}")

    raw = ad.read_h5ad(raw_path)
    if "cell_id" in raw.obs.columns:
        raw.obs_names = raw.obs["cell_id"].astype(str).values
    raw = raw[sct_cells]
    n_genes = np.asarray((raw.X > 0).sum(axis=1)).ravel().astype(float)
    donor = raw.obs["donor_id"].astype(str)
    del raw
    gc.collect()
    dummies = pd.get_dummies(donor, prefix="donor", drop_first=True).astype(float)
    dummies.index = sct_cells
    cov = pd.concat(
        [pd.DataFrame({"const": 1.0, "n_genes": n_genes}, index=sct_cells), dummies],
        axis=1,
    )
    print(f"  rhesus covariates: const + n_genes + {dummies.shape[1]} donor dummies "
          f"({donor.nunique()} donors)", flush=True)

    data = ad.AnnData(
        X=np.asarray(sct.todense(), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(sct_cells)),
        var=pd.DataFrame(index=pd.Index(sct_genes)),
    )
    del sct
    gc.collect()
    scdrs.preprocess(data, cov=cov, n_mean_bin=20, n_var_bin=20, copy=False)
    output = pd.DataFrame(index=data.obs_names.copy())
    gene_sets = {}
    for trait in TRAITS:
        path = (GENE_SETS / f"{trait}.gs").resolve()
        genes, weights = load_gene_set(path, data.var_names)
        score = scdrs.score_cell(
            data=data,
            gene_list=genes,
            gene_weight=weights,
            ctrl_match_key="mean_var",
            n_ctrl=1_000,
            weight_opt="vs",
            return_ctrl_norm_score=False,
            random_seed=SEED,
            verbose=False,
        )
        output[f"scdrs_{trait}_norm_score"] = score["norm_score"].reindex(output.index)
        output[f"scdrs_{trait}_zscore"] = score["zscore"].reindex(output.index)
        gene_sets[trait] = {"path": str(path), "sha256": sha256(path), "matched_genes": len(genes)}
        print(f"  scored rhesus {trait}: matched_genes={len(genes)}", flush=True)
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("rhesus HMBA scDRS scores contain nonfinite values")
    score_path = RELEASE / "rhesus_hmba_scdrs_scores.parquet"
    atomic_parquet(output, score_path, index=True)
    manifest = {
        "schema_version": "1.0.0",
        "release_id": "development",
        "status": "complete",
        "analysis_role": "primary_within_da_donor_aware",
        "source": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256(matrix_path),
            "covariate_source": str(raw_path.resolve()),
            "cells": data.n_obs,
            "genes": data.n_vars,
            "biological_donors": int(donor.nunique()),
            "species_scientific_name": "Macaca mulatta",
        },
        "parameters": {
            "normalization": "none (SCTransform v2 corrected counts used directly)",
            "n_mean_bin": 20,
            "n_var_bin": 20,
            "ctrl_match_key": "mean_var",
            "n_ctrl": 1_000,
            "weight_opt": "vs",
            "random_seed": SEED,
            "scoring_option": "B",
            "covariates": ["const", "n_genes", "donor_id (one-hot drop_first)"],
            "matrix_representation": "dense float32",
        },
        "gene_sets": gene_sets,
        "software": {
            "python": platform.python_version(),
            "anndata": package_version("anndata"),
            "scanpy": package_version("scanpy"),
            "scdrs": package_version("scdrs"),
        },
        "output": {"path": str(score_path.resolve()), "rows": len(output), "sha256": sha256(score_path)},
        "pipeline_files": {str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
    }
    atomic_json(manifest, RELEASE / "rhesus_hmba_scdrs_scores_manifest.json")
    return output


def _effect_inputs(metadata: pd.DataFrame, score: pd.Series, population: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "donor_id": metadata["donor_id"].astype(str),
            "source_donor_id": metadata["donor_id"].astype(str),
            "study": "Allen HMBA",
            "condition": "Control",
            "population_id": population.astype(str),
            "score": pd.to_numeric(score, errors="coerce"),
        },
        index=metadata.index,
    )


def build_population_effects() -> dict[str, pd.DataFrame]:
    metadata = pd.read_parquet(RELEASE / "rhesus_hmba_focal_cell_registry.parquet").set_index("cell_id")
    scores = pd.read_parquet(RELEASE / "rhesus_hmba_scdrs_scores.parquet")
    if not scores.index.equals(metadata.index):
        scores = scores.reindex(metadata.index)
    if scores.isna().any(axis=None):
        raise ValueError("rhesus HMBA scores do not align with the cell registry")
    effect_parts = []
    lead_parts = []
    donor_parts = []
    for trait in TRAITS:
        score = scores[f"scdrs_{trait}_norm_score"]
        for level in ("family", "leaf"):
            effects, donor_means = estimate_population_effects(
                _effect_inputs(metadata, score, metadata[level]),
                min_cells_per_donor_population=10,
            )
            effects["parent_population_id"] = (
                "DA" if level == "family" else effects["population_id"].str.split(":", n=1).str[0]
            )
            leads = select_population_lead_set(
                effects, donor_means, min_common_donors=3, min_donors=3
            )
            common = {
                "release_id": "development",
                "species": "macaque",
                "trait_id": trait,
                "hierarchy_level": level,
                "analysis_cohort": "allen_hmba_rhesus_control",
                "is_primary_cohort": True,
                "analysis_role": "primary",
            }
            for frame in (effects, leads, donor_means):
                for column, value in common.items():
                    frame[column] = value
            effect_parts.append(effects)
            lead_parts.append(leads)
            donor_parts.append(donor_means)

        merged_membership = {
            "Anxa1": metadata["leaf"].isin(ANXA1_LEAVES),
            "Sox6_nonAnxa": (
                metadata["leaf"].astype(str).str.startswith("Sox6:")
                & ~metadata["leaf"].isin(ANXA1_LEAVES)
            ),
        }
        common = {
            "release_id": "development",
            "species": "macaque",
            "trait_id": trait,
            "hierarchy_level": "family",
            "analysis_cohort": "allen_hmba_rhesus_control",
            "is_primary_cohort": True,
            "analysis_role": "primary",
        }
        for merged_id, member_mask in merged_membership.items():
            if not bool(member_mask.any()):
                raise ValueError(f"rhesus DA selection contains no {merged_id} cells")
            merged_population = pd.Series(
                np.where(member_mask, merged_id, "rest_DA"), index=metadata.index
            )
            merged_effects, merged_donors = estimate_population_effects(
                _effect_inputs(metadata, score, merged_population),
                min_cells_per_donor_population=10,
            )
            merged_effects = merged_effects[
                merged_effects["population_id"].eq(merged_id)
            ].copy()
            merged_donors = merged_donors[
                merged_donors["population_id"].isin([merged_id, "rest_DA"])
            ].copy()
            merged_effects["parent_population_id"] = "DA"
            for frame in (merged_effects, merged_donors):
                for column, value in common.items():
                    frame[column] = value
            effect_parts.append(merged_effects)
            donor_parts.append(merged_donors)

            lead_parts.append(
                pd.DataFrame(
                    {
                        "population_id": [merged_id],
                        "lead_status": ["independent_focal_contrast"],
                        "leader_id": [""],
                        "delta_from_leader": [np.nan],
                        "delta_ci_low": [np.nan],
                        "delta_ci_high": [np.nan],
                        "n_common_donors": [int(merged_effects["n_donors"].iloc[0])],
                        "selection_reason": [
                            f"{merged_id} versus pooled rest_DA is reported independently "
                            "and is not used to choose a family leader"
                        ],
                        **{column: [value] for column, value in common.items()},
                    }
                )
            )

    additions = {
        "effects": pd.concat(effect_parts, ignore_index=True),
        "leads": pd.concat(lead_parts, ignore_index=True),
        "donor_means": pd.concat(donor_parts, ignore_index=True),
    }
    expected = population_members()
    effect_placeholders = []
    lead_placeholders = []
    for trait in TRAITS:
        observed = set(
            additions["effects"].loc[
                additions["effects"]["trait_id"].astype(str).eq(trait),
                "population_id",
            ].astype(str)
        )
        for population in sorted(set(expected) - observed):
            level = "leaf" if ":" in population else "family"
            parent = population.split(":", 1)[0] if level == "leaf" else "DA"
            n_cells = int(
                metadata["leaf"].astype(str).eq(population).sum()
                if level == "leaf"
                else metadata["family"].astype(str).eq(population).sum()
            )
            common = {
                "release_id": "development",
                "species": "macaque",
                "trait_id": trait,
                "hierarchy_level": level,
                "analysis_cohort": "allen_hmba_rhesus_control",
                "is_primary_cohort": True,
                "analysis_role": "primary",
            }
            effect_placeholders.append(
                {
                    "population_id": population,
                    "effect": np.nan,
                    "standard_error": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_donors": 0,
                    "n_studies": 1,
                    "n_cells": n_cells,
                    "min_cells_per_donor": 0,
                    "median_cells_per_donor": 0.0,
                    "positive_donor_fraction": np.nan,
                    "loo_effect_min": np.nan,
                    "loo_effect_max": np.nan,
                    "status": "insufficient_donor_cell_support",
                    "inference_model": (
                        "unavailable because fewer than three donors have at least ten cells"
                    ),
                    "parent_population_id": parent,
                    **common,
                }
            )
            lead_placeholders.append(
                {
                    "population_id": population,
                    "lead_status": "insufficient_data",
                    "leader_id": "",
                    "delta_from_leader": np.nan,
                    "delta_ci_low": np.nan,
                    "delta_ci_high": np.nan,
                    "n_common_donors": 0,
                    "selection_reason": (
                        "fewer than three donors have at least ten cells"
                    ),
                    **common,
                }
            )
    if effect_placeholders:
        additions["effects"] = pd.concat(
            [additions["effects"], pd.DataFrame(effect_placeholders)],
            ignore_index=True,
        )
        additions["leads"] = pd.concat(
            [additions["leads"], pd.DataFrame(lead_placeholders)],
            ignore_index=True,
        )
    paths = {
        "effects": RELEASE / "scdrs_population_effects.parquet",
        "leads": RELEASE / "trait_population_leads.parquet",
        "donor_means": RELEASE / "scdrs_donor_population_means.parquet",
    }
    combined = {}
    for name, path in paths.items():
        existing = pd.read_parquet(path)
        existing = existing[
            ~existing["species"].astype(str).eq("macaque")
            & ~existing["hierarchy_level"].astype(str).eq("branch")
        ]
        columns = list(dict.fromkeys([*existing.columns, *additions[name].columns]))
        combined[name] = pd.concat(
            [existing.reindex(columns=columns), additions[name].reindex(columns=columns)],
            ignore_index=True,
        )
    validate_table("scdrs_population_effects", combined["effects"])
    validate_table("trait_population_leads", combined["leads"])
    for name, path in paths.items():
        atomic_parquet(combined[name], path)
    manifest_path = RELEASE / "scdrs_population_effects_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["primary_cohorts"]["macaque"] = "allen_hmba_rhesus_control"
    manifest["parameters"]["branch_populations_included"] = False
    manifest["counts"] = {
        "effects": len(combined["effects"]),
        "lead_rows": len(combined["leads"]),
        "donor_population_rows": len(combined["donor_means"]),
    }
    manifest.setdefault("validation", {})["macaque_rhesus_traits"] = len(TRAITS)
    manifest["validation"]["macaque_inferential_statistics_available"] = True
    manifest["validation"]["macaque_species_scientific_name"] = "Macaca mulatta"
    for path_text, record in manifest.get("inputs", {}).items():
        source = Path(path_text).expanduser()
        if source.is_file() and source.suffix == ".py":
            record["size_bytes"] = source.stat().st_size
            record["sha256"] = sha256(source)
    for path_text in list(manifest.get("pipeline_files", {})):
        source = Path(path_text).expanduser()
        if source.is_file():
            manifest["pipeline_files"][path_text] = sha256(source)
    manifest.setdefault("pipeline_files", {})[str(Path(__file__).resolve())] = sha256(
        Path(__file__).resolve()
    )
    for name, path in paths.items():
        manifest["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": len(combined[name]),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    atomic_json(manifest, manifest_path)
    return combined


def strict_target_registry() -> pd.DataFrame:
    edges = pd.read_parquet(RELEASE / "drug_target_edges.parquet")
    labels = build_direct_label_edges(pd.read_parquet(RELEASE / "active_moieties.parquet"))
    edges = add_direct_label_edges(edges, labels)
    targets = direct_target_pairs(
        edges,
        pd.read_parquet(RELEASE / "regulatory_substance_target_evidence.parquet"),
    )
    registry = build_target_gene_registry(
        targets,
        species="macaque",
        orthologs=pd.read_csv(ORTHOLOGS, sep="\t", dtype=str),
    )
    if len(registry) != 533:
        raise ValueError(f"strict rhesus target registry differs from 533: {len(registry)}")
    return registry


def load_chiou_reference() -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    manifest = json.loads(CHIOU_REFERENCE_MANIFEST.read_text())
    validation = manifest.get("validation", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("donor_matching") is not False
        or validation.get("all_source_da_excluded_before_aggregation") is not True
        or validation.get("reference_is_donor_matched") is not False
    ):
        raise ValueError("Chiou brain-wide reference provenance is not release-valid")
    matrices_path = CHIOU_REFERENCE / "reference_matrices.npz"
    metadata_path = CHIOU_REFERENCE / "reference_metadata.parquet"
    features_path = CHIOU_REFERENCE / "features.parquet"
    for path in (matrices_path, metadata_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    matrices = np.load(matrices_path)
    expression = np.asarray(matrices["expression"], dtype=float)
    detection = np.asarray(matrices["detection"], dtype=float)
    metadata = pd.read_parquet(metadata_path)
    features = pd.read_parquet(features_path)
    measured = features["measured_in_chiou"].astype(bool).to_numpy()
    genes = features.loc[measured, "feature_name"].astype(str).tolist()
    if (
        expression.shape != detection.shape
        or expression.shape[0] != len(metadata)
        or expression.shape[1] != len(features)
        or not genes
    ):
        raise ValueError("Chiou brain-wide reference outputs do not align")
    return expression[:, measured], detection[:, measured], genes, metadata


def focal_population_arrays(
    pseudobulk,
    feature_names: tuple[str, ...],
    *,
    members: set[str],
    genes: list[str],
    minimum_cells_per_donor: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    feature_index = {str(gene): index for index, gene in enumerate(feature_names)}
    common = [gene for gene in genes if gene in feature_index]
    if common != genes:
        missing = sorted(set(genes) - set(common))
        raise ValueError(f"Allen HMBA is missing Chiou target genes: {missing[:5]}")
    columns = np.asarray([feature_index[gene] for gene in genes], dtype=int)
    metadata = pseudobulk.metadata.reset_index(drop=True)
    expression = []
    detection = []
    donors = []
    total_cells = 0
    for donor in sorted(metadata["donor_id"].astype(str).unique()):
        mask = (
            metadata["donor_id"].astype(str).eq(donor)
            & metadata["population_id"].astype(str).isin(members)
        ).to_numpy()
        cells = int(metadata.loc[mask, "n_cells"].sum())
        library = float(metadata.loc[mask, "library_size"].sum())
        if cells < minimum_cells_per_donor or library <= 0:
            continue
        counts = np.asarray(
            pseudobulk.counts[mask][:, columns].sum(axis=0)
        ).ravel()
        detected = np.asarray(
            pseudobulk.detected_cells[mask][:, columns].sum(axis=0)
        ).ravel()
        expression.append(np.log2((counts + 0.5) / library * 1e6))
        detection.append(detected / cells)
        donors.append(donor)
        total_cells += cells
    if not expression:
        return (
            np.empty((0, len(genes))),
            np.empty((0, len(genes))),
            [],
            0,
        )
    return np.vstack(expression), np.vstack(detection), donors, total_cells


def build_selectivity_axes() -> None:
    from fda_atlas_v2.pseudobulk import load_sparse_pseudobulk_checkpoint

    checkpoint = PSEUDOBULK / "rhesus_hmba_da_leaf.manifest.json"
    pseudobulk, names = load_sparse_pseudobulk_checkpoint(checkpoint)
    (
        reference_expression,
        reference_detection,
        reference_genes,
        reference_metadata,
    ) = load_chiou_reference()
    registry = strict_target_registry()
    mapped = registry[
        registry["gene_mapping_status"].eq("reciprocal_one_to_one_macaque_ortholog")
    ]
    wanted = set(mapped["species_gene"].astype(str))
    hmba_genes = set(map(str, names))
    common_genes = [gene for gene in reference_genes if gene in hmba_genes]
    reference_positions = np.asarray(
        [reference_genes.index(gene) for gene in common_genes], dtype=int
    )
    reference_expression = reference_expression[:, reference_positions]
    reference_detection = reference_detection[:, reference_positions]
    reference_regions = reference_metadata["region"].astype(str).tolist()
    reference_cells = reference_metadata["n_non_da_cells"].astype(int).tolist()
    reference_donors = (
        reference_metadata["n_reference_donors"].astype(int).tolist()
    )
    AXIS_CACHE.mkdir(parents=True, exist_ok=True)
    region_parts = []
    for population, members in population_members().items():
        pooled = build_pooled_rest_pseudobulk(pseudobulk, members, population)
        genomewide = chunked_strongest_competitor_selectivity(
            pooled,
            names,
            focal_population=population,
            comparator_populations=["rest_DA"],
            gene_chunk_size=2_000,
            minimum_common_donors=3,
            minimum_cells_per_donor_population=10,
            minimum_focal_detection=0.05,
        )
        within = genomewide[genomewide["gene_id"].astype(str).isin(wanted)].copy()
        focal_expression, focal_detection, focal_donors, focal_cells = (
            focal_population_arrays(
                pseudobulk,
                names,
                members=set(members),
                genes=common_genes,
            )
        )
        if len(focal_donors) >= 3:
            brainwide = cross_cohort_brainwide_selectivity(
                focal_expression,
                focal_detection,
                genes=common_genes,
                focal_donor_ids=focal_donors,
                focal_n_cells=focal_cells,
                reference_expression=reference_expression,
                reference_detection=reference_detection,
                reference_population_ids=reference_regions,
                reference_n_cells=reference_cells,
                reference_donor_counts=reference_donors,
                minimum_focal_detection=0.05,
            )
            region_effects = (
                focal_expression.mean(axis=0)[np.newaxis, :]
                - reference_expression
            ).T
            region_part = pd.DataFrame(
                region_effects,
                columns=[f"region::{region}" for region in reference_regions],
            )
            region_part.insert(0, "gene_id", common_genes)
            region_part.insert(0, "population_id", population)
            region_part.insert(0, "species", "macaque")
            region_parts.append(region_part)
        else:
            brainwide = pending_brainwide_axis(common_genes)
        prefix = population.replace(":", "__")
        for frame, axis_id in ((genomewide, "within_da"), (brainwide, "brainwide")):
            if frame.empty:
                continue
            frame["release_id"] = "development"
            frame["species"] = "macaque"
            frame["focal_population_id"] = population
            frame["axis_id"] = axis_id
            frame["dataset_id"] = "allen_hmba_rhesus_bg_raw_counts"
            frame["analysis_cohort"] = "allen_hmba_rhesus_control"
        atomic_parquet(genomewide, AXIS_CACHE / f"{prefix}.within_da.parquet")
        atomic_parquet(brainwide, AXIS_CACHE / f"{prefix}.brainwide.parquet")
        estimated = int(within["status"].eq("estimated").sum())
        donors = int(pd.to_numeric(within["n_common_donors"], errors="coerce").max())
        brain_estimated = int(brainwide["status"].eq("estimated").sum())
        print(
            f"  rhesus selectivity {population}: within={estimated}, "
            f"brainwide={brain_estimated}, focal_donors={len(focal_donors)}",
            flush=True,
        )
    if not region_parts:
        raise ValueError("no rhesus brain-wide region contrasts were estimated")
    atomic_parquet(pd.concat(region_parts, ignore_index=True), MACAQUE_REGION_CONTRASTS)
    atomic_parquet(registry, ROOT / "results/direct_target_rebuild/macaque_target_gene_registry.parquet")
    manifest = {
        "release_id": "development",
        "species_key": "macaque",
        "species_scientific_name": "Macaca mulatta",
        "dataset_id": "allen_hmba_rhesus_bg_raw_counts",
        "population_count": len(population_members()),
        "traits_covered_downstream": list(TRAITS),
        "within_da": {
            "comparator": "pooled rest of DA",
            "minimum_common_donors": 3,
            "minimum_cells_per_donor_population": 10,
            "minimum_focal_detection": 0.05,
            "detection_difference_is_gate": False,
            "confidence_interval_is_rank_gate": False,
            "FDR_scope": "genome-wide within each focal population",
        },
        "brainwide": {
            "status": "available_cross_cohort_reference",
            "focal_dataset": "Allen HMBA rhesus taxonomy-defined DA",
            "reference_dataset": "Chiou adult rhesus whole-brain sci-RNA-seq3",
            "reference_scope": "all source regions after exact exclusion of every Chiou CL:0000700 dopaminergic cell",
            "donor_matching": False,
            "reference_weighting": "equal mean across available Chiou donor-region log2 CPM profiles",
            "minimum_focal_donors": 3,
            "minimum_cells_per_focal_donor": 10,
            "minimum_focal_detection": 0.05,
            "detection_difference_is_gate": False,
            "confidence_interval_is_rank_gate": False,
            "FDR_scope": "eligible strict direct FDA target genes within each focal population",
        },
        "inputs": {
            str(checkpoint.resolve()): sha256(checkpoint),
            str(ORTHOLOGS.resolve()): sha256(ORTHOLOGS),
            str(CHIOU_REFERENCE_MANIFEST.resolve()): sha256(
                CHIOU_REFERENCE_MANIFEST
            ),
        },
        "pipeline_files": {str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
    }
    atomic_json(manifest, RELEASE / "rhesus_hmba_selectivity_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "score", "effects", "axes", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        build_membership_and_pseudobulk()
    if args.stage in ("score", "all"):
        score_traits()
    if args.stage in ("effects", "all"):
        build_population_effects()
    if args.stage in ("axes", "all"):
        build_selectivity_axes()


if __name__ == "__main__":
    main()
