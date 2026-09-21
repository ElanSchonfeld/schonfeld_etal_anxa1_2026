import json

import pandas as pd
import pytest

from fda_atlas_v2.contracts import validate_table


def registry():
    common = {
        "release_id": "r", "assay": "snRNA-seq", "source_name": "source",
        "source_release": "v", "source_paths_json": json.dumps(["/data/source"]),
        "source_manifest_path": "/release/manifest.json",
        "source_manifest_sha256": "a" * 64,
        "matrix_representation": "raw counts",
        "normalization_or_transformation": "donor pseudobulk",
        "n_cells": 100, "n_features": 20, "rank_cohort_n_cells": 80,
        "rank_cohort_n_donors": 4, "availability_status": "available_verified",
        "identity_role": "source identity", "sha256": "b" * 64,
        "checksum_status": "manifest_recorded_content_sha256",
    }
    rows = []
    for species in ("human", "mouse"):
        within = dict(common, dataset_id=f"{species}_da", species=species,
                      dataset_role="integrated_da_atlas", anatomy_scope="DA",
                      rank_role="primary_within_da")
        if species == "human":
            within.update(rank_cohort_n_cells=16_507, rank_cohort_n_donors=11,
                          identity_role="Kamath-grounded integrated Kamath-Siletti controls")
        rows.append(within)
        rows.append(dict(common, dataset_id=f"{species}_brain", species=species,
                         dataset_role="whole_brain_comparator", anatomy_scope="brain",
                         rank_role="primary_brainwide"))
    rows.append(dict(common, dataset_id="allen_hmba_rhesus_bg_raw_counts", species="macaque",
                     dataset_role="focal_da_atlas", anatomy_scope="SNpc",
                     rank_cohort_n_cells=3_368, rank_cohort_n_donors=4,
                     rank_role="primary_within_da",
                     identity_role="Allen HMBA NCBITaxon:9544 Macaca mulatta taxonomy-defined DA and HMoE populations"))
    rows.append(dict(common, dataset_id="chiou_rhesus_brainwide_scrnaseq3", species="macaque",
                     dataset_role="whole_brain_comparator", anatomy_scope="brain",
                     rank_cohort_n_cells=2_580_322, rank_cohort_n_donors=5,
                     rank_role="primary_brainwide",
                     identity_role="cross-cohort reference with all 3,645 exact source DA cells excluded"))
    return pd.DataFrame(rows)


def test_dataset_registry_preserves_primary_roles_and_pending_truth():
    validate_table("dataset_registry", registry())


def test_pending_dataset_cannot_claim_counts_or_checksum():
    frame = registry()
    pending = frame.iloc[[0]].copy()
    pending["dataset_id"] = "future_pending_dataset"
    pending["availability_status"] = "pending_external_release"
    pending["source_paths_json"] = "[]"
    pending["n_cells"] = 1
    pending["n_features"] = pd.NA
    pending["sha256"] = ""
    pending["checksum_status"] = "unavailable_pending_release"
    frame = pd.concat([frame, pending], ignore_index=True)
    with pytest.raises(ValueError, match="cannot claim unverified dimensions"):
        validate_table("dataset_registry", frame)


def test_human_primary_within_da_must_be_kamath_grounded():
    frame = registry()
    frame.loc[frame["dataset_id"].eq("human_da"), "identity_role"] = "Siletti only"
    with pytest.raises(ValueError, match="not the Kamath-grounded integration"):
        validate_table("dataset_registry", frame)


def test_hmba_rhesus_source_requires_four_donors_and_exact_taxon_identity():
    frame = registry()
    mask = frame["dataset_id"].eq("allen_hmba_rhesus_bg_raw_counts")
    frame.loc[mask, "rank_cohort_n_donors"] = 1
    with pytest.raises(ValueError, match="strict rhesus HMBA"):
        validate_table("dataset_registry", frame)

    frame = registry()
    frame.loc[mask, "identity_role"] = "generic macaque"
    with pytest.raises(ValueError, match="strict rhesus HMBA"):
        validate_table("dataset_registry", frame)
