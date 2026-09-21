"""Semantic validation for the source datasets that define atlas v2."""

from __future__ import annotations

import json

import pandas as pd


PRIMARY_RANK_ROLES = {
    "human": {"primary_within_da", "primary_brainwide"},
    "mouse": {"primary_within_da", "primary_brainwide"},
    "macaque": {"primary_within_da", "primary_brainwide"},
}


def validate_dataset_registry_semantics(frame: pd.DataFrame) -> None:
    """Fail closed on invented availability, counts, or rank-driving roles."""

    if frame.empty:
        raise ValueError("dataset_registry cannot be empty")
    for value in frame["source_paths_json"].astype(str):
        try:
            paths = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("dataset_registry source_paths_json is invalid") from exc
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("dataset_registry source_paths_json must be a string list")

    available = frame["availability_status"].isin(
        ["available_verified", "inputs_ready_resource_gated"]
    )
    counts = frame.loc[available, ["n_cells", "n_features"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if counts.isna().any(axis=None) or (counts <= 0).any(axis=None):
        raise ValueError("available dataset_registry rows require positive source dimensions")

    pending = frame["availability_status"].eq("pending_external_release")
    if frame.loc[pending, ["n_cells", "n_features"]].notna().any(axis=None):
        raise ValueError("pending external datasets cannot claim unverified dimensions")
    if frame.loc[pending, "sha256"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("pending external datasets cannot claim a content checksum")
    if frame.loc[pending, "source_paths_json"].astype(str).ne("[]").any():
        raise ValueError("pending external datasets cannot claim acquired source paths")

    checksummed = frame["checksum_status"].isin(
        [
            "manifest_recorded_content_sha256",
            "manifest_recorded_component_set_sha256",
        ]
    )
    valid_digest = frame.loc[checksummed, "sha256"].fillna("").astype(str).str.fullmatch(
        r"[0-9a-f]{64}"
    )
    if not valid_digest.all():
        raise ValueError("recorded dataset content checksums must be lowercase SHA-256")
    unchecksummed = frame["checksum_status"].isin(
        ["component_content_sha256_not_computed_resource_gate",
         "component_content_sha256_not_recomputed",
         "unavailable_pending_release"]
    )
    if frame.loc[unchecksummed, "sha256"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("dataset rows without a content checksum must leave sha256 blank")

    for species, required_roles in PRIMARY_RANK_ROLES.items():
        observed = set(frame.loc[frame["species"].eq(species), "rank_role"].astype(str))
        missing = required_roles - observed
        if missing:
            raise ValueError(
                f"dataset_registry {species} is missing rank roles: {sorted(missing)}"
            )

    macaque = frame[frame["species"].eq("macaque")]
    hmba = macaque[
        macaque["dataset_id"].eq("allen_hmba_rhesus_bg_raw_counts")
    ]
    if (
        len(hmba) != 1
        or hmba.iloc[0]["dataset_role"] != "focal_da_atlas"
        or hmba.iloc[0]["rank_role"] != "primary_within_da"
        or int(hmba.iloc[0]["rank_cohort_n_cells"]) != 3_368
        or int(hmba.iloc[0]["rank_cohort_n_donors"]) != 4
        or "Macaca mulatta" not in str(hmba.iloc[0]["identity_role"])
        or "NCBITaxon:9544" not in str(hmba.iloc[0]["identity_role"])
    ):
        raise ValueError("dataset_registry requires strict rhesus HMBA within-DA source")
    rhesus = macaque[
        macaque["dataset_id"].eq("chiou_rhesus_brainwide_scrnaseq3")
    ]
    if (
        len(rhesus) != 1
        or rhesus.iloc[0]["dataset_role"] != "whole_brain_comparator"
        or rhesus.iloc[0]["rank_role"] != "primary_brainwide"
        or rhesus.iloc[0]["availability_status"] != "available_verified"
        or int(rhesus.iloc[0]["rank_cohort_n_cells"]) != 2_580_322
        or int(rhesus.iloc[0]["rank_cohort_n_donors"]) != 5
        or "3,645" not in str(rhesus.iloc[0]["identity_role"])
        or "cross-cohort" not in str(rhesus.iloc[0]["identity_role"]).lower()
    ):
        raise ValueError("dataset_registry requires the verified Chiou non-DA reference")

    human = frame[
        frame["species"].eq("human")
        & frame["rank_role"].eq("primary_within_da")
    ]
    if len(human) != 1:
        raise ValueError("dataset_registry requires one human primary within-DA dataset")
    row = human.iloc[0]
    if (
        int(row["rank_cohort_n_cells"]) != 16_507
        or int(row["rank_cohort_n_donors"]) != 11
        or "Kamath" not in str(row["identity_role"])
        or "Siletti" not in str(row["identity_role"])
        or "integrated" not in str(row["identity_role"]).lower()
    ):
        raise ValueError(
            "human primary within-DA provenance is not the Kamath-grounded integration"
        )
