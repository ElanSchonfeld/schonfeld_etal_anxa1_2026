"""Contracts for the integrated human scDRS checkpoint."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = (
    "pd", "scz", "adhd", "nicotine_dep", "bipolar",
    "chronic_pain", "mdd", "oud", "ocd", "anxiety_any",
    "fibromyalgia",
)
STUDY_CELL_COUNTS = {"Kamath": 22_048, "Siletti": 823}
STUDY_CONDITION_CELL_COUNTS = {
    ("Kamath", "Ctrl"): 15_684,
    ("Kamath", "PD"): 2_715,
    ("Kamath", "LBD"): 3_649,
    ("Siletti", "Control"): 823,
}


def score_columns() -> list[str]:
    return [
        f"scdrs_{trait}_{kind}"
        for trait in TRAITS
        for kind in ("norm_score", "zscore")
    ]


def validate_integrated_score_table(
    scores: pd.DataFrame,
    observations: pd.DataFrame,
) -> None:
    """Require one complete score vector per trait across both human studies."""

    required = {"study", *score_columns()}
    if missing := sorted(required - set(scores.columns)):
        raise ValueError(f"integrated human scDRS scores lack columns: {missing}")
    if not scores.index.is_unique:
        raise ValueError("integrated human scDRS cell identifiers are not unique")
    if not observations.index.is_unique or "study" not in observations:
        raise ValueError("integrated human observations lack unique IDs or study")
    if not scores.index.equals(observations.index):
        raise ValueError("integrated human scDRS cell order or identity differs from H5AD")
    expected_study = observations["study"].astype(str)
    observed_study = scores["study"].astype(str)
    if not observed_study.equals(expected_study):
        raise ValueError("integrated human scDRS study identities differ from H5AD")
    if "disease_status" in observations:
        if "condition" not in scores:
            raise ValueError("integrated human scDRS scores lack condition identities")
        if not scores["condition"].astype(str).equals(
            observations["disease_status"].astype(str)
        ):
            raise ValueError("integrated human scDRS condition identities differ from H5AD")
    counts = observed_study.value_counts().to_dict()
    if counts != STUDY_CELL_COUNTS:
        raise ValueError(
            f"integrated human scDRS study counts are {counts}, expected {STUDY_CELL_COUNTS}"
        )
    values = scores[score_columns()].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("integrated human scDRS scores contain non-finite values")


def validate_score_manifest(
    manifest_path: Path,
    score_path: Path,
    *,
    expected_analysis_role: str = "primary",
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    validation = manifest.get("validation", {})
    if validation.get("raw_count_scoring_complete") is not True:
        raise ValueError("human scDRS raw-count scoring checkpoint is incomplete")
    if validation.get("integrated_human_analysis_count") != len(TRAITS):
        raise ValueError(
            f"human scDRS checkpoint does not contain exactly {len(TRAITS)} analyses"
        )
    source_mode = manifest.get("parameters", {}).get("source_mode", "recomputed")
    if source_mode == "existing_dopabase_human_score_artifact":
        required = {
            "complete_score_vector_count": len(TRAITS),
            "source_score_column_count": len(score_columns()),
            "cell_ids_and_order_exact": True,
            "csv_values_equal_embedded_float32_values": True,
            "all_scores_finite": True,
            "raw_integer_count_source_verified": True,
            "condition_split_scoring_used": False,
        }
        if any(validation.get(key) != value for key, value in required.items()):
            raise ValueError("imported human scDRS artifact validation is incomplete")
    elif expected_analysis_role == "duplicate_corrected_sensitivity":
        n_study_by_trait = len(STUDY_CELL_COUNTS) * len(TRAITS)
        if validation.get("completed_trait_checkpoint_count") != n_study_by_trait:
            raise ValueError(
                f"human duplicate-corrected checkpoint does not contain all "
                f"{n_study_by_trait} study-by-trait strata"
            )
        parameters = manifest.get("parameters", {})
        if (
            parameters.get("condition_split_scoring_used") is not False
            or validation.get("study_level_calibration_preserved") is not True
        ):
            raise ValueError(
                "human duplicate-corrected checkpoint does not preserve study-level "
                "scDRS calibration"
            )
    elif validation.get("completed_trait_checkpoint_count") != (
        len(STUDY_CONDITION_CELL_COUNTS) * len(TRAITS)
    ):
        n_strata = len(STUDY_CONDITION_CELL_COUNTS) * len(TRAITS)
        raise ValueError(
            f"human scDRS checkpoint does not contain all {n_strata} resumable strata"
        )
    if manifest.get("parameters", {}).get("analysis_design") != (
        "one integrated Kamath-plus-Siletti human result per trait"
    ):
        raise ValueError("human scDRS checkpoint has the wrong analysis design")
    observed_role = manifest.get("parameters", {}).get("analysis_role", "primary")
    if observed_role != expected_analysis_role:
        raise ValueError(
            f"human scDRS checkpoint role is {observed_role}, expected "
            f"{expected_analysis_role}"
        )
    output = manifest.get("output", {})
    if Path(str(output.get("path", ""))).expanduser().resolve() != Path(score_path).resolve():
        raise ValueError("human scDRS checkpoint points to a different score table")
    digest = hashlib.sha256()
    with Path(score_path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if output.get("sha256") != digest.hexdigest():
        raise ValueError("human scDRS checkpoint score-table hash is stale")
    inputs = manifest.get("inputs", {})
    if not inputs:
        raise ValueError("human scDRS checkpoint has no hashed source inputs")
    for path_text, record in inputs.items():
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise ValueError(f"human scDRS checkpoint source is missing: {path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"human scDRS checkpoint source size is stale: {path}")
        source_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                source_digest.update(block)
        if record.get("sha256") != source_digest.hexdigest():
            raise ValueError(f"human scDRS checkpoint source hash is stale: {path}")
    if source_mode == "existing_dopabase_human_score_artifact":
        return manifest
    checkpoints = manifest.get("trait_checkpoints", {})
    if expected_analysis_role == "duplicate_corrected_sensitivity":
        expected_checkpoints = {
            f"{study}|all_conditions|{trait}"
            for study in STUDY_CELL_COUNTS
            for trait in TRAITS
        }
    else:
        expected_checkpoints = {
            f"{study}|{condition}|{trait}"
            for study, condition in STUDY_CONDITION_CELL_COUNTS
            for trait in TRAITS
        }
    if set(checkpoints) != expected_checkpoints:
        raise ValueError("human scDRS checkpoint identities are incomplete or duplicated")
    for checkpoint_id, record in checkpoints.items():
        checkpoint_path = Path(str(record.get("manifest", ""))).expanduser()
        if not checkpoint_path.is_file():
            raise ValueError(f"human scDRS trait checkpoint is missing: {checkpoint_id}")
        checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if record.get("sha256") != checkpoint_digest:
            raise ValueError(f"human scDRS trait checkpoint hash is stale: {checkpoint_id}")
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("status") != "complete":
            raise ValueError(f"human scDRS trait checkpoint is incomplete: {checkpoint_id}")
        checkpoint_role = checkpoint.get("analysis_role", "primary")
        if checkpoint_role != expected_analysis_role:
            raise ValueError(
                f"human scDRS trait checkpoint role is stale: {checkpoint_id}"
            )
        observed_id = "|".join(
            str(checkpoint.get(field, ""))
            for field in ("study", "condition", "trait")
        )
        if observed_id != checkpoint_id:
            raise ValueError(f"human scDRS trait checkpoint identity is stale: {checkpoint_id}")
        checkpoint_output = checkpoint.get("output", {})
        checkpoint_output_path = Path(
            str(checkpoint_output.get("path", ""))
        ).expanduser()
        if not checkpoint_output_path.is_file():
            raise ValueError(f"human scDRS trait score is missing: {checkpoint_id}")
        output_digest = hashlib.sha256(checkpoint_output_path.read_bytes()).hexdigest()
        if checkpoint_output.get("sha256") != output_digest:
            raise ValueError(f"human scDRS trait score hash is stale: {checkpoint_id}")
    return manifest
