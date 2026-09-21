"""Structured citations, versions, and reuse terms for every atlas source."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import pandas as pd


EXPECTED_SOURCE_IDS = frozenset(
    {
        "kamath_da_2022", "siletti_whb_2023", "gaertner_oram_lrrk2_2025",
        "allen_mouse_wmb_2023", "allen_hmba_bg_2025",
        "chiou_rhesus_brainwide_2023", "scdrs_1_0_2",
        "magma_1_10", "drugsfda_2026_07_09", "purple_book_2026_06",
        "gsrs_2026_07_06", "openfda_labels_2026_07_10", "chembl_37",
        "gtopdb_2026_2", "drugcentral_2021_09_01", "b3db_75dab1cc",
        "open_targets_26_06", "ensembl_compara_116", "hgnc_2026_07_07",
    }
)

RANK_INPUT_ROLES = {
    "kamath_da_2022": "rank_driving_expression",
    "siletti_whb_2023": "rank_driving_expression",
    "gaertner_oram_lrrk2_2025": "rank_driving_expression",
    "allen_mouse_wmb_2023": "rank_driving_expression",
    "allen_hmba_bg_2025": "rank_driving_expression",
    "chiou_rhesus_brainwide_2023": "rank_driving_expression",
    "scdrs_1_0_2": "trait_population_selection",
    "magma_1_10": "trait_population_selection",
    "drugsfda_2026_07_09": "regulatory_denominator",
    "purple_book_2026_06": "regulatory_denominator",
    "gsrs_2026_07_06": "identity_mapping",
    "openfda_labels_2026_07_10": "supporting_evidence",
    "chembl_37": "supporting_evidence",
    "gtopdb_2026_2": "supporting_evidence",
    "drugcentral_2021_09_01": "supporting_evidence",
    "b3db_75dab1cc": "supporting_evidence",
    "open_targets_26_06": "supporting_evidence",
    "ensembl_compara_116": "orthology_mapping",
    "hgnc_2026_07_07": "identity_mapping",
}


def build_source_citations(release_id: str, records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = dict(record)
        row["release_id"] = release_id
        row["rank_input_role"] = RANK_INPUT_ROLES[str(row["source_id"])]
        row["release_tables_json"] = json.dumps(
            row.pop("release_tables"), separators=(",", ":")
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [
        "release_id", "source_id", "source_category", "source_name",
        "source_release", "citation", "doi", "pmid", "accession",
        "source_url", "license_name", "license_url", "license_scope",
        "reuse_status", "used_for", "release_tables_json",
        "rank_input_role", "affects_dual_selectivity_rank", "availability_status",
    ]
    return frame[columns].sort_values("source_id", kind="stable").reset_index(drop=True)


def validate_source_citations_semantics(frame: pd.DataFrame) -> None:
    if set(frame["source_id"].astype(str)) != set(EXPECTED_SOURCE_IDS):
        raise ValueError("source_citations does not contain the exact frozen source set")
    text_columns = [
        "source_name", "source_release", "citation", "source_url",
        "license_name", "license_url", "license_scope", "used_for",
    ]
    if frame[text_columns].fillna("").astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=None):
        raise ValueError("source_citations contains a blank required description")
    for column in ("source_url", "license_url"):
        invalid = frame[column].astype(str).map(
            lambda value: urlparse(value).scheme != "https"
            or not urlparse(value).netloc
        )
        if invalid.any():
            raise ValueError(f"source_citations contains an invalid HTTPS {column}")
    for value in frame["release_tables_json"].astype(str):
        try:
            tables = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("source_citations release_tables_json is invalid") from exc
        if not isinstance(tables, list) or not tables or not all(
            isinstance(table, str) and table for table in tables
        ):
            raise ValueError("source_citations requires a nonempty release table list")
    pending = frame["availability_status"].eq("pending_external")
    if set(frame.loc[pending, "source_id"]) != {"openfda_labels_2026_07_10"}:
        raise ValueError("source_citations pending sources differ from external gates")
    if frame.loc[pending, "reuse_status"].eq("open_license").any():
        raise ValueError("pending external sources cannot claim unrestricted acquisition")
    observed_roles = dict(zip(frame["source_id"], frame["rank_input_role"]))
    if observed_roles != RANK_INPUT_ROLES:
        raise ValueError("source_citations rank-input source classification changed")
    if frame["affects_dual_selectivity_rank"].astype(bool).any():
        raise ValueError("source citation rows cannot alter dual-selectivity rank")
