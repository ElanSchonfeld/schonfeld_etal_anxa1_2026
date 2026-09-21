"""Preserve source-reported DrugCentral activity without cross-assay conversion."""

from __future__ import annotations

import hashlib

import pandas as pd


def _stable_id(prefix: str, *values: object) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return f"{prefix}:{hashlib.sha1(payload.encode()).hexdigest()[:16]}"


def reconstruct_drugcentral_activity(interactions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "DRUG_NAME", "STRUCT_ID", "TARGET_NAME", "ACCESSION", "GENE",
        "ACT_VALUE", "ACT_UNIT", "ACT_TYPE", "ACT_COMMENT", "ACT_SOURCE",
        "RELATION", "ACT_SOURCE_URL", "ACTION_TYPE", "ORGANISM",
    }
    if missing := sorted(required - set(interactions.columns)):
        raise ValueError(f"DrugCentral activity source is missing columns: {missing}")
    columns = list(interactions.columns)
    rows = []
    for record in interactions[columns].drop_duplicates().to_dict("records"):
        symbols = [
            value.strip() for value in str(record["GENE"]).split("|") if value.strip()
        ] or [""]
        for symbol in symbols:
            rows.append(
                {
                    "source_record_id": _stable_id(
                        "DRUGCENTRAL_EDGE",
                        *(record[column] for column in columns),
                        symbol,
                    ),
                    "source_drug_id_raw": str(record["STRUCT_ID"]),
                    "source_drug_name_raw": str(record["DRUG_NAME"]),
                    "source_target_id_raw": str(record["ACCESSION"]),
                    "source_target_name_raw": str(record["TARGET_NAME"]),
                    "reported_gene_symbol": symbol,
                    "activity_endpoint_type": str(record["ACT_TYPE"]),
                    "reported_activity_value": str(record["ACT_VALUE"]),
                    "reported_unit": str(record["ACT_UNIT"]),
                    "activity_relation": str(record["RELATION"]),
                    "activity_source": str(record["ACT_SOURCE"]),
                    "activity_source_url": str(record["ACT_SOURCE_URL"]),
                    "activity_comment": str(record["ACT_COMMENT"]),
                    "source_action_type_raw": str(record["ACTION_TYPE"]),
                    "target_organism_raw": str(record["ORGANISM"]),
                }
            )
    result = pd.DataFrame(rows)
    if result["source_record_id"].duplicated().any():
        raise ValueError("DrugCentral reconstructed source IDs are not unique")
    numeric = pd.to_numeric(result["reported_activity_value"], errors="coerce")
    invalid = result["reported_activity_value"].ne("") & numeric.isna()
    if invalid.any():
        raise ValueError("DrugCentral has a nonnumeric reported activity value")
    result["activity_value_numeric"] = numeric
    return result


def build_drug_target_activity_evidence(
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    release_id: str,
) -> pd.DataFrame:
    source = reconstruct_drugcentral_activity(interactions)
    blocks = []
    for entity_type, entity_column, edges in (
        ("active_moiety", "moiety_id", moiety_edges),
        ("regulatory_substance", "substance_id", substance_edges),
    ):
        required = {
            entity_column, "target_id", "human_gene", "source_record_id",
            "source_database", "source_release", "source_drug_id",
            "source_drug_name", "source_target_id", "source_target_name",
            "target_organism", "action_type", "target_scope", "evidence_kind",
        }
        if missing := sorted(required - set(edges.columns)):
            raise ValueError(f"{entity_type} target edges are missing: {missing}")
        selected = edges[edges["source_database"].eq("DrugCentral")].copy()
        joined = selected.merge(
            source,
            on="source_record_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{entity_type} DrugCentral activity records do not all join")
        identity_checks = {
            "source_drug_id": "source_drug_id_raw",
            "source_drug_name": "source_drug_name_raw",
            "source_target_id": "source_target_id_raw",
            "source_target_name": "source_target_name_raw",
            "target_organism": "target_organism_raw",
        }
        for edge_column, source_column in identity_checks.items():
            if not joined[edge_column].fillna("").astype(str).eq(
                joined[source_column].fillna("").astype(str)
            ).all():
                raise ValueError(f"DrugCentral activity identity differs: {edge_column}")
        joined["release_id"] = release_id
        joined["entity_type"] = entity_type
        joined["entity_id"] = joined[entity_column].astype(str)
        joined["activity_availability_status"] = joined[
            "reported_activity_value"
        ].ne("").map({True: "source_reported", False: "not_reported_by_source"})
        joined["activity_scale"] = joined["reported_activity_value"].ne("").map(
            {True: "source_reported_unitless", False: "not_reported"}
        )
        if joined["reported_unit"].ne("").any():
            raise ValueError("DrugCentral activity units changed from the pinned release")
        joined["comparability_status"] = "not_harmonized_across_assays"
        joined["affects_dual_selectivity_rank"] = False
        blocks.append(joined)

    result = pd.concat(blocks, ignore_index=True)
    columns = [
        "release_id", "entity_type", "entity_id", "target_id", "human_gene",
        "source_record_id", "source_database", "source_release", "source_drug_id",
        "source_drug_name", "source_target_id", "source_target_name",
        "target_organism", "activity_endpoint_type", "reported_activity_value",
        "activity_value_numeric", "reported_unit", "activity_relation",
        "activity_source", "activity_source_url", "activity_comment",
        "activity_availability_status", "activity_scale", "comparability_status",
        "action_type", "target_scope", "evidence_kind",
        "affects_dual_selectivity_rank",
    ]
    result = result[columns].sort_values(
        ["entity_type", "entity_id", "target_id", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)
    if result.duplicated(
        ["release_id", "entity_type", "entity_id", "target_id", "source_record_id"]
    ).any():
        raise ValueError("DrugCentral activity evidence primary key is duplicated")
    return result
