"""B3DB CNS exposure evidence at drug grain."""

from __future__ import annotations

import json

import pandas as pd
from rdkit import Chem

from .contracts import TRAITS, validate_table


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare_b3db_records(
    classification: pd.DataFrame,
    regression: pd.DataFrame,
) -> pd.DataFrame:
    required = {"NO.", "compound_name", "CID", "logBB", "Inchi", "reference", "group"}
    if required - set(classification) or required - set(regression):
        raise ValueError("B3DB inputs lack required provenance")
    if "BBB+/BBB-" not in classification:
        raise ValueError("B3DB classification labels are absent")

    blocks = []
    for evidence_type, frame in (
        ("curated_bbb_classification", classification),
        ("measured_numeric_logbb", regression),
    ):
        block = frame.copy()
        block["b3db_record_number"] = pd.to_numeric(block["NO."], errors="raise").astype(int)
        block["inchikey"] = block["Inchi"].map(Chem.InchiToInchiKey)
        if block["inchikey"].eq("").any() or block["inchikey"].isna().any():
            raise ValueError("B3DB contains an InChI that cannot produce an InChIKey")
        if block["inchikey"].duplicated().any():
            raise ValueError(f"B3DB {evidence_type} InChIKey is duplicated")
        block["evidence_type"] = evidence_type
        for column in ("compound_name", "CID", "reference", "group", "comments"):
            block[column] = block[column].fillna("").astype(str)
        block["logBB"] = pd.to_numeric(block["logBB"], errors="raise")
        block["bbb_label"] = (
            block["BBB+/BBB-"].astype(str)
            if evidence_type == "curated_bbb_classification" else ""
        )
        if evidence_type == "curated_bbb_classification" and not set(
            block["bbb_label"]
        ).issubset({"BBB+", "BBB-"}):
            raise ValueError("B3DB classification vocabulary changed")
        block["source_record_id"] = block["b3db_record_number"].map(
            lambda value: f"b3db_{'classification' if evidence_type.startswith('curated') else 'regression'}:{value}"
        )
        blocks.append(
            block[
                [
                    "evidence_type", "source_record_id", "b3db_record_number",
                    "compound_name", "inchikey", "CID", "logBB", "bbb_label",
                    "group", "reference", "comments",
                ]
            ]
        )
    records = pd.concat(blocks, ignore_index=True)
    numeric_keys = set(
        records.loc[records["evidence_type"].eq("measured_numeric_logbb"), "inchikey"]
    )
    categorical_a = set(
        records.loc[
            records["evidence_type"].eq("curated_bbb_classification")
            & records["group"].eq("A"),
            "inchikey",
        ]
    )
    if numeric_keys != categorical_a:
        raise ValueError("B3DB numeric records no longer equal categorical group A")
    return records.sort_values(
        ["inchikey", "evidence_type", "b3db_record_number"], kind="stable"
    ).reset_index(drop=True)


def build_b3db_entity_evidence(
    release_id: str,
    records: pd.DataFrame,
    structures: pd.DataFrame,
    entity_edges: pd.DataFrame,
    snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if {"id", "name", "inchikey"} - set(structures):
        raise ValueError("DrugCentral structures lack exact chemical identity")
    edge_required = {
        "entity_type", "entity_id", "drugcentral_struct_id", "identity_status"
    }
    if edge_required - set(entity_edges):
        raise ValueError("DrugCentral entity bridge lacks required identity provenance")
    structure_keys = structures[["id", "name", "inchikey"]].dropna(subset=["inchikey"])
    if structure_keys["inchikey"].duplicated().any():
        raise ValueError("DrugCentral InChIKey is not unique")
    structure_keys = structure_keys.rename(
        columns={"id": "drugcentral_struct_id", "name": "drugcentral_name"}
    )
    matches = records.merge(structure_keys, on="inchikey", how="inner", validate="many_to_one")
    matches["match_method"] = "exact_stereochemical_inchikey"
    matches["source_snapshot_id"] = snapshot_id
    matched_entities = matches.merge(
        entity_edges[
            ["entity_type", "entity_id", "drugcentral_struct_id", "identity_status"]
        ].drop_duplicates(),
        on="drugcentral_struct_id",
        how="inner",
        validate="many_to_many",
    )
    matched_entities["release_id"] = release_id
    matched_entities["affects_dual_selectivity_rank"] = False
    entity_key = ["entity_type", "entity_id", "evidence_type", "source_record_id"]
    if matched_entities.duplicated(entity_key).any():
        raise ValueError("B3DB entity evidence is duplicated")
    match_columns = [
        "source_snapshot_id", "evidence_type", "source_record_id",
        "b3db_record_number", "compound_name", "inchikey", "CID", "logBB",
        "bbb_label", "group", "reference", "comments", "drugcentral_struct_id",
        "drugcentral_name", "match_method",
    ]
    evidence_columns = [
        "release_id", "entity_type", "entity_id", *match_columns,
        "identity_status", "affects_dual_selectivity_rank",
    ]
    return (
        matches[match_columns].sort_values(
            ["drugcentral_struct_id", "evidence_type"], kind="stable"
        ).reset_index(drop=True),
        matched_entities[evidence_columns].sort_values(entity_key, kind="stable").reset_index(drop=True),
    )


def materialize_b3db_contexts(
    release_id: str,
    entity_evidence: pd.DataFrame,
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat trait-independent CNS evidence across target portfolios and traits."""

    pairs = pd.concat(
        [
            moiety_edges[["moiety_id", "target_id"]]
            .rename(columns={"moiety_id": "entity_id"})
            .assign(entity_type="active_moiety"),
            substance_edges[["substance_id", "target_id"]]
            .rename(columns={"substance_id": "entity_id"})
            .assign(entity_type="regulatory_substance"),
        ],
        ignore_index=True,
    )[["entity_type", "entity_id", "target_id"]].drop_duplicates()
    portfolios = {
        key: sorted(group["target_id"].astype(str).unique())
        for key, group in pairs.groupby(["entity_type", "entity_id"], sort=True)
    }
    joined = entity_evidence.merge(
        pairs, on=["entity_type", "entity_id"], how="inner", validate="many_to_many"
    )
    rows = []
    for record in joined.itertuples(index=False):
        entity_key = (str(record.entity_type), str(record.entity_id))
        numeric = str(record.evidence_type) == "measured_numeric_logbb"
        if numeric:
            value = (
                f"logBB={float(record.logBB):.3f} (log10 brain/blood concentration ratio); "
                f"B3DB numeric group {record.group}"
            )
            status = "available_measured_logbb"
        else:
            value = f"{record.bbb_label}; B3DB categorical group {record.group}"
            status = "available_curated_bbb_classification"
        identity_path = [
            {
                "b3db_inchikey": str(record.inchikey),
                "b3db_record_id": str(record.source_record_id),
                "drugcentral_struct_id": int(record.drugcentral_struct_id),
                "match_method": str(record.match_method),
                "source_snapshot_id": str(record.source_snapshot_id),
            }
        ]
        for trait_id in sorted(TRAITS):
            rows.append(
                {
                    "release_id": release_id,
                    "entity_type": str(record.entity_type),
                    "entity_id": str(record.entity_id),
                    "target_id": str(record.target_id),
                    "trait_id": trait_id,
                    "context_type": "cns_exposure",
                    "evidence_source": str(record.source_snapshot_id),
                    "value": value,
                    "direction": "unknown",
                    "status": status,
                    "source_record_id": str(record.source_record_id),
                    "missing_reason": "",
                    "label_id": "",
                    "label_section_source_record_id": "",
                    "label_section_name": "",
                    "identity_path_json": _canonical_json(identity_path),
                    "polypharmacology_target_ids_json": _canonical_json(portfolios[entity_key]),
                    "adjudication_method": (
                        f"exact_stereochemical_inchikey:b3db_{record.evidence_type}_group_{record.group}"
                    ),
                    "direction_evidence_source": "",
                    "direction_known": False,
                    "affects_dual_selectivity_rank": False,
                }
            )
    output = pd.DataFrame(rows)
    validate_table("drug_target_trait_contexts", output)
    return output.sort_values(
        ["trait_id", "entity_type", "entity_id", "target_id", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)
