"""DrugCentral context evidence at drug grain."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .contracts import TRAITS, validate_table


TRAIT_PATTERNS = {
    "pd": r"parkinson",
    "scz": r"schizo",
    "adhd": r"attention.deficit|hyperactiv",
    "nicotine_dep": r"nicotine|tobacco|smoking|cigarette",
    "bipolar": r"bipolar|manic",
    "chronic_pain": r"chronic pain|neuropathic pain|persistent pain",
    "fibromyalgia": r"fibromyalgia|fibrositis",
    "mdd": r"major depress|depressive disorder|depression",
    "oud": r"opioid.*(?:depend|use disorder|addict)|opiate.*(?:depend|use disorder|addict)",
    "ocd": r"obsessive.compulsive",
    "anxiety_any": r"anxiety|anxious|panic disorder|social phobia|generalized anxiety",
}


def _identifier(*values: object) -> str:
    payload = "\x1f".join(map(str, values)).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_drugcentral_trait_candidates(
    relationships: pd.DataFrame,
    concept_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Classify every lexical concept candidate before trait attribution."""

    required = {
        "context_relationship_id", "entity_type", "entity_id",
        "drugcentral_struct_id", "drugcentral_name", "identity_status",
        "drug_fda_approval_status", "relationship_name", "concept_id",
        "concept_name", "umls_cui", "snomed_full_name", "snomed_conceptid",
        "jurisdiction_status", "source_snapshot_id", "source_record_id",
    }
    registry_required = {
        "trait_id", "concept_id", "expected_concept_name", "mapping_tier",
        "mapping_rationale",
    }
    if required - set(relationships) or registry_required - set(concept_registry):
        raise ValueError("DrugCentral trait candidate inputs lack required provenance")
    if set(concept_registry["trait_id"].astype(str)) - set(TRAITS):
        raise ValueError("DrugCentral concept registry contains an unknown trait")
    if concept_registry.duplicated(["trait_id", "concept_id"]).any():
        raise ValueError("DrugCentral concept registry key is duplicated")
    if not set(concept_registry["mapping_tier"]).issubset(
        {"exact_trait", "nested_subtype", "related_construct"}
    ):
        raise ValueError("DrugCentral concept registry has an unsupported mapping tier")
    observed_names = relationships[["concept_id", "concept_name"]].drop_duplicates()
    registry_check = concept_registry.merge(
        observed_names,
        on="concept_id",
        how="left",
    )
    name_match = registry_check["expected_concept_name"].astype(str).str.casefold().eq(
        registry_check["concept_name"].fillna("").astype(str).str.casefold()
    )
    if not name_match.groupby(
        [registry_check["trait_id"], registry_check["concept_id"]]
    ).any().all():
        raise ValueError("DrugCentral curated concept identity or name is stale")
    registry = concept_registry.set_index(["trait_id", "concept_id"]).to_dict("index")
    text = relationships[["concept_name", "snomed_full_name"]].fillna("").astype(str).agg(
        " | ".join, axis=1
    )
    blocks = []
    found_curated = set()
    for trait_id, pattern in TRAIT_PATTERNS.items():
        block = relationships[text.str.contains(pattern, case=False, regex=True)].copy()
        block["trait_id"] = trait_id
        tiers = []
        rationales = []
        admitted = []
        for row in block.itertuples(index=False):
            key = (trait_id, int(row.concept_id))
            record = registry.get(key)
            if record is None:
                tiers.append("lexical_match_not_admitted")
                rationales.append(
                    "Lexical overlap alone is insufficient for the frozen trait construct"
                )
                admitted.append(False)
            else:
                tiers.append(str(record["mapping_tier"]))
                rationales.append(str(record["mapping_rationale"]))
                admitted.append(True)
                found_curated.add(key)
        block["mapping_tier"] = tiers
        block["mapping_rationale"] = rationales
        block["admitted_to_target_trait_context"] = admitted
        block["trait_candidate_id"] = block.apply(
            lambda row: "drugcentral_trait_candidate:"
            + _identifier(row["trait_id"], row["context_relationship_id"]),
            axis=1,
        )
        block["affects_dual_selectivity_rank"] = False
        blocks.append(block)
    missing_curated = set(registry) - found_curated
    if missing_curated:
        raise ValueError(f"DrugCentral curated concepts missed lexical audit: {missing_curated}")
    output = pd.concat(blocks, ignore_index=True)
    if output.duplicated("trait_candidate_id").any():
        raise ValueError("DrugCentral trait candidate identity is duplicated")
    return output.sort_values(
        ["trait_id", "mapping_tier", "relationship_name", "concept_name", "entity_type", "entity_id"],
        kind="stable",
    ).reset_index(drop=True)


def materialize_drugcentral_trait_contexts(
    release_id: str,
    candidates: pd.DataFrame,
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Expand admitted drug-grain concepts across a complete target portfolio."""

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
    admitted = candidates[candidates["admitted_to_target_trait_context"].astype(bool)].copy()
    joined = admitted.merge(
        pairs, on=["entity_type", "entity_id"], how="inner", validate="many_to_many"
    )
    context_type = {
        "indication": "structured_indication_jurisdiction_unresolved",
        "contraindication": "structured_contraindication_jurisdiction_unresolved",
        "off-label use": "structured_off_label_use",
    }
    joined = joined[joined["relationship_name"].isin(context_type)].copy()
    rows = []
    for row in joined.itertuples(index=False):
        entity_key = (str(row.entity_type), str(row.entity_id))
        identity_path = [
            {
                "context_relationship_id": str(row.context_relationship_id),
                "drugcentral_struct_id": int(row.drugcentral_struct_id),
                "identity_status": str(row.identity_status),
                "source_snapshot_id": str(row.source_snapshot_id),
            }
        ]
        rows.append(
            {
                "release_id": release_id,
                "entity_type": str(row.entity_type),
                "entity_id": str(row.entity_id),
                "target_id": str(row.target_id),
                "trait_id": str(row.trait_id),
                "context_type": context_type[str(row.relationship_name)],
                "evidence_source": str(row.source_snapshot_id),
                "value": (
                    f"{row.concept_name} [DrugCentral concept {int(row.concept_id)}; "
                    f"{row.jurisdiction_status}]"
                ),
                "direction": "unknown",
                "status": (
                    "available_supporting_off_label"
                    if str(row.relationship_name) == "off-label use"
                    else "available_supporting_jurisdiction_unresolved"
                ),
                "source_record_id": str(row.source_record_id),
                "missing_reason": "",
                "label_id": "",
                "label_section_source_record_id": "",
                "label_section_name": "",
                "identity_path_json": _canonical_json(identity_path),
                "polypharmacology_target_ids_json": _canonical_json(portfolios[entity_key]),
                "adjudication_method": (
                    f"curated_drugcentral_concept_registry:{row.mapping_tier}"
                ),
                "direction_evidence_source": "",
                "direction_known": False,
                "affects_dual_selectivity_rank": False,
            }
        )
    output = pd.DataFrame(rows)
    validate_table("drug_target_trait_contexts", output)
    return output.sort_values(
        ["trait_id", "entity_type", "entity_id", "target_id", "context_type", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_drugcentral_context_bridge(
    release_id: str,
    source_frames: list[tuple[str, str, pd.DataFrame]],
    structures: pd.DataFrame,
    synonyms: pd.DataFrame,
    approvals: pd.DataFrame,
    relationships: pd.DataFrame,
    snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = []
    for entity_type, entity_column, frame in source_frames:
        required = {entity_column, "source_database", "source_drug_id", "source_drug_name"}
        if required - set(frame):
            raise ValueError(f"DrugCentral {entity_type} source bridge lacks columns")
        selected = frame[frame["source_database"].astype(str).eq("DrugCentral")][
            [entity_column, "source_drug_id", "source_drug_name"]
        ].drop_duplicates()
        selected = selected.rename(columns={entity_column: "entity_id"})
        selected["entity_type"] = entity_type
        source_rows.append(selected)
    source = pd.concat(source_rows, ignore_index=True).drop_duplicates()
    source["drugcentral_struct_id"] = pd.to_numeric(
        source["source_drug_id"], errors="coerce"
    ).astype("Int64")
    if source["drugcentral_struct_id"].isna().any():
        raise ValueError("DrugCentral source IDs are not integer structure IDs")
    identities = structures[["id", "name"]].rename(
        columns={"id": "drugcentral_struct_id", "name": "drugcentral_name"}
    )
    source = source.merge(
        identities, on="drugcentral_struct_id", how="left", validate="many_to_one"
    )
    if source["drugcentral_name"].isna().any():
        raise ValueError("DrugCentral source ID is absent from pinned structures")
    synonym_keys = set(
        zip(
            synonyms["id"].dropna().astype(int),
            synonyms.loc[synonyms["id"].notna(), "name"].astype(str).str.strip().str.casefold(),
        )
    )
    current_name_match = source["source_drug_name"].astype(str).str.strip().str.casefold().eq(
        source["drugcentral_name"].astype(str).str.strip().str.casefold()
    )
    synonym_match = pd.Series(
        [
            (int(row.drugcentral_struct_id), str(row.source_drug_name).strip().casefold())
            in synonym_keys
            for row in source.itertuples(index=False)
        ],
        index=source.index,
    )
    source["identity_status"] = "source_id_name_mismatch"
    source.loc[synonym_match, "identity_status"] = "exact_source_id_synonym"
    source.loc[current_name_match, "identity_status"] = "exact_source_id_current_name"
    if source["identity_status"].eq("source_id_name_mismatch").any():
        raise ValueError("DrugCentral source ID and name cannot be jointly verified")
    fda_ids = set(
        approvals.loc[approvals["type"].astype(str).eq("FDA"), "struct_id"]
        .dropna().astype(int)
    )
    source["drug_fda_approval_status"] = source["drugcentral_struct_id"].map(
        lambda value: "drug_has_fda_approval" if int(value) in fda_ids
        else "fda_approval_not_recorded"
    )
    source["release_id"] = release_id
    source["source_snapshot_id"] = snapshot_id
    source["source_record_id"] = source["drugcentral_struct_id"].map(
        lambda value: f"drugcentral_structure:{int(value)}"
    )
    source["affects_dual_selectivity_rank"] = False
    source["drugcentral_context_entity_edge_id"] = source.apply(
        lambda row: "drugcentral_context_edge:"
        + _identifier(
            row["entity_type"], row["entity_id"], row["drugcentral_struct_id"]
        ),
        axis=1,
    )
    edge_columns = [
        "release_id", "drugcentral_context_entity_edge_id", "entity_type",
        "entity_id", "drugcentral_struct_id", "source_drug_name",
        "drugcentral_name", "identity_status", "drug_fda_approval_status",
        "source_snapshot_id", "source_record_id", "affects_dual_selectivity_rank",
    ]
    edges = source[edge_columns].drop_duplicates().sort_values(
        ["entity_type", "entity_id", "drugcentral_struct_id"], kind="stable"
    ).reset_index(drop=True)
    if edges.duplicated("drugcentral_context_entity_edge_id").any():
        raise ValueError("DrugCentral context entity edge identity is duplicated")

    relationship_columns = [
        "id", "struct_id", "concept_id", "relationship_name", "concept_name",
        "umls_cui", "snomed_full_name", "cui_semantic_type", "snomed_conceptid",
    ]
    relations = edges.merge(
        relationships[relationship_columns],
        left_on="drugcentral_struct_id",
        right_on="struct_id",
        how="inner",
        validate="many_to_many",
    )
    relations["jurisdiction_status"] = "relationship_jurisdiction_unresolved"
    relations["context_relationship_id"] = relations.apply(
        lambda row: "drugcentral_context_relationship:"
        + _identifier(row["entity_type"], row["entity_id"], row["id"]),
        axis=1,
    )
    relations["source_record_id"] = relations["id"].map(
        lambda value: f"drugcentral_omop_relationship:{int(value)}"
    )
    relations["affects_dual_selectivity_rank"] = False
    relation_columns = [
        "release_id", "context_relationship_id", "entity_type", "entity_id",
        "drugcentral_struct_id", "drugcentral_name", "identity_status",
        "drug_fda_approval_status", "relationship_name", "concept_id",
        "concept_name", "umls_cui", "snomed_full_name", "cui_semantic_type",
        "snomed_conceptid", "jurisdiction_status", "source_snapshot_id",
        "source_record_id", "affects_dual_selectivity_rank",
    ]
    relations = relations[relation_columns].sort_values(
        ["entity_type", "entity_id", "relationship_name", "concept_id", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)
    if relations.duplicated("context_relationship_id").any():
        raise ValueError("DrugCentral context relationship identity is duplicated")
    if edges["affects_dual_selectivity_rank"].any() or relations[
        "affects_dual_selectivity_rank"
    ].any():
        raise ValueError("DrugCentral context evidence cannot alter rank")
    return edges, relations
