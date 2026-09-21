"""Normalize matched openFDA labels without attributing drug text to gene targets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import pandas as pd

from .contracts import validate_table


SECTION_CATEGORIES = {
    "indications_and_usage": "indication_or_use",
    "purpose": "indication_or_use",
    "mechanism_of_action": "mechanism_or_pharmacology",
    "clinical_pharmacology": "mechanism_or_pharmacology",
    "boxed_warning": "safety_or_use_constraint",
    "contraindications": "safety_or_use_constraint",
    "warnings": "safety_or_use_constraint",
    "warnings_and_cautions": "safety_or_use_constraint",
    "adverse_reactions": "safety_or_use_constraint",
    "drug_interactions": "safety_or_use_constraint",
    "information_for_patients": "safety_or_use_constraint",
}


OPENFDA_FIELDS = (
    "application_number",
    "brand_name",
    "generic_name",
    "substance_name",
    "unii",
    "product_ndc",
    "spl_set_id",
    "route",
    "product_type",
    "manufacturer_name",
    "pharm_class_epc",
    "pharm_class_moa",
)

LABEL_COLUMNS = (
    "release_id", "label_id", "set_id", "label_version", "effective_time",
    "source_release", "source_partition", "source_record_sha256",
    "affects_dual_selectivity_rank",
    *(f"{field}_json" for field in OPENFDA_FIELDS),
)
EDGE_COLUMNS = (
    "release_id", "label_entity_edge_id", "label_id", "entity_type", "entity_id",
    "match_method", "source_value", "propagation_source_entity_id",
    "identity_scope", "evidence_source", "source_record_id",
    "affects_dual_selectivity_rank",
)
SECTION_COLUMNS = (
    "release_id", "label_id", "section_name", "section_index",
    "section_category", "section_text", "section_text_sha256", "evidence_source",
    "source_record_id", "trait_adjudication_status",
    "affects_dual_selectivity_rank",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _identity_scope(method: str, propagation_source_entity_id: str) -> str:
    if propagation_source_entity_id:
        return "propagated_identity"
    if method == "exact_application_number":
        return "application_level_product_match"
    if method == "exact_active_moiety_unii":
        return "direct_active_moiety_match"
    if method.endswith("_to_substance"):
        return "unique_substance_name_match"
    if method.endswith("_to_product"):
        return "unique_product_name_match"
    return "direct_identity"


def normalize_compact_labels(
    release_id: str,
    source_release: str,
    records: Iterable[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build label, identity-edge, and section tables at their native evidence grain."""

    label_rows = []
    edge_rows = []
    section_rows = []
    seen_labels = set()
    for record in records:
        label_id = str(record.get("label_id", "")).strip()
        if not label_id:
            raise ValueError("matched openFDA record has no label_id")
        if label_id in seen_labels:
            raise ValueError(f"duplicate openFDA label_id: {label_id}")
        seen_labels.add(label_id)
        if not record.get("matched"):
            raise ValueError(f"unmatched record entered label release: {label_id}")
        openfda = record.get("openfda")
        sections = record.get("sections")
        evidence = record.get("match_evidence")
        if not isinstance(openfda, dict) or not isinstance(sections, dict):
            raise ValueError(f"label record lacks openFDA or sections mapping: {label_id}")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"label record lacks exact identity evidence: {label_id}")
        source_digest = _hash_text(_canonical_json(record))
        label_row = {
            "release_id": release_id,
            "label_id": label_id,
            "set_id": str(record.get("set_id", "")),
            "label_version": str(record.get("version", "")),
            "effective_time": str(record.get("effective_time", "")),
            "source_release": source_release,
            "source_partition": str(record.get("source_partition", "")),
            "source_record_sha256": source_digest,
            "affects_dual_selectivity_rank": False,
        }
        for field in OPENFDA_FIELDS:
            label_row[f"{field}_json"] = _canonical_json(_ordered_strings(openfda.get(field)))
        label_rows.append(label_row)

        for item in evidence:
            required = {
                "entity_type", "entity_id", "match_method", "source_value",
                "propagation_source_entity_id",
            }
            if not isinstance(item, dict) or required - set(item):
                raise ValueError(f"invalid identity evidence for label {label_id}")
            identity = {
                key: str(item[key]).strip()
                for key in required
            }
            if not identity["entity_id"] or not identity["entity_type"]:
                raise ValueError(f"blank identity edge for label {label_id}")
            payload = "\x1f".join(
                [
                    label_id,
                    identity["entity_type"],
                    identity["entity_id"],
                    identity["match_method"],
                    identity["source_value"],
                    identity["propagation_source_entity_id"],
                ]
            )
            edge_rows.append(
                {
                    "release_id": release_id,
                    "label_entity_edge_id": "label_edge:" + _hash_text(payload)[:24],
                    "label_id": label_id,
                    **identity,
                    "identity_scope": _identity_scope(
                        identity["match_method"],
                        identity["propagation_source_entity_id"],
                    ),
                    "evidence_source": source_release,
                    "source_record_id": "openfda_label:" + label_id,
                    "affects_dual_selectivity_rank": False,
                }
            )

        for section_name, values in sorted(sections.items()):
            if section_name not in SECTION_CATEGORIES:
                raise ValueError(f"unknown compact label section: {section_name}")
            for section_index, section_text in enumerate(
                _ordered_strings(values), start=1
            ):
                text_digest = _hash_text(section_text)
                section_rows.append(
                    {
                        "release_id": release_id,
                        "label_id": label_id,
                        "section_name": section_name,
                        "section_index": section_index,
                        "section_category": SECTION_CATEGORIES[section_name],
                        "section_text": section_text,
                        "section_text_sha256": text_digest,
                        "evidence_source": source_release,
                        "source_record_id": (
                            f"openfda_label_section:{label_id}:{section_name}:"
                            f"{section_index}:{text_digest[:16]}"
                        ),
                        "trait_adjudication_status": "not_adjudicated",
                        "affects_dual_selectivity_rank": False,
                    }
                )

    labels = pd.DataFrame(label_rows, columns=LABEL_COLUMNS)
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    sections = pd.DataFrame(section_rows, columns=SECTION_COLUMNS)
    for name, frame in (
        ("regulatory_labels", labels),
        ("regulatory_label_entity_edges", edges),
        ("regulatory_label_sections", sections),
    ):
        if name != "regulatory_label_sections" and frame.empty:
            raise ValueError(f"{name} is empty")
        validate_table(name, frame)
    label_ids = set(labels["label_id"].astype(str))
    if set(edges["label_id"].astype(str)) - label_ids:
        raise ValueError("label identity edges contain orphan labels")
    if set(sections["label_id"].astype(str)) - label_ids:
        raise ValueError("label sections contain orphan labels")
    return (
        labels.sort_values("label_id", kind="stable").reset_index(drop=True),
        edges.sort_values(
            ["label_id", "entity_type", "entity_id", "label_entity_edge_id"],
            kind="stable",
        ).reset_index(drop=True),
        sections.sort_values(
            ["label_id", "section_name", "section_index"], kind="stable"
        ).reset_index(drop=True),
    )
