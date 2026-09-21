"""Materialize explicit GSRS identity and active-moiety relationships."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


REGULATORY_NAME_ROLES = frozenset({"active_ingredient", "proper_name"})


@dataclass(frozen=True)
class GsrsIdentityRelease:
    substances: pd.DataFrame
    identity_edges: pd.DataFrame
    active_moieties: pd.DataFrame
    active_moiety_edges: pd.DataFrame
    audit: dict[str, Any]


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _gsrs_identity_id(row: pd.Series) -> str:
    unii = str(row.get("unii", "")).strip().upper()
    uuid = str(row.get("gsrs_uuid", "")).strip().lower()
    record_id = str(row.get("gsrs_record_id", "")).strip()
    if unii:
        return f"gsrs:{unii}"
    if uuid:
        return f"gsrs_uuid:{uuid}"
    if record_id:
        return f"gsrs_record:{record_id}"
    raise ValueError(f"selected GSRS query {row.get('query_id')} has no identity")


def _active_moiety_id(target: dict[str, Any]) -> str:
    unii = str(target.get("target_unii") or "").strip().upper()
    uuid = str(target.get("target_uuid") or "").strip().lower()
    if unii:
        return f"gsrs:{unii}"
    if uuid:
        return f"gsrs_uuid:{uuid}"
    fallback = str(
        target.get("target_linking_id") or target.get("target_name") or ""
    ).strip()
    if not fallback:
        raise ValueError("explicit GSRS active-moiety target has no stable identity")
    return f"gsrs_fallback:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"


def _preferred(values: list[str]) -> str:
    nonempty = [str(value).strip() for value in values if str(value).strip()]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    return sorted(value for value, count in counts.items() if count == max(counts.values()))[0]


def _single_or_empty(values: list[str], field: str, entity_id: str) -> str:
    nonempty = {str(value).strip() for value in values if str(value).strip()}
    if len(nonempty) > 1:
        raise ValueError(f"{entity_id} has conflicting {field} values: {sorted(nonempty)}")
    return next(iter(nonempty), "")


def build_gsrs_identity_release(
    substances: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    release_id: str,
) -> GsrsIdentityRelease:
    required_substance = {"substance_id", "exact_name_key"}
    required_crosswalk = {
        "query_id",
        "name_role",
        "regulatory_name_exact_key",
        "match_status",
        "match_method",
        "match_confidence",
        "candidate_selected",
        "gsrs_record_id",
        "gsrs_uuid",
        "unii",
        "explicit_active_moiety_targets_json",
        "gsrs_source_version",
    }
    if missing := sorted(required_substance - set(substances.columns)):
        raise ValueError(f"regulatory substances missing columns: {missing}")
    if missing := sorted(required_crosswalk - set(crosswalk.columns)):
        raise ValueError(f"GSRS crosswalk missing columns: {missing}")

    relevant = crosswalk[crosswalk["name_role"].isin(REGULATORY_NAME_ROLES)].copy()
    if relevant["query_id"].duplicated().any():
        raise ValueError("GSRS query IDs are not unique")
    if substances["exact_name_key"].duplicated().any():
        raise ValueError("regulatory exact-name keys are not unique")
    key_to_substance = dict(zip(substances["exact_name_key"], substances["substance_id"]))
    query_keys = set(relevant["regulatory_name_exact_key"])
    substance_keys = set(key_to_substance)
    if query_keys != substance_keys:
        raise ValueError(
            "GSRS and regulatory exact-name universes differ: "
            f"missing_queries={len(substance_keys - query_keys)}, "
            f"extra_queries={len(query_keys - substance_keys)}"
        )

    selected = relevant[relevant["candidate_selected"].map(_truthy)].copy()
    invalid_selected = selected[~selected["match_status"].eq("resolved")]
    if not invalid_selected.empty:
        raise ValueError("a non-resolved GSRS query was selected")
    resolved_by_key: dict[str, str] = {}
    for key, group in selected.groupby("regulatory_name_exact_key", sort=False):
        identities = {_gsrs_identity_id(row) for _, row in group.iterrows()}
        if len(identities) != 1:
            raise ValueError(f"GSRS exact key {key!r} resolves to multiple identities")
        resolved_by_key[key] = next(iter(identities))

    identity_rows: list[dict[str, Any]] = []
    active_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_edges: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"query_ids": set(), "gsrs_record_ids": set(), "versions": set()}
    )
    for _, row in selected.sort_values("query_id", kind="stable").iterrows():
        substance_id = key_to_substance[row["regulatory_name_exact_key"]]
        target_identity = _gsrs_identity_id(row)
        evidence_source = f"GSRS:{row['gsrs_source_version']}"
        identity_rows.append(
            {
                "release_id": release_id,
                "source_substance_id": substance_id,
                "target_substance_id": target_identity,
                "relationship_type": "identity_match",
                "evidence_source": evidence_source,
                "source_record_id": row["query_id"],
                "query_id": row["query_id"],
                "source_name_role": row["name_role"],
                "match_status": row["match_status"],
                "match_method": row["match_method"],
                "match_confidence": row["match_confidence"],
                "gsrs_source_version": row["gsrs_source_version"],
                "gsrs_record_id": row["gsrs_record_id"],
                "unii": row.get("unii", ""),
            }
        )
        raw_targets = row["explicit_active_moiety_targets_json"] or "[]"
        try:
            targets = json.loads(raw_targets)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid active-moiety JSON for {row['query_id']}") from exc
        if not isinstance(targets, list):
            raise ValueError(f"active-moiety targets are not a list for {row['query_id']}")
        for target in targets:
            relationship_type = str(target.get("type") or "").strip()
            if not relationship_type.startswith("ACTIVE MOIETY"):
                raise ValueError(
                    f"non-active relationship in active-moiety field: {relationship_type}"
                )
            moiety_id = _active_moiety_id(target)
            active_metadata[moiety_id].append(target)
            edge_key = (substance_id, moiety_id, relationship_type)
            active_edges[edge_key]["query_ids"].add(str(row["query_id"]))
            active_edges[edge_key]["gsrs_record_ids"].add(str(row["gsrs_record_id"]))
            active_edges[edge_key]["versions"].add(str(row["gsrs_source_version"]))

    moiety_rows: list[dict[str, Any]] = []
    for moiety_id, records in sorted(active_metadata.items()):
        unii = _single_or_empty(
            [row.get("target_unii", "") for row in records], "UNII", moiety_id
        ).upper()
        uuid = _single_or_empty(
            [row.get("target_uuid", "") for row in records], "UUID", moiety_id
        ).lower()
        linking_id = _single_or_empty(
            [row.get("target_linking_id", "") for row in records],
            "linking ID",
            moiety_id,
        )
        substance_class = _single_or_empty(
            [row.get("target_substance_class", "") for row in records],
            "substance class",
            moiety_id,
        )
        versions = sorted(
            {
                str(value)
                for key, payload in active_edges.items()
                if key[1] == moiety_id
                for value in payload["versions"]
            }
        )
        if len(versions) != 1:
            raise ValueError(f"active moiety {moiety_id} spans GSRS versions {versions}")
        moiety_rows.append(
            {
                "release_id": release_id,
                "moiety_id": moiety_id,
                "preferred_name": _preferred(
                    [row.get("target_name", "") for row in records]
                ),
                "normalization_status": "explicit_gsrs_relationship",
                "resolution_class": "explicit_active_moiety",
                "unii": unii or pd.NA,
                "gsrs_uuid": uuid or pd.NA,
                "linking_id": linking_id or pd.NA,
                "substance_class": substance_class or pd.NA,
                "gsrs_source_version": versions[0],
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for (substance_id, moiety_id, relationship_type), payload in sorted(
        active_edges.items()
    ):
        versions = sorted(payload["versions"])
        if len(versions) != 1:
            raise ValueError(
                f"active-moiety edge {substance_id}/{moiety_id} spans versions {versions}"
            )
        edge_digest = hashlib.sha256(
            f"{substance_id}\x1f{moiety_id}\x1f{relationship_type}".encode("utf-8")
        ).hexdigest()[:20]
        edge_rows.append(
            {
                "release_id": release_id,
                "substance_id": substance_id,
                "moiety_id": moiety_id,
                "relationship_type": relationship_type,
                "evidence_source": f"GSRS:{versions[0]}",
                "source_record_id": f"gsrs_active_moiety_edge:{edge_digest}",
                "query_ids": " | ".join(sorted(payload["query_ids"])),
                "gsrs_record_ids": " | ".join(sorted(payload["gsrs_record_ids"])),
                "gsrs_source_version": versions[0],
            }
        )

    edge_substances = {row["substance_id"] for row in edge_rows}
    updated = substances.copy()
    updated["matched_gsrs_identity_id"] = updated["exact_name_key"].map(
        resolved_by_key
    )
    updated["has_explicit_active_moiety_relationship"] = updated[
        "substance_id"
    ].isin(edge_substances)
    updated["identity_status"] = "unresolved_identity"
    resolved_mask = updated["exact_name_key"].isin(resolved_by_key)
    updated.loc[resolved_mask, "identity_status"] = (
        "resolved_no_explicit_active_moiety_relationship"
    )
    updated.loc[
        updated["has_explicit_active_moiety_relationship"], "identity_status"
    ] = "resolved_explicit_active_moiety"
    updated["identity_source"] = updated["exact_name_key"].map(
        lambda key: (
            "GSRS exact-name query"
            if key in resolved_by_key
            else "GSRS exact-name query unresolved"
        )
    )

    identity_frame = pd.DataFrame(identity_rows)
    moiety_frame = pd.DataFrame(moiety_rows)
    active_edge_frame = pd.DataFrame(edge_rows)
    audit = {
        "regulatory_substances": len(updated),
        "gsrs_queries": len(relevant),
        "resolved_queries": len(selected),
        "resolved_regulatory_substances": len(resolved_by_key),
        "unresolved_regulatory_substances": len(updated) - len(resolved_by_key),
        "active_moieties": len(moiety_frame),
        "substance_active_moiety_edges": len(active_edge_frame),
        "substances_with_explicit_active_moiety": len(edge_substances),
        "resolved_without_explicit_active_moiety": (
            len(resolved_by_key) - len(edge_substances)
        ),
        "inferred_self_edges": 0,
    }
    return GsrsIdentityRelease(
        updated,
        identity_frame,
        moiety_frame,
        active_edge_frame,
        audit,
    )
