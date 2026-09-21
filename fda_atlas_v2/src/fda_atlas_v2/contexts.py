"""Explicit availability rows for drug-target-trait supporting evidence."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .contracts import ContractError, TRAITS, validate_table


def _record_id(
    entity_type: str, entity_id: str, target_id: str, trait_id: str
) -> str:
    payload = "\x1f".join((entity_type, entity_id, target_id, trait_id)).encode()
    return "context_coverage:" + hashlib.sha256(payload).hexdigest()[:20]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _target_portfolios(
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
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
    return {
        (str(entity_type), str(entity_id)): sorted(
            group["target_id"].dropna().astype(str).unique()
        )
        for (entity_type, entity_id), group in pairs.groupby(
            ["entity_type", "entity_id"], sort=True
        )
    }


def prepare_label_context_candidates(
    release_id: str,
    label_edges: pd.DataFrame,
    label_sections: pd.DataFrame,
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare drug-grain label evidence for separate trait adjudication."""

    required_edges = {
        "label_id", "entity_type", "entity_id", "match_method", "source_value",
        "propagation_source_entity_id", "identity_scope", "source_record_id",
    }
    required_sections = {
        "label_id", "section_name", "section_index", "section_category",
        "section_text_sha256", "source_record_id", "trait_adjudication_status",
    }
    if required_edges - set(label_edges):
        raise ValueError("label context candidates lack identity-edge provenance")
    if required_sections - set(label_sections):
        raise ValueError("label context candidates lack section provenance")
    admitted = label_edges[
        label_edges["entity_type"].isin(["active_moiety", "regulatory_substance"])
    ].copy()
    if admitted.empty or label_sections.empty:
        return pd.DataFrame(
            columns=[
                "release_id", "candidate_id", "label_id", "entity_type",
                "entity_id", "section_name", "section_index", "section_category",
                "section_text_sha256", "label_section_source_record_id",
                "identity_path_json", "polypharmacology_target_ids_json",
                "n_portfolio_targets", "trait_adjudication_status",
                "affects_dual_selectivity_rank",
            ]
        )
    portfolios = _target_portfolios(moiety_edges, substance_edges)
    identity_paths: dict[tuple[str, str, str], str] = {}
    for key, group in admitted.groupby(
        ["label_id", "entity_type", "entity_id"], sort=True
    ):
        paths = [
            {
                "identity_scope": str(row.identity_scope),
                "match_method": str(row.match_method),
                "propagation_source_entity_id": str(
                    row.propagation_source_entity_id
                ),
                "source_record_id": str(row.source_record_id),
                "source_value": str(row.source_value),
            }
            for row in group.sort_values(
                ["identity_scope", "match_method", "source_value", "source_record_id"],
                kind="stable",
            ).itertuples(index=False)
        ]
        identity_paths[tuple(map(str, key))] = _canonical_json(paths)
    rows = []
    entity_rows = admitted[["label_id", "entity_type", "entity_id"]].drop_duplicates()
    joined = entity_rows.merge(
        label_sections[sorted(required_sections)],
        on="label_id",
        how="inner",
        validate="many_to_many",
    )
    for row in joined.sort_values(
        ["label_id", "entity_type", "entity_id", "section_name", "section_index"],
        kind="stable",
    ).itertuples(index=False):
        entity_key = (str(row.entity_type), str(row.entity_id))
        targets = portfolios.get(entity_key, [])
        payload = "\x1f".join(
            [
                str(row.label_id), str(row.entity_type), str(row.entity_id),
                str(row.section_name), str(row.section_index),
                str(row.section_text_sha256),
            ]
        )
        rows.append(
            {
                "release_id": release_id,
                "candidate_id": "label_context_candidate:"
                + hashlib.sha256(payload.encode()).hexdigest()[:24],
                "label_id": str(row.label_id),
                "entity_type": str(row.entity_type),
                "entity_id": str(row.entity_id),
                "section_name": str(row.section_name),
                "section_index": int(row.section_index),
                "section_category": str(row.section_category),
                "section_text_sha256": str(row.section_text_sha256),
                "label_section_source_record_id": str(row.source_record_id),
                "identity_path_json": identity_paths[
                    (str(row.label_id), str(row.entity_type), str(row.entity_id))
                ],
                "polypharmacology_target_ids_json": _canonical_json(targets),
                "n_portfolio_targets": len(targets),
                "trait_adjudication_status": str(row.trait_adjudication_status),
                "affects_dual_selectivity_rank": False,
            }
        )
    output = pd.DataFrame(rows)
    if output.duplicated("candidate_id").any():
        raise ValueError("label context candidates contain duplicate identities")
    if output["affects_dual_selectivity_rank"].astype(bool).any():
        raise ValueError("label context candidates cannot alter rank")
    return output.reset_index(drop=True)


def validate_context_semantics(frame: pd.DataFrame) -> None:
    """Require complete portfolios and fail-closed adjudication provenance."""

    parsed_portfolios = []
    parsed_paths = []
    for row in frame.itertuples(index=False):
        try:
            portfolio = json.loads(str(row.polypharmacology_target_ids_json))
            identity_path = json.loads(str(row.identity_path_json))
        except json.JSONDecodeError as exc:
            raise ContractError("drug contexts contain invalid provenance JSON") from exc
        if not isinstance(portfolio, list) or not all(
            isinstance(value, str) and value for value in portfolio
        ):
            raise ContractError("drug contexts require a target-ID portfolio list")
        if portfolio != sorted(set(portfolio)):
            raise ContractError("drug context target portfolios must be sorted and unique")
        if str(row.target_id) not in portfolio:
            raise ContractError("drug context target is absent from its complete portfolio")
        if not isinstance(identity_path, list):
            raise ContractError("drug context identity path must be a JSON list")
        parsed_portfolios.append(_canonical_json(portfolio))
        parsed_paths.append(identity_path)
    if len(frame):
        observed = frame.assign(_portfolio=parsed_portfolios).groupby(
            ["entity_type", "entity_id"], sort=False
        )["_portfolio"].nunique()
        if observed.gt(1).any():
            raise ContractError("drug context entity has inconsistent target portfolios")
    known = frame["direction_known"].astype(bool)
    if not known.eq(~frame["direction"].astype(str).eq("unknown")).all():
        raise ContractError("drug context direction-known flag disagrees with direction")
    available = frame["value"].astype(str).ne("not_available")
    if (
        available
        & frame["adjudication_method"].astype(str).eq("not_adjudicated")
    ).any():
        raise ContractError("available drug contexts require an adjudication method")
    if any(available.iloc[index] and not parsed_paths[index] for index in range(len(frame))):
        raise ContractError("available drug contexts require an exact identity path")
    has_label = frame["label_id"].astype(str).str.len().gt(0)
    label_fields = frame[
        ["label_section_source_record_id", "label_section_name"]
    ].astype(str).apply(lambda column: column.str.len().gt(0))
    if (has_label != label_fields.all(axis=1)).any():
        raise ContractError("drug context label provenance is incomplete")


def build_pending_context_coverage(
    release_id: str,
    moiety_edges: pd.DataFrame,
    substance_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Retain one pending row for every entity, target, and frozen trait."""

    portfolios = _target_portfolios(moiety_edges, substance_edges)
    pairs = pd.DataFrame(
        [
            {"entity_type": entity_type, "entity_id": entity_id, "target_id": target_id}
            for (entity_type, entity_id), targets in sorted(portfolios.items())
            for target_id in targets
        ]
    )
    rows = []
    for pair in pairs.sort_values(
        ["entity_type", "entity_id", "target_id"], kind="stable"
    ).itertuples(index=False):
        for trait in sorted(TRAITS):
            rows.append(
                {
                    "release_id": release_id,
                    "entity_type": str(pair.entity_type),
                    "entity_id": str(pair.entity_id),
                    "target_id": str(pair.target_id),
                    "trait_id": trait,
                    "context_type": "context_coverage",
                    "evidence_source": "atlas_v2_coverage",
                    "value": "not_available",
                    "direction": "unknown",
                    "status": "pending_label_cns_safety_and_directional_evidence",
                    "source_record_id": _record_id(
                        str(pair.entity_type), str(pair.entity_id),
                        str(pair.target_id), trait
                    ),
                    "missing_reason": (
                        "pinned label, CNS, safety, indication, and independent "
                        "directional evidence have not all been acquired"
                    ),
                    "label_id": "",
                    "label_section_source_record_id": "",
                    "label_section_name": "",
                    "identity_path_json": "[]",
                    "polypharmacology_target_ids_json": _canonical_json(
                        portfolios[(str(pair.entity_type), str(pair.entity_id))]
                    ),
                    "adjudication_method": "not_adjudicated",
                    "direction_evidence_source": "",
                    "direction_known": False,
                    "affects_dual_selectivity_rank": False,
                }
            )
    contexts = pd.DataFrame(rows)
    validate_table("drug_target_trait_contexts", contexts)
    return contexts
