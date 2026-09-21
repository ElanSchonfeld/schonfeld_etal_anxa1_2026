"""Source-grained target actions and complete drug-portfolio summaries."""

from __future__ import annotations

import json

import pandas as pd


ACTION_DIRECTION = {
    "AGONIST": "activation",
    "PARTIAL AGONIST": "activation",
    "ACTIVATOR": "activation",
    "OPENER": "activation",
    "POSITIVE MODULATOR": "activation",
    "POSITIVE ALLOSTERIC MODULATOR": "activation",
    "EXOGENOUS PROTEIN": "activation",
    "EXOGENOUS GENE": "activation",
    "INHIBITOR": "inhibition",
    "ANTAGONIST": "inhibition",
    "BLOCKER": "inhibition",
    "NEGATIVE ALLOSTERIC MODULATOR": "inhibition",
    "ALLOSTERIC ANTAGONIST": "inhibition",
    "INVERSE AGONIST": "inhibition",
    "GATING INHIBITOR": "inhibition",
    "ANTISENSE INHIBITOR": "inhibition",
    "RNA INTERFERENCE INHIBITOR": "inhibition",
    "DEGRADER": "inhibition",
    "DISRUPTING AGENT": "inhibition",
    "GENE EDITING NEGATIVE MODULATOR": "inhibition",
    "UNSPECIFIED": "ambiguous",
    "MODULATOR": "ambiguous",
    "SPLICING MODULATOR": "ambiguous",
    "ALLOSTERIC MODULATOR": "ambiguous",
    "ANTIBODY BINDING": "ambiguous",
    "BINDING AGENT": "ambiguous",
    "HYDROLYTIC ENZYME": "ambiguous",
    "PROTEOLYTIC ENZYME": "ambiguous",
    "RELEASING AGENT": "ambiguous",
    "STABILISER": "ambiguous",
    "SUBSTRATE": "ambiguous",
    "OTHER": "ambiguous",
    "CROSS-LINKING AGENT": "ambiguous",
    "PHARMACOLOGICAL CHAPERONE": "ambiguous",
}


def _ordered_json(values) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}))


def summarize_target_actions(
    edges: pd.DataFrame,
    *,
    release_id: str,
    entity_type: str,
    entity_column: str,
) -> pd.DataFrame:
    required = {
        entity_column, "target_id", "human_gene", "action_type", "target_scope",
        "source_family", "source_database",
    }
    if missing := sorted(required - set(edges.columns)):
        raise ValueError(f"target action evidence is missing: {missing}")
    source = edges.copy()
    source["action_normalized"] = (
        source["action_type"].fillna("unspecified").astype(str).str.strip().str.upper()
    )
    unknown = sorted(set(source["action_normalized"]) - set(ACTION_DIRECTION))
    if unknown:
        raise ValueError(f"unclassified pharmacology actions: {unknown}")
    source["action_direction"] = source["action_normalized"].map(ACTION_DIRECTION)
    source["exact_scope"] = source["target_scope"].astype(str).eq("single_protein")
    rows = []
    for (entity_id, target_id), group in source.groupby(
        [entity_column, "target_id"], sort=True, dropna=False
    ):
        if pd.isna(entity_id) or pd.isna(target_id):
            raise ValueError("target action evidence has a null entity or target ID")
        genes = sorted(set(group["human_gene"].dropna().astype(str)))
        if len(genes) > 1:
            raise ValueError(f"{target_id} has conflicting approved gene symbols")
        exact = group[group["exact_scope"]]
        broad = group[~group["exact_scope"]]
        exact_activation = exact[exact["action_direction"].eq("activation")]
        exact_inhibition = exact[exact["action_direction"].eq("inhibition")]
        exact_ambiguous = exact[exact["action_direction"].eq("ambiguous")]
        broad_directional = broad[broad["action_direction"].isin(["activation", "inhibition"])]
        broad_ambiguous = broad[broad["action_direction"].eq("ambiguous")]
        has_activation = len(exact_activation) > 0
        has_inhibition = len(exact_inhibition) > 0
        if has_activation and has_inhibition:
            consensus = "conflicting_activation_and_inhibition"
            activation_families = set(exact_activation["source_family"].astype(str))
            inhibition_families = set(exact_inhibition["source_family"].astype(str))
            independent = any(
                left != right
                for left in activation_families
                for right in inhibition_families
            )
            conflict_status = (
                "independent_source_family_conflict"
                if independent else "single_source_family_conflict"
            )
        elif has_activation:
            consensus = "activation_only"
            conflict_status = "not_conflicting"
        elif has_inhibition:
            consensus = "inhibition_only"
            conflict_status = "not_conflicting"
        elif len(broad_directional):
            consensus = "broad_scope_direction_only"
            conflict_status = "not_conflicting"
        else:
            consensus = "directional_action_unavailable"
            conflict_status = "not_conflicting"
        n_directional_exact = len(exact_activation) + len(exact_inhibition)
        if len(exact) and n_directional_exact == len(exact):
            coverage = "complete_directional_exact"
        elif n_directional_exact:
            coverage = "partial_directional_exact"
        else:
            coverage = "no_directional_exact"
        if has_activation and has_inhibition:
            clinical_status = "source_conflict_unadjudicated"
        elif has_activation or has_inhibition or len(broad_directional):
            clinical_status = "source_only_unadjudicated"
        else:
            clinical_status = "direction_unavailable_unadjudicated"
        rows.append(
            {
                "release_id": release_id,
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "target_id": str(target_id),
                "human_gene": genes[0] if genes else "",
                "n_source_rows": len(group),
                "n_exact_scope_rows": len(exact),
                "n_directional_exact_rows": n_directional_exact,
                "n_directional_broad_rows": len(broad_directional),
                "n_ambiguous_exact_rows": len(exact_ambiguous),
                "n_ambiguous_broad_rows": len(broad_ambiguous),
                "activation_actions_json": _ordered_json(
                    group.loc[group["action_direction"].eq("activation"), "action_normalized"]
                ),
                "inhibition_actions_json": _ordered_json(
                    group.loc[group["action_direction"].eq("inhibition"), "action_normalized"]
                ),
                "ambiguous_actions_json": _ordered_json(
                    group.loc[group["action_direction"].eq("ambiguous"), "action_normalized"]
                ),
                "activation_source_families_json": _ordered_json(
                    exact_activation["source_family"]
                ),
                "inhibition_source_families_json": _ordered_json(
                    exact_inhibition["source_family"]
                ),
                "source_databases_json": _ordered_json(group["source_database"]),
                "target_scopes_json": _ordered_json(group["target_scope"]),
                "action_consensus": consensus,
                "conflict_evidence_status": conflict_status,
                "exact_action_coverage": coverage,
                "clinical_action_status": clinical_status,
                "affects_dual_selectivity_rank": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_portfolios(
    target_summary: pd.DataFrame,
    raw_edges: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    release_id: str,
    entity_type: str,
    entity_id_column: str,
) -> pd.DataFrame:
    """Retain every drug identity, including entities with zero resolved targets."""

    required_coverage = {entity_id_column, "preferred_name", "target_coverage_status"}
    if missing := sorted(required_coverage - set(coverage.columns)):
        raise ValueError(f"portfolio coverage is missing: {missing}")
    summary_groups = {
        str(entity_id): group
        for entity_id, group in target_summary.groupby("entity_id", sort=False)
    }
    edge_counts = raw_edges.groupby(entity_id_column).size().to_dict()
    rows = []
    for record in coverage.sort_values(entity_id_column, kind="stable").itertuples(index=False):
        entity_id = str(getattr(record, entity_id_column))
        group = summary_groups.get(entity_id, target_summary.iloc[0:0])
        n_targets = len(group)
        consensus = group.get("action_consensus", pd.Series(dtype=str)).astype(str)
        exact_coverage = group.get("exact_action_coverage", pd.Series(dtype=str)).astype(str)
        counts = {
            "activation": int(consensus.eq("activation_only").sum()),
            "inhibition": int(consensus.eq("inhibition_only").sum()),
            "conflicting": int(consensus.eq("conflicting_activation_and_inhibition").sum()),
            "broad": int(consensus.eq("broad_scope_direction_only").sum()),
            "unavailable": int(consensus.eq("directional_action_unavailable").sum()),
            "partial": int(exact_coverage.eq("partial_directional_exact").sum()),
        }
        if n_targets == 0:
            status = "no_resolved_human_targets"
        elif counts["conflicting"]:
            status = "target_action_conflict_present"
        elif counts["activation"] + counts["inhibition"] == n_targets and not counts["partial"]:
            status = "target_actions_fully_directional_no_conflicts"
        elif counts["activation"] + counts["inhibition"] == 0:
            status = "directional_actions_unavailable"
        else:
            status = "target_actions_partially_unavailable"
        rows.append(
            {
                "release_id": release_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "preferred_name": str(record.preferred_name),
                "target_coverage_status": str(record.target_coverage_status),
                "n_target_evidence_rows": int(edge_counts.get(entity_id, 0)),
                "n_targets": n_targets,
                "n_targets_activation_only": counts["activation"],
                "n_targets_inhibition_only": counts["inhibition"],
                "n_targets_conflicting": counts["conflicting"],
                "n_targets_broad_scope_only": counts["broad"],
                "n_targets_direction_unavailable": counts["unavailable"],
                "n_targets_partial_directional": counts["partial"],
                "polypharmacology_status": status,
                "benefit_liability_status": (
                    "not_interpretable_without_trait_direction_and_complete_portfolio"
                ),
                "affects_dual_selectivity_rank": False,
            }
        )
    return pd.DataFrame(rows)
