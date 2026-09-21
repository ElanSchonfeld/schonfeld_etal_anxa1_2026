"""Map source pharmacology onto the v2 moiety universe."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

from .contracts import validate_table


def legacy_name_key(value: object) -> str:
    """Reproduce the punctuation-insensitive key used by the prior evidence build."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").upper()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def build_substance_evidence_bridge(
    regulatory_substances: pd.DataFrame,
    legacy_coverage: pd.DataFrame,
) -> pd.DataFrame:
    required_new = {"release_id", "substance_id", "preferred_name"}
    required_old = {"regulatory_substance_id", "regulatory_substance_norm"}
    if missing := sorted(required_new - set(regulatory_substances.columns)):
        raise ValueError(f"regulatory substances are missing columns: {missing}")
    if missing := sorted(required_old - set(legacy_coverage.columns)):
        raise ValueError(f"legacy coverage is missing columns: {missing}")

    new = regulatory_substances[
        ["release_id", "substance_id", "preferred_name"]
    ].copy()
    old = legacy_coverage[
        ["regulatory_substance_id", "regulatory_substance_norm"]
    ].copy()
    new["legacy_name_key"] = new["preferred_name"].map(legacy_name_key)
    old["legacy_name_key"] = old["regulatory_substance_norm"].map(legacy_name_key)
    if old["legacy_name_key"].duplicated().any():
        raise ValueError("legacy substance keys are not unique")
    if (new["legacy_name_key"] == "").any() or (old["legacy_name_key"] == "").any():
        raise ValueError("empty legacy substance key")

    bridge = new.merge(old, on="legacy_name_key", how="left", validate="many_to_one")
    new_key_counts = new.groupby("legacy_name_key")["substance_id"].transform("size")
    bridge["n_v2_substances_for_legacy_key"] = new_key_counts.to_numpy()
    bridge["bridge_status"] = "exact_legacy_key"
    bridge.loc[
        bridge["n_v2_substances_for_legacy_key"].gt(1), "bridge_status"
    ] = "legacy_key_split_across_exact_v2_substances"
    bridge.loc[bridge["regulatory_substance_id"].isna(), "bridge_status"] = "unmatched"
    return bridge.sort_values("substance_id", kind="stable").reset_index(drop=True)


def build_substance_target_edges(
    substance_bridge: pd.DataFrame,
    legacy_target_evidence: pd.DataFrame,
) -> pd.DataFrame:
    evidence = legacy_target_evidence.copy()
    evidence = evidence[
        evidence["human_gene"].notna()
        & evidence["human_gene"].astype(str).ne("")
        & evidence["hgnc_id"].notna()
        & evidence["hgnc_id"].astype(str).ne("")
        & evidence["evidence_tier"].astype(str).ne("T0")
    ].copy()
    joined = (
        substance_bridge[
            [
                "release_id",
                "substance_id",
                "regulatory_substance_id",
                "bridge_status",
            ]
        ]
        .loc[lambda frame: frame["bridge_status"].eq("exact_legacy_key")]
        .merge(
            evidence,
            on="regulatory_substance_id",
            how="inner",
            validate="many_to_many",
        )
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "release_id", "substance_id", "target_id", "evidence_source",
                "evidence_kind", "action_type", "directness", "target_scope",
                "confidence", "source_record_id",
            ]
        )

    joined["target_id"] = joined["hgnc_id"].astype(str)
    joined["evidence_source"] = (
        joined["source_database"].astype(str)
        + ":"
        + joined["source_release"].astype(str)
    )
    joined["source_record_id"] = joined["source_record_key"].astype(str)
    joined["directness"] = joined["evidence_kind"].astype(str)
    joined["action_type"] = joined["action_type"].fillna("").astype(str)
    joined.loc[joined["action_type"].eq(""), "action_type"] = "unspecified"
    joined["confidence"] = joined["drug_match_confidence"].fillna("unresolved")

    columns = [
        "release_id", "substance_id", "target_id", "evidence_source",
        "evidence_kind", "action_type", "directness", "target_scope",
        "confidence", "source_record_id", "bridge_status",
        "human_gene", "hgnc_id", "evidence_tier", "tier_reason",
        "source_database", "source_release", "source_family",
        "source_drug_id", "source_drug_name", "source_target_id",
        "source_target_name", "source_target_type", "target_organism",
        "primary_target_annotation", "mechanism_description",
        "drug_match_method", "drug_match_confidence",
        "n_convergent_direct_source_families",
    ]
    edges = joined[columns].sort_values(
        ["substance_id", "target_id", "evidence_source", "source_record_id"],
        kind="stable",
    )
    key = [
        "release_id", "substance_id", "target_id", "evidence_source", "source_record_id"
    ]
    edges = edges.drop_duplicates(key, keep="first").reset_index(drop=True)
    if edges.duplicated(key).any():
        raise ValueError("duplicate substance target edge key")
    return edges


def build_substance_target_coverage(
    regulatory_substances: pd.DataFrame,
    substance_target_edges: pd.DataFrame,
    substance_bridge: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if substance_target_edges.empty:
        summary = pd.DataFrame(columns=["substance_id"])
    else:
        summary = (
            substance_target_edges.groupby("substance_id", sort=True)
            .agg(
                n_target_edges=("target_id", "size"),
                n_human_targets=("target_id", "nunique"),
                human_target_ids=(
                    "target_id", lambda values: " | ".join(sorted(set(values.astype(str))))
                ),
                human_genes=(
                    "human_gene", lambda values: " | ".join(sorted(set(values.astype(str))))
                ),
                best_evidence_tier=(
                    "evidence_tier",
                    lambda values: min(values.astype(str), key=lambda value: int(value[1:])),
                ),
            )
            .reset_index()
        )
    coverage = regulatory_substances[
        [
            "release_id", "substance_id", "preferred_name", "identity_status",
            "matched_gsrs_identity_id", "has_explicit_active_moiety_relationship",
        ]
    ].merge(summary, on="substance_id", how="left", validate="one_to_one")
    coverage["n_target_edges"] = coverage["n_target_edges"].fillna(0).astype(int)
    coverage["n_human_targets"] = coverage["n_human_targets"].fillna(0).astype(int)
    coverage["target_coverage_status"] = "resolved_human_target"
    coverage.loc[
        coverage["n_human_targets"].eq(0), "target_coverage_status"
    ] = "no_resolved_human_target"
    if substance_bridge is not None:
        split_ids = set(
            substance_bridge.loc[
                substance_bridge["bridge_status"].eq(
                    "legacy_key_split_across_exact_v2_substances"
                ),
                "substance_id",
            ].astype(str)
        )
        coverage.loc[
            coverage["substance_id"].astype(str).isin(split_ids)
            & coverage["n_human_targets"].eq(0),
            "target_coverage_status",
        ] = "ambiguous_legacy_identity_withheld"
    return coverage.sort_values("substance_id", kind="stable").reset_index(drop=True)


def build_drug_target_edges_with_audit(
    substance_target_edges: pd.DataFrame,
    substance_moiety_edges: pd.DataFrame,
    active_moieties: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate only evidence with unambiguous moiety attribution."""

    required_moiety = {"moiety_id", "preferred_name"}
    if missing := sorted(required_moiety - set(active_moieties.columns)):
        raise ValueError(f"active moieties are missing columns: {missing}")
    links = substance_moiety_edges[["substance_id", "moiety_id"]].drop_duplicates()
    link_counts = links.groupby("substance_id")["moiety_id"].nunique()
    joined = substance_target_edges.merge(
        links,
        on="substance_id",
        how="inner",
        validate="many_to_many",
    ).merge(
        active_moieties[["moiety_id", "preferred_name"]].rename(
            columns={"preferred_name": "moiety_preferred_name"}
        ),
        on="moiety_id",
        how="left",
        validate="many_to_one",
    )
    if joined["moiety_preferred_name"].isna().any():
        raise ValueError("substance-moiety edge references an unknown active moiety")
    joined["n_moieties_for_substance"] = joined["substance_id"].map(link_counts).astype(int)
    source_key = joined["source_drug_name"].map(legacy_name_key)
    moiety_key = joined["moiety_preferred_name"].map(legacy_name_key)
    joined["moiety_attribution_status"] = "single_explicit_moiety"
    multiple = joined["n_moieties_for_substance"].gt(1)
    exact_name = source_key.eq(moiety_key) & source_key.ne("")
    joined.loc[multiple & exact_name, "moiety_attribution_status"] = (
        "multi_moiety_exact_source_drug_name"
    )
    joined.loc[multiple & ~exact_name, "moiety_attribution_status"] = (
        "ambiguous_multi_moiety_attribution_withheld"
    )
    withheld = joined[
        joined["moiety_attribution_status"].eq(
            "ambiguous_multi_moiety_attribution_withheld"
        )
    ].copy()
    eligible = joined[~joined.index.isin(withheld.index)].copy()
    audit_columns = [
        "release_id", "substance_id", "moiety_id", "target_id",
        "evidence_source", "source_record_id", "source_drug_id",
        "source_drug_name", "moiety_preferred_name", "n_moieties_for_substance",
        "moiety_attribution_status",
    ]
    withheld = withheld[audit_columns].sort_values(
        ["substance_id", "moiety_id", "target_id", "evidence_source", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)

    if eligible.empty:
        edges = pd.DataFrame(
            columns=[
                "release_id", "moiety_id", "target_id", "evidence_source",
                "evidence_kind", "action_type", "directness", "target_scope",
                "confidence", "source_record_id",
            ]
        )
        return edges, withheld
    columns = [
        "release_id", "moiety_id", "target_id", "evidence_source",
        "evidence_kind", "action_type", "directness", "target_scope",
        "confidence", "source_record_id", "substance_id", "bridge_status",
        "human_gene", "hgnc_id", "evidence_tier", "tier_reason",
        "source_database", "source_release", "source_family",
        "source_drug_id", "source_drug_name", "source_target_id",
        "source_target_name", "source_target_type", "target_organism",
        "primary_target_annotation", "mechanism_description",
        "drug_match_method", "drug_match_confidence",
        "n_convergent_direct_source_families", "moiety_attribution_status",
    ]
    edges = eligible[columns].sort_values(
        ["moiety_id", "target_id", "evidence_source", "source_record_id", "substance_id"],
        kind="stable",
    )
    key = [
        "release_id", "moiety_id", "target_id", "evidence_source", "source_record_id"
    ]
    edges = edges.drop_duplicates(key, keep="first").reset_index(drop=True)
    validate_table("drug_target_edges", edges)
    return edges, withheld


def build_drug_target_edges(
    substance_target_edges: pd.DataFrame,
    substance_moiety_edges: pd.DataFrame,
    active_moieties: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Propagate substance evidence only through explicit active-moiety edges."""

    if active_moieties is not None:
        return build_drug_target_edges_with_audit(
            substance_target_edges, substance_moiety_edges, active_moieties
        )[0]

    joined = substance_target_edges.merge(
        substance_moiety_edges[["substance_id", "moiety_id"]],
        on="substance_id",
        how="inner",
        validate="many_to_many",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "release_id", "moiety_id", "target_id", "evidence_source",
                "evidence_kind", "action_type", "directness", "target_scope",
                "confidence", "source_record_id",
            ]
        )
    columns = [
        "release_id", "moiety_id", "target_id", "evidence_source",
        "evidence_kind", "action_type", "directness", "target_scope",
        "confidence", "source_record_id", "substance_id", "bridge_status",
        "human_gene", "hgnc_id", "evidence_tier", "tier_reason",
        "source_database", "source_release", "source_family",
        "source_drug_id", "source_drug_name", "source_target_id",
        "source_target_name", "source_target_type", "target_organism",
        "primary_target_annotation", "mechanism_description",
        "drug_match_method", "drug_match_confidence",
        "n_convergent_direct_source_families",
    ]
    edges = joined[columns].sort_values(
        ["moiety_id", "target_id", "evidence_source", "source_record_id", "substance_id"],
        kind="stable",
    )
    key = [
        "release_id", "moiety_id", "target_id", "evidence_source", "source_record_id"
    ]
    edges = edges.drop_duplicates(key, keep="first").reset_index(drop=True)
    validate_table("drug_target_edges", edges)
    return edges


def build_moiety_target_coverage(
    active_moieties: pd.DataFrame,
    drug_target_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Retain every explicit moiety, including those without resolved targets."""

    if drug_target_edges.empty:
        summary = pd.DataFrame(columns=["moiety_id"])
    else:
        summary = (
            drug_target_edges.groupby("moiety_id", sort=True)
            .agg(
                n_target_edges=("target_id", "size"),
                n_human_targets=("target_id", "nunique"),
                human_target_ids=(
                    "target_id", lambda values: " | ".join(sorted(set(values.astype(str))))
                ),
                human_genes=(
                    "human_gene", lambda values: " | ".join(sorted(set(values.astype(str))))
                ),
                best_evidence_tier=(
                    "evidence_tier",
                    lambda values: min(values.astype(str), key=lambda value: int(value[1:])),
                ),
            )
            .reset_index()
        )
    coverage = active_moieties[
        ["release_id", "moiety_id", "preferred_name", "resolution_class", "unii"]
    ].merge(summary, on="moiety_id", how="left", validate="one_to_one")
    coverage["n_target_edges"] = coverage["n_target_edges"].fillna(0).astype(int)
    coverage["n_human_targets"] = coverage["n_human_targets"].fillna(0).astype(int)
    coverage["target_coverage_status"] = "resolved_human_target"
    coverage.loc[
        coverage["n_human_targets"].eq(0), "target_coverage_status"
    ] = "no_resolved_human_target"
    return coverage.sort_values("moiety_id", kind="stable").reset_index(drop=True)
