"""Strict direct single-gene FDA target universe."""

from __future__ import annotations

import hashlib

import pandas as pd


DIRECT_LABEL_TARGETS = (
    ("sevabertinib", "ERBB2", "HGNC:3430", "INHIBITOR"),
    ("icotrokinra", "IL23R", "HGNC:19100", "ANTAGONIST"),
    ("eplontersen", "TTR", "HGNC:12405", "ANTISENSE INHIBITOR"),
    ("tofersen", "SOD1", "HGNC:11179", "ANTISENSE INHIBITOR"),
    ("nusinersen", "SMN2", "HGNC:11118", "SPLICING MODULATOR"),
    ("risdiplam", "SMN2", "HGNC:11118", "SPLICING MODULATOR"),
    ("nedosiran", "LDHA", "HGNC:6535", "RNA INTERFERENCE INHIBITOR"),
    ("mipomersen", "APOB", "HGNC:603", "ANTISENSE INHIBITOR"),
    ("inclisiran", "PCSK9", "HGNC:20001", "RNA INTERFERENCE INHIBITOR"),
    ("acoltremon", "TRPM8", "HGNC:17961", "AGONIST"),
)


def primary_direct_single_gene(frame: pd.DataFrame) -> pd.DataFrame:
    """Return direct primary single-protein evidence only."""

    required = {
        "evidence_tier",
        "primary_target_annotation",
        "evidence_kind",
        "target_scope",
        "target_id",
        "human_gene",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"target evidence is missing columns: {missing}")
    return frame[
        frame["evidence_tier"].astype(str).isin(("T1", "T2"))
        & frame["primary_target_annotation"].astype(bool)
        & frame["evidence_kind"].astype(str).eq("direct_mechanism")
        & frame["target_scope"].astype(str).eq("single_protein")
    ].copy()


def build_direct_label_edges(
    active_moieties: pd.DataFrame,
    *,
    release_id: str = "development",
) -> pd.DataFrame:
    required = {"moiety_id", "preferred_name"}
    if missing := sorted(required - set(active_moieties.columns)):
        raise ValueError(f"active moieties are missing columns: {missing}")
    lookup = (
        active_moieties.assign(
            _name=active_moieties["preferred_name"].astype(str).str.casefold()
        )
        .set_index("_name")["moiety_id"]
        .to_dict()
    )
    records = []
    for drug, gene, target_id, action in DIRECT_LABEL_TARGETS:
        moiety_id = lookup.get(drug.casefold())
        if not moiety_id:
            raise ValueError(f"direct-label drug lacks an active moiety: {drug}")
        identity = f"{moiety_id}|{target_id}|fda_label_direct"
        records.append(
            {
                "release_id": release_id,
                "moiety_id": str(moiety_id),
                "target_id": target_id,
                "evidence_source": "FDA prescribing information:2026-07-20",
                "evidence_kind": "direct_mechanism",
                "action_type": action,
                "directness": "direct_mechanism",
                "target_scope": "single_protein",
                "confidence": "high",
                "source_record_id": "FDA_LABEL_DIRECT:"
                + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "substance_id": None,
                "bridge_status": "exact_active_moiety",
                "human_gene": gene,
                "hgnc_id": target_id,
                "evidence_tier": "T2",
                "tier_reason": "direct single-gene mechanism stated in FDA prescribing information",
                "source_database": "FDA prescribing information",
                "source_release": "2026-07-20",
                "source_family": "FDA label",
                "source_drug_id": str(moiety_id),
                "source_drug_name": drug,
                "source_target_id": target_id,
                "source_target_name": gene,
                "source_target_type": "SINGLE PROTEIN",
                "target_organism": "Homo sapiens",
                "primary_target_annotation": True,
                "mechanism_description": "FDA label direct molecular mechanism",
                "drug_match_method": "exact_active_moiety",
                "drug_match_confidence": "high",
                "n_convergent_direct_source_families": 1,
                "moiety_attribution_status": "single_explicit_moiety",
            }
        )
    result = pd.DataFrame(records)
    if result.duplicated(["moiety_id", "target_id"]).any():
        raise ValueError("direct-label evidence contains duplicate drug-target pairs")
    return result


def add_direct_label_edges(
    drug_target_edges: pd.DataFrame,
    direct_label_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Add nonduplicated direct-label pairs while preserving the source schema."""

    missing = sorted(set(drug_target_edges.columns) - set(direct_label_edges.columns))
    if missing:
        raise ValueError(f"direct-label evidence is missing edge columns: {missing}")
    additions = direct_label_edges.reindex(columns=drug_target_edges.columns)
    existing_pairs = set(
        zip(
            drug_target_edges["moiety_id"].astype(str),
            drug_target_edges["target_id"].astype(str),
        )
    )
    additions = additions[
        [
            (str(row.moiety_id), str(row.target_id)) not in existing_pairs
            for row in additions.itertuples(index=False)
        ]
    ]
    result = pd.concat([drug_target_edges, additions], ignore_index=True)
    return result.drop_duplicates().reset_index(drop=True)


def direct_target_pairs(
    drug_target_edges: pd.DataFrame,
    regulatory_substance_target_evidence: pd.DataFrame,
) -> pd.DataFrame:
    selected = pd.concat(
        [
            primary_direct_single_gene(drug_target_edges),
            primary_direct_single_gene(regulatory_substance_target_evidence),
        ],
        ignore_index=True,
    )[["target_id", "human_gene"]].drop_duplicates()
    conflicts = selected.groupby("target_id")["human_gene"].nunique()
    if (conflicts > 1).any():
        raise ValueError("one direct target ID maps to multiple human genes")
    selected = selected.sort_values("target_id", kind="stable").reset_index(drop=True)
    if len(selected) != 533:
        raise ValueError(f"strict direct target universe must contain 533 genes, found {len(selected)}")
    return selected
