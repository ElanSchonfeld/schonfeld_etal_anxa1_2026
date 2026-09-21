"""Build the cross-entity search index used by the public atlas."""

from __future__ import annotations

import pandas as pd

from .contracts import validate_table


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _append(
    rows: list[dict],
    *,
    release_id: str,
    entity_type: str,
    entity_id: str,
    display_name: str,
    search_term: object,
    source: str,
) -> None:
    term = _clean(search_term)
    if not term:
        return
    rows.append(
        {
            "release_id": release_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "display_name": display_name,
            "search_term": term,
            "source": source,
        }
    )


def build_entity_search(
    release_id: str,
    *,
    drug_target_edges: pd.DataFrame,
    regulatory_substance_target_evidence: pd.DataFrame,
    active_moieties: pd.DataFrame,
    regulatory_substances: pd.DataFrame,
    regulatory_products: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    targets = pd.concat(
        [
            drug_target_edges[["target_id", "human_gene", "hgnc_id"]],
            regulatory_substance_target_evidence[
                ["target_id", "human_gene", "hgnc_id"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates()
    for row in targets.itertuples(index=False):
        display = _clean(row.human_gene) or _clean(row.target_id)
        for term, source in (
            (row.human_gene, "drug_target_edges.human_gene"),
            (row.hgnc_id, "drug_target_edges.hgnc_id"),
            (row.target_id, "drug_target_edges.target_id"),
        ):
            _append(
                rows,
                release_id=release_id,
                entity_type="target",
                entity_id=_clean(row.target_id),
                display_name=display,
                search_term=term,
                source=source,
            )

    for row in active_moieties.itertuples(index=False):
        display = _clean(row.preferred_name) or _clean(row.moiety_id)
        for term, source in (
            (row.preferred_name, "active_moieties.preferred_name"),
            (row.unii, "active_moieties.unii"),
            (row.moiety_id, "active_moieties.moiety_id"),
        ):
            _append(
                rows,
                release_id=release_id,
                entity_type="active_moiety",
                entity_id=_clean(row.moiety_id),
                display_name=display,
                search_term=term,
                source=source,
            )

    for row in regulatory_substances.itertuples(index=False):
        display = _clean(row.preferred_name) or _clean(row.substance_id)
        terms = [
            (row.preferred_name, "regulatory_substances.preferred_name"),
            (row.normalized_name, "regulatory_substances.normalized_name"),
            (row.substance_id, "regulatory_substances.substance_id"),
            (row.matched_gsrs_identity_id, "regulatory_substances.gsrs_identity"),
        ]
        for variant in _clean(row.name_variants).split(" | "):
            terms.append((variant, "regulatory_substances.name_variant"))
        for term, source in terms:
            _append(
                rows,
                release_id=release_id,
                entity_type="regulatory_substance",
                entity_id=_clean(row.substance_id),
                display_name=display,
                search_term=term,
                source=source,
            )

    for row in regulatory_products.itertuples(index=False):
        display = next(
            (
                value
                for value in (
                    _clean(row.proprietary_name),
                    _clean(row.nonproprietary_name),
                    _clean(row.ingredient_text),
                    _clean(row.application_number),
                    _clean(row.product_id),
                )
                if value
            ),
            _clean(row.product_id),
        )
        for term, source in (
            (row.proprietary_name, "regulatory_products.brand_name"),
            (row.nonproprietary_name, "regulatory_products.generic_name"),
            (row.ingredient_text, "regulatory_products.ingredient"),
            (row.application_number, "regulatory_products.application_number"),
            (row.product_id, "regulatory_products.product_id"),
        ):
            _append(
                rows,
                release_id=release_id,
                entity_type="regulatory_product",
                entity_id=_clean(row.product_id),
                display_name=display,
                search_term=term,
                source=source,
            )

    search = pd.DataFrame(rows).drop_duplicates(
        ["release_id", "entity_type", "entity_id", "search_term", "source"]
    )
    search = search.sort_values(
        ["entity_type", "display_name", "entity_id", "source", "search_term"],
        kind="stable",
    ).reset_index(drop=True)
    validate_table("entity_search", search)
    return search


def audit_entity_search_coverage(
    release_id: str,
    search: pd.DataFrame,
    *,
    drug_target_edges: pd.DataFrame,
    regulatory_substance_target_evidence: pd.DataFrame,
    active_moieties: pd.DataFrame,
    regulatory_substances: pd.DataFrame,
    regulatory_products: pd.DataFrame,
) -> tuple[dict, list[str]]:
    sources = {
        "drug_target_edges": drug_target_edges,
        "regulatory_substance_target_evidence": regulatory_substance_target_evidence,
        "active_moieties": active_moieties,
        "regulatory_substances": regulatory_substances,
        "regulatory_products": regulatory_products,
    }
    expected = build_entity_search(release_id, **sources)
    observed = search.reset_index(drop=True)
    issues = []
    exact = list(observed.columns) == list(expected.columns) and observed.equals(expected)
    if not exact:
        issues.append("entity search does not exactly reproduce every frozen search term")
    expected_entities = {
        entity_type: set(
            expected.loc[expected["entity_type"].eq(entity_type), "entity_id"].astype(str)
        )
        for entity_type in (
            "target", "active_moiety", "regulatory_substance", "regulatory_product"
        )
    }
    observed_entities = {
        entity_type: set(
            observed.loc[observed["entity_type"].eq(entity_type), "entity_id"].astype(str)
        )
        for entity_type in expected_entities
    }
    missing = {
        entity_type: len(expected_entities[entity_type] - observed_entities[entity_type])
        for entity_type in expected_entities
    }
    extra = {
        entity_type: len(observed_entities[entity_type] - expected_entities[entity_type])
        for entity_type in expected_entities
    }
    for entity_type in expected_entities:
        if missing[entity_type] or extra[entity_type]:
            issues.append(
                f"entity search {entity_type} coverage differs: "
                f"missing={missing[entity_type]}, extra={extra[entity_type]}"
            )
    return {
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "exact_term_index": exact,
        "expected_entities": {
            key: len(value) for key, value in expected_entities.items()
        },
        "missing_entities": missing,
        "extra_entities": extra,
    }, issues
