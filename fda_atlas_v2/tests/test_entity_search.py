import pandas as pd

from fda_atlas_v2.entity_search import (
    audit_entity_search_coverage,
    build_entity_search,
)


def test_entity_search_covers_target_moiety_brand_generic_and_application():
    inputs = {
        "drug_target_edges": pd.DataFrame(
            {"target_id": ["HGNC:1"], "human_gene": ["DRD2"], "hgnc_id": ["HGNC:1"]}
        ),
        "regulatory_substance_target_evidence": pd.DataFrame(
            {"target_id": ["HGNC:2"], "human_gene": ["DRD3"], "hgnc_id": ["HGNC:2"]}
        ),
        "active_moieties": pd.DataFrame(
            {"moiety_id": ["m1"], "preferred_name": ["DOPAMINE"], "unii": ["VTD58H1Z2X"]}
        ),
        "regulatory_substances": pd.DataFrame(
            {
                "substance_id": ["s1"],
                "preferred_name": ["ROPINIROLE"],
                "normalized_name": ["ropinirole"],
                "name_variants": ["ROPINIROLE | Ropinirole HCl"],
                "matched_gsrs_identity_id": ["g1"],
            }
        ),
        "regulatory_products": pd.DataFrame(
            {
                "product_id": ["p1"],
                "proprietary_name": ["REQUIP"],
                "nonproprietary_name": ["ROPINIROLE"],
                "ingredient_text": ["ROPINIROLE HYDROCHLORIDE"],
                "application_number": ["NDA020658"],
            }
        ),
    }
    search = build_entity_search("r", **inputs)
    terms = set(search["search_term"])
    assert {"DRD2", "DRD3", "VTD58H1Z2X", "REQUIP", "ROPINIROLE", "NDA020658"} <= terms
    assert set(search["entity_type"]) == {
        "target", "active_moiety", "regulatory_substance", "regulatory_product"
    }
    invariants, issues = audit_entity_search_coverage("r", search, **inputs)
    assert not issues
    assert invariants["exact_term_index"]
    assert invariants["missing_entities"] == {
        "target": 0,
        "active_moiety": 0,
        "regulatory_substance": 0,
        "regulatory_product": 0,
    }
    incomplete = search.loc[search["search_term"].ne("NDA020658")].reset_index(drop=True)
    _, issues = audit_entity_search_coverage("r", incomplete, **inputs)
    assert "entity search does not exactly reproduce every frozen search term" in issues
