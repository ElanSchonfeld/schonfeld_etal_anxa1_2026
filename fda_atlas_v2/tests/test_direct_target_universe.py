import pandas as pd

from fda_atlas_v2.direct_target_universe import (
    add_direct_label_edges,
    build_direct_label_edges,
    primary_direct_single_gene,
)


def test_primary_direct_single_gene_excludes_family_and_secondary_rows():
    rows = pd.DataFrame(
        {
            "target_id": ["HGNC:1", "HGNC:2", "HGNC:3"],
            "human_gene": ["A", "B", "C"],
            "evidence_tier": ["T2", "T5", "T3"],
            "primary_target_annotation": [True, True, False],
            "evidence_kind": ["direct_mechanism"] * 3,
            "target_scope": ["single_protein", "family_group_component", "single_protein"],
        }
    )
    observed = primary_direct_single_gene(rows)
    assert observed["human_gene"].tolist() == ["A"]


def test_direct_label_edges_add_only_missing_drug_target_pairs():
    moieties = pd.DataFrame(
        {
            "moiety_id": [
                "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"
            ],
            "preferred_name": [
                "sevabertinib", "icotrokinra", "eplontersen", "tofersen",
                "nusinersen", "risdiplam", "nedosiran", "mipomersen",
                "inclisiran", "acoltremon",
            ],
        }
    )
    additions = build_direct_label_edges(moieties)
    existing = additions.iloc[[0]].copy()
    observed = add_direct_label_edges(existing, additions)
    assert len(observed) == 10
    assert set(observed["human_gene"]) >= {"IL23R", "SOD1", "SMN2", "LDHA", "APOB"}
    assert not observed["target_scope"].eq("family_group_component").any()
