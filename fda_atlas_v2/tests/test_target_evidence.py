import pandas as pd

from fda_atlas_v2.target_evidence import (
    build_drug_target_edges,
    build_drug_target_edges_with_audit,
    build_moiety_target_coverage,
    build_substance_target_edges,
    build_substance_evidence_bridge,
    build_substance_target_coverage,
    legacy_name_key,
)


def _target_evidence():
    return pd.DataFrame(
        {
            "regulatory_substance_id": ["old1", "old2"],
            "human_gene": ["GENE1", "GENE2"],
            "hgnc_id": ["HGNC:1", "HGNC:2"],
            "evidence_tier": ["T1", "T1"],
            "source_database": ["DB", "DB"],
            "source_release": ["v", "v"],
            "source_record_key": ["e1", "e2"],
            "evidence_kind": ["direct_mechanism", "direct_mechanism"],
            "action_type": ["AGONIST", "INHIBITOR"],
            "target_scope": ["single_protein", "single_protein"],
            "drug_match_confidence": ["high", "high"],
            "tier_reason": ["x", "x"],
            "source_family": ["DB", "DB"],
            "source_drug_id": ["d1", "d2"],
            "source_drug_name": ["drug1", "drug2"],
            "source_target_id": ["t1", "t2"],
            "source_target_name": ["target1", "target2"],
            "source_target_type": ["protein", "protein"],
            "target_organism": ["Homo sapiens", "Homo sapiens"],
            "primary_target_annotation": [True, True],
            "mechanism_description": ["x", "x"],
            "drug_match_method": ["exact", "exact"],
            "n_convergent_direct_source_families": [2, 2],
        }
    )


def test_exact_v2_names_are_not_collapsed_by_the_legacy_bridge():
    new = pd.DataFrame(
        {
            "release_id": ["r", "r"],
            "substance_id": ["s1", "s2"],
            "preferred_name": ["Factor (Human)", "Factor (Recombinant)"],
        }
    )
    old = pd.DataFrame(
        {
            "regulatory_substance_id": ["old1"],
            "regulatory_substance_norm": ["FACTOR"],
        }
    )
    bridge = build_substance_evidence_bridge(new, old)

    assert legacy_name_key("Factor (Human)") == "FACTOR"
    assert len(bridge) == 2
    assert bridge["substance_id"].nunique() == 2
    assert set(bridge["bridge_status"]) == {
        "legacy_key_split_across_exact_v2_substances"
    }


def test_target_edges_require_explicit_moieties_and_retain_zero_target_coverage():
    bridge = pd.DataFrame(
        {
            "release_id": ["r", "r"],
            "substance_id": ["s1", "s2"],
            "regulatory_substance_id": ["old1", "old2"],
            "bridge_status": ["exact_legacy_key", "exact_legacy_key"],
        }
    )
    substance_moiety = pd.DataFrame(
        {"substance_id": ["s1"], "moiety_id": ["m1"]}
    )
    evidence = _target_evidence()
    substance_edges = build_substance_target_edges(bridge, evidence)
    assert set(substance_edges["target_id"]) == {"HGNC:1", "HGNC:2"}
    substances = pd.DataFrame(
        {
            "release_id": ["r", "r", "r"],
            "substance_id": ["s1", "s2", "s3"],
            "preferred_name": ["one", "two", "three"],
            "identity_status": ["resolved", "resolved", "unresolved"],
            "matched_gsrs_identity_id": ["g1", "g2", None],
            "has_explicit_active_moiety_relationship": [True, False, False],
        }
    )
    substance_coverage = build_substance_target_coverage(substances, substance_edges)
    assert len(substance_coverage) == 3
    assert substance_coverage.set_index("substance_id").loc[
        "s3", "target_coverage_status"
    ] == "no_resolved_human_target"
    edges = build_drug_target_edges(substance_edges, substance_moiety)
    assert list(edges["target_id"]) == ["HGNC:1"]

    moieties = pd.DataFrame(
        {
            "release_id": ["r", "r"],
            "moiety_id": ["m1", "m2"],
            "preferred_name": ["one", "two"],
            "resolution_class": ["explicit", "explicit"],
            "unii": ["u1", "u2"],
        }
    )
    coverage = build_moiety_target_coverage(moieties, edges)
    assert len(coverage) == 2
    assert coverage.set_index("moiety_id").loc["m2", "target_coverage_status"] == (
        "no_resolved_human_target"
    )


def test_ambiguous_legacy_splits_and_multi_moiety_cartesian_edges_are_withheld():
    split_bridge = pd.DataFrame(
        {
            "release_id": ["r", "r"],
            "substance_id": ["split1", "split2"],
            "regulatory_substance_id": ["old1", "old1"],
            "bridge_status": [
                "legacy_key_split_across_exact_v2_substances",
                "legacy_key_split_across_exact_v2_substances",
            ],
        }
    )
    assert build_substance_target_edges(split_bridge, _target_evidence()).empty

    exact_bridge = pd.DataFrame(
        {
            "release_id": ["r"],
            "substance_id": ["s1"],
            "regulatory_substance_id": ["old1"],
            "bridge_status": ["exact_legacy_key"],
        }
    )
    substance_edges = build_substance_target_edges(exact_bridge, _target_evidence())
    links = pd.DataFrame(
        {"substance_id": ["s1", "s1"], "moiety_id": ["m1", "m2"]}
    )
    moieties = pd.DataFrame(
        {
            "moiety_id": ["m1", "m2"],
            "preferred_name": ["drug1", "unrelated component"],
        }
    )
    edges, withheld = build_drug_target_edges_with_audit(
        substance_edges, links, moieties
    )
    assert edges[["moiety_id", "target_id"]].to_dict("records") == [
        {"moiety_id": "m1", "target_id": "HGNC:1"}
    ]
    assert withheld[["moiety_id", "target_id"]].to_dict("records") == [
        {"moiety_id": "m2", "target_id": "HGNC:1"}
    ]
    assert set(withheld["moiety_attribution_status"]) == {
        "ambiguous_multi_moiety_attribution_withheld"
    }
