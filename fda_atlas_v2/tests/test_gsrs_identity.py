import json

import pandas as pd

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.gsrs_identity import build_gsrs_identity_release
from fda_atlas_v2.regulatory_sources import regulatory_substance_id


def _substances():
    names = ["DRUG A SALT", "DRUG B", "UNRESOLVED PRODUCT"]
    return pd.DataFrame(
        {
            "release_id": ["test"] * 3,
            "substance_id": [regulatory_substance_id(name) for name in names],
            "preferred_name": names,
            "normalized_name": [name.casefold() for name in names],
            "identity_status": ["pending_gsrs_identity"] * 3,
            "identity_source": ["exact_regulatory_name"] * 3,
            "exact_name_key": [name.casefold() for name in names],
        }
    )


def _query(query_id, name, *, selected, target=None, role="active_ingredient"):
    return {
        "query_id": query_id,
        "name_role": role,
        "regulatory_name_exact_key": name.casefold(),
        "match_status": "resolved" if selected else "unresolved",
        "match_method": "exact_unique_name" if selected else "unresolved",
        "match_confidence": "high" if selected else "none",
        "candidate_selected": str(selected),
        "gsrs_record_id": f"record-{name}" if selected else "",
        "gsrs_uuid": f"uuid-{name}" if selected else "",
        "unii": f"UNII-{name}" if selected else "",
        "explicit_active_moiety_targets_json": json.dumps(
            [target] if target is not None else []
        ),
        "gsrs_source_version": "2026-07-06",
    }


def test_explicit_active_moieties_are_materialized_without_self_inference():
    target = {
        "type": "ACTIVE MOIETY",
        "target_unii": "ACTIVEA",
        "target_uuid": "active-a-uuid",
        "target_linking_id": "ACTIVEA",
        "target_name": "DRUG A",
        "target_substance_class": "chemical",
    }
    crosswalk = pd.DataFrame(
        [
            _query("q1", "DRUG A SALT", selected=True, target=target),
            _query(
                "q2", "DRUG A SALT", selected=True, target=target,
                role="proper_name"
            ),
            _query("q3", "DRUG B", selected=True),
            _query("q4", "UNRESOLVED PRODUCT", selected=False),
        ]
    )
    result = build_gsrs_identity_release(
        _substances(), crosswalk, release_id="test"
    )

    assert len(result.identity_edges) == 3
    assert len(result.active_moieties) == 1
    assert len(result.active_moiety_edges) == 1
    assert result.audit["inferred_self_edges"] == 0
    assert result.active_moiety_edges.iloc[0]["moiety_id"] == "gsrs:ACTIVEA"
    assert result.active_moiety_edges.iloc[0]["query_ids"] == "q1 | q2"

    status = result.substances.set_index("preferred_name")["identity_status"]
    assert status["DRUG A SALT"] == "resolved_explicit_active_moiety"
    assert status["DRUG B"] == "resolved_no_explicit_active_moiety_relationship"
    assert status["UNRESOLVED PRODUCT"] == "unresolved_identity"

    validate_table("regulatory_substances", result.substances)
    validate_table("substance_identity_edges", result.identity_edges)
    validate_table("active_moieties", result.active_moieties)
    validate_table("substance_active_moiety_edges", result.active_moiety_edges)


def test_conflicting_selected_identities_for_one_exact_name_fail():
    substances = _substances().iloc[[0]].copy()
    first = _query("q1", "DRUG A SALT", selected=True)
    second = _query("q2", "DRUG A SALT", selected=True, role="proper_name")
    second["unii"] = "DIFFERENT"
    crosswalk = pd.DataFrame([first, second])

    try:
        build_gsrs_identity_release(substances, crosswalk, release_id="test")
    except ValueError as exc:
        assert "multiple identities" in str(exc)
    else:
        raise AssertionError("conflicting GSRS identities were accepted")
