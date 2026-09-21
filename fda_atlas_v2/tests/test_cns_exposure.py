import pandas as pd
import pytest
from rdkit import Chem

from fda_atlas_v2.cns_exposure import (
    build_b3db_entity_evidence,
    materialize_b3db_contexts,
    prepare_b3db_records,
)


INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
INCHIKEY = Chem.InchiToInchiKey(INCHI)


def b3db_frames():
    common = {
        "NO.": [1], "compound_name": ["ethanol"], "CID": [702],
        "logBB": [-0.3], "Inchi": [INCHI], "reference": ["R1|"],
        "group": ["A"], "comments": [""],
    }
    return pd.DataFrame({**common, "BBB+/BBB-": ["BBB+"]}), pd.DataFrame(common)


def entity_evidence():
    classification, regression = b3db_frames()
    records = prepare_b3db_records(classification, regression)
    structures = pd.DataFrame(
        {"id": [7], "name": ["ethanol"], "inchikey": [INCHIKEY]}
    )
    edges = pd.DataFrame(
        {
            "entity_type": ["active_moiety"], "entity_id": ["gsrs:test"],
            "drugcentral_struct_id": [7],
            "identity_status": ["exact_source_id_current_name"],
        }
    )
    return build_b3db_entity_evidence(
        "development", records, structures, edges, "b3db:test"
    )


def test_b3db_bridge_requires_exact_structure_identity_and_preserves_axes():
    matches, evidence = entity_evidence()
    assert len(matches) == 2
    assert set(evidence["evidence_type"]) == {
        "curated_bbb_classification", "measured_numeric_logbb"
    }
    assert evidence["match_method"].eq("exact_stereochemical_inchikey").all()
    assert not evidence["affects_dual_selectivity_rank"].any()


def test_b3db_numeric_records_must_equal_categorical_group_a():
    classification, regression = b3db_frames()
    classification.loc[0, "group"] = "B"
    with pytest.raises(ValueError, match="numeric records"):
        prepare_b3db_records(classification, regression)


def test_b3db_contexts_repeat_native_evidence_across_targets_and_traits():
    _, evidence = entity_evidence()
    moiety_edges = pd.DataFrame(
        {"moiety_id": ["gsrs:test"], "target_id": ["HGNC:1"]}
    )
    substance_edges = pd.DataFrame(columns=["substance_id", "target_id"])
    contexts = materialize_b3db_contexts(
        "development", evidence, moiety_edges, substance_edges
    )
    assert len(contexts) == 22
    assert contexts["trait_id"].nunique() == 11
    assert contexts["context_type"].eq("cns_exposure").all()
    assert set(contexts["status"]) == {
        "available_curated_bbb_classification", "available_measured_logbb"
    }
    assert contexts["value"].str.contains("brain/blood").sum() == 11
    assert not contexts["affects_dual_selectivity_rank"].any()
