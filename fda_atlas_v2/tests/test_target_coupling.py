import pandas as pd
import pytest

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.target_coupling import (
    estimate_target_trait_coupling,
    unavailable_rhesus_coupling_rows,
    validate_selectivity_primary_coverage,
    validate_target_coupling_semantics,
)


def test_rhesus_coupling_never_reuses_single_donor_kamath_claims():
    selectivity = pd.DataFrame(
        {
            "species": ["macaque"],
            "trait_id": ["pd"],
            "population_id": ["Anxa1"],
            "target_id": ["HGNC:1"],
            "eligible": [True],
            "gene_mapping_status": ["reciprocal_one_to_one_macaque_ortholog"],
            "within_da_status": ["estimated"],
            "within_da_focal_n_cells": [1241],
            "within_da_n_common_donors": [4],
        }
    )
    result = unavailable_rhesus_coupling_rows(
        selectivity, release_id="development"
    ).iloc[0]
    assert result["analysis_partition"] == "allen_hmba_rhesus_control"
    assert result["n_donors"] == 4
    assert result["study_donor_counts_json"] == '{"Allen HMBA":4}'
    assert result["status"] == "unavailable_score"
    assert result["circularity_status"] == "not_evaluated_target_aware_leaveout"
    assert "Macaca mulatta" in result["analysis_unit"]
    assert pd.isna(result["estimate"])


def summaries():
    rows = []
    for target, values, primary in (
        ("HGNC:1", [1.0, 2.0, 3.0, 4.0], True),
        ("HGNC:2", [4.0, 3.0, 2.0, 1.0], True),
    ):
        for index, expression in enumerate(values):
            study = "Kamath" if index < 2 else "Siletti"
            condition = "Ctrl" if study == "Kamath" else "Control"
            rows.append(
                {
                    "species": "human", "trait_id": "pd",
                    "population_id": "Sox6:Tafa1", "target_id": target,
                    "method": "leave_target_out", "donor_id": f"d{index}",
                    "analysis_partition": "human_integrated_control",
                    "analysis_role": "primary",
                    "gene_mapping_status": "human_direct",
                    "expression_mapping_available": True,
                    "score_availability_status": "available",
                    "study": study, "condition": condition,
                    "stratum_id": f"{study}|{condition}|primary",
                    "target_expression": expression, "score": index + 1.0,
                    "n_cells": 10_000 if index == 0 else 20,
                    "circularity_status": "target_removed",
                    "is_primary_method": primary,
                    "expression_representation": "log2_cpm_from_raw_donor_pseudobulk",
                    "score_representation": "scDRS target-removed donor-population mean",
                }
            )
    return pd.DataFrame(rows)


def test_coupling_uses_donors_not_cells_and_is_rank_inert():
    result = estimate_target_trait_coupling(summaries(), release_id="r")
    assert len(result) == 2
    assert result.set_index("target_id").loc["HGNC:1", "estimate"] == pytest.approx(1.0)
    assert result.set_index("target_id").loc["HGNC:2", "estimate"] == pytest.approx(-1.0)
    assert set(result["n_donors"]) == {4}
    assert set(result["n_studies"]) == {2}
    assert set(result["study_donor_counts_json"]) == {
        '{"Kamath":2,"Siletti":2}'
    }
    assert not result["affects_dual_selectivity_rank"].any()
    validate_table("target_trait_coupling", result)


def test_coupling_marks_insufficient_donors_without_inventing_statistics():
    result = estimate_target_trait_coupling(
        summaries()[lambda frame: frame["donor_id"].isin(["d0", "d1"])],
        release_id="r",
    )
    assert set(result["status"]) == {"insufficient_donors"}
    assert result["estimate"].isna().all()


def test_coupling_rejects_a_mismatched_leave_out_status():
    frame = summaries()
    frame["circularity_status"] = "target_chromosome_removed"
    with pytest.raises(ValueError, match="invalid circularity status"):
        estimate_target_trait_coupling(frame, release_id="r")


def test_coupling_accepts_target_absent_from_corrected_baseline():
    frame = summaries()
    frame["circularity_status"] = "target_not_scored_in_corrected_baseline"
    result = estimate_target_trait_coupling(frame, release_id="r")
    assert set(result["circularity_status"]) == {
        "target_not_scored_in_corrected_baseline"
    }


def test_coupling_distinguishes_unavailable_expression_mapping():
    frame = summaries()
    frame["target_expression"] = float("nan")
    frame["expression_mapping_available"] = False
    frame["gene_mapping_status"] = "no_one_to_one_mouse_ortholog"
    result = estimate_target_trait_coupling(frame, release_id="r")
    assert set(result["status"]) == {"unavailable_expression_mapping"}
    assert result["estimate"].isna().all()


def test_coupling_removes_condition_strata_before_association():
    frame = summaries().loc[lambda value: value["target_id"].eq("HGNC:1")].copy()
    frame["stratum_id"] = ["control", "control", "disease", "disease"]
    frame["target_expression"] = [1.0, 2.0, 10.0, 11.0]
    frame["score"] = [1.0, 0.0, 10.0, 9.0]
    result = estimate_target_trait_coupling(frame, release_id="r")
    assert result.loc[0, "estimate"] == pytest.approx(-1.0)
    assert result.loc[0, "n_strata"] == 2
    assert result.loc[0, "residual_degrees_of_freedom"] == 1


def test_coupling_contract_rejects_supporting_evidence_that_changes_rank():
    result = estimate_target_trait_coupling(summaries(), release_id="r")
    result["affects_dual_selectivity_rank"] = True
    with pytest.raises(ValueError, match="cannot alter"):
        validate_table("target_trait_coupling", result)


def test_human_primary_coupling_requires_integrated_kamath_siletti_partition():
    frame = summaries()
    frame["analysis_partition"] = "human_kamath_control"
    with pytest.raises(ValueError, match="human_integrated_control"):
        estimate_target_trait_coupling(frame, release_id="r")


def test_human_integrated_coupling_requires_both_study_strata():
    frame = summaries()
    frame["study"] = "Kamath"
    frame["condition"] = "Ctrl"
    frame["stratum_id"] = "Kamath|Ctrl|primary"
    result = estimate_target_trait_coupling(frame, release_id="r")
    assert set(result["status"]) == {"insufficient_study_support"}
    assert result["estimate"].isna().all()


def test_coupling_semantics_rejects_false_integrated_study_support():
    result = estimate_target_trait_coupling(summaries(), release_id="r")
    result["n_studies"] = 1
    result["study_donor_counts_json"] = '{"Kamath":4}'
    with pytest.raises(ValueError, match="lacks Kamath and Siletti"):
        validate_target_coupling_semantics(result)


def test_coupling_requires_primary_coverage_of_every_selectivity_key():
    result = estimate_target_trait_coupling(summaries(), release_id="r")
    selectivity = result[
        ["release_id", "species", "trait_id", "population_id", "target_id"]
    ].drop_duplicates()
    validate_selectivity_primary_coverage(result, selectivity, release_id="r")

    incomplete = result.loc[result["target_id"].ne("HGNC:2")].copy()
    with pytest.raises(ValueError, match="missing=1, extra=0"):
        validate_selectivity_primary_coverage(
            incomplete,
            selectivity,
            release_id="r",
        )


def test_coupling_coverage_rejects_mixed_release_identity():
    result = estimate_target_trait_coupling(summaries(), release_id="r")
    selectivity = result[
        ["release_id", "species", "trait_id", "population_id", "target_id"]
    ].drop_duplicates()
    selectivity.loc[selectivity.index[0], "release_id"] = "other"

    with pytest.raises(ValueError, match="selectivity coverage release ID"):
        validate_selectivity_primary_coverage(result, selectivity, release_id="r")
