import pandas as pd

from fda_atlas_v2.open_targets_genetics import (
    attach_hgnc_and_fda_targets,
    build_approved_hgnc_map,
    classify_biosample_relevance,
    normalize_credible_set_variants,
    normalize_study_payload,
    select_study_candidates,
)


def studies(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "id", "studyType", "pubmedId", "traitFromSource", "nSamples",
            "hasSumstats", "projectId",
        ],
    )


def test_preferred_exact_study_id_wins_without_dropping_pubmed_candidates():
    source = studies(
        [
            ["GCST009325", "gwas", "30957308", "Parkinson disease", 100, True, "GCST"],
            ["OTHER", "gwas", "30957308", "Parkinson subset", 50, False, "GCST"],
        ]
    )
    result = select_study_candidates("pd", "30957308", source)
    assert set(result["id"]) == {"GCST009325", "OTHER"}
    assert result.loc[result["selected_for_evidence"], "id"].tolist() == ["GCST009325"]
    assert result["mapping_status"].eq("selected_exact_study_id").all()


def test_multiple_pubmed_candidates_are_not_selected_automatically():
    source = studies(
        [
            ["A", "gwas", "123", "trait A", 100, True, "X"],
            ["B", "gwas", "123", "trait B", 200, True, "X"],
        ]
    )
    result = select_study_candidates("scz", "123", source)
    assert not result["selected_for_evidence"].any()
    assert result["mapping_status"].eq("requires_manual_study_selection").all()


def test_preferred_study_with_conflicting_pubmed_is_never_selected():
    source = studies(
        [["GCST009325", "gwas", "different", "Parkinson disease", 100, True, "GCST"]]
    )
    result = select_study_candidates("pd", "30957308", source)
    assert not result["selected_for_evidence"].any()
    assert result["candidate_basis"].tolist() == ["exact_study_id_pubmed_conflict"]
    assert result["mapping_status"].tolist() == ["study_id_pubmed_conflict"]


def test_pending_or_absent_pubmed_yields_explicit_empty_candidates():
    source = studies([["A", "gwas", "123", "trait", 100, True, "X"]])
    result = select_study_candidates("anxiety_any", "pending", source)
    assert result.empty
    assert "mapping_status" in result


def test_normalize_study_payload_preserves_l2g_and_gene_colocalisation():
    loci = [
        {
            "studyLocusId": "locus-1",
            "chromosome": "1",
            "position": 123,
            "region": "1:100-200",
            "locusStart": 100,
            "locusEnd": 200,
            "pValueMantissa": 2.5,
            "pValueExponent": -9,
            "beta": 0.1,
            "standardError": 0.02,
            "finemappingMethod": "SuSiE",
            "confidence": "high",
            "qualityControls": [],
            "variant": {"id": "1_123_A_G", "rsIds": ["rs1"]},
            "l2GPredictions": [
                {"score": 0.2, "target": {"id": "ENSG000001.4", "approvedSymbol": "A"}},
                {"score": 0.01, "target": {"id": "ENSG000002", "approvedSymbol": "B"}},
            ],
            "colocalisation": [
                {
                    "rightStudyType": "eqtl",
                    "clpp": 0.8,
                    "h3": 0.1,
                    "h4": 0.9,
                    "betaRatioSignAverage": -1.0,
                    "colocalisationMethod": "COLOC",
                    "otherStudyLocus": {
                        "studyLocusId": "qtl-1",
                        "studyId": "QTL1",
                        "qtlGeneId": "ENSG000001",
                        "study": {
                            "target": {"id": "ENSG000001", "approvedSymbol": "A"},
                            "biosample": {"biosampleId": "UBERON:1", "biosampleName": "brain"},
                        },
                    },
                },
                {
                    "rightStudyType": "gwas",
                    "clpp": 0.7,
                    "colocalisationMethod": "COLOC",
                    "otherStudyLocus": {"studyLocusId": "gwas-1", "studyId": "G2", "study": {}},
                },
            ],
        }
    ]
    locus_frame, evidence = normalize_study_payload(
        release_id="test",
        source_release="26.06",
        trait_id="pd",
        study_id="G1",
        study_trait="Parkinson disease",
        loci=loci,
    )
    assert len(locus_frame) == 1
    assert locus_frame.loc[0, "n_colocalisation_rows"] == 2
    assert len(evidence) == 3
    assert set(evidence["evidence_type"]) == {"locus_to_gene", "colocalisation_eqtl"}
    assert set(evidence["effect_direction"]) == {"not_interpreted"}
    assert not evidence["affects_dual_selectivity_rank"].any()
    low = evidence[evidence["ensembl_gene_id"].eq("ENSG000002")].iloc[0]
    assert low.status == "at_or_below_0.05_support_threshold"
    coloc = evidence[evidence["evidence_type"].eq("colocalisation_eqtl")].iloc[0]
    assert coloc.biosample_relevance == "other"


def test_approved_hgnc_mapping_and_fda_membership_are_explicit():
    hgnc = pd.DataFrame(
        {
            "hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3"],
            "symbol": ["A", "B", "OLD"],
            "status": ["Approved", "Approved", "Entry Withdrawn"],
            "ensembl_gene_id": ["ENSG000001", "ENSG000002.7", "ENSG000003"],
        }
    )
    mapping = build_approved_hgnc_map(hgnc)
    evidence = pd.DataFrame(
        {"ensembl_gene_id": ["ENSG000001", "ENSG000002", "ENSG000003"]}
    )
    result = attach_hgnc_and_fda_targets(evidence, mapping, {"HGNC:1"})
    assert result["target_mapping_status"].tolist() == [
        "fda_target", "not_fda_target", "no_approved_hgnc_mapping"
    ]
    assert result.loc[0, "human_gene"] == "A"


def test_biosample_relevance_uses_ontology_ancestry_not_name_matching():
    assert classify_biosample_relevance(
        {"biosampleId": "X", "biosampleName": "unexpected", "ancestors": ["CL_0000700"]}
    ) == "dopaminergic_neuron"
    assert classify_biosample_relevance(
        {"biosampleId": "X", "biosampleName": "unexpected", "ancestors": ["CL_0000540"]}
    ) == "neuronal"
    assert classify_biosample_relevance(
        {"biosampleId": "X", "biosampleName": "unexpected", "ancestors": ["UBERON_0000955"]}
    ) == "brain_tissue"


def test_credible_set_variant_membership_and_posteriors_are_preserved():
    frame = normalize_credible_set_variants(
        release_id="test", source_release="26.06", trait_id="pd", study_id="G1",
        loci=[{
            "studyLocusId": "L1",
            "credibleSetVariants": [{
                "posteriorProbability": 0.8, "is95CredibleSet": True,
                "is99CredibleSet": True, "logBF": 4.0, "beta": 0.1,
                "standardError": 0.02, "pValueMantissa": 2.0,
                "pValueExponent": -9,
                "variant": {"id": "1_10_A_G", "rsIds": ["rs1"],
                            "chromosome": "1", "position": 10,
                            "referenceAllele": "A", "alternateAllele": "G"},
            }],
        }],
    )
    assert len(frame) == 1
    assert frame.loc[0, "posterior_probability"] == 0.8
    assert frame.loc[0, "is_95_credible_set"]
    assert not frame.loc[0, "affects_dual_selectivity_rank"]
