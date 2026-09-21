"""Study matching for Open Targets genetics evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


TRAIT_DISEASE_IDS = {
    "pd": "MONDO_0005180",
    "scz": "MONDO_0005090",
    "adhd": "MONDO_0007743",
    "nicotine_dep": "MONDO_0008575",
    "bipolar": "MONDO_0004985",
    "chronic_pain": "HP_0012532",
    "mdd": "MONDO_0002009",
    "oud": "EFO_0010702",
    "ocd": "MONDO_0008114",
    "anxiety_any": "MONDO_0005618",
    "fibromyalgia": "MONDO_0005546",
}


def require_disease_id(trait_id: str) -> str:
    """Ontology id for a trait, or a loud failure if it was never curated."""
    value = TRAIT_DISEASE_IDS.get(trait_id)
    if value is None:
        raise KeyError(
            f"trait {trait_id!r} has no curated Open Targets ontology id; "
            "set TRAIT_DISEASE_IDS[{0!r}] before running causal-genetics steps".format(trait_id)
        )
    return value

PREFERRED_STUDY_IDS = {
    "pd": {"GCST009325"},
    "nicotine_dep": {"GCST90104535"},
    "mdd": {"GCST005839"},
}

L2G_SUPPORT_THRESHOLD = 0.05
BIOSAMPLE_RELEVANCE = frozenset(
    {"dopaminergic_neuron", "neuronal", "brain_tissue", "nervous_system", "other", "not_available"}
)


def select_study_candidates(
    trait_id: str,
    pinned_pubmed_id: str,
    studies: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "id", "studyType", "pubmedId", "traitFromSource", "nSamples",
        "hasSumstats", "projectId",
    }
    if missing := sorted(required - set(studies.columns)):
        raise ValueError(f"Open Targets study rows are missing: {missing}")
    source = studies.loc[studies["studyType"].astype(str).eq("gwas")].copy()
    source["id"] = source["id"].astype(str)
    source["pubmedId"] = source["pubmedId"].fillna("").astype(str)
    preferred = PREFERRED_STUDY_IDS.get(trait_id, set())
    preferred_rows = source[source["id"].isin(preferred)].copy()
    pubmed = str(pinned_pubmed_id).strip()
    pinned_pubmed_available = bool(pubmed and pubmed.lower() != "pending")
    pubmed_rows = (
        source[source["pubmedId"].eq(pubmed)].copy()
        if pinned_pubmed_available
        else source.iloc[0:0].copy()
    )
    candidates = pd.concat([preferred_rows, pubmed_rows], ignore_index=True).drop_duplicates("id")
    if candidates.empty:
        columns = list(source.columns) + [
            "trait_id", "disease_id", "candidate_basis", "mapping_status",
            "selected_for_evidence",
        ]
        return pd.DataFrame(columns=columns)
    candidates["trait_id"] = trait_id
    candidates["disease_id"] = require_disease_id(trait_id)
    preferred_conflict_ids = set(
        preferred_rows.loc[
            pinned_pubmed_available & preferred_rows["pubmedId"].ne(pubmed), "id"
        ].astype(str)
    )
    candidates["candidate_basis"] = candidates.apply(
        lambda row: (
            "exact_study_id_pubmed_conflict"
            if row["id"] in preferred_conflict_ids
            else "exact_study_id"
            if row["id"] in preferred
            else "exact_pubmed_id"
        ),
        axis=1,
    )
    if preferred_conflict_ids:
        selected_id = ""
        status = "study_id_pubmed_conflict"
    elif len(preferred_rows) == 1:
        selected_id = str(preferred_rows.iloc[0]["id"])
        status = "selected_exact_study_id"
    elif len(candidates) == 1:
        selected_id = str(candidates.iloc[0]["id"])
        status = "selected_unique_exact_pubmed"
    else:
        selected_id = ""
        status = "requires_manual_study_selection"
    candidates["mapping_status"] = status
    candidates["selected_for_evidence"] = candidates["id"].eq(selected_id)
    return candidates.sort_values("id", kind="stable").reset_index(drop=True)


def build_approved_hgnc_map(hgnc: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-Ensembl map restricted to approved HGNC records."""

    required = {"hgnc_id", "symbol", "status", "ensembl_gene_id"}
    if missing := sorted(required - set(hgnc.columns)):
        raise ValueError(f"HGNC table is missing: {missing}")
    frame = hgnc.loc[
        hgnc["status"].astype(str).eq("Approved")
        & hgnc["ensembl_gene_id"].notna(),
        ["ensembl_gene_id", "hgnc_id", "symbol"],
    ].copy()
    frame["ensembl_gene_id"] = (
        frame["ensembl_gene_id"].astype(str).str.split(".", regex=False).str[0]
    )
    frame = frame[frame["ensembl_gene_id"].str.startswith("ENSG")]
    counts = frame.groupby("ensembl_gene_id", sort=False)["hgnc_id"].transform("nunique")
    frame["n_approved_hgnc_mappings"] = counts.astype(int)
    frame["hgnc_mapping_status"] = np.where(
        counts.eq(1), "unique_approved_hgnc_mapping", "ambiguous_approved_hgnc_mapping"
    )
    ambiguous = frame["hgnc_mapping_status"].eq("ambiguous_approved_hgnc_mapping")
    frame.loc[ambiguous, ["hgnc_id", "symbol"]] = pd.NA
    return (
        frame.drop_duplicates("ensembl_gene_id")
        .sort_values("ensembl_gene_id", kind="stable")
        .reset_index(drop=True)
    )


def _source_record_id(prefix: str, values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return f"ot:{prefix}:" + hashlib.sha256(payload.encode()).hexdigest()


def _value(value):
    return np.nan if value is None else value


def classify_biosample_relevance(biosample: dict) -> str:
    """Classify QTL context from source ontology identity and ancestors."""

    identifiers = {
        str(value).replace(":", "_")
        for value in [biosample.get("biosampleId"), *(biosample.get("ancestors") or [])]
        if value
    }
    if not identifiers:
        return "not_available"
    if "CL_0000700" in identifiers:
        return "dopaminergic_neuron"
    if "CL_0000540" in identifiers:
        return "neuronal"
    if "UBERON_0000955" in identifiers:
        return "brain_tissue"
    if {"UBERON_0001016", "UBERON_0001017"} & identifiers:
        return "nervous_system"
    return "other"


def normalize_study_payload(
    *,
    release_id: str,
    source_release: str,
    trait_id: str,
    study_id: str,
    study_trait: str,
    loci: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize all loci and gene-linked L2G/colocalisation source rows."""

    locus_rows = []
    evidence_rows = []
    for locus in loci:
        if "gwasColocalisation" in locus or "qtlColocalisation" in locus:
            gwas_colocalisation = locus.get("gwasColocalisation") or []
            qtl_colocalisation = locus.get("qtlColocalisation") or []
            colocalisation = [*gwas_colocalisation, *qtl_colocalisation]
        else:
            colocalisation = locus.get("colocalisation") or []
            gwas_colocalisation = [
                row for row in colocalisation
                if str(row.get("rightStudyType") or "").lower() == "gwas"
            ]
            qtl_colocalisation = [
                row for row in colocalisation
                if str(row.get("rightStudyType") or "").lower() != "gwas"
            ]
        locus_id = str(locus["studyLocusId"])
        variant = locus.get("variant") or {}
        lead_variant_id = str(variant.get("id") or "")
        locus_base = {
            "release_id": release_id,
            "trait_id": trait_id,
            "study_id": study_id,
            "study_trait": study_trait,
            "locus_id": locus_id,
            "lead_variant_id": lead_variant_id,
            "chromosome": str(locus.get("chromosome") or ""),
            "position": _value(locus.get("position")),
            "region": str(locus.get("region") or ""),
            "locus_start": _value(locus.get("locusStart")),
            "locus_end": _value(locus.get("locusEnd")),
            "p_value_mantissa": _value(locus.get("pValueMantissa")),
            "p_value_exponent": _value(locus.get("pValueExponent")),
            "beta": _value(locus.get("beta")),
            "standard_error": _value(locus.get("standardError")),
            "finemapping_method": str(locus.get("finemappingMethod") or ""),
            "credible_set_confidence": str(locus.get("confidence") or ""),
            "quality_controls_json": json.dumps(
                locus.get("qualityControls") or [], sort_keys=True, separators=(",", ":")
            ),
            "n_l2g_rows": len(locus.get("l2GPredictions") or []),
            "n_colocalisation_rows": len(colocalisation),
            "n_gwas_colocalisation_rows": len(gwas_colocalisation),
            "n_qtl_colocalisation_rows": len(qtl_colocalisation),
        }
        locus_rows.append(locus_base)

        for prediction in locus.get("l2GPredictions") or []:
            target = prediction.get("target") or {}
            ensembl = str(target.get("id") or "").split(".")[0]
            if not ensembl:
                continue
            score = float(prediction["score"])
            identity = {
                "trait_id": trait_id,
                "study_id": study_id,
                "locus_id": locus_id,
                "ensembl_gene_id": ensembl,
                "evidence_type": "locus_to_gene",
                "score": score,
            }
            evidence_rows.append(
                {
                    **locus_base,
                    "ensembl_gene_id": ensembl,
                    "source_gene_symbol": str(target.get("approvedSymbol") or ""),
                    "source": "Open Targets Platform",
                    "source_release": source_release,
                    "evidence_type": "locus_to_gene",
                    "evidence_value": score,
                    "evidence_value_type": "L2G score",
                    "effect_direction": "not_interpreted",
                    "direction_basis": (
                        "L2G prioritization does not establish therapeutic direction"
                    ),
                    "biosample_id": "",
                    "biosample_name": "",
                    "biosample_relevance": "not_available",
                    "biosample_ontology_ancestors_json": "[]",
                    "right_study_type": "",
                    "other_study_id": "",
                    "other_locus_id": "",
                    "colocalisation_method": "",
                    "clpp": np.nan,
                    "h3": np.nan,
                    "h4": np.nan,
                    "beta_ratio_sign_average": np.nan,
                    "source_record_id": _source_record_id("l2g", identity),
                    "status": (
                        "above_0.05_support_threshold"
                        if score > L2G_SUPPORT_THRESHOLD
                        else "at_or_below_0.05_support_threshold"
                    ),
                    "gene_identity_status": "source_gene_present",
                    "affects_dual_selectivity_rank": False,
                }
            )

        for coloc in colocalisation:
            other = coloc.get("otherStudyLocus") or {}
            other_study = other.get("study") or {}
            other_target = other_study.get("target") or {}
            qtl_gene = str(other.get("qtlGeneId") or "").split(".")[0]
            target_gene = str(other_target.get("id") or "").split(".")[0]
            genes = {value for value in (qtl_gene, target_gene) if value}
            if not genes:
                continue
            gene_status = "source_gene_present" if len(genes) == 1 else "source_gene_conflict"
            ensembl = sorted(genes)[0]
            biosample = other_study.get("biosample") or {}
            clpp = _value(coloc.get("clpp"))
            h4 = _value(coloc.get("h4"))
            if pd.notna(clpp):
                evidence_value = float(clpp)
                evidence_value_type = "CLPP"
            elif pd.notna(h4):
                evidence_value = float(h4)
                evidence_value_type = "H4 posterior probability"
            else:
                evidence_value = np.nan
                evidence_value_type = "not_available"
            evidence_type = "colocalisation_" + str(
                coloc.get("rightStudyType") or "unknown"
            ).lower()
            identity = {
                "trait_id": trait_id,
                "study_id": study_id,
                "locus_id": locus_id,
                "other_locus_id": str(other.get("studyLocusId") or ""),
                "ensembl_gene_id": ensembl,
                "evidence_type": evidence_type,
                "method": str(coloc.get("colocalisationMethod") or ""),
            }
            evidence_rows.append(
                {
                    **locus_base,
                    "ensembl_gene_id": ensembl,
                    "source_gene_symbol": str(other_target.get("approvedSymbol") or ""),
                    "source": "Open Targets Platform",
                    "source_release": source_release,
                    "evidence_type": evidence_type,
                    "evidence_value": evidence_value,
                    "evidence_value_type": evidence_value_type,
                    "effect_direction": "not_interpreted",
                    "direction_basis": (
                        "colocalisation is gene-prioritization evidence, not treatment direction"
                    ),
                    "biosample_id": str(biosample.get("biosampleId") or ""),
                    "biosample_name": str(biosample.get("biosampleName") or ""),
                    "biosample_relevance": classify_biosample_relevance(biosample),
                    "biosample_ontology_ancestors_json": json.dumps(
                        biosample.get("ancestors") or [],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "right_study_type": str(coloc.get("rightStudyType") or ""),
                    "other_study_id": str(other.get("studyId") or ""),
                    "other_locus_id": str(other.get("studyLocusId") or ""),
                    "colocalisation_method": str(coloc.get("colocalisationMethod") or ""),
                    "clpp": clpp,
                    "h3": _value(coloc.get("h3")),
                    "h4": h4,
                    "beta_ratio_sign_average": _value(coloc.get("betaRatioSignAverage")),
                    "source_record_id": _source_record_id("coloc", identity),
                    "status": "gene_linked_colocalisation",
                    "gene_identity_status": gene_status,
                    "affects_dual_selectivity_rank": False,
                }
            )
    loci_frame = pd.DataFrame(locus_rows)
    evidence_frame = pd.DataFrame(evidence_rows)
    if not loci_frame.empty and loci_frame["locus_id"].duplicated().any():
        raise ValueError(f"{study_id} contains duplicate credible-set identifiers")
    if not evidence_frame.empty and evidence_frame["source_record_id"].duplicated().any():
        raise ValueError(f"{study_id} contains duplicate normalized genetics records")
    return loci_frame, evidence_frame


def attach_hgnc_and_fda_targets(
    evidence: pd.DataFrame,
    hgnc_map: pd.DataFrame,
    fda_target_ids: set[str],
) -> pd.DataFrame:
    """Attach approved HGNC identities and explicit FDA-universe membership."""

    frame = evidence.merge(hgnc_map, on="ensembl_gene_id", how="left", validate="many_to_one")
    frame["target_id"] = frame["hgnc_id"]
    mapped = frame["hgnc_id"].notna()
    frame["target_mapping_status"] = np.where(
        frame["hgnc_mapping_status"].eq("ambiguous_approved_hgnc_mapping"),
        "ambiguous_approved_hgnc_mapping",
        np.where(
            ~mapped,
            "no_approved_hgnc_mapping",
            np.where(frame["target_id"].isin(fda_target_ids), "fda_target", "not_fda_target"),
        ),
    )
    frame = frame.rename(columns={"symbol": "human_gene"})
    return frame


def normalize_credible_set_variants(
    *,
    release_id: str,
    source_release: str,
    trait_id: str,
    study_id: str,
    loci: list[dict],
) -> pd.DataFrame:
    """Normalize every variant and posterior from each selected credible set."""

    rows = []
    for locus in loci:
        locus_id = str(locus["studyLocusId"])
        for record in locus.get("credibleSetVariants") or []:
            variant = record.get("variant") or {}
            variant_id = str(variant.get("id") or "")
            if not variant_id:
                raise ValueError(f"{locus_id} credible-set member lacks a variant ID")
            identity = {
                "trait_id": trait_id,
                "study_id": study_id,
                "locus_id": locus_id,
                "variant_id": variant_id,
            }
            rows.append(
                {
                    "release_id": release_id,
                    "trait_id": trait_id,
                    "study_id": study_id,
                    "locus_id": locus_id,
                    "variant_id": variant_id,
                    "rs_ids_json": json.dumps(
                        variant.get("rsIds") or [], sort_keys=True, separators=(",", ":")
                    ),
                    "chromosome": str(variant.get("chromosome") or ""),
                    "position": _value(variant.get("position")),
                    "reference_allele": str(variant.get("referenceAllele") or ""),
                    "alternate_allele": str(variant.get("alternateAllele") or ""),
                    "posterior_probability": float(record["posteriorProbability"]),
                    "is_95_credible_set": bool(record["is95CredibleSet"]),
                    "is_99_credible_set": bool(record["is99CredibleSet"]),
                    "log_bayes_factor": _value(record.get("logBF")),
                    "beta": _value(record.get("beta")),
                    "standard_error": _value(record.get("standardError")),
                    "p_value_mantissa": _value(record.get("pValueMantissa")),
                    "p_value_exponent": _value(record.get("pValueExponent")),
                    "source": "Open Targets Platform",
                    "source_release": source_release,
                    "source_record_id": _source_record_id("credible_set_variant", identity),
                    "affects_dual_selectivity_rank": False,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if frame.duplicated(["locus_id", "variant_id"]).any():
            raise ValueError("duplicate variant within an Open Targets credible set")
        probability = pd.to_numeric(frame["posterior_probability"], errors="coerce")
        if probability.isna().any() or not probability.between(0, 1).all():
            raise ValueError("credible-set posterior probabilities must be finite in [0, 1]")
        if (frame["is_95_credible_set"] & ~frame["is_99_credible_set"]).any():
            raise ValueError("95 percent credible-set membership must imply 99 percent membership")
    return frame
