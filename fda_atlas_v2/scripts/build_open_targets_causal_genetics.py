#!/usr/bin/env python3
"""Acquire Open Targets gene evidence for the matched studies and join it to the FDA target universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_atlas_v2.contracts import TRAITS, validate_table  # noqa: E402
from fda_atlas_v2.open_targets_genetics import (  # noqa: E402
    attach_hgnc_and_fda_targets,
    build_approved_hgnc_map,
    normalize_credible_set_variants,
    normalize_study_payload,
)


ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_RELEASE = ROOT / "release/development"
DATA = Path(os.environ.get("DOPABASE_DATA_ROOT", ROOT / "data")).expanduser()
DEFAULT_CACHE = DATA / "cache/open_targets/26.06/causal_genetics"
DEFAULT_HGNC = DATA / "cache/hgnc/2026-07-07/hgnc_complete_set.txt"
META_QUERY = """
query Meta { meta { name apiVersion { x y z suffix } dataVersion { year month iteration } } }
"""
LOCI_QUERY = """
query Loci($ids: [String!], $page: Pagination) {
  credibleSets(studyIds: $ids, page: $page) {
    count
    rows {
      studyLocusId studyId chromosome position region locusStart locusEnd
      pValueMantissa pValueExponent beta standardError finemappingMethod
      confidence qualityControls
      variant { id rsIds }
    }
  }
}
"""
EVIDENCE_QUERY = """
query Evidence($id: String!, $page: Pagination) {
  credibleSet(studyLocusId: $id) {
    studyLocusId
    l2GPredictions(page: $page) {
      count
      rows { score target { id approvedSymbol approvedName } }
    }
    colocalisation(studyTypes: [gwas], page: $page) {
      count
      rows {
        rightStudyType chromosome clpp betaRatioSignAverage h3 h4 studyLocusId
        numberColocalisingVariants colocalisationMethod
        otherStudyLocus {
          studyLocusId studyId qtlGeneId beta standardError
          pValueMantissa pValueExponent
          study {
            id studyType traitFromSource
            target { id approvedSymbol }
            biosample { biosampleId biosampleName description parents ancestors }
          }
        }
      }
    }
  }
}
"""
QTL_EVIDENCE_QUERY = """
query QtlEvidence($id: String!, $types: [StudyTypeEnum!], $page: Pagination) {
  credibleSet(studyLocusId: $id) {
    studyLocusId
    colocalisation(studyTypes: $types, page: $page) {
      count
      rows {
        rightStudyType chromosome clpp betaRatioSignAverage h3 h4 studyLocusId
        numberColocalisingVariants colocalisationMethod
        otherStudyLocus {
          studyLocusId studyId qtlGeneId beta standardError
          pValueMantissa pValueExponent
          study {
            id studyType traitFromSource
            target { id approvedSymbol }
            biosample { biosampleId biosampleName description parents ancestors }
          }
        }
      }
    }
  }
}
"""
QTL_STUDY_TYPES = [
    "eqtl", "sqtl", "pqtl", "sceqtl", "scsqtl", "scpqtl", "tuqtl", "sctuqtl"
]
VARIANT_QUERY = """
query Variants($id: String!, $page: Pagination) {
  credibleSet(studyLocusId: $id) {
    studyLocusId
    locus(page: $page) {
      count
      rows {
        is99CredibleSet logBF beta is95CredibleSet posteriorProbability
        pValueExponent pValueMantissa r2Overall standardError
        variant {
          id rsIds chromosome position referenceAllele alternateAllele
        }
      }
    }
  }
}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)


def graphql(query: str, variables: dict | None = None, attempts: int = 4) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DopaBase-FDA-atlas/2.0",
        },
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise RuntimeError(f"Open Targets GraphQL errors: {payload['errors']}")
            return payload["data"]
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Open Targets request failed after {attempts} attempts") from last_error


def fetch_loci(study_id: str, page_size: int = 100) -> list[dict]:
    rows = []
    expected = None
    page_index = 0
    while expected is None or len(rows) < expected:
        result = graphql(
            LOCI_QUERY,
            {"ids": [study_id], "page": {"index": page_index, "size": page_size}},
        )["credibleSets"]
        expected = int(result["count"])
        block = result["rows"]
        if not block:
            break
        rows.extend(block)
        page_index += 1
    if len(rows) != expected:
        raise ValueError(f"{study_id} loci expected {expected}, observed {len(rows)}")
    identifiers = [str(row["studyLocusId"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{study_id} API response has duplicate credible sets")
    return rows


def fetch_locus_evidence(locus_id: str, page_size: int = 250) -> dict:
    l2g = []
    coloc = []
    expected_l2g = None
    expected_coloc = None
    page_index = 0
    while (
        expected_l2g is None
        or len(l2g) < expected_l2g
        or len(coloc) < expected_coloc
    ):
        result = graphql(
            EVIDENCE_QUERY,
            {"id": locus_id, "page": {"index": page_index, "size": page_size}},
        )["credibleSet"]
        if result is None or str(result["studyLocusId"]) != locus_id:
            raise ValueError(f"Open Targets did not return credible set {locus_id}")
        l2g_result = result["l2GPredictions"]
        coloc_result = result["colocalisation"]
        expected_l2g = int(l2g_result["count"])
        expected_coloc = int(coloc_result["count"])
        l2g.extend(l2g_result["rows"])
        coloc.extend(coloc_result["rows"])
        page_index += 1
        if page_index > 100:
            raise ValueError(f"pagination did not terminate for {locus_id}")
    if len(l2g) != expected_l2g or len(coloc) != expected_coloc:
        raise ValueError(
            f"{locus_id} nested counts differ: L2G {len(l2g)}/{expected_l2g}; "
            f"colocalisation {len(coloc)}/{expected_coloc}"
        )
    return {
        "studyLocusId": locus_id,
        "L2G_count": expected_l2g,
        "colocalisation_count": expected_coloc,
        "L2GPredictions": l2g,
        "colocalisation": coloc,
    }


def fetch_qtl_colocalisation(locus_id: str, page_size: int = 250) -> dict:
    rows = []
    expected = None
    page_index = 0
    while expected is None or len(rows) < expected:
        result = graphql(
            QTL_EVIDENCE_QUERY,
            {
                "id": locus_id,
                "types": QTL_STUDY_TYPES,
                "page": {"index": page_index, "size": page_size},
            },
        )["credibleSet"]
        if result is None or str(result["studyLocusId"]) != locus_id:
            raise ValueError(f"Open Targets did not return credible set {locus_id}")
        block = result["colocalisation"]
        expected = int(block["count"])
        if not block["rows"]:
            break
        rows.extend(block["rows"])
        page_index += 1
    if len(rows) != expected:
        raise ValueError(
            f"{locus_id} QTL colocalisation expected {expected}, observed {len(rows)}"
        )
    return {"QTL_colocalisation_count": expected, "QTL_colocalisation": rows}


def fetch_credible_set_variants(locus_id: str, page_size: int = 500) -> dict:
    rows = []
    expected = None
    page_index = 0
    while expected is None or len(rows) < expected:
        result = graphql(
            VARIANT_QUERY,
            {"id": locus_id, "page": {"index": page_index, "size": page_size}},
        )["credibleSet"]
        if result is None or str(result["studyLocusId"]) != locus_id:
            raise ValueError(f"Open Targets did not return credible set {locus_id}")
        block = result["locus"]
        expected = int(block["count"])
        if not block["rows"]:
            break
        rows.extend(block["rows"])
        page_index += 1
    if len(rows) != expected:
        raise ValueError(
            f"{locus_id} credible-set variants expected {expected}, observed {len(rows)}"
        )
    return {"credible_set_variant_count": expected, "credibleSetVariants": rows}


def load_or_fetch_study(
    study_id: str,
    cache_dir: Path,
    *,
    refresh: bool,
) -> tuple[list[dict], list[Path]]:
    study_dir = cache_dir / study_id
    loci_path = study_dir / "credible_sets.json"
    if refresh or not loci_path.is_file():
        atomic_json({"study_id": study_id, "rows": fetch_loci(study_id)}, loci_path)
    locus_payload = json.loads(loci_path.read_text())
    if locus_payload.get("study_id") != study_id:
        raise ValueError(f"cached study identifier differs in {loci_path}")
    loci = locus_payload.get("rows", [])
    cache_paths = [loci_path]
    completed = []
    for index, locus in enumerate(loci, start=1):
        locus_id = str(locus["studyLocusId"])
        evidence_path = study_dir / "loci" / f"{locus_id}.json"
        if refresh or not evidence_path.is_file():
            evidence = fetch_locus_evidence(locus_id)
            evidence.update(fetch_qtl_colocalisation(locus_id))
            evidence["QTL_biosample_ontology_fields_version"] = 2
            evidence.update(fetch_credible_set_variants(locus_id))
            evidence["credible_set_variant_fields_version"] = 1
            atomic_json(evidence, evidence_path)
        evidence = json.loads(evidence_path.read_text())
        if (
            "QTL_colocalisation" not in evidence
            or evidence.get("QTL_biosample_ontology_fields_version") != 2
        ):
            evidence.update(fetch_qtl_colocalisation(locus_id))
            evidence["QTL_biosample_ontology_fields_version"] = 2
            atomic_json(evidence, evidence_path)
        if (
            "credibleSetVariants" not in evidence
            or evidence.get("credible_set_variant_fields_version") != 1
        ):
            evidence.update(fetch_credible_set_variants(locus_id))
            evidence["credible_set_variant_fields_version"] = 1
            atomic_json(evidence, evidence_path)
        if evidence.get("studyLocusId") != locus_id:
            raise ValueError(f"cached locus identifier differs in {evidence_path}")
        if len(evidence.get("L2GPredictions", [])) != int(evidence.get("L2G_count", -1)):
            raise ValueError(f"cached L2G count differs in {evidence_path}")
        if len(evidence.get("colocalisation", [])) != int(
            evidence.get("colocalisation_count", -1)
        ):
            raise ValueError(f"cached colocalisation count differs in {evidence_path}")
        if len(evidence.get("QTL_colocalisation", [])) != int(
            evidence.get("QTL_colocalisation_count", -1)
        ):
            raise ValueError(f"cached QTL colocalisation count differs in {evidence_path}")
        if len(evidence.get("credibleSetVariants", [])) != int(
            evidence.get("credible_set_variant_count", -1)
        ):
            raise ValueError(f"cached credible-set variant count differs in {evidence_path}")
        completed.append(
            {
                **locus,
                "l2GPredictions": evidence["L2GPredictions"],
                "gwasColocalisation": evidence["colocalisation"],
                "qtlColocalisation": evidence["QTL_colocalisation"],
                "credibleSetVariants": evidence["credibleSetVariants"],
            }
        )
        cache_paths.append(evidence_path)
        if index % 25 == 0 or index == len(loci):
            print(f"{study_id}: {index}/{len(loci)} loci cached", flush=True)
    return completed, cache_paths


def main(args: argparse.Namespace) -> None:
    release_dir = args.release_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    hgnc_path = args.hgnc.expanduser().resolve()
    for path in (release_dir, hgnc_path):
        if not path.exists():
            raise FileNotFoundError(path)
    candidate_path = release_dir / "open_targets_genetics_study_candidates.parquet"
    study_coverage_path = release_dir / "open_targets_genetics_trait_coverage.parquet"
    moiety_target_path = release_dir / "drug_target_edges.parquet"
    substance_target_path = release_dir / "regulatory_substance_target_evidence.parquet"
    for path in (
        candidate_path, study_coverage_path, moiety_target_path, substance_target_path
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    meta = graphql(META_QUERY)["meta"]
    data_version = meta["dataVersion"]
    observed_release = f"{int(data_version['year'])}.{int(data_version['month']):02d}"
    if observed_release != args.expected_source_release:
        raise ValueError(
            f"Open Targets release expected {args.expected_source_release}, "
            f"observed {observed_release}"
        )
    candidates = pd.read_parquet(candidate_path)
    selected = candidates.loc[candidates["selected_for_evidence"].astype(bool)].copy()
    if selected["trait_id"].duplicated().any():
        raise ValueError("more than one Open Targets study is selected for a trait")
    coverage = pd.read_parquet(study_coverage_path)
    if set(coverage["trait_id"].astype(str)) != set(TRAITS):
        raise ValueError("Open Targets coverage does not contain the frozen 11 traits")

    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    hgnc_map = build_approved_hgnc_map(hgnc)
    fda_targets = set(
        pd.read_parquet(moiety_target_path, columns=["target_id"])["target_id"].astype(str)
    ) | set(
        pd.read_parquet(substance_target_path, columns=["target_id"])["target_id"].astype(str)
    )
    if len(fda_targets) != 1_720:
        raise ValueError(f"FDA target universe expected 1,720, observed {len(fda_targets)}")

    locus_frames = []
    evidence_frames = []
    variant_frames = []
    raw_cache_paths = []
    for row in selected.sort_values("trait_id", kind="stable").itertuples(index=False):
        loci, paths = load_or_fetch_study(
            str(row.id), cache_dir, refresh=args.refresh
        )
        locus_frame, evidence_frame = normalize_study_payload(
            release_id=args.release_id,
            source_release=observed_release,
            trait_id=str(row.trait_id),
            study_id=str(row.id),
            study_trait=str(row.traitFromSource),
            loci=loci,
        )
        locus_frames.append(locus_frame)
        evidence_frames.append(evidence_frame)
        variant_frames.append(
            normalize_credible_set_variants(
                release_id=args.release_id,
                source_release=observed_release,
                trait_id=str(row.trait_id),
                study_id=str(row.id),
                loci=loci,
            )
        )
        raw_cache_paths.extend(paths)

    loci = pd.concat(locus_frames, ignore_index=True)
    all_evidence = pd.concat(evidence_frames, ignore_index=True)
    variants = pd.concat(variant_frames, ignore_index=True).sort_values(
        ["trait_id", "study_id", "locus_id", "posterior_probability", "variant_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    all_evidence = attach_hgnc_and_fda_targets(all_evidence, hgnc_map, fda_targets)
    final = all_evidence.loc[
        all_evidence["gene_identity_status"].eq("source_gene_present")
        & all_evidence["target_mapping_status"].eq("fda_target")
    ].copy()
    final_columns = [
        "release_id", "trait_id", "target_id", "human_gene", "hgnc_id",
        "ensembl_gene_id", "source", "source_release", "study_id", "study_trait",
        "locus_id", "lead_variant_id", "region", "finemapping_method",
        "credible_set_confidence", "evidence_type", "evidence_value",
        "evidence_value_type", "effect_direction", "direction_basis", "biosample_id",
        "biosample_name", "biosample_relevance", "biosample_ontology_ancestors_json",
        "clpp", "h3", "h4", "beta_ratio_sign_average",
        "source_record_id", "status", "affects_dual_selectivity_rank",
    ]
    final = final[final_columns].sort_values(
        ["trait_id", "target_id", "locus_id", "evidence_type", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)
    validate_table("causal_genetics_evidence", final)
    validate_table("causal_genetics_credible_set_variants", variants)

    evidence_coverage_rows = []
    for row in coverage.sort_values("trait_id", kind="stable").itertuples(index=False):
        trait_id = str(row.trait_id)
        trait_loci = loci[loci["trait_id"].eq(trait_id)]
        trait_all = all_evidence[all_evidence["trait_id"].eq(trait_id)]
        trait_final = final[final["trait_id"].eq(trait_id)]
        source_coloc = int(trait_loci.get("n_colocalisation_rows", pd.Series(dtype=int)).sum())
        gene_coloc = int(trait_all["evidence_type"].str.startswith("colocalisation_").sum())
        evidence_coverage_rows.append(
            {
                "release_id": args.release_id,
                "trait_id": trait_id,
                "study_mapping_status": str(row.study_mapping_status),
                "selected_study_id": (
                    str(selected.loc[selected["trait_id"].eq(trait_id), "id"].iloc[0])
                    if selected["trait_id"].eq(trait_id).any() else ""
                ),
                "n_credible_sets": len(trait_loci),
                "n_source_l2g_rows": int(trait_loci.get("n_l2g_rows", pd.Series(dtype=int)).sum()),
                "n_source_colocalisation_rows": source_coloc,
                "n_source_gwas_colocalisation_rows": int(
                    trait_loci.get("n_gwas_colocalisation_rows", pd.Series(dtype=int)).sum()
                ),
                "n_source_qtl_colocalisation_rows": int(
                    trait_loci.get("n_qtl_colocalisation_rows", pd.Series(dtype=int)).sum()
                ),
                "n_gene_linked_colocalisation_rows": gene_coloc,
                "n_all_gene_evidence_rows": len(trait_all),
                "n_approved_hgnc_rows": int(trait_all["hgnc_id"].notna().sum()),
                "n_fda_target_evidence_rows": len(trait_final),
                "n_fda_targets_with_evidence": trait_final["target_id"].nunique(),
                "n_fda_brain_or_neuronal_evidence_rows": int(
                    trait_final["biosample_relevance"].isin(
                        ["dopaminergic_neuron", "neuronal", "brain_tissue", "nervous_system"]
                    ).sum()
                ),
                "evidence_status": (
                    "available" if len(trait_final)
                    else "selected_study_no_fda_target_evidence" if len(trait_loci)
                    else "unavailable_no_exact_pinned_study"
                ),
                "affects_dual_selectivity_rank": False,
            }
        )
    evidence_coverage = pd.DataFrame(evidence_coverage_rows)

    paths = {
        "loci": release_dir / "open_targets_genetics_loci.parquet",
        "all_evidence": release_dir / "open_targets_genetics_gene_evidence_all.parquet",
        "causal": release_dir / "causal_genetics_evidence.parquet",
        "variants": release_dir / "causal_genetics_credible_set_variants.parquet",
        "coverage": release_dir / "open_targets_genetics_evidence_coverage.parquet",
        "manifest": release_dir / "open_targets_causal_genetics_manifest.json",
    }
    for key, frame in (
        ("loci", loci), ("all_evidence", all_evidence), ("causal", final),
        ("variants", variants),
        ("coverage", evidence_coverage),
    ):
        atomic_parquet(frame, paths[key])
    cache_manifest = [
        {"path": str(path), "sha256": sha256(path)} for path in sorted(raw_cache_paths)
    ]
    manifest = {
        "release_id": args.release_id,
        "source": "Open Targets Platform GraphQL API",
        "endpoint": ENDPOINT,
        "source_release": observed_release,
        "api_version": meta["apiVersion"],
        "official_documentation": {
            "genetics": "https://platform-docs.opentargets.org/gentropy",
            "data_access": "https://platform-docs.opentargets.org/data-access/graphql-api",
        },
        "scientific_rules": {
            "exact_pinned_study_required": True,
            "all_l2g_scores_preserved": True,
            "l2g_support_threshold": 0.05,
            "all_source_colocalisations_counted": True,
            "all_credible_set_variants_and_posteriors_preserved": True,
            "qtl_study_types": QTL_STUDY_TYPES,
            "target_rows_require_gene_linked_colocalisation": True,
            "biosample_relevance_uses_source_ontology_ancestors": True,
            "approved_hgnc_mapping_required": True,
            "release_table_restricted_to_fda_target_universe": True,
            "effect_direction_interpreted": False,
            "affects_dual_selectivity_rank": False,
        },
        "inputs": {
            "study_candidates": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
            "study_coverage": {"path": str(study_coverage_path), "sha256": sha256(study_coverage_path)},
            "hgnc": {"path": str(hgnc_path), "sha256": sha256(hgnc_path)},
            "moiety_targets": {"path": str(moiety_target_path), "sha256": sha256(moiety_target_path)},
            "substance_targets": {"path": str(substance_target_path), "sha256": sha256(substance_target_path)},
            "raw_api_cache": cache_manifest,
        },
        "results": {
            "traits": len(evidence_coverage),
            "traits_with_exact_study": len(selected),
            "traits_without_exact_study": len(evidence_coverage) - len(selected),
            "credible_sets": len(loci),
            "source_l2g_rows": int(loci["n_l2g_rows"].sum()),
            "source_colocalisation_rows": int(loci["n_colocalisation_rows"].sum()),
            "source_gwas_colocalisation_rows": int(
                loci["n_gwas_colocalisation_rows"].sum()
            ),
            "source_qtl_colocalisation_rows": int(
                loci["n_qtl_colocalisation_rows"].sum()
            ),
            "gene_linked_evidence_rows": len(all_evidence),
            "fda_target_evidence_rows": len(final),
            "fda_targets_with_evidence": final["target_id"].nunique(),
            "fda_brain_or_neuronal_evidence_rows": int(
                final["biosample_relevance"].isin(
                    ["dopaminergic_neuron", "neuronal", "brain_tissue", "nervous_system"]
                ).sum()
            ),
            "credible_set_variant_rows": len(variants),
            "credible_set_95_members": int(variants["is_95_credible_set"].sum()),
            "credible_set_99_members": int(variants["is_99_credible_set"].sum()),
        },
        "tables": {
            key: {"path": str(paths[key]), "rows": len(frame), "sha256": sha256(paths[key])}
            for key, frame in (
                ("loci", loci), ("all_evidence", all_evidence), ("causal", final),
                ("variants", variants),
                ("coverage", evidence_coverage),
            )
        },
        "pipeline_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(ROOT / "src/fda_atlas_v2/contracts.py"): sha256(
                ROOT / "src/fda_atlas_v2/contracts.py"
            ),
            str(ROOT / "src/fda_atlas_v2/open_targets_genetics.py"): sha256(
                ROOT / "src/fda_atlas_v2/open_targets_genetics.py"
            ),
        },
    }
    atomic_json(manifest, paths["manifest"])
    print(json.dumps(manifest["results"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hgnc", type=Path, default=DEFAULT_HGNC)
    parser.add_argument("--release-id", default="development")
    parser.add_argument("--expected-source-release", default="26.06")
    parser.add_argument("--refresh", action="store_true")
    main(parser.parse_args())
