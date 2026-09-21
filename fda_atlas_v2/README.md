# FDA Drug Target Atlas within DopaBase Human

This directory contains the analysis code for the FDA Drug Target Atlas
within DopaBase Human. It links regulatory drug identities and targets to
dopaminergic cell populations, trait scores, cross-species evidence, and
within-DA and brain-wide target selectivity.

## Layout

- `src/fda_atlas_v2/`: analysis and release-table functions.
- `scripts/`: command-line builders.
- `configs/`: cohort, trait, source, and population specifications.
- `tests/`: unit and integration tests with synthetic inputs.

## Builders

| Script | Output content |
| --- | --- |
| `scripts/build_regulatory_source_universe.py` | pinned Drugs@FDA and Purple Book source universe |
| `scripts/build_gsrs_identity_release.py` | regulatory identity and active-moiety tables |
| `scripts/build_drug_target_activity_release.py` | DrugCentral activity evidence |
| `scripts/build_polypharmacology_action_release.py` | target-action and portfolio summaries per drug identity |
| `scripts/build_openfda_label_release.py` | normalized drug-grain openFDA label tables |
| `scripts/build_open_targets_causal_genetics.py` | Open Targets gene evidence joined to the FDA target universe |
| `scripts/build_pending_drug_contexts.py` | context coverage for every entity-target-trait tuple |
| `scripts/build_dataset_registry.py` | source-dataset registry |
| `scripts/build_release_registries.py` | trait and DA-population registries |
| `scripts/build_cell_partition_audit.py` | DA/non-DA partition table |
| `scripts/build_scdrs_population_effects.py` | donor-aware scDRS population effects for human and mouse |
| `scripts/append_anxa1_population_effects.py` | merged Anxa1 focal population appended to the scDRS effects |
| `scripts/build_hmba_rhesus_fda.py` | rhesus HMBA focal-DA evidence |
| `scripts/build_cross_species_effects.py` | same-target, same-trait cross-species evidence |
| `scripts/build_target_trait_coupling.py` | donor-level target to leave-out scDRS coupling evidence |
| `scripts/rebuild_direct_target_selectivity.py` | 533-target selectivity tables |
| `scripts/rebuild_within_da_rest.py` | within-DA selectivity axis, focal versus pooled rest of DA |
| `scripts/recompute_brainwide_all_da_donors.py` | human brain-wide selectivity axis using all DA donors |
| `scripts/rebuild_direct_target_downstream.py` | edge-dependent tables for the direct-target atlas |
| `scripts/build_entity_search.py` | search index for genes, drugs, and products |
| `scripts/build_data_dictionary.py` | machine-readable data dictionary |
| `scripts/build_source_citations.py` | source-citation and reuse-terms registry |

Source citations and pinned source releases are recorded in
`configs/source_citations.json` and in manifests produced by the builders.

## Environment and tests

Use the repository-root `environment.yml`, which pins Python 3.10.19 and the
Python package versions. From the repository root, run:

```bash
make test
```

The suite contains 29 test files and 145 tests. The tests use small synthetic
fixtures and need no external data.

## Inputs and paths

Raw single-cell matrices and generated release tables are not distributed in
this directory. Builders accept explicit path arguments where their
command-line interface exposes them. Shared defaults are repository-relative
and can be overridden with these environment variables:

- `DOPABASE_DATA_ROOT`: raw inputs, caches, and large derived checkpoints.
- `DOPABASE_FDA_ROOT`: this package root when resolving configured release
  outputs. It defaults to the directory containing this README.
- `DOPABASE_SOURCE_ROOT`: upstream analysis inputs that are not
  distributed in this package.
- `DOPABASE_EXTERNAL_ROOT`: mounted storage root checked before large jobs.
- `DOPABASE_HMOE_ROOT`: HMoE inference package, defaulting to the sibling
  `../hmoe_annotate` directory.

```bash
export DOPABASE_DATA_ROOT=/path/to/data
export DOPABASE_SOURCE_ROOT=/path/to/source-workspace
export DOPABASE_EXTERNAL_ROOT=/path/to/mounted-storage
python scripts/build_scdrs_population_effects.py --help
```

The rhesus HMBA builder uses the HMoE API and model from `hmoe_annotate`. Its
input matrix must contain raw nonnegative integer counts; the model computes
its deep-gate SCT representation internally.
