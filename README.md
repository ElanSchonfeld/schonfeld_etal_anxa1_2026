# ANXA1-associated dopaminergic neuron analyses

This repository contains the analysis and figure code for the Awatramani Lab
manuscript on ANXA1-associated dopaminergic neurons, Parkinson disease vulnerability, projection topography, DaTscan
depletion, and DopaBase Human.

## Scope

The repository covers:

- Figure 1: mouse hierarchical mixture-of-experts classifier and validation.
- Figure 2: transfer of the classifier to Kamath and Siletti human dopamine
  neurons.
- Figure 3: lesion and human disease-vulnerability analyses.
- Figure 4: mouse, human, Kraft Slide-tags, and rhesus projection analyses.
- Figure 5: DaTscan depletion, staging, genotype, trajectory, spatial-null,
  and conversion-generalization analyses.
- Figure 6: DopaBase Human and its FDA Drug Target Atlas.
- The released HMoE inference model and DopaBase Human atlas-building lineage.

## Repository map

| Path | Contents |
| --- | --- |
| `figures/fig1_hmoe` | Mouse HMoE classification and Salmani validation |
| `figures/fig2_human` | Kamath and Siletti human transfer analyses |
| `figures/fig3_vulnerability` | Disease vulnerability and depletion |
| `figures/fig4_projections` | Cross-species projection analyses |
| `figures/fig5_datscan_depletion` | PPMI DaTscan analyses |
| `figures/fig6_dopabase_human` | DopaBase Human manuscript panels |
| `hmoe_annotate` | Standalone HMoE inference tool and model |
| `human_integration` | DopaBase Human atlas build |
| `fda_atlas_v2` | FDA Drug Target Atlas analysis core and tests |
| `common` | Shared figure and provenance helpers |

Each figure directory has a README or manifest listing its display scripts,
compute scripts, and required frozen inputs.

## Reproduction model

The repository separates three kinds of material:

1. GitHub contains source code, tests, small configuration files, and the HMoE
   model.
2. The derived inputs under `figures/.../frozen/` are written by the scripts in
   each figure's `compute/` directory. A small number of inputs listed in the
   figure READMEs and manifests have no producer here and are available from
   the authors on request.
3. Raw or controlled third-party data remain at their source repositories or
   access portals. In particular, participant-level PPMI records and imaging
   data are not redistributed.

Generated panels, caches, and large expression or imaging objects are not
tracked. One rendered panel is tracked:
`figures/fig5_datscan_depletion/output/panels/fig5_conversion_generalises.png`.

## Environment and commands

`environment.yml` installs Python 3.10.19 and the Python package versions
reported in the manuscript. Install R 4.5.0 separately; `R_PACKAGES.txt` lists
the R package versions.

After installing the environment and unpacking the frozen-data overlay:

```bash
make test
make fig1
make fig2
make fig3
make fig4
make fig5
make fig6
```

The `make figN` targets render the figure panels from the frozen inputs. The
scripts under each figure's `compute/` directory rebuild those inputs from
authorized source data; see the figure README for the required inputs.

## Pinned DopaBase Human deployment

- Space: `elanschonfeld/DopaBase-Human`, revision
  `4b698678f094f50ec3154c9a47c36b2f0354c9e8`
- Dataset: `elanschonfeld/dopabase-human-data`, revision
  `37257a55528df3c1cf7eb762c31a66def851a7e2`

The deployed application is preserved at those Hugging Face revisions and is
not duplicated in this repository.

## License

Original code and Awatramani Lab-generated HMoE weights in this repository are
released under the [MIT License](LICENSE), copyright Awatramani Lab,
Northwestern University. External datasets, third-party software, and other
separately distributed resources retain their own terms and licenses.
