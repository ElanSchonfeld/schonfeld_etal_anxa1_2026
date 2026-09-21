# Figure 2: HMoE in human dopamine neurons

This directory contains the display code for the Kamath and Siletti HMoE
transfer panels and the compute scripts that build their frozen inputs.

## Display code

| Script | Output under `output/panels/` | Inputs under `frozen/` |
| --- | --- | --- |
| `code/panel_kamath_1x2_family.py` | `kamath_1x2_family_all` | `kamath_percell.parquet` |
| `code/panel_kamath_2x3.py` | `kamath_2x3_all` | `kamath_percell.parquet`, `Kamath_alra_tafa1_markers.csv`, `Kamath_alra_imputed.csv` |
| `code/panel_kamath_family_confusion.py` | `kamath_family_confusion` | `kamath_percell.parquet` |
| `code/panel_benchmark_roc_auc.py` | `benchmark_roc_auc` | `benchmark_fair_accuracy.csv` |
| `code/panel_kamath_height5.py` | `height_5` | `kamath_percell.parquet`, `kamath_height5_partition.json` |
| `code/panel_kamath_groundtruth_sankey.py` | `kamath_groundtruth_subtype_sankey` | `kamath_percell.parquet` |
| `code/panel_kamath_subtype_heatmap.py` | `kamath_subtype_heatmap` | `kamath_heatmap_matrix.parquet`, `kamath_heatmap_agg.parquet`, `kamath_heatmap_meta.json` |
| `code/panel_kamath_scanvi_vs_hmoe.py` | `kamath_umap_scanvi_vs_hmoe` | `kamath_percell.parquet`, `scanvi_subtype_predictions.csv` |
| `code/panel_siletti_umap.py` | `siletti_umap` | `siletti_percell.parquet` |
| `code/panel_kamath_confidence.py` | `kamath_hmoe_confidence` | `kamath_percell.parquet`, `kamath_hmoe_softprobs.parquet`, `scanvi_subtype_softprobs.csv` |
| `code/panel_kamath_gad2_exclusion.py` | `kamath_gad2_exclusion` | `kamath_gad2_percell.parquet`, `kamath_gad2_stats.json` |
| `code/assemble_fig2.py` | `output/fig2_main`, `output/fig2_supp` montages | `output/panels/*.png` |

`code/kamath_common.py` holds the shared palettes and helpers.

Build the `frozen/` inputs with the scripts under `compute/`, then run from
the repository root:

```bash
make fig2
```

## Compute code

| Script | Output under `frozen/` | Inputs |
| --- | --- | --- |
| `compute/build_kamath_dopamine.py` | `Data/Human/Kamath/Human_Kamath_Dopamine.h5ad` | CZ CELLxGENE Discover dataset version `a41c9e65-1abd-428b-aa0a-1d11474bfbe7.h5ad`, fetched by URL into `Data/Human/Kamath/` and checked against a recorded SHA-256; `Data/Human/Kamath/Human_Kamath_Cell_Metadata.tsv` (Broad Single Cell Portal SCP1768) |
| `compute/build_kamath_da_predicted.py` | `Data/Human/Kamath/Kamath_DA_predicted.h5ad` | `Data/Human/Kamath/Human_Kamath_Dopamine.h5ad`, `hmoe_annotate/model/` |
| `compute/extract_kamath_percell.py` | `kamath_percell.parquet` | `Kamath_DA_predicted.h5ad` |
| `compute/build_kamath_hmoe_softprobs.py` | `kamath_hmoe_softprobs.parquet` | `Kamath_DA_predicted.h5ad` |
| `compute/build_height5_partition.py` | `kamath_height5_partition.json` | `hmoe_annotate/model/v4_improved_unified/meta.json` |
| `compute/build_kamath_heatmap_matrix.py` | `kamath_heatmap_matrix.parquet`, `kamath_heatmap_agg.parquet`, `kamath_heatmap_meta.json` | `Kamath_DA_predicted.h5ad` |
| `compute/freeze_fig2_gad2.py` | `kamath_gad2_percell.parquet`, `kamath_gad2_stats.json` | `Kamath_DA_predicted.h5ad` |
| `compute/build_scanvi_subtype_transfer.py` | `scanvi_subtype_predictions.csv`, `scanvi_subtype_softprobs.csv` | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad` |
| `compute/run_alra_kamath.R` | `Kamath_alra_imputed.csv`, `Kamath_alra_tafa1_markers.csv` | `Data/Human/Kamath/Human_Kamath_Dopamine.h5ad` |
| `compute/benchmark/benchmark_common.py` | none, imported by the `run_*.py` scripts | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad` |
| `compute/benchmark/run_classical.py` | `benchmark/classical.csv` | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad` |
| `compute/benchmark/run_hmoe.py` | `benchmark/hmoe.csv` | `Kamath_DA_predicted.h5ad`, `hmoe_annotate/model/` |
| `compute/benchmark/run_singlecellnet.py` | `benchmark/singlecellnet.csv` | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad` |
| `compute/benchmark/run_scvi_scanvi.py` | `benchmark/scvi_scanvi.csv` | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad` |
| `compute/benchmark/run_scgpt.py` | `benchmark/scgpt.csv` | `Data/Mouse/Mouse_LRRK2_Dopamine.h5ad`, `Kamath_DA_predicted.h5ad`, `Data/Mouse_Human_Gene_Orthologs.tsv`, scGPT `scGPT_human` checkpoint directory in `SCGPT_MODEL_DIR` |
| `compute/benchmark/build_benchmark_table.py` | `benchmark_fair_accuracy.csv` | `frozen/benchmark/classical.csv`, `hmoe.csv`, `singlecellnet.csv`, `scvi_scanvi.csv`, `scgpt.csv`, or the same files under `BENCHMARK_RESULTS_DIR` |
| `compute/build_siletti_groundtruth.py` | `siletti_groundtruth.csv` | `human_integration/results/siletti_da_whb_crosswalk.csv`, Siletti `subcluster_annotation.xlsx` and `cluster_to_cluster_annotation_membership.csv` |
| `compute/extract_siletti_percell.py` | `siletti_percell.parquet` | `human_integration/data/siletti_da_neurons.h5ad`, `human_integration/results/siletti_hmoe_unified.csv`, `frozen/siletti_groundtruth.csv` |

Set `M2H_SOURCE_ROOT` to a source workspace containing
`Data/Human/Kamath/Kamath_DA_predicted.h5ad` and, for
`build_siletti_groundtruth.py`, `Data/Human/Siletti/`. Run each script from
the repository root, for example:

```bash
export M2H_SOURCE_ROOT=/path/to/source-workspace
python figures/fig2_human/compute/extract_kamath_percell.py
```

## Transformations

For the marker heatmaps, raw Kamath counts are library-size normalized to
10,000 counts per cell and log1p transformed before marker selection and
z-scaling. Siletti ground-truth labels come from the CELLxGENE Census
`soma_joinid` crosswalk built by
`human_integration/scripts/00d_siletti_whb_crosswalk.py`.
