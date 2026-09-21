# Figure 4: cross-species dopamine projection maps

This directory contains the Figure 4 and Supplementary Figure 4 display code, the
manuscript-statistics script, and the compute scripts that build their frozen inputs.

Most inputs under `frozen/` are written by the compute scripts in this directory from source
data located by `M2H_SOURCE_ROOT` and `M2H_DATA_ROOT`. The inputs listed under External inputs
below have no producer here and are available from the authors on request.

## Display code

| Manuscript role | Script | Input under `frozen/` |
|---|---|---|
| Main mouse projection heatmap | `code/panel_B_mouse.py` | `panelB_mouse_{matrix,sig}.csv` |
| Main mouse anatomical enrichment | `code/panel_mouse_enrichment_series.py` | `mouse_enrichment_fields.npz`, `mouse_enrichment_{meta,structures,scales}.csv` |
| Main anterograde SNc eYFP/mCherry projection-tracing validation | `code/panel_mouse_tracing_validation.py` | `mouse_tracing_panelC.pdf`, `mouse_tracing_panelD.pdf` |
| Main human projection heatmap | `code/panel_A_human.py` | `panelA_human_{matrix,sig}.csv` |
| Main extended human atlas | `compute/render_extended_arm.py`, `BV_ARM=arm2_matched` | `extended_arm/` inputs listed below |
| Supplementary Salmani replication | `code/panel_D_salmani.py` | `panelD_salmani_{matrix,sig}.csv` |
| Supplementary Kraft map | `code/panel_F_kraft.py` | `panelF_kraft_{matrix,sig}.csv` |
| Supplementary rhesus macaque map | `code/panel_G_macaque.py` | `panelG_macaque_{matrix,sig}.csv` |
| Supplementary extended human atlas | `compute/render_extended_arm.py`, `BV_ARM=arm1_matched` | `extended_arm/` inputs listed below |
| Supplementary LR drivers | `code/supp_lr_drivers.py` | five `lr_drivers/*_magnitude.tsv` tables |
| Manuscript statistics | `code/manuscript_stats.py` | `stats/{LRRK2,Salmani}_{dlvm_com,superbins}.csv`, `panelB_mouse_{matrix,sig}.csv`, `panelD_salmani_{matrix,sig}.csv` |
| Preview montages `output/fig4_main`, `output/fig4_supp` | `code/assemble_fig4.py` | `output/panels/*.png`, including both extended-atlas PNGs |

`code/fig4_common.py`, `code/heatmap.py`, and `code/significance.py` are shared modules.
`code/manuscript_stats.py` computes the exact three-group Jonckheere-Terpstra test and the
per-target family contrasts.

The five heatmaps use three driver groups: Anxa1-associated (`Sox6:Tafa1` and `Sox6:Vcan`),
all six Sox6 subtypes, and all ten Calb1 subtypes. Significance uses 5,000 label permutations
with Benjamini-Hochberg correction across receiver regions within each sender family; a cell is
marked when its per-family q value is below 0.05.

After installing the repository environment and building the `frozen/` inputs with the compute
scripts, run from the repository root:

```bash
make fig4
```

or run the scripts individually:

```bash
python figures/fig4_projections/code/panel_B_mouse.py
python figures/fig4_projections/code/panel_mouse_enrichment_series.py
python figures/fig4_projections/code/panel_mouse_tracing_validation.py
python figures/fig4_projections/code/panel_A_human.py
python figures/fig4_projections/code/panel_D_salmani.py
python figures/fig4_projections/code/panel_F_kraft.py
python figures/fig4_projections/code/panel_G_macaque.py
python figures/fig4_projections/code/supp_lr_drivers.py
python figures/fig4_projections/code/manuscript_stats.py
```

Outputs are written under `output/`. `make fig4-stats` runs `manuscript_stats.py` alone.
`make fig4-preview` runs `make fig4` and then `assemble_fig4.py`; run it after the extended-atlas
renders below.

## Extended human atlases

`compute/render_extended_arm.py` is the entry point for the two horizontal bivariate atlases. It
uses the modules under `compute/extended_arm/` and these files:

```text
figures/fig4_projections/frozen/extended_arm/
  fig11b_panelA_constrained_display_targets.tsv
  fig11b_panelA_constrained_targets.tsv
  fig11b_method_tangram_uniform_density_targets.tsv
  projection_matrix_human.tsv
  bin_metadata.csv
  arm1_matched_panelA_locked_bins.parquet
  arm2_matched_panelA_locked_bins.parquet
```

The human projection analysis uses Allen HMBA-BG 10x expression together with the HMBA-BG
MERSCOPE spatial reference. Arm 1 uses that Allen reference alone. Arm 2 adds Kraft Slide-tags
when fitting the combined COVET niche model. Both read the MERSCOPE CCF coordinates, lock region
values to the broad human panel, display Caudate, Putamen, and NAc, and mirror the
donor-sampled left-hemisphere field to the right hemisphere. Color encodes centered
within-structure texture; saturation encodes a shared cross-column magnitude.

The renders require a locally installed BrainGlobe `allen_human_500um` atlas:

```bash
BV_ARM=arm2_matched python figures/fig4_projections/compute/render_extended_arm.py
BV_ARM=arm1_matched python figures/fig4_projections/compute/render_extended_arm.py
```

The commands write PNG and PDF files under `output/panels/`.

## Compute code

The compute scripts read source data located by two environment variables and write staged
tables under `work/` and frozen tables under `frozen/`:

- `M2H_SOURCE_ROOT`: workspace holding `Data/Human/Kamath/Kamath_DA_predicted.h5ad`,
  `Data/Mouse/Mouse_LRRK2_Dopamine_RAW.h5ad`,
  `Data/Mouse/YaghmaeianSalmani/YaghmaeianSalmani_authorDA_predicted.h5ad`,
  `cell2cell_final_v2/results/stage1/cache/wholebrain_target_structures.h5ad`,
  `cell2cell_final_v2/data/senders/lrrk2_da_senders_spatial_fixed.h5ad`,
  `cell2cell_final_v2/results/stage2_finalized_nonspatial/parafac_projection_matrix_full18.tsv`, and
  `fMRI/projection_atlas/canonical/h4_bin_435_metadata.csv`.
- `M2H_DATA_ROOT`: directory holding `human/atlas/allen/basal_ganglia/multiome/`,
  `macaque/allen/`, and `human/spatial/kraft_macosko/`.

| Script | Writes | Inputs |
|---|---|---|
| `compute/mouse/build_receiver_bins.py` | `work/mouse/receiver_bins_ccf.csv`, `work/mouse/recv_bin_expr.npz`, `work/mouse/recv_bin_genes.json` | `M2H_SOURCE_ROOT` whole-brain MERFISH cache |
| `compute/mouse/build_mouse_projection.py` | `work/mouse/allenFinal_bin_level_{LRRK2,Salmani}.tsv`, `work/mouse/allenFinal_receiver_bins.csv` | `work/mouse/` receiver bins, the two mouse sender h5ads, the sender gene list, LIANA `mouseconsensus` |
| `compute/refreeze_mouse_centroid_ap.py` | `panelB_mouse_{matrix,sig}.csv`, `panelD_salmani_{matrix,sig}.csv` | `work/mouse/` projections, sender h5ads (cell counts) |
| `compute/mouse/build_enrichment_fields.py` | `mouse_enrichment_fields.npz`, `mouse_enrichment_{meta,structures,scales}.csv` | `work/mouse/` projection and receiver bins, `M2H_SOURCE_ROOT` ventral target matrix, BrainGlobe `allen_mouse_25um` |
| `compute/freeze_manuscript_stats.py` | `stats/{LRRK2,Salmani}_{dlvm_com,superbins}.csv` | `work/mouse/` projections, sender h5ads (cell counts) |
| `compute/human/prepare_human_inputs.py` | `work/human/{senders,receivers}_pseudobulk.csv`, `work/human/bin_metadata.csv`, `work/human/receiver_genes.json`, `work/human/sender_subtype_counts.json` | Kamath h5ad, HMBA-BG human 10x multiome h5ad and its sidecar tables |
| `compute/human/select_lr_pairs.py` | `work/<name>/lr_pairs_filtered.csv` | `work/<name>/` pseudobulks, LIANA `consensus` |
| `compute/human/build_finalized_844.py` | `work/human/finalized_844/*`, `work/human/constrained_targets.tsv`, `work/human/constrained_spatial_bins.csv` | `work/human/` inputs, `h4_bin_435_metadata.csv` |
| `compute/human/freeze_extended_arm_targets.py` | `extended_arm/fig11b_panelA_constrained_targets.tsv`, `extended_arm/fig11b_panelA_constrained_display_targets.tsv` | `work/human/constrained_targets.tsv` |
| `compute/kraft/prepare_kraft_inputs.py` | `work/kraft/{senders,receivers}_pseudobulk.csv`, `work/kraft/{bin_metadata.csv,receiver_genes.json,lr_pairs_filtered.csv,sender_subtype_counts.json}` | Kamath h5ad, 19 Kraft Slide-tags h5ads, `work/human/lr_pairs_filtered.csv` |
| `compute/kraft/kraft_spatialbins.py` | `work/kraft/projection_matrix_spatialbins_liana_hier.tsv`, `work/kraft/anchored_mass_liana_hier.tsv`, `work/kraft/anchored_bins_liana.csv`, `work/kraft/liana_foldin_factors.npz` | `work/kraft/` inputs, 19 Kraft Slide-tags h5ads |
| `compute/macaque/build_macaque_senders.py` | `work/macaque/senders_pseudobulk.csv`, `work/macaque/sender_subtype_counts.json` | HMBA-BG macaque h5ad, joint-taxonomy sidecars, `hmoe_annotate/model` |
| `compute/macaque/build_macaque_receivers.py` | `work/macaque/{receivers_pseudobulk.csv,bin_metadata.csv,receiver_genes.json}` and the same three files under `work/macaque/donor/` | HMBA-BG macaque h5ad, joint-taxonomy sidecars, `donor_metadata.csv` |
| `compute/macaque/build_rhesus_refit.py` | `work/macaque/refit/{projection_bin_level.tsv,projection_matrix_macaque.tsv}` | `work/macaque/` senders, `work/macaque/donor/` receivers, `work/macaque/lr_pairs_filtered.csv` |
| `compute/freeze_display_panels.py` | `panelA_human_{matrix,sig}.csv`, `panelF_kraft_{matrix,sig}.csv`, `panelG_macaque_{matrix,sig}.csv` | `work/human/`, `work/kraft/`, `work/macaque/` |
| `compute/drivers/recon_and_magnitude.py` | `work/drivers/{human,kraft,macaque,mouse_LRRK2,salmani}_recon.csv`, `work/drivers/perfamily_magnitude.csv` | `work/human/`, `work/kraft/`, `work/macaque/`, `work/mouse/` |
| `compute/freeze_lr_drivers.py` | five `lr_drivers/*_magnitude.tsv` tables and `lr_drivers/LR_PROVENANCE.json` | `work/drivers/`, `work/*/lr_pairs_filtered.csv`, `work/human/finalized_844/` |
| `compute/render_extended_arm.py` | `output/panels/extended_arm{1,2}_horizontal.{png,pdf}` | `compute/extended_arm/`, `frozen/extended_arm/` |

`compute/lib/` holds the tensor construction, PARAFAC projection, and display-matrix modules the
dataset scripts share. All PARAFAC fits use rank 20, `l1_receiver` 0.05, `n_iter_max` 300, seed 42,
and the top-3 plus within-family hierarchical reconstruction. The mouse receiver drops the
amygdala and septal slabs before the tensor is built. Ligand-receptor tables keep the row order of
the LIANA resource; the pair lists are never sorted or de-duplicated downstream.

Sender pseudobulks: both mouse datasets pool all cells of each subtype; the human panel-A senders
pool all Kamath cells per predicted subtype; the Kraft senders keep Kamath cells with
`pred_confident`, without `gad2_excluded`, and with `Status == Ctrl`. Pair counts per panel are
409 (mouse), 844 (human), 837 (Kraft), and 932 (rhesus macaque).

Run order:

```bash
export M2H_SOURCE_ROOT=/path/to/source-workspace
export M2H_DATA_ROOT=/path/to/atlas-data
cd figures/fig4_projections/compute
python mouse/build_receiver_bins.py
python mouse/build_mouse_projection.py
python refreeze_mouse_centroid_ap.py
python freeze_manuscript_stats.py
python mouse/build_enrichment_fields.py
python human/prepare_human_inputs.py
python human/select_lr_pairs.py human
python human/build_finalized_844.py
python human/freeze_extended_arm_targets.py
python kraft/prepare_kraft_inputs.py
python kraft/kraft_spatialbins.py
python macaque/build_macaque_senders.py
python macaque/build_macaque_receivers.py
python human/select_lr_pairs.py macaque
python macaque/build_rhesus_refit.py
python freeze_display_panels.py
python drivers/recon_and_magnitude.py
python freeze_lr_drivers.py
```

`frozen/mouse_enrichment_fields.npz` and `frozen/mouse_enrichment_{meta,structures,scales}.csv`
are written by `compute/mouse/build_enrichment_fields.py`. `frozen/mouse_tracing_panelC.pdf` and
`frozen/mouse_tracing_panelD.pdf` are built from anterograde SNc eYFP/mCherry tracing data listed
in the Key Resource Table as available from the Awatramani Lab.

## External inputs

These inputs have no producer in this repository and are external inputs to Figure 4:

| Input | Description |
|---|---|
| `frozen/mouse_tracing_panelC.pdf`, `frozen/mouse_tracing_panelD.pdf` | Built from the anterograde SNc eYFP/mCherry tracing data listed in the Key Resource Table as available from the Awatramani Lab |
| `frozen/extended_arm/arm1_matched_panelA_locked_bins.parquet`, `frozen/extended_arm/arm2_matched_panelA_locked_bins.parquet` | Per-bin spatial placements from the combined niche model fitted on the HMBA-BG MERSCOPE reference, arm 2 with Kraft Slide-tags added |
| `frozen/extended_arm/fig11b_method_tangram_uniform_density_targets.tsv` | Broad-region targets from the Tangram uniform-density spatial transfer |
| `frozen/extended_arm/projection_matrix_human.tsv`, `frozen/extended_arm/bin_metadata.csv` | Kraft Slide-tags zone-level projection matrix and its bin table from the 11-donor zone-by-cell-type fit |
| `fMRI/projection_atlas/canonical/h4_bin_435_metadata.csv` under `M2H_SOURCE_ROOT` | 435 HMBA spatial bins with region, cell type, cell count, and CCF centroid per bin |
| `cell2cell_final_v2/results/stage2_finalized_nonspatial/parafac_projection_matrix_full18.tsv` under `M2H_SOURCE_ROOT` | Mouse broad-target projection matrix over 18 receiver zones, used for the ventral target signs of the enrichment fields |

