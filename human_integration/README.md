# DopaBase Human atlas build

This folder contains the code that builds the integrated DopaBase Human atlas: 22,048
Kamath cells plus the machinery-selected 823-cell Siletti cohort. The deployed browser
defaults to the scCRAFT embedding and also offers scVI and Seurat RPCA embeddings.

## Pinned public deployment

- DopaBase Human Space: [`elanschonfeld/DopaBase-Human`](https://huggingface.co/spaces/elanschonfeld/DopaBase-Human/tree/4b698678f094f50ec3154c9a47c36b2f0354c9e8), revision `4b698678f094f50ec3154c9a47c36b2f0354c9e8`
- Data repository: [`elanschonfeld/dopabase-human-data`](https://huggingface.co/datasets/elanschonfeld/dopabase-human-data/tree/37257a55528df3c1cf7eb762c31a66def851a7e2), revision `37257a55528df3c1cf7eb762c31a66def851a7e2`
- CZ CELLxGENE Census release used for Siletti selection: `2023-12-15`

The application code, including its ALRA implementation, is in the pinned Space revision.

## Deployed enrichment resources

DopaBase Human exposes preranked GSEA, over-representation analysis, and
gene-set coloring through the pinned Space. The deployed human library menu
includes `MSigDB_Hallmark_2020`; the canonical-pathway group
`KEGG_2026`, `Reactome_Pathways_2024`, `WikiPathways_2024_Human`, and
`BioCarta_2016`; and `GO_Biological_Process_2026`,
`GO_Cellular_Component_2026`, and `GO_Molecular_Function_2026`. The UI's
`MSigDB C2 (Combined)` option uses the four canonical-pathway libraries above.

## Inputs

Set these paths before running the build:

| Variable | Required input and contract |
|---|---|
| `DOPABASE_KAMATH_H5AD` | Kamath raw-count H5AD. `X` must contain integer counts; `obs` must include `donor_id`, `Cell_Type`, and preferably `Status`. |
| `DOPABASE_SILETTI_REFERENCE_H5AD` | Author-labelled Siletti DA raw-count H5AD with `obs["subcluster_id"]`, used for the expression-kNN ground-truth transfer. |
| `DOPABASE_KAMATH_PRED_H5AD` | Kamath HMoE/native-layout H5AD with `pred_subtype`, `pred_family`, `confidence`, `donor_id`, and `obsm["X_umap"]`. |
| `DOPABASE_SCT_ROOT` | Per-study SCTransform corrected-count inputs. Both `sct_Kamath` and `sct_Siletti` must contain a cells-by-genes `matrix.mtx` plus aligned `cells.txt` and `genes.txt`. |
| `DOPABASE_GENESET_DIR` | scDRS directory containing one weighted `<trait>.gs` file for each of the 11 deployed traits. |

Step `00_select_siletti_da.py` also needs network access to the pinned Census. The
optional WHB identity crosswalk accepts `DOPABASE_CENSUS_CACHE`,
`DOPABASE_WHB_METADATA`, `DOPABASE_WHB_TAXONOMY`,
`DOPABASE_SILETTI_BROAD_H5AD`, and `DOPABASE_SILETTI_REFERENCE_H5AD`; see
`00d_siletti_whb_crosswalk.py` for the expected schemas. All generated files default to
`human_integration/data`, `human_integration/results`, and
`human_integration/figures`. For the root-level secondary integration and postprocessing
scripts, `DOPABASE_DATA_DIR` and `DOPABASE_WORK_DIR` override the data and RPCA scratch
locations.

## Build order

Run from the repository root. The first command creates the local output and input
directories.

```bash
mkdir -p human_integration/data human_integration/results human_integration/figures \
  human_integration/inputs/{kamath,siletti,scdrs_gene_sets,scdrs_sct} \
  human_integration/work

python human_integration/scripts/00_select_siletti_da.py
python human_integration/scripts/00b_build_combined.py
python human_integration/scripts/00c_siletti_labels.py
python human_integration/scripts/01_integrate.py
python human_integration/scripts/02_cluster_qc.py
python human_integration/scripts/09_finalize_atlas.py

python human_integration/scripts/03_apply_hmoe.py \
  human_integration/data/siletti_da_neurons.h5ad \
  human_integration/results/siletti_hmoe_unified.csv

python scripts/03a_integrate_scvi.py
python scripts/03b_export_for_seurat.py
Rscript scripts/03b_integrate_rpca.R
python scripts/03b_finalize_rpca.py

python human_integration/scripts/13_build_browser.py
python human_integration/scripts/score_human_sct_pooled.py \
  --sct-root human_integration/inputs/scdrs_sct \
  --gene-set-dir human_integration/inputs/scdrs_gene_sets \
  --browser-h5ad human_integration/data/human_da_browser_new.h5ad \
  --output human_integration/results/human_scores_sct_pooled.parquet
python human_integration/scripts/update_browser_scdrs_sct.py \
  --scores human_integration/results/human_scores_sct_pooled.parquet \
  --browser-h5ad human_integration/data/human_da_browser_new.h5ad \
  --output-h5ad human_integration/data/human_da_browser_scored.h5ad
python scripts/06_slim_browser.py \
  human_integration/data/human_da_browser_scored.h5ad \
  human_integration/data/human_da_browser_slim.h5ad
```

The optional WHB identity crosswalk is:

```bash
python human_integration/scripts/00d_siletti_whb_crosswalk.py \
  --download-missing --overwrite
```

## Transformations and fixed parameters

1. Siletti selection fetches neurons from five midbrain dissections, applies CP10K plus
   `log1p` only for scoring, and uses a two-component GMM (`seed=42`, `n_init=10`) on
   mean TH/DDC/SLC6A3/SLC18A2 expression. Cells must pass the high component and detect
   at least two machinery genes; TPH2-high and DBH-high aminergic cells are filtered out.
   The saved 823-cell `X` holds raw counts.
2. The Kamath/Siletti merge preserves integer counts, uppercases symbols, sums duplicate
   symbols, and retains only the shared gene intersection. Raw counts are stored in both
   `X` and `layers["counts"]` at this stage.
3. Primary scCRAFT integration filters genes detected in fewer than five cells, performs
   CP10K plus `log1p`, selects 2,000 study-aware HVGs after excluding mitochondrial and
   ribosomal candidates, trains for 150 epochs, and exports a 50-dimensional PCA of the
   learned embedding. Seeds are fixed at 42, with scCRAFT's documented internal seeds.
4. Discovery clustering uses a 15-neighbour graph, Leiden resolution 1.2, and iterative
   nearest-centroid merges. A retained split requires at least 10 genes in each direction
   with absolute log2 fold change greater than 1 and BH-FDR below 0.05.
5. The HMoE wrapper evaluates all 823 raw-count cells in one call with the model in
   `hmoe_annotate/model/`. It writes barcode, subtype, family, and maximum leaf
   probability.
6. Atlas finalization assigns family with GAD2 GMM then SOX6/CALB1 comparison, measures
   stability across 30 seeded 80% subsamples, builds within-family Ward tiers, names
   clusters using held-out Poisson-thinned counts, and computes the canonical PAGA-seeded
   UMAP.
7. Secondary scVI uses raw counts, 3,000 study-aware Seurat-v3 HVGs, `batch_key="study"`,
   and a 30-dimensional latent. RPCA exports the same 3,000-HVG raw matrix, then Seurat v5
   applies LogNormalize, 50 PCs, Kamath-reference `RPCAIntegration`, dimensions 1:30, and
   `k.anchor=5`.
8. Browser construction combines the scCRAFT, scVI, RPCA, Kamath-native, and
   Siletti-native 2D/3D layouts. scCRAFT PAGA is the default. The staging browser keeps
   raw integer counts; slimming removes duplicate layers and neighbour graphs.
9. Human scDRS uses the per-study Kamath and Siletti SCTransform corrected-count
   matrices as-is, without CPM or `log1p`. Genes are intersected, the matrices are
   pooled as dense `float32`, and all cells are scored in one shared calibration. The
   covariates are `const`, raw-browser detected-gene count (`n_genes`), and donor
   indicators. Preprocessing uses 20 mean bins and 20 variance bins. Each of the 11
   weighted traits uses `ctrl_match_key="mean_var"`, 1,000 matched controls,
   `weight_opt="vs"`, and seed 42.
10. Score insertion verifies exact cell order and finite values, copies the staging
    browser, replaces its scDRS columns with the 22 `float32` score columns, and verifies
    the new H5AD before publishing it at the requested output path.

## Environment

Use the repository-root `environment.yml` and `R_PACKAGES.txt`. They pin the software
stack, including `scCRAFT==1.0.0`, `cellxgene-census==1.17.0`, `scvi-tools==1.3.3`, and
the NumPy, pandas, SciPy, Scanpy, and AnnData versions.
