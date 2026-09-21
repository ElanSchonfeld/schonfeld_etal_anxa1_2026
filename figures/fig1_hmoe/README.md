# Figure 1: mouse HMoE classifier

This directory contains the Figure 1 display and model-training code for the
HMoE hierarchy, cross-validation metrics, and Salmani atlas panels.

## Display code

| Script | Output content |
| --- | --- |
| `code/panel_A_gates.py` | HMoE hierarchy with ground-truth cell counts |
| `code/panel_persubtype_bars.py` | per-subtype cross-validation metrics |
| `code/salmani_panels.py` | Salmani territory, neighborhood, HMoE, and ANXA1 panels |
| `code/salmani_confusion_sankey.py` | author-label to HMoE concordance |
| `code/assemble_fig1.py` | `output/fig1_main` and `output/fig1_supp` montages |

`code/palette.py` and `code/plot_salmani_author_vs_hmoe_umap.py` are shared
modules imported by the panel scripts.

Build the `frozen/` inputs with the scripts under `compute/`, then run from
the repository root:

```bash
make fig1
```

Panels are written to `output/panels/` and montages to `output/` as PNG, PDF,
and AI files. These are ignored by Git.

## Frozen overlay inputs

Under `frozen/`:

- `hmoe_dendrogram_meta.pkl`
- `dendrogram_cellcount_cache.npz`
- `cv5_metrics_mouse.json`
- `salmani_author_annotations.parquet`
- `YaghmaeianSalmani_percell_gad2allowed.parquet`

The pickle and object-containing NPZ must come from the deposited overlay.
The display scripts read these annotations and metrics and apply no
normalization, integration, or batch correction.

## Compute code

`v4_improved_unified` is the trained model bundle, stored under
`hmoe_annotate/model/v4_improved_unified/`. The training implementation is in
`src/awatramani_lab/moe/v4/` (`pipelines.py`, `roc_router.py`,
`child_models.py`, and `roc_markers.py`).

| Script | Output content |
| --- | --- |
| `compute/retrain_hmoe_sct_hybrid.py` | hybrid model trained on all mouse LRRK2 cells; raw counts for gates at depth 3 or shallower, SCT Pearson residuals for deeper gates |
| `compute/eval_unified_heldout_mouse.py` | `heldout_test_metrics_mouse.json`: same architecture trained on the seed-42 90% split and evaluated on the held-out 10% |
| `compute/cv_unified_mouse.py` | `cv5_metrics_mouse.json`: seed-42 five-fold stratified cross-validation |
| `compute/build_dendrogram_cache.py` | `frozen/hmoe_dendrogram_meta.pkl` and `frozen/dendrogram_cellcount_cache.npz` from the `v4_improved_unified` topology and the mouse labels |
| `compute/build_salmani_percell.py` | `frozen/YaghmaeianSalmani_percell_gad2allowed.parquet` and `frozen/salmani_author_annotations.parquet` from a pre-predicted Salmani H5AD |
| `compute/test_unified_training.py` | unit tests for the training entry point, run by `make test` |

Set `M2H_SOURCE_ROOT` to a source workspace containing
`Data/Mouse/Mouse_LRRK2_Dopamine.h5ad` with raw counts. Training reads the
hierarchy from the model bundle. Training outputs default to
`artifacts/fig1_hmoe/`; `HMOE_ARTIFACT_ROOT` overrides that location.
`build_salmani_percell.py` reads
`Data/Mouse/YaghmaeianSalmani/YaghmaeianSalmani_authorDA_predicted.h5ad` and
`Data/Mouse/YaghmaeianSalmani/salmani_author_annotations.parquet` under
`M2H_SOURCE_ROOT`. Salmani predictions are produced with `hmoe_annotate/`.

```bash
export M2H_SOURCE_ROOT=/path/to/source-workspace
make fig1-train
make fig1-heldout
make fig1-cv
python figures/fig1_hmoe/compute/build_dendrogram_cache.py
python figures/fig1_hmoe/compute/build_salmani_percell.py
```
