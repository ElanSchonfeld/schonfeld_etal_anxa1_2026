# HMoE annotator

This directory contains the inference code and trained model that assign
Gaertner mouse dopaminergic-neuron subtypes to mouse and human single-cell or
single-nucleus datasets.

## Files

- `annotate.py`: command-line annotator.
- `hmoe_model.py`: model loading and inference.
- `model/v4_improved_unified/`: hierarchy metadata (`meta.json`) and 34
  child-classifier checkpoints.
- `model/family_gate/features.json`: the 21,604-gene training feature space.
- `requirements.txt`: pinned dependency versions.
- `tests/test_hmoe_model.py`: model test on synthetic counts, run by
  `make test`.

## Input

`annotate.py` expects an AnnData `.h5ad` with:

- raw, non-negative integer counts in `.X`;
- gene symbols in `var_names`; and
- predominantly dopaminergic cells.

Do not log-normalize, scale, or batch-correct `.X` before inference. The deep
HMoE gates calculate SCT Pearson residuals internally from population-level
statistics; run the complete query dataset in one call. Query genes are
matched by symbol, with a limited ZNF-to-Zfp symbol fallback; unmatched genes
are set to zero.

## Run

```bash
python hmoe_annotate/annotate.py input.h5ad -o annotated.h5ad
```

By default, Gad2-family predictions are reassigned to the best dopaminergic
leaf, as in the Kamath analysis. `--keep-gad2` retains Gad2 leaves, as in the
823-cell Siletti analysis run by `human_integration/scripts/03_apply_hmoe.py`.

## Output

The annotated AnnData contains:

- `.obs`: `hmoe_subtype`, `hmoe_family`, `hmoe_subtype_confidence`,
  `hmoe_margin`, and `hmoe_gad2_reassigned`;
- `.obsm["X_hmoe_P_sub"]`: full leaf probabilities; and
- `.uns`: `hmoe_subtypes` and `hmoe_model`.

## Training

The training implementation is under `src/awatramani_lab/moe/v4/`. The entry
points for training `v4_improved_unified`, held-out evaluation, and five-fold
cross-validation are under `figures/fig1_hmoe/compute/`; see
`figures/fig1_hmoe/README.md` for commands and inputs.
