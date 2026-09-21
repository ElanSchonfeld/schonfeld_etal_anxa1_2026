# Figure 6: DopaBase Human

This directory produces Figure 6, which shows the DopaBase Human atlas and the
FDA Drug Target Atlas within it. The display code crops full-window browser
captures under `frozen/` and assembles the figure.

## Display code

| Script | Output content |
| --- | --- |
| `code/make_panels.py` | panel crops under `output/panels/` from the captures under `frozen/` |
| `code/assemble_fig6.py` | `output/fig6_dopabase_human` as PNG, PDF, and AI |

`code/fig6_common.py` holds the shared paths and settings.

Build the `frozen/` inputs with the scripts under `compute/`, then run from
the repository root:

```bash
make fig6
```

## Compute code

| Script | Output content |
| --- | --- |
| `compute/capture_states.py` | full-window PNG captures and `capture_manifest.json` under `frozen/`, taken with Playwright from a running DopaBase Human server (`--url`, default `http://127.0.0.1:5012/app`) |
| `compute/capture_dossier_tall.py` | CACNA1G dossier capture in a 1680x2600 viewport (`--url`, same default) |
| `compute/verify_scdrs_panel.py` | disease-score heatmap matrix recomputed from the browser H5AD given by `--h5ad` or `DOPABASE_BROWSER_H5AD` |

`config/scdrs_traits.yaml` holds the trait order and labels of the deployed
atlas.

```bash
python figures/fig6_dopabase_human/compute/capture_states.py
python figures/fig6_dopabase_human/compute/capture_dossier_tall.py
DOPABASE_BROWSER_H5AD=/path/to/browser.h5ad \
  python figures/fig6_dopabase_human/compute/verify_scdrs_panel.py
```

## Pinned deployment

- DopaBase Human Space revision
  `4b698678f094f50ec3154c9a47c36b2f0354c9e8`
- DopaBase Human dataset revision
  `37257a55528df3c1cf7eb762c31a66def851a7e2`

The atlas-building code is in `human_integration/`; the FDA Atlas code is in
`fda_atlas_v2/`.
