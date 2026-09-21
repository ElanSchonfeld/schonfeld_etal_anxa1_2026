# Figure 5: DaTscan depletion and conversion

PPMI DaTscan depletion, staging, genetic-group, hemisphere, spatial-null, and
phenoconversion panels.

## Display

```bash
make fig5
```

`make fig5` runs the `code/panel_*.py` scripts. They read only the aggregate
CSV, JSON, NPZ, and rendered-slice inputs under `frozen/` and write to
`output/panels/`. `fig5_common.py`, `slice_render.py`, and
`vector_cells.py` are shared helpers.

`output/panels/fig5_conversion_generalises.png` is tracked in the repository.

## Compute

The scripts in `compute/` rebuild the `frozen/` inputs from PPMI data. Set
`PPMI_ROOT` to an authorized local PPMI data directory. Participant IDs, visit
dates, scans, registered volumes, and per-participant caches are not written to
`frozen/` and must not be committed.

### Upstream inputs

These scripts build the derived PPMI products that the freezers read. Run them
in the order listed.

| Script | Writes | Inputs |
| --- | --- | --- |
| `fetch_public_templates.py` | `PPMI_ROOT/derived/av133_spatial/mni152_fsl_{1,2}mm.nii.gz`, `PPMI_ROOT/derived/datscan_spatial/{osgk_striatum.nii.gz,fpcit_template_mni.nii,fpcit_fsl_grid.nii.gz}`, `frozen/annotations_compressed_700.nii.gz` | TemplateFlow, NeuroVault image 406338, SourceForge `spmtemplates`, Allen atlas assets (each checked against a recorded MD5) |
| `build_ppmi_tables.py` | `PPMI_ROOT/derived/{datscan_recon_cohort.csv,ppmi_scan_datscan.parquet,datscan_backbone.parquet}` | `PPMI_ROOT/study_info/`, `PPMI_ROOT/demographics/`, `PPMI_ROOT/images/datscan_spect/_zips/` |
| `register_datscan_baselines.py` | `PPMI_ROOT/derived/datscan_spatial/fpcit_mni/`, `datscan_pct_depletion.nii.gz` | `datscan_recon_cohort.csv`, `fpcit_fsl_grid.nii.gz`, `osgk_striatum.nii.gz`, ICBM152 white-matter prior, DaTscan DICOM archives |
| `register_datscan_visits.py` | `PPMI_ROOT/derived/datscan_spatial/fpcit_mni_visits/` | `ppmi_scan_datscan.parquet`, `datscan_recon_cohort.csv`, `fpcit_fsl_grid.nii.gz`, DaTscan DICOM archives |
| `build_projection_weights_and_caches.py` | `PPMI_ROOT/derived/datscan_spatial/{anxa1_putaminal_weight_fsl.nii.gz,subtype_weight_*_fsl.nii.gz,anxa1_putaminal_spatial.parquet,anxa_vs_nonanxa_putamen.parquet}` | `projection_fields.py`, `osgk_striatum.nii.gz`, `fpcit_mni/`, `fpcit_mni_visits/` |
| `build_staging_depmaps.py` | `PPMI_ROOT/derived/datscan_spatial/{fig12_staging_all_depmaps.npz,fig12_staging_all_depmaps_hemi.npz}` | `ppmi_scan_datscan.parquet`, `datscan_recon_cohort.csv`, `fpcit_mni/`, `fpcit_mni_visits/` |
| `build_group_stats.py` | `PPMI_ROOT/derived/datscan_spatial/fig12_group_stats.csv` | `projection_fields.py`, `ppmi_scan_datscan.parquet`, `datscan_recon_cohort.csv`, `fpcit_mni/` |

`datscan_registration.py` holds the archive index, the striatal volumes of
interest, the white-matter reference, the SUVR normalizer, and the staged PD
visit selection shared by `register_datscan_visits.py` and
`freeze_fig5_slices.py`.

Registration calls ANTs without a random seed, so registered volumes reproduce
within numerical tolerance, not bitwise.

### Freezers

| Script | Writes to `frozen/` |
| --- | --- |
| `freeze_fig5_inputs.py` | staging and delta matrices, hemisphere matrices, scatter inputs, `brainsmash_inputs/` |
| `brainsmash_map_correspondence.py` | `bilateral_variogram_smash.csv` |
| `freeze_fig5_slices.py` | `slices_final_depletion.npz`, `slices_genetics_spatial.npz`, `slices_staging_time.npz` |
| `freeze_fig5_genetics.py` | `genetics_driver_points.csv`, `genetics_driver_groups.csv`, `genetics_contrasts.csv`, `genetics_contrasts_headers.csv` |
| `freeze_fig5_hemi_genetics.py` | `driver_trajectory_hemi.csv`, `hemi_confound_bars.csv`, `hemi_confound_concordance.csv`, `hemi_confound_contrasts.csv`, `hemi_confound_inset.npz`, `hemi_confound_meta.json` |
| `freeze_fig5_conversion_*.py` | inputs of `panel_conversion_generalises.py` (below) |

Helpers: `projection_fields.py`, `hemisphere_datscan_data.py`,
`hemisphere_voxel_analysis.py`, `updrs_util.py`, `conversion_paths.py`,
`survutil.py` (Cox models, tested by `test_survutil.py`).

LRRK2+GBA double carriers are excluded from all genetic-group analyses. The
more-affected hemisphere is defined by clinical `DOMSIDE`.

### Additional inputs

`projection_fields.py` reads:

- `figures/fig4_projections/frozen/extended_arm/arm2_matched_panelA_locked_bins.parquet`,
  which has no producer in this repository and is available from the authors on
  request
- `figures/fig5_datscan_depletion/frozen/annotations_compressed_700.nii.gz`,
  written by `compute/fetch_public_templates.py`

### Spatial null

1. Run `freeze_fig5_inputs.py` to write the target, mask, and signed fields
   under `frozen/brainsmash_inputs/`.
2. Run `brainsmash_map_correspondence.py` to write
   `frozen/bilateral_variogram_smash.csv`.
3. Run `freeze_fig5_inputs.py` again to add the p-values to the scatter inputs.

### Conversion panel

Run in this order: `freeze_fig5_conversion_matched_sbr.py`,
`freeze_fig5_conversion_cohorts.py`, `freeze_fig5_conversion_fingerprint.py`,
`freeze_fig5_conversion_fingerprint_ancova.py`,
`freeze_fig5_conversion_where.py`, `freeze_fig5_conversion_irbd.py`,
`freeze_fig5_conversion_otsu.py`.

The conversion scripts write their participant-level intermediates to
`PPMI_ROOT/derived/fig5_conversion/`, not to `frozen/`.

`code/panel_conversion_generalises.py` reads:

- `conversion_generalises_irbd.csv`
- `conversion_generalises_meta.json`
- `family_fingerprint_striatum_ancova.csv`
- `where_median.csv`
- `where_median_km.csv`
- `partition_asymmetry_otsu.csv`
