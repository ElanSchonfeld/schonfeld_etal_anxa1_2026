PY ?= python

FIG1 := figures/fig1_hmoe/code
FIG2 := figures/fig2_human/code
FIG3 := figures/fig3_vulnerability/code
FIG4 := figures/fig4_projections/code
FIG5 := figures/fig5_datscan_depletion/code
FIG6 := figures/fig6_dopabase_human/code

export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/fda_atlas_v2/src:$(CURDIR):$(PYTHONPATH)
export MPLCONFIGDIR ?= $(CURDIR)/.matplotlib-cache

.PHONY: all fig1 fig1-train fig1-heldout fig1-cv fig2 fig3 fig4 fig4-preview fig4-stats fig5 fig6 test

all: fig1 fig2 fig3 fig4 fig5 fig6

fig1:
	$(PY) $(FIG1)/panel_A_gates.py
	$(PY) $(FIG1)/panel_persubtype_bars.py
	$(PY) $(FIG1)/salmani_panels.py
	$(PY) $(FIG1)/salmani_confusion_sankey.py
	$(PY) $(FIG1)/assemble_fig1.py

fig1-train:
	$(PY) figures/fig1_hmoe/compute/retrain_hmoe_sct_hybrid.py

fig1-heldout:
	$(PY) figures/fig1_hmoe/compute/eval_unified_heldout_mouse.py

fig1-cv:
	$(PY) figures/fig1_hmoe/compute/cv_unified_mouse.py

fig2:
	$(PY) $(FIG2)/panel_kamath_1x2_family.py
	$(PY) $(FIG2)/panel_kamath_2x3.py
	$(PY) $(FIG2)/panel_kamath_family_confusion.py
	$(PY) $(FIG2)/panel_benchmark_roc_auc.py
	$(PY) $(FIG2)/panel_kamath_height5.py
	$(PY) $(FIG2)/panel_kamath_groundtruth_sankey.py
	$(PY) $(FIG2)/panel_kamath_subtype_heatmap.py
	$(PY) $(FIG2)/panel_kamath_scanvi_vs_hmoe.py
	$(PY) $(FIG2)/panel_siletti_umap.py
	$(PY) $(FIG2)/panel_kamath_confidence.py
	$(PY) $(FIG2)/panel_kamath_gad2_exclusion.py
	$(PY) $(FIG2)/assemble_fig2.py

fig3:
	$(PY) $(FIG3)/salmani_lesion_masc.py
	$(PY) $(FIG3)/kamath_anxa1_depletion_scdrs.py
	$(PY) $(FIG3)/kamath_vulnerability_4panel.py
	$(PY) $(FIG3)/kamath_subtype_depletion.py
	$(PY) $(FIG3)/kamath_anxa1_depletion_scanvi_vs_hmoe.py
	$(PY) $(FIG3)/assemble_fig3.py

fig4:
	$(PY) $(FIG4)/panel_B_mouse.py
	$(PY) $(FIG4)/panel_D_salmani.py
	$(PY) $(FIG4)/panel_F_kraft.py
	$(PY) $(FIG4)/panel_G_macaque.py
	$(PY) $(FIG4)/panel_A_human.py
	$(PY) $(FIG4)/panel_mouse_enrichment_series.py
	$(PY) $(FIG4)/panel_mouse_tracing_validation.py
	$(PY) $(FIG4)/supp_lr_drivers.py
	$(PY) $(FIG4)/manuscript_stats.py

fig4-preview: fig4
	$(PY) $(FIG4)/assemble_fig4.py

fig4-stats:
	$(PY) $(FIG4)/manuscript_stats.py

fig5:
	$(PY) $(FIG5)/panel_staging_rainbow.py
	$(PY) $(FIG5)/panel_delta_matrices.py
	$(PY) $(FIG5)/panel_depletion_vs_enrichment.py
	$(PY) $(FIG5)/panel_hemi_rainbows.py
	$(PY) $(FIG5)/panel_genetics_driver.py
	$(PY) $(FIG5)/panel_genetics_pairwise_contrasts.py
	$(PY) $(FIG5)/panel_driver_trajectory_hemi.py
	$(PY) $(FIG5)/panel_hemi_genetics_confound.py
	$(PY) $(FIG5)/panel_final_depletion.py
	$(PY) $(FIG5)/panel_genetics_spatial.py
	$(PY) $(FIG5)/panel_staging_time.py
	$(PY) $(FIG5)/panel_conversion_generalises.py

fig6:
	$(PY) $(FIG6)/make_panels.py
	$(PY) $(FIG6)/assemble_fig6.py

test:
	$(PY) -m pytest hmoe_annotate/tests fda_atlas_v2/tests figures/fig1_hmoe/compute/test_unified_training.py figures/fig5_datscan_depletion/compute/test_survutil.py
