#!/usr/bin/env python3
"""Crop the Figure 6 panels out of the frozen full-window captures."""
from fig6_common import crop, PANELS

A_HERO = (0, 0, 1680, 810)

A_INSET = (520, 165, 840, 470)

CROPS = [
    ("A_hero", "D1_drug_umap", A_HERO, 183,
     "full app: 'ethosuximide' recolours the atlas by CACNA1G (ALRA, Cool-warm)"),
    ("A_inset", "A_umap_hmoe", A_INSET, 34,
     "same cells under the default HMoE-subtype colouring"),

    ("B_step2_forest", "B3_fda_step3", (365, 336, 1275, 286), 183,
     "Step 2: disease liability across families, Anxa1 focal"),
    ("C_step3_landscape", "B3_fda_step3", (365, 920, 1275, 520), 99,
     "Step 3: tier chips + two-axis landscape + ranked drug table"),

    ("D_dossier", "F_dossier_tall", (688, 900, 970, 600), 81,
     "CACNA1G dossier upper: tiles, cross-species, brain-region profile"),
    ("D_dossier_lower", "F_dossier_tall", (688, 1660, 970, 520), 81,
     "CACNA1G dossier lower: target-disease coupling, cross-disease relevance"),

    ("E_scdrs_heatmap", "C_scdrs_heatmap", (365, 862, 1285, 390), 99,
     "11 GWAS traits x 16 HMoE subtypes, control donors, per-trait z, BH-FDR stars"),
]


def main():
    print(f"writing panels to {PANELS}\n")
    for name, state, box, mm, note in CROPS:
        crop(state, box, name, panel_mm=mm)
        print(f"       {note}")


if __name__ == "__main__":
    main()
