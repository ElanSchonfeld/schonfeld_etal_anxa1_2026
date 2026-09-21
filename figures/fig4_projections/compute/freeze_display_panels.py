#!/usr/bin/env python3
"""Freeze the display matrices and significance tables for Figure 4 panels A (human), F (Kraft), and G (rhesus macaque).

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python freeze_display_panels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.display import GROUPS, SIG, dump, group_matrix, pooled_mass
from lib.paths import work_dir
from macaque.donor_mass import donor_mass

HUMAN_FINE = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs"]
KRAFT_ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
MACAQUE_REGIONS = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NAC"]


def bh(p):
    return SIG._bh(np.asarray(p, float))


def sig_from_mass(mass, regions, targets):
    table = SIG.compute_table(mass.assign(region=list(regions)), targets)
    out = {}
    for group in GROUPS:
        d = table[table.group == group]
        q = bh(d.p_perm.to_numpy())
        out.update({(group, r): bool(v < 0.05) for r, v in zip(d.region, q)})
    return out


def freeze_human():
    d = work_dir("human")
    counts = json.loads((d / "sender_subtype_counts.json").read_text())
    target = pd.read_csv(d / "constrained_targets.tsv", sep="\t", index_col=0)
    bins = pd.read_csv(d / "constrained_spatial_bins.csv")
    bins = bins[bins.region.isin(HUMAN_FINE)].reset_index(drop=True)
    mass_cols = [c for c in bins.columns if c.startswith("mass_")]
    proj = bins[mass_cols].T.rename(index={c: c.replace("mass_", "").replace("_", ":", 1)
                                           for c in mass_cols})
    proj.columns = [f"bin_{i}" for i in range(proj.shape[1])]
    dump("panelA_human", group_matrix(target, counts, HUMAN_FINE),
         sig_from_mass(pooled_mass(proj, counts), bins.region, HUMAN_FINE))


def freeze_kraft():
    d = work_dir("kraft")
    counts = json.loads((work_dir("human") / "sender_subtype_counts.json").read_text())
    raw = pd.read_csv(d / "projection_matrix_spatialbins_liana_hier.tsv", sep="\t", index_col=0)
    mass = pd.read_csv(d / "anchored_mass_liana_hier.tsv", sep="\t", index_col=0)
    zones = pd.read_csv(d / "anchored_bins_liana.csv")["zone"]
    dump("panelF_kraft", group_matrix(raw, counts, KRAFT_ZONES, renormalize=True),
         sig_from_mass(pooled_mass(mass, counts), zones, KRAFT_ZONES))


def freeze_macaque():
    d = work_dir("macaque")
    counts = json.loads((d / "donor/sender_subtype_counts.json").read_text())
    raw = pd.read_csv(d / "refit/projection_matrix_macaque.tsv", sep="\t", index_col=0)
    bin_level = pd.read_csv(d / "refit/projection_bin_level.tsv", sep="\t", index_col=0)
    bin_meta = pd.read_csv(d / "donor/bin_metadata.csv")
    mass, regions = donor_mass(raw[MACAQUE_REGIONS], bin_level, bin_meta, MACAQUE_REGIONS)
    dump("panelG_macaque", group_matrix(raw, counts, MACAQUE_REGIONS),
         sig_from_mass(pooled_mass(mass, counts), regions, MACAQUE_REGIONS))


if __name__ == "__main__":
    freeze_human()
    freeze_kraft()
    freeze_macaque()
