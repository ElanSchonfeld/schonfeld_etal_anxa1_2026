#!/usr/bin/env python3
"""Freeze the five ligand-receptor driver tables and their pair-count provenance for Supplementary Figure 4.

Run: python freeze_lr_drivers.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drivers.driver_tables import DATASETS, driver_table
from lib.paths import FROZEN, work_dir


def main():
    out = FROZEN / "lr_drivers"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, mouse_case in DATASETS:
        t = driver_table(key, mouse_case)
        t.to_csv(out / f"{key}_magnitude.tsv", sep="\t", index=False)
        counts[key] = int(len(t))

    human = work_dir("human/finalized_844")
    lr = {name: pd.read_csv(work_dir(name) / "lr_pairs_filtered.csv") for name in
          ("human", "kraft", "macaque")}
    prov = {
        "note": "Figure 4 projection panels use LIANA-consensus ligand-receptor pairs.",
        "panels": {
            "panelA_human": {
                "n_pairs": int((lr["human"].source == "liana_consensus").sum()),
                "projection_sha256": hashlib.sha256(
                    (human / "projection_matrix_human.tsv").read_bytes()).hexdigest(),
                "bin_level_sha256": hashlib.sha256(
                    (human / "projection_bin_level.tsv").read_bytes()).hexdigest(),
            },
            "panelB_mouse": {"n_pairs": 409, "set": "mouse-consensus LIANA"},
            "panelD_salmani": {"n_pairs": 409, "set": "mouse-consensus LIANA"},
            "panelF_kraft": {"n_pairs": int((lr["kraft"].source == "liana_consensus").sum())},
            "panelG_macaque": {"n_pairs": int((lr["macaque"].source == "liana_consensus").sum())},
        },
        "driver_tables_post_gate_pairs": counts,
    }
    (out / "LR_PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    main()
