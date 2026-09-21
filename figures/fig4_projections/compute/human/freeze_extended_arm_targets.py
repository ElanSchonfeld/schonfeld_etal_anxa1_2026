#!/usr/bin/env python3
"""Write the broad-region target tables the extended human atlas reads under frozen/extended_arm/.

Run: python human/freeze_extended_arm_targets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import FROZEN, work_dir

REGION_ORDER = ["CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "VeP"]


def main():
    target = pd.read_csv(work_dir("human") / "constrained_targets.tsv", sep="\t", index_col=0)
    target = target.reindex(columns=REGION_ORDER)
    sd = target.std(axis=0, ddof=1).replace(0, np.nan)
    display = ((target - target.mean(axis=0)) / sd).fillna(0.0)
    out = FROZEN / "extended_arm"
    out.mkdir(parents=True, exist_ok=True)
    target.to_csv(out / "fig11b_panelA_constrained_targets.tsv", sep="\t")
    display.to_csv(out / "fig11b_panelA_constrained_display_targets.tsv", sep="\t")


if __name__ == "__main__":
    main()
