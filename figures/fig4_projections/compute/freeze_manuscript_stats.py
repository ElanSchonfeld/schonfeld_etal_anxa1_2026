#!/usr/bin/env python3
"""Freeze the per-subtype dorsolateral-ventromedial centres of mass and super-bin group masses read by code/manuscript_stats.py.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python freeze_manuscript_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import FROZEN
from mouse.panels import DATASETS, dlvm_com, load, superbins


def main():
    out = FROZEN / "stats"
    out.mkdir(parents=True, exist_ok=True)
    for tag in DATASETS:
        bl, bins, w = load(tag)
        dlvm_com(bl, bins).to_csv(out / f"{tag}_dlvm_com.csv", index=False)
        superbins(bl, bins, w).to_csv(out / f"{tag}_superbins.csv", index=False)


if __name__ == "__main__":
    main()
