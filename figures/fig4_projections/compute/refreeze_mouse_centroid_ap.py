#!/usr/bin/env python3
"""Freeze the display matrices and significance tables for Figure 4 panels B (Gaertner mouse) and D (Salmani mouse).

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python refreeze_mouse_centroid_ap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.display import GROUPS, SIG, dump
from mouse.panels import REGIONS, display_matrix, load, superbins


def main():
    for stem, tag in (("panelB_mouse", "LRRK2"), ("panelD_salmani", "Salmani")):
        bl, bins, w = load(tag)
        agg = superbins(bl, bins, w)
        dump(stem, display_matrix(bl, bins, w), SIG.compute(agg[["region"] + GROUPS], REGIONS))


if __name__ == "__main__":
    main()
