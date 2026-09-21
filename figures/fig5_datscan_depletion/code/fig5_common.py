#!/usr/bin/env python3
"""Shared constants + paths for the Figure-5 drive-free render scripts."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio  # noqa: E402,F401

FROZEN = HERE.parent / "frozen"
PANELS = HERE.parent / "output" / "panels"
PANELS.mkdir(parents=True, exist_ok=True)

GORD = ["Anxa1", "Sox6-all", "Sox6-rest", "Calb1"]
GLAB = {"Anxa1": "Anxa1-associated", "Sox6-all": "Sox6 (all)", "Sox6-rest": "Sox6 (non-Anxa1)", "Calb1": "Calb1"}
GLAB_S = {"Anxa1": "Anxa1-assoc.", "Sox6-all": "Sox6 (all)", "Sox6-rest": "Sox6 (non-Anxa1)", "Calb1": "Calb1"}
FCOL = {"Anxa1": "#22336b", "Sox6-all": "#2f6fb2", "Sox6-rest": "#9ec4e0", "Calb1": "#e2861f"}
SCHEME_TITLE = {"time": "Time since motor onset", "hy": "Hoehn & Yahr", "updrs": "MDS-UPDRS III"}
