#!/usr/bin/env python3
"""Shared helpers for the Figure 4 display panels."""
from pathlib import Path
import pandas as pd

FROZEN = Path(__file__).resolve().parent.parent / "frozen"
OUT = Path(__file__).resolve().parent.parent / "output" / "panels"

SIG_NOTE = "* significant (label-permutation, per-family BH q<0.05);  hatch = n.s."


def load_panel(stem):
    """Return (matrix DataFrame [group x region], sig dict {(group,region): bool})."""
    mat = pd.read_csv(FROZEN / f"{stem}_matrix.csv", index_col=0)
    s = pd.read_csv(FROZEN / f"{stem}_sig.csv")
    sig = {(r.group, r.region): bool(r.significant) for r in s.itertuples()}
    return mat, sig
