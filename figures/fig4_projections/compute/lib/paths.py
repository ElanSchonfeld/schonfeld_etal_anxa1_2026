"""Locations of the Figure 4 source data, intermediate tables, and frozen tables."""
import os
from pathlib import Path

FIG = Path(__file__).resolve().parents[2]
FROZEN = FIG / "frozen"
WORK = FIG / "work"


def _root(name, hint):
    if name not in os.environ:
        raise RuntimeError(f"set {name} to {hint}")
    return Path(os.environ[name]).expanduser().resolve()


def source_root():
    return _root("M2H_SOURCE_ROOT", "the source workspace containing Data/")


def data_root():
    return _root("M2H_DATA_ROOT", "the directory containing the downloaded atlas datasets")


def work_dir(name):
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    return d
