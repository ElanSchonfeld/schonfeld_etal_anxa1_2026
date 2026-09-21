#!/usr/bin/env python
"""Shared, SANITISED loading of MDS-UPDRS Part III item-level scores."""
import numpy as np
import pandas as pd

UR_CODE = 101
ITEM_MAX = 4


def sanitize_items(df, cols):
    """Return a copy of df[cols] numeric, with out-of-range codes (101 etc.) set to NaN."""
    out = df[cols].apply(pd.to_numeric, errors='coerce')
    return out.where((out >= 0) & (out <= ITEM_MAX))


def item_score(df, cols, min_frac=1.0, prorate=False):
    """Sum of a sanitised item set."""
    v = sanitize_items(df, cols)
    n_ok = v.notna().sum(axis=1)
    need = int(np.ceil(min_frac * len(cols)))
    s = v.mean(axis=1) * len(cols) if prorate else v.sum(axis=1)
    return s.where(n_ok >= need)


def load_part3(path, off_only=True):
    """MDS-UPDRS III with a parsed date, optionally restricted to OFF / untreated visits."""
    d = pd.read_csv(path, low_memory=False)
    d['dt'] = pd.to_datetime(d.INFODT, format='%m/%Y', errors='coerce')
    d = d.dropna(subset=['dt'])
    if off_only:
        d = d[d.PDSTATE.isna() | (d.PDSTATE == 'OFF')]
    return d


BRADY_R = ["NP3RIGRU", "NP3RIGRL", "NP3FTAPR", "NP3HMOVR", "NP3PRSPR", "NP3TTAPR", "NP3LGAGR"]
BRADY_L = ["NP3RIGLU", "NP3RIGLL", "NP3FTAPL", "NP3HMOVL", "NP3PRSPL", "NP3TTAPL", "NP3LGAGL"]
BRADY = BRADY_R + BRADY_L + ["NP3RIGN"]
RTREM = ["NP3RTALJ", "NP3RTALL", "NP3RTALU", "NP3RTARL", "NP3RTARU", "NP3RTCON"]
ATREM = ["NP3PTRML", "NP3PTRMR", "NP3KTRML", "NP3KTRMR"]
AXIAL = ["NP3SPCH", "NP3FACXP", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]
LAT_R = ["NP3RIGRU", "NP3RIGRL", "NP3FTAPR", "NP3HMOVR", "NP3PRSPR", "NP3TTAPR", "NP3LGAGR",
         "NP3RTARU", "NP3RTARL", "NP3PTRMR", "NP3KTRMR"]
LAT_L = ["NP3RIGLU", "NP3RIGLL", "NP3FTAPL", "NP3HMOVL", "NP3PRSPL", "NP3TTAPL", "NP3LGAGL",
         "NP3RTALU", "NP3RTALL", "NP3PTRML", "NP3KTRML"]
