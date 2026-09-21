"""Three-group display matrices, per-bin group masses, and frozen panel writers shared by the Figure 4 freeze scripts."""
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

from lib.paths import FIG, FROZEN

sys.path.insert(0, str(FIG / "code"))
import significance as SIG

GROUPS = SIG.GROUPS
ANXA = ("Sox6:Tafa1", "Sox6:Vcan")


def members(index):
    return OrderedDict([
        (GROUPS[0], [s for s in ANXA if s in index]),
        (GROUPS[1], [s for s in index if str(s).startswith("Sox6:")]),
        (GROUPS[2], [s for s in index if str(s).startswith("Calb1:")]),
    ])


def group_matrix(raw, counts, order, renormalize=False):
    raw = raw.reindex(columns=[c for c in order if c in raw.columns])
    frac = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) if renormalize \
        else raw.div(raw.sum(axis=1), axis=0)
    subs = list(frac.index)
    w = np.array([counts.get(s, 1.0) for s in subs], dtype=float)
    rows = []
    for label, mem in members(frac.index).items():
        idx = [subs.index(s) for s in mem]
        rows.append(pd.Series(frac.iloc[idx].mul(w[idx], axis=0).sum(axis=0) / w[idx].sum(), name=label))
    M = pd.DataFrame(rows).fillna(0.0)
    return M.div(M.sum(axis=1), axis=0) if renormalize else M


def pooled_mass(proj_bin, counts):
    out = {}
    for label, mem in members(proj_bin.index).items():
        ww = np.array([counts.get(m, 1.0) for m in mem], float)
        ww = ww / ww.sum() if ww.sum() > 0 else ww
        out[label] = (proj_bin.loc[mem].to_numpy() * ww[:, None]).sum(0)
    return pd.DataFrame(out)


def dump(stem, mat, sig):
    FROZEN.mkdir(parents=True, exist_ok=True)
    mat.to_csv(FROZEN / f"{stem}_matrix.csv")
    pd.DataFrame([{"group": g, "region": r, "significant": bool(v)} for (g, r), v in sig.items()]
                 ).to_csv(FROZEN / f"{stem}_sig.csv", index=False)
