"""Spread a macaque region projection over donor-stratified receiver bins, preserving each region's cell-weighted mean."""
import numpy as np
import pandas as pd


def donor_mass(panel, bin_level, bin_meta, regions):
    frozen = bin_level.copy()
    frozen.columns = [str(c).split("_", 1)[1] if str(c).startswith("bin_") else str(c)
                      for c in frozen.columns]
    bm = bin_meta.copy()
    bm["bin_id"] = bm["bin_id"].astype(str)
    bm = bm[bm["region"].isin(regions)]
    cols = [b for b in bm["bin_id"] if b in frozen.columns]
    frozen = frozen[cols]
    bm = bm.set_index("bin_id").loc[cols]
    subs = [s for s in frozen.index if s in panel.index]
    frozen = frozen.loc[subs]
    panel = panel.loc[subs]
    mass = pd.DataFrame(0.0, index=subs, columns=cols)
    for r in regions:
        vs = [c for c in cols if bm.loc[c, "region"] == r]
        if not vs:
            continue
        nc = bm.loc[vs, "n_cells"].to_numpy(float)
        sh = frozen[vs].to_numpy()
        wmean = (sh * nc[None, :]).sum(1) / nc.sum()
        score = np.divide(panel[r].to_numpy()[:, None] * sh, wmean[:, None],
                          out=np.zeros_like(sh), where=wmean[:, None] > 0)
        mass.loc[subs, vs] = score * (nc / nc.sum())[None, :]
    mass.columns = [f"bin_{i}" for i in range(mass.shape[1])]
    return mass, bm["region"].reset_index(drop=True)
