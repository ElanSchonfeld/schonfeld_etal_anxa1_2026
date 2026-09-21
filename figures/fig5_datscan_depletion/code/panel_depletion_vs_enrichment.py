#!/usr/bin/env python3
"""Fig 5 panel - striatal % dopamine loss vs each group's projection enrichment (bilateral)."""
import numpy as np
import matplotlib.pyplot as plt
from fig5_common import figio, FROZEN, PANELS, GORD, GLAB, FCOL

d = np.load(FROZEN / "scatter_bilateral.npz"); xs = d["band_xs"]
def _fmtp(v):
    return ("%.3f" % v).rstrip('0').lstrip('0') if v >= 0.001 else ("%.0e" % v)
def _psfmt(p): return "" if not np.isfinite(p) else f" (p={_fmtp(p)})"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for g in GORD:
    ax.fill_between(xs, d[f"{g}_blo"], d[f"{g}_bhi"], color=FCOL[g], alpha=0.11, lw=0, zorder=1)
    r, p = float(d[f"{g}_rho"]), float(d[f"{g}_p"])
    ax.plot([0, 100], d[f"{g}_fit"], '-', color=FCOL[g], lw=2.6, label=f"{GLAB[g]}: ρ={r:+.2f}{_psfmt(p)}", zorder=3)
ax.set_xlabel("Striatal voxels ranked by that group's projection enrichment (low to high, %)", fontsize=9)
ax.set_ylabel("% dopamine loss (PD vs HC)", fontsize=9)
ax.set_title("Depletion vs projection enrichment\nline = fit over all voxels; band = block-bootstrap 95% CI; p = variogram-null spatial test",
             fontsize=8.5, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left', frameon=False); ax.tick_params(labelsize=8); ax.set_xlim(-2, 102)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
figio.save_panel(fig, PANELS / "depletion_vs_enrichment")
