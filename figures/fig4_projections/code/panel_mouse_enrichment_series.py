#!/usr/bin/env python3
"""Figure 4 mouse panel: MERFISH enrichment series across five coronal levels for three driver groups."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import distance_transform_edt, gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import figio                                               # noqa: E402
from fig4_common import FROZEN, OUT                        # noqa: E402

CCF_AP_BREGMA_OFFSET_MM = 5.4
NORM = TwoSlopeNorm(0, -1, 1)


def _bregma_mm(ap_mm):
    b = CCF_AP_BREGMA_OFFSET_MM - float(ap_mm)
    return "0.0" if abs(b) < 0.05 else f"{b:+.1f}"


def main():
    z = np.load(FROZEN / "mouse_enrichment_fields.npz")
    ap_levels = list(z["ap_mm"])
    meta = pd.read_csv(FROZEN / "mouse_enrichment_meta.csv")
    titles = meta["title"].tolist()
    structs = pd.read_csv(FROZEN / "mouse_enrichment_structures.csv")["structure"].tolist()
    sc = pd.read_csv(FROZEN / "mouse_enrichment_scales.csv")
    scales = {(int(r.group_idx), r.structure): float(r.scale) for r in sc.itertuples()}

    nrow, ncol = len(ap_levels), len(titles)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.9 + 0.6, nrow * 2.5),
                             gridspec_kw=dict(left=0.085, right=0.995, top=0.94,
                                              bottom=0.07, wspace=0.06, hspace=0.10))
    labeled = set()
    for ri, ap in enumerate(ap_levels):
        dapi = z[f"dapi_{ri}"]
        ext = list(z[f"ext_{ri}"])
        masks = {s: z[f"mask_{ri}_{s}"] for s in structs if f"mask_{ri}_{s}" in z.files}
        for ci, title in enumerate(titles):
            ax = axes[ri, ci]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#333333"); sp.set_linewidth(0.8)
            ax.imshow(dapi, extent=ext, aspect="equal", interpolation="lanczos")
            for s in structs:
                mask = masks.get(s)
                if mask is None or not mask.any():
                    continue
                key = f"field_{ri}_{ci}_{s}"
                esc = scales.get((ci, s), 0.0)
                if key in z.files and esc > 1e-9:
                    fe = z[key]
                    fn = np.clip(fe / esc, -1, 1)
                    rgb = plt.cm.coolwarm(NORM(fn))[:, :, :3]
                    overlay_mask = mask & (np.abs(fe) > 1e-12)
                    comp = dapi.copy()
                    for c in range(3):
                        comp[overlay_mask, c] = rgb[overlay_mask, c]
                    ax.imshow(np.where(overlay_mask[..., None], comp, np.nan), extent=ext,
                              aspect="equal", interpolation="lanczos")
                outline = gaussian_filter(mask.astype(float), 0.8)
                ax.contour(outline, levels=[0.5], colors="#222222", linewidths=0.9,
                           extent=ext, origin="upper")
            if ci == 0:
                callouts = []
                for s in structs:
                    mm = masks.get(s)
                    if mm is None or not mm.any() or s in labeled:
                        continue
                    d = distance_transform_edt(mm)
                    r_, c_ = np.unravel_index(int(np.argmax(d)), mm.shape)
                    l, rgt, bot, top = ext
                    H, W = mm.shape
                    sx = l + (c_ + 0.5) / W * (rgt - l)
                    sy = top - (r_ + 0.5) / H * (top - bot)
                    callouts.append((s, sx, sy))
                    labeled.add(s)
                callouts.sort(key=lambda t: t[2])
                slots = np.linspace(0.72, 0.28, len(callouts)) if len(callouts) > 1 else [0.5]
                for (nm, sx, sy), slot in zip(callouts, slots):
                    if ri == 0 and ci == 0 and nm == "OT":
                        slot = 0.16
                    ax.annotate(nm, xy=(sx, sy), xycoords="data",
                                xytext=(1.08, float(slot)), textcoords="axes fraction",
                                fontsize=9.5, fontweight="bold", color="#222222",
                                ha="left", va="center", annotation_clip=False, zorder=7,
                                arrowprops=dict(arrowstyle="-", lw=0.8, color="#333333", shrinkB=3))
            if ri == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"AP +{ap:.1f} mm\n(bregma {_bregma_mm(ap)} mm)",
                              fontsize=11, fontweight="bold", linespacing=1.1)

    cax = fig.add_axes([0.40, 0.022, 0.32, 0.012])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=TwoSlopeNorm(0, -1, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["Depleted (-1)", "0", "Enriched (+1)"], fontsize=10, fontweight="bold")
    cb.set_label("Subtype projection relative to other subtypes (centered log2 enrichment)",
                 fontsize=10, fontweight="bold")
    cb.outline.set_edgecolor("#999999"); cb.outline.set_linewidth(0.5)
    fig.text(0.022, 0.50, "Coronal (AP, mouse)", rotation=90, ha="center", va="center",
             fontsize=13, fontweight="bold")

    figio.save_panel(fig, OUT / "mouse_enrichment_series")


if __name__ == "__main__":
    main()
