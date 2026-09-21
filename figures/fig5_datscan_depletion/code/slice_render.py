#!/usr/bin/env python3
"""Drive-free anatomical-montage renderer for the three DaTscan depletion panels."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fig5_common import figio


def render_montage(npz_path, stem):
    z = np.load(npz_path, allow_pickle=True)
    bg_ax, sm_ax, bg_cor, sm_cor = z['bg_ax'], z['sm_ax'], z['bg_cor'], z['sm_cor']
    ov_ax, ov_cor = z['ov_ax'], z['ov_cor']
    zmm, ymm = z['zmm'], z['ymm']
    stages, ns_labels, colors = z['stages'], z['ns_labels'], z['colors']
    vmin, vmax = float(z['vmin']), float(z['vmax']); pct = z['pct']
    title = str(z['title']); colorbar_label = str(z['colorbar_label'])

    ns = len(stages); NC = bg_ax.shape[0]
    Hpx, Wpx = bg_ax.shape[1], bg_ax.shape[2]
    ph = 2.0
    figw = ph * NC * Wpx / Hpx; figh = ph * 2 * ns + 3.4
    fig = plt.figure(figsize=(figw, figh), facecolor='white')
    gs = gridspec.GridSpec(2 * ns, NC, wspace=0.02, hspace=0.16, top=1 - 1.15 / figh, bottom=2.3 / figh)
    im = None

    def panel(gr, gc, bg, ov, sm, ttl):
        nonlocal im
        ax = fig.add_subplot(gs[gr, gc]); ax.imshow(bg, interpolation='bilinear')
        im = ax.imshow(ov, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='nearest')
        if sm.max() > 0.1: ax.contour(sm, levels=[0.5], colors='#333333', linewidths=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor('#cfcfcf'); s.set_linewidth(0.5)
        if ttl: ax.set_title(ttl, fontsize=17, pad=6)
        return ax

    for r in range(ns):
        for c in range(NC):
            ax = panel(2 * r, c, bg_ax[c], ov_ax[r, c], sm_ax[c], (f"z = {zmm[c]:+d} mm" if r == 0 else None))
            if c == 0:
                ax.text(-0.58, 0.5, f"{stages[r]}\n(n={ns_labels[r]})", transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=19, fontweight='bold', color=str(colors[r]))
                ax.text(-0.24, 0.5, "axial", transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=14, style='italic', color='#555')
        for c in range(NC):
            ax = panel(2 * r + 1, c, bg_cor[c], ov_cor[r, c], sm_cor[c], (f"y = {ymm[c]:+d} mm" if r == 0 else None))
            if c == 0:
                ax.text(-0.24, 0.5, "coronal", transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=14, style='italic', color='#555')

    fig.text(0.5, 1.35 / figh, "L        R", ha='center', fontsize=18, color='#333', fontweight='bold')
    cax = fig.add_axes([0.34, 0.72 / figh, 0.32, 0.20 / figh])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(f"{colorbar_label}   (p{pct[0]}–p{pct[1]}: {vmin:.0f}–{vmax:.0f}%)", fontsize=16)
    cb.ax.tick_params(labelsize=13); cb.outline.set_linewidth(0.7)
    fig.suptitle(title, fontsize=22, y=1 - 0.42 / figh, fontweight='bold')
    figio.save_panel(fig, stem)
