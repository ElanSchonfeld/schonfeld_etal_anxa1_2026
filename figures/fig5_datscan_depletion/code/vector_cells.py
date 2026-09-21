#!/usr/bin/env python3
"""Draw a matrix as vector cells."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_cells(ax, M, cmap, vmin, vmax, aspect="auto"):
    """Render 2-D array M as vector rectangles on ax, matching imshow's cell geometry (cell (i,j) spans x in [j-0.5, j+0.5], y in [i-0.5, i+0.5]; row 0 at top)."""
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                         facecolor=sm.to_rgba(float(M[i, j])),
                         edgecolor="none", lw=0, zorder=0))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect(aspect)
    return sm
