#!/usr/bin/env python3
"""Fig 4 Panel B (main) - Mouse (Gaertner & Oram) projection: 3 driver groups x CP/ACB/OT/Amygdala, cell-weighted row-norm fractions -> centered +/-1 display."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                              # noqa: E402
from heatmap import plot_grouped_broad_heatmap, GROUP_COLORS, panel_fig  # noqa: E402
from fig4_common import load_panel, SIG_NOTE, OUT         # noqa: E402


def main():
    mat, sig = load_panel("panelB_mouse")
    row_color = {g: GROUP_COLORS[g] for g in mat.index}
    fig, ax = panel_fig(len(mat.columns), top_in=0.42, bottom_in=0.55)
    plot_grouped_broad_heatmap(
        ax, mat, row_color, title="Mouse (Gaertner and Oram et al.)",
        colorbar_label="enrichment / depletion (norm. -1..1)",
        show_territories=False, sig=sig, sig_note=None)
    ax.text(0.5, -0.26, SIG_NOTE, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.0, color="#333333")
    figio.save_panel(fig, OUT / "panelB_mouse_projection")


if __name__ == "__main__":
    main()
