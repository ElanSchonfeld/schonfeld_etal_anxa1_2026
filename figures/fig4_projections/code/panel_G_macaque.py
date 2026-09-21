#!/usr/bin/env python3
"""Fig 4 Panel G (supp) - Macaque (Allen HMBA) projection: 3 driver groups x CaH/CaB/CaT/PuR/PuC/PuPV/NAC, cell-weighted family fractions -> centered +/-1 display."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                              # noqa: E402
from heatmap import plot_grouped_broad_heatmap, GROUP_COLORS, panel_fig  # noqa: E402
from fig4_common import load_panel, SIG_NOTE, OUT         # noqa: E402


def main():
    mat, sig = load_panel("panelG_macaque")
    row_color = {g: GROUP_COLORS[g] for g in mat.index}
    fig, ax = panel_fig(len(mat.columns), top_in=0.80, bottom_in=0.58)
    plot_grouped_broad_heatmap(
        ax, mat, row_color, title="Macaque (Allen HMBA)",
        colorbar_label="enrichment / depletion (norm. -1..1)",
        show_territories=True, sig=sig, sig_note=SIG_NOTE)
    figio.save_panel(fig, OUT / "panelG_macaque_projection")


if __name__ == "__main__":
    main()
