#!/usr/bin/env python3
"""Figure 4 panel F: human (Kraft, Slide-tags) projection enrichment by driver group."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE))
import figio                                              # noqa: E402
from heatmap import plot_grouped_broad_heatmap, GROUP_COLORS, panel_fig  # noqa: E402
from fig4_common import load_panel, SIG_NOTE, OUT         # noqa: E402

KRAFT_TERR = {"Z1": "Dorsal", "Z2": "Caudate", "Z3": "Caudate",
              "Z4": "NAc", "Z5": "Putamen", "Z6": "Putamen"}


def main():
    mat, sig = load_panel("panelF_kraft")
    row_color = {g: GROUP_COLORS[g] for g in mat.index}
    fig, ax = panel_fig(len(mat.columns), top_in=0.80, bottom_in=0.58)
    plot_grouped_broad_heatmap(
        ax, mat, row_color, title="Human (Kraft et al.)",
        colorbar_label="enrichment / depletion (norm. -1..1)",
        show_territories=True, territory_fn=KRAFT_TERR.get,
        sig=sig, sig_note=SIG_NOTE)
    figio.save_panel(fig, OUT / "panelF_kraft_projection")


if __name__ == "__main__":
    main()
