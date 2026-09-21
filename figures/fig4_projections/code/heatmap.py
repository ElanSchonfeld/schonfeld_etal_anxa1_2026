#!/usr/bin/env python3
"""Shared grouped-projection heatmap renderer for Figure 4 (mouse/human/macaque/Kraft projection panels B, D, F, G and the human broad-region Panel A)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GROUP_COLORS = {
    "Anxa1-associated": "#08306b",
    "Sox6 family": "#4292c6",
    "Calb1 family": "#DE8F05",
}


def _group_display_label(label: str) -> str:
    if label == "Sox6 family":
        return "Sox6⁺ family"
    if label == "Calb1 family":
        return "Calb1⁺ family"
    return label


def _territory_of(col: str) -> str:
    if col.startswith("Ca"):
        return "Caudate"
    if col.startswith("Pu"):
        return "Putamen"
    if col.startswith("NAC"):
        return "Nucleus accumbens"
    return col


def _territory_spans(cols, territory_fn=None):
    fn = territory_fn or _territory_of
    spans, cur, start = [], None, 0
    for i, c in enumerate(cols):
        t = fn(c)
        if cur is None:
            cur, start = t, i
        elif t != cur:
            spans.append((cur, start, i - 1)); cur, start = t, i
    if cur is not None:
        spans.append((cur, start, len(cols) - 1))
    return spans


def draw_sig_marks(ax, dispm, sig, note=None, star_fontsize=9.0,
                   note_xy=(1.0, 1.035), note_ha="right", note_va="bottom"):
    """`*` in the top-right corner of significant cells; see-through diagonal hatch on cells tested but not significant. sig maps (row_label, col_label) -> bool."""
    for yi, r in enumerate(dispm.index):
        for xi, c in enumerate(dispm.columns):
            if (r, c) not in sig:
                continue
            if sig[(r, c)]:
                val = float(dispm.iloc[yi, xi])
                ax.text(xi + 0.30, yi - 0.30, "*", ha="center", va="center",
                        fontsize=star_fontsize, fontweight="bold",
                        color="white" if abs(val) >= 0.55 else "black", zorder=10)
            else:
                ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1.0, 1.0, fill=False,
                             facecolor="none", hatch="////", edgecolor="0.28",
                             linewidth=0.0, zorder=6))
    if note:
        ax.text(note_xy[0], note_xy[1], note, transform=ax.transAxes, ha=note_ha,
                va=note_va, fontsize=5.6, color="#333333", clip_on=False)


def plot_grouped_broad_heatmap(ax, mat, row_color, title, colorbar_label,
                               xtick_fontsize=8.0, ytick_fontsize=8.0,
                               cbar_label_fontsize=7.5, cbar_tick_fontsize=7.0,
                               title_fontsize=8.0, territory_fontsize=7.5,
                               show_territories=True, territory_fn=None,
                               sig=None, sig_note=None):
    cols = list(mat.columns)
    n_rows, n_cols = len(mat.index), len(cols)

    C = mat.sub(mat.mean(axis=0), axis=1)
    vmax = float(np.abs(C.to_numpy()).max()) or 1.0
    dispm = C / vmax
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1.0, vmax=1.0), cmap="RdBu_r")
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                         facecolor=sm.to_rgba(float(dispm.iat[i, j])),
                         edgecolor="none", lw=0, zorder=0))
    ax.set_aspect("auto")
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(dispm.iat[i, j])
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=max(xtick_fontsize - 0.5, 5.5), zorder=8,
                    color="white" if abs(v) > 0.7 else "black")
    for i in range(1, n_rows):
        ax.plot([-0.5, n_cols - 0.5], [i - 0.5, i - 0.5], color="black",
                lw=0.9, zorder=3, solid_capstyle="butt")

    sb_w, sb_gap = 0.42, 0.16
    sb_right = -0.5 - sb_gap
    sb_left = sb_right - sb_w
    for i, label in enumerate(mat.index):
        ax.add_patch(Rectangle((sb_left, i - 0.5), sb_w, 1.0,
                     facecolor=row_color[label], edgecolor="white", lw=1.0,
                     zorder=4, clip_on=False))
    ax.set_xlim(sb_left - 0.05, n_cols - 0.5)
    ax.add_patch(Rectangle((-0.5, -0.5), n_cols, n_rows, fill=False,
                 edgecolor="black", lw=0.8, zorder=5, clip_on=False))

    title_pad = 6.0
    if show_territories:
        spans = _territory_spans(cols, territory_fn)
        for _, _, _e in spans[:-1]:
            ax.plot([_e + 0.5, _e + 0.5], [-0.5, n_rows - 0.5], color="black",
                    lw=0.9, zorder=3, solid_capstyle="butt")
        for label, _s, _e in spans:
            ax.plot([_s - 0.4, _e + 0.4], [-0.64, -0.64], color="black", lw=1.0,
                    zorder=6, clip_on=False, solid_capstyle="butt")
            disp = "Nucleus\naccumbens" if label == "Nucleus accumbens" else label
            ax.text((_s + _e) / 2.0, -0.74, disp, ha="center", va="bottom",
                    fontsize=territory_fontsize, fontweight="bold", color="black",
                    clip_on=False, linespacing=0.95)
        title_pad = min(title_fontsize * 2.5, 22.0)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=xtick_fontsize)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_group_display_label(s) for s in mat.index],
                       fontsize=ytick_fontsize, fontweight="bold")
    for tick, label in zip(ax.get_yticklabels(), mat.index):
        tick.set_color(row_color[label])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if sig is not None:
        if show_territories:
            _nxy, _nha, _nva = (0.5, -0.24), "center", "top"
        else:
            _nxy, _nha, _nva = (1.0, 1.035), "right", "bottom"
        draw_sig_marks(ax, dispm, sig, note=sig_note,
                       star_fontsize=max(ytick_fontsize + 1.0, 8.0),
                       note_xy=_nxy, note_ha=_nha, note_va=_nva)

    ax.set_title(title, fontsize=title_fontsize, pad=title_pad, fontweight="bold")
    cb = plt.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label(colorbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize, length=2)
    cb.outline.set_linewidth(0.5)


_CELL_IN, _LEFT_IN, _RIGHT_IN = 0.62, 1.35, 0.95


def panel_fig(n_cols, top_in, bottom_in, n_rows=3):
    figw = _LEFT_IN + n_cols * _CELL_IN + _RIGHT_IN
    figh = top_in + n_rows * _CELL_IN + bottom_in
    fig = plt.figure(figsize=(figw, figh))
    ax = fig.add_axes([_LEFT_IN / figw, bottom_in / figh,
                       (n_cols * _CELL_IN) / figw, (n_rows * _CELL_IN) / figh])
    return fig, ax
