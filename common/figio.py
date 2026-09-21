"""Shared figure I/O for the m2h reproducibility package."""
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_fonts():
    """Use available system sans-serif fonts and keep vector text editable."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_panel(fig, stem, dpi=600, pad_inches=0.06):
    """Write <stem>.png (600 dpi) and <stem>.ai (vector PDF); close the figure."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        fig.savefig(f"{stem}.ai", format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"  wrote {stem.name}.{{png,ai}}")


setup_fonts()
