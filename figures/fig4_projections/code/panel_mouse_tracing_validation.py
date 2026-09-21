#!/usr/bin/env python3
"""Figure 4 mouse Panel C tracing validation."""
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fig4_common import FROZEN, OUT  # noqa: E402

GAP_PT = 13.0
DPI = 600


def main():
    src_c = fitz.open(FROZEN / "mouse_tracing_panelC.pdf")
    src_d = fitz.open(FROZEN / "mouse_tracing_panelD.pdf")
    rc, rd = src_c[0].rect, src_d[0].rect

    width = rc.width
    d_h = rd.height * (width / rd.width)
    out = fitz.open()
    page = out.new_page(width=width, height=rc.height + GAP_PT + d_h)
    page.show_pdf_page(fitz.Rect(0, 0, width, rc.height), src_c, 0)
    page.show_pdf_page(fitz.Rect(0, rc.height + GAP_PT, width, rc.height + GAP_PT + d_h), src_d, 0)

    OUT.mkdir(parents=True, exist_ok=True)
    out.save(OUT / "mouse_tracing_validation.ai", garbage=3, deflate=True)
    page.get_pixmap(dpi=DPI).save(OUT / "mouse_tracing_validation.png")
    print(f"  wrote mouse_tracing_validation.{{png,ai}}  "
          f"({len(page.get_text().strip())} text chars, {len(page.get_drawings())} vector drawings)")


if __name__ == "__main__":
    main()
