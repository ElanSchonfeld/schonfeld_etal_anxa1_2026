#!/usr/bin/env python3
"""Assemble Figure 6: DopaBase Human and the FDA Drug Target Atlas."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from fig6_common import OUTPUT, PANELS, HAVE_FIGIO, PANEL_DPI

MM = 25.4
FIG_W_MM = 183.0
FONT = "Helvetica" if HAVE_FIGIO else "DejaVu Sans"
if not HAVE_FIGIO:
    print("WARNING: rendering labels in DejaVu Sans, NOT Helvetica.")

PT_LETTER, PT_TITLE, PT_CALL = 9, 7.5, 7.0
INK, MUTED, ACCENT = "#111111", "#5a6772", "#1d7a46"

TITLE_H = 4.0
GAP = 3.0

COL_GAP = 3.0
COL_L_W = 99.0
COL_R_X = COL_L_W + COL_GAP
COL_R_W = FIG_W_MM - COL_R_X

PLACED = [("A_hero", FIG_W_MM), ("A_inset", 34.0),
          ("B_step2_forest", FIG_W_MM),
          ("C_step3_landscape", COL_L_W), ("E_scdrs_heatmap", COL_L_W),
          ("D_dossier", COL_R_W), ("D_dossier_lower", COL_R_W)]


VARIANT = ""


def aspect_h(png, w_mm):
    im = Image.open(PANELS / f"{png}{VARIANT}.png")
    return w_mm * im.height / im.width


def build(fig, total_h):
    def mmx(v):
        return v / FIG_W_MM

    def mmy(v):
        return v / total_h

    def place(png, x_mm, y_mm, w_mm, h_mm=None, zorder=2):
        im = Image.open(PANELS / f"{png}{VARIANT}.png")
        h = h_mm if h_mm is not None else w_mm * im.height / im.width
        ax = fig.add_axes([mmx(x_mm), 1 - mmy(y_mm + h), mmx(w_mm), mmy(h)], zorder=zorder)
        ax.imshow(im, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return h

    def letter(ch, x_mm, y_mm):
        fig.text(mmx(x_mm), 1 - mmy(y_mm), ch, fontsize=PT_LETTER, fontweight="bold",
                 family=FONT, color=INK, va="top", ha="left")

    def title(txt, x_mm, y_mm):
        fig.text(mmx(x_mm), 1 - mmy(y_mm), txt, fontsize=PT_TITLE, fontweight="bold",
                 family=FONT, color=INK, va="top", ha="left")

    def call(txt, x_mm, y_mm, color=MUTED, ha="left", weight="normal", z=6):
        fig.text(mmx(x_mm), 1 - mmy(y_mm), txt, fontsize=PT_CALL, family=FONT,
                 color=color, va="top", ha=ha, fontweight=weight, zorder=z)

    y = 0.0

    letter("a", 0, y)
    title("DopaBase Human: searching a drug recolours the atlas by its target", 5.0, y + 0.1)
    y += TITLE_H
    h = place("A_hero", 0, y, FIG_W_MM)

    iw = 34.0
    ih = aspect_h("A_inset", iw)
    ix, iy = 43.0, y + 33.0
    fig.patches.append(plt.Rectangle(
        (mmx(ix - 1.2), 1 - mmy(iy + ih + 4.0)), mmx(iw + 2.4), mmy(ih + 5.2),
        transform=fig.transFigure, facecolor="white", edgecolor="#8b959d",
        linewidth=0.6, zorder=4))
    place("A_inset", ix, iy, iw, zorder=5)
    call("default colouring: HMoE subtype", ix, iy + ih + 1.2, color=INK, weight="bold")

    y += h + 1.0
    call("Ethosuximide -> CACNA1G  ·  ALRA-imputed expression, Cool-warm  ·  "
         "22,871 human dopamine neurons", 0.5, y, color=ACCENT, weight="bold")
    call("elanschonfeld-dopabase-human.hf.space/app", FIG_W_MM - 0.5, y, ha="right")
    y += 3.4 + GAP

    letter("b", 0, y)
    title("Step 2  ·  where does the disease risk live?", 5.0, y + 0.1)
    y += TITLE_H
    y += place("B_step2_forest", 0, y, FIG_W_MM) + GAP

    letter("c", 0, y)
    title("Step 3  ·  what can we drug there?", 5.0, y + 0.1)
    letter("d", COL_R_X, y)
    title("Target dossier: CACNA1G", COL_R_X + 5.0, y + 0.1)
    y_row = y + TITLE_H

    hc = place("C_step3_landscape", 0, y_row, COL_L_W)
    y_e = y_row + hc + TITLE_H + GAP
    letter("e", 0, y_e - TITLE_H)
    title("Eleven GWAS traits across dopamine subtypes", 5.0, y_e - TITLE_H + 0.1)
    he = place("E_scdrs_heatmap", 0, y_e, COL_L_W)

    hd = place("D_dossier", COL_R_X, y_row, COL_R_W)
    y_d2 = y_row + hd + TITLE_H + GAP
    title("lower dossier: target-disease coupling", COL_R_X + 5.0, y_d2 - TITLE_H + 0.1)
    hd2 = place("D_dossier_lower", COL_R_X, y_d2, COL_R_W)

    return max(y_e + he, y_d2 + hd2)


def _mark_images_interpolatable(path):
    """Set /Interpolate true on every embedded image in a PDF (or .ai)."""
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import BooleanObject, NameObject
    except Exception:                                    # noqa: BLE001
        print(f"  (pypdf unavailable; {path.name} left without /Interpolate)")
        return 0
    reader = PdfReader(str(path))
    writer = PdfWriter()
    n = 0
    for page in reader.pages:
        res = page.get("/Resources")
        xo = res.get("/XObject") if res else None
        if xo:
            for key in list(xo.keys()):
                obj = xo[key].get_object()
                if obj.get("/Subtype") == "/Image":
                    obj[NameObject("/Interpolate")] = BooleanObject(True)
                    n += 1
        writer.add_page(page)
    with open(path, "wb") as fh:
        writer.write(fh)
    return n


def main():
    h = TITLE_H + aspect_h("A_hero", FIG_W_MM) + 1.0 + 3.4 + GAP
    h += TITLE_H + aspect_h("B_step2_forest", FIG_W_MM) + GAP
    left = (aspect_h("C_step3_landscape", COL_L_W) + TITLE_H + GAP
            + aspect_h("E_scdrs_heatmap", COL_L_W))
    right = (aspect_h("D_dossier", COL_R_W) + TITLE_H + GAP
             + aspect_h("D_dossier_lower", COL_R_W))
    h += TITLE_H + max(left, right)
    total_h = round(h + 2.0, 1)

    fig = plt.figure(figsize=(FIG_W_MM / MM, total_h / MM))
    fig.patch.set_facecolor("white")
    used = build(fig, total_h)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dpi = PANEL_DPI
    print(f"saving at {dpi} dpi; panels are pixel-exact so nothing is resampled")
    global VARIANT

    p = OUTPUT / f"fig6_dopabase_human.png"
    fig.savefig(p, facecolor="white", dpi=dpi)
    print(f"wrote {p.name}  ({p.stat().st_size/1024/1024:.1f} MB, panels 1:1 at {dpi} dpi)")
    plt.close(fig)

    VARIANT = ".native"
    fig2 = plt.figure(figsize=(FIG_W_MM / MM, total_h / MM))
    fig2.patch.set_facecolor("white")
    build(fig2, total_h)
    pdf = OUTPUT / f"fig6_dopabase_human.pdf"
    fig2.savefig(pdf, facecolor="white", dpi=dpi, format="pdf")
    marked = _mark_images_interpolatable(pdf)
    print(f"wrote {pdf.name}  ({pdf.stat().st_size/1024/1024:.1f} MB, native panels, "
          f"{marked} marked interpolatable)")

    ai = OUTPUT / f"fig6_dopabase_human.ai"
    ai.write_bytes(pdf.read_bytes())
    print(f"wrote {ai.name}  (byte-identical copy of the .pdf)")
    plt.close(fig2)
    VARIANT = ""

    print(f"\ncanvas {FIG_W_MM} x {total_h} mm; content ends at {used:.1f} mm "
          f"({'fits' if used <= total_h else 'OVERFLOW'})")
    if total_h > 247:
        print(f"NOTE: {total_h:.0f} mm exceeds the 247 mm full-page limit by "
              f"{total_h-247:.0f} mm.")


if __name__ == "__main__":
    main()
