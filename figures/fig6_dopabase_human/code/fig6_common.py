#!/usr/bin/env python3
"""Shared paths, fonts, and crop helpers for the Figure 6 render scripts."""
import json
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent
FROZEN = FIGDIR / "frozen"
OUTPUT = FIGDIR / "output"
PANELS = OUTPUT / "panels"
PANELS.mkdir(parents=True, exist_ok=True)

def _find_common():
    for _anc in [FIGDIR, *FIGDIR.parents]:
        _cand = _anc / "common"
        if (_cand / "figio.py").is_file():
            return _cand
    return None


_COMMON = _find_common()
if _COMMON is not None:
    sys.path.insert(0, str(_COMMON))
try:
    import figio  # noqa: F401
    HAVE_FIGIO = True
except Exception as _exc:                            # noqa: BLE001
    print(f"WARNING: figio not importable ({_exc}); labels will NOT be Helvetica. "
          f"searched from {FIGDIR}")
    HAVE_FIGIO = False

MM_PER_IN = 25.4
def _capture_dsf(default=4):
    try:
        import json as _json
        return int(_json.loads((FROZEN / "capture_manifest.json").read_text())
                   .get("device_scale_factor", default))
    except Exception:                                # noqa: BLE001
        return default


DSF = _capture_dsf()
APP_FONT_PX = 14.0
MIN_PT = 7.0

PANEL_DPI = 700

_manifest = None


def manifest():
    global _manifest
    if _manifest is None:
        _manifest = json.loads((FROZEN / "capture_manifest.json").read_text())
    return _manifest


def regions(state, kind=None):
    """Recorded bounding boxes (CSS px) for a captured state."""
    geo = manifest()["states"][state]["geometry"]
    return [g for g in geo if kind is None or g["kind"] == kind]


def max_panel_mm(crop_w_css, css_px=APP_FONT_PX, min_pt=MIN_PT):
    """Largest printed width (mm) at which `css_px` app text still reaches min_pt."""
    return min_pt * crop_w_css / css_px / 72 * MM_PER_IN


def printed_pt(crop_w_css, panel_mm, css_px=APP_FONT_PX):
    return css_px / crop_w_css * (panel_mm / MM_PER_IN) * 72


def required_dpi(png, panel_mm):
    """Dpi needed to embed `png` at native resolution when placed `panel_mm` wide."""
    im = Image.open(PANELS / f"{png}.png")
    return im.width / (panel_mm / MM_PER_IN)


def crop(state, box, name, panel_mm=None):
    """Crop a captured state."""
    src = Image.open(FROZEN / f"{state}.png")
    x, y, w, h = (int(round(v * DSF)) for v in box)
    x, y = max(0, x), max(0, y)
    w, h = min(w, src.width - x), min(h, src.height - y)
    out = src.crop((x, y, x + w, y + h))
    if panel_mm:
        (PANELS / f"{name}.native.png").parent.mkdir(parents=True, exist_ok=True)
        out.save(PANELS / f"{name}.native.png")
        exact_px = int(round(PANEL_DPI * panel_mm / MM_PER_IN))
        if out.width != exact_px:
            out = out.resize((exact_px, max(1, round(out.height * exact_px / out.width))),
                             Image.LANCZOS)
    p = PANELS / f"{name}.png"
    out.save(p)
    note = ""
    if panel_mm:
        pt = printed_pt(box[2], panel_mm)
        note = f" | at {panel_mm}mm app text = {pt:.1f}pt" + ("" if pt >= MIN_PT else "  *** BELOW 7pt ***")
    print(f"  {name:26s} {out.width}x{out.height}px  (crop {box[2]}x{box[3]} css, "
          f"legible to {max_panel_mm(box[2]):.0f}mm){note}")
    return out


def union(boxes, pad=8):
    """Bounding box (x, y, w, h) enclosing recorded regions, padded."""
    x0 = min(b["x"] for b in boxes) - pad
    y0 = min(b["y"] for b in boxes) - pad
    x1 = max(b["x"] + b["w"] for b in boxes) + pad
    y1 = max(b["y"] + b["h"] for b in boxes) + pad
    return (x0, y0, x1 - x0, y1 - y0)
