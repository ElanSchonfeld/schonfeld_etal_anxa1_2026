#!/usr/bin/env python3
"""Capture the CACNA1G dossier whole, in one very tall viewport."""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
VIEWPORT = {"width": 1680, "height": 2600}
DSF = 4

GEOMETRY = """
() => {
  const hits = [];
  const push = (kind, el, text) => {
    const r = el.getBoundingClientRect();
    if (r.width < 40 || r.height < 20) return;
    hits.push({kind, text: (text||'').slice(0,80), x: Math.round(r.x), y: Math.round(r.y),
               w: Math.round(r.width), h: Math.round(r.height)});
  };
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) {
      const cl = el.classList; if (!cl) continue;
      if (cl.contains('js-plotly-plot')) push('plotly', el);
      else if (cl.contains('fda-rd-hero')) push('hero', el, el.innerText);
      else if (cl.contains('tabulator')) push('tabulator', el);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document); return hits;
}
"""
MENU = """
() => { const o=[]; const w=r=>{for(const e of r.querySelectorAll('.bk-menu > *')){const t=(e.innerText||'').trim(); if(t)o.push(t);}
  for(const e of r.querySelectorAll('*')) if(e.shadowRoot) w(e.shadowRoot);}; w(document); return o; }
"""


def assert_not_blank(path, min_ink=0.005, bands=8):
    """Fail loudly if any horizontal band of a capture came back empty."""
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(path).convert("L"))
    h = a.shape[0]
    dead = [i for i in range(bands)
            if (a[i * h // bands:(i + 1) * h // bands] < 250).mean() < min_ink]
    if dead:
        raise RuntimeError(
            f"{Path(path).name}: bands {dead} of {bands} are blank -- the capture was "
            f"truncated by the renderer ({a.shape[1]}x{h}px). Lower DSF or the viewport.")


def omni(pg, q, pre):
    box = pg.locator("input[placeholder='gene or drug…']").first
    box.scroll_into_view_if_needed(); box.click(); box.fill("")
    pg.wait_for_timeout(300); box.type(q, delay=90)
    for _ in range(16):
        pg.wait_for_timeout(500)
        hit = [t for t in pg.evaluate(MENU) if t.upper().startswith(pre)]
        if hit:
            pg.get_by_text(hit[0], exact=True).first.click()
            pg.wait_for_timeout(14_000)
            return hit[0]
    raise RuntimeError(f"omni {q!r}: no {pre}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5012/app")
    a = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport=VIEWPORT, device_scale_factor=DSF)
        ctx.add_init_script(
            "try{for(const v of ['v1','v2','v3'])"
            "localStorage.setItem('dopabase_tour_'+v, String(Date.now()));}catch(e){}"
        )
        pg = ctx.new_page()
        t0 = time.time()
        pg.goto(a.url, wait_until="networkidle", timeout=180_000)
        pg.wait_for_timeout(16_000)
        pg.evaluate(
            "()=>{const s=document.querySelector('.tour-skip'); if(s) s.click();"
            "document.querySelectorAll('.tour-pop,.tour-ring').forEach(e=>e.remove());}"
        )
        pg.wait_for_timeout(500)
        pg.locator("div.bk-tab", has_text="FDA Drug Target Atlas").first.click()
        pg.wait_for_timeout(24_000)
        sels = pg.locator("select")
        for i in range(sels.count()):
            s = sels.nth(i)
            try:
                if s.is_visible() and "Anxa1" in s.locator("option").all_text_contents():
                    s.select_option(label="Anxa1"); pg.wait_for_timeout(14_000); break
            except Exception:                       # noqa: BLE001
                pass
        out = {"viewport": VIEWPORT, "device_scale_factor": DSF, "states": {}}
        for name, q, pre in (("F_dossier_tall", "CACNA1G", "GENE"),
                             ("F_drug_tall", "ethosux", "DRUG")):
            picked = omni(pg, q, pre)
            pg.mouse.move(4, 4); pg.wait_for_timeout(900)
            p_ = FROZEN / f"{name}.png"
            pg.screenshot(path=str(p_))
            assert_not_blank(p_)
            out["states"][name] = {"note": picked, "geometry": pg.evaluate(GEOMETRY)}
            print(f"  -> {name}.png ({p_.stat().st_size/1024/1024:.1f} MB)  {picked}")
            for g in out["states"][name]["geometry"]:
                print(f"       {g['kind']:10s} x={g['x']:5d} y={g['y']:5d} "
                      f"w={g['w']:5d} h={g['h']:5d} {g['text'][:40]}")
        (FROZEN / "capture_manifest_tall.json").write_text(json.dumps(out, indent=1))
        print(f"\ntotal {time.time()-t0:.0f}s")
        b.close()


if __name__ == "__main__":
    main()
