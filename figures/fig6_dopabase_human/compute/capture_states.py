#!/usr/bin/env python3
"""Capture the Figure 6 panel states from a running DopaBase Human server."""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
FROZEN.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1680, "height": 1500}
DSF = 4
SETTLE_PAGE = 14_000
SETTLE_TAB = 22_000
SETTLE_VIEW = 14_000

FOCAL_POPULATION = "Anxa1"

EXPRESSION_SCALE = ("Imputed", "ALRA (Imputed)")
GENE_COLORSCALE_PREFS = ("Cool-warm", "coolwarm", "RdBu_r")

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
      const cl = el.classList;
      if (!cl) continue;
      if (cl.contains('js-plotly-plot')) push('plotly', el);
      else if (cl.contains('fda-rd-chips')) push('tier_chips', el, el.innerText);
      else if (cl.contains('tabulator')) push('tabulator', el);
      else if (cl.contains('fda-rd-hero')) push('hero', el, el.innerText);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document);
  return hits;
}
"""

CHIPS = """
() => { const o=[]; const w=(r)=>{ for(const e of r.querySelectorAll('.fda-rd-chips')) o.push(e.innerText.trim());
  for(const e of r.querySelectorAll('*')) if(e.shadowRoot) w(e.shadowRoot); }; w(document); return o; }
"""

MENU_ITEMS = """
() => { const o=[]; const w=(r)=>{ for(const e of r.querySelectorAll('.bk-menu > *')){const t=(e.innerText||'').trim(); if(t) o.push(t);}
  for(const e of r.querySelectorAll('*')) if(e.shadowRoot) w(e.shadowRoot); }; w(document); return o; }
"""

SELECTS = """
() => { const o=[]; const w=(r)=>{ for(const s of r.querySelectorAll('select'))
  o.push({value: s.value, options: [...s.options].map(x=>x.text)});
  for(const e of r.querySelectorAll('*')) if(e.shadowRoot) w(e.shadowRoot); }; w(document); return o; }
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


class Session:
    def __init__(self, pg):
        self.pg = pg
        self.manifest = {"viewport": VIEWPORT, "device_scale_factor": DSF,
                         "focal_population": FOCAL_POPULATION, "states": {}}

    def _park_cursor(self):
        """Park the pointer where it can do no harm."""
        self.pg.mouse.move(4, 4)

    def shot(self, name, note=""):
        self._park_cursor()
        self.pg.wait_for_timeout(700)
        p = FROZEN / f"{name}.png"
        self.pg.screenshot(path=str(p))
        assert_not_blank(p)
        geo = self.pg.evaluate(GEOMETRY)
        self.manifest["states"][name] = {"note": note, "geometry": geo}
        print(f"  -> {name}.png ({p.stat().st_size/1024/1024:.1f} MB, {len(geo)} regions)")
        self._park_cursor()
        return geo

    def tab(self, text, settle=SETTLE_TAB):
        self.pg.locator("div.bk-tab", has_text=text).first.click()
        self.pg.wait_for_timeout(settle)

    def button(self, text, settle=8_000):
        b = self.pg.get_by_role("button", name=text, exact=True).first
        b.scroll_into_view_if_needed()
        b.click()
        self.pg.wait_for_timeout(settle)

    def reveal(self, text, settle=2_500):
        """Scroll the first VISIBLE element containing `text` into view."""
        self._park_cursor()
        loc = self.pg.get_by_text(text, exact=False)
        for i in range(min(loc.count(), 12)):
            el = loc.nth(i)
            try:
                if not el.is_visible():
                    continue
                el.scroll_into_view_if_needed(timeout=8_000)
                self.pg.wait_for_timeout(settle)
                return True
            except Exception:                        # noqa: BLE001
                continue
        print(f"   reveal({text!r}): no visible match, leaving scroll position")
        return False

    def to_top(self):
        """Scroll every scrollable container back to the top, from JavaScript."""
        self.pg.evaluate("""() => {
            const scrollAll = (root) => {
                for (const el of root.querySelectorAll('*')) {
                    if (el.scrollHeight > el.clientHeight + 4) el.scrollTop = 0;
                    if (el.shadowRoot) scrollAll(el.shadowRoot);
                }
            };
            window.scrollTo(0, 0);
            scrollAll(document);
        }""")
        self.pg.wait_for_timeout(2_000)

    def _select_by_options(self, wanted, must_contain=()):
        """Find the visible <select> offering `wanted` and choose it."""
        sels = self.pg.locator("select")
        for i in range(sels.count()):
            s = sels.nth(i)
            try:
                if not s.is_visible():
                    continue
                opts = s.locator("option").all_text_contents()
            except Exception:                        # noqa: BLE001
                continue
            if all(m in opts for m in must_contain):
                for w in ([wanted] if isinstance(wanted, str) else wanted):
                    if w in opts:
                        s.scroll_into_view_if_needed()
                        s.select_option(label=w)
                        self.pg.wait_for_timeout(6_000)
                        return w
        return None

    def set_expression_scale(self, label):
        got = self._select_by_options(label, must_contain=("Raw", "Normalized"))
        print(f"   expression scale -> {got}")
        return got

    def set_gene_colorscale(self, prefs):
        """The gene colorscale select only exists once Color-by is Gene, so this must run AFTER the drug/gene search has switched the map into gene mode."""
        sels = self.pg.locator("select")
        avail = []
        for i in range(sels.count()):
            s = sels.nth(i)
            try:
                if s.is_visible():
                    avail.append(s.locator("option").all_text_contents())
            except Exception:                        # noqa: BLE001
                pass
        flat = {o for lst in avail for o in lst}
        print(f"   colorscale candidates present: {sorted(flat & set(prefs)) or 'NONE'}")
        got = self._select_by_options(list(prefs))
        print(f"   gene colorscale -> {got}")
        return got

    def set_focal_population(self, name):
        """Pick the focal population by finding the VISIBLE <select> that offers it."""
        sels = self.pg.locator("select")
        for i in range(sels.count()):
            s = sels.nth(i)
            try:
                if not s.is_visible():
                    continue
                opts = s.locator("option").all_text_contents()
            except Exception:                        # noqa: BLE001
                continue
            if name in opts and "Sox6" in opts:
                s.scroll_into_view_if_needed()
                s.select_option(label=name)
                self.pg.wait_for_timeout(SETTLE_VIEW)
                print(f"   focal population -> {name}")
                return
        raise RuntimeError(f"no visible <select> offering {name!r}")

    def omni(self, query, want_prefix):
        """Type into the unified 'gene or drug…' search and pick the option whose text starts with want_prefix ('DRUG' or 'GENE')."""
        box = self.pg.locator("input[placeholder='gene or drug…']").first
        box.scroll_into_view_if_needed()
        box.click()
        box.fill("")
        self.pg.wait_for_timeout(300)
        box.type(query, delay=90)
        for _ in range(16):
            self.pg.wait_for_timeout(500)
            hit = [t for t in self.pg.evaluate(MENU_ITEMS)
                   if t.upper().startswith(want_prefix.upper())]
            if hit:
                self.pg.get_by_text(hit[0], exact=True).first.click()
                self.pg.wait_for_timeout(SETTLE_VIEW)
                return hit[0]
        raise RuntimeError(f"omni {query!r}: no {want_prefix} option "
                           f"(saw {self.pg.evaluate(MENU_ITEMS)[:6]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5012/app")
    ap.add_argument("--only", default="",
                    help="comma-separated section letters (e.g. 'C'); default all")
    args = ap.parse_args()
    wanted = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    def want(letter):
        return not wanted or letter in wanted

    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport=VIEWPORT, device_scale_factor=DSF)
        ctx.add_init_script(
            "try{for(const v of ['v1','v2','v3'])"
            "localStorage.setItem('dopabase_tour_'+v, String(Date.now()));}catch(e){}"
        )
        pg = ctx.new_page()
        s = Session(pg)
        t0 = time.time()
        pg.goto(args.url, wait_until="networkidle", timeout=180_000)
        pg.wait_for_timeout(SETTLE_PAGE)
        pg.evaluate(
            "()=>{const s=document.querySelector('.tour-skip'); if(s) s.click();"
            "document.querySelectorAll('.tour-pop,.tour-ring').forEach(e=>e.remove());}"
        )
        pg.wait_for_timeout(500)
        print(f"loaded in {time.time()-t0:.0f}s\n")

        if want("A"):
            print("[A] 3D UMAP, HMoE subtype")
            s.shot("A_umap_hmoe", "landing; colour=Subtype, source=HMoE; 22,871 cells")

        if want("B"):
            print("[B] FDA Drug Target Atlas")
            s.tab("FDA Drug Target Atlas")
            s.set_focal_population(FOCAL_POPULATION)
            chips = pg.evaluate(CHIPS)
            s.manifest["tier_chips"] = chips
            print("   chips:", chips)

            s.reveal("STEP 1")
            s.shot("B1_fda_step1", "Step 1: disease / species / resolution")
            s.reveal("Disease liability across dopamine subtypes")
            s.shot("B2_fda_step2", f"Step 2: risk forest across families, focal={FOCAL_POPULATION}")
            s.reveal("FDA-drug targets ranked by selectivity")
            s.shot("B3_fda_step3", "Step 3: two-axis landscape + ranked drug table")

        if want("C"):
            print("[C] scDRS heatmap")
            s.to_top()
            s.tab("Disease Risk (scDRS)", settle=18_000)
            s.button("Heatmap", settle=14_000)
            s.reveal("scDRS trait")
            s.shot("C_scdrs_heatmap",
                   "11 traits x 16 HMoE subtypes, control donors, per-trait z, BH-FDR stars")

        if want("D"):
            print("[D] ethosuximide -> CACNA1G")
            s.to_top()
            s.tab("FDA Drug Target Atlas")
            s.manifest["expression_scale"] = s.set_expression_scale(EXPRESSION_SCALE)
            picked = s.omni("ethosux", "DRUG")
            print(f"   picked: {picked}")
            s.manifest["drug_option"] = picked
            s.manifest["gene_colorscale"] = s.set_gene_colorscale(GENE_COLORSCALE_PREFS)
            s.to_top()
            s.shot("D1_drug_umap", f"3D atlas recoloured by the target of {picked}")
            s.reveal("Ethosuximide")
            s.shot("D2_drug_dossier", "drug dossier: mechanism, indication, CNS")

        if want("E"):
            print("[E] CACNA1G target dossier")
            s.to_top()
            picked = s.omni("CACNA1G", "GENE")
            print(f"   picked: {picked}")
            s.reveal("CACNA1G")
            s.shot("E1_cacna1g_hero", "CACNA1G dossier: hero + selectivity tiles")
            for i, anchor in enumerate(("Cross-species", "Brain-wide", "Approved"), start=2):
                try:
                    s.reveal(anchor)
                    s.shot(f"E{i}_cacna1g_{anchor.split()[0].lower()}", f"dossier section: {anchor}")
                except Exception as e:
                    print(f"   section {anchor!r} not found: {str(e)[:90]}")

        mpath = FROZEN / "capture_manifest.json"
        out = s.manifest
        if wanted and mpath.exists():
            out = json.loads(mpath.read_text())
            out.update({k: v for k, v in s.manifest.items() if k != "states"})
            out.setdefault("states", {}).update(s.manifest["states"])
        mpath.write_text(json.dumps(out, indent=1))
        print(f"\nwrote frozen/capture_manifest.json "
              f"({'merged ' + ','.join(sorted(wanted)) if wanted else 'full'}) "
              f"| total {time.time()-t0:.0f}s")
        b.close()


if __name__ == "__main__":
    main()
