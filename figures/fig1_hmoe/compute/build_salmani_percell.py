#!/usr/bin/env python
"""Export the Salmani per-cell HMoE prediction table.

Run: python build_salmani_percell.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from os import environ
from pathlib import Path

import anndata as ad
import pandas as pd


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
MANUSCRIPT = Path(environ["M2H_SOURCE_ROOT"])
SALMANI = MANUSCRIPT / "Data" / "Mouse" / "YaghmaeianSalmani"

SRC = SALMANI / "YaghmaeianSalmani_authorDA_predicted.h5ad"
OUT_CANONICAL = SALMANI / "YaghmaeianSalmani_percell_gad2allowed.parquet"
OUT_FROZEN = FROZEN / "YaghmaeianSalmani_percell_gad2allowed.parquet"

ANNOT_CANONICAL = SALMANI / "salmani_author_annotations.parquet"
ANNOT_FROZEN = FROZEN / "salmani_author_annotations.parquet"

N_CELLS_EXPECTED = 24_178
FAMILIES_EXPECTED = {"Sox6", "Calb1", "Gad2"}
COLUMNS = ["barcode", "animal", "condition",
           "pred_subtype", "pred_family", "territory", "neighborhood"]


def build() -> pd.DataFrame:
    if not SRC.is_file():
        raise SystemExit(f"ERROR: missing input {SRC}\n"
                         f"  run fig5_projection_code/robustness_sweep/"
                         f"predict_salmani_authorDA.py first")
    obs = ad.read_h5ad(SRC, backed="r").obs

    if len(obs) != N_CELLS_EXPECTED:
        raise SystemExit(f"ERROR: expected {N_CELLS_EXPECTED} cells, got {len(obs)}")

    for kind in ("subtype", "family"):
        a = obs[f"pred_{kind}"].astype(str)
        b = obs[f"pred_{kind}_unified"].astype(str)
        if not a.equals(b):
            n = int((a != b).sum())
            raise SystemExit(
                f"ERROR: pred_{kind} and pred_{kind}_unified disagree on {n} cells; "
                f"resolve before exporting (project rule: unified model only)")

    out = pd.DataFrame({
        "barcode": obs.index.astype(str),
        "animal": obs["animal"].astype(str).values,
        "condition": obs["condition"].astype(str).values,
        "pred_subtype": obs["pred_subtype_unified"].astype(str).values,
        "pred_family": obs["pred_family_unified"].astype(str).values,
        "territory": obs["territory"].astype(str).values,
        "neighborhood": obs["neighborhood"].astype(str).values,
    })[COLUMNS]

    fams = set(out["pred_family"].unique())
    if fams != FAMILIES_EXPECTED:
        raise SystemExit(
            f"ERROR: expected families {sorted(FAMILIES_EXPECTED)}, got {sorted(fams)}. "
            f"A Gad2-suppressing upstream change would look exactly like this.")
    if out["barcode"].duplicated().any():
        raise SystemExit("ERROR: duplicate barcodes")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="rebuild and compare against the shipped file; write nothing")
    a = p.parse_args()

    out = build()
    print(f"  source : {SRC.name}")
    print(f"  built  : {len(out)} cells, "
          f"{out['pred_subtype'].nunique()} subtypes, "
          f"{out['pred_family'].nunique()} families "
          f"({out['pred_family'].value_counts().to_dict()})")

    if OUT_CANONICAL.is_file():
        old = pd.read_parquet(OUT_CANONICAL)
        same = (len(old) == len(out) and
                out.sort_values("barcode").reset_index(drop=True).equals(
                    old[COLUMNS].sort_values("barcode").reset_index(drop=True)))
        print(f"  matches shipped file: {same}")
        if a.check:
            raise SystemExit(0 if same else 1)
        if not same:
            for c in ("pred_subtype", "pred_family"):
                m = out.set_index("barcode")[c].reindex(old["barcode"].values).values
                print(f"    {c}: {(m != old[c].values).sum()} cells differ")
    elif a.check:
        raise SystemExit("ERROR: nothing to check against")

    if a.check:
        return

    out.to_parquet(OUT_CANONICAL, index=False)
    FROZEN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_CANONICAL, OUT_FROZEN)
    print(f"  wrote {OUT_CANONICAL}")
    print(f"  wrote {OUT_FROZEN}")

    if ANNOT_CANONICAL.is_file():
        before = ANNOT_FROZEN.is_file() and _sha(ANNOT_FROZEN) == _sha(ANNOT_CANONICAL)
        shutil.copy2(ANNOT_CANONICAL, ANNOT_FROZEN)
        print(f"  synced {ANNOT_FROZEN.name} "
              f"({'already current' if before else 'UPDATED - frozen copy was stale'})")
    else:
        print(f"  WARNING: {ANNOT_CANONICAL.name} missing from M2H_SOURCE_ROOT")


if __name__ == "__main__":
    main()
