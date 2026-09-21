#!/usr/bin/env python3
"""Apply the released HMoE model to the 823-cell Siletti DA atlas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hmoe_annotate import hmoe_model as hm  # noqa: E402


def build_predictions(input_path: Path, output_path: Path) -> pd.DataFrame:
    atlas = ad.read_h5ad(input_path)
    if atlas.n_obs != 823:
        raise ValueError(f"expected 823 Siletti DA cells, found {atlas.n_obs}")
    if atlas.n_vars == 0:
        raise ValueError("input atlas has no genes")

    bundle = hm.load_model(ROOT / "hmoe_annotate" / "model")
    aligned, n_matched, _ = hm.align_features(
        atlas.var_names,
        atlas.X,
        bundle["feature_names"],
    )
    if n_matched == 0:
        raise ValueError("no model genes matched the Siletti atlas")

    probabilities = hm.predict_proba(bundle, aligned)
    leaves = np.asarray(bundle["leaves"], dtype=object)
    subtype = leaves[probabilities.argmax(axis=1)]
    family = np.asarray([hm.infer_family(value) for value in subtype])
    confidence = probabilities.max(axis=1).astype(np.float32).astype(np.float64)

    barcodes = [
        value if value.endswith("-Siletti") else f"{value}-Siletti"
        for value in map(str, atlas.obs_names)
    ]
    result = pd.DataFrame(
        {
            "barcode": barcodes,
            "hmoe_subtype": subtype,
            "hmoe_family": family,
            "hmoe_confidence": confidence,
        }
    )
    if result["barcode"].duplicated().any():
        raise ValueError("Siletti barcodes are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 823-cell Siletti HMoE prediction table.",
    )
    parser.add_argument("input", type=Path, help="raw-count Siletti DA h5ad")
    parser.add_argument("output", type=Path, help="output prediction CSV")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")
    result = build_predictions(args.input, args.output)
    print(f"wrote {args.output} ({len(result)} cells)")


if __name__ == "__main__":
    main()
