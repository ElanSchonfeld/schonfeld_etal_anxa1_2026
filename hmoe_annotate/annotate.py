#!/usr/bin/env python3
"""Annotate a dopaminergic snRNA/scRNA dataset with Gaertner DA subtypes."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import hmoe_model as hm

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = Path(__file__).resolve().parent / "model"

DA_MARKERS = ["TH", "SLC6A3", "SLC18A2", "DDC", "DRD2", "RET", "NR4A2"]


def _looks_like_counts(X) -> bool:
    """True if a small sample of X looks like non-negative integer counts."""
    sample = X[: min(200, X.shape[0])]
    if sp.issparse(sample):
        vals = sample.data
    else:
        vals = np.asarray(sample).ravel()
    if vals.size == 0:
        return True
    if vals.min() < 0:
        return False
    return bool(np.allclose(vals, np.round(vals)))


def _da_marker_warning(ad) -> None:
    upper = {str(g).upper(): i for i, g in enumerate(ad.var_names)}
    present = [m for m in DA_MARKERS if m in upper]
    if not present:
        warnings.warn(
            "None of the canonical DA markers "
            f"{DA_MARKERS} were found in var_names. This may not be a "
            "dopaminergic dataset, or gene symbols may be non-standard. "
            "Predictions will still be produced but may be unreliable.",
            stacklevel=2,
        )
        return
    n = ad.shape[0]
    frac_pos = []
    for m in present:
        col = ad.X[:, upper[m]]
        col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
        frac_pos.append(float((col > 0).mean()))
    best = max(frac_pos)
    if best < 0.05:
        warnings.warn(
            "Canonical DA markers are detected but expressed in <5% of cells "
            f"(max positive fraction = {best:.1%} across {present}). "
            "This dataset may not be predominantly dopaminergic.",
            stacklevel=2,
        )


def annotate(
    input_path: Path,
    output_path: Path,
    keep_gad2: bool = False,
) -> None:
    import scanpy as sc

    print(f"[1/5] Loading {input_path}")
    ad = sc.read_h5ad(input_path)
    print(f"      {ad.shape[0]:,} cells x {ad.shape[1]:,} genes")

    if ad.n_obs == 0 or ad.n_vars == 0:
        sys.exit("ERROR: empty AnnData (no cells or no genes).")
    if not _looks_like_counts(ad.X):
        warnings.warn(
            "Input .X does not look like raw integer counts (found negative "
            "or non-integer values). The HMoE pipeline expects RAW counts; it "
            "computes SCT residuals internally. Proceeding anyway.",
            stacklevel=2,
        )
    _da_marker_warning(ad)

    print("[2/5] Loading HMoE model (v4_improved_unified)")
    bundle = hm.load_model(MODEL_DIR)
    subtypes = list(bundle["leaves"])
    gate_feats = bundle["feature_names"]

    print("[3/5] Aligning genes to the model's 21,604-gene training space")
    X_aligned, n_matched, n_ortho = hm.align_features(
        ad.var_names, ad.X, gate_feats
    )
    pct = 100.0 * n_matched / len(gate_feats)
    print(f"      matched {n_matched:,}/{len(gate_feats):,} model genes "
          f"({pct:.1f}%, {n_ortho} via Zfp<->ZNF ortholog)")
    if n_matched == 0:
        sys.exit(
            "ERROR: zero model genes matched the input var_names. Check that "
            "var_names are gene symbols (not Ensembl IDs)."
        )
    if pct < 50:
        warnings.warn(
            f"Only {pct:.1f}% of model genes matched. Low overlap can degrade "
            "predictions. Verify gene symbols are standard HGNC/MGI names.",
            stacklevel=2,
        )

    print("[4/5] Running HMoE routing (raw shallow + SCT deep gates)")
    P_sub = hm.predict_proba(bundle, X_aligned)

    pred_subtypes = np.array([subtypes[j] for j in P_sub.argmax(axis=1)])
    pred_family = np.array([hm.infer_family(s) for s in pred_subtypes])

    gad_mask = pred_family == "Gad2"
    n_gad = int(gad_mask.sum())
    if not keep_gad2 and n_gad:
        gad_leaf_idx = [i for i, l in enumerate(subtypes) if "Gad2" in l]
        for idx in np.where(gad_mask)[0]:
            dp = P_sub[idx].copy()
            dp[gad_leaf_idx] = 0
            pred_subtypes[idx] = subtypes[int(dp.argmax())]
            pred_family[idx] = hm.infer_family(pred_subtypes[idx])
        print(f"      Gad2 excluded: {n_gad:,} cells reassigned to best DA subtype")
    elif n_gad:
        print(f"      Gad2 kept: {n_gad:,} cells labeled Gad2:* (--keep-gad2)")

    confidence = np.max(P_sub, axis=1)
    sorted_probs = np.sort(P_sub, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    ad.obs["hmoe_subtype"] = pred_subtypes
    ad.obs["hmoe_subtype_confidence"] = confidence.astype(np.float32)
    ad.obs["hmoe_family"] = pred_family
    ad.obs["hmoe_margin"] = margin.astype(np.float32)
    ad.obs["hmoe_gad2_reassigned"] = gad_mask if not keep_gad2 else np.zeros(ad.n_obs, bool)
    ad.obsm["X_hmoe_P_sub"] = P_sub.astype(np.float32)
    ad.uns["hmoe_subtypes"] = list(subtypes)
    ad.uns["hmoe_model"] = "v4_improved_unified"

    print(f"      mean confidence: {confidence.mean():.3f}  "
          f"median: {np.median(confidence):.3f}")
    print("      subtype counts (top 8):")
    uniq, counts = np.unique(ad.obs["hmoe_subtype"].values, return_counts=True)
    order = np.argsort(counts)[::-1]
    for i in order[:8]:
        print(f"        {uniq[i]:18s} {counts[i]:>7,}  "
              f"({100*counts[i]/ad.n_obs:.1f}%)")

    print(f"[5/5] Writing {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(output_path)
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate a raw-counts DA h5ad with Gaertner HMoE subtypes.",
    )
    p.add_argument("input", type=Path, help="input .h5ad (raw counts)")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output .h5ad path")
    p.add_argument("--keep-gad2", action="store_true",
                   help="keep Gad2:* labels instead of reassigning to best DA "
                        "subtype (default matches Kamath; Siletti uses --keep-gad2)")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input not found: {args.input}")
    annotate(args.input, args.output, keep_gad2=args.keep_gad2)


if __name__ == "__main__":
    main()
