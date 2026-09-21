#!/usr/bin/env python3
"""Build the human LIANA-consensus projection under work/human/finalized_844/ and its region targets and spatial-bin masses on the 435 HMBA spatial bins.

Run: M2H_SOURCE_ROOT=/path/to/source-workspace python build_finalized_844.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.parafac_pipeline import aggregate_to_regions, build_tensor, run_parafac_pipeline
from lib.paths import source_root, work_dir

SEED = 42
DIRECT_REGIONS = ("CaH", "CaB", "CaT", "PuR", "PuC", "PuPV", "NACc", "NACs", "VeP")
NAC_FINE_REGIONS = ("NACc", "NACs")
SHAPE_RATIO_CLIP = (0.70, 1.45)


def project(d, out):
    sd = pd.read_csv(d / "senders_pseudobulk.csv", index_col=0)
    recv = pd.read_csv(d / "receivers_pseudobulk.csv").values.astype(np.float32)
    recv_genes = json.loads((d / "receiver_genes.json").read_text())
    bm = pd.read_csv(d / "bin_metadata.csv")
    lr = pd.read_csv(d / "lr_pairs_filtered.csv")
    lr = lr[lr.source == "liana_consensus"][["ligand_gene", "receptor_gene"]].reset_index(drop=True)
    subs = list(sd.index)
    T = build_tensor(sd.values.astype(np.float32), list(sd.columns), recv, recv_genes, lr)
    proj, _ = run_parafac_pipeline(T, subs, n_factors=20, l1_receiver=0.05, seed=SEED)
    bin_level = pd.DataFrame(proj, index=subs, columns=[f"bin_{i}" for i in range(proj.shape[1])])
    bin_level.to_csv(out / "projection_bin_level.tsv", sep="\t")
    region, region_z = aggregate_to_regions(proj, subs, bm)
    region.to_csv(out / "projection_matrix_human.tsv", sep="\t")
    region_z.to_csv(out / "projection_matrix_human_z.tsv", sep="\t")
    bm.to_csv(out / "bin_metadata.csv", index=False)
    return region, bin_level, bm


def spatial_bins(bin_level, bin_meta, h4_path):
    meta = pd.read_csv(h4_path)
    meta = meta[meta["region"].isin([r for r in DIRECT_REGIONS if r != "VeP"])].copy()
    meta = meta.rename(columns={"x_ccf_mean": "x_ccf_mm", "y_ccf_mean": "y_ccf_mm", "z_ccf_mean": "z_ccf_mm"})
    bins = meta[["bin_id", "region", "group_name", "n_cells", "x_ccf_mm", "y_ccf_mm", "z_ccf_mm"]].copy()
    for col in ("x_ccf_mm", "y_ccf_mm", "z_ccf_mm"):
        bins[col] = pd.to_numeric(bins[col], errors="coerce")
    bins = bins.dropna(subset=["x_ccf_mm", "y_ccf_mm", "z_ccf_mm"])
    score = {(str(r.region), str(r.group_name)): f"bin_{int(r.bin_id)}" for r in bin_meta.itertuples()}
    regions = set(bin_meta["region"].astype(str))
    for subtype in bin_level.index:
        values = []
        for row in bins.itertuples(index=False):
            if str(row.region) not in regions:
                values.append(1.0)
            else:
                col = score.get((str(row.region), str(row.group_name)))
                values.append(float(bin_level.loc[subtype, col]) if col else 0.0)
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        bins[subtype] = np.maximum(values, 0.0)
    return bins.sort_values(["region", "y_ccf_mm", "x_ccf_mm", "z_ccf_mm"]).reset_index(drop=True)


def target_region_totals(panel, bins):
    target = panel.reindex(columns=list(DIRECT_REGIONS)).copy()
    fine_counts = bins.groupby("region")["n_cells"].sum()
    denom = float(fine_counts.reindex(NAC_FINE_REGIONS).fillna(0).sum())
    if "NAC" in panel.columns and denom > 0:
        for region in NAC_FINE_REGIONS:
            target[region] = target[region] + panel["NAC"] * float(fine_counts.get(region, 0)) / denom
    return target


def unit_mean_shape(shape, weights):
    mean_shape = float(np.sum(shape * weights) / np.sum(weights))
    if not np.isfinite(mean_shape) or mean_shape <= 0:
        return np.ones(len(shape), dtype=float)
    ratio = np.clip(shape / mean_shape, SHAPE_RATIO_CLIP[0], SHAPE_RATIO_CLIP[1])
    mean_ratio = float(np.sum(ratio * weights) / np.sum(weights))
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        return np.ones(len(shape), dtype=float)
    return ratio / mean_ratio


def constrain(panel, bins):
    out = bins.copy()
    target = target_region_totals(panel, out)
    n_cells = out["n_cells"].to_numpy(dtype=float)
    region_arr = out["region"].astype(str).to_numpy()
    for subtype in panel.index:
        shape = np.maximum(out[subtype].to_numpy(dtype=float), 0.0)
        mass = np.zeros(len(out), dtype=float)
        for region in DIRECT_REGIONS:
            idx = np.where(region_arr == region)[0]
            if len(idx) == 0:
                continue
            local = float(target.loc[subtype, region]) * unit_mean_shape(shape[idx], n_cells[idx])
            mass[idx] = local * n_cells[idx] / np.sum(n_cells[idx])
        out["mass_" + subtype.replace(":", "_")] = mass
    return out, target


def main():
    d = work_dir("human")
    out = work_dir("human/finalized_844")
    panel, bin_level, bin_meta = project(d, out)
    h4 = source_root() / "fMRI/projection_atlas/canonical/h4_bin_435_metadata.csv"
    bins, target = constrain(panel, spatial_bins(bin_level, bin_meta, h4))
    target.to_csv(d / "constrained_targets.tsv", sep="\t")
    bins.to_csv(d / "constrained_spatial_bins.csv", index=False)


if __name__ == "__main__":
    main()
