#!/usr/bin/env python3
"""Portable BrainSMASH test for spatial correspondence between NIfTI maps."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from brainsmash.mapgen.base import Base
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--field",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="Fixed projection field; repeat for every field.",
    )
    parser.add_argument(
        "--target-labels",
        type=float,
        nargs="+",
        help="Treat each listed value as a separate one-hot nominal target.",
    )
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--n-surrogates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"--field must be NAME=FILE, received: {specification}")
    name, filename = specification.split("=", 1)
    if not name or not filename:
        raise ValueError(f"--field must be NAME=FILE, received: {specification}")
    return name, Path(filename)


def same_grid(reference: nib.spatialimages.SpatialImage, other: nib.spatialimages.SpatialImage) -> bool:
    return reference.shape == other.shape and np.allclose(
        reference.affine, other.affine, rtol=0.0, atol=1e-5
    )


def main() -> None:
    args = arguments()
    if args.n_surrogates < 1:
        raise ValueError("--n-surrogates must be positive")

    target_image = nib.load(args.target)
    target = target_image.get_fdata()
    mask_image = nib.load(args.mask)
    if not same_grid(target_image, mask_image):
        raise ValueError("Target and mask are not on the same voxel grid")
    common_mask = (mask_image.get_fdata() > 0) & np.isfinite(target)

    fields: dict[str, np.ndarray] = {}
    field_paths: dict[str, Path] = {}
    for specification in args.field:
        name, filename = named_path(specification)
        if name in fields:
            raise ValueError(f"Duplicate field name: {name}")
        field_image = nib.load(filename)
        if not same_grid(target_image, field_image):
            raise ValueError(f"Field {name} is not on the target voxel grid: {filename}")
        values = field_image.get_fdata()
        common_mask &= np.isfinite(values)
        fields[name] = values
        field_paths[name] = filename

    n_voxels = int(common_mask.sum())
    if n_voxels < 30:
        raise ValueError(f"Only {n_voxels} valid masked voxels")
    field_vectors = {name: values[common_mask] for name, values in fields.items()}
    for name, values in field_vectors.items():
        if np.ptp(values) == 0:
            raise ValueError(f"Field {name} is constant in the common mask")

    ijk = np.array(np.where(common_mask)).T
    coordinates_mm = nib.affines.apply_affine(target_image.affine, ijk)
    distances = cdist(coordinates_mm, coordinates_mm).astype(np.float32)

    if args.target_labels:
        targets = [
            (f"{args.target_name}_label_{label:g}", (target[common_mask] == label).astype(float))
            for label in args.target_labels
        ]
    else:
        targets = [(args.target_name, target[common_mask])]

    records: list[dict] = []
    for target_index, (target_name, target_vector) in enumerate(targets):
        if np.ptp(target_vector) == 0:
            raise ValueError(f"Target {target_name} is constant in the common mask")
        print(
            f"[BrainSMASH] {target_name}: {n_voxels} voxels, "
            f"{args.n_surrogates} surrogates",
            flush=True,
        )
        surrogates = Base(
            x=target_vector,
            D=distances,
            resample=True,
            deltas=np.arange(0.1, 1.0, 0.2),
            n_jobs=args.n_jobs,
            seed=args.seed + target_index,
        )(args.n_surrogates)
        if not np.isfinite(surrogates).all():
            raise RuntimeError(f"BrainSMASH produced non-finite values for {target_name}")
        if args.target_labels and not np.allclose(
            surrogates.sum(axis=1), target_vector.sum()
        ):
            raise RuntimeError(f"Surrogates did not preserve the size of {target_name}")

        for field_name, field_vector in field_vectors.items():
            observed = float(spearmanr(field_vector, target_vector).statistic)
            null = np.array(
                [spearmanr(field_vector, surrogate).statistic for surrogate in surrogates],
                dtype=float,
            )
            p_smash = float(
                (1 + np.sum(np.abs(null) >= abs(observed))) / (args.n_surrogates + 1)
            )
            records.append(
                {
                    "target": target_name,
                    "group": field_name,
                    "r_spearman": observed,
                    "p_smash": p_smash,
                    "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=0)),
                    "n_voxels": n_voxels,
                    "n_surrogates": args.n_surrogates,
                    "seed": args.seed + target_index,
                    "target_file": str(args.target.resolve()),
                    "mask_file": str(args.mask.resolve()),
                    "projection_file": str(field_paths[field_name].resolve()),
                }
            )
            print(f"  {field_name:16s} rho={observed:+.3f} p={p_smash:.4g}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
