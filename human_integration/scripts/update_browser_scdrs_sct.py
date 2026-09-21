#!/usr/bin/env python
"""Insert the deployed pooled SCT scDRS scores into a DopaBase Human H5AD."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent.parent
TRAITS = (
    "pd",
    "scz",
    "adhd",
    "nicotine_dep",
    "bipolar",
    "chronic_pain",
    "mdd",
    "oud",
    "ocd",
    "anxiety_any",
    "fibromyalgia",
)
KINDS = ("norm_score", "zscore")


def _env_path(name: str, fallback: Path) -> Path:
    return Path(os.environ.get(name, fallback))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a browser H5AD and insert the 22 deployed scDRS columns."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=_env_path(
            "DOPABASE_SCDRS_OUTPUT",
            HERE / "results" / "human_scores_sct_pooled.parquet",
        ),
        help="pooled SCT score Parquet; env: DOPABASE_SCDRS_OUTPUT",
    )
    parser.add_argument(
        "--browser-h5ad",
        type=Path,
        default=_env_path(
            "DOPABASE_BROWSER_H5AD", HERE / "data" / "human_da_browser_new.h5ad"
        ),
        help="unmodified staging browser; env: DOPABASE_BROWSER_H5AD",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=_env_path(
            "DOPABASE_SCDRS_BROWSER_OUT",
            HERE / "data" / "human_da_browser_scored.h5ad",
        ),
        help="new scored browser path; env: DOPABASE_SCDRS_BROWSER_OUT",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output only after a new copy verifies successfully",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs without copying or writing"
    )
    return parser


def _obs_index(h5ad: Path) -> pd.Index:
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs["_index"]
        return pd.Index(np.asarray(obs[index_key]).astype(str))


def _validate_inputs(
    scores_path: Path, browser_path: Path
) -> tuple[pd.DataFrame, list[str], pd.Index]:
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    if not browser_path.is_file():
        raise FileNotFoundError(browser_path)

    scores = pd.read_parquet(scores_path)
    scores.index = scores.index.astype(str)
    wanted = [f"scdrs_{trait}_{kind}" for trait in TRAITS for kind in KINDS]
    missing = [column for column in wanted if column not in scores.columns]
    if missing:
        raise ValueError(f"score table is missing {len(missing)} columns: {missing}")
    if scores.index.has_duplicates:
        raise ValueError("score table index contains duplicate cell identifiers")

    browser_index = _obs_index(browser_path)
    if not scores.index.equals(browser_index):
        shared = browser_index.intersection(scores.index)
        raise ValueError(
            "score index must exactly match browser obs order: "
            f"{len(shared)} of {len(browser_index)} browser cells are shared"
        )
    values = scores[wanted].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("score table contains non-finite values")
    return scores, wanted, browser_index


def _insert_scores(path: Path, scores: pd.DataFrame, wanted: list[str]) -> int:
    with h5py.File(path, "r+") as handle:
        obs = handle["obs"]
        existing = [key for key in obs.keys() if key.startswith("scdrs_")]
        for key in existing:
            del obs[key]
        for column in wanted:
            dataset = obs.create_dataset(
                column, data=scores[column].to_numpy(dtype=np.float32)
            )
            dataset.attrs["encoding-type"] = "array"
            dataset.attrs["encoding-version"] = "0.2.0"
        order = [
            column
            for column in obs.attrs["column-order"].astype(str)
            if not column.startswith("scdrs_")
        ]
        obs.attrs["column-order"] = np.asarray(
            order + wanted, dtype=h5py.string_dtype()
        )
    return len(existing)


def _verify(path: Path, scores: pd.DataFrame, wanted: list[str], index: pd.Index) -> None:
    if not _obs_index(path).equals(index):
        raise RuntimeError("browser obs index changed during score insertion")
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        found = sorted(key for key in obs.keys() if key.startswith("scdrs_"))
        if found != sorted(wanted):
            raise RuntimeError("scDRS column verification failed")
        for column in wanted:
            written = np.asarray(obs[column])
            expected = scores[column].to_numpy(dtype=np.float32)
            if written.dtype != np.float32 or not np.array_equal(written, expected):
                raise RuntimeError(f"verification failed for {column}")


def main() -> None:
    args = _parser().parse_args()
    source = args.browser_h5ad.resolve()
    output = args.output_h5ad.resolve()
    if source == output:
        raise ValueError("--output-h5ad must differ from --browser-h5ad")

    scores, wanted, browser_index = _validate_inputs(args.scores, source)
    print(f"  browser: {source}")
    print(f"  scores: {args.scores.resolve()} ({len(browser_index)} cells, {len(TRAITS)} traits)")
    if args.dry_run:
        print("  dry run: inputs valid; nothing written")
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".h5ad", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        removed = _insert_scores(temporary, scores, wanted)
        _verify(temporary, scores, wanted, browser_index)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"  removed {removed} prior scDRS columns and wrote {len(wanted)} float32 columns")
    print(f"  verified {output}")


if __name__ == "__main__":
    main()
