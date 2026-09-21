#!/usr/bin/env python
"""Score the 11 DopaBase Human scDRS traits with the deployed SCT workflow."""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent.parent
STUDIES = ("Kamath", "Siletti")
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
SEED = 42
N_CTRL = 1000


def _env_path(name: str, fallback: Path) -> Path:
    return Path(os.environ.get(name, fallback))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pooled human scDRS scoring on per-study SCT corrected counts."
    )
    parser.add_argument(
        "--sct-root",
        type=Path,
        default=_env_path("DOPABASE_SCT_ROOT", HERE / "inputs" / "scdrs_sct"),
        help=(
            "directory containing sct_Kamath and sct_Siletti, each with "
            "matrix.mtx, genes.txt, and cells.txt; env: DOPABASE_SCT_ROOT"
        ),
    )
    parser.add_argument(
        "--gene-set-dir",
        type=Path,
        default=_env_path(
            "DOPABASE_GENESET_DIR", HERE / "inputs" / "scdrs_gene_sets"
        ),
        help="directory containing one <trait>.gs file per trait; env: DOPABASE_GENESET_DIR",
    )
    parser.add_argument(
        "--browser-h5ad",
        type=Path,
        default=_env_path(
            "DOPABASE_BROWSER_H5AD", HERE / "data" / "human_da_browser_new.h5ad"
        ),
        help=(
            "raw-count staging browser used for cell order, donor, and n_genes; "
            "env: DOPABASE_BROWSER_H5AD"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_env_path(
            "DOPABASE_SCDRS_OUTPUT",
            HERE / "results" / "human_scores_sct_pooled.parquet",
        ),
        help="output Parquet path; env: DOPABASE_SCDRS_OUTPUT",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output after the new file has been written successfully",
    )
    return parser


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty input: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate identifiers in {path}")
    return values


def _input_labels(sct_root: Path) -> dict[str, tuple[list[str], list[str]]]:
    labels = {}
    for study in STUDIES:
        directory = sct_root / f"sct_{study}"
        labels[study] = (
            _read_lines(directory / "genes.txt"),
            _read_lines(directory / "cells.txt"),
        )
        if not (directory / "matrix.mtx").is_file():
            raise FileNotFoundError(directory / "matrix.mtx")
    return labels


def _dense_pool(
    sct_root: Path,
    labels: dict[str, tuple[list[str], list[str]]],
    common_genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    import scipy.io as sio
    import scipy.sparse as sp

    cells = [cell for study in STUDIES for cell in labels[study][1]]
    dense = np.empty((len(cells), len(common_genes)), dtype=np.float32)
    offset = 0

    for study in STUDIES:
        genes, study_cells = labels[study]
        matrix_path = sct_root / f"sct_{study}" / "matrix.mtx"
        matrix = sp.csr_matrix(sio.mmread(matrix_path))
        expected = (len(study_cells), len(genes))
        if matrix.shape != expected:
            raise ValueError(
                f"{study} SCT orientation is {matrix.shape}, expected {expected} "
                "(cells by genes)"
            )

        positions = {gene: index for index, gene in enumerate(genes)}
        columns = np.asarray([positions[gene] for gene in common_genes], dtype=np.int64)
        for start in range(0, matrix.shape[0], 512):
            stop = min(start + 512, matrix.shape[0])
            dense[offset + start : offset + stop] = matrix[start:stop, columns].toarray()
        offset += matrix.shape[0]
        print(f"  {study}: {matrix.shape[0]} cells x {matrix.shape[1]} genes", flush=True)

    return dense, cells


def _browser_metadata(
    browser_path: Path, cells: list[str], expected_studies: list[str]
) -> tuple[pd.Index, pd.DataFrame, pd.Series, pd.Series]:
    import anndata as ad

    if not browser_path.is_file():
        raise FileNotFoundError(browser_path)

    browser = ad.read_h5ad(browser_path, backed="r")
    try:
        required = {"study", "donor", "disease_status"}
        missing_columns = sorted(required.difference(browser.obs.columns))
        if missing_columns:
            raise ValueError(f"browser obs is missing columns: {missing_columns}")

        browser_index = pd.Index(browser.obs_names.astype(str))
        pool_index = pd.Index(cells)
        if browser_index.has_duplicates or pool_index.has_duplicates:
            raise ValueError("browser and SCT cell identifiers must be unique")
        if set(browser_index) != set(pool_index):
            missing = browser_index.difference(pool_index)
            extra = pool_index.difference(browser_index)
            raise ValueError(
                "SCT cells do not match browser cells exactly: "
                f"{len(missing)} missing and {len(extra)} extra"
            )

        observed_studies = browser.obs["study"].astype(str).reindex(pool_index)
        if not np.array_equal(observed_studies.to_numpy(), np.asarray(expected_studies)):
            raise ValueError("SCT study membership does not match browser obs['study']")

        donor = browser.obs["donor"].reindex(pool_index)
        if donor.isna().any():
            raise ValueError("missing donor values for SCT cells")
        donor = donor.astype(str)

        raw_counts = browser.X[:]
        n_genes = pd.Series(
            np.asarray((raw_counts > 0).sum(axis=1)).ravel().astype(float),
            index=browser_index,
            name="n_genes",
        ).reindex(pool_index)

        metadata = browser.obs[["study", "disease_status"]].copy()
        metadata.index = browser_index
        return browser_index, metadata, donor, n_genes
    finally:
        browser.file.close()


def _atomic_parquet(frame: pd.DataFrame, output: Path, overwrite: bool) -> None:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".parquet", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite to replace it: {args.output.resolve()}"
        )

    import anndata as ad
    import scdrs

    labels = _input_labels(args.sct_root)
    siletti_genes = set(labels["Siletti"][0])
    common_genes = [gene for gene in labels["Kamath"][0] if gene in siletti_genes]
    if not common_genes:
        raise ValueError("Kamath and Siletti SCT inputs have no shared genes")
    print(f"  gene intersection: {len(common_genes)}", flush=True)

    pool_x, cells = _dense_pool(args.sct_root, labels, common_genes)
    expected_studies = [
        study for study in STUDIES for _ in range(len(labels[study][1]))
    ]
    browser_index, metadata, donor, n_genes = _browser_metadata(
        args.browser_h5ad, cells, expected_studies
    )

    pool = ad.AnnData(
        X=pool_x,
        obs=pd.DataFrame(index=pd.Index(cells)),
        var=pd.DataFrame(index=pd.Index(common_genes)),
    )
    print(f"  pooled: {pool.n_obs} cells x {pool.n_vars} genes (dense float32)", flush=True)

    donor_dummies = pd.get_dummies(donor, prefix="donor", drop_first=True).astype(float)
    covariates = pd.concat(
        [
            pd.DataFrame({"const": 1.0, "n_genes": n_genes}, index=pool.obs_names),
            donor_dummies,
        ],
        axis=1,
    )
    print(
        "  covariates: const + n_genes + "
        f"{donor_dummies.shape[1]} donor indicators ({donor.nunique()} donors)",
        flush=True,
    )

    scdrs.preprocess(pool, cov=covariates, n_mean_bin=20, n_var_bin=20, copy=False)
    print("  preprocessing complete", flush=True)

    scores = pd.DataFrame(index=pool.obs_names)
    for trait in TRAITS:
        started = time.time()
        gene_set_path = args.gene_set_dir / f"{trait}.gs"
        if not gene_set_path.is_file():
            raise FileNotFoundError(gene_set_path)
        loaded = scdrs.util.load_gs(
            str(gene_set_path),
            src_species="human",
            dst_species="human",
            to_intersect=list(pool.var_names),
        )
        if len(loaded) != 1:
            raise ValueError(f"expected one gene set in {gene_set_path}, found {len(loaded)}")
        genes, weights = next(iter(loaded.values()))
        if not genes:
            raise ValueError(f"no genes from {gene_set_path} intersect the pooled matrix")
        frame = scdrs.score_cell(
            pool,
            genes,
            weights,
            ctrl_match_key="mean_var",
            n_ctrl=N_CTRL,
            weight_opt="vs",
            random_seed=SEED,
            verbose=False,
        )
        scores[f"scdrs_{trait}_norm_score"] = frame["norm_score"].astype(np.float32)
        scores[f"scdrs_{trait}_zscore"] = frame["zscore"].astype(np.float32)
        print(
            f"  {trait}: {len(genes)} genes, {time.time() - started:.0f}s",
            flush=True,
        )

    scores = scores.reindex(browser_index)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("scDRS produced non-finite output values")
    scores.insert(0, "study", metadata["study"].astype(str).to_numpy())
    scores.insert(0, "condition", metadata["disease_status"].astype(str).to_numpy())
    _atomic_parquet(scores, args.output, args.overwrite)
    print(f"  wrote {args.output.resolve()}: {scores.shape}", flush=True)


if __name__ == "__main__":
    main()
