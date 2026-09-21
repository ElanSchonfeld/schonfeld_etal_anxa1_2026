#!/usr/bin/env python3
"""Freeze the active hemisphere-stratified Figure 5 display inputs.

Run: ``PPMI_ROOT=/path/to/ppmi python freeze_fig5_hemi_genetics.py``
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st

from hemisphere_datscan_data import (
    add_clinical_hemisphere_columns,
    add_spatial_anxa1_columns,
    exclude_double_carriers,
    genetic_group,
    load_scan_with_hemisphere_fields,
)
from hemisphere_voxel_analysis import compute_hemisphere_voxel_analysis

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"

PPMI_VALUE = os.environ.get("PPMI_ROOT")
if not PPMI_VALUE:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PPMI = Path(PPMI_VALUE).expanduser().resolve()
if not PPMI.is_dir():
    raise FileNotFoundError(f"PPMI_ROOT is not a directory: {PPMI}")


def _star(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def freeze_driver_trajectory(output_dir: Path = FROZEN) -> None:
    """Freeze the bilateral prodromal and MA/LA diagnosed trajectories."""
    bilateral = "SBR_anxa1put"
    more_affected = "SBR_anxa1put_MA"
    less_affected = "SBR_anxa1put_LA"
    scans = exclude_double_carriers(load_scan_with_hemisphere_fields(PPMI))
    scans["carr"] = np.where(
        scans.LRRK2_carrier.fillna(False).astype(bool),
        "LRRK2",
        np.where(scans.GBA_carrier.fillna(False).astype(bool), "GBA", "none"),
    )
    scans["updrs"] = pd.to_numeric(scans.MDSUPDRS3_total, errors="coerce")
    scans.loc[(scans.updrs < 0) | (scans.updrs > 108), "updrs"] = np.nan

    diagnosed = scans[scans.COHORT_DEFINITION == "Parkinson's Disease"].copy()
    baseline = (
        diagnosed.sort_values(["PATNO", "scan_date"])
        .drop_duplicates("PATNO")
        .copy()
    )
    baseline["grp3"] = baseline.apply(genetic_group, axis=1)
    diagnosed["grp3"] = diagnosed.PATNO.map(
        baseline.set_index("PATNO")["grp3"]
    )
    healthy_control = (
        scans[scans.COHORT_DEFINITION == "Healthy Control"]
        .sort_values(["PATNO", "scan_date"])
        .drop_duplicates("PATNO")[bilateral]
        .mean()
    )

    presymptomatic_rows = (
        scans[
            (scans.COHORT_DEFINITION == "Prodromal") & (scans.updrs <= 3)
        ]
        .sort_values(["PATNO", "scan_date"])
        .drop_duplicates("PATNO")
    )
    presymptomatic = {
        carrier: presymptomatic_rows[
            presymptomatic_rows.carr == carrier
        ][bilateral].dropna()
        for carrier in ("none", "LRRK2", "GBA")
    }
    early = diagnosed[
        (diagnosed.yrs_since_onset >= 0) & (diagnosed.yrs_since_onset < 2)
    ]

    def early_side(group: str, column: str) -> pd.Series:
        return early[early.grp3 == group][column].dropna()

    prodromal_bins = [
        (0, 2),
        (2, 4),
        (4, 7),
        (7, 10),
        (10, 14),
        (14, 19),
        (19, 26),
        (26, 34),
    ]
    diagnosed_bins = [(year, year + 1) for year in range(12)]
    minimum_count = 5
    prodromal = {
        carrier: scans[
            (scans.COHORT_DEFINITION == "Prodromal")
            & (scans.carr == carrier)
        ]
        .dropna(subset=[bilateral, "updrs"])
        .sort_values("scan_date")
        .drop_duplicates("PATNO")
        for carrier in ("none", "LRRK2", "GBA")
    }

    def prodromal_bin_count(carrier: str, low: float, high: float) -> int:
        return int(
            (
                (prodromal[carrier].updrs >= low)
                & (prodromal[carrier].updrs < high)
            ).sum()
        )

    prodromal_x_cap = max(
        (low + high) / 2
        for low, high in prodromal_bins
        if all(
            prodromal_bin_count(carrier, low, high) >= minimum_count
            for carrier in ("LRRK2", "GBA")
        )
    )

    def prodromal_trajectory(carrier: str):
        subset = prodromal[carrier]
        x_values = []
        means = []
        errors = []
        for low, high in prodromal_bins:
            if (low + high) / 2 > prodromal_x_cap:
                continue
            values = subset[
                (subset.updrs >= low) & (subset.updrs < high)
            ][bilateral]
            if len(values) >= minimum_count:
                x_values.append((low + high) / 2)
                means.append(values.mean())
                errors.append(values.sem())
        return np.array(x_values), np.array(means), np.array(errors)

    def diagnosed_trajectory(group: str, column: str):
        subset = diagnosed[diagnosed.grp3 == group].dropna(
            subset=[column, "yrs_since_onset"]
        )
        x_values = []
        means = []
        errors = []
        for low, high in diagnosed_bins:
            values = subset[
                (subset.yrs_since_onset >= low)
                & (subset.yrs_since_onset < high)
            ][column]
            if len(values) >= 4:
                x_values.append((low + high) / 2 if high <= 8 else 8.5)
                means.append(values.mean())
                errors.append(values.sem())
        return np.array(x_values), np.array(means), np.array(errors)

    groups = [
        ("none", "idiopathic"),
        ("LRRK2", "LRRK2"),
        ("GBA", "GBA"),
    ]
    rows = [
        {
            "kind": "hc",
            "side": "",
            "group": "",
            "x": np.nan,
            "mean": float(healthy_control),
            "sem": np.nan,
            "n": np.nan,
        },
        {
            "kind": "xcap",
            "side": "",
            "group": "",
            "x": float(prodromal_x_cap),
            "mean": np.nan,
            "sem": np.nan,
            "n": np.nan,
        },
    ]
    for carrier, group in groups:
        values = presymptomatic[carrier]
        rows.append(
            {
                "kind": "presymp",
                "side": "bilateral",
                "group": group,
                "x": np.nan,
                "mean": float(values.mean()),
                "sem": float(values.sem()) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    for side, column in (("MA", more_affected), ("LA", less_affected)):
        for _, group in groups:
            values = early_side(group, column)
            rows.append(
                {
                    "kind": "early",
                    "side": side,
                    "group": group,
                    "x": np.nan,
                    "mean": float(values.mean()),
                    "sem": float(values.sem()) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
            )
    for carrier, _ in groups:
        x_values, means, errors = prodromal_trajectory(carrier)
        for x_value, mean, error in zip(x_values, means, errors):
            rows.append(
                {
                    "kind": "prod_traj",
                    "side": "bilateral",
                    "group": carrier,
                    "x": float(x_value),
                    "mean": float(mean),
                    "sem": float(error),
                    "n": np.nan,
                }
            )
    for side, column in (("MA", more_affected), ("LA", less_affected)):
        for group in ("idiopathic", "LRRK2", "GBA"):
            x_values, means, errors = diagnosed_trajectory(group, column)
            for x_value, mean, error in zip(x_values, means, errors):
                rows.append(
                    {
                        "kind": "diag_traj",
                        "side": side,
                        "group": group,
                        "x": float(x_value),
                        "mean": float(mean),
                        "sem": float(error),
                        "n": np.nan,
                    }
                )

    def stat_row(kind, side, group, x=np.nan, mean=np.nan, sem=np.nan, n=np.nan):
        return {
            "kind": kind,
            "side": side,
            "group": group,
            "x": x,
            "mean": mean,
            "sem": sem,
            "n": n,
        }

    def slope_row(kind: str, side: str, group: str, subset: pd.DataFrame):
        model = smf.ols("ly ~ dur", data=subset[subset.grp3 == group]).fit()
        return stat_row(
            kind,
            side,
            group,
            mean=float(100 * (np.exp(model.params["dur"]) - 1)),
            sem=float(model.pvalues["dur"]),
            n=int(subset[subset.grp3 == group].PATNO.nunique()),
        )

    def interaction_row(kind: str, side: str, group: str, subset: pd.DataFrame):
        pair = subset[subset.grp3.isin(["idiopathic", group])].copy()
        pair["gg"] = (pair.grp3 == group).astype(int)
        model = smf.ols("ly ~ dur * gg", data=pair).fit()
        return stat_row(
            kind,
            side,
            group,
            mean=float(model.params["dur:gg"]),
            sem=float(model.pvalues["dur:gg"]),
        )

    for side, column in (("MA", more_affected), ("LA", less_affected)):
        decline = diagnosed.dropna(subset=[column, "yrs_since_onset"])
        decline = decline[
            (decline.yrs_since_onset >= 0)
            & (decline.yrs_since_onset < 12)
            & (decline[column] > 0)
        ].copy()
        decline["ly"] = np.log(decline[column])
        decline["dur"] = decline.yrs_since_onset
        for group in ("idiopathic", "LRRK2", "GBA"):
            rows.append(slope_row("diag_slope", side, group, decline))
        for group in ("LRRK2", "GBA"):
            rows.append(interaction_row("diag_slope_vs_idio", side, group, decline))
        under_eight = decline[decline.dur < 8]
        for group in ("idiopathic", "LRRK2", "GBA"):
            rows.append(slope_row("diag_slope_lt8", side, group, under_eight))
        rows.append(
            interaction_row("diag_slope_lt8_vs_idio", side, "GBA", under_eight)
        )

        def gap(frame: pd.DataFrame, low: float, high: float) -> float:
            window = frame[(frame.dur >= low) & (frame.dur < high)]
            return (
                window[window.grp3 == "GBA"][column].mean()
                - window[window.grp3 == "idiopathic"][column].mean()
            )

        rng = np.random.default_rng(42)
        patients = {
            group: decline[decline.grp3 == group].PATNO.unique()
            for group in ("GBA", "idiopathic")
        }
        observed = gap(decline, 4, 12) - gap(decline, 0, 2)
        bootstrap = []
        for _ in range(2000):
            sample = pd.concat(
                [
                    decline[
                        decline.PATNO.isin(
                            rng.choice(
                                patients[group], len(patients[group]), replace=True
                            )
                        )
                    ]
                    for group in patients
                ]
            )
            bootstrap.append(gap(sample, 4, 12) - gap(sample, 0, 2))
        bootstrap = np.array(bootstrap)
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        p_value = float(2 * min(np.mean(bootstrap >= 0), np.mean(bootstrap <= 0)))
        rows.append(stat_row("gap_early", side, "GBA", mean=float(gap(decline, 0, 2))))
        rows.append(stat_row("gap_late", side, "GBA", mean=float(gap(decline, 4, 12))))
        rows.append(
            stat_row("gap_widening", side, "GBA", mean=float(observed), sem=p_value)
        )
        rows.append(
            stat_row(
                "gap_widening_ci", side, "GBA", x=float(low), mean=float(high)
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "driver_trajectory_hemi.csv", index=False)
    print(
        f"  driver_trajectory_hemi.csv ({len(rows)} rows) "
        f"HC={healthy_control:.3f} XCAP={prodromal_x_cap:.1f}"
    )


def _build_confound_data():
    cohort = pd.read_csv(PPMI / "derived/datscan_recon_cohort.csv")
    cohort["PATNO"] = cohort.PATNO.astype(int)
    scans = pd.read_parquet(PPMI / "derived/ppmi_scan_datscan.parquet")
    scans["PATNO"] = scans.PATNO.astype(int)
    backbone = pd.read_parquet(PPMI / "derived/datscan_backbone.parquet")
    backbone["PATNO"] = backbone.PATNO.astype(int)
    scans = scans.merge(
        backbone[["PATNO", "EVENT_ID", "yrs_since_dx"]],
        on=["PATNO", "EVENT_ID"],
        how="left",
    )
    scans = exclude_double_carriers(scans)
    add_clinical_hemisphere_columns(scans)
    add_spatial_anxa1_columns(scans, PPMI)
    pd_patients = set(
        cohort[
            (cohort.cohort == "Parkinson's Disease") & cohort.in_xing_sbr
        ].PATNO
    )
    data = scans[scans.PATNO.isin(pd_patients) & scans.ma_hemi.notna()].copy()
    baseline = (
        data.sort_values(["PATNO", "scan_date"])
        .drop_duplicates("PATNO")
        .copy()
    )
    baseline["grp3"] = baseline.apply(genetic_group, axis=1)
    data["grp3"] = data.PATNO.map(baseline.set_index("PATNO")["grp3"])
    data = data.rename(
        columns={
            "AGE_AT_VISIT": "age",
            "yrs_since_onset": "dur",
            "yrs_since_dx": "ddx",
        }
    )
    data["sex"] = data.SEX
    baseline = data.sort_values(["PATNO", "scan_date"]).drop_duplicates("PATNO").copy()
    return data, baseline


def freeze_confound(
    output_dir: Path = FROZEN,
    *,
    locked_bins_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
) -> dict:
    """Freeze adjusted means, matched contrasts, and voxel concordance."""
    groups = ("idiopathic", "LRRK2", "GBA")
    rng = np.random.default_rng(42)
    data, baseline = _build_confound_data()

    def adjusted_means(outcome: str, rhs: str):
        needed = [
            column
            for column in rhs.replace("+", " ").split()
            if column not in ("", "C(grp3)")
        ]
        subset = baseline[baseline.grp3.isin(groups)].dropna(
            subset=[outcome] + needed
        ).copy()
        subset["grp3"] = pd.Categorical(subset.grp3, categories=list(groups))
        model = smf.ols(f"{outcome} ~ C(grp3) + {rhs}", data=subset).fit()
        at = {
            column: subset[column].mean()
            for column in needed
            if subset[column].dtype != object
        }
        if "sex" in needed:
            at["sex"] = subset.sex.mean()
        predictions = {}
        p_values = {}
        intervals = {}
        for group in groups:
            prediction = model.get_prediction(pd.DataFrame([{**at, "grp3": group}]))
            predictions[group] = float(prediction.predicted_mean[0])
            low, high = prediction.conf_int(alpha=0.05)[0]
            intervals[group] = (float(low), float(high))
            p_values[group] = model.pvalues.get(f"C(grp3)[T.{group}]", np.nan)
        return predictions, p_values, intervals, len(subset)

    def matched_contrast(
        group1: str, group2: str, outcome: str, caliper: float = 0.4
    ):
        pool = data[data.grp3 == group1].dropna(
            subset=[outcome, "ddx", "age"]
        ).copy()
        target = data[data.grp3 == group2].dropna(
            subset=[outcome, "ddx", "age"]
        ).copy()
        used = set()
        group_values = []
        reference_values = []
        for _, case in target.iterrows():
            candidates = pool[
                (pool.sex == case.sex)
                & (pool.ddx.sub(case.ddx).abs() <= caliper)
                & (pool.age.sub(case.age).abs() <= 5)
                & (~pool.index.isin(used))
            ]
            if len(candidates) == 0:
                continue
            pick = candidates.iloc[
                candidates.ddx.sub(case.ddx).abs().values.argmin()
            ]
            used.add(pick.name)
            group_values.append(case[outcome])
            reference_values.append(pick[outcome])
        difference = np.array(group_values) - np.array(reference_values)
        if len(difference) < 4:
            return np.nan, np.nan, np.nan, np.nan, len(difference)
        bootstrap = np.array(
            [
                difference[rng.integers(0, len(difference), len(difference))].mean()
                for _ in range(2000)
            ]
        )
        return (
            float(difference.mean()),
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
            float(st.wilcoxon(difference).pvalue),
            len(difference),
        )

    bars = []
    rhs = "age + sex + dur + ddx"
    for suffix, hemisphere in (("MA", "MA"), ("LA", "LA")):
        for region, label in (("Caudate", "caudate"), ("anxa1put", "anxa1")):
            predictions, p_values, intervals, count = adjusted_means(
                f"SBR_{region}_{suffix}", rhs
            )
            for group in groups:
                low, high = intervals[group]
                annotation = (
                    _star(p_values[group])
                    if group != "idiopathic" and hemisphere == "MA"
                    else ""
                )
                bars.append(
                    {
                        "subpanel": "B",
                        "key": label,
                        "hemi": hemisphere,
                        "group": group,
                        "value": predictions[group],
                        "lo": low,
                        "hi": high,
                        "sem": np.nan,
                        "star": annotation,
                        "pval": (
                            float(p_values[group])
                            if np.isfinite(p_values[group])
                            else np.nan
                        ),
                        "n": int(count),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bars).to_csv(output_dir / "hemi_confound_bars.csv", index=False)
    print(f"  hemi_confound_bars.csv ({len(bars)} active rows)")

    comparisons = [
        ("idiopathic", "LRRK2"),
        ("idiopathic", "GBA"),
        ("LRRK2", "GBA"),
    ]
    contrasts = []
    for group1, group2 in comparisons:
        for suffix, hemisphere in (("MA", "MA"), ("LA", "LA")):
            effect, low, high, p_value, count = matched_contrast(
                group1, group2, f"SBR_anxa1put_{suffix}"
            )
            contrasts.append(
                {
                    "subpanel": "D",
                    "contrast": f"{group2} vs {group1}",
                    "g1": group1,
                    "g2": group2,
                    "hemi": hemisphere,
                    "eff": effect,
                    "lo": low,
                    "hi": high,
                    "pval": p_value,
                    "star": _star(p_value),
                    "n": int(count),
                }
            )
    pd.DataFrame(contrasts).to_csv(
        output_dir / "hemi_confound_contrasts.csv", index=False
    )
    print(f"  hemi_confound_contrasts.csv ({len(contrasts)} rows)")

    print("  computing MA/LA voxel excess and concordance ...", flush=True)
    voxel = compute_hemisphere_voxel_analysis(
        PPMI,
        locked_bins_path=locked_bins_path,
        annotation_path=annotation_path,
    )
    concordance_rows = []
    for hemisphere in ("MA", "LA"):
        values = voxel[hemisphere]
        for territory, key in (
            ("Anxa1", "anxa1"),
            ("Sox6", "sox6all"),
            ("Sox6-rest", "sox6rest"),
        ):
            concordance_rows.append(
                {
                    "subpanel": "F",
                    "hemi": hemisphere,
                    "territory": territory,
                    "rho": values[f"rho_{key}"],
                    "lo": values[f"ci_{key}"][0],
                    "hi": values[f"ci_{key}"][1],
                    "pval": values[f"p_{key}"],
                    "star": _star(values[f"p_{key}"]),
                    "dRho": values["dRho"],
                    "dRho_lo": values["ci"][0],
                    "dRho_hi": values["ci"][1],
                    "dRho_p": values["p"],
                    "dRho_star": _star(values["p"]),
                }
            )
    pd.DataFrame(concordance_rows).to_csv(
        output_dir / "hemi_confound_concordance.csv", index=False
    )
    pair_count = int(voxel["n_pairs"])
    with (output_dir / "hemi_confound_meta.json").open("w") as handle:
        json.dump(
            {
                "n_pairs": pair_count,
                "input_volume_count": int(voxel["input_volume_count"]),
                "input_volume_set_sha256": voxel["input_volume_set_sha256"],
            },
            handle,
        )
    print(f"  hemi_confound_concordance.csv n_pairs={pair_count}")

    excess = voxel["excess_fsl_canon"]
    striatum = voxel["STR_fsl"]
    putamen = voxel["PUT_fsl"]
    affine = voxel["aff_fsl"]
    mni = voxel["fsl_reference"].get_fdata()
    low, high = np.nanpercentile(excess[striatum], [25, 98])
    inset = {"elo": float(low), "ehi": float(high)}
    z_index = int(round((0 - affine[2, 3]) / affine[2, 2]))
    for hemisphere, hemisphere_mask in (
        ("MA", voxel["left_fsl"]),
        ("LA", voxel["right_fsl"]),
    ):
        x_extent = np.where(hemisphere_mask.any(axis=(1, 2)))[0]
        y_extent = np.where(hemisphere_mask.any(axis=(0, 2)))[0]
        x0, x1 = x_extent.min() - 3, x_extent.max() + 3
        y0, y1 = y_extent.min() - 3, y_extent.max() + 3
        inset[f"{hemisphere}_bg"] = np.rot90(
            mni[x0:x1, y0:y1, z_index]
        ).astype(np.float32)
        inset[f"{hemisphere}_disp"] = np.rot90(
            np.where(
                hemisphere_mask[x0:x1, y0:y1, z_index],
                excess[x0:x1, y0:y1, z_index],
                np.nan,
            )
        ).astype(np.float32)
        inset[f"{hemisphere}_put"] = np.rot90(
            (putamen & hemisphere_mask)[x0:x1, y0:y1, z_index].astype(float)
        ).astype(np.float32)
    np.savez_compressed(output_dir / "hemi_confound_inset.npz", **inset)
    print(f"  hemi_confound_inset.npz (elo={low:.3f} ehi={high:.3f})")
    return voxel


if __name__ == "__main__":
    freeze_driver_trajectory()
    voxel_result = freeze_confound()
    sys.path.insert(0, str(HERE.parents[2] / "common"))
    import freeze_provenance as provenance

    upstream = [
        Path(__file__).resolve(),
        HERE / "hemisphere_datscan_data.py",
        HERE / "hemisphere_voxel_analysis.py",
        HERE / "projection_fields.py",
        PPMI / "derived/ppmi_scan_datscan.parquet",
        PPMI / "derived/datscan_recon_cohort.csv",
        PPMI / "derived/datscan_backbone.parquet",
        PPMI / "derived/datscan_spatial/anxa1_putaminal_spatial.parquet",
        *voxel_result["input_paths"],
    ]
    provenance.stamp(FROZEN, "fig5_hemi_genetics", list(dict.fromkeys(upstream)))
    print("done (hemisphere genetics group).")
