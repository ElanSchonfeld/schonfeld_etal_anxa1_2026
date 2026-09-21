#!/usr/bin/env python3
"""Freeze the two active Figure 5 genetic-stratum panels from PPMI tables.

Run: ``PPMI_ROOT=/path/to/ppmi python freeze_fig5_genetics.py``
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemisphere_datscan_data import load_scan_with_hemisphere_fields

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
FROZEN.mkdir(parents=True, exist_ok=True)

PPMI_VALUE = os.environ.get("PPMI_ROOT")
if not PPMI_VALUE:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PPMI = Path(PPMI_VALUE).expanduser().resolve()

READOUT = "SBR_anxa1put"
COLORS = {
    "idiopathic": "#6b6b6b",
    "LRRK2": "#3f8fd0",
    "GBA": "#2ca25f",
    "HC": "#111111",
    "none": "#c7c7c7",
}


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def group3(row):
    lrrk2 = bool(row.LRRK2_carrier) if pd.notna(row.LRRK2_carrier) else False
    gba = bool(row.GBA_carrier) if pd.notna(row.GBA_carrier) else False
    return "LRRK2" if lrrk2 else ("GBA" if gba else "idiopathic")


def load_scan_table():
    scans = load_scan_with_hemisphere_fields(PPMI)
    double = scans.LRRK2_carrier.fillna(False).astype(bool) & scans.GBA_carrier.fillna(False).astype(bool)
    scans = scans[~double].copy()
    scans["carr"] = np.where(
        scans.LRRK2_carrier.fillna(False).astype(bool),
        "LRRK2",
        np.where(scans.GBA_carrier.fillna(False).astype(bool), "GBA", "none"),
    )
    return scans


def freeze_driver(scans):
    scans = scans.copy()
    scans["updrs"] = pd.to_numeric(scans.MDSUPDRS3_total, errors="coerce")
    scans.loc[(scans.updrs < 0) | (scans.updrs > 108), "updrs"] = np.nan
    pre = scans[
        (scans.COHORT_DEFINITION == "Healthy Control")
        | ((scans.COHORT_DEFINITION == "Prodromal") & (scans.updrs <= 3))
    ].sort_values(["PATNO", "scan_date"]).drop_duplicates("PATNO").copy()
    pre["lab"] = np.where(pre.COHORT_DEFINITION == "Healthy Control", "HC", pre.carr)
    pre = pre.dropna(subset=[READOUT])

    model_data = pre[pre.lab.isin(["HC", "LRRK2", "GBA"])].dropna(
        subset=["AGE_AT_VISIT", "sex"]
    ).copy()
    model_data["lab"] = pd.Categorical(model_data.lab, categories=["HC", "LRRK2", "GBA"])
    model = smf.ols(f"{READOUT} ~ C(lab) + AGE_AT_VISIT + sex", data=model_data).fit()

    order = ["none", "LRRK2", "GBA", "HC"]
    labels = ["non-genetic\n(RBD/hyposmia)", "LRRK2", "GBA", "HC"]
    rng = np.random.default_rng(42)
    points = []
    groups = []
    for idx, (group, label) in enumerate(zip(order, labels)):
        values = pre.loc[pre.lab == group, READOUT].to_numpy()
        x_values = rng.normal(idx, 0.06, len(values))
        points.extend(
            {"group": group, "x": float(x), "y": float(y)}
            for x, y in zip(x_values, values)
        )
        annotation = ""
        if group in ("LRRK2", "GBA"):
            key = f"C(lab)[T.{group}]"
            annotation = f"vs HC {model.params[key]:+.2f}\n{star(model.pvalues[key])}"
        groups.append(
            {
                "idx": idx,
                "group": group,
                "label": label,
                "color": COLORS[group],
                "mean": float(values.mean()),
                "ann": annotation,
                "n": int(len(values)),
            }
        )

    pd.DataFrame(points).to_csv(FROZEN / "genetics_driver_points.csv", index=False)
    pd.DataFrame(groups).to_csv(FROZEN / "genetics_driver_groups.csv", index=False)
    print(f"froze genetics driver: {len(points)} participant-level display points")


def load_contrast_data(scans):
    cohort = pd.read_csv(PPMI / "derived/datscan_recon_cohort.csv")
    cohort["PATNO"] = cohort.PATNO.astype(int)
    backbone = pd.read_parquet(PPMI / "derived/datscan_backbone.parquet")
    backbone["PATNO"] = backbone.PATNO.astype(int)

    data = scans.merge(
        backbone[["PATNO", "EVENT_ID", "yrs_since_dx"]],
        on=["PATNO", "EVENT_ID"],
        how="left",
    )
    pd_patients = set(
        cohort[
            (cohort.cohort == "Parkinson's Disease") & cohort.in_xing_sbr
        ].PATNO
    )
    data = data[data.PATNO.isin(pd_patients)].copy()
    baseline = data.sort_values(["PATNO", "scan_date"]).drop_duplicates("PATNO").copy()
    baseline["grp3"] = baseline.apply(group3, axis=1)
    data["grp3"] = data.PATNO.map(baseline.set_index("PATNO")["grp3"])
    data = data.rename(
        columns={
            "AGE_AT_VISIT": "age",
            "yrs_since_onset": "dur",
            "yrs_since_dx": "ddx",
            "NHY": "hy",
        }
    )
    data["sex"] = data.SEX
    data["scan_dt"] = pd.to_datetime(data.scan_date, errors="coerce")
    data["scan_mo"] = data.scan_dt.values.astype("datetime64[M]")

    medication = pd.read_csv(
        PPMI / "study_info/LEDD_Concomitant_Medication_Log_13Jul2026.csv",
        low_memory=False,
    )
    medication["PATNO"] = medication.PATNO.astype(int)
    medication["start"] = pd.to_datetime(
        medication.STARTDT, format="%m/%Y", errors="coerce"
    ).values.astype("datetime64[M]")
    medication["stop"] = pd.to_datetime(
        medication.STOPDT, format="%m/%Y", errors="coerce"
    ).values.astype("datetime64[M]")
    medication["LEDD"] = pd.to_numeric(medication.LEDD, errors="coerce")
    medication = medication.dropna(subset=["start"])

    def ledd_at(patient, month):
        if pd.isna(month):
            return np.nan
        rows = medication[
            (medication.PATNO == patient)
            & (medication.start <= month)
            & ((medication.stop > month) | medication.stop.isna())
        ]
        return float(rows.LEDD.sum()) if len(rows) else 0.0

    data["LEDD"] = [
        ledd_at(patient, month)
        for patient, month in zip(data.PATNO, data.scan_mo)
    ]
    return data


def pair_contrast(data, group1, group2, rhs):
    needed = [value for value in rhs.replace("+", " ").split() if value]
    subset = data[data.grp3.isin([group1, group2])].dropna(
        subset=["SBR_Putamen"] + needed
    ).copy()
    subset["gg"] = (subset.grp3 == group2).astype(int)
    model = smf.mixedlm(
        f"SBR_Putamen ~ gg + {rhs}", data=subset, groups=subset.PATNO
    ).fit(method="lbfgs")
    interval = model.conf_int().loc["gg"]
    return (
        float(model.params["gg"]),
        float(interval.iloc[0]),
        float(interval.iloc[1]),
        float(model.pvalues["gg"]),
    )


def matched_contrast(data, group1, group2, rng, caliper=0.4):
    pool = data[data.grp3 == group1].dropna(
        subset=["SBR_Putamen", "ddx", "age"]
    ).copy()
    target = data[data.grp3 == group2].dropna(
        subset=["SBR_Putamen", "ddx", "age"]
    ).copy()
    used = set()
    target_values = []
    reference_values = []
    for _, row in target.iterrows():
        candidates = pool[
            (pool.sex == row.sex)
            & (pool.ddx.sub(row.ddx).abs() <= caliper)
            & (pool.age.sub(row.age).abs() <= 5)
            & (~pool.index.isin(used))
        ]
        if len(candidates) == 0:
            continue
        pick = candidates.iloc[candidates.ddx.sub(row.ddx).abs().values.argmin()]
        used.add(pick.name)
        target_values.append(row.SBR_Putamen)
        reference_values.append(pick.SBR_Putamen)
    difference = np.asarray(target_values) - np.asarray(reference_values)
    bootstrap = np.asarray(
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
    )


def freeze_contrasts(scans):
    data = load_contrast_data(scans)
    comparisons = [
        ("idiopathic", "LRRK2"),
        ("idiopathic", "GBA"),
        ("LRRK2", "GBA"),
    ]
    steps = [
        ("age+sex", "age + sex", "#b0b0b0"),
        ("+onset (disease time)", "age + sex + dur", "#565656"),
        ("+H&Y (severity)", "age + sex + dur + hy", "#111111"),
        ("+LEDD (medication)", "age + sex + dur + LEDD", "#7b3294"),
    ]
    rng = np.random.default_rng(42)
    rows = []
    headers = []
    y = 0.0
    for group1, group2 in comparisons:
        header_color = COLORS[group2] if group1 == "idiopathic" else "#444"
        headers.append(
            {
                "text": f"{group2} vs {group1}",
                "x": -0.30,
                "y": y - 0.78,
                "color": header_color,
            }
        )
        for label, rhs, color in steps:
            effect, low, high, p_value = pair_contrast(data, group1, group2, rhs)
            rows.append(
                {
                    "y": round(y, 1),
                    "eff": effect,
                    "lo": low,
                    "hi": high,
                    "color": color,
                    "marker": "o",
                    "lw": 2.8,
                    "stat_text": f"{effect:+.3f}  p={p_value:.0e} {star(p_value)}",
                    "ylabel": label,
                }
            )
            y += 1.0
        effect, low, high, p_value = matched_contrast(
            data, group1, group2, rng
        )
        rows.append(
            {
                "y": round(y, 1),
                "eff": effect,
                "lo": low,
                "hi": high,
                "color": "#c1121f",
                "marker": "D",
                "lw": 3.0,
                "stat_text": f"{effect:+.3f}  p={p_value:.0e} {star(p_value)}",
                "ylabel": "dx-matched",
            }
        )
        y += 2.2

    pd.DataFrame(rows).to_csv(FROZEN / "genetics_contrasts.csv", index=False)
    pd.DataFrame(headers).to_csv(
        FROZEN / "genetics_contrasts_headers.csv", index=False
    )
    print(f"froze genetic-stratum contrasts: {len(rows)} model estimates")


if __name__ == "__main__":
    scan_table = load_scan_table()
    freeze_driver(scan_table)
    freeze_contrasts(scan_table)
    sys.path.insert(0, str(HERE.parents[2] / "common"))
    import freeze_provenance as provenance

    provenance.stamp(
        FROZEN,
        "fig5_genetics",
        [
            PPMI / "derived/ppmi_scan_datscan.parquet",
            PPMI / "derived/datscan_recon_cohort.csv",
            PPMI / "derived/datscan_backbone.parquet",
            PPMI / "study_info/LEDD_Concomitant_Medication_Log_13Jul2026.csv",
        ],
    )
