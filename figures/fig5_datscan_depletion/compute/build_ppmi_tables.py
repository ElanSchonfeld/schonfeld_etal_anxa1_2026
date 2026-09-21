#!/usr/bin/env python3
"""Build the DaTscan recon cohort table, the per-scan DaTscan table and the DaTscan backbone table.

Run: PPMI_ROOT=/path/to/ppmi python compute/build_ppmi_tables.py
"""
import os
from pathlib import Path

import pandas as pd

from datscan_registration import recon_members

STAMP = "13Jul2026"
REGIONS = {
    "PuR": "PRECOMMISSURAL_PUTAMEN",
    "PuC": "POSCOMMISSURAL_PUTAMEN",
    "PuPV": "POSVENTRALPUTAMEN",
    "PuPD": "POSDORSALPUTAMEN",
    "PuAV": "PREVENTRALPUTAMEN",
    "PuAD": "PREDORSALPUTAMEN",
    "Caudate": "CAUDATE",
    "Putamen": "PUTAMEN",
    "Striatum": "STRIATUM",
}
VISIT_MONTH = {
    "SC": -1.5, "BL": 0.0, "V02": 6.0, "V04": 12.0, "V05": 18.0, "V06": 24.0,
    "V08": 36.0, "V10": 48.0, "V12": 60.0, "V13": 72.0, "V14": 84.0, "V15": 96.0,
    "V17": 120.0,
}
GENES = ["LRRK2", "GBA", "SNCA", "APOE"]
CARRIER_GENES = ["LRRK2", "GBA", "SNCA"]
DIAGNOSIS_LABEL = {
    1.0: "PD",
    7.0: "EssentialTremor",
    17.0: "Control",
    24.0: "Prodromal_motor",
    25.0: "Prodromal_synuclein",
}

_root = os.environ.get("PPMI_ROOT")
if not _root:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PPMI = Path(_root).expanduser().resolve()
DERIVED = PPMI / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


def read(folder, name):
    return pd.read_csv(PPMI / folder / f"{name}_{STAMP}.csv", low_memory=False)


def as_patno(frame):
    frame = frame.copy()
    frame["PATNO"] = frame.PATNO.astype(int)
    return frame


xing = as_patno(read("study_info", "Xing_Core_Lab_-_Quant_SBR"))
status = as_patno(read("demographics", "Participant_Status"))
status_first = status.drop_duplicates("PATNO").set_index("PATNO")

members = recon_members(PPMI / "images" / "datscan_spect" / "_zips")
dates = {}
for patno, date in members:
    dates.setdefault(patno, []).append(date)
xing_ids = set(xing.PATNO)
cohort_rows = []
for patno, visit_dates in dates.items():
    label = status_first.COHORT_DEFINITION.get(patno)
    cohort_rows.append({
        "PATNO": patno,
        "cohort": label if isinstance(label, str) else "?",
        "baseline_date": min(visit_dates),
        "n_recon_scans": len(set(visit_dates)),
        "in_xing_sbr": patno in xing_ids,
    })
cohort = pd.DataFrame(cohort_rows)
cohort.to_csv(DERIVED / "datscan_recon_cohort.csv", index=False)
print(f"datscan_recon_cohort.csv {cohort.shape}")

scans = xing[xing.DATSCAN_ANALYZED == "Yes"].drop_duplicates(["PATNO", "EVENT_ID"]).reset_index(drop=True)
keys = pd.MultiIndex.from_arrays([scans.PATNO, scans.EVENT_ID])
base = pd.DataFrame({"PATNO": scans.PATNO.to_numpy(), "EVENT_ID": scans.EVENT_ID.to_numpy()})
base["visit_month"] = base.EVENT_ID.map(VISIT_MONTH).astype(float)
base["scan_date"] = pd.to_datetime(scans.DATSCAN_DATE, format="%m/%Y", errors="coerce").to_numpy()
base["DATSCAN_LIGAND"] = scans.DATSCAN_LIGAND.to_numpy()

ages = as_patno(read("demographics", "Age_at_visit"))
ages = ages.drop_duplicates(["PATNO", "EVENT_ID"]).set_index(["PATNO", "EVENT_ID"])
demographics = as_patno(read("demographics", "Demographics")).drop_duplicates("PATNO").set_index("PATNO")

updrs = as_patno(read("study_info", "MDS-UPDRS_Part_III"))
updrs["np3_sum"] = pd.to_numeric(updrs.NP3TOT, errors="coerce")
updrs_sorted = updrs.sort_values(["PATNO", "EVENT_ID", "REC_ID"], kind="stable")
updrs_last = updrs_sorted.drop_duplicates(["PATNO", "EVENT_ID"], keep="last").set_index(["PATNO", "EVENT_ID"])
nhy_last = (
    updrs_sorted.dropna(subset=["NHY"])
    .drop_duplicates(["PATNO", "EVENT_ID"], keep="last")
    .set_index(["PATNO", "EVENT_ID"])
)

history = as_patno(read("study_info", "PD_Diagnosis_History")).drop_duplicates("PATNO").set_index("PATNO")
onset = pd.to_datetime(base.PATNO.map(history.SXDT), format="%m/%Y", errors="coerce")
diagnosis = pd.to_datetime(base.PATNO.map(history.PDDXDT), format="%m/%Y", errors="coerce")

sbr = {}
for short, long in REGIONS.items():
    left = scans[f"{long}_L_REF_CWM"].astype(float).to_numpy()
    right = scans[f"{long}_R_REF_CWM"].astype(float).to_numpy()
    sbr[f"SBR_{short}"] = (left + right) / 2
    sbr[f"SBR_{short}_L"] = left
    sbr[f"SBR_{short}_R"] = right

scan_table = base.copy()
scan_table["AGE_AT_VISIT"] = keys.map(ages.AGE_AT_VISIT).astype(float)
scan_table["SEX"] = scan_table.PATNO.map(demographics.SEX).astype(float)
scan_table["sex_label"] = scan_table.SEX.map({0.0: "Female", 1.0: "Male"})
scan_table["COHORT_DEFINITION"] = scan_table.PATNO.map(status_first.COHORT_DEFINITION)
scan_table["ENROLL_STATUS"] = scan_table.PATNO.map(status_first.ENROLL_STATUS)
scan_table["NHY"] = keys.map(updrs_last.NHY).astype(float)
scan_table["MDSUPDRS3_total"] = keys.map(updrs_last.np3_sum).astype(float)
scan_table["DOMSIDE"] = scan_table.PATNO.map(history.DOMSIDE).astype(float)
scan_table["yrs_since_onset"] = (scan_table.scan_date - onset).dt.days / 365.25

consensus = as_patno(read("demographics", "iu_genetic_consensus_20251025")).drop_duplicates("PATNO").set_index("PATNO")
for gene in GENES:
    values = scan_table.PATNO.map(consensus[gene])
    scan_table[gene] = values.map(lambda value: None if pd.isna(value) else str(value).strip())
for gene in CARRIER_GENES:
    scan_table[f"{gene}_carrier"] = scan_table[gene].map(lambda value: None if value is None else value != "0")
for column, values in sbr.items():
    scan_table[column] = values
assert len(scan_table) == len(scans)
scan_table.to_parquet(DERIVED / "ppmi_scan_datscan.parquet", index=False)
print(f"ppmi_scan_datscan.parquet {scan_table.shape}")

diagnoses = as_patno(read("study_info", "Primary_Research_Diagnosis"))
diagnoses["infodt"] = pd.to_datetime(diagnoses.INFODT, format="%m/%Y", errors="coerce")
latest = (
    diagnoses.sort_values(["PATNO", "infodt", "REC_ID"], kind="stable")
    .drop_duplicates("PATNO", keep="last")
    .set_index("PATNO")
)
backbone = base.copy()
primary_diagnosis = backbone.PATNO.map(latest.PRIMDIAG).astype(float)
backbone["cohort"] = primary_diagnosis.map(
    lambda value: None if pd.isna(value) else DIAGNOSIS_LABEL.get(value, f"Other_{int(value)}")
)
backbone["PRIMDIAG"] = primary_diagnosis
backbone["NHY"] = keys.map(nhy_last.NHY).astype(float)
backbone["DOMSIDE"] = scan_table.DOMSIDE.to_numpy()
backbone["yrs_since_onset"] = scan_table.yrs_since_onset.to_numpy()
backbone["yrs_since_dx"] = (backbone.scan_date - diagnosis).dt.days / 365.25
for column, values in sbr.items():
    backbone[column] = values
assert len(backbone) == len(scans)
backbone.to_parquet(DERIVED / "datscan_backbone.parquet", index=False)
print(f"datscan_backbone.parquet {backbone.shape}")
