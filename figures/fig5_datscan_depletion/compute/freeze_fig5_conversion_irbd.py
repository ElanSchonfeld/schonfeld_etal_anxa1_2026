#!/usr/bin/env python
"""Freeze the three aggregate iRBD odds ratios plotted in the Figure 5 conversion panel.

Run: python freeze_fig5_conversion_irbd.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from conversion_paths import PRIVATE

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen"
FAMILIES = [
    ("anxa1_str", "Anxa1 family"),
    ("sox6_str", "Sox6-all"),
    ("calb1_str", "Calb1"),
]


def zscore(values):
    return (values - np.nanmean(values)) / np.nanstd(values)


survival = pd.read_parquet(PRIVATE / "cohort_phenoconversion.parquet")
territories = pd.read_parquet(PRIVATE / "family_terr_sbr.parquet")
family_keys = [key for key, _ in FAMILIES]

data = survival.drop(columns=["postput", "striatum"], errors="ignore")
data = data.merge(territories[["PATNO", "striatum"] + family_keys], on="PATNO").reset_index(drop=True)
data = data[data.tyears_pd > 0].reset_index(drop=True)
data["age_z"] = zscore(data.age)
data["sexf"] = data.sex.astype(float)
data["burden"] = -zscore(data.striatum)
data["enrlrbd"] = pd.to_numeric(data.ENRLRBD, errors="coerce").fillna(0)

rows = []
for key, label in FAMILIES:
    predictor = "dep_" + key
    data[predictor] = -zscore(data[key])
    columns = [predictor, "burden", "age_z", "sexf"]
    model = sm.Logit(data.enrlrbd, sm.add_constant(data[columns])).fit(disp=0)
    beta = float(model.params[predictor])
    se = float(model.bse[predictor])
    rows.append({
        "pred": key,
        "label": label,
        "n": int(len(data)),
        "events": int(data.enrlrbd.sum()),
        "HR": float(np.exp(beta)),
        "lo": float(np.exp(beta - 1.96 * se)),
        "hi": float(np.exp(beta + 1.96 * se)),
        "p": float(model.pvalues[predictor]),
    })

result = pd.DataFrame(rows)
result.to_csv(FROZEN / "conversion_generalises_irbd.csv", index=False)
meta = {
    "n": int(len(data)),
    "events_pd": int(pd.to_numeric(data.event_pd, errors="coerce").fillna(0).sum()),
    "events_lbd": int(pd.to_numeric(data.event_lbd, errors="coerce").fillna(0).sum()),
    "n_irbd": int(data.enrlrbd.sum()),
    "events_irbd": int(data.loc[data.enrlrbd == 1, "event_pd"].sum()),
}
with open(FROZEN / "conversion_generalises_meta.json", "w") as handle:
    json.dump(meta, handle, indent=2)

print(result.to_string(index=False))
print(json.dumps(meta, indent=2))
