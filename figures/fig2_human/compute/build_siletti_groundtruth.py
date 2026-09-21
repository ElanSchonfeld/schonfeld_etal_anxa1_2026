#!/usr/bin/env python3
"""Siletti ground-truth labels for the 823 DA cells.

Run: python build_siletti_groundtruth.py
"""
from pathlib import Path
from os import environ
import re
import pandas as pd

MAN = Path(environ["M2H_SOURCE_ROOT"])
HI = Path(__file__).resolve().parents[3] / "human_integration"
FROZEN = Path(__file__).resolve().parent.parent / "frozen"

MARKER = {1869: "EBF2", 1870: "GRP", 1871: "CHRNB3", 1872: "SCUBE1",
          1873: "DLK1", 1874: "CCKAR", 1875: "ASB4", 1876: "AGTR1"}

cw = pd.read_csv(HI / "results" / "siletti_da_whb_crosswalk.csv")
sub = pd.read_excel(MAN / "Data" / "Human" / "Siletti" / "subcluster_annotation.xlsx")
mem = pd.read_csv(MAN / "Data" / "Human" / "Siletti" / "cluster_to_cluster_annotation_membership.csv")

sub = sub[sub["Subcluster"].isin(MARKER)].set_index("Subcluster")
color = (mem[(mem.cluster_annotation_term_set_name == "subcluster") & mem.cluster_alias.isin(MARKER)]
         .set_index("cluster_alias")["color_hex_triplet"].to_dict())


def nt_of(sid):
    nt = str(sub.loc[sid, "Neurotransmitter"])
    return "DA" if "NT-DA" in nt else ("GABA" if "NT-GABA" in nt else "VGLUT")


def region_of(sid):
    return re.split(r":", str(sub.loc[sid, "Top ROI"]))[0].strip()


rows = []
for _, r in cw.iterrows():
    barcode = f"{int(r['census_query_row'])}-Siletti"
    if bool(r["source_cluster_395"]):
        sid = int(r["source_subcluster_id"])
        rows.append({"barcode": barcode, "siletti_subcluster": f"Splat_395_{sid}",
                     "siletti_subtype": f"{MARKER[sid]} ({nt_of(sid)}, {region_of(sid)})",
                     "siletti_nt": nt_of(sid), "siletti_color": color.get(sid, "#888888"),
                     "in_siletti_da": True})
    else:
        rows.append({"barcode": barcode, "siletti_subcluster": r["source_supercluster"],
                     "siletti_subtype": "Novel_DA", "siletti_nt": "Novel",
                     "siletti_color": "#cfcfcf", "in_siletti_da": False})

out = pd.DataFrame(rows)
out.to_csv(FROZEN / "siletti_groundtruth.csv", index=False)
print(f"wrote siletti_groundtruth.csv: {len(out)} cells")
print("  in Siletti DA cluster 395:", int(out.in_siletti_da.sum()), "| Novel_DA:", int((~out.in_siletti_da).sum()))
print(out.siletti_subtype.value_counts().to_string())
