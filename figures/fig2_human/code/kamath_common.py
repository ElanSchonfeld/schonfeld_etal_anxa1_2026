"""Shared palettes and helpers for the Figure 2 (human) panels."""

FAMILY_COLORS = {"Sox6": "#0173B2", "Calb1": "#DE8F05", "Gad2": "#2ca02c"}
FAMILY_ORDER = ["Sox6", "Calb1", "Gad2"]

SUBTYPE_COLORS = {
    "Calb1:Ccdc192": "#1f77b4", "Calb1:Chrm2": "#aec7e8", "Calb1:Gipr": "#ff7f0e",
    "Calb1:Kctd8": "#ffbb78", "Calb1:Lpar1": "#2ca02c", "Calb1:Pde11a": "#d62728",
    "Calb1:Ptprt": "#ff9896", "Calb1:Stac": "#9467bd", "Calb1:Sulf1": "#8c564b",
    "Calb1:Sox6": "#c5b0d5", "Gad2:Ebf2": "#c49c94", "Gad2:Egfr": "#e377c2",
    "Sox6:Arhgap28": "#f7b6d2", "Sox6:Kcnmb2": "#c7c7c7", "Sox6:March3": "#bcbd22",
    "Sox6:Tafa1": "#dbdb8d", "Sox6:Tmem132d": "#17becf", "Sox6:Vcan": "#9edae5",
}

KAMATH_CT_COLORS = {
    "SOX6_AGTR1": "#08519c", "SOX6_DDT": "#2171b5", "SOX6_GFRA2": "#4292c6",
    "SOX6_PART1": "#6baed6", "SOX6_SERPINE2": "#9ecae1",
    "CALB1_CALCR": "#d94701", "CALB1_CRYM_CCDC68": "#e6550d", "CALB1_GEM": "#f16913",
    "CALB1_PPP1R17": "#fd8d3c", "CALB1_RBP4": "#fdae6b", "CALB1_TRHR": "#fdd0a2",
    "CALB1_TH": "#fee6ce",
}
KAMATH_SOX6_TYPES = ["SOX6_AGTR1", "SOX6_DDT", "SOX6_GFRA2", "SOX6_PART1", "SOX6_SERPINE2"]
KAMATH_CALB1_TYPES = ["CALB1_CALCR", "CALB1_CRYM_CCDC68", "CALB1_GEM", "CALB1_PPP1R17",
                      "CALB1_RBP4", "CALB1_TRHR", "CALB1_TH"]


def spell_celltype(ct: str) -> str:
    """SOX6_AGTR1 -> Sox6:Agtr1 ; CALB1_CRYM_CCDC68 -> Calb1:Crym_Ccdc68."""
    s = str(ct)
    if s.upper().startswith("SOX6_"):
        fam, rest = "Sox6", s[5:]
    elif s.upper().startswith("CALB1_"):
        fam, rest = "Calb1", s[6:]
    else:
        return s
    suffix = "_".join(tok.capitalize() for tok in rest.split("_"))
    return f"{fam}:{suffix}"


def infer_family(subtype: str) -> str:
    """Family from a subtype / Kamath Cell_Type string (Calb1 checked before Sox6)."""
    s = str(subtype)
    if ":" in s:
        pref = s.split(":", 1)[0]
        if pref in {"Sox6", "Calb1", "Gad2", "Lef1"}:
            return pref
    u = s.upper()
    if "CALB1" in u:
        return "Calb1"
    if "SOX6" in u:
        return "Sox6"
    if "GAD" in u:
        return "Gad2"
    return "Other"
