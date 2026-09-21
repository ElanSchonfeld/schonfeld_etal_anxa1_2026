"""Routing specification for MoE v4."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

ROOT_NODE = "node"
GAD_POS_NODE = "node|gad_pos"
GAD_NEG_NODE = "node|gad_neg"
SOX6_NODE = "node|gad_neg|sox6"
CALB_NODE = "node|gad_neg|calb"

EXCLUDED_LEAVES = [
    "Lef1",
    "Gad2:Syndig1",
]

EXCLUSION_RATIONALE = (
    "Exclude Lef1 and Gad2:Syndig1 because they are non-midbrain DA populations "
    "without stable mouse-human correspondence."
)

ROUTING_TREE = {
    ROOT_NODE: {
        "gad_pos": {},
        "gad_neg": {
            "sox6": {},
            "calb": {},
        },
    }
}


def _family_from_mapping(subtype: str, subtype_to_family: Optional[Dict[str, str]]) -> Optional[str]:
    if not subtype_to_family:
        return None
    return subtype_to_family.get(subtype) or subtype_to_family.get(str(subtype))


def _has_token(value: str, token: str) -> bool:
    return token.casefold() in value.casefold()


def is_gad_positive(subtype: str, subtype_to_family: Optional[Dict[str, str]] = None) -> bool:
    family = _family_from_mapping(subtype, subtype_to_family)
    if family is not None:
        return _has_token(family, "Gad")
    return _has_token(subtype, "Gad")


def is_sox6_family(subtype: str, subtype_to_family: Optional[Dict[str, str]] = None) -> bool:
    family = _family_from_mapping(subtype, subtype_to_family)
    if family is not None:
        return _has_token(family, "Sox6")
    return _has_token(subtype, "Sox6")


def is_calb_family(subtype: str, subtype_to_family: Optional[Dict[str, str]] = None) -> bool:
    family = _family_from_mapping(subtype, subtype_to_family)
    if family is not None:
        return _has_token(family, "Calb")
    return _has_token(subtype, "Calb")


def assign_gadneg_branch(
    subtype: str,
    subtype_to_family: Optional[Dict[str, str]] = None,
    default_branch: str = "calb",
) -> str:
    """Assign gad_neg subtypes to sox6 vs calb; default if unknown."""

    if is_sox6_family(subtype, subtype_to_family=subtype_to_family):
        return "sox6"
    if is_calb_family(subtype, subtype_to_family=subtype_to_family):
        return "calb"
    return default_branch


def build_node_to_leaves(
    leaves: List[str],
    subtype_to_family: Optional[Dict[str, str]] = None,
    default_gadneg_branch: str = "calb",
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Build a node->leaves mapping for the routing spec."""

    node_to_leaves: Dict[str, List[str]] = {
        ROOT_NODE: [],
        GAD_POS_NODE: [],
        GAD_NEG_NODE: [],
        SOX6_NODE: [],
        CALB_NODE: [],
    }
    unknown: List[str] = []

    for leaf in leaves:
        node_to_leaves[ROOT_NODE].append(leaf)
        if is_gad_positive(leaf, subtype_to_family=subtype_to_family):
            node_to_leaves[GAD_POS_NODE].append(leaf)
        else:
            node_to_leaves[GAD_NEG_NODE].append(leaf)
            branch = assign_gadneg_branch(
                leaf, subtype_to_family=subtype_to_family, default_branch=default_gadneg_branch
            )
            if branch == "sox6":
                node_to_leaves[SOX6_NODE].append(leaf)
            elif branch == "calb":
                node_to_leaves[CALB_NODE].append(leaf)
            else:
                unknown.append(leaf)

    return node_to_leaves, unknown


def filter_excluded_leaves(
    leaves: List[str],
    excluded_leaves: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Filter excluded leaves from a list of leaf labels."""

    excluded = set(excluded_leaves or EXCLUDED_LEAVES)
    kept = [leaf for leaf in leaves if leaf not in excluded]
    removed = [leaf for leaf in leaves if leaf in excluded]
    return kept, removed


def node_dir_name(node_name: str) -> str:
    return node_name.replace("|", "__")


def infer_family_from_subtype(subtype: str) -> str:
    """Infer family from a mouse subtype label (e.g. ``"Sox6:Vcan"`` -> ``"Sox6"``)."""
    s = str(subtype)
    if ":" in s:
        pref = s.split(":", 1)[0]
        if pref in {"Gad2", "Sox6", "Calb1", "Lef1"}:
            return pref
    if "Gad2" in s:
        return "Gad2"
    if "Sox6" in s:
        return "Sox6"
    if "Calb1" in s:
        return "Calb1"
    if "Lef1" in s:
        return "Lef1"
    return "Other"


def infer_family_from_human_celltype(cell_type: str) -> str:
    """Infer family from a human Cell_Type label (e.g. ``"SOX6_AGTR1"`` -> ``"Sox6"``)."""
    s = str(cell_type).upper()
    if s.startswith("SOX6"):
        return "Sox6"
    if s.startswith("CALB1"):
        return "Calb1"
    if s.startswith("GAD2"):
        return "Gad2"
    if s.startswith("LEF1"):
        return "Lef1"
    return "Other"


def infer_family(subtype: str) -> str:
    """Simple family inference: split on ``:`` and return the prefix."""
    if subtype == "Lef1":
        return "Lef1"
    if ":" in subtype:
        return subtype.split(":", 1)[0]
    return "Other"
