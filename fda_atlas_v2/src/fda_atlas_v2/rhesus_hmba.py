"""Strict rhesus-only Allen HMBA inputs for the FDA Atlas."""

from __future__ import annotations

import pandas as pd


RHESUS_TAXON = "NCBITaxon:9544"
RHESUS_NAME = "Macaca mulatta"
HMBA_DA_GROUPS = (
    "SN SOX6 Dopa",
    "SN-VTR CALB1 Dopa",
    "SN-VTR GAD2 Dopa",
)
CANONICAL_LEAVES = (
    "Calb1:Ccdc192",
    "Calb1:Chrm2",
    "Calb1:Gipr",
    "Calb1:Kctd8",
    "Calb1:Lpar1",
    "Calb1:Pde11a",
    "Calb1:Ptprt",
    "Calb1:Sox6",
    "Calb1:Stac",
    "Calb1:Sulf1",
    "Gad2:Ebf2",
    "Gad2:Egfr",
    "Sox6:Arhgap28",
    "Sox6:Kcnmb2",
    "Sox6:March3",
    "Sox6:Tafa1",
    "Sox6:Tmem132d",
    "Sox6:Vcan",
)
ANXA1_LEAVES = ("Sox6:Tafa1", "Sox6:Vcan")


def validate_rhesus_donors(donors: pd.DataFrame) -> pd.DataFrame:
    required = {"donor_label", "donor_species", "species_scientific_name"}
    if missing := sorted(required - set(donors.columns)):
        raise ValueError(f"HMBA donor registry is missing columns: {missing}")
    frame = donors.copy()
    if frame["donor_label"].astype(str).duplicated().any():
        raise ValueError("HMBA donor labels must be unique")
    taxon_is_rhesus = frame["donor_species"].astype(str).eq(RHESUS_TAXON)
    name_is_rhesus = frame["species_scientific_name"].astype(str).eq(RHESUS_NAME)
    if not taxon_is_rhesus.equals(name_is_rhesus):
        raise ValueError("HMBA donor taxon and scientific name disagree")
    selected = frame.loc[taxon_is_rhesus].copy()
    if selected.empty:
        raise ValueError("HMBA donor registry contains no verified rhesus donors")
    return selected.sort_values("donor_label", kind="stable").reset_index(drop=True)


def select_rhesus_da_cells(
    membership: pd.DataFrame,
    hierarchy: pd.DataFrame,
    cell_metadata: pd.DataFrame,
    donors: pd.DataFrame,
) -> pd.DataFrame:
    """Select taxonomy-defined midbrain DA cells from verified rhesus donors."""

    membership_required = {"cell_label", "cluster_alias"}
    hierarchy_required = {
        "cluster_alias",
        "cluster_annotation_term_set_name",
        "cluster_annotation_term_name",
    }
    metadata_required = {"cell_label", "donor_label", "library_label"}
    for label, frame, required in (
        ("membership", membership, membership_required),
        ("hierarchy", hierarchy, hierarchy_required),
        ("cell metadata", cell_metadata, metadata_required),
    ):
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"HMBA {label} is missing columns: {missing}")
    if membership["cell_label"].astype(str).duplicated().any():
        raise ValueError("HMBA cell membership contains duplicate cell labels")
    if cell_metadata["cell_label"].astype(str).duplicated().any():
        raise ValueError("HMBA cell metadata contains duplicate cell labels")

    rhesus = validate_rhesus_donors(donors)
    groups = hierarchy[
        hierarchy["cluster_annotation_term_set_name"].astype(str).eq("Group")
        & hierarchy["cluster_annotation_term_name"].astype(str).isin(HMBA_DA_GROUPS)
    ][["cluster_alias", "cluster_annotation_term_name"]].drop_duplicates()
    if groups["cluster_alias"].astype(str).duplicated().any():
        raise ValueError("HMBA DA cluster aliases map to multiple taxonomy groups")
    selected = membership.merge(
        groups,
        on="cluster_alias",
        how="inner",
        validate="many_to_one",
    ).merge(
        cell_metadata,
        on="cell_label",
        how="inner",
        validate="one_to_one",
    ).merge(
        rhesus[["donor_label", "donor_species", "species_scientific_name"]],
        on="donor_label",
        how="inner",
        validate="many_to_one",
    )
    if selected.empty:
        raise ValueError("strict rhesus HMBA DA selection contains no cells")
    if not set(selected["donor_species"].astype(str)) == {RHESUS_TAXON}:
        raise ValueError("non-rhesus donor leaked into the HMBA DA selection")
    return selected.sort_values("cell_label", kind="stable").reset_index(drop=True)


def population_members() -> dict[str, set[str]]:
    """Return the branch-free public HMoE population membership map."""

    members = {
        family: {leaf for leaf in CANONICAL_LEAVES if leaf.startswith(f"{family}:")}
        for family in ("Calb1", "Gad2", "Sox6")
    }
    members["Anxa1"] = set(ANXA1_LEAVES)
    members.update({leaf: {leaf} for leaf in CANONICAL_LEAVES})
    return members
