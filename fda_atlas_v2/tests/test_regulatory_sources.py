import csv
import json
import zipfile

import pandas as pd

from fda_atlas_v2.contracts import validate_table
from fda_atlas_v2.regulatory_sources import (
    build_regulatory_substances,
    exact_name_key,
    load_drugsfda_archive,
    load_purple_book,
    regulatory_substance_id,
)


def _write_drugsfda_fixture(path):
    records = {
        "meta": {"last_updated": "2026-07-09", "results": {"total": 4}},
        "results": [
            {
                "application_number": "NDA002386",
                "sponsor_name": "GD SEARLE LLC",
                "submissions": [
                    {
                        "submission_type": "ORIG",
                        "submission_status": "AP",
                        "submission_status_date": "19400419",
                    }
                ],
                "products": [
                    {
                        "product_number": "002",
                        "reference_drug": "No",
                        "reference_standard": "No",
                        "brand_name": "AMINOPHYLLIN",
                        "active_ingredients": [
                            {"name": "AMINOPHYLLINE", "strength": "100MG"}
                        ],
                        "dosage_form": "TABLET",
                        "route": "ORAL",
                        "marketing_status": "Discontinued",
                    }
                ],
            },
            {
                "application_number": "NDA216974",
                "sponsor_name": "INNOVIVA SPECIALTY THERAPEUTICS",
                "submissions": [],
                "products": [
                    {
                        "product_number": "001",
                        "reference_drug": "Yes",
                        "brand_name": "XACDURO (COPACKAGED)",
                        "active_ingredients": [
                            {"name": "SULBACTAM SODIUM", "strength": "1G/VIAL"},
                            {"name": "DURLOBACTAM SODIUM", "strength": "0.5G/VIAL"},
                            {"name": "DURLOBACTAM SODIUM", "strength": "0.5G/VIAL"},
                        ],
                        "dosage_form": "POWDER;POWDER",
                        "route": "INTRAVENOUS;INTRAVENOUS",
                        "marketing_status": "Prescription",
                    }
                ],
            },
            {
                "application_number": "ANDA208097",
                "sponsor_name": "TEST",
                "submissions": [],
                "products": [
                    {
                        "product_number": "001",
                        "brand_name": "TENTATIVE COMBINATION",
                        "active_ingredients": [
                            {"name": "ETHINYL ESTRADIOL", "strength": "A"},
                            {"name": "ETHINYL ESTRADIOL", "strength": "A"},
                        ],
                        "marketing_status": "None (Tentative Approval)",
                    }
                ],
            },
            {
                "application_number": "NDA221337",
                "sponsor_name": "AIR COMPANY",
                "submissions": [],
                "products": [
                    {
                        "product_number": "001",
                        "brand_name": "MEDICAL AIR, USP",
                        "active_ingredients": [],
                        "dosage_form": "GAS",
                        "route": "RESPIRATORY (INHALATION)",
                        "marketing_status": "Prescription",
                    }
                ],
            },
        ],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("drug-drugsfda.json", json.dumps(records))


def _purple_headers():
    return [
        "N/R/U", "Applicant", "BLA Number", "Proprietary Name", "Proper Name",
        "License Type", "Strength", "Dosage Form", "Route of Administration",
        "Product Presentation", "Marketing Status", "Licensure", "Approval Date",
        "Inter. Approval Date", "Ref. Product Proper Name",
        "Ref. Product Proprietary Name", "Supplement Number", "Submission Type",
        "Inter. Supplement Number", "License Number", "Product Number", "Center",
        "Date of First Licensure", "Exclusivity Expiration Date",
        "First Interchangeable Exclusivity Exp. Date",
        "Ref. Product Exclusivity Exp. Date", "Orphan Exclusivity Exp. Date",
        "Patent List Provided",
    ]


def _purple_row(*, change="", bla, product, name, brand, licensure="Licensed",
                marketing="Rx", route="Intravenous"):
    row = {column: "" for column in _purple_headers()}
    row.update(
        {
            "N/R/U": change,
            "Applicant": "APPLICANT",
            "BLA Number": bla,
            "Proprietary Name": brand,
            "Proper Name": name,
            "License Type": "351(a)",
            "Strength": "1MG",
            "Dosage Form": "Injection",
            "Route of Administration": route,
            "Product Presentation": "Single-Dose Vial",
            "Marketing Status": marketing,
            "Licensure": licensure,
            "Approval Date": "1-Jan-20",
            "Product Number": product,
            "Center": "CBER",
        }
    )
    return [row[column] for column in _purple_headers()]


def _write_purple_fixture(path):
    headers = _purple_headers()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Purple Book Monthly Historical Data Changes Report"])
        writer.writerow([])
        writer.writerow(["Newly Approved Products"])
        writer.writerow(headers)
        writer.writerow(
            _purple_row(
                change="U", bla="101084", product="001",
                name="Normal Horse Serum", brand="N/A"
            )
        )
        writer.writerow([])
        writer.writerow(headers)
        writer.writerow(
            _purple_row(
                bla="101084", product="001", name="Normal Horse Serum", brand="N/A"
            )
        )
        writer.writerow(
            _purple_row(
                bla="125720", product="001", name="valoctocogene roxaparvovec-rvox",
                brand="ROCTAVIAN", licensure="Voluntarily Revoked", marketing="Disc"
            )
        )
        writer.writerow(
            _purple_row(
                bla="103634", product="001", name="Imaging agent", brand="MYOSCINT",
                route=""
            )
        )


def test_exact_name_identity_preserves_biologically_material_qualifiers():
    pairs = [
        ("Antihemophilic Factor (Human)", "Antihemophilic Factor (Recombinant)"),
        ("Thrombin, Topical (Bovine)", "Thrombin, Topical (Human)"),
        ("Antivenin (Latrodectus mactans)", "Antivenin (Micrurus fulvius)"),
    ]
    for left, right in pairs:
        assert exact_name_key(left) != exact_name_key(right)
        assert regulatory_substance_id(left) != regulatory_substance_id(right)


def test_drugsfda_parser_preserves_products_duplicate_ingredients_and_exclusions(tmp_path):
    path = tmp_path / "drugs.zip"
    _write_drugsfda_fixture(path)
    parsed = load_drugsfda_archive(path, release_id="test-release")

    assert len(parsed.products) == 3
    assert len(parsed.ingredients) == 4
    assert parsed.products["product_id"].is_unique
    assert set(parsed.products["product_id"]) == {
        "drugsfda:NDA002386:002",
        "drugsfda:NDA216974:001",
        "drugsfda:NDA221337:001",
    }

    xacduro = parsed.ingredients[
        parsed.ingredients["product_id"].eq("drugsfda:NDA216974:001")
    ]
    assert len(xacduro) == 3
    duplicated = xacduro[xacduro["ingredient_text"].eq("DURLOBACTAM SODIUM")]
    assert duplicated["duplicate_ordinal"].tolist() == [1, 2]
    assert duplicated["source_row_id"].nunique() == 2

    air = parsed.products.set_index("product_id").loc["drugsfda:NDA221337:001"]
    assert air["resolution_status"] == "no_ingredient_in_source"
    assert not parsed.ingredients["product_id"].eq(air.name).any()

    tentative = parsed.reconciliation[
        parsed.reconciliation["product_id"].eq("drugsfda:ANDA208097:001")
    ]
    assert set(tentative["reason"]) == {
        "excluded_tentative_product",
        "excluded_parent_tentative",
    }
    assert len(tentative) == 3
    assert parsed.products["approval_date"].isna().all()
    assert (
        parsed.products.set_index("product_id")
        .loc["drugsfda:NDA002386:002", "earliest_approved_orig_submission_date"]
        == "19400419"
    )

    validate_table("regulatory_products", parsed.products)
    validate_table("product_ingredients", parsed.ingredients)
    validate_table("source_row_reconciliation", parsed.reconciliation)


def test_purple_parser_uses_last_header_and_reconciles_change_summary(tmp_path):
    path = tmp_path / "purple.csv"
    _write_purple_fixture(path)
    parsed = load_purple_book(path, release_id="test-release", source_release="2026-06")

    assert len(parsed.products) == 3
    assert len(parsed.ingredients) == 3
    assert parsed.products["product_id"].is_unique
    assert parsed.audit["change_summary_rows"] == 1
    assert parsed.audit["full_product_rows"] == 3

    change = parsed.reconciliation[
        parsed.reconciliation["source_section"].eq("monthly_change_summary")
    ].iloc[0]
    assert change["disposition"] == "excluded"
    assert change["reason"] == "monthly_change_summary_duplicate"
    assert change["product_id"] == "purplebook:101084:001"

    roctavian = parsed.products.set_index("product_id").loc["purplebook:125720:001"]
    assert bool(roctavian["revoked"])
    assert bool(roctavian["discontinued"])
    assert not bool(roctavian["currently_marketed"])
    myoscint = parsed.products.set_index("product_id").loc["purplebook:103634:001"]
    assert pd.isna(myoscint["route"])

    validate_table("regulatory_products", parsed.products)
    validate_table("product_ingredients", parsed.ingredients)
    validate_table("source_row_reconciliation", parsed.reconciliation)


def test_substances_merge_only_identical_conservative_name_keys(tmp_path):
    drugs_path = tmp_path / "drugs.zip"
    purple_path = tmp_path / "purple.csv"
    _write_drugsfda_fixture(drugs_path)
    _write_purple_fixture(purple_path)
    drugs = load_drugsfda_archive(drugs_path, release_id="test-release")
    purple = load_purple_book(
        purple_path, release_id="test-release", source_release="2026-06"
    )
    edges = pd.concat([drugs.ingredients, purple.ingredients], ignore_index=True)
    substances = build_regulatory_substances(edges, release_id="test-release")

    assert substances["substance_id"].is_unique
    assert set(substances["normalized_name"]) == {
        exact_name_key(name)
        for name in edges["ingredient_text"]
    }
    validate_table("regulatory_substances", substances)
