"""Lossless source parsing for the pinned FDA regulatory denominator."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TENTATIVE_STATUS = "None (Tentative Approval)"
DRUGSFDA_SOURCE = "Drugs@FDA"
PURPLE_SOURCE = "Purple Book"


@dataclass(frozen=True)
class ParsedRegulatorySource:
    products: pd.DataFrame
    ingredients: pd.DataFrame
    reconciliation: pd.DataFrame
    audit: dict[str, Any]


def build_regulatory_substances(ingredients: pd.DataFrame, *, release_id: str) -> pd.DataFrame:
    required = {
        "substance_id",
        "ingredient_text",
        "ingredient_role",
        "product_id",
        "source_row_id",
    }
    missing = sorted(required - set(ingredients.columns))
    if missing:
        raise ValueError(f"product ingredients missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for substance_id, group in ingredients.groupby("substance_id", sort=True):
        names = group["ingredient_text"].dropna().astype(str)
        keys = {exact_name_key(name) for name in names}
        if len(keys) != 1:
            raise ValueError(f"substance ID {substance_id} maps to multiple exact keys")
        counts = names.value_counts()
        preferred = sorted(counts[counts.eq(counts.max())].index)[0]
        variants = sorted(set(names))
        sources = sorted(
            {
                DRUGSFDA_SOURCE if str(value).startswith("openfda_drugsfda:")
                else PURPLE_SOURCE if str(value).startswith("purplebook:")
                else "unknown"
                for value in group["source_row_id"]
            }
        )
        rows.append(
            {
                "release_id": release_id,
                "substance_id": substance_id,
                "preferred_name": preferred,
                "normalized_name": next(iter(keys)),
                "identity_status": "pending_gsrs_identity",
                "identity_source": "exact_regulatory_name",
                "exact_name_key": next(iter(keys)),
                "name_variants": " | ".join(variants),
                "source_roles": " | ".join(
                    sorted(set(group["ingredient_role"].astype(str)))
                ),
                "regulatory_sources": " | ".join(sources),
                "n_product_edges": len(group),
                "n_products": group["product_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _clean(value: Any) -> Any:
    if value is None:
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _text(value: Any) -> str:
    cleaned = _clean(value)
    return "" if pd.isna(cleaned) else str(cleaned)


def exact_name_key(value: Any) -> str:
    """Create a conservative identity key without dropping qualifiers."""

    text = html.unescape(_text(value))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def regulatory_substance_id(value: Any) -> str:
    key = exact_name_key(value)
    if not key:
        return ""
    return f"regsub:{hashlib.sha256(key.encode('utf-8')).hexdigest()}"


def _canonical_sha256(record: Any) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _earliest_approved_orig_submission(application: dict[str, Any]) -> Any:
    dates = sorted(
        _text(row.get("submission_status_date"))
        for row in application.get("submissions", [])
        if _text(row.get("submission_type")).upper() == "ORIG"
        and _text(row.get("submission_status")).upper() == "AP"
        and _text(row.get("submission_status_date"))
    )
    return dates[0] if dates else pd.NA


def _join_exact(values: list[Any]) -> Any:
    strings = [_text(value) for value in values if _text(value)]
    return " + ".join(strings) if strings else pd.NA


def _read_single_json_from_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(
                f"expected one JSON member in {path.name}, found {len(members)}"
            )
        with archive.open(members[0]) as handle:
            return json.load(handle)


def load_drugsfda_archive(
    path: str | Path,
    *,
    release_id: str,
    source_release: str | None = None,
) -> ParsedRegulatorySource:
    """Parse all Drugs@FDA application, product, and ingredient source nodes."""

    path = Path(path)
    payload = _read_single_json_from_zip(path)
    applications = payload.get("results")
    if not isinstance(applications, list):
        raise ValueError("Drugs@FDA archive has no results list")
    meta = payload.get("meta", {})
    detected_release = _text(meta.get("last_updated"))
    release = source_release or detected_release
    if not release:
        raise ValueError("Drugs@FDA source release is unavailable")
    if source_release and detected_release and source_release != detected_release:
        raise ValueError(
            f"Drugs@FDA release mismatch: requested {source_release}, "
            f"archive reports {detected_release}"
        )
    expected = meta.get("results", {}).get("total")
    if expected is not None and int(expected) != len(applications):
        raise ValueError(
            f"Drugs@FDA application count mismatch: meta={expected}, "
            f"archive={len(applications)}"
        )

    products: list[dict[str, Any]] = []
    ingredients: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    seen_product_keys: set[tuple[str, str]] = set()
    tentative_products = 0
    tentative_ingredients = 0
    included_without_ingredient = 0

    for app_index, application in enumerate(applications):
        app = _text(application.get("application_number")).upper()
        if not app:
            raise ValueError(f"Drugs@FDA application {app_index} has no application number")
        app_source_id = f"openfda_drugsfda:{release}:application:{app}"
        app_products = application.get("products") or []
        reconciliation.append(
            {
                "release_id": release_id,
                "regulatory_source": DRUGSFDA_SOURCE,
                "source_release": release,
                "source_row_id": app_source_id,
                "source_section": "application",
                "record_type": "application",
                "parent_source_row_id": pd.NA,
                "natural_key": app,
                "source_location": f"$.results[{app_index}]",
                "source_record_sha256": _canonical_sha256(application),
                "disposition": "container",
                "product_id": pd.NA,
                "reason": "application_container" if app_products else "no_products_in_export",
                "output_table": pd.NA,
                "output_id": pd.NA,
            }
        )
        earliest_orig = _earliest_approved_orig_submission(application)
        for product_index, product in enumerate(app_products):
            number = _text(product.get("product_number"))
            natural_key = (app, number)
            if not number:
                raise ValueError(f"Drugs@FDA product {app}[{product_index}] has no number")
            if natural_key in seen_product_keys:
                raise ValueError(f"duplicate Drugs@FDA product key {app}/{number}")
            seen_product_keys.add(natural_key)

            product_id = f"drugsfda:{app}:{number}"
            product_source_id = f"openfda_drugsfda:{release}:product:{app}:{number}"
            product_path = f"$.results[{app_index}].products[{product_index}]"
            active = product.get("active_ingredients") or []
            tentative = _text(product.get("marketing_status")) == TENTATIVE_STATUS
            if tentative:
                tentative_products += 1
            if not active and not tentative:
                included_without_ingredient += 1
            product_reason = (
                "excluded_tentative_product"
                if tentative
                else "included_product_without_ingredient"
                if not active
                else "included_product"
            )
            reconciliation.append(
                {
                    "release_id": release_id,
                    "regulatory_source": DRUGSFDA_SOURCE,
                    "source_release": release,
                    "source_row_id": product_source_id,
                    "source_section": "product",
                    "record_type": "product",
                    "parent_source_row_id": app_source_id,
                    "natural_key": f"{app}/{number}",
                    "source_location": product_path,
                    "source_record_sha256": _canonical_sha256(product),
                    "disposition": "excluded" if tentative else "retained",
                    "product_id": product_id,
                    "reason": product_reason,
                    "output_table": pd.NA if tentative else "regulatory_products",
                    "output_id": pd.NA if tentative else product_id,
                }
            )

            ingredient_names = [row.get("name") for row in active]
            strengths = [row.get("strength") for row in active]
            product_sha = _canonical_sha256(product)
            if not tentative:
                status = _text(product.get("marketing_status"))
                products.append(
                    {
                        "release_id": release_id,
                        "product_id": product_id,
                        "regulatory_source": DRUGSFDA_SOURCE,
                        "source_row_id": product_source_id,
                        "source_release": release,
                        "source_record_sha256": product_sha,
                        "application_number": app,
                        "product_number": number,
                        "proprietary_name": _clean(product.get("brand_name")),
                        "nonproprietary_name": _join_exact(ingredient_names),
                        "ingredient_text": _join_exact(ingredient_names),
                        "strength": _join_exact(strengths),
                        "dosage_form": _clean(product.get("dosage_form")),
                        "route": _clean(product.get("route")),
                        "product_presentation": pd.NA,
                        "marketing_status": _clean(product.get("marketing_status")),
                        "licensure_status": pd.NA,
                        "currently_marketed": status in {"Prescription", "Over-the-counter"},
                        "discontinued": status == "Discontinued",
                        "revoked": False,
                        "approval_date": pd.NA,
                        "product_approval_date_availability": (
                            "not_supplied_at_product_grain"
                        ),
                        "earliest_approved_orig_submission_date": earliest_orig,
                        "applicant": _clean(application.get("sponsor_name")),
                        "reference_product": _clean(product.get("reference_drug")),
                        "reference_standard": _clean(product.get("reference_standard")),
                        "resolution_status": (
                            "pending_substance_identity"
                            if active
                            else "no_ingredient_in_source"
                        ),
                        "te_code": _clean(product.get("te_code")),
                    }
                )

            duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
            for ingredient_index, ingredient in enumerate(active, start=1):
                name = _text(ingredient.get("name"))
                strength = _text(ingredient.get("strength"))
                identity_tuple = (exact_name_key(name), exact_name_key(strength))
                duplicate_counts[identity_tuple] += 1
                duplicate_ordinal = duplicate_counts[identity_tuple]
                identity_digest = hashlib.sha256(
                    f"{identity_tuple[0]}\x1f{identity_tuple[1]}".encode("utf-8")
                ).hexdigest()[:16]
                ingredient_source_id = (
                    f"openfda_drugsfda:{release}:ingredient:{app}:{number}:"
                    f"{identity_digest}:{duplicate_ordinal}"
                )
                ingredient_path = f"{product_path}.active_ingredients[{ingredient_index - 1}]"
                if tentative:
                    tentative_ingredients += 1
                else:
                    ingredients.append(
                        {
                            "release_id": release_id,
                            "product_id": product_id,
                            "ingredient_index": ingredient_index,
                            "product_ingredient_id": (
                                f"{product_id}:ingredient:{ingredient_index:03d}"
                            ),
                            "source_row_id": ingredient_source_id,
                            "ingredient_role": "active_ingredient",
                            "ingredient_text": _clean(name),
                            "substance_id": regulatory_substance_id(name),
                            "strength": _clean(strength),
                            "dosage_form": _clean(product.get("dosage_form")),
                            "route": _clean(product.get("route")),
                            "duplicate_ordinal": duplicate_ordinal,
                            "resolution_status": "pending_gsrs_identity",
                        }
                    )
                reconciliation.append(
                    {
                        "release_id": release_id,
                        "regulatory_source": DRUGSFDA_SOURCE,
                        "source_release": release,
                        "source_row_id": ingredient_source_id,
                        "source_section": "active_ingredients",
                        "record_type": "ingredient",
                        "parent_source_row_id": product_source_id,
                        "natural_key": (
                            f"{app}/{number}/{identity_digest}/{duplicate_ordinal}"
                        ),
                        "source_location": ingredient_path,
                        "source_record_sha256": _canonical_sha256(ingredient),
                        "disposition": "excluded" if tentative else "retained",
                        "product_id": product_id,
                        "reason": (
                            "excluded_parent_tentative"
                            if tentative
                            else "included_product_ingredient"
                        ),
                        "output_table": pd.NA if tentative else "product_ingredients",
                        "output_id": (
                            pd.NA
                            if tentative
                            else f"{product_id}:ingredient:{ingredient_index:03d}"
                        ),
                    }
                )

    product_frame = pd.DataFrame(products)
    ingredient_frame = pd.DataFrame(ingredients)
    reconciliation_frame = pd.DataFrame(reconciliation)
    audit = {
        "source_release": release,
        "application_nodes": len(applications),
        "product_nodes": len(seen_product_keys),
        "ingredient_nodes": len(ingredients) + tentative_ingredients,
        "included_products": len(product_frame),
        "included_ingredient_edges": len(ingredient_frame),
        "tentative_products_excluded": tentative_products,
        "tentative_ingredient_nodes_excluded": tentative_ingredients,
        "included_products_without_ingredient": included_without_ingredient,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    _validate_products_and_edges(product_frame, ingredient_frame)
    return ParsedRegulatorySource(
        product_frame, ingredient_frame, reconciliation_frame, audit
    )


def _read_csv_rows(path: Path) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append((reader.line_num, row))
    return rows


def _is_purple_header(row: list[str]) -> bool:
    return len(row) >= 3 and row[:3] == ["N/R/U", "Applicant", "BLA Number"]


def _purple_record(header: list[str], row: list[str]) -> dict[str, str]:
    values = row + [""] * max(0, len(header) - len(row))
    return dict(zip(header, values[: len(header)]))


def load_purple_book(
    path: str | Path,
    *,
    release_id: str,
    source_release: str,
) -> ParsedRegulatorySource:
    """Parse the authoritative full Purple Book table after its last header."""

    path = Path(path)
    rows = _read_csv_rows(path)
    header_indices = [index for index, (_, row) in enumerate(rows) if _is_purple_header(row)]
    if len(header_indices) < 2:
        raise ValueError("Purple Book source must contain change and full-table headers")
    change_header_index = header_indices[0]
    full_header_index = header_indices[-1]
    change_header = rows[change_header_index][1]
    full_header = rows[full_header_index][1]

    change_rows: list[tuple[int, dict[str, str]]] = []
    for line_number, row in rows[change_header_index + 1 : full_header_index]:
        if not any(_text(value) for value in row) or _is_purple_header(row):
            continue
        record = _purple_record(change_header, row)
        if _text(record.get("BLA Number")) and _text(record.get("Product Number")):
            change_rows.append((line_number, record))

    full_rows: list[tuple[int, dict[str, str]]] = []
    for line_number, row in rows[full_header_index + 1 :]:
        if not any(_text(value) for value in row):
            continue
        if _is_purple_header(row):
            raise ValueError(f"unexpected repeated Purple Book header on line {line_number}")
        record = _purple_record(full_header, row)
        if not _text(record.get("BLA Number")) or not _text(record.get("Product Number")):
            raise ValueError(f"Purple Book data row {line_number} has an incomplete key")
        full_rows.append((line_number, record))

    products: list[dict[str, Any]] = []
    ingredients: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    full_keys: set[tuple[str, str]] = set()
    for line_number, record in full_rows:
        bla = _text(record["BLA Number"])
        number = _text(record["Product Number"])
        key = (bla, number)
        if key in full_keys:
            raise ValueError(f"duplicate Purple Book product key {bla}/{number}")
        full_keys.add(key)
        product_id = f"purplebook:{bla}:{number}"
        source_row_id = f"purplebook:{source_release}:full:{bla}:{number}"
        name = _text(record["Proper Name"])
        if not name:
            raise ValueError(f"Purple Book product {bla}/{number} has no Proper Name")
        marketing = _text(record["Marketing Status"])
        licensure = _text(record["Licensure"])
        licensure_upper = licensure.upper()
        source_sha = _canonical_sha256(record)
        products.append(
            {
                "release_id": release_id,
                "product_id": product_id,
                "regulatory_source": PURPLE_SOURCE,
                "source_row_id": source_row_id,
                "source_release": source_release,
                "source_record_sha256": source_sha,
                "application_number": bla,
                "product_number": number,
                "proprietary_name": _clean(record["Proprietary Name"]),
                "nonproprietary_name": _clean(name),
                "ingredient_text": _clean(name),
                "strength": _clean(record["Strength"]),
                "dosage_form": _clean(record["Dosage Form"]),
                "route": _clean(record["Route of Administration"]),
                "product_presentation": _clean(record["Product Presentation"]),
                "marketing_status": _clean(marketing),
                "licensure_status": _clean(licensure),
                "currently_marketed": (
                    licensure == "Licensed" and marketing in {"Rx", "OTC"}
                ),
                "discontinued": marketing in {"Disc", "Disc*"},
                "revoked": "REVOKED" in licensure_upper,
                "approval_date": _clean(record["Approval Date"]),
                "product_approval_date_availability": "supplied_by_source",
                "earliest_approved_orig_submission_date": pd.NA,
                "applicant": _clean(record["Applicant"]),
                "reference_product": _clean(record["Ref. Product Proper Name"]),
                "reference_standard": pd.NA,
                "resolution_status": "pending_substance_identity",
                "change_status": _clean(record["N/R/U"]),
                "license_type": _clean(record["License Type"]),
                "interchangeable_approval_date": _clean(record["Inter. Approval Date"]),
                "reference_product_proprietary_name": _clean(
                    record["Ref. Product Proprietary Name"]
                ),
                "supplement_number": _clean(record["Supplement Number"]),
                "submission_type": _clean(record["Submission Type"]),
                "interchangeable_supplement_number": _clean(
                    record["Inter. Supplement Number"]
                ),
                "license_number": _clean(record["License Number"]),
                "center": _clean(record["Center"]),
                "date_of_first_licensure": _clean(record["Date of First Licensure"]),
                "exclusivity_expiration_date": _clean(
                    record["Exclusivity Expiration Date"]
                ),
                "first_interchangeable_exclusivity_expiration_date": _clean(
                    record["First Interchangeable Exclusivity Exp. Date"]
                ),
                "reference_product_exclusivity_expiration_date": _clean(
                    record["Ref. Product Exclusivity Exp. Date"]
                ),
                "orphan_exclusivity_expiration_date": _clean(
                    record["Orphan Exclusivity Exp. Date"]
                ),
                "patent_list_provided": _clean(record["Patent List Provided"]),
            }
        )
        ingredients.append(
            {
                "release_id": release_id,
                "product_id": product_id,
                "ingredient_index": 1,
                "product_ingredient_id": f"{product_id}:proper_name",
                "source_row_id": source_row_id,
                "ingredient_role": "proper_name",
                "ingredient_text": _clean(name),
                "substance_id": regulatory_substance_id(name),
                "strength": _clean(record["Strength"]),
                "dosage_form": _clean(record["Dosage Form"]),
                "route": _clean(record["Route of Administration"]),
                "duplicate_ordinal": 1,
                "resolution_status": "pending_gsrs_identity",
            }
        )
        reconciliation.append(
            {
                "release_id": release_id,
                "regulatory_source": PURPLE_SOURCE,
                "source_release": source_release,
                "source_row_id": source_row_id,
                "source_section": "full_table",
                "record_type": "product",
                "parent_source_row_id": pd.NA,
                "natural_key": f"{bla}/{number}",
                "source_location": f"physical_line:{line_number}",
                "source_record_sha256": source_sha,
                "disposition": "retained",
                "product_id": product_id,
                "reason": "included_full_product",
                "output_table": "regulatory_products;product_ingredients",
                "output_id": product_id,
            }
        )

    for line_number, record in change_rows:
        bla = _text(record["BLA Number"])
        number = _text(record["Product Number"])
        if (bla, number) not in full_keys:
            raise ValueError(
                f"Purple Book change row {line_number} has no full-table product"
            )
        product_id = f"purplebook:{bla}:{number}"
        reconciliation.append(
            {
                "release_id": release_id,
                "regulatory_source": PURPLE_SOURCE,
                "source_release": source_release,
                "source_row_id": (
                    f"purplebook:{source_release}:change:line:{line_number}"
                ),
                "source_section": "monthly_change_summary",
                "record_type": "change_summary_product",
                "parent_source_row_id": pd.NA,
                "natural_key": f"{bla}/{number}",
                "source_location": f"physical_line:{line_number}",
                "source_record_sha256": _canonical_sha256(record),
                "disposition": "excluded",
                "product_id": product_id,
                "reason": "monthly_change_summary_duplicate",
                "output_table": "regulatory_products",
                "output_id": product_id,
            }
        )

    product_frame = pd.DataFrame(products)
    ingredient_frame = pd.DataFrame(ingredients)
    reconciliation_frame = pd.DataFrame(reconciliation)
    audit = {
        "source_release": source_release,
        "physical_csv_records": len(rows),
        "header_count": len(header_indices),
        "change_header_line": rows[change_header_index][0],
        "full_header_line": rows[full_header_index][0],
        "change_summary_rows": len(change_rows),
        "full_product_rows": len(full_rows),
        "revoked_products": int(product_frame["revoked"].sum()),
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    _validate_products_and_edges(product_frame, ingredient_frame)
    return ParsedRegulatorySource(
        product_frame, ingredient_frame, reconciliation_frame, audit
    )


def _validate_products_and_edges(
    products: pd.DataFrame, ingredients: pd.DataFrame
) -> None:
    if products["product_id"].duplicated().any():
        raise ValueError("regulatory product IDs are not unique")
    if ingredients["product_ingredient_id"].duplicated().any():
        raise ValueError("product ingredient IDs are not unique")
    orphan = set(ingredients["product_id"]) - set(products["product_id"])
    if orphan:
        raise ValueError(f"product ingredient orphans found: {sorted(orphan)[:3]}")
