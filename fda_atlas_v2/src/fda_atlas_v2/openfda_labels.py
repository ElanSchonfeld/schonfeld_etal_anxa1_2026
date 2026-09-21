"""Streaming utilities for the pinned openFDA drug-label bulk export."""

from __future__ import annotations

import json
import re
import unicodedata
import codecs
from collections import defaultdict
from typing import BinaryIO, Iterator

import pandas as pd


def normalized_name(value: object) -> str:
    """Return an ordered, punctuation-insensitive key that retains all words."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").upper()
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", text).split())


def iter_json_results(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Yield objects from a top-level openFDA ``results`` array incrementally."""

    json_decoder = json.JSONDecoder()
    text_decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    started = False
    exhausted = False
    while not exhausted:
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += text_decoder.decode(chunk, final=False) if isinstance(chunk, bytes) else chunk
        else:
            if hasattr(stream, "read"):
                buffer += text_decoder.decode(b"", final=True)
            exhausted = True
        if not started:
            match = re.search(r'"results"\s*:\s*\[', buffer)
            if match is None:
                if exhausted:
                    raise ValueError("JSON document has no results array")
                buffer = buffer[-64:]
                continue
            buffer = buffer[match.end():]
            started = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer:
                break
            try:
                value, end = json_decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if exhausted:
                    raise ValueError("truncated object in results array")
                break
            if not isinstance(value, dict):
                raise ValueError("results array contains a non-object value")
            yield value
            buffer = buffer[end:]
    raise ValueError("unterminated results array")


def _values(record: dict, field: str) -> list[str]:
    value = record.get("openfda", {}).get(field, [])
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def build_match_indices(
    regulatory_products: pd.DataFrame,
    product_ingredients: pd.DataFrame,
    regulatory_substances: pd.DataFrame,
    active_moieties: pd.DataFrame,
    substance_moiety_edges: pd.DataFrame,
) -> dict:
    product_to_substances = (
        product_ingredients.groupby("product_id")["substance_id"]
        .agg(lambda values: sorted(set(values.dropna().astype(str))))
        .to_dict()
    )
    application_to_products = defaultdict(set)
    name_to_products = defaultdict(set)
    for row in regulatory_products.itertuples(index=False):
        product_id = str(row.product_id)
        application = "" if pd.isna(row.application_number) else str(row.application_number).strip().upper()
        if application:
            application_to_products[application].add(product_id)
        for value in (row.proprietary_name, row.nonproprietary_name, row.ingredient_text):
            key = normalized_name(value)
            if key:
                name_to_products[key].add(product_id)

    name_to_substances = defaultdict(set)
    for row in regulatory_substances.itertuples(index=False):
        variants = "" if pd.isna(row.name_variants) else str(row.name_variants)
        for value in [row.preferred_name, *variants.split(" | ")]:
            key = normalized_name(value)
            if key:
                name_to_substances[key].add(str(row.substance_id))

    moiety_to_substances = (
        substance_moiety_edges.groupby("moiety_id")["substance_id"]
        .agg(lambda values: sorted(set(values.dropna().astype(str))))
        .to_dict()
    )
    unii_to_moieties = defaultdict(set)
    for row in active_moieties.itertuples(index=False):
        unii = "" if pd.isna(row.unii) else str(row.unii).strip().upper()
        if unii:
            unii_to_moieties[unii].add(str(row.moiety_id))
    return {
        "product_to_substances": product_to_substances,
        "application_to_products": {k: sorted(v) for k, v in application_to_products.items()},
        "name_to_products": {k: sorted(v) for k, v in name_to_products.items()},
        "name_to_substances": {k: sorted(v) for k, v in name_to_substances.items()},
        "moiety_to_substances": moiety_to_substances,
        "unii_to_moieties": {k: sorted(v) for k, v in unii_to_moieties.items()},
    }


def summarize_match_indices(indices: dict) -> dict[str, int]:
    """Report admitted and withheld identity-key multiplicities."""

    substance_names = indices["name_to_substances"]
    product_names = indices["name_to_products"]
    unii = indices["unii_to_moieties"]
    return {
        "application_number_keys": len(indices["application_to_products"]),
        "product_name_keys": len(product_names),
        "unique_product_name_keys": sum(len(ids) == 1 for ids in product_names.values()),
        "ambiguous_product_name_keys_withheld": sum(
            len(ids) > 1 for ids in product_names.values()
        ),
        "substance_name_keys": len(substance_names),
        "unique_substance_name_keys": sum(
            len(ids) == 1 for ids in substance_names.values()
        ),
        "ambiguous_substance_name_keys_withheld": sum(
            len(ids) > 1 for ids in substance_names.values()
        ),
        "active_moiety_unii_keys": len(unii),
        "ambiguous_active_moiety_unii_keys": sum(len(ids) > 1 for ids in unii.values()),
    }


def match_label_record(record: dict, indices: dict) -> dict:
    evidence: set[tuple[str, str, str, str, str]] = set()
    methods = set()

    def add(entity_type, entity_id, method, source_value, source_entity_id=""):
        evidence.add(
            (
                str(entity_type), str(entity_id), str(method),
                str(source_value), str(source_entity_id),
            )
        )

    for application in _values(record, "application_number"):
        matched = indices["application_to_products"].get(application.upper(), [])
        if matched:
            methods.add("exact_application_number")
            for product_id in matched:
                add(
                    "regulatory_product", product_id, "exact_application_number",
                    application.upper(),
                )
    for unii in _values(record, "unii"):
        matched = indices["unii_to_moieties"].get(unii.upper(), [])
        if matched:
            methods.add("exact_active_moiety_unii")
            for moiety_id in matched:
                add(
                    "active_moiety", moiety_id, "exact_active_moiety_unii",
                    unii.upper(),
                )
    for field in ("substance_name", "generic_name", "brand_name"):
        for value in _values(record, field):
            key = normalized_name(value)
            substance_matches = indices["name_to_substances"].get(key, [])
            product_matches = indices["name_to_products"].get(key, [])
            if len(substance_matches) == 1:
                method = f"unique_punctuation_insensitive_{field}_to_substance"
                methods.add(method)
                add("regulatory_substance", substance_matches[0], method, value)
            if len(product_matches) == 1:
                method = f"unique_punctuation_insensitive_{field}_to_product"
                methods.add(method)
                add("regulatory_product", product_matches[0], method, value)

    direct = sorted(evidence)
    for entity_type, entity_id, method, source_value, _ in direct:
        if entity_type == "regulatory_product":
            for substance_id in indices["product_to_substances"].get(entity_id, []):
                add(
                    "regulatory_substance", substance_id,
                    "propagated_from_product_match", source_value, entity_id,
                )
        elif entity_type == "active_moiety":
            for substance_id in indices["moiety_to_substances"].get(entity_id, []):
                add(
                    "regulatory_substance", substance_id,
                    "propagated_from_active_moiety_match", source_value, entity_id,
                )

    products = sorted({row[1] for row in evidence if row[0] == "regulatory_product"})
    substances = sorted({row[1] for row in evidence if row[0] == "regulatory_substance"})
    moieties = sorted({row[1] for row in evidence if row[0] == "active_moiety"})
    return {
        "matched": bool(evidence),
        "product_ids": products,
        "substance_ids": substances,
        "moiety_ids": moieties,
        "match_methods": sorted(methods),
        "match_evidence": [
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "match_method": method,
                "source_value": source_value,
                "propagation_source_entity_id": source_entity_id,
            }
            for entity_type, entity_id, method, source_value, source_entity_id
            in sorted(evidence)
        ],
    }


def validate_download_metadata(payload: dict, expected_export_date: str) -> dict:
    try:
        label = payload["results"]["drug"]["label"]
    except (KeyError, TypeError) as exc:
        raise ValueError("download metadata lacks results.drug.label") from exc
    export_date = str(label.get("export_date", ""))
    if export_date != expected_export_date:
        raise ValueError(
            f"openFDA export date {export_date} does not match {expected_export_date}"
        )
    partitions = label.get("partitions", [])
    if not partitions:
        raise ValueError("openFDA label export has no partitions")
    for index, partition in enumerate(partitions, start=1):
        url = str(partition.get("file", ""))
        if not url.startswith("https://download.open.fda.gov/drug/label/"):
            raise ValueError(f"unexpected openFDA partition URL: {url}")
        if int(partition.get("records", 0)) <= 0 or float(partition.get("size_mb", 0)) <= 0:
            raise ValueError(f"invalid openFDA partition {index}")
    total_records = sum(int(partition["records"]) for partition in partitions)
    if total_records != int(label["total_records"]):
        raise ValueError("openFDA partition record counts do not reconcile")
    return {
        "export_date": export_date,
        "total_records": total_records,
        "n_partitions": len(partitions),
        "total_size_mb": sum(float(partition["size_mb"]) for partition in partitions),
        "partitions": partitions,
    }
