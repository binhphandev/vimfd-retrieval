"""Validate a generated query-review workbook against the test split."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from data.annotation.build_query_list_v2 import (
    CATEGORY_LABELS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    QUOTAS,
    SIZE_TYPES,
    _clean_full_description,
    _extract_colors,
    _extract_sizes,
    _extract_terms,
    _match_product_type,
    _material_is_compatible,
    _normalize,
    _query_text,
)
from data.annotation.build_query_list_v2 import MATERIAL_TERMS


def _validate(
    products: list[dict[str, Any]],
    workbook_path: Path,
) -> dict[str, Any]:
    products_by_id = {
        str(product.get("id")): product
        for product in products
        if product.get("id")
    }
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if "Query List" not in workbook.sheetnames:
        raise ValueError("Workbook is missing the 'Query List' sheet.")
    sheet = workbook["Query List"]
    header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    required_headers = (
        "STT", "Nhóm", "Truy vấn", "Ý định", "Số thuộc tính", "Nguồn",
        "ID mẫu", "Tên sản phẩm (đã làm sạch)", "Ảnh đại diện (URL)",
        "Màu", "Màu nguồn", "Cỡ", "Cỡ nguồn", "Từ khóa mô tả",
        "Đoạn trích mô tả",
    )
    missing_headers = [name for name in required_headers if name not in header]
    if missing_headers:
        raise ValueError(f"Workbook is missing columns: {missing_headers}")
    column = {name: header.index(name) for name in required_headers}

    errors = []
    rows = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if not values or values[column["STT"]] is None:
            continue
        row = {name: values[index] for name, index in column.items()}
        rows.append(row)

    if len(rows) != sum(sum(levels.values()) for levels in QUOTAS.values()):
        errors.append(f"Expected 200 rows, got {len(rows)}.")
    normalized_queries = [_normalize(str(row["Truy vấn"] or "")) for row in rows]
    if len(set(normalized_queries)) != len(normalized_queries):
        errors.append("Normalized query text contains duplicates.")

    group_counts = Counter(row["Nhóm"] for row in rows)
    attribute_counts = Counter(row["Số thuộc tính"] for row in rows)
    group_attribute_counts = Counter(
        (row["Nhóm"], row["Số thuộc tính"]) for row in rows
    )
    source_counts = Counter(str(row["Nguồn"] or "") for row in rows)
    expected_group_counts = Counter({
        CATEGORY_LABELS[category]: sum(levels.values())
        for category, levels in QUOTAS.items()
    })
    expected_attribute_counts = Counter({
        level: sum(levels.get(level, 0) for levels in QUOTAS.values())
        for level in (1, 2, 3)
    })
    expected_group_attribute_counts = Counter({
        (CATEGORY_LABELS[category], level): count
        for category, levels in QUOTAS.items()
        for level, count in levels.items()
    })
    if group_counts != expected_group_counts:
        errors.append(f"Category quota mismatch: {dict(group_counts)}.")
    if attribute_counts != expected_attribute_counts:
        errors.append(f"Attribute quota mismatch: {dict(attribute_counts)}.")
    if group_attribute_counts != expected_group_attribute_counts:
        errors.append(
            f"Category/attribute quota mismatch: {dict(group_attribute_counts)}."
        )

    type_color_counts = Counter()
    bundles = Counter()
    for number, row in enumerate(rows, start=2):
        seed_id = str(row["ID mẫu"] or "")
        product = products_by_id.get(seed_id)
        if product is None:
            errors.append(f"Row {number}: seed ID is not in test.json: {seed_id}.")
            continue

        product_type = str(product.get("type") or "")
        category = str(product.get("category") or "")
        if row["Nhóm"] != CATEGORY_LABELS.get(category):
            errors.append(f"Row {number}: category does not match seed product.")
        expected_source = seed_id.split("_", 1)[0].casefold()
        if row["Nguồn"] != expected_source:
            errors.append(f"Row {number}: source does not match seed ID.")
        if not _match_product_type(product_type, str(product.get("name") or "")):
            errors.append(f"Row {number}: product type is not evidenced by name.")

        raw_colors = product.get("colors")
        if not isinstance(raw_colors, list) or len(raw_colors) != 1:
            errors.append(
                f"Row {number}: seed product must have exactly one colors entry."
            )

        raw_color = str(row["Màu nguồn"] or "")
        color_pairs = dict((raw, normalized) for normalized, raw in _extract_colors(product))
        if raw_color not in color_pairs:
            errors.append(f"Row {number}: source color is not a valid single color.")
        elif row["Màu"] != color_pairs[raw_color]:
            errors.append(f"Row {number}: normalized color does not match source.")

        size = str(row["Cỡ"] or "")
        raw_size = str(row["Cỡ nguồn"] or "")
        if size:
            if product_type not in SIZE_TYPES:
                errors.append(f"Row {number}: size used for a non-size product type.")
            if (size, raw_size) not in _extract_sizes(product):
                errors.append(f"Row {number}: size is not evidenced by source metadata.")

        description = _clean_full_description(product.get("description"))
        matched_materials = [
            item for item in _extract_terms(description, MATERIAL_TERMS)
            if _material_is_compatible(
                product_type,
                item["term"],
                str(product.get("name") or ""),
                description,
            )
        ]
        term_evidence = {
            item["term"]: item["alias"]
            for item in matched_materials
        }
        terms = [
            term.strip()
            for term in str(row["Từ khóa mô tả"] or "").split(" + ")
            if term.strip()
        ]
        excerpt = str(row["Đoạn trích mô tả"] or "")
        if any(term not in term_evidence for term in terms):
            errors.append(
                f"Row {number}: query material is not supported by description."
            )
        if any(
            _normalize(term_evidence[term]) not in _normalize(excerpt)
            for term in terms
            if term in term_evidence
        ):
            errors.append(f"Row {number}: excerpt does not contain every query term.")

        expected_attribute_count = 1 + bool(size) + len(terms)
        if row["Số thuộc tính"] != expected_attribute_count:
            errors.append(f"Row {number}: attribute count does not match query fields.")
        if not size and row["Số thuộc tính"] == 3:
            if len(terms) != 2 or len(set(terms)) != 2:
                errors.append(
                    f"Row {number}: a size-less three-attribute query needs "
                    "two distinct materials."
                )
        if size and row["Số thuộc tính"] == 3 and len(terms) != 1:
            errors.append(
                f"Row {number}: a sized three-attribute query needs one material."
            )
        query = _query_text(
            str(_match_product_type(product_type, str(product.get("name") or "")) or ""),
            str(row["Màu"] or ""),
            size,
            terms,
        )
        if _normalize(query) != _normalize(str(row["Truy vấn"] or "")):
            errors.append(f"Row {number}: query text does not match evidence fields.")

        type_color = (product_type, str(row["Màu"] or ""))
        type_color_counts[type_color] += 1
        bundle = (
            product_type,
            str(row["Màu"] or ""),
            size,
            tuple(sorted(terms)),
        )
        bundles[bundle] += 1

        image_url = str(row["Ảnh đại diện (URL)"] or "")
        if not image_url.startswith(("http://", "https://")):
            errors.append(f"Row {number}: representative image URL is missing or invalid.")

    if any(count > 3 for count in type_color_counts.values()):
        errors.append("A product-type/color pair occurs more than three times.")
    if any(count > 1 for count in bundles.values()):
        errors.append(
            "A product-type/color/size/material variant is duplicated."
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "row_count": len(rows),
        "group_counts": dict(group_counts),
        "attribute_counts": dict(attribute_counts),
        "group_attribute_counts": {
            f"{group}:{level}": count
            for (group, level), count in group_attribute_counts.items()
        },
        "source_counts": dict(source_counts),
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a query-review workbook against test.json."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        products = json.load(file)
    if not isinstance(products, list):
        raise ValueError(f"{args.input} must contain a JSON array.")
    report = _validate(products, args.workbook)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
