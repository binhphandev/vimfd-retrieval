"""Read-only audit for processed product splits, text fields, and image files.

Run from the repository root:
    python data/audit_processed_data.py --output data/reports/audit.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = ("id", "shop", "category", "images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_TEXT_FIELD = "retrieval_text"
TEXT_PREPROCESSING_VERSION = 1
FORMAT_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
    "TIFF": {".tif", ".tiff"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit for processed split JSON files and image files."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Thư mục chứa train.json, val.json, test.json.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Thư mục ảnh; mặc định là <processed-dir>/images.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Chỉ kiểm tra đường dẫn tồn tại, không giải mã nội dung ảnh.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=25,
        help="Số ví dụ lỗi tối đa lưu cho mỗi loại (mặc định: 25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Tùy chọn lưu JSON report; mặc định chỉ in ra màn hình.",
    )
    return parser.parse_args()


def _resolve_image_path(local_path: str, image_dir: Path) -> Path:
    path = Path(local_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].casefold() == "images":
        path = Path(*path.parts[1:])
    return image_dir / path


def _add_issue(
    report: dict[str, Any],
    kind: str,
    severity: str,
    *,
    split: str | None = None,
    product_id: str | None = None,
    path: str | None = None,
    detail: str | None = None,
) -> None:
    report["issue_counts"][kind] += 1
    if severity == "error":
        report["error_count"] += 1
    samples = report["issues"].setdefault(kind, [])
    if len(samples) < report["max_issues"]:
        issue: dict[str, str] = {"severity": severity}
        if split is not None:
            issue["split"] = split
        if product_id is not None:
            issue["product_id"] = product_id
        if path is not None:
            issue["path"] = path
        if detail is not None:
            issue["detail"] = detail
        samples.append(issue)


def audit(processed_dir: Path, image_dir: Path, decode_images: bool, max_issues: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "processed_dir": str(processed_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "decode_images": decode_images,
        "text_processing": {
            "model_text_field": MODEL_TEXT_FIELD,
            "expected_version": TEXT_PREPROCESSING_VERSION,
        },
        "splits": {},
        "checks": {},
        "issue_counts": Counter(),
        "issues": {},
        "error_count": 0,
        "max_issues": max_issues,
    }
    id_locations: dict[str, list[str]] = {}
    referenced_paths: set[str] = set()
    referenced_image_count = 0
    readable_images = 0
    image_formats: Counter[str] = Counter()
    extension_mismatches = 0

    for split in SPLITS:
        split_path = processed_dir / f"{split}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy split: {split_path}")
        with split_path.open(encoding="utf-8") as file:
            products = json.load(file)
        if not isinstance(products, list):
            raise ValueError(f"{split_path} phải chứa một JSON array.")

        split_ids: set[str] = set()
        split_summary: dict[str, Any] = {
            "products": len(products),
            "duplicate_ids": 0,
            "missing_fields": Counter(),
            "missing_description": 0,
            "missing_tags": 0,
            "empty_retrieval_text": 0,
            "text_preprocessing_versions": Counter(),
            "retrieval_text_word_count_min": None,
            "retrieval_text_word_count_max": 0,
            "categories": Counter(),
            "shops": Counter(),
            "image_references": 0,
            "unique_image_paths": 0,
            "missing_images": 0,
            "unreadable_images": 0,
        }
        if not products:
            _add_issue(
                report, "empty_split", "error",
                split=split, detail="Split contains no products",
            )
        split_paths: set[str] = set()

        for row_number, product in enumerate(products, start=1):
            if not isinstance(product, dict):
                _add_issue(
                    report, "invalid_product_record", "error",
                    split=split, detail=f"JSON row {row_number} is not an object",
                )
                continue

            product_id_value = product.get("id")
            product_id = str(product_id_value) if product_id_value is not None else ""

            for field in REQUIRED_FIELDS:
                value = product.get(field)
                if field == "images":
                    missing = not isinstance(value, list) or not value
                else:
                    missing = value is None or value == ""
                if missing:
                    split_summary["missing_fields"][field] += 1
                    _add_issue(
                        report, f"missing_{field}", "error",
                        split=split, product_id=product_id or None,
                        detail=f"JSON row {row_number}",
                    )

            if not product.get("description"):
                split_summary["missing_description"] += 1
            if not product.get("tags"):
                split_summary["missing_tags"] += 1
            model_text = product.get(MODEL_TEXT_FIELD)
            if not isinstance(model_text, str) or not model_text.strip():
                split_summary["empty_retrieval_text"] += 1
                _add_issue(
                    report, "empty_retrieval_text", "error",
                    split=split, product_id=product_id or None,
                    detail=f"JSON row {row_number}",
                )
            else:
                word_count = len(model_text.split())
                current_min = split_summary["retrieval_text_word_count_min"]
                split_summary["retrieval_text_word_count_min"] = (
                    word_count if current_min is None else min(current_min, word_count)
                )
                split_summary["retrieval_text_word_count_max"] = max(
                    split_summary["retrieval_text_word_count_max"], word_count
                )
            text_version = product.get("text_preprocessing_version")
            split_summary["text_preprocessing_versions"][str(text_version)] += 1
            if text_version != TEXT_PREPROCESSING_VERSION:
                _add_issue(
                    report, "unexpected_text_preprocessing_version", "warning",
                    split=split, product_id=product_id or None,
                    detail=f"version={text_version!r}",
                )
            if product.get("category") is not None:
                split_summary["categories"][str(product["category"])] += 1
            if product.get("shop") is not None:
                split_summary["shops"][str(product["shop"])] += 1

            if product_id:
                if product_id in split_ids:
                    split_summary["duplicate_ids"] += 1
                    _add_issue(
                        report, "duplicate_id_within_split", "error",
                        split=split, product_id=product_id,
                    )
                split_ids.add(product_id)
                id_locations.setdefault(product_id, []).append(split)

            images = product.get("images")
            if not isinstance(images, list):
                continue
            for image_meta in images:
                split_summary["image_references"] += 1
                referenced_image_count += 1
                if referenced_image_count % 5000 == 0:
                    print(
                        f"Checked {referenced_image_count} image references...",
                        file=sys.stderr,
                    )
                if not isinstance(image_meta, dict):
                    _add_issue(
                        report, "invalid_image_record", "error",
                        split=split, product_id=product_id or None,
                    )
                    continue
                local_path = image_meta.get("local_path")
                if not isinstance(local_path, str) or not local_path.strip():
                    _add_issue(
                        report, "missing_local_path", "error",
                        split=split, product_id=product_id or None,
                    )
                    continue

                image_path = _resolve_image_path(local_path, image_dir)
                normalized_path = str(image_path.resolve()).casefold()
                split_paths.add(normalized_path)
                referenced_paths.add(normalized_path)
                if not image_path.is_file():
                    split_summary["missing_images"] += 1
                    _add_issue(
                        report, "missing_image_file", "error",
                        split=split, product_id=product_id or None,
                        path=str(image_path),
                    )
                    continue

                if decode_images:
                    try:
                        with Image.open(image_path) as image:
                            image_format = image.format or "UNKNOWN"
                            extension = image_path.suffix.casefold()
                            image_formats[image_format] += 1
                            if extension not in FORMAT_EXTENSIONS.get(image_format, set()):
                                extension_mismatches += 1
                                _add_issue(
                                    report, "extension_content_mismatch", "warning",
                                    split=split, product_id=product_id or None,
                                    path=str(image_path),
                                    detail=f"extension={extension}, format={image_format}",
                                )
                            image.load()
                        readable_images += 1
                    except (OSError, ValueError) as error:
                        split_summary["unreadable_images"] += 1
                        _add_issue(
                            report, "unreadable_image", "error",
                            split=split, product_id=product_id or None,
                            path=str(image_path), detail=str(error),
                        )

        split_summary["unique_image_paths"] = len(split_paths)
        split_summary["missing_fields"] = dict(split_summary["missing_fields"])
        split_summary["text_preprocessing_versions"] = dict(
            split_summary["text_preprocessing_versions"]
        )
        split_summary["categories"] = dict(split_summary["categories"])
        split_summary["shops"] = dict(split_summary["shops"])
        report["splits"][split] = split_summary

    overlaps = {
        product_id: sorted(set(splits))
        for product_id, splits in id_locations.items()
        if len(set(splits)) > 1
    }
    for product_id, splits in overlaps.items():
        _add_issue(
            report, "id_across_splits", "error",
            product_id=product_id, detail=", ".join(splits),
        )

    image_files = {
        str(path.resolve()).casefold()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    } if image_dir.is_dir() else set()
    report["checks"] = {
        "split_id_overlap_count": len(overlaps),
        "split_id_overlap_samples": list(overlaps.items())[:max_issues],
        "referenced_image_count": referenced_image_count,
        "unique_referenced_image_count": len(referenced_paths),
        "duplicate_image_reference_count": referenced_image_count - len(referenced_paths),
        "image_files_on_disk": len(image_files),
        "unreferenced_image_file_count": len(image_files - referenced_paths),
        "readable_images": readable_images if decode_images else None,
        "image_formats": dict(image_formats) if decode_images else None,
        "extension_content_mismatches": extension_mismatches if decode_images else None,
    }
    report["issue_counts"] = dict(report["issue_counts"])
    report["status"] = "FAIL" if report["error_count"] else "PASS"
    report.pop("max_issues")
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    image_dir = (args.image_dir or processed_dir / "images").resolve()

    try:
        report = audit(
            processed_dir=processed_dir,
            image_dir=image_dir,
            decode_images=not args.skip_decode,
            max_issues=max(0, args.max_issues),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"AUDIT ERROR: {error}", file=sys.stderr)
        return 2

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Đã lưu báo cáo: {args.output.resolve()}", file=sys.stderr)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
