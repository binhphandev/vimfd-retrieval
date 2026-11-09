"""
Pipeline:
    1. Load & validate metadata_clean.json
    2. Làm sạch name/description/tags, tạo retrieval_text và word-segment
    3. Chia train/val/test theo stratify (shop x category) — tỉ lệ 80/10/10
    4. Resize tối đa 4 ảnh mỗi sản phẩm về 224x224, lưu ra processed/images/
    5. Lưu splits ra processed/train.json, val.json, test.json

Cách dùng:
    python data/data_preprocess.py
    python data/data_preprocess.py --data-dir /path/to/data --workers 8

Cài đặt:
    pip install underthesea pillow scikit-learn tqdm
"""
import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    IMAGE_LIST_FIELD,
    IMAGE_PATH_FIELD,
    IMAGE_SIZE,
    MODEL_TEXT_FIELD,
    PRODUCT_ID_FIELD,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
)

from src.text_processing.text_utils import (
    build_text,
    clean_description,
    clean_name,
    clean_tags,
    segment_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MIN_STRATUM_SIZE = 20  # nhóm nhỏ hơn ngưỡng này sẽ được gom theo category
# Increment when the cleaning/segmentation rules change.
TEXT_PREPROCESSING_VERSION = 1
MAX_IMAGES_PER_PRODUCT = 4

def process_product_text(product: dict) -> dict:
    p = product.copy()
    if (
        p.get("text_preprocessing_version") == TEXT_PREPROCESSING_VERSION
        and isinstance(p.get(MODEL_TEXT_FIELD), str)
        and p[MODEL_TEXT_FIELD].strip()
    ):
        return p

    name = product.get("name")
    description = product.get("description")
    p[MODEL_TEXT_FIELD] = segment_text(build_text(product))
    p["name"] = clean_name(name if isinstance(name, str) else "")
    p["description"] = clean_description(
        description if isinstance(description, str) else ""
    )
    p["tags"] = clean_tags(product.get("tags"))
    p["text_preprocessing_version"] = TEXT_PREPROCESSING_VERSION
    return p


# --- Bước 1: Load & validate ---

def load_and_validate(metadata_path: Path) -> list[dict]:
    log.info("Loading %s ...", metadata_path)
    with metadata_path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{metadata_path} phải chứa một JSON array.")

    total = len(data)
    valid = []
    skipped = {
        "no_product_id": 0,
        "no_category": 0,
        "no_shop": 0,
        "no_images": 0,
    }
    truncated_image_count = 0

    for row_number, p in enumerate(data, start=1):
        if not isinstance(p, dict):
            raise ValueError(f"{metadata_path}, dòng {row_number} không phải object.")
        if not p.get(PRODUCT_ID_FIELD):
            skipped["no_product_id"] += 1
            continue
        if not p.get("category"):
            skipped["no_category"] += 1
            continue
        if not p.get("shop"):
            skipped["no_shop"] += 1
            continue
        images = p.get(IMAGE_LIST_FIELD)
        if not isinstance(images, list) or not images:
            skipped["no_images"] += 1
            continue
        if len(images) > MAX_IMAGES_PER_PRODUCT:
            truncated_image_count += len(images) - MAX_IMAGES_PER_PRODUCT
            p = p.copy()
            p[IMAGE_LIST_FIELD] = images[:MAX_IMAGES_PER_PRODUCT]
        valid.append(p)

    log.info(
        "  Total: %s | Valid: %s | Skipped: %s",
        total,
        len(valid),
        skipped,
    )
    if truncated_image_count:
        log.info(
            "  Truncated %s image references beyond the %s-image per-product limit.",
            truncated_image_count,
            MAX_IMAGES_PER_PRODUCT,
        )

    if skipped["no_product_id"] > 0:
        log.warning(
            f"  {skipped['no_product_id']} sản phẩm thiếu field '{PRODUCT_ID_FIELD}'."
        )

    return valid


# --- Bước 2: Word-segment text ---

def segment_all(data: list[dict]) -> list[dict]:
    log.info(
        "Cleaning and word-segmenting %s products (model text: %s) ...",
        len(data),
        MODEL_TEXT_FIELD,
    )
    return [process_product_text(p) for p in tqdm(data, desc="Process text")]


# --- Bước 3: Chia split ---

def build_strata(data: list[dict]) -> list[str]:
    # Tầng 1: nhóm nhỏ gom về other_<category>
    strata = [f"{p['shop']}_{p['category']}" for p in data]
    counts = Counter(strata)
    strata = [
        s if counts[s] >= MIN_STRATUM_SIZE else f"other_{p['category']}"
        for s, p in zip(strata, data)
    ]

    # Tầng 2: other_<category> vẫn nhỏ thì gộp vào nhóm lớn nhất cùng category
    counts = Counter(strata)
    for category in {p["category"] for p in data}:
        small = f"other_{category}"
        if counts.get(small, 0) == 0 or counts[small] >= MIN_STRATUM_SIZE:
            continue
        candidates = [
            stratum
            for stratum in counts
            if stratum != small
            and stratum.endswith(f"_{category}")
            and counts[stratum] >= MIN_STRATUM_SIZE
        ]
        target = max(candidates, key=lambda s: counts[s])
        log.info(
            "Gộp %s sản phẩm của %s vào %s",
            counts[small],
            small,
            target,
        )
        strata = [target if s == small else s for s in strata]
        counts = Counter(strata)

    final_counts = Counter(strata)
    log.info("Strata sau khi gom: %s nhóm", len(final_counts))
    for name, n in sorted(final_counts.items(), key=lambda x: x[1])[:5]:
        log.info("  nhóm nhỏ nhất: %s = %s", name, n)
    return strata


def split_data(data: list[dict]) -> dict[str, list[dict]]:
    strata = build_strata(data)

    train, temp, _, strata_temp = train_test_split(
        data,
        strata,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=strata,
    )
    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=strata_temp,
    )

    splits = {"train": train, "val": val, "test": test}
    log.info("Split result:")
    for name, subset in splits.items():
        log.info(
            "  %s: %s products (%.1f%%)",
            name,
            len(subset),
            len(subset) / len(data) * 100,
        )

    return splits


# --- Bước 4: Lưu split ---

def save_splits(splits: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        out_path = out_dir / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(subset, file, ensure_ascii=False, indent=2)
            file.write("\n")
        log.info("Saved %s (%s records)", out_path, len(subset))


# --- Bước 5: Resize ảnh ---

def resize_one(args: tuple[Path, Path]) -> None:
    src_path, dest_path = args
    if not src_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh nguồn: {src_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f"{dest_path.stem}.tmp.jpg")
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
            img.save(temp_path, "JPEG", quality=95)
        temp_path.replace(dest_path)
    except OSError as error:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Không xử lý được ảnh {src_path}: {error}") from error


def _image_relative_path(local_path: str) -> Path:
    relative_path = Path(local_path.replace("\\", "/"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Đường dẫn ảnh phải nằm trong thư mục images/: {local_path}")
    if relative_path.parts and relative_path.parts[0].casefold() == "images":
        relative_path = Path(*relative_path.parts[1:])
    if not relative_path.parts:
        raise ValueError(f"Đường dẫn ảnh không hợp lệ: {local_path}")
    return relative_path


def resize_images(
    data: list[dict],
    raw_images_dir: Path,
    out_images_dir: Path,
    num_workers: int = 4,
) -> None:
    sources: dict[str, tuple[Path, Path]] = {}
    output_sources: dict[str, str] = {}
    image_updates: list[tuple[dict, str]] = []
    for product in data:
        images = product.get(IMAGE_LIST_FIELD, [])
        if not isinstance(images, list):
            raise ValueError(
                f"Sản phẩm {product.get(PRODUCT_ID_FIELD)!r} có images không phải list."
            )
        for image in images:
            if not isinstance(image, dict):
                raise ValueError(
                    f"Sản phẩm {product.get(PRODUCT_ID_FIELD)!r} có metadata ảnh không hợp lệ."
                )
            local_path = image.get(IMAGE_PATH_FIELD)
            if not isinstance(local_path, str) or not local_path.strip():
                raise ValueError(
                    f"Sản phẩm {product.get(PRODUCT_ID_FIELD)!r} có ảnh thiếu "
                    f"{IMAGE_PATH_FIELD}."
                )

            relative_path = _image_relative_path(local_path)
            source_key = relative_path.as_posix().casefold()
            output_relative_path = relative_path.with_suffix(".jpg")
            output_key = output_relative_path.as_posix().casefold()
            prior_source = output_sources.get(output_key)
            if prior_source is not None and prior_source != source_key:
                raise ValueError(
                    f"Nhiều ảnh nguồn cùng ánh xạ tới {output_relative_path}: "
                    f"{prior_source}, {source_key}"
                )
            output_sources[output_key] = source_key
            image_updates.append(
                (image, f"images/{output_relative_path.as_posix()}")
            )
            sources[source_key] = (
                raw_images_dir / relative_path,
                out_images_dir / output_relative_path,
            )

    tasks = list(sources.values())
    log.info("Resizing %s images to %sx%s ...", len(tasks), IMAGE_SIZE, IMAGE_SIZE)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        resized_images = executor.map(resize_one, tasks)
        for _ in tqdm(resized_images, total=len(tasks), desc="Resize"):
            pass

    for image, output_path in image_updates:
        image[IMAGE_PATH_FIELD] = output_path

    log.info("Done — resized %s images.", len(tasks))


# --- Main ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViMFD data preprocessing pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    metadata_path = data_dir / "raw" / "metadata_clean.json"
    raw_images_dir = data_dir / "raw" / "images"
    processed_dir = data_dir / "processed"
    out_images_dir = processed_dir / "images"

    data = load_and_validate(metadata_path)
    data = segment_all(data)
    resize_images(data, raw_images_dir, out_images_dir, num_workers=args.workers)
    log.info("Splitting dataset (80 / 10 / 10) ...")
    splits = split_data(data)
    save_splits(splits, processed_dir)

    log.info("Preprocessing complete.")
    log.info("Split files: processed/{train,val,test}.json")
    log.info("Images: processed/images/")


if __name__ == "__main__":
    main()