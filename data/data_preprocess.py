"""
Pipeline:
    1. Load & validate metadata_clean.json
    2. Word-segment các field text (PhoBERT yêu cầu)
    3. Chia train/val/test theo stratify (shop x category) — tỉ lệ 80/10/10
    4. Lưu splits ra processed/train.json, val.json, test.json
    5. Resize ảnh về 224x224 (CLIP ViT-B/32) và lưu ra processed/images/

Cách dùng:
    python data_preprocess.py
    python data_preprocess.py --skip-images
    python data_preprocess.py --skip-split
    python data_preprocess.py --data-dir /path/to/data --workers 8

Cài đặt:
    pip install underthesea pillow scikit-learn tqdm
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PRODUCT_ID_FIELD,
    TEXT_FIELDS,
    IMAGE_SIZE,
    IMAGE_LIST_FIELD,
    IMAGE_PRIMARY_FIELD,
    IMAGE_POSITION_FIELD,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- Word segmentation ---

def _load_segmenter() -> Callable[[str], str]:
    try:
        from underthesea import word_tokenize
        log.info("Word segmenter: underthesea ✓")
        return lambda text: str(word_tokenize(text, format="text"))
    except ImportError:
        raise ImportError(
            "Thiếu underthesea. Cài đặt: pip install underthesea\n"
            "PhoBERT yêu cầu text đã word-segment — không thể bỏ qua bước này."
        )

_segment_fn: Callable[[str], str] = _load_segmenter()

def segment_text(text: str) -> str:
    return _segment_fn(text) if text else text

def segment_product(product: dict) -> dict:
    p = product.copy()
    for field in TEXT_FIELDS:
        value = p.get(field)
        if isinstance(value, str) and value.strip():
            p[field] = segment_text(value)
        elif isinstance(value, list):
            p[field] = [segment_text(v) if isinstance(v, str) else v for v in value]
    return p


# --- Bước 1: Load & validate ---

def load_and_validate(metadata_path: Path) -> list[dict]:
    log.info(f"Loading {metadata_path} ...")
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    valid = []
    skipped = {
        "no_product_id": 0,
        "no_category": 0,
        "no_shop": 0,
        "no_images": 0,
    }

    for p in data:
        if not p.get(PRODUCT_ID_FIELD):
            skipped["no_product_id"] += 1
            continue
        if not p.get("category"):
            skipped["no_category"] += 1
            continue
        if not p.get("shop"):
            skipped["no_shop"] += 1
            continue
        if not p.get("images"):
            skipped["no_images"] += 1
            continue
        valid.append(p)

    log.info(f"  Total: {total} | Valid: {len(valid)} | Skipped: {skipped}")

    if skipped["no_product_id"] > 0:
        log.warning(
            f"  {skipped['no_product_id']} sản phẩm thiếu field '{PRODUCT_ID_FIELD}'."
        )

    return valid


# --- Bước 2: Word-segment text ---

def segment_all(data: list[dict]) -> list[dict]:
    log.info(f"Word-segmenting {len(data)} products (fields: {TEXT_FIELDS}) ...")
    return [segment_product(p) for p in tqdm(data, desc="Segment")]


# --- Bước 3: Chia split ---

def split_data(data: list[dict]) -> dict[str, list[dict]]:
    strata = [f"{p['shop']}_{p['category']}" for p in data]

    train, temp, _, strata_temp = train_test_split(
        data, strata,
        test_size=(VAL_RATIO + TEST_RATIO),
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
        log.info(f"  {name}: {len(subset)} products ({len(subset)/len(data)*100:.1f}%)")

    return splits


# --- Bước 4: Lưu split ---

def _pick_primary_image(images: list[dict]) -> dict | None:
    if not images:
        return None
    for img in images:
        if img.get(IMAGE_PRIMARY_FIELD):
            return img
    for img in images:
        if img.get(IMAGE_POSITION_FIELD) == 0:
            return img
    return images[0]

def save_splits(splits: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        cleaned = []
        for p in subset:
            p = p.copy()
            primary = _pick_primary_image(p.get(IMAGE_LIST_FIELD, []))
            p[IMAGE_LIST_FIELD] = [primary] if primary else []
            cleaned.append(p)

        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        log.info(f"  Saved {out_path} ({len(cleaned)} records)")


# --- Bước 5: Resize ảnh ---

def resize_one(args: tuple) -> tuple[bool, str]:
    src_path, dest_path = args
    if dest_path.exists():
        return True, f"skip (exists): {dest_path}"

    if not src_path.exists():
        for alt_ext in (".webp", ".jpg", ".png"):
            if alt_ext == src_path.suffix:
                continue
            alt_path = src_path.with_suffix(alt_ext)
            if alt_path.exists():
                src_path = alt_path
                break
        else:
            return False, f"src not found: {src_path}"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
            img.save(dest_path, "JPEG", quality=95)
        return True, str(dest_path)
    except FileNotFoundError:
        return False, f"src not found: {src_path}"
    except UnidentifiedImageError:
        return False, f"cannot identify image: {src_path}"
    except Exception as e:
        return False, f"error {src_path}: {e}"

def resize_images(
    data: list[dict],
    raw_images_dir: Path,
    out_images_dir: Path,
    num_workers: int = 4,
) -> None:
    tasks = []
    for product in data:
        primary = _pick_primary_image(product.get(IMAGE_LIST_FIELD, []))
        if not primary:
            continue
        rel = Path(primary["local_path"])
        if rel.parts[0] == "images":
            rel = Path(*rel.parts[1:])
        tasks.append((raw_images_dir / rel, out_images_dir / rel))

    tasks = list({t[1]: t for t in tasks}.values())
    log.info(f"Resizing {len(tasks)} images to {IMAGE_SIZE}x{IMAGE_SIZE} ...")

    ok_count = skipped_count = err_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(resize_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Resize"):
            success, msg = future.result()
            if not success:
                err_count += 1
                errors.append(msg)
            elif msg.startswith("skip"):
                skipped_count += 1
            else:
                ok_count += 1

    log.info(f"  Done — resized: {ok_count} | skipped: {skipped_count} | errors: {err_count}")
    if errors:
        log.warning("  First 10 errors:")
        for e in errors[:10]:
            log.warning(f"    {e}")


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description="ViMFD data preprocessing pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()

    data_dir = args.data_dir
    metadata_path = data_dir / "raw" / "metadata_clean.json"
    raw_images_dir = data_dir / "raw" / "images"
    processed_dir = data_dir / "processed"
    out_images_dir = processed_dir / "images"

    data = load_and_validate(metadata_path)

    if not args.skip_split:
        data = segment_all(data)
        log.info("Splitting dataset (80 / 10 / 10) ...")
        splits = split_data(data)
        save_splits(splits, processed_dir)
    else:
        log.info("Skipping split + segment step.")

    if not args.skip_images:
        if not raw_images_dir.exists():
            log.error(f"raw images dir not found: {raw_images_dir}")
            return
        resize_images(data, raw_images_dir, out_images_dir, num_workers=args.workers)
    else:
        log.info("Skipping image resize step.")

    log.info("Preprocessing complete.")
    log.info("  Split files : processed/{train,val,test}.json")
    log.info("  Images      : processed/images/")


if __name__ == "__main__":
    main()