"""
Tiền xử lý dữ liệu ViMFD cho hệ thống retrieval.

Pipeline:
    1. Load & validate metadata_clean.json
    2. Chia train/val/test theo stratify (shop x category)
    3. Lưu splits ra data/processed/splits/
    4. Resize ảnh về 224x224 (chuẩn ViT CLIP/B32) và lưu ra data/processed/images/

Cách dùng:
    # Chạy toàn bộ pipeline
    python data_preprocess.py

    # Chỉ chia split (không resize ảnh)
    python data_preprocess.py --skip-images

    # Chỉ resize ảnh (đã có split rồi)
    python data_preprocess.py --skip-split

    # Tuỳ chỉnh đường dẫn
    python data_preprocess.py --data-dir /path/to/data --workers 8
"""

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

# Kích thước đầu vào của ViT CLIP/B32
IMAGE_SIZE = 224

# Tỉ lệ split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15  # = 1 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bước 1: Load & validate
# ---------------------------------------------------------------------------

def load_and_validate(metadata_path: Path) -> list[dict]:
    """Load metadata_clean.json và loại các record không hợp lệ."""
    log.info(f"Loading {metadata_path} ...")
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    valid = []
    skipped = {"no_category": 0, "no_shop": 0, "no_images": 0}

    for p in data:
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
    return valid


# ---------------------------------------------------------------------------
# Bước 2: Chia split
# ---------------------------------------------------------------------------

def split_data(data: list[dict]) -> dict[str, list[dict]]:
    """
    Stratify theo (shop x category) để đảm bảo mỗi nhóm
    đều có đại diện ở cả 3 tập.
    """
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
        from collections import Counter
        dist = Counter(f"{p['shop']}_{p['category']}" for p in subset)
        log.info(f"  {name}: {len(subset)} products")
        for stratum, cnt in sorted(dist.items()):
            log.info(f"    {stratum}: {cnt}")

    return splits


def save_splits(splits: dict[str, list[dict]], out_dir: Path) -> None:
    """Lưu mỗi split ra file JSON riêng."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        log.info(f"  Saved {out_path} ({len(subset)} records)")


# ---------------------------------------------------------------------------
# Bước 3: Resize ảnh
# ---------------------------------------------------------------------------

def resize_one(args: tuple) -> tuple[bool, str]:
    """
    Resize một ảnh về IMAGE_SIZE x IMAGE_SIZE bằng BICUBIC và lưu ra dest.
    Trả về (success, message).
    """
    src_path, dest_path = args
    if dest_path.exists():
        return True, f"skip (exists): {dest_path}"

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
    """
    Resize toàn bộ ảnh trong dataset (không lặp lại nếu đã có).
    local_path trong metadata có dạng: images/<shop>/<filename>.jpg
    → raw_images_dir / <shop> / <filename>.jpg
    """
    tasks = []
    for product in data:
        for img_meta in product["images"]:
            # local_path: "images/aristino/aristino_xxx_1.jpg"
            # → bỏ prefix "images/" vì raw_images_dir đã trỏ vào folder images/
            rel = Path(img_meta["local_path"])
            if rel.parts[0] == "images":
                rel = Path(*rel.parts[1:])  # shop/filename.jpg

            src  = raw_images_dir / rel
            dest = out_images_dir / rel
            tasks.append((src, dest))

    # Dedup (một ảnh có thể xuất hiện trong nhiều product nếu có shared image)
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
        log.warning(f"  First 10 errors:")
        for e in errors[:10]:
            log.warning(f"    {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ViMFD data preprocessing pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--skip-split", action="store_true",
        help="Bỏ qua bước chia split (dùng khi split đã có)",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Bỏ qua bước resize ảnh",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Số thread để resize ảnh song song (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir       = args.data_dir
    metadata_path  = data_dir / "raw" / "metadata_clean.json"
    raw_images_dir = data_dir / "raw" / "images"
    splits_dir     = data_dir / "processed" / "splits"
    out_images_dir = data_dir / "processed" / "images"

    # --- Bước 1: Load ---
    data = load_and_validate(metadata_path)

    # --- Bước 2: Split ---
    if not args.skip_split:
        log.info("Splitting dataset ...")
        splits = split_data(data)
        save_splits(splits, splits_dir)
    else:
        log.info("Skipping split step.")

    # --- Bước 3: Resize ảnh ---
    if not args.skip_images:
        if not raw_images_dir.exists():
            log.error(f"raw images dir not found: {raw_images_dir}")
            log.error("Hãy chắc chắn folder ảnh gốc đặt tại data/raw/images/")
            return
        resize_images(data, raw_images_dir, out_images_dir, num_workers=args.workers)
    else:
        log.info("Skipping image resize step.")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()