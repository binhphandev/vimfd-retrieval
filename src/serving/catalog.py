"""
Quy tắc dữ liệu dùng CHUNG cho build_index.py (lúc tạo index) và engine.py (lúc phục vụ).
Hai nơi cùng import từ đây nên không thể lệch nhau về:
  - nguồn dữ liệu : SPLIT_DIR/{train,val,test}.jsonl  (không dùng PROCESSED_DIR)
  - ảnh đại diện  : is_featured -> position nhỏ nhất, chỉ xét ảnh có local_path
  - đường dẫn ảnh : chịu được local_path ghi .png nhưng file thật là .jpg
  - transform ảnh : đúng get_image_transform('test') của data/dataset.py (BICUBIC)
  - dấu vân tay checkpoint, để biết index được build từ checkpoint nào
Các quy tắc này khớp với train_colab.ipynb.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import config as cfg

log = logging.getLogger(__name__)

EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ---------- dữ liệu ----------
def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON lỗi tại {path}:{n}") from e
    return out


def load_products(split_dir=None, splits=("train", "val", "test")) -> list[dict]:
    """Toàn bộ sản phẩm từ SPLIT_DIR/*.jsonl, bỏ trùng id (giữ bản đầu tiên)."""
    split_dir = Path(split_dir or cfg.SPLIT_DIR)
    seen, out = set(), []
    for s in splits:
        p = split_dir / f"{s}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy {p}. Hãy chạy build_splits.py trước.")
        for item in _read_jsonl(p):
            pid = str(item[cfg.PRODUCT_ID_FIELD])
            if pid not in seen:
                seen.add(pid)
                out.append(item)
    log.info(f"Loaded {len(out)} products từ {split_dir}")
    return out


# ---------- ảnh ----------
def primary_image(product: dict) -> dict | None:
    """Ảnh đại diện: is_featured, nếu không có thì position nhỏ nhất. Chỉ xét ảnh có local_path."""
    imgs = [im for im in (product.get(cfg.IMAGE_LIST_FIELD) or [])
            if isinstance(im, dict) and im.get(cfg.IMAGE_PATH_FIELD)]
    if not imgs:
        return None
    for im in imgs:
        if im.get(cfg.IMAGE_PRIMARY_FIELD):
            return im

    def pos(im):
        p = im.get(cfg.IMAGE_POSITION_FIELD)
        return p if isinstance(p, (int, float)) else float("inf")

    return min(imgs, key=pos)


def resolve_image(local_path: str) -> Path | None:
    """local_path trong JSON có thể ghi .png nhưng file thật là .jpg."""
    if not local_path:
        return None
    rel = Path(local_path)
    if rel.is_absolute():
        p = rel
    else:
        if rel.parts and rel.parts[0] == "images":
            rel = Path(*rel.parts[1:])
        p = Path(cfg.IMAGE_DIR) / rel
    if p.exists():
        return p
    for e in EXTS:
        if p.with_suffix(e).exists():
            return p.with_suffix(e)
    return None


def get_eval_transform():
    """Đúng transform eval lúc train/đánh giá (data/dataset.py)."""
    try:
        from data.dataset import get_image_transform
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
        from dataset import get_image_transform
    return get_image_transform("test")


# ---------- dấu vân tay checkpoint / metadata index ----------
def ckpt_fingerprint(path) -> str:
    """sha256 toàn file (vài giây). Đọc một phần là không đủ vì các lớp bị đóng băng giống hệt nhau giữa các checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def meta_path(index_file) -> Path:
    return Path(index_file).with_suffix(".meta.json")


def write_index_meta(index_file, **info) -> Path:
    p = meta_path(index_file)
    info["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return p