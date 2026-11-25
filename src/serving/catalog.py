"""
Quy tắc dữ liệu dùng CHUNG cho build_index.py (lúc tạo index) và engine.py (lúc phục vụ).
Hai nơi cùng import từ đây nên không thể lệch nhau về:
  - nguồn dữ liệu : SPLIT_DIR/{train,val,test}.jsonl  (không dùng PROCESSED_DIR)
  - ảnh đại diện  : is_featured -> position nhỏ nhất, chỉ xét ảnh có local_path
  - đường dẫn ảnh : chịu được local_path ghi .png nhưng file thật là .jpg
  - transform ảnh : đúng get_image_transform('test') của data/dataset.py (BICUBIC)
  - ảnh truy vấn  : đi qua to_square (đệm trắng thành vuông) như ảnh gallery trước khi vào transform
  - dấu vân tay checkpoint, để biết index được build từ checkpoint nào
Các quy tắc này khớp với train_colab.ipynb.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

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


def to_square(img: Image.Image, size: int | None = None) -> Image.Image:
    """Sao chép NGUYÊN VĂN common.to_square (lúc export ảnh gallery): xoay theo EXIF, ghép nền trắng cho ảnh
    trong suốt, resize LANCZOS giữ tỉ lệ về cạnh dài = size rồi đặt giữa canvas trắng size x size.
    Ảnh upload nhờ vậy đi qua đúng phép biến đổi (cùng một lần resample) như ảnh gallery.
    Nếu đổi common.to_square thì phải đổi ở đây."""
    size = size or cfg.IMAGE_SIZE
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def prepare_query_image(img: Image.Image) -> Image.Image:
    """Ảnh người dùng -> ảnh cùng dạng với gallery. Gọi trước get_eval_transform()."""
    return to_square(img)


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


# ---------- chuẩn hóa màu ----------
# Màu chuẩn -> các từ khóa (tiếng Việt/Anh) xuất hiện trong tên sản phẩm, trường `colors` và query.
# "xanh" đứng một mình là màu chưa rõ sắc độ nên có nhóm riêng, không gán bừa vào xanh dương/lá.
COLOR_VOCAB: dict[str, list[str]] = {
    "trắng": ["trắng", "white", "ivory"],
    "đen": ["đen", "black"],
    "xám": ["xám", "ghi", "grey", "gray"],
    "xanh navy": ["xanh navy", "navy", "xanh đen", "xanh than", "xanh tím than"],
    "xanh dương": ["xanh dương", "xanh biển", "xanh da trời", "xanh coban", "xanh lam", "blue"],
    "xanh rêu": ["xanh rêu", "rêu", "olive", "xanh olive", "xanh quân đội"],
    "xanh lá": ["xanh lá", "xanh lục", "green"],
    "xanh": ["xanh"],
    "be": ["be", "kem", "cream", "beige", "nude"],
    "nâu": ["nâu", "brown", "cà phê", "caramel"],
    "đỏ": ["đỏ", "đỏ đô", "red"],
    "vàng": ["vàng", "yellow", "mustard"],
    "hồng": ["hồng", "pink"],
    "tím": ["tím", "purple"],
    "cam": ["cam", "orange"],
}
_COLOR_KW = sorted(((kw, c) for c, kws in COLOR_VOCAB.items() for kw in kws), key=lambda x: -len(x[0]))


def extract_colors(text: str) -> list[str]:
    """Màu chuẩn xuất hiện trong text. Từ khóa dài khớp trước nên "xanh navy" không bị tính thêm "xanh"."""
    import re
    t = " " + re.sub(r"[^\w\s]|_", " ", (text or "").lower()) + " "
    found = []
    for kw, canon in _COLOR_KW:
        pat = f" {kw} "
        if pat in t:
            if canon not in found:
                found.append(canon)
            t = t.replace(pat, "  ")
    return found


def strip_colors(text: str, canons) -> str:
    """Bỏ mọi từ khóa thuộc các màu chuẩn `canons` khỏi text (giữ nguyên hoa/thường phần còn lại)."""
    import re
    for kw, canon in _COLOR_KW:
        if canon in canons:
            text = re.sub(rf"(?<!\w){re.escape(kw)}(?!\w)", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()