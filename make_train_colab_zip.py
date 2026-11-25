#!/usr/bin/env python3
"""Đóng gói tối thiểu để chạy train_colab.ipynb (thay cho việc zip thủ công cả dự án).

Đặt file này ở GỐC dự án (cạnh config.py). Chạy:

    python make_train_colab_zip.py                 # 1 file đủ cả code + splits + ảnh (notebook chạy ngay)
    python make_train_colab_zip.py --no-images     # chỉ code + splits (nhẹ, up lại mỗi lần sửa code)
    python make_train_colab_zip.py --images-only   # chỉ ảnh (zip 1 lần, ít khi đổi)

Hoặc gọi hàm:  from make_train_colab_zip import make_train_colab_zip; make_train_colab_zip(mode="code")

Trong zip có gì:
  - config.py, src/**/*.py (+ json/yaml/txt nhỏ), data/dataset.py (+ data/__init__.py)
  - data/splits/*.jsonl, split.json
  - các ảnh được train/val/test.jsonl tham chiếu (đúng đường dẫn dataset.py sẽ mở)
Không có: .git, venv, checkpoints/, index/, results/, frontend/, vncorenlp/ (notebook tự tải),
data/raw, các *_clean.jsonl trong data/processed, __pycache__.

Mọi file nằm dưới thư mục gốc `vimfd-train/` trong zip để khớp với notebook
(`extractall('/content/')` rồi ROOT = '/content/vimfd-train'); nhiều zip có cùng thư mục gốc
nên giải nén chồng lên nhau được (code zip + images zip).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402  (lấy đường dẫn + tên trường để luôn khớp với dự án)

TOP_DIR_DEFAULT = "vimfd-train"
SPLITS = ("train", "val", "test")
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "node_modules"}
CODE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".txt"}
MAX_CODE_FILE_BYTES = 5 * 1024 * 1024        # chặn lỡ tay nhét file model/dữ liệu lớn vào src/
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
REQUIRED = (
    "config.py",
    "data/dataset.py",
    "src/models/model.py",
    "src/models/loss.py",
    "src/training/train.py",
    "src/serving/catalog.py",
    "src/serving/text_utils.py",
)


def _walk(base: Path):
    """Duyệt file dưới base, bỏ thư mục rác và file ẩn."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            yield Path(dirpath) / name


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"{path} nằm ngoài thư mục dự án {root}; notebook sẽ không tìm thấy file này.")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _collect_code(root: Path) -> list[Path]:
    files = [root / "config.py", root / "data" / "dataset.py", root / "data" / "__init__.py"]
    src = root / "src"
    if src.is_dir():
        for p in _walk(src):
            if p.suffix not in CODE_SUFFIXES:
                continue
            if p.stat().st_size > MAX_CODE_FILE_BYTES:
                print(f"  ⚠ bỏ qua file quá lớn trong src/: {_rel(p, root)} ({p.stat().st_size / 1e6:.1f} MB)")
                continue
            files.append(p)
    return [p for p in files if p.is_file()]


def _collect_splits() -> list[Path]:
    split_dir = Path(config.SPLIT_DIR)
    for s in SPLITS:
        if not (split_dir / f"{s}.jsonl").is_file():
            raise FileNotFoundError(f"Thiếu {split_dir / (s + '.jsonl')} (chạy build_splits.py trước).")
    return [p for p in sorted(split_dir.iterdir()) if p.is_file() and p.suffix in {".jsonl", ".json"}]


def _collect_images() -> tuple[list[Path], Counter]:
    """Ảnh được 3 split tham chiếu, theo đúng quy tắc mở file của dataset.py:
    bỏ tiền tố 'images/' rồi nối vào IMAGE_DIR. Nếu file thật lệch đuôi thì vẫn lấy file thật
    (và đếm vào ext_mismatch: dataset.py sẽ lỗi khi train, notebook sẽ báo ở bước kiểm tra dữ liệu)."""
    image_dir = Path(config.IMAGE_DIR)
    seen: set[Path] = set()
    stats: Counter = Counter()
    out: list[Path] = []
    for s in SPLITS:
        for product in _read_jsonl(Path(config.SPLIT_DIR) / f"{s}.jsonl"):
            for im in product.get(config.IMAGE_LIST_FIELD) or []:
                if not (isinstance(im, dict) and im.get(config.IMAGE_PATH_FIELD)):
                    continue
                stats["referenced"] += 1
                rel = Path(im[config.IMAGE_PATH_FIELD])
                if rel.parts and rel.parts[0] == "images":
                    rel = Path(*rel.parts[1:])
                p = image_dir / rel
                if not p.is_file():
                    alt = next((p.with_suffix(e) for e in IMAGE_EXTS if p.with_suffix(e).is_file()), None)
                    if alt is None:
                        stats["missing"] += 1
                        if stats["missing"] <= 5:
                            print(f"  ⚠ thiếu ảnh: {p}")
                        continue
                    stats["ext_mismatch"] += 1
                    p = alt
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out, stats


def _fmt(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def make_train_colab_zip(
    output: str | os.PathLike | None = None,
    mode: str = "full",
    top_dir: str = TOP_DIR_DEFAULT,
    extra: tuple[str, ...] = (),
    root: str | os.PathLike = ROOT_DIR,
) -> Path:
    """Tạo zip cho Colab và trả về đường dẫn file zip.

    mode: "full"   = code + splits + ảnh (1 file, notebook chạy ngay)
          "code"   = code + splits (không ảnh)
          "images" = chỉ ảnh
    extra: đường dẫn (tương đối gốc dự án) của file/thư mục muốn thêm vào zip.
    """
    if mode not in {"full", "code", "images"}:
        raise ValueError("mode phải là 'full', 'code' hoặc 'images'")
    root = Path(root).resolve()
    default_names = {"full": f"{top_dir}.zip", "code": f"{top_dir}-code.zip", "images": f"{top_dir}-images.zip"}
    output = Path(output) if output else root / default_names[mode]
    output = output.resolve()

    entries: list[tuple[Path, str, str]] = []     # (đường dẫn thật, tên trong zip, nhóm)
    if mode in ("full", "code"):
        missing = [r for r in REQUIRED if not (root / r).is_file()]
        if missing:
            raise FileNotFoundError("Thiếu file bắt buộc để chạy notebook: " + ", ".join(missing))
        entries += [(p, _rel(p, root), "code") for p in _collect_code(root)]
        entries += [(p, _rel(p, root), "splits") for p in _collect_splits()]
        for e in extra:
            base = root / e
            if not base.exists():
                raise FileNotFoundError(f"--extra không tồn tại: {base}")
            for p in ([base] if base.is_file() else _walk(base)):
                entries.append((p, _rel(p, root), "extra"))
    if mode in ("full", "images"):
        print("Đang quét ảnh từ train/val/test.jsonl ...")
        imgs, stats = _collect_images()
        entries += [(p, _rel(p, root), "images") for p in imgs]
        print(f"  ảnh tham chiếu: {stats['referenced']:,} | file duy nhất: {len(imgs):,} | "
              f"thiếu: {stats['missing']} | lệch đuôi: {stats['ext_mismatch']}")
        if stats["missing"] or stats["ext_mismatch"]:
            print("  ⚠ Có ảnh thiếu/lệch đuôi: notebook (ô kiểm tra dữ liệu) sẽ dừng; sửa dữ liệu trước khi train.")

    entries = [e for e in entries if e[0].resolve() != output]       # không tự nhét zip vào chính nó
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    sizes: Counter = Counter()
    counts: Counter = Counter()
    try:
        from tqdm import tqdm
        it = tqdm(entries, desc="nén", unit="file")
    except ImportError:
        it = entries
    with zipfile.ZipFile(tmp, "w") as zf:
        for path, rel, group in it:
            # ảnh jpg/png/webp đã nén sẵn -> chỉ lưu (nhanh); còn lại nén deflate
            ctype = zipfile.ZIP_STORED if group == "images" else zipfile.ZIP_DEFLATED
            zf.write(path, f"{top_dir}/{rel}", compress_type=ctype)
            sizes[group] += path.stat().st_size
            counts[group] += 1
    tmp.replace(output)

    print(f"\nĐã tạo {output}  ({_fmt(output.stat().st_size)})")
    for g in ("code", "splits", "extra", "images"):
        if counts[g]:
            print(f"  {g:7s}: {counts[g]:>6,} file, {_fmt(sizes[g])} (chưa nén)")
    var = "IMAGES_ZIP_PATH" if mode == "images" else "ZIP_PATH"
    print(f"Trong notebook (ô cấu hình): {var} = '<đường dẫn Drive>/{output.name}'")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo zip tối thiểu để chạy train_colab.ipynb.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-images", action="store_true", help="chỉ code + splits (không ảnh)")
    g.add_argument("--images-only", action="store_true", help="chỉ ảnh được 3 split tham chiếu")
    ap.add_argument("-o", "--output", default=None, help="đường dẫn file zip đầu ra")
    ap.add_argument("--top-dir", default=TOP_DIR_DEFAULT,
                    help="thư mục gốc trong zip (phải khớp ROOT trong notebook)")
    ap.add_argument("--extra", nargs="*", default=[], help="file/thư mục thêm vào zip (tương đối gốc dự án)")
    a = ap.parse_args()
    mode = "code" if a.no_images else "images" if a.images_only else "full"
    make_train_colab_zip(a.output, mode=mode, top_dir=a.top_dir, extra=tuple(a.extra))


if __name__ == "__main__":
    main()