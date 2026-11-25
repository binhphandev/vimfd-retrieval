#!/usr/bin/env python3
"""Đóng gói tối thiểu để chạy V7 / V7b / V8 trên Colab (thay cho việc zip hoặc đẩy cả dự án lên git).

Đặt file này ở GỐC dự án (cạnh config.py). Chạy:

    python make_eval_colab_zip.py                    # code + test.jsonl + queries + ảnh đại diện của sản phẩm Test
    python make_eval_colab_zip.py --with-ckpt        # thêm checkpoint dual-encoder và cross_encoder_best (nặng)
    python make_eval_colab_zip.py --with-ckpt --slim-ckpt   # như trên, nhưng checkpoint dual-encoder chỉ giữ trọng số mô hình
    python make_eval_colab_zip.py --no-images        # chỉ code + dữ liệu nhỏ (nhẹ, up lại mỗi lần sửa code)
    python make_eval_colab_zip.py --images-only      # chỉ ảnh Test (zip một lần)
    python make_eval_colab_zip.py --ckpt-only        # chỉ checkpoint (zip một lần, KHÔNG đưa lên git)

Trong zip có gì:
  - config.py, src/**/*.py (+ json/yaml/txt nhỏ), data/annotation/*.py (V7, V7b, V8, seg_utils.py...), data/dataset.py nếu có
  - data/splits/test.jsonl (KHÔNG có train/val)
  - data/annotation/benchmark/*.csv|json|jsonl (queries_200.csv, translations.csv nếu đã có...)
  - ảnh đại diện (is_featured, hoặc position nhỏ nhất) của từng sản phẩm Test, chọn đúng như V7/V8
  - với --with-ckpt: checkpoint dual-encoder (VIMFD_DE_CKPT > config.CHECKPOINT_BEST > checkpoints/best_24112026.pt)
    và thư mục checkpoints/cross_encoder_best/
Không có: .git, venv, index/, results/, train/val, ảnh train/val, vncorenlp/ (notebook tự tải), __pycache__.

Mọi file nằm dưới thư mục gốc `vimfd-eval/` trong zip (đổi bằng --top-dir); nhiều zip cùng thư mục gốc
giải nén chồng lên nhau được (code zip + images zip + ckpt zip).
Lưu ý: checkpoint thường quá lớn cho git; để chúng trên Drive, chỉ đưa zip code (--no-images) lên git.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
try:
    import config  # noqa: E402  (lấy PROCESSED_DIR, CHECKPOINT_BEST, CKPT_MODEL_KEY để khớp với dự án)
except Exception as e:  # noqa: BLE001
    print(f"  ⚠ không import được config.py ({type(e).__name__}: {e}); dùng giá trị mặc định")
    config = None

TOP_DIR_DEFAULT = "vimfd-eval"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "node_modules"}
CODE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".txt"}
DATA_SUFFIXES = {".csv", ".json", ".jsonl"}
MAX_CODE_FILE_BYTES = 5 * 1024 * 1024
TEST_REL = "data/splits/test.jsonl"
CE_DIR_REL = "checkpoints/cross_encoder_best"
DE_DEFAULT_REL = "checkpoints/best_short_24112026.pt"
REQUIRED = (
    "config.py",
    "src/models/model.py",
    TEST_REL,
    "data/annotation/benchmark/queries_200.csv",
    "data/annotation/seg_utils.py",
    "data/annotation/v7_baselines.py",
    "data/annotation/v8_run_vimfr.py",
)


def _cfg(name, default=None):
    return getattr(config, name, default) if config is not None else default


def _walk(base: Path):
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
        raise ValueError(f"{path} nằm ngoài thư mục dự án {root}; trên Colab các script sẽ không tìm thấy file này. "
                         f"Đưa file vào trong thư mục dự án hoặc dùng đường dẫn tương đối trong config.py.")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _featured_image(product: dict, root: Path):
    """Giống V5/V7/V8: is_featured trước, rồi position nhỏ nhất; thử local_path, root/local_path, PROCESSED_DIR/local_path.
    Trả về (Path tồn tại | None, lý do_thiếu)."""
    images = product.get("images") or []
    if not images:
        return None, "no_images"
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None, "no_local_path"
    local = Path(lp)
    cands = [local, root / local]
    proc = _cfg("PROCESSED_DIR")
    if proc and not local.is_absolute():
        cands.append(Path(proc) / local)
    for c in cands:
        if c.exists():
            return c, ""
    return None, "file_missing"


def _collect_code(root: Path) -> list[Path]:
    files = [root / "config.py", root / "data" / "dataset.py", root / "data" / "__init__.py", root / "requirements.txt"]
    for sub in ("src",):
        base = root / sub
        if base.is_dir():
            for p in _walk(base):
                if p.suffix not in CODE_SUFFIXES:
                    continue
                if p.stat().st_size > MAX_CODE_FILE_BYTES:
                    print(f"  ⚠ bỏ qua file quá lớn trong {sub}/: {_rel(p, root)} ({p.stat().st_size / 1e6:.1f} MB)")
                    continue
                files.append(p)
    ann = root / "data" / "annotation"
    if ann.is_dir():
        files += sorted(p for p in ann.glob("*.py") if p.is_file())
    return [p for p in files if p.is_file()]


def _collect_data(root: Path) -> list[Path]:
    files = [root / TEST_REL]
    bench = root / "data" / "annotation" / "benchmark"
    if bench.is_dir():
        for p in sorted(bench.iterdir()):
            if p.is_file() and p.suffix in DATA_SUFFIXES and not p.name.startswith(".") \
                    and p.stat().st_size <= MAX_CODE_FILE_BYTES:
                files.append(p)
    return files


def _collect_images(root: Path) -> tuple[list[Path], Counter]:
    seen: set[Path] = set()
    out: list[Path] = []
    stats: Counter = Counter()
    shops: Counter = Counter()
    for product in _read_jsonl(root / TEST_REL):
        stats["products"] += 1
        p, why = _featured_image(product, root)
        if p is None:
            stats[why] += 1
            shops[str(product.get("id", "?")).split("_")[0]] += 1
            continue
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    stats["missing"] = stats["no_images"] + stats["no_local_path"] + stats["file_missing"]
    if shops:
        print("  thiếu ảnh theo shop:", dict(shops))
    return out, stats


def _resolve_de_ckpt(root: Path) -> Path:
    p = os.environ.get("VIMFD_DE_CKPT") or _cfg("CHECKPOINT_BEST") or DE_DEFAULT_REL
    p = Path(p)
    return p if p.is_absolute() else root / p


def _fmt(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def make_eval_colab_zip(
    output: str | os.PathLike | None = None,
    mode: str = "full",
    with_ckpt: bool = False,
    slim_ckpt: bool = False,
    top_dir: str = TOP_DIR_DEFAULT,
    extra: tuple[str, ...] = (),
    root: str | os.PathLike = ROOT_DIR,
) -> Path:
    """mode: "full" = code + dữ liệu nhỏ + ảnh Test | "code" = không ảnh | "images" = chỉ ảnh | "ckpt" = chỉ checkpoint."""
    if mode not in {"full", "code", "images", "ckpt"}:
        raise ValueError("mode phải là 'full', 'code', 'images' hoặc 'ckpt'")
    root = Path(root).resolve()
    names = {"full": f"{top_dir}.zip", "code": f"{top_dir}-code.zip", "images": f"{top_dir}-images.zip",
             "ckpt": f"{top_dir}-ckpt.zip"}
    output = (Path(output) if output else root / names[mode]).resolve()
    if mode == "ckpt":
        with_ckpt = True

    entries: list[tuple[Path, str, str]] = []
    if mode in ("full", "code"):
        missing = [r for r in REQUIRED if not (root / r).is_file()]
        if missing:
            raise FileNotFoundError("Thiếu file bắt buộc để chạy V7/V8 trên Colab: " + ", ".join(missing))
        entries += [(p, _rel(p, root), "code") for p in _collect_code(root)]
        entries += [(p, _rel(p, root), "data") for p in _collect_data(root)]
        for e in extra:
            base = root / e
            if not base.exists():
                raise FileNotFoundError(f"--extra không tồn tại: {base}")
            for p in ([base] if base.is_file() else _walk(base)):
                entries.append((p, _rel(p, root), "extra"))
    if mode in ("full", "images"):
        if not (root / TEST_REL).is_file():
            raise FileNotFoundError(f"Thiếu {root / TEST_REL}")
        print("Đang chọn ảnh đại diện của sản phẩm Test ...")
        imgs, st = _collect_images(root)
        entries += [(p, _rel(p, root), "images") for p in imgs]
        print(f"  sản phẩm Test: {st['products']:,} | ảnh duy nhất: {len(imgs):,} | thiếu ảnh: {st['missing']} "
              f"(không có danh sách ảnh {st['no_images']}, không có local_path {st['no_local_path']}, "
              f"file không tồn tại {st['file_missing']})")
        if st["missing"]:
            print("  ⚠ Sản phẩm thiếu ảnh sẽ KHÔNG vào chỉ mục ảnh (V7/V8 đã tính điều này), nhưng nên kiểm tra PROCESSED_DIR.")

    tmpdir = None
    if with_ckpt:
        ce = root / CE_DIR_REL
        if not ce.is_dir():
            raise FileNotFoundError(f"Thiếu {ce}")
        entries += [(p, _rel(p, root), "ckpt") for p in _walk(ce)]
        de = _resolve_de_ckpt(root)
        if not de.is_file():
            raise FileNotFoundError(f"Thiếu checkpoint dual-encoder {de} (đặt VIMFD_DE_CKPT hoặc sửa config.CHECKPOINT_BEST)")
        de_rel = _rel(de, root)
        print(f"Checkpoint dual-encoder: {de_rel} ({_fmt(de.stat().st_size)}); "
              f"nhớ: PHẢI train trên split hiện tại.")
        if slim_ckpt:
            import torch
            tmpdir = tempfile.TemporaryDirectory()
            key = _cfg("CKPT_MODEL_KEY", "model")
            ck = torch.load(de, map_location="cpu", weights_only=False)
            sd = ck.get(key, ck) if isinstance(ck, dict) else ck
            slim = Path(tmpdir.name) / de.name
            torch.save({key: sd}, slim)
            print(f"  --slim-ckpt: {_fmt(de.stat().st_size)} → {_fmt(slim.stat().st_size)} (chỉ giữ '{key}')")
            entries.append((slim, de_rel, "ckpt"))
        else:
            entries.append((de, de_rel, "ckpt"))

    entries = [e for e in entries if e[0].resolve() != output]
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
            ctype = zipfile.ZIP_STORED if group in ("images", "ckpt") else zipfile.ZIP_DEFLATED
            zf.write(path, f"{top_dir}/{rel}", compress_type=ctype)
            sizes[group] += path.stat().st_size
            counts[group] += 1
    tmp.replace(output)
    if tmpdir is not None:
        tmpdir.cleanup()

    print(f"\nĐã tạo {output}  ({_fmt(output.stat().st_size)})")
    for g in ("code", "data", "extra", "images", "ckpt"):
        if counts[g]:
            print(f"  {g:7s}: {counts[g]:>6,} file, {_fmt(sizes[g])} (chưa nén)")
    print(f"Trong notebook: giải nén vào /content rồi ROOT = '/content/{top_dir}'  (ZIP_PATH = '<Drive>/{output.name}')")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo zip tối thiểu để chạy V7/V7b/V8 trên Colab.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-images", action="store_true", help="chỉ code + dữ liệu nhỏ (không ảnh, không checkpoint)")
    g.add_argument("--images-only", action="store_true", help="chỉ ảnh đại diện của sản phẩm Test")
    g.add_argument("--ckpt-only", action="store_true", help="chỉ checkpoint (dual-encoder + cross_encoder_best)")
    ap.add_argument("--with-ckpt", action="store_true", help="thêm checkpoint vào zip (nặng)")
    ap.add_argument("--slim-ckpt", action="store_true",
                    help="checkpoint dual-encoder chỉ giữ trọng số mô hình (bỏ optimizer...); cần torch")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--top-dir", default=TOP_DIR_DEFAULT, help="thư mục gốc trong zip (khớp ROOT trong notebook)")
    ap.add_argument("--extra", nargs="*", default=[], help="file/thư mục thêm (tương đối gốc dự án)")
    a = ap.parse_args()
    mode = "code" if a.no_images else "images" if a.images_only else "ckpt" if a.ckpt_only else "full"
    if a.slim_ckpt and not (a.with_ckpt or a.ckpt_only):
        ap.error("--slim-ckpt chỉ có nghĩa khi dùng cùng --with-ckpt hoặc --ckpt-only")
    make_eval_colab_zip(a.output, mode=mode, with_ckpt=a.with_ckpt, slim_ckpt=a.slim_ckpt,
                        top_dir=a.top_dir, extra=tuple(a.extra))


if __name__ == "__main__":
    main()