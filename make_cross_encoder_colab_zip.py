"""
make_cross_encoder_colab_zip.py: đóng gói dự án để chạy V5, V6 trên Colab (chạy TẠI THƯ MỤC GỐC dự án, nơi có config.py).
    python make_cross_encoder_colab_zip.py
    python make_cross_encoder_colab_zip.py --ckpt checkpoints/best_24112026.pt --out vimfd-cross-encoder.zip
Gói gồm: config.py, src/, data/splits/{train,val}.jsonl (KHÔNG có test), data/annotation/*.py,
data/annotation/cross_encoder/* (product_attrs.json, attr_dictionary.xlsx, pseudo_*.csv), checkpoint dual-encoder,
và CHỈ ảnh đại diện của sản phẩm Train+Val (V5 chỉ dùng ảnh đại diện). Giữ nguyên đường dẫn tương đối.
Sau đó tải file zip lên Google Drive (ví dụ MyDrive/vimfd/vimfd-cross-encoder.zip).
"""
import argparse, json, os, sys, zipfile
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
import config

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--out", default="vimfd-cross-encoder.zip")
args = ap.parse_args()

PREFIX = "vimfd-retrieval"
ckpt = Path(config.CHECKPOINT_BEST)
ckpt = Path(ckpt) if Path(ckpt).is_absolute() else ROOT / ckpt
if not ckpt.exists():
    raise SystemExit(f"Không thấy checkpoint: {ckpt}. Dùng --ckpt <đường dẫn>.")

files = {}   # arcname -> path
def add(path):
    path = Path(path)
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        print("  [BỎ QUA] ngoài thư mục dự án:", path); return
    files[str(rel).replace("\\", "/")] = path

add(ROOT / "config.py")
for p in (ROOT / "src").rglob("*.py"): add(p)
for p in (ROOT / "data/annotation").glob("*.py"): add(p)
for p in (ROOT / "data/annotation/cross_encoder").glob("*"):
    if p.is_file(): add(p)
for s in ("train", "val"): add(ROOT / f"data/splits/{s}.jsonl")
add(ckpt)

n_img = n_miss = 0
for s in ("train", "val"):
    for line in open(ROOT / f"data/splits/{s}.jsonl", encoding="utf-8"):
        if not line.strip(): continue
        p = json.loads(line)
        ims = sorted(p.get("images") or [], key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
        if not ims or not ims[0].get("local_path"): n_miss += 1; continue
        lp = Path(ims[0]["local_path"])
        cands = [lp, ROOT / lp]
        if getattr(config, "PROCESSED_DIR", None) and not lp.is_absolute():
            cands.append(Path(config.PROCESSED_DIR) / lp)
        hit = next((c for c in cands if c.exists()), None)
        if hit is None: n_miss += 1; continue
        add(hit); n_img += 1
print(f"Ảnh đại diện: {n_img} có, {n_miss} thiếu")

with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:     # ảnh jpg đã nén sẵn nên không nén lại
    for arc, path in files.items():
        z.write(path, f"{PREFIX}/{arc}")
print(f"Đã tạo {args.out}: {len(files)} file, {Path(args.out).stat().st_size / 1e6:.0f} MB")