"""
Chẩn đoán: vì sao V5 báo "thiếu ảnh: 7511"?
Đặt cùng thư mục với v5_hard_negatives.py (để ROOT giống V5) rồi chạy:  python check_images.py
In ra: số ảnh đại diện tìm thấy / không thấy theo từng shop, thư mục ảnh thật trên đĩa,
và nếu một shop mất toàn bộ ảnh thì tìm xem file ảnh đó nằm ở đâu.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import config
except ImportError:
    config = None

BASES = [ROOT]
if config is not None:
    for attr in ("PROCESSED_DIR", "DATA_DIR", "RAW_DIR", "IMAGE_DIR", "IMAGES_DIR"):
        v = getattr(config, attr, None)
        if v:
            BASES.append(Path(v))
print("Các thư mục gốc thử:", [str(b) for b in BASES])


def featured_local_path(p):
    imgs = sorted(p.get("images") or [], key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    return imgs[0].get("local_path") if imgs else None


def find(lp):
    for b in BASES:
        for cand in (Path(lp), b / lp):
            if cand.exists():
                return cand
    return None


for split in ("train", "val"):
    path = ROOT / f"data/splits/{split}.jsonl"
    prods = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok, miss, example = Counter(), Counter(), {}
    for p in prods:
        shop = p["id"].split("_")[0]
        lp = featured_local_path(p)
        if lp and find(lp):
            ok[shop] += 1
        else:
            miss[shop] += 1
            example.setdefault(shop, lp)
    print(f"\n=== {split}: {len(prods)} sản phẩm | thấy ảnh {sum(ok.values())} | KHÔNG thấy {sum(miss.values())} ===")
    for shop in sorted(set(ok) | set(miss), key=lambda s: -(ok[s] + miss[s])):
        print(f"  {shop:<10} thấy {ok[shop]:>5}   không thấy {miss[shop]:>5}   ví dụ thiếu: {example.get(shop)}")

    # shop mất (gần) toàn bộ ảnh -> tìm file ở đâu
    for shop, n in miss.items():
        if n and ok[shop] == 0 and example.get(shop):
            name = Path(example[shop]).name
            print(f"\n  Tìm '{name}' dưới {ROOT} ...")
            hit = next(iter(ROOT.rglob(name)), None)
            print(f"    -> {hit if hit else 'KHÔNG tìm thấy ở đâu (ảnh chưa tải hoặc nằm ngoài ROOT)'}")

print("\nThư mục ảnh trên đĩa:")
for b in BASES:
    d = b / "images"
    if d.exists():
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                n = sum(1 for _ in sub.glob("*.*"))
                print(f"  {sub}: {n} file")