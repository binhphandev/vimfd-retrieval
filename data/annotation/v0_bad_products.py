"""
V0: Kiểm tra đầu vào (kế hoạch mục 0) và tạo bad_products.txt
Đầu vào : data/splits/{train,val,test}.jsonl
Đầu ra  : data/annotation/benchmark/bad_products.txt          (mỗi dòng một product_id của TEST)
          data/annotation/benchmark/bad_products_reasons.csv  (id, shop, lý do)
          results/logs/log_A.md  (nối thêm kết quả kiểm tra)
Chạy    : python v0_bad_products.py          (đặt cùng thư mục với v3_attributes.py, v5_hard_negatives.py)

Sản phẩm Test bị liệt kê khi: không có file ảnh đại diện trên đĩa, hoặc description rỗng. Các sản phẩm này
không được dùng làm hạt giống truy vấn (V1) và không được dùng khi kiểm tra bộ truy vấn (mục 4.4).
Ảnh đại diện: is_featured nếu có, nếu không lấy position nhỏ nhất; thử các đường dẫn như V5:
  local_path, ROOT/local_path, config.PROCESSED_DIR/local_path.

--no-image-check chỉ để thử script khi chưa có ảnh trên máy; KHÔNG dùng khi tạo bad_products.txt thật.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():          # chạy từ thư mục khác: dùng thư mục hiện tại
    ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
try:
    import config
except ImportError:
    config = None

SPLIT_DIR = ROOT / "data/splits"
OUT_DIR   = ROOT / "data/annotation/benchmark"
LOG_DIR   = ROOT / "results/logs"
EXPECTED  = {"train": 10690, "val": 1339, "test": 1339}
SCHEMA    = ["id", "name", "category", "type", "colors", "material", "description", "text", "text_seg", "images"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def featured_image_path(product):
    """Cùng quy tắc với V5: is_featured trước, rồi position nhỏ nhất; trả về Path tồn tại hoặc None."""
    images = product.get("images") or []
    if not images:
        return None
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None
    local_path = Path(lp)
    candidates = [local_path, ROOT / local_path]
    if config is not None and getattr(config, "PROCESSED_DIR", None) and not local_path.is_absolute():
        candidates.append(Path(config.PROCESSED_DIR) / local_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-image-check", action="store_true", help="Bỏ qua kiểm tra ảnh (chỉ để thử script)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["\n## V0 — kiểm tra đầu vào"]
    ok_all = True

    # 1. kích thước, tách rời theo id và text_group, schema
    data = {s: load_jsonl(SPLIT_DIR / f"{s}.jsonl") for s in ("train", "val", "test")}
    print("=== Kích thước các tập ===")
    for s, rows in data.items():
        flag = "OK " if len(rows) == EXPECTED[s] else "LỆCH"
        ok_all &= len(rows) == EXPECTED[s]
        print(f"  [{flag}] {s}: {len(rows)} (kỳ vọng {EXPECTED[s]})")
        lines.append(f"- {s}: {len(rows)} sản phẩm (kỳ vọng {EXPECTED[s]})")

    ids = {s: {p["id"] for p in rows} for s, rows in data.items()}
    groups = {s: {p.get("text_group") for p in rows if p.get("text_group")} for s, rows in data.items()}
    print("\n=== Tách rời giữa các tập ===")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n_id, n_g = len(ids[a] & ids[b]), len(groups[a] & groups[b])
        ok = n_id == 0 and n_g == 0
        ok_all &= ok
        print(f"  [{'OK ' if ok else 'LỖI'}] {a} ∩ {b}: id trùng {n_id}, text_group trùng {n_g}")
        lines.append(f"- {a} ∩ {b}: id trùng {n_id}, text_group trùng {n_g}")
    missing_keys = Counter(k for rows in data.values() for p in rows for k in SCHEMA if k not in p)
    print(f"\n=== Schema === trường thiếu: {dict(missing_keys) if missing_keys else 'không'}")
    ok_all &= not missing_keys

    # 2. sản phẩm Test không dùng được
    test = data["test"]
    reasons = []
    for p in test:
        why = []
        if not (p.get("description") or "").strip():
            why.append("mo_ta_rong")
        if not args.no_image_check:
            if not (p.get("images") or []):
                why.append("khong_co_images")
            elif featured_image_path(p) is None:
                why.append("khong_co_file_anh")
        if why:
            reasons.append((p["id"], p["id"].split("_")[0], "+".join(why)))

    bad_ids = sorted(r[0] for r in reasons)
    with open(OUT_DIR / "bad_products.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(bad_ids) + ("\n" if bad_ids else ""))
    with open(OUT_DIR / "bad_products_reasons.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "shop", "reason"])
        w.writerows(sorted(reasons))

    print(f"\n=== bad_products (Test): {len(bad_ids)}/{len(test)} sản phẩm không dùng được ===")
    by_reason = Counter(r[2] for r in reasons)
    by_shop = Counter(r[1] for r in reasons)
    tot_shop = Counter(p["id"].split("_")[0] for p in test)
    print("  theo lý do:", dict(by_reason))
    for shop, n in tot_shop.most_common():
        print(f"  {shop:<10} {by_shop[shop]:>4}/{n:<4} ({by_shop[shop] / n:.0%})")
    print(f"  Còn dùng được làm ứng viên hạt giống: {len(test) - len(bad_ids)}")
    if args.no_image_check:
        print("  [CẢNH BÁO] --no-image-check: KHÔNG kiểm tra ảnh, file này chưa dùng được thật.")
    lines.append(f"- bad_products: {len(bad_ids)}/{len(test)}; lý do {dict(by_reason)}; theo shop {dict(by_shop)}")
    lines.append(f"- ứng viên hạt giống còn lại: {len(test) - len(bad_ids)}")
    lines.append(f"- đường dẫn ảnh thử: ROOT, ROOT/local_path, config.PROCESSED_DIR "
                 f"({getattr(config, 'PROCESSED_DIR', None) if config else 'không có config'})")

    with open(LOG_DIR / "log_A.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nĐã lưu: {OUT_DIR / 'bad_products.txt'}")
    print("V0 " + ("hoàn thành." if ok_all else "xong nhưng CÓ MỤC LỆCH/LỖI ở trên: xem lại trước khi chạy V1."))


if __name__ == "__main__":
    main()