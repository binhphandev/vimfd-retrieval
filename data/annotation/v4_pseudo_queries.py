"""
V4 (bản viết lại): Sinh truy vấn giả từ Train và Val
Đầu vào:
  - data/annotation/cross_encoder/product_attrs.json   (từ V3)
  - data/annotation/cross_encoder/attr_dictionary.xlsx (từ V3, sheet synonyms, đã được B duyệt)
  - data/splits/train.jsonl, val.jsonl
Đầu ra:
  - data/annotation/cross_encoder/pseudo_train.csv
  - data/annotation/cross_encoder/pseudo_val.csv
    Cột: pq_id, query_text (ĐÃ TÁCH TỪ), pos_id, attrs (JSON, giá trị chuẩn), n_attrs, query_raw (chưa tách từ)
  - results/logs/log_A.md (nối thêm thống kê V4)
Chạy: python v4_pseudo_queries.py
Yêu cầu: pip install openpyxl underthesea

Khác bản cũ (cần ghi vào mục 12 / log_A.md):
  1. Truy vấn trùng chữ: giữ TỐI ĐA 3 (mỗi truy vấn gắn một sản phẩm dương khác nhau), thay vì 1 (kế hoạch 5.2).
     Sản phẩm nào gặp truy vấn đã đủ 3 thì rút lại ngẫu nhiên (tối đa MAX_TRIES lần) trước khi bỏ qua.
  2. query_text được tách từ bằng underthesea.word_tokenize(..., format="text") đúng mục 3.2/5.3;
     bản chưa tách từ lưu ở cột query_raw (để đọc và bỏ trùng).
  3. Mọi mẫu câu đều chứa ĐỦ các thuộc tính đã chọn. Bản cũ có mẫu bỏ rơi thuộc tính (cột attrs ghi 3 thuộc tính
     nhưng câu chỉ nêu 2) làm V5 lọc âm giả sai, nên bị sửa.
  4. Dùng RNG riêng theo từng sản phẩm (seed 42 + id) và duyệt sản phẩm theo thứ tự xáo seed 42, nên kết quả tái
     lập được và việc giới hạn 3 truy vấn trùng không thiên vị shop nào đứng đầu file.
  5. Không thay đồng nghĩa nếu từ đồng nghĩa đã nằm trong tên loại (tránh "thắt lưng da da", "quần jean jean").
"""

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl
import seg_utils

SEED = 42

# ── Tham số ───────────────────────────────────────────────────────
MAX_DUP        = 3      # tối đa số truy vấn cùng chữ (chốt của nhóm)
MAX_TRIES      = 5      # số lần rút lại cho một sản phẩm khi truy vấn đã đủ MAX_DUP
P_SYNONYM      = 0.3    # xác suất thay giá trị bằng đồng nghĩa (nếu có)
P_SHUFFLE      = 0.5    # xác suất xáo thứ tự thuộc tính
N_ATTR_WEIGHTS = {1: 0.3, 2: 0.4, 3: 0.3}

ROOT       = Path(".")
ATTRS_PATH = ROOT / "data/annotation/cross_encoder/product_attrs.json"
DICT_PATH  = ROOT / "data/annotation/cross_encoder/attr_dictionary.xlsx"
TRAIN_PATH = ROOT / "data/splits/train.jsonl"
VAL_PATH   = ROOT / "data/splits/val.jsonl"
OUT_DIR    = ROOT / "data/annotation/cross_encoder"
LOG_DIR    = ROOT / "results/logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ATTR_GROUPS = ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]


# ── Hàm chuẩn hóa (mục 3.2) ───────────────────────────────────────
def normalize(text) -> str:
    """Unicode NFC, chữ thường, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def segment(text: str) -> str:
    """Tách từ đúng cách đã dùng để tạo text_seg của sản phẩm."""
    return seg_utils.segment_text(text)


# ── Load dữ liệu ──────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


print("Load product_attrs.json...")
with open(ATTRS_PATH, encoding="utf-8") as f:
    product_attrs = json.load(f)

print("Load bảng đồng nghĩa từ attr_dictionary.xlsx (sheet synonyms)...")
wb = openpyxl.load_workbook(DICT_PATH, read_only=True)
# synonym_map[(attr_type, value_norm)] = [synonym, ...]   (khóa theo cả nhóm để không lẫn kaki màu/chất liệu)
synonym_map = {}
for row in wb["synonyms"].iter_rows(min_row=2, values_only=True):
    norm, atype, syn = row[:3]
    if norm and atype and syn:
        synonym_map.setdefault((atype, normalize(norm)), []).append(normalize(syn))
n_pairs = sum(len(v) for v in synonym_map.values())
print(f"  {n_pairs} cặp đồng nghĩa")

train_data = load_jsonl(TRAIN_PATH)
val_data   = load_jsonl(VAL_PATH)
print(f"  Train: {len(train_data)} | Val: {len(val_data)}")


# ── Mẫu câu (mục 5.2) ─────────────────────────────────────────────
# Mỗi mẫu nhận: loai, kd (kiểu dáng), pc (phong cách), mau, cl (chất liệu), others (các thuộc tính còn lại, đã
# nối thành chuỗi). Luôn chèn others vào cuối để câu chứa ĐỦ mọi thuộc tính đã chọn.
def tpl_generic(loai, vals, g):           # {loại} {thuộc tính 1} {thuộc tính 2}
    return f"{loai} " + " ".join(v for _, v in vals)

def tpl_mau(loai, vals, g):               # {loại} màu {màu} {thuộc tính khác}
    others = " ".join(v for t, v in vals if t != "MAU")
    return f"{loai} màu {g['MAU']} {others}"

def tpl_kd_pc(loai, vals, g):             # {loại} {kiểu dáng} phong cách {phong cách} (+ thuộc tính khác)
    others = " ".join(v for t, v in vals if t not in ("KIEU_DANG", "PHONG_CACH"))
    return f"{loai} {g['KIEU_DANG']} phong cách {g['PHONG_CACH']} {others}"

def tpl_chat(loai, vals, g):              # tìm {loại} {chất liệu} {thuộc tính khác}
    others = " ".join(v for t, v in vals if t != "CHAT_LIEU")
    return f"tìm {loai} {g['CHAT_LIEU']} {others}"

def tpl_pc_mau(loai, vals, g):            # {loại} {phong cách} {màu} (+ thuộc tính khác)
    others = " ".join(v for t, v in vals if t not in ("PHONG_CACH", "MAU"))
    return f"{loai} {g['PHONG_CACH']} {g['MAU']} {others}"

# (tên, hàm, nhóm bắt buộc có mặt)
TEMPLATES = [
    ("generic", tpl_generic, set()),
    ("mau",     tpl_mau,     {"MAU"}),
    ("kd_pc",   tpl_kd_pc,   {"KIEU_DANG", "PHONG_CACH"}),
    ("chat",    tpl_chat,    {"CHAT_LIEU"}),
    ("pc_mau",  tpl_pc_mau,  {"PHONG_CACH", "MAU"}),
]


def build_query(loai: str, chosen: dict, rng: random.Random):
    """chosen = {nhóm: giá trị chuẩn}. Trả về (câu_truy_vấn, tên_mẫu, số_giá_trị_được_thay_đồng_nghĩa)."""
    # 1. Thay đồng nghĩa với xác suất P_SYNONYM (bỏ qua nếu từ đồng nghĩa đã nằm trong tên loại)
    surface = {}
    n_syn = 0
    for t, v in chosen.items():
        syns = [s for s in synonym_map.get((t, v), []) if f" {s} " not in f" {loai} "]
        if syns and rng.random() < P_SYNONYM:
            surface[t] = rng.choice(syns)
            n_syn += 1
        else:
            surface[t] = v

    # 2. Xáo thứ tự thuộc tính ở P_SHUFFLE trường hợp
    order = list(chosen.keys())
    if rng.random() < P_SHUFFLE:
        rng.shuffle(order)
    vals = [(t, surface[t]) for t in order]

    # 3. Chọn ngẫu nhiên một mẫu phù hợp với các nhóm đã chọn
    ok = [(name, fn) for name, fn, need in TEMPLATES if need <= set(chosen)]
    name, fn = rng.choice(ok)
    query = re.sub(r"\s+", " ", fn(loai, vals, surface)).strip()

    # 4. Kiểm tra: mọi giá trị phải có mặt trong câu
    for t, s in surface.items():
        assert s in query, f"Thiếu thuộc tính {t}={s} trong câu: {query}"
    return query, name, n_syn


# ── Sinh truy vấn giả cho một split ───────────────────────────────
def generate_pseudo_queries(split_data, split_name):
    split_ids = {p["id"] for p in split_data}
    order = list(range(len(split_data)))
    random.Random(SEED).shuffle(order)           # duyệt theo thứ tự xáo để giới hạn trùng không thiên vị shop

    raw_count = Counter()                        # số truy vấn đã nhận theo chữ (đã chuẩn hóa)
    accepted = []                                # (chỉ_số_gốc, dict)
    stat = Counter()
    tpl_used = Counter()

    for idx in order:
        product = split_data[idx]
        pid = product["id"]
        attrs = product_attrs.get(pid)
        if attrs is None:
            stat["thiếu_attrs"] += 1
            continue
        if not attrs.get("LOAI"):
            stat["không_có_loại"] += 1
            continue
        loai = attrs["LOAI"][0]

        available = {g: attrs[g][0] for g in ATTR_GROUPS if attrs.get(g)}
        if not available:
            stat["không_có_thuộc_tính"] += 1
            continue

        rng = random.Random(f"{SEED}:{split_name}:{pid}")
        done = False
        for _ in range(MAX_TRIES):
            cands = [n for n in (1, 2, 3) if n <= len(available)]
            n = rng.choices(cands, weights=[N_ATTR_WEIGHTS[c] for c in cands], k=1)[0]
            keys = rng.sample(list(available), n)
            chosen = {k: available[k] for k in keys}

            query, tpl, n_syn = build_query(loai, chosen, rng)
            key = normalize(query)
            if raw_count[key] >= MAX_DUP:
                continue                         # đã đủ 3 truy vấn cùng chữ -> rút lại
            raw_count[key] += 1
            accepted.append((idx, {
                "query_raw": query,
                "pos_id":    pid,
                "attrs":     json.dumps(chosen, ensure_ascii=False),
                "n_attrs":   n,
            }))
            stat["syn_used"] += n_syn
            tpl_used[tpl] += 1
            done = True
            break
        if not done:
            stat["bỏ_vì_trùng"] += 1

    # Sắp lại theo thứ tự gốc trong file split, gán pq_id, rồi tách từ
    accepted.sort(key=lambda x: x[0])
    rows = []
    for i, (_, r) in enumerate(accepted, start=1):
        assert r["pos_id"] in split_ids              # không lẫn sản phẩm split khác / Test
        rows.append({
            "pq_id":      f"{split_name}_pq_{i:05d}",
            "query_text": segment(r["query_raw"]),
            "pos_id":     r["pos_id"],
            "attrs":      r["attrs"],
            "n_attrs":    r["n_attrs"],
            "query_raw":  r["query_raw"],
        })

    n_prod_ok = len(rows)
    print(f"  {split_name}: {n_prod_ok} truy vấn giả | bỏ qua: {dict((k, v) for k, v in stat.items() if k != 'syn_used')}")
    return rows, stat, tpl_used, raw_count


print("\nSinh truy vấn giả...")
train_pseudo, train_stat, train_tpl, train_cnt = generate_pseudo_queries(train_data, "train")
val_pseudo,   val_stat,   val_tpl,   val_cnt   = generate_pseudo_queries(val_data,   "val")

# ── Lưu CSV ───────────────────────────────────────────────────────
fieldnames = ["pq_id", "query_text", "pos_id", "attrs", "n_attrs", "query_raw"]
for name, rows in [("pseudo_train", train_pseudo), ("pseudo_val", val_pseudo)]:
    out_path = OUT_DIR / f"{name}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã lưu: {out_path}  ({len(rows)} dòng)")

# ── Thống kê và kiểm tra ──────────────────────────────────────────
lines = ["\n## V4 — truy vấn giả",
         f"- Tham số: seed={SEED}, MAX_DUP={MAX_DUP}, MAX_TRIES={MAX_TRIES}, P_SYNONYM={P_SYNONYM}, "
         f"P_SHUFFLE={P_SHUFFLE}, n_attrs~{N_ATTR_WEIGHTS}"]
for name, rows, stat, tpl, cnt, data in [("train", train_pseudo, train_stat, train_tpl, train_cnt, train_data),
                                         ("val",   val_pseudo,   val_stat,   val_tpl,   val_cnt,   val_data)]:
    n = len(rows)
    dist = Counter(r["n_attrs"] for r in rows)
    n_unique = len(cnt)
    dup_hist = Counter(cnt.values())
    n_usable = n + stat["bỏ_vì_trùng"]
    lines.append(f"- {name}: {n}/{n_usable} sản phẩm dùng được có truy vấn ({n/max(n_usable,1):.1%}); "
                 f"bỏ vì trùng: {stat['bỏ_vì_trùng']}; chữ truy vấn khác nhau: {n_unique}")
    lines.append(f"  n_attrs: " + ", ".join(f"n={k}: {dist[k]} ({dist[k]/n:.0%})" for k in sorted(dist)))
    lines.append(f"  số truy vấn cùng chữ (1/2/3 lần): " + "/".join(str(dup_hist.get(k, 0)) for k in (1, 2, 3)))
    lines.append(f"  mẫu câu: " + ", ".join(f"{k}={v}" for k, v in tpl.most_common()))
    lines.append(f"  thay đồng nghĩa: {stat['syn_used']} giá trị")
    top = Counter(r["query_raw"] for r in rows).most_common(5)
    lines.append(f"  5 truy vấn lặp nhiều nhất: " + "; ".join(f"{q} ({c})" for q, c in top))

report = "\n".join(lines)
print(report)
with open(LOG_DIR / "log_A.md", "a", encoding="utf-8") as f:
    f.write(report + "\n")

# Kiểm tra bắt buộc: không giao nhau giữa pos_id của Train và Val, và không có chữ nào trùng quá MAX_DUP
assert not ({r["pos_id"] for r in train_pseudo} & {r["pos_id"] for r in val_pseudo})
assert max(train_cnt.values()) <= MAX_DUP and max(val_cnt.values()) <= MAX_DUP

print("\n=== 15 truy vấn mẫu (đọc bằng mắt) ===")
random.Random(SEED).shuffle(train_pseudo)
for r in train_pseudo[:15]:
    print(f"  {r['query_raw']:<55} -> {r['query_text']:<60} {r['attrs']}")

print("\nV4 hoàn thành! Đọc 15 truy vấn mẫu ở trên: câu phải tự nhiên và chứa đủ thuộc tính trong cột attrs.")