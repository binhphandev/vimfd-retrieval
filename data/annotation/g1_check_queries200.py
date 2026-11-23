"""
G1: Kiểm tra queries_200.csv trước khi chạy V8
Đầu vào : data/annotation/benchmark/queries_200.csv
Đầu ra  : in kết quả kiểm tra, ghi vào results/logs/log_A.md
Chạy    : python g1_check_queries200.py
"""

import csv, re, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data").exists():
    ROOT = Path(".")

QUERIES_PATH = ROOT / "data/annotation/benchmark/queries_200.csv"
LOG_PATH     = ROOT / "results/logs/log_A.md"

# ── Tải bad_products (không được là hạt giống) ───────────────────
BAD_PATH = ROOT / "data/annotation/benchmark/bad_products.txt"
bad_ids  = set()
if BAD_PATH.exists():
    bad_ids = set(BAD_PATH.read_text(encoding="utf-8").split())

# ── Load ──────────────────────────────────────────────────────────
rows = []
with open(QUERIES_PATH, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"=== G1 Kiểm tra queries_200.csv ({len(rows)} dòng) ===\n")
errors   = []
warnings = []

# ── 1. Đủ 200 dòng ───────────────────────────────────────────────
if len(rows) != 200:
    errors.append(f"Số dòng = {len(rows)}, kỳ vọng 200")
else:
    print("✅ Đủ 200 dòng")

# ── 2. query_id không trùng, đúng định dạng Q001–Q200 ────────────
qids = [r["query_id"] for r in rows]
dup  = [q for q, c in Counter(qids).items() if c > 1]
if dup:
    errors.append(f"query_id trùng: {dup}")
else:
    print("✅ query_id không trùng")

bad_fmt = [q for q in qids if not re.match(r"^Q\d{3}$", q)]
if bad_fmt:
    errors.append(f"query_id sai định dạng: {bad_fmt[:5]}")
else:
    print("✅ Định dạng query_id đúng (Q001–Q200)")

# ── 3. query_text: 2–12 từ, không rỗng ───────────────────────────
bad_len = []
for r in rows:
    q = (r.get("query_text") or "").strip()
    n = len(q.split())
    if not q or n < 2 or n > 12:
        bad_len.append((r["query_id"], n, q))
if bad_len:
    errors.append(f"{len(bad_len)} truy vấn ngoài khoảng 2–12 từ:")
    for qid, n, q in bad_len[:5]:
        errors.append(f"  {qid}: {n} từ — '{q}'")
else:
    print("✅ Mọi query_text đều 2–12 từ")

# ── 4. n_attrs hợp lệ (1, 2, 3+) và phân bố đúng chỉ tiêu ───────
nt_c = Counter(r.get("n_attrs","") for r in rows)
bad_nt = [v for v in nt_c if v not in ("1","2","3+","3")]
if bad_nt:
    errors.append(f"n_attrs sai giá trị: {bad_nt}")

# Chỉ tiêu: 35/35/30 (±5 mỗi nhóm)
n1 = nt_c.get("1",0)
n2 = nt_c.get("2",0)
n3 = nt_c.get("3+",0) + nt_c.get("3",0)
print(f"\nPhân bố n_attrs: 1={n1} | 2={n2} | 3+={n3}")
if abs(n1-35)>5 or abs(n2-35)>5 or abs(n3-30)>5:
    warnings.append(f"Phân bố n_attrs lệch chỉ tiêu 35/35/30: {n1}/{n2}/{n3}")
else:
    print("✅ Phân bố n_attrs đạt chỉ tiêu")

# ── 5. attrs_in_query: đúng định dạng "nhóm=giá trị; nhóm=giá trị" ─
VALID_KEYS = {"màu","chất liệu","kiểu dáng","phong cách"}
bad_attrs  = []
for r in rows:
    a = (r.get("attrs_in_query") or "").strip()
    if not a:
        bad_attrs.append((r["query_id"], "rỗng"))
        continue
    parts = [p.strip() for p in a.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            bad_attrs.append((r["query_id"], f"thiếu '=' trong '{p}'"))
        else:
            key = p.split("=")[0].strip()
            if key not in VALID_KEYS:
                warnings.append(f"{r['query_id']}: key lạ '{key}' (chấp nhận được nếu có lý do)")
if bad_attrs:
    errors.append(f"{len(bad_attrs)} dòng attrs_in_query sai định dạng:")
    for qid, reason in bad_attrs[:5]:
        errors.append(f"  {qid}: {reason}")
else:
    print("✅ attrs_in_query đúng định dạng")

# ── 6. Danh mục Áo/Quần/Phụ kiện ────────────────────────────────
cat_c = Counter(r.get("category","") for r in rows)
print(f"\nPhân bố danh mục: {dict(cat_c)}")
ao = cat_c.get("áo",0)
quan = cat_c.get("quần",0)
pk = cat_c.get("phụ kiện",0)
if abs(ao-80)>8 or abs(quan-60)>8 or abs(pk-60)>8:
    warnings.append(f"Danh mục lệch chỉ tiêu Áo80/Quần60/PK60: {ao}/{quan}/{pk}")
else:
    print("✅ Phân bố danh mục đạt chỉ tiêu")

# ── 7. Hạt giống không nằm trong bad_products ────────────────────
if bad_ids:
    bad_seed = [r["query_id"] for r in rows if r.get("seed_id","") in bad_ids]
    if bad_seed:
        errors.append(f"Hạt giống trong bad_products: {bad_seed}")
    else:
        print("✅ Không có hạt giống nào trong bad_products")

# ── 8. Không chép name (cụm 4 từ liên tiếp) ─────────────────────
def ngrams(text, n):
    words = text.lower().split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

copy_issues = []
for r in rows:
    q    = (r.get("query_text") or "").lower()
    name = (r.get("name") or "").lower()
    if not q or not name: continue
    q_grams    = set(ngrams(q, 4))
    name_grams = set(ngrams(name, 4))
    overlap    = q_grams & name_grams
    if overlap:
        copy_issues.append((r["query_id"], list(overlap)[:2]))
if copy_issues:
    errors.append(f"{len(copy_issues)} truy vấn chép cụm ≥4 từ từ name:")
    for qid, grams in copy_issues[:5]:
        errors.append(f"  {qid}: {grams}")
else:
    print("✅ Không có cụm 4 từ nào bị chép từ name")

# ── 9. Không chứa thương hiệu ────────────────────────────────────
BRANDS = ["aristino","johnhenry","heboz","henstore","icondenim",
          "ninomaxx","novelty","speedo","puma","nike","aligro","mrm"]
brand_issues = []
for r in rows:
    q = (r.get("query_text") or "").lower()
    found = [b for b in BRANDS if b in q]
    if found:
        brand_issues.append((r["query_id"], found))
if brand_issues:
    errors.append(f"{len(brand_issues)} truy vấn chứa tên thương hiệu:")
    for qid, brands in brand_issues[:5]:
        errors.append(f"  {qid}: {brands}")
else:
    print("✅ Không có tên thương hiệu")

# ── 10. Tỉ lệ từ đồng nghĩa ≥ 30% ───────────────────────────────
SYN_WORDS = ["ôm sát","form slim","form vừa","suông vừa","dáng suông",
             "form rộng","dáng rộng","vải bông","da bò","da thật",
             "vải denim","vải bò","lịch sự","văn phòng","đường phố",
             "đời thường","sporty","vải poly","len ","vải sợi tre",
             "chất bông","chất da","năng động"]
syn_count = sum(1 for r in rows if any(w in (r.get("query_text","")).lower() for w in SYN_WORDS))
pct = syn_count / len(rows) * 100
print(f"\nDùng từ đồng nghĩa: {syn_count}/200 ({pct:.0f}%)")
if pct < 30:
    warnings.append(f"Tỉ lệ từ đồng nghĩa {pct:.0f}% < 30%")
else:
    print("✅ Tỉ lệ từ đồng nghĩa đạt ≥ 30%")

# ── Tổng kết ──────────────────────────────────────────────────────
print("\n" + "="*50)
if errors:
    print(f"❌ {len(errors)} LỖI CẦN SỬA TRƯỚC KHI CHẠY V8:")
    for e in errors:
        print(f"  {e}")
else:
    print("✅ KHÔNG CÓ LỖI — Có thể chạy V8")

if warnings:
    print(f"\n⚠️  {len(warnings)} CẢNH BÁO (không bắt buộc sửa):")
    for w in warnings:
        print(f"  {w}")

# ── Ghi log ──────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n## G1 — Kiểm tra queries_200.csv\n")
    f.write(f"- Tổng: {len(rows)} dòng\n")
    f.write(f"- n_attrs: 1={n1} / 2={n2} / 3+={n3}\n")
    f.write(f"- Danh mục: áo={ao} / quần={quan} / phụ kiện={pk}\n")
    f.write(f"- Từ đồng nghĩa: {syn_count}/200\n")
    f.write(f"- Lỗi: {len(errors)} | Cảnh báo: {len(warnings)}\n")

print(f"\nĐã ghi log vào {LOG_PATH}")