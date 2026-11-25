"""
G1: Kiểm tra queries_200.csv trước khi chạy V8
Đầu vào : data/annotation/benchmark/queries_200.csv
Đầu ra  : in kết quả kiểm tra, ghi vào results/logs/log_A.md
Chạy    : python g1_check_queries200.py
"""

import csv, re, json
from pathlib import Path
from collections import Counter

# ── Đường dẫn ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data").exists():
    ROOT = Path(".")

QUERIES_PATH = ROOT / "data/annotation/benchmark/queries_200.csv"
BAD_PATH     = ROOT / "data/annotation/benchmark/bad_products.txt"
TEST_PATH    = ROOT / "data/splits/test.jsonl"
LOG_PATH     = ROOT / "results/logs/log_A.md"

# ── Hằng số kiểm tra ─────────────────────────────────────────────
REQUIRED_COLS = ["query_id", "query_text", "seed_id", "n_attrs",
                 "attrs_in_query", "author", "is_overlap"]
VALID_ATTR_KEYS = {"màu", "chất liệu", "kiểu dáng", "phong cách"}
BRANDS = [
    "aristino", "johnhenry", "john henry", "heboz", "henstore",
    "icondenim", "icon denim", "ninomaxx", "novelty", "speedo",
    "puma", "nike", "aligro", "mrm", "owen", "tiki", "lazada",
]
SYN_WORDS = [
    "ôm sát", "form slim", "form vừa", "suông vừa", "dáng suông",
    "form rộng", "dáng rộng", "vải bông", "da bò", "da thật",
    "vải denim", "vải bò", "lịch sự", "văn phòng", "đường phố",
    "đời thường", "sporty", "vải poly", "len ", "vải sợi tre",
    "chất bông", "chất da", "năng động",
]
FORBID_RE = re.compile(
    r"(?<!\w)(giá|giảm giá|sale|khuyến mãi|freeship|free ship|size|siz|mã|sku)(?!\w)"
    r"|\d+\s?(k|đ|vnđ|vnd|nghìn|ngàn|triệu)(?!\w)"
)

# ── Tiện ích ─────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Chuẩn hóa truy vấn: chữ thường, bỏ dấu câu, bỏ khoảng trắng thừa."""
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()

def _ngrams(text: str, n: int) -> set:
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def _qnum(qid: str) -> int:
    digits = "".join(ch for ch in qid if ch.isdigit())
    return int(digits) if digits else -1

def _is_overlap_flag(val: str) -> bool:
    return val.strip() in ("1", "1.0", "True", "true")

# ── Nạp dữ liệu ──────────────────────────────────────────────────
bad_ids: set = set()
if BAD_PATH.exists():
    bad_ids = set(BAD_PATH.read_text(encoding="utf-8").split())

test_name: dict = {}
if TEST_PATH.exists():
    for line in open(TEST_PATH, encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            test_name[p["id"]] = p.get("name", "")

rows: list = []
with open(QUERIES_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"=== G1 Kiểm tra queries_200.csv ({len(rows)} dòng) ===\n")
errors: list   = []
warnings: list = []

# ── Kiểm tra 1: cột bắt buộc ─────────────────────────────────────
miss_cols = [c for c in REQUIRED_COLS if rows and c not in rows[0]]
if miss_cols:
    errors.append(f"Thiếu cột bắt buộc: {miss_cols}")
else:
    print("    Đủ cột bắt buộc")

# ── Kiểm tra 2: đủ 200 dòng ──────────────────────────────────────
if len(rows) != 200:
    errors.append(f"Số dòng = {len(rows)}, kỳ vọng 200")
else:
    print("    Đủ 200 dòng")

# ── Kiểm tra 3: query_id không trùng, đúng định dạng Q001–Q200 ───
qids = [r["query_id"] for r in rows]
dup_qid = [q for q, c in Counter(qids).items() if c > 1]
if dup_qid:
    errors.append(f"query_id trùng: {dup_qid}")
else:
    print("    query_id không trùng")

bad_fmt = [q for q in qids if not re.match(r"^Q\d{3}$", q)]
if bad_fmt:
    errors.append(f"query_id sai định dạng: {bad_fmt[:5]}")
else:
    print("    Định dạng query_id đúng (Q001–Q200)")

# ── Kiểm tra 4: query_text 2–12 từ ───────────────────────────────
bad_len = [
    (r["query_id"], len(q.split()), q)
    for r in rows
    for q in [(r.get("query_text") or "").strip()]
    if not q or not (2 <= len(q.split()) <= 12)
]
if bad_len:
    errors.append(f"{len(bad_len)} truy vấn ngoài khoảng 2–12 từ:")
    for qid, n, q in bad_len[:5]:
        errors.append(f"  {qid}: {n} từ — '{q}'")
else:
    print("    Mọi query_text đều 2–12 từ")

# ── Kiểm tra 5: không trùng truy vấn sau chuẩn hóa ───────────────
dup_q = [q for q, c in Counter(_norm(r.get("query_text")) for r in rows).items() if c > 1]
if dup_q:
    errors.append(f"{len(dup_q)} truy vấn trùng nhau sau chuẩn hóa: {dup_q[:5]}")
else:
    print("    Không có truy vấn trùng sau chuẩn hóa")

# ── Kiểm tra 6: n_attrs hợp lệ và phân bố đúng chỉ tiêu ─────────
nt_c   = Counter(r.get("n_attrs", "") for r in rows)
bad_nt = [v for v in nt_c if v not in ("1", "2", "3+", "3")]
if bad_nt:
    errors.append(f"n_attrs sai giá trị: {bad_nt}")

n1 = nt_c.get("1", 0)
n2 = nt_c.get("2", 0)
n3 = nt_c.get("3+", 0) + nt_c.get("3", 0)
print(f"\nPhân bố n_attrs: 1={n1} | 2={n2} | 3+={n3}")
if abs(n1 - 70) > 8 or abs(n2 - 70) > 8 or abs(n3 - 60) > 8:
    warnings.append(f"Phân bố n_attrs lệch chỉ tiêu 70/70/60: {n1}/{n2}/{n3}")
else:
    print("    Phân bố n_attrs đạt chỉ tiêu")

# ── Kiểm tra 7: attrs_in_query đúng định dạng, số nhóm khớp n_attrs
bad_attrs = []
bad_n     = []
for r in rows:
    raw   = (r.get("attrs_in_query") or "").strip()
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if not raw:
        bad_attrs.append((r["query_id"], "rỗng"))
        continue
    for p in parts:
        if "=" not in p:
            bad_attrs.append((r["query_id"], f"thiếu '=' trong '{p}'"))
        else:
            key = p.split("=")[0].strip()
            if key not in VALID_ATTR_KEYS:
                warnings.append(f"{r['query_id']}: key lạ '{key}' (chấp nhận được nếu có lý do)")

    keys = {p.split("=")[0].strip() for p in parts if "=" in p}
    nt   = (r.get("n_attrs") or "").strip()
    ok   = (len(keys) >= 3) if nt in ("3", "3+") else (len(keys) == int(nt) if nt.isdigit() else False)
    if not ok or len(keys) != len(parts):
        bad_n.append((r["query_id"], nt, len(keys)))

if bad_attrs:
    errors.append(f"{len(bad_attrs)} dòng attrs_in_query sai định dạng:")
    for qid, reason in bad_attrs[:5]:
        errors.append(f"  {qid}: {reason}")
else:
    print("    attrs_in_query đúng định dạng")

if bad_n:
    errors.append(f"{len(bad_n)} dòng số nhóm thuộc tính không khớp n_attrs: {bad_n[:5]}")
else:
    print("    Số nhóm trong attrs_in_query khớp n_attrs")

# ── Kiểm tra 8: danh mục Áo/Quần/Phụ kiện ───────────────────────
cat_c = Counter(r.get("category", "") for r in rows)
ao, quan, pk = cat_c.get("áo", 0), cat_c.get("quần", 0), cat_c.get("phụ kiện", 0)
print(f"\nPhân bố danh mục: {dict(cat_c)}")
if abs(ao - 80) > 8 or abs(quan - 60) > 8 or abs(pk - 60) > 8:
    warnings.append(f"Danh mục lệch chỉ tiêu Áo80/Quần60/PK60: {ao}/{quan}/{pk}")
else:
    print("    Phân bố danh mục đạt chỉ tiêu")

# ── Kiểm tra 9: seed_id không trùng và thuộc Test ────────────────
seeds = [r.get("seed_id", "").strip() for r in rows]
dup_seed = [s for s, c in Counter(seeds).items() if s and c > 1]
if dup_seed:
    errors.append(f"seed_id trùng (mỗi sản phẩm chỉ làm hạt giống một lần): {dup_seed[:5]}")
else:
    print("    seed_id không trùng")

if test_name:
    not_test = [r["query_id"] for r in rows if r.get("seed_id", "").strip() not in test_name]
    if not_test:
        errors.append(f"{len(not_test)} hạt giống không thuộc test.jsonl: {not_test[:5]}")
    else:
        print("    Mọi hạt giống thuộc Test")
else:
    warnings.append(f"Không thấy {TEST_PATH}: bỏ qua kiểm tra seed_id thuộc Test")

# ── Kiểm tra 10: hạt giống không trong bad_products ──────────────
if bad_ids:
    bad_seed = [r["query_id"] for r in rows if r.get("seed_id", "") in bad_ids]
    if bad_seed:
        errors.append(f"Hạt giống trong bad_products: {bad_seed}")
    else:
        print("    Không có hạt giống nào trong bad_products")

# ── Kiểm tra 11: không chép name (cụm 4 từ liên tiếp) ───────────
# Ưu tiên cột name trong CSV; nếu thiếu thì tra theo seed_id từ test.jsonl.
copy_issues = []
no_name_checked = 0
for r in rows:
    q    = (r.get("query_text") or "").lower()
    name = (r.get("name") or "").strip()
    if not name and test_name:
        name = test_name.get(r.get("seed_id", "").strip(), "")
        no_name_checked += 1
    if not q or not name:
        continue
    overlap = _ngrams(q, 4) & _ngrams(name.lower(), 4)
    if overlap:
        copy_issues.append((r["query_id"], sorted(overlap)[:2]))
if copy_issues:
    errors.append(f"{len(copy_issues)} truy vấn chép cụm ≥4 từ từ name:")
    for qid, grams in copy_issues[:5]:
        errors.append(f"  {qid}: {grams}")
else:
    suffix = f" ({no_name_checked} dòng tra theo seed_id)" if no_name_checked else ""
    print(f"    Không có cụm 4 từ nào bị chép từ name{suffix}")

# ── Kiểm tra 12: không chứa thương hiệu ──────────────────────────
brand_issues = [
    (r["query_id"], [b for b in BRANDS if b in (r.get("query_text") or "").lower()])
    for r in rows
    if any(b in (r.get("query_text") or "").lower() for b in BRANDS)
]
if brand_issues:
    errors.append(f"{len(brand_issues)} truy vấn chứa tên thương hiệu:")
    for qid, found in brand_issues[:5]:
        errors.append(f"  {qid}: {found}")
else:
    print("    Không có tên thương hiệu")

# ── Kiểm tra 13: không chứa giá/size/khuyến mãi ─────────────────
bad_f = []
for r in rows:
    q = (r.get("query_text") or "").lower()
    m = FORBID_RE.search(q)
    if m:
        bad_f.append((r["query_id"], m.group(0)))
if bad_f:
    errors.append(f"{len(bad_f)} truy vấn chứa giá/size/khuyến mãi/mã: {bad_f[:5]}")
else:
    print("    Không có từ giá/size/khuyến mãi")

# ── Kiểm tra 14: is_overlap đúng (Q086–Q100, Q186–Q200) ──────────
exp_ov = {q for q in qids if 86 <= _qnum(q) <= 100 or 186 <= _qnum(q) <= 200}
got_ov = {r["query_id"] for r in rows if _is_overlap_flag(r.get("is_overlap") or "")}
if got_ov != exp_ov:
    errors.append(
        f"is_overlap sai: thiếu {sorted(exp_ov - got_ov)[:5]}, thừa {sorted(got_ov - exp_ov)[:5]}"
    )
else:
    print(f"    is_overlap đúng ({len(got_ov)} truy vấn: Q086–Q100, Q186–Q200)")

# ── Kiểm tra 15: tỉ lệ từ đồng nghĩa ≥ 30% ──────────────────────
syn_count = sum(1 for r in rows if any(w in (r.get("query_text") or "").lower() for w in SYN_WORDS))
syn_pct   = syn_count / len(rows) * 100
print(f"\nDùng từ đồng nghĩa: {syn_count}/200 ({syn_pct:.0f}%)")
if syn_pct < 30:
    warnings.append(f"Tỉ lệ từ đồng nghĩa {syn_pct:.0f}% < 30%")
else:
    print("    Tỉ lệ từ đồng nghĩa đạt ≥ 30%")

# ── Kiểm tra 16: không quá 40% truy vấn dạng "loại, màu, ..." ────
def _color_first(r: dict) -> bool:
    q   = (r.get("query_text") or "").lower()
    pos = {}
    for p in (r.get("attrs_in_query") or "").split(";"):
        if "=" in p:
            k, v = [x.strip().lower() for x in p.split("=", 1)]
            i = q.find(v)
            if v and i >= 0:
                pos[k] = i
    return "màu" in pos and pos["màu"] == min(pos.values()) and pos["màu"] > 0

n_cf    = sum(1 for r in rows if _color_first(r))
cf_pct  = n_cf / max(len(rows), 1)
print(f"\nƯớc lượng truy vấn theo thứ tự 'loại, màu, ...': {n_cf}/{len(rows)} ({cf_pct:.0%})")
if cf_pct > 0.40:
    warnings.append(
        f"'loại, màu, ...' chiếm {cf_pct:.0%} > 40% (ước lượng; đếm lại bằng mắt nếu sát ngưỡng)"
    )

# ── Tổng kết ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
if errors:
    print(f"❌ {len(errors)} LỖI CẦN SỬA TRƯỚC KHI CHẠY V8:")
    for e in errors:
        print(f"  {e}")
else:
    print("    KHÔNG CÓ LỖI — Có thể chạy V8")

if warnings:
    print(f"\n⚠️  {len(warnings)} CẢNH BÁO (không bắt buộc sửa):")
    for w in warnings:
        print(f"  {w}")

# ── Ghi log ───────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n## G1 — Kiểm tra queries_200.csv\n")
    f.write(f"- Tổng: {len(rows)} dòng\n")
    f.write(f"- n_attrs: 1={n1} / 2={n2} / 3+={n3}\n")
    f.write(f"- Danh mục: áo={ao} / quần={quan} / phụ kiện={pk}\n")
    f.write(f"- Từ đồng nghĩa: {syn_count}/200\n")
    f.write(f"- Lỗi: {len(errors)} | Cảnh báo: {len(warnings)}\n")
    if errors:
        f.write("### Lỗi\n")
        for e in errors:
            f.write(f"  {e}\n")
    if warnings:
        f.write("### Cảnh báo\n")
        for w in warnings:
            f.write(f"  {w}\n")

print(f"\nĐã ghi log vào {LOG_PATH}")