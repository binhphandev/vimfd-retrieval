"""
build_queries200.py — Gộp seeds_A.xlsx và seeds_B.xlsx thành queries_200.csv

Chạy:
    python build_queries200.py

Đầu vào (mặc định trong cùng thư mục script):
    seeds_A.xlsx  — sheet "seeds", slot_id Q001–Q100
    seeds_B.xlsx  — sheet "seeds", slot_id Q101–Q200

Đầu ra:
    data/annotation/benchmark/queries_200.csv

Quy tắc:
  - author = A cho Q001–Q100, B cho Q101–Q200
  - is_overlap = 1 cho Q086–Q100 và Q186–Q200, còn lại 0
  - Các cột từ seeds giữ nguyên: seed_id, category, name, n_attrs,
    attrs_in_query, query_text
  - reserve=1 → dùng dữ liệu dự phòng; script cảnh báo nhưng vẫn ghi
  - skipped_reason không rỗng → cảnh báo người dùng kiểm tra lại
"""

import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Thiếu pandas. Chạy: pip install pandas openpyxl")

# ── Đường dẫn ─────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if not (ROOT / "data").exists():
    ROOT = HERE          # chạy từ thư mục tuỳ ý

SEEDS_A = next(p for p in [HERE/"benchmark/seeds_A.xlsx", HERE/"seeds_A.xlsx"] if p.exists())
SEEDS_B = next(p for p in [HERE/"benchmark/seeds_B.xlsx", HERE/"seeds_B.xlsx"] if p.exists())
OUT_PATH  = ROOT / "data/annotation/benchmark/queries_200.csv"


def _qnum(qid: str) -> int:
    d = "".join(c for c in qid if c.isdigit())
    return int(d) if d else -1


def load_seeds(path: Path, expected_author: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="seeds", dtype=str).fillna("")
    df = df.rename(columns={"n_target": "n_attrs"})

    missing = [c for c in ["slot_id", "seed_id", "query_text", "attrs_in_query",
                            "category", "n_attrs"] if c not in df.columns]
    if missing:
        sys.exit(f"[LỖI] {path.name} thiếu cột: {missing}")

    # Lọc bỏ dòng slot_id rỗng (hàng trống cuối file)
    df = df[df["slot_id"].str.match(r"^Q\d{3}$")].copy()

    # Cảnh báo reserve
    if "reserve" in df.columns:
        res = df[df["reserve"].str.strip() == "1"]["slot_id"].tolist()
        if res:
            print(f"⚠️  {path.name}: {len(res)} dòng reserve=1 "
                  f"(đã dùng hạt giống dự phòng): {res[:5]}")

    # Cảnh báo skipped_reason
    if "skipped_reason" in df.columns:
        sk = df[df["skipped_reason"].str.strip() != ""]["slot_id"].tolist()
        if sk:
            print(f"⚠️  {path.name}: {len(sk)} dòng có skipped_reason "
                  f"(nên kiểm tra lại): {sk[:5]}")

    df["author"] = expected_author
    return df


# ── Nạp dữ liệu ───────────────────────────────────────────────────
print("Đang nạp seeds_A.xlsx …")
df_a = load_seeds(SEEDS_A, "A")
print(f"  → {len(df_a)} dòng ({df_a['slot_id'].iloc[0]}–{df_a['slot_id'].iloc[-1]})")

print("Đang nạp seeds_B.xlsx …")
df_b = load_seeds(SEEDS_B, "B")
print(f"  → {len(df_b)} dòng ({df_b['slot_id'].iloc[0]}–{df_b['slot_id'].iloc[-1]})")

df = pd.concat([df_a, df_b], ignore_index=True)

# ── Đặt is_overlap ────────────────────────────────────────────────
df["is_overlap"] = df["slot_id"].apply(
    lambda q: 1 if 86 <= _qnum(q) <= 100 or 186 <= _qnum(q) <= 200 else 0
)

# ── Chọn và đặt tên cột đầu ra ────────────────────────────────────
# Giữ cột name nếu có (G1 dùng để kiểm tra chép name)
optional = ["name"] if "name" in df.columns else []

out_cols = (["slot_id", "seed_id", "category"] + optional +
            ["n_attrs", "attrs_in_query", "query_text", "author", "is_overlap"])

df_out = df[out_cols].rename(columns={"slot_id": "query_id"})

# ── Kiểm tra nhanh trước khi ghi ─────────────────────────────────
n = len(df_out)
if n != 200:
    print(f"⚠️  Tổng dòng = {n}, kỳ vọng 200. Kiểm tra lại hai file seeds.")

dup = df_out["query_id"][df_out["query_id"].duplicated()].tolist()
if dup:
    print(f"⚠️  query_id trùng: {dup}")

empty_q = df_out[df_out["query_text"].str.strip() == ""]["query_id"].tolist()
if empty_q:
    print(f"⚠️  {len(empty_q)} dòng query_text rỗng: {empty_q[:5]}")

ov_got  = df_out[df_out["is_overlap"] == 1]["query_id"].tolist()
ov_want = [q for q in df_out["query_id"] if 86 <= _qnum(q) <= 100 or 186 <= _qnum(q) <= 200]
if set(ov_got) != set(ov_want):
    print(f"⚠️  is_overlap sai: got={sorted(ov_got)[:5]}, want={sorted(ov_want)[:5]}")

# ── Ghi file ──────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

ov_count = int((df_out["is_overlap"] == 1).sum())
print(f"\n✅ Đã ghi {n} dòng → {OUT_PATH}")
print(f"   Overlap: {ov_count} truy vấn (Q086–Q100, Q186–Q200)")
print(f"   Author A: {(df_out['author']=='A').sum()} | Author B: {(df_out['author']=='B').sum()}")
print("\nBước tiếp theo: python data/annotation/g1_check_queries200.py")