"""
V9: Pooling và xuất file annotation (kế hoạch mục 8)
Đầu vào : results/runs/{dual_encoder,vimfr}.run (V8) và sáu baseline của V7:
          {tfidf,bm25,clip_vi2en,bienc_text,fashionclip_vi2en,viclip_ot}.run
          data/annotation/benchmark/queries_200.csv   (query_id, query_text, attrs_in_query, seed_id, ...)
          data/splits/test.jsonl
          KHÔNG còn phụ thuộc attr_utils.py (cột type hiển thị là type gốc của nguồn, đã chuẩn hóa chữ thường).
Đầu ra  : data/annotation/benchmark/annotation/annotation_A_01..05.xlsx, annotation_B_01..05.xlsx
          data/annotation/benchmark/annotation/_private/pool_sources.csv   (HAI NGƯỜI KHÔNG MỞ khi gán nhãn)
          results/logs/log_B.md
Chạy    : python v9_pooling.py

Quy trình (8.1–8.4):
  - K: dual_encoder 10, vimfr 10; sáu baseline mỗi cái 5 (--skip-baselines để bỏ hệ thống nào đó, ghi vào log và mục 12). Hợp nhất, loại trùng theo product_id, thêm hạt giống.
  - Cổng G3: nếu N > 3.800 thì hạ K của 4 baseline xuống 3, rồi 2. KHÔNG hạ K của dual_encoder và vimfr.
  - Mỗi truy vấn xáo thứ tự cặp bằng seed (42 cho A, 43 cho B) + số thứ tự truy vấn.
  - A gán: overlap (Q086–Q100, Q186–Q200) rồi Q101–Q185; B gán: overlap rồi Q001–Q085; mỗi phần 85 truy vấn chia 4 file.
  - annotation_A_01 và annotation_B_01 chứa CÙNG các dòng nhưng THỨ TỰ khác nhau.
  - File không có cột lộ nguồn/hạt giống/thứ hạng/điểm/nhãn của người kia.
  - --reserve-r tạo thêm annotation_R_helper.xlsx (tập dự phòng R của bước 2b), chỉ giao khi kappa vòng 1 < 0,61.
LƯU Ý: file overlap có 30 truy vấn (> 25 truy vấn/file trong 8.4) vì kế hoạch dồn cả 30 truy vấn overlap vào *_01.
Ghi mục 12 nếu giữ nguyên (hoặc dùng --split-overlap để tách *_01 thành hai file 15 truy vấn).
"""

import argparse
import csv
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
try:
    import config
except Exception:
    config = None

RUNS_DIR    = ROOT / "results/runs"
QUERIES_CSV = ROOT / "data/annotation/benchmark/queries_200.csv"
TEST_PATH   = ROOT / "data/splits/test.jsonl"
ANN_DIR     = ROOT / "data/annotation/benchmark/annotation"
PRIVATE     = ANN_DIR / "_private"
LOG_DIR     = ROOT / "results/logs"

MAIN_K = {"dual_encoder": 10, "vimfr": 10}
ALL_BASELINES = ["tfidf", "bm25", "clip_vi2en", "bienc_text", "fashionclip_vi2en", "viclip_ot"]
BASELINES = list(ALL_BASELINES)   # có thể rút gọn bằng --skip-baselines
N_MAX = 3800
SECONDS_PER_PAIR = 6
DESC_LIMIT = 300
THUMB_PX = 160
OVERLAP = [(86, 100), (186, 200)]
B_OWN = (1, 85)          # B gán nhãn Q001–Q085
A_OWN = (101, 185)       # A gán nhãn Q101–Q185
SEEDS = {"A": 42, "B": 43}
HEADERS = ["query_id", "query_text", "attrs_in_query", "product_id", "shop", "category", "type", "name",
           "material", "colors", "description", "image", "label", "note"]
FORBIDDEN_COLS = {"seed", "hat_giong", "system", "source", "rank", "score", "tags", "tag", "nguon", "run"}


def _norm(text):
    """Chữ thường, NFC, gọn khoảng trắng (thay cho attr_utils.normalize, chỉ dùng để hiển thị)."""
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", str(text or "")).lower().split())


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).replace("|", ";").split(";") if v.strip()]


def qnum(qid):
    return int("".join(ch for ch in qid if ch.isdigit()))


def in_ranges(n, ranges):
    return any(a <= n <= b for a, b in ranges)


def load_run(path):
    """{qid: [product_id theo hạng tăng dần]}; bỏ product_id lặp (giữ hạng tốt nhất)."""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            p = line.split()
            if len(p) >= 6:
                out[p[0]].append((int(p[3]), n, p[2]))
    res = {}
    for q, v in out.items():
        seen, ranked = set(), []
        for _, _, pid in sorted(v):
            if pid not in seen:
                seen.add(pid)
                ranked.append(pid)
        res[q] = ranked
    return res


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def featured_image_path(product):
    images = product.get("images") or []
    if not images:
        return None
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None
    local_path = Path(str(lp).replace("\\", "/"))
    cands = [local_path, ROOT / local_path]
    if config is not None and not local_path.is_absolute():
        if getattr(config, "PROCESSED_DIR", None):
            cands.append(Path(config.PROCESSED_DIR) / local_path)
        if getattr(config, "IMAGE_DIR", None):
            cands.append(Path(config.IMAGE_DIR) / local_path.name)
    for c in cands:
        if c.exists():
            return c
    return None


# ══════════════════════════════════════════════════════════════════
def build_pool(runs, queries, seeds, k_base):
    """pool[qid] = {pid: {"sys": {tên: hạng}, "seed": bool}}"""
    ks = dict(MAIN_K)
    ks.update({b: k_base for b in BASELINES})
    pool = {}
    for q in queries:
        qid = q["query_id"]
        d = {}
        for s, k in ks.items():
            for rank, pid in enumerate(runs[s].get(qid, [])[:k], start=1):
                d.setdefault(pid, {"sys": {}, "seed": False})["sys"][s] = rank
        sp = seeds.get(qid)
        if sp:
            d.setdefault(sp, {"sys": {}, "seed": False})["seed"] = True
        pool[qid] = d
    return pool


def n_pairs(pool):
    return sum(len(v) for v in pool.values())


def make_thumb_cache():
    cache = {}

    def get(path):
        if path not in cache:
            try:
                from PIL import Image
                im = Image.open(path).convert("RGB")
                im.thumbnail((THUMB_PX, THUMB_PX))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
                cache[path] = buf.getvalue()
            except Exception:
                cache[path] = None
        return cache[path]
    return get


def write_annotation(path, rows, thumb):
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "annotation"
    ws.append(HEADERS)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    widths = {"query_id": 9, "query_text": 30, "attrs_in_query": 26, "product_id": 20, "shop": 12, "category": 10,
              "type": 14, "name": 38, "material": 14, "colors": 16, "description": 60, "image": 24, "label": 8, "note": 20}
    for i, h in enumerate(HEADERS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths[h]
    ws.freeze_panes = "E2"
    img_col = HEADERS.index("image") + 1
    keep = []
    n_noimg = 0
    for r, row in enumerate(rows, start=2):
        ws.append([row[h] if h != "image" else None for h in HEADERS])
        ws.row_dimensions[r].height = 125
        for c in ws[r]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        data = thumb(row["_img_path"]) if row["_img_path"] else None
        if data:
            buf = io.BytesIO(data)
            keep.append(buf)
            ws.add_image(XLImage(buf), f"{get_column_letter(img_col)}{r}")
        else:
            ws.cell(row=r, column=img_col, value="(không có ảnh)")
            n_noimg += 1
    dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Nhãn không hợp lệ", error="Chỉ nhận 0, 1 hoặc 2")
    ws.add_data_validation(dv)
    label_col = get_column_letter(HEADERS.index("label") + 1)
    dv.add(f"{label_col}2:{label_col}{len(rows) + 1}")
    wb.save(path)
    return n_noimg


def split_even(items, parts):
    base, extra = divmod(len(items), parts)
    out, i = [], 0
    for p in range(parts):
        size = base + (1 if p < extra else 0)
        out.append(items[i:i + size])
        i += size
    return out


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-overlap", action="store_true", help="Tách *_01 thành *_01a và *_01b (15 truy vấn mỗi file)")
    ap.add_argument("--reserve-r", action="store_true",
                    help="Tạo thêm annotation_R_helper.xlsx (tập dự phòng R: Q071–Q080 và Q171–Q180) cho NGƯỜI THỨ HAI; "
                         "chỉ giao nếu kappa vòng 1 < 0,61 (mục 9.6, bước 2b)")
    ap.add_argument("--skip-baselines", default="",
                    help="Danh sách baseline bỏ khỏi pool, cách nhau dấu phẩy (vd. viclip_ot nếu V7 không chạy được); "
                         "hệ thống bị bỏ sẽ KHÔNG được đánh giá công bằng, phải ghi mục 12")
    args = ap.parse_args()
    global BASELINES
    skip = {x.strip() for x in args.skip_baselines.split(",") if x.strip()}
    bad = skip - set(ALL_BASELINES)
    if bad:
        raise SystemExit(f"--skip-baselines có tên lạ: {sorted(bad)}; hợp lệ: {ALL_BASELINES}")
    BASELINES = [b for b in ALL_BASELINES if b not in skip]

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── nạp dữ liệu ──
    runs = {}
    for s in list(MAIN_K) + BASELINES:
        p = RUNS_DIR / f"{s}.run"
        if not p.exists():
            raise SystemExit(f"Thiếu {p}")
        runs[s] = load_run(p)
    with open(QUERIES_CSV, encoding="utf-8-sig", newline="") as f:
        queries = [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    queries.sort(key=lambda q: qnum(q["query_id"]))
    assert len(queries) == 200, f"queries_200.csv có {len(queries)} dòng"
    products = {p["id"]: p for p in load_jsonl(TEST_PATH)}
    seeds = {q["query_id"]: q.get("seed_id") for q in queries}
    for s, r in runs.items():
        assert set(r) == {q["query_id"] for q in queries}, f"{s}.run không đủ 200 truy vấn"
    missing_seed = [q for q, sp in seeds.items() if not sp or sp not in products]
    assert not missing_seed, f"Hạt giống thiếu hoặc không thuộc Test: {missing_seed[:5]}"
    unknown = {pid for r in runs.values() for v in r.values() for pid in v} - set(products)
    assert not unknown, f"{len(unknown)} product_id trong file run không thuộc test.jsonl, ví dụ {sorted(unknown)[:3]}"

    # ── cổng G3 ──
    print("=== Kích thước pool theo K của baseline ===")
    table = {}
    for k in (5, 3, 2):
        table[k] = n_pairs(build_pool(runs, queries, seeds, k))
        print(f"  K baseline = {k}: N = {table[k]} cặp"
              f"  → ước tính {table[k] * 0.575 * SECONDS_PER_PAIR / 3600:.1f} giờ mỗi người (hai người),"
              f" {table[k] * 1.15 * SECONDS_PER_PAIR / 3600:.1f} giờ nếu MỘT người gán cả hai phần")
    k_base = 5
    for k in (5, 3, 2):
        k_base = k
        if table[k] <= N_MAX:
            break
    pool = build_pool(runs, queries, seeds, k_base)
    N = n_pairs(pool)
    gate = "ĐẠT" if N <= N_MAX else f"VƯỢT {N_MAX} ngay cả khi K=2 (giữ K=2, chấp nhận kéo dài buổi gán nhãn)"
    print(f"\n→ K baseline cuối = {k_base}; N = {N}; cổng G3: {gate}")

    # ── thống kê pool ──
    sizes = Counter(len(v) for v in pool.values())
    seed_only = sum(1 for v in pool.values() for info in v.values() if info["seed"] and not info["sys"])
    seed_in_sys = sum(1 for v in pool.values() for info in v.values() if info["seed"] and info["sys"])
    print(f"  Cỡ pool mỗi truy vấn: nhỏ nhất {min(sizes)}, lớn nhất {max(sizes)}, trung bình {N / 200:.1f}")
    print(f"  Hạt giống đã nằm trong top-K của ít nhất một hệ thống: {seed_in_sys}/200; chỉ có nhờ được thêm vào: {seed_only}/200")

    # ── pool_sources.csv (private) ──
    with open(PRIVATE / "pool_sources.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "product_id", "systems_and_ranks", "is_seed"])
        for q in queries:
            for pid, info in sorted(pool[q["query_id"]].items()):
                w.writerow([q["query_id"], pid, "|".join(f"{s}:{r}" for s, r in sorted(info["sys"].items())),
                            int(info["seed"])])

    # ── dòng annotation theo truy vấn ──
    thumb = make_thumb_cache()
    qmeta = {q["query_id"]: q for q in queries}
    path_cache = {}

    def product_row(qid, pid):
        p = products[pid]
        if pid not in path_cache:
            path_cache[pid] = featured_image_path(p)
        return {
            "query_id": qid, "query_text": qmeta[qid]["query_text"], "attrs_in_query": qmeta[qid].get("attrs_in_query", ""),
            "product_id": pid, "shop": p.get("shop"), "category": _norm(p.get("category")),
            "type": _norm(p.get("type")), "name": p.get("name"),
            "material": "; ".join(_as_list(p.get("material"))), "colors": "; ".join(_as_list(p.get("colors"))),
            "description": (p.get("description") or "")[:DESC_LIMIT], "image": None, "label": None, "note": None,
            "_img_path": path_cache[pid],
        }

    def rows_for(person, qids):
        rows = []
        for qid in qids:
            pids = sorted(pool[qid])
            random.Random(SEEDS[person] + qnum(qid)).shuffle(pids)
            rows.extend(product_row(qid, pid) for pid in pids)
        return rows

    all_q = [q["query_id"] for q in queries]
    overlap_q = [q for q in all_q if in_ranges(qnum(q), OVERLAP)]
    own = {"A": [q for q in all_q if A_OWN[0] <= qnum(q) <= A_OWN[1]],
           "B": [q for q in all_q if B_OWN[0] <= qnum(q) <= B_OWN[1]]}
    assert len(overlap_q) == 30 and len(own["A"]) == 85 and len(own["B"]) == 85

    plan = {}                                    # tên file -> (người, danh sách query_id)
    for person in ("A", "B"):
        if args.split_overlap:
            half = split_even(overlap_q, 2)
            plan[f"annotation_{person}_01a"] = (person, half[0])
            plan[f"annotation_{person}_01b"] = (person, half[1])
        else:
            plan[f"annotation_{person}_01"] = (person, overlap_q)
        for i, part in enumerate(split_even(own[person], 4), start=2):
            plan[f"annotation_{person}_{i:02d}"] = (person, part)

    if args.reserve_r:
        r_q = [q for q in all_q if 71 <= qnum(q) <= 80 or 171 <= qnum(q) <= 180]
        plan["annotation_R_helper"] = ("B", r_q)

    print("\n=== Xuất file annotation ===")
    summary, pair_sets, order_of = [], {}, {}
    seen_pairs = defaultdict(Counter)
    for name, (person, qids) in plan.items():
        rows = rows_for(person, qids)
        n_noimg = write_annotation(ANN_DIR / f"{name}.xlsx", rows, thumb)
        keys = [(r["query_id"], r["product_id"]) for r in rows]
        assert len(keys) == len(set(keys)), f"{name}: có (query_id, product_id) lặp"
        pair_sets[name], order_of[name] = set(keys), keys
        for k in keys:
            seen_pairs[person][k] += 1
        summary.append((name, len(qids), len(rows), n_noimg))
        print(f"  {name}.xlsx: {len(qids)} truy vấn, {len(rows)} cặp, thiếu ảnh {n_noimg}")

    # ── kiểm tra 8.5 ──
    assert not (set(h.lower() for h in HEADERS) & FORBIDDEN_COLS), "Có cột làm lộ nguồn/hạt giống"
    if not args.split_overlap:
        a1, b1 = "annotation_A_01", "annotation_B_01"
        assert pair_sets[a1] == pair_sets[b1], "A_01 và B_01 phải chứa cùng các dòng"
        same_pos = sum(1 for x, y in zip(order_of[a1], order_of[b1]) if x == y) / len(order_of[a1])
        print(f"  A_01 và B_01: cùng {len(pair_sets[a1])} dòng; tỉ lệ vị trí giống nhau {same_pos:.1%} (phải thấp)")
        assert same_pos < 0.5, "Thứ tự A_01 và B_01 quá giống nhau"
    for person in ("A", "B"):
        assert all(v >= 1 for v in seen_pairs[person].values())
    total_rows = {p: sum(r for name, q, r, _ in summary if plan[name][0] == p and not name.startswith("annotation_R"))
                  for p in ("A", "B")}
    print(f"  Tổng cặp phải gán: A {total_rows['A']}, B {total_rows['B']}")
    no_img = sum(1 for p in path_cache.values() if p is None)
    print(f"  Sản phẩm trong pool không có file ảnh: {no_img}/{len(path_cache)}")

    # ── log ──
    over_pairs = sum(len(pool[q]) for q in overlap_q)
    log = ["\n## V9 — pooling",
           f"- hệ thống trong pool: dual_encoder, vimfr + {len(BASELINES)} baseline ({', '.join(BASELINES)})" + (f"; BỎ: {', '.join(sorted(skip))}" if skip else ""),
           f"- K: dual_encoder 10, vimfr 10, baseline {k_base} (mỗi baseline); thêm hạt giống; N = {N} (ngưỡng {N_MAX}: {gate})",
           f"- N theo K baseline: " + ", ".join(f"K={k}: {v}" for k, v in table.items()),
           f"- cỡ pool mỗi truy vấn: min {min(sizes)}, max {max(sizes)}, TB {N / 200:.1f}; cặp thuộc overlap: {over_pairs}",
           f"- hạt giống đã nằm trong top-K của hệ thống: {seed_in_sys}/200; chỉ có nhờ thêm: {seed_only}/200",
           f"- file annotation: " + "; ".join(f"{n}: {q} truy vấn/{r} cặp" for n, q, r, _ in summary),
           f"- ước tính công gán nhãn: {N * 0.575 * SECONDS_PER_PAIR / 3600:.1f} giờ mỗi người (hai người); "
           f"{(N + over_pairs) * SECONDS_PER_PAIR / 3600:.1f} giờ nếu một người gán cả hai phần",
           "- Hệ quả cần ghi vào báo cáo: sản phẩm trong top-10 của baseline nhưng ngoài top-K pool và ngoài pool của mọi hệ thống khác "
           "bị coi là không liên quan, nên chỉ số baseline có thể bị đánh giá thấp (có lợi cho ViMFR); V11 báo thêm judged@10 để đo mức này."]
    if not args.split_overlap:
        log.append("- file *_01 có 30 truy vấn (> 25 truy vấn/file ở mục 8.4): ghi mục 12 hoặc dùng --split-overlap")
    with open(LOG_DIR / "log_B.md", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print(f"\nĐã lưu vào {ANN_DIR}. Không mở {PRIVATE / 'pool_sources.csv'} khi gán nhãn.")
    print("V9 hoàn thành. Mở thử một file: ảnh hiển thị đúng, cột label chỉ nhận 0/1/2 (8.5).")


if __name__ == "__main__":
    main()