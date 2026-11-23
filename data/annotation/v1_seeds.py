"""
V1: Chọn hạt giống truy vấn kiểm thử (kế hoạch mục 4.1–4.2)
Đầu vào : data/splits/test.jsonl, data/annotation/benchmark/bad_products.txt (từ v0_bad_products.py), attr_utils.py
Đầu ra  : data/annotation/benchmark/seeds_A.xlsx, seeds_B.xlsx      (mỗi file 100 slot chính + hạt giống dự phòng)
          data/annotation/benchmark/_private/seeds_meta.csv          (siêu dữ liệu để kiểm tra; không mở khi viết/gán nhãn)
          results/logs/log_B.md                                       (tham số thật, kết quả, cảnh báo)
Chạy    : python v1_seeds.py

Quy trình:
  1. Ứng viên = test.jsonl trừ bad_products.txt. Với mỗi sản phẩm, xác định các nhóm thuộc tính mà người viết CÓ THỂ
     xác nhận trên thẻ hạt giống, bằng đúng bộ trích của V3 (attr_utils.py):
       màu (colors có ít nhất một màu chuẩn hóa được), chất liệu (material), kiểu dáng, phong cách
       (kiểu dáng/phong cách dò trong name + 300 ký tự đầu của description, vì thẻ chỉ hiện 300 ký tự).
  2. n_target = 1/2/≥3 cần ít nhất 1/2/3 nhóm; n_target ≥ 3 chỉ cho Áo/Quần (và kiểu dáng chỉ dò cho Áo/Quần như V3).
  3. Chia lớp (danh mục × nhóm shop), lấy mẫu ngẫu nhiên không hoàn lại (seed 42) cho từng người, rồi thêm 20% dự phòng
     cho mỗi lớp. Danh sách A và B không có sản phẩm chung.
Chỉ tiêu mỗi người (4.1): 100 truy vấn; n_attrs 35/35/30; Áo/Quần/Phụ kiện 40/30/30; ≥30 hạt giống nội địa.
Đổi chỉ tiêu thì ghi mục 12 TRƯỚC 09:40.
"""

import csv
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # để import attr_utils cùng thư mục
import attr_utils as au

try:
    import config
except ImportError:
    config = None

SEED = 42
TEST_PATH = ROOT / "data/splits/test.jsonl"
BENCH_DIR = ROOT / "data/annotation/benchmark"
PRIVATE   = BENCH_DIR / "_private"
LOG_DIR   = ROOT / "results/logs"

DESC_LIMIT      = 300      # thẻ hạt giống chỉ hiện 300 ký tự mô tả
THUMB_PX        = 160
RESERVE_RATE    = 0.20
DOMESTIC_SHARE  = 0.35     # mục tiêu tỉ lệ nội địa mỗi ô (chỉ tiêu tối thiểu là 30%)
MIN_DOMESTIC    = 30       # mỗi người
DOMESTIC = {"aristino", "johnhenry", "henstore", "icondenim"}

# Số slot mỗi người theo (danh mục, n_target). Biên: n_target 35/35/30; Áo/Quần/Phụ kiện 40/30/30; n_target 3 chỉ Áo/Quần.
CELLS = {
    ("áo", 1): 11, ("áo", 2): 11, ("áo", 3): 18,
    ("quần", 1): 9, ("quần", 2): 9, ("quần", 3): 12,
    ("phụ kiện", 1): 15, ("phụ kiện", 2): 15,
}
CATEGORIES = ["áo", "quần", "phụ kiện"]
PERSONS = ["A", "B"]
SLOT_START = {"A": 1, "B": 101}        # A: Q001–Q100, B: Q101–Q200


# ══════════════════════════════════════════════════════════════════
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
    local_path = Path(lp)
    candidates = [local_path, ROOT / local_path]
    if config is not None and getattr(config, "PROCESSED_DIR", None) and not local_path.is_absolute():
        candidates.append(Path(config.PROCESSED_DIR) / local_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def shop_group(pid):
    return "noi_dia" if pid.split("_")[0] in DOMESTIC else "san_tmdt"


def seed_groups(p):
    """Các nhóm thuộc tính người viết có thể xác nhận trên thẻ; trả về {nhóm: [giá trị chuẩn]} chỉ gồm nhóm có giá trị."""
    name = au.normalize(p.get("name"))
    desc = au.normalize((p.get("description") or "")[:DESC_LIMIT])
    cat = au.normalize(p.get("category"))
    loai = au.map_type(p.get("type", ""), p.get("name", ""))
    g = {}

    colors = sorted({c for c in (au.map_color(x) for x in au.as_list(p.get("colors"))) if c and c not in loai})
    if colors:
        g["MAU"] = colors
    mats = []
    for m in au.as_list(p.get("material")):
        m = au.normalize(m)
        if m and m not in mats and m not in loai:
            mats.append(m)
    if mats:
        g["CHAT_LIEU"] = mats
    if cat in au.KIEU_DANG_CATEGORIES:
        kd = [v for v in au.detect_kieu_dang(name, desc) if v not in loai]
        if kd:
            g["KIEU_DANG"] = kd
    pc = [v for v in au.detect_phong_cach(name, desc) if v not in loai]
    if pc:
        g["PHONG_CACH"] = pc
    return g


def eligible_for(info, nt):
    """Có đủ nhóm cho n_target = nt không (n_target ≥ 3 chỉ Áo/Quần)."""
    if nt >= 3 and info["category"] not in ("áo", "quần"):
        return False
    return len(info["groups"]) >= nt


# ══════════════════════════════════════════════════════════════════
def main():
    bad_path = BENCH_DIR / "bad_products.txt"
    if not bad_path.exists():
        raise SystemExit(f"Chưa có {bad_path}. Chạy v0_bad_products.py trước.")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    bad = {line.strip() for line in open(bad_path, encoding="utf-8") if line.strip()}
    test = load_jsonl(TEST_PATH)
    test_ids = {p["id"] for p in test}
    print(f"Test: {len(test)} | bad_products: {len(bad)} | ứng viên: {len(test) - len(bad & test_ids)}")

    infos = {}
    for p in sorted(test, key=lambda x: x["id"]):
        if p["id"] in bad:
            continue
        if not (p.get("description") or "").strip() or featured_image_path(p) is None:
            continue                                   # an toàn kép: V0 đã loại, nhưng kiểm tra lại
        g = seed_groups(p)
        if not g:
            continue
        infos[p["id"]] = {
            "product": p, "category": au.normalize(p.get("category")), "sg": shop_group(p["id"]),
            "groups": g, "n_groups": len(g),
            "loai": au.map_type(p.get("type", ""), p.get("name", "")),
        }
    print(f"Ứng viên có ít nhất 1 nhóm thuộc tính xác nhận được: {len(infos)}")

    # bảng khả năng cung cấp trước khi lấy mẫu
    print("\n=== Số ứng viên đủ điều kiện (danh mục × n_target ≥ × nhóm shop) ===")
    print(f"  {'':<10} {'nt≥1 nội địa/tmđt':>20} {'nt≥2':>16} {'nt≥3':>16}")
    for cat in CATEGORIES:
        row = []
        for nt in (1, 2, 3):
            d = sum(1 for i in infos.values() if i["category"] == cat and i["sg"] == "noi_dia" and eligible_for(i, nt))
            m = sum(1 for i in infos.values() if i["category"] == cat and i["sg"] == "san_tmdt" and eligible_for(i, nt))
            row.append(f"{d}/{m}")
        print(f"  {cat:<10} {row[0]:>20} {row[1]:>16} {row[2]:>16}")

    rng = random.Random(SEED)
    remaining = dict(infos)
    chosen = {p: [] for p in PERSONS}           # danh sách slot chính
    warnings = []

    def candidates(cat, nt, sg):
        return sorted(pid for pid, i in remaining.items()
                      if i["category"] == cat and i["sg"] == sg and eligible_for(i, nt))

    # Giai đoạn 1: slot chính, n_target giảm dần (khan hiếm trước), xen kẽ A/B trong từng ô
    for nt in (3, 2, 1):
        for cat in CATEGORIES:
            size = CELLS.get((cat, nt), 0)
            if not size:
                continue
            for person in PERSONS:
                dom_pool, mkt_pool = candidates(cat, nt, "noi_dia"), candidates(cat, nt, "san_tmdt")
                k_dom = min(math.ceil(DOMESTIC_SHARE * size), len(dom_pool))
                k_mkt = size - k_dom
                if len(mkt_pool) < k_mkt:
                    k_mkt = len(mkt_pool)
                    k_dom = min(size - k_mkt, len(dom_pool))
                short = size - k_dom - k_mkt
                if short > 0:
                    warnings.append(f"THIẾU {short} hạt giống cho người {person}, ô ({cat}, n_target={nt})")
                picked = [(pid, "noi_dia") for pid in rng.sample(dom_pool, k_dom)] + \
                         [(pid, "san_tmdt") for pid in rng.sample(mkt_pool, k_mkt)]
                for pid, sg in picked:
                    chosen[person].append({"pid": pid, "nt": nt, "cat": cat, "sg": sg, "reserve": 0})
                    del remaining[pid]

    # Giai đoạn 2: dự phòng 20% mỗi lớp (cùng danh mục, cùng n_target, cùng nhóm shop)
    for person in PERSONS:
        strata = Counter((s["cat"], s["nt"], s["sg"]) for s in chosen[person] if not s["reserve"])
        for (cat, nt, sg), k in sorted(strata.items()):
            r = math.ceil(RESERVE_RATE * k)
            pool = candidates(cat, nt, sg)
            take = min(r, len(pool))
            if take < r:
                warnings.append(f"Dự phòng thiếu {r - take} cho người {person}, lớp ({cat}, n_target={nt}, {sg})")
            for pid in rng.sample(pool, take):
                chosen[person].append({"pid": pid, "nt": nt, "cat": cat, "sg": sg, "reserve": 1})
                del remaining[pid]

    # Giai đoạn 3: xáo thứ tự slot chính, gán slot_id
    for idx, person in enumerate(PERSONS):
        mains = [s for s in chosen[person] if not s["reserve"]]
        reserves = sorted([s for s in chosen[person] if s["reserve"]], key=lambda s: (s["cat"], s["nt"], s["sg"], s["pid"]))
        random.Random(SEED + idx).shuffle(mains)
        for j, s in enumerate(mains):
            s["slot_id"] = f"Q{SLOT_START[person] + j:03d}"
        for j, s in enumerate(reserves, start=1):
            s["slot_id"] = f"R{person}{j:02d}"
        chosen[person] = mains + reserves

    # ── Kiểm tra ──
    print("\n=== Kiểm tra chỉ tiêu (mỗi người, chỉ slot chính) ===")
    errors = []
    for person in PERSONS:
        mains = [s for s in chosen[person] if not s["reserve"]]
        nt_c = Counter(min(s["nt"], 3) for s in mains)
        cat_c = Counter(s["cat"] for s in mains)
        dom = sum(1 for s in mains if s["sg"] == "noi_dia")
        print(f"  {person}: {len(mains)} slot | n_target {dict(sorted(nt_c.items()))} | danh mục {dict(cat_c)} | "
              f"nội địa {dom} | dự phòng {sum(1 for s in chosen[person] if s['reserve'])}")
        if len(mains) != 100:
            errors.append(f"{person}: chỉ có {len(mains)}/100 slot")
        if dom < MIN_DOMESTIC:
            errors.append(f"{person}: nội địa {dom} < {MIN_DOMESTIC}")
    all_pids = [s["pid"] for p in PERSONS for s in chosen[p]]
    assert len(all_pids) == len(set(all_pids)), "Có sản phẩm bị chọn hai lần"
    assert set(all_pids) <= test_ids and not (set(all_pids) & bad), "Hạt giống không thuộc Test hoặc nằm trong bad_products"
    for w in warnings:
        print("  [CẢNH BÁO]", w)
    for e in errors:
        print("  [LỖI CHỈ TIÊU]", e)

    # ── Xuất xlsx ──
    write_xlsx(chosen, infos)
    with open(PRIVATE / "seeds_meta.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "slot_id", "seed_id", "reserve", "category", "n_target", "shop_group", "n_groups",
                    "groups", "raw_type", "loai"])
        for person in PERSONS:
            for s in chosen[person]:
                i = infos[s["pid"]]
                w.writerow([person, s["slot_id"], s["pid"], s["reserve"], s["cat"], s["nt"], s["sg"], i["n_groups"],
                            json.dumps(i["groups"], ensure_ascii=False), i["product"].get("type"), i["loai"]])

    log = ["\n## V1 — chọn hạt giống",
           f"- seed={SEED}; DESC_LIMIT={DESC_LIMIT}; DOMESTIC_SHARE={DOMESTIC_SHARE}; RESERVE_RATE={RESERVE_RATE}",
           f"- ứng viên: {len(infos)}/{len(test)} (bad_products {len(bad)})",
           f"- ô chỉ tiêu mỗi người: {', '.join(f'{k[0]}/n{k[1]}={v}' for k, v in CELLS.items())}"]
    for person in PERSONS:
        mains = [s for s in chosen[person] if not s["reserve"]]
        log.append(f"- {person}: {len(mains)} slot, nội địa {sum(1 for s in mains if s['sg'] == 'noi_dia')}, "
                   f"dự phòng {sum(1 for s in chosen[person] if s['reserve'])}")
    log += [f"- CẢNH BÁO: {w}" for w in warnings] + [f"- LỖI: {e}" for e in errors]
    with open(LOG_DIR / "log_B.md", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print(f"\nĐã lưu: {BENCH_DIR / 'seeds_A.xlsx'}, {BENCH_DIR / 'seeds_B.xlsx'}")
    print("V1 hoàn thành." if not errors else "V1 xong NHƯNG chưa đủ chỉ tiêu: ghi mục 12 hoặc bổ sung trước khi viết truy vấn.")


# ══════════════════════════════════════════════════════════════════
HEADERS = ["slot_id", "seed_id", "shop", "category", "type", "name", "material", "colors", "description", "image",
           "n_target", "reserve", "query_text", "attrs_in_query", "skipped_reason"]
RULES = [
    "Mỗi slot một truy vấn; không tự chọn hạt giống. Hạt giống không dùng được: lấy dự phòng đầu tiên cùng danh mục "
    "và cùng n_target, ghi lý do vào skipped_reason.",
    "Tiếng Việt tự nhiên, 2–12 từ. Bắt buộc có loại sản phẩm và ĐÚNG n_target thuộc tính ngoài loại.",
    "Mỗi thuộc tính phải xác nhận trên thẻ: chất liệu từ material/description; kiểu dáng, phong cách từ description/name; "
    "màu phải có trong colors và nhìn thấy trong ảnh đại diện. Mỗi nhóm tối đa một giá trị.",
    "CẤM: chép tên sản phẩm hoặc cụm ≥ 4 từ liên tiếp trong name; mã sản phẩm; tên thương hiệu/shop; giá; size; từ khuyến mãi.",
    "Tối đa 40% truy vấn theo thứ tự loại, màu, thuộc tính khác. Ít nhất 30% dùng cách nói đồng nghĩa có trong "
    "attr_dictionary.xlsx (vd 'ôm sát' thay cho 'slim fit').",
    "Không xem: truy vấn của người kia, kết quả của hệ thống nào, truy vấn giả (pseudo_*.csv). Không dùng công cụ gợi ý hay mô hình ngôn ngữ lớn.",
    "attrs_in_query: dạng nhóm=giá trị, ngăn cách bằng ';', ví dụ 'màu=xanh navy; kiểu dáng=regular fit'. Không ghi loại sản phẩm.",
    "Mục tiêu tốc độ: không quá 55 giây mỗi truy vấn.",
]


def write_xlsx(chosen, infos):
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None
        print("  [CẢNH BÁO] Chưa cài Pillow (pip install pillow): file xlsx sẽ không có ảnh.")

    widths = {"slot_id": 9, "seed_id": 20, "shop": 12, "category": 10, "type": 14, "name": 38, "material": 14,
              "colors": 16, "description": 60, "image": 24, "n_target": 9, "reserve": 9, "query_text": 38,
              "attrs_in_query": 30, "skipped_reason": 22}
    keep_alive = []
    for person in PERSONS:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "seeds"
        ws.append(HEADERS)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="DDEBF7")
        for i, h in enumerate(HEADERS, start=1):
            ws.column_dimensions[get_column_letter(i)].width = widths[h]
        ws.freeze_panes = "C2"
        img_col = HEADERS.index("image") + 1

        for r, s in enumerate(chosen[person], start=2):
            info = infos[s["pid"]]
            p = info["product"]
            ws.append([s["slot_id"], s["pid"], p.get("shop"), s["cat"], info["loai"], p.get("name"),
                       "; ".join(au.as_list(p.get("material"))), "; ".join(au.as_list(p.get("colors"))),
                       (p.get("description") or "")[:DESC_LIMIT], None, s["nt"] if s["nt"] < 3 else "3+",
                       s["reserve"], None, None, None])
            ws.row_dimensions[r].height = 125
            for c in ws[r]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
            if s["reserve"]:
                for c in ws[r]:
                    c.fill = PatternFill("solid", fgColor="FFF2CC")
            path = featured_image_path(p)
            if PILImage is not None and path is not None:
                try:
                    im = PILImage.open(path).convert("RGB")
                    im.thumbnail((THUMB_PX, THUMB_PX))
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    buf.seek(0)
                    xl = XLImage(buf)
                    keep_alive.append(buf)
                    ws.add_image(xl, f"{get_column_letter(img_col)}{r}")
                except Exception as e:                   # ảnh hỏng: ghi chú vào ô thay vì dừng
                    ws.cell(row=r, column=img_col, value=f"(lỗi ảnh: {e.__class__.__name__})")

        ws2 = wb.create_sheet("huong_dan")
        ws2.column_dimensions["A"].width = 130
        ws2.append(["Quy tắc viết truy vấn (kế hoạch mục 4.3). Dòng nền vàng là hạt giống dự phòng."])
        ws2["A1"].font = Font(bold=True)
        for rule in RULES:
            ws2.append([rule])
            ws2.cell(row=ws2.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        ws2.append([f"Slot chính của người {person}: Q{SLOT_START[person]:03d}–Q{SLOT_START[person] + 99:03d}. "
                    "Overlap (cả hai cùng gán nhãn): Q086–Q100 và Q186–Q200."])
        wb.save(BENCH_DIR / f"seeds_{person}.xlsx")


if __name__ == "__main__":
    main()