"""Kiểm định dữ liệu đầu vào của mô hình truy hồi ảnh ↔ text (ViMFD).

Luồng dữ liệu:
    common.py (từng nguồn)  -> data/processed/{source}_clean.jsonl + images/ (224x224)
    build_splits.py         -> data/splits/{train,val,test}.jsonl + split.json

Mô hình thực sự nhận:
    ảnh : file 224x224 RGB (giữ tỉ lệ, đệm trắng) tại <image-root>/<local_path>
    text: `text_seg` = tên + "Màu ..." + câu mô tả, lowercase, đã tách từ VnCoreNLP;
          PhoBERT cắt ở MAX_TEXT_LENGTH token
    nhãn: các ảnh và text của CÙNG một sản phẩm là cặp positive; sản phẩm khác trong batch
          là negative (InfoNCE) nên "anh em" gần giống nhau là negative sai.

Vì vậy script chỉ kiểm tra những gì ảnh hưởng đến việc train và đánh giá:
    1. Hợp đồng dữ liệu : field, id, ảnh/position/local_path, text_group khớp text, khớp split.json
    2. Split            : tỉ lệ, phân bố (nguồn x category), rò rỉ theo id / text_group / URL ảnh /
                          nhóm anh em (đúng các khoá build_splits.py dùng để gom)
    3. Ảnh              : file có đọc được, đúng 224x224 RGB, đúng định dạng, không trắng, sản phẩm
                          không quá nhỏ; thống kê mean/std để chuẩn hoá
    4. Rò rỉ theo ảnh   : pHash tính TỪ FILE ẢNH (common.py không còn xuất phash) để bắt ảnh trùng /
                          gần trùng giữa các split, ảnh trùng trong cùng split
    5. Text             : độ dài token PhoBERT, text_seg khớp text, rác còn sót, câu lặp (boilerplate),
                          text chỉ có tên, thiếu màu
    6. Nhãn và cặp      : category so với `type`/tên, số ảnh mỗi sản phẩm (mất cân bằng positive)
    7. Giá trị retrieval: số query, gallery, cụm anh em (trần R@1), negative sai trong batch

Chạy:
    python data_validation.py
    python data_validation.py --splits-dir data/splits --image-root data/processed
    python data_validation.py --skip-images            # bỏ qua đọc ảnh (mục 3, 4)
    python data_validation.py --max-images 3000        # chỉ đọc mẫu N ảnh mỗi split
    python data_validation.py --no-tokenizer           # không nạp PhoBERT, ước lượng độ dài token

Kết quả: màn hình + <out-dir>/validation_report.json + <out-dir>/validation_flags.csv
Mã thoát: 1 nếu có mục FAIL, ngược lại 0.
Cần: numpy, Pillow. Tuỳ chọn: tqdm, transformers (đo token chính xác bằng tokenizer PhoBERT).
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # chạy được khi không có tqdm
    def tqdm(it, **_):
        return it

SPLITS = ("train", "val", "test")
EXPECTED_RATIO = {"train": 0.80, "val": 0.10, "test": 0.10}
CATEGORIES = {"áo", "quần", "phụ kiện"}
REQUIRED = ("id", "name", "source", "category", "text", "text_seg", "text_group", "images")
FORMAT_OF_EXT = {".jpg": "JPEG", ".png": "PNG"}   # common.py chỉ xuất .jpg / .png
MAX_SEG_SYLLABLES = 300                            # giống common.py (segment cắt ở 300 âm tiết)

# ---- Ngưỡng cảnh báo ----
RATIO_TOL = 0.015          # lệch tỉ lệ split so với 80/10/10
STRATUM_TOL = 0.05         # lệch tỉ lệ của từng (nguồn, category) so với 80/10/10
MIN_STRATUM_EVAL = 20      # số sản phẩm tối thiểu của mỗi (nguồn, category) trong val/test
NEAR_DUP_WARN, NEAR_DUP_FAIL = 0.005, 0.03   # tỉ lệ sản phẩm val/test có ảnh gần trùng với split trước
NAME_LEAK_WARN = 0.10      # tên (bỏ số/dấu) trùng với split trước; chỉ WARN vì tên chung chung hay trùng
TIE_WARN = 0.02            # sản phẩm có ảnh / text trùng hệt sản phẩm khác cùng split
TRUNC_WARN = 0.10          # tỉ lệ text bị PhoBERT cắt
SHORT_TEXT_WORDS = 5
NAME_ONLY_WARN = 0.25      # tỉ lệ text chỉ có tên + màu (không có mô tả)
NO_COLOR_WARN = 0.15       # có field colors nhưng text không có màu
FLAT_WARN = 0.005
TINY_CONTENT_WARN = 0.05   # ảnh mà sản phẩm chiếm < 10% khung hình
HARD_JUNK_WARN = 0.01      # HTML, URL, SĐT, emoji, ký tự ngoại ngữ, quảng cáo
SOFT_JUNK_WARN = 0.05      # hướng dẫn giặt ủi, thông số cỡ/cân nặng, mã sản phẩm trong mô tả
BOILER_DF = 0.02           # câu xuất hiện trong ≥2% text train = boilerplate
BOILER_WARN = 0.10
SUSPECT_CAT_WARN = 0.02
FN_BATCH_WARN = 0.02       # % sản phẩm có ≥1 negative sai trong batch
MIN_EVAL_QUERIES = 200
MIN_GALLERY = 500

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://|www\.", re.I)
_RE_PHONE = re.compile(r"(?<![\dA-Za-z])0\d{8,10}(?![\dA-Za-z])")
_RE_EMOJI = re.compile("[☀-➿\U0001F000-\U0001FAFF️‍]")
_RE_FOREIGN = re.compile("[一-鿿぀-ヿ฀-๿가-힯]")
_RE_PROMO = re.compile(r"free ?ship|miễn phí vận chuyển|voucher|khuyến mãi|giảm giá|inbox|hotline|zalo|fanpage"
                       r"|liên hệ|đặt hàng|đổi trả|hoàn tiền|cam kết", re.I)
_RE_CARE = re.compile(r"(?<!\w)(?:là\s*/?\s*ủi|ủi|phơi|nước giặt|thuốc tẩy|giặt (?:tay|máy|nhẹ|riêng))(?!\w)", re.I)
_RE_SIZE = re.compile(r"\d+\s*[-–]\s*\d+\s*kg|\bbảng size\b|\bcao\s*\d[\.,]\d", re.I)
_RE_CODE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{5,}\b")
_NOT_CODE = re.compile(r"(?:UV|SPF)\d+|\d+-?\d*(?:GSM|INCH|PCS|KG|CM|ML|MM|G)|\d-WAY|\d+-\d+|\d?X{0,3}[SML]|\d?XL", re.I)
_RE_NONWORD = re.compile(r"[\W\d_]+", re.U)

# Giữ đồng bộ với build_splits.py (khoá gom nhóm "anh em")
_SKU_KEY = re.compile(r"\b(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{4,}\b")

ACCESSORY_WORDS = (
    "ví", "túi", "balo", "ba lô", "cặp", "thắt lưng", "dây nịt", "nón", "mũ", "kính", "đồng hồ",
    "giày", "dép", "tất", "vớ", "cà vạt", "dây chuyền", "nhẫn", "vòng tay", "khăn", "găng tay",
    "móc khóa", "nước hoa",
)


def _kw(words):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)")


KW_AO, KW_QUAN, KW_ACC = _kw(("áo",)), _kw(("quần",)), _kw(ACCESSORY_WORDS)


# ---------------------------------------------------------------- tiện ích
class Report:
    def __init__(self):
        self.checks, self.flags, self.data, self.section = [], [], {}, ""

    def check(self, name, value, status, note=""):
        self.checks.append({"section": self.section, "name": name, "value": value, "status": status, "note": note})

    def flag(self, split, pid, issue, detail=""):
        self.flags.append({"split": split, "id": pid, "issue": issue, "detail": detail})


def grade(rate, warn, fail=None):
    if fail is not None and rate > fail:
        return "FAIL"
    return "WARN" if rate > warn else "PASS"


def must_be_zero(n):
    return "FAIL" if n > 0 else "PASS"


def pct(a, b):
    return a / b if b else 0.0


def fp(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def pcts(values, qs=(50, 90, 95, 99)):
    if len(values) == 0:
        return {**{f"p{q}": 0.0 for q in qs}, "max": 0.0}
    a = np.asarray(values, dtype=float)
    return {**{f"p{q}": float(np.percentile(a, q)) for q in qs}, "max": float(a.max())}


def head(rep, title):
    rep.section = title
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max([len(str(h))] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).rjust(w[i]) if i else str(h).ljust(w[i]) for i, h in enumerate(headers)))
    for r in rows:
        print("  " + "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)))


def norm_name(s):
    """Tên bỏ số và dấu câu: 'Áo polo 2024 ABC' -> 'áo polo abc'."""
    return _RE_NONWORD.sub(" ", unicodedata.normalize("NFC", str(s or "").casefold())).strip()


def fold(s):
    """Bỏ dấu + khoảng trắng + '_': so text với text_seg (VnCoreNLP chuẩn hoá vị trí dấu: họa -> hoạ)."""
    s = unicodedata.normalize("NFD", str(s or "").casefold().replace("đ", "d"))
    return re.sub(r"[\W_]+", "", "".join(c for c in s if not unicodedata.combining(c)))


def name_no_sku(name):
    return re.sub(r"\s+", " ", _SKU_KEY.sub("", name or "")).strip().casefold()


def sib_key(r):
    """Khoá nhóm anh em, giống build_splits.build_groups: (nguồn | shop | tên bỏ SKU)."""
    nsk = name_no_sku(r.get("name", ""))
    return f"{r.get('source')}|{r.get('shop', '')}|{nsk}" if nsk else None


def calc_text_group(text):
    """Giống common.text_group."""
    return hashlib.md5(re.sub(r"\W+", " ", text.casefold()).strip().encode()).hexdigest()[:16]


def split_text(r):
    """text = tên [. Màu ...] [. câu mô tả ...] (xem common.build_text) -> (tên, phần sau tên)."""
    t, name = str(r.get("text") or ""), str(r.get("name") or "").rstrip(".!?… ")
    return name, (t[len(name):] if t.startswith(name) else t).lstrip(". ")


def text_parts(r):
    """(có đoạn màu, số từ mô tả không tính tên và màu)."""
    _, body = split_text(r)
    ps = body.split(". ")
    has_color = bool(ps) and ps[0].startswith("Màu ")
    extra = len(body.split()) - (len(ps[0].split()) if has_color else 0)
    return has_color, max(extra, 0)


class UnionFind:
    def __init__(self):
        self._p = {}

    def find(self, x):
        self._p.setdefault(x, x)
        root = x
        while self._p[root] != root:
            root = self._p[root]
        while self._p[x] != root:
            self._p[x], x = root, self._p[x]
        return root

    def union(self, a, b):
        self._p[self.find(a)] = self.find(b)


def load_data(args):
    out = {}
    for s in SPLITS:
        path = Path(args.splits_dir) / f"{s}.jsonl"
        if not path.exists():
            sys.exit(f"Không tìm thấy {path}. Hãy chạy build_splits.py trước.")
        rows = []
        with path.open(encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        sys.exit(f"{path}, dòng {ln}: JSON lỗi ({e})")
        out[s] = rows
    man_path = Path(args.manifest) if args.manifest else Path(args.splits_dir) / "split.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    return out, manifest


# ---------------------------------------------------------------- 1. hợp đồng dữ liệu
def sec_schema(data, manifest, rep):
    head(rep, "1. HỢP ĐỒNG DỮ LIỆU (schema common.py / build_splits.py cam kết)")
    ids, paths = Counter(), Counter()
    rows_out, tot = [], Counter()
    for s, rows in data.items():
        c = Counter()
        for r in rows:
            pid = r.get("id")
            ids[str(pid)] += 1
            for k in REQUIRED:
                v = r.get(k)
                if v is None or (isinstance(v, (str, list)) and len(v) == 0):
                    c["missing"] += 1
                    rep.flag(s, pid, f"thieu_{k}")
            if r.get("category") not in CATEGORIES:
                c["bad_cat"] += 1
                rep.flag(s, pid, "category_la", str(r.get("category")))
            if not r.get("shop"):
                c["no_shop"] += 1
            imgs = r.get("images") if isinstance(r.get("images"), list) else []
            bad_meta = False
            for im in imgs:
                lp = im.get("local_path") if isinstance(im, dict) else None
                if not lp or not im.get("url") or not lp.startswith("images/") or Path(lp).suffix.lower() not in FORMAT_OF_EXT:
                    bad_meta = True
                else:
                    paths[lp] += 1
            if bad_meta:
                c["bad_img_meta"] += 1
                rep.flag(s, pid, "anh_meta_loi")
            pos = [im.get("position") for im in imgs if isinstance(im, dict)]
            if pos != list(range(1, len(imgs) + 1)):
                c["bad_pos"] += 1
                rep.flag(s, pid, "position_khong_lien_tuc", str(pos[:10]))
            urls = [im.get("url") for im in imgs if isinstance(im, dict)]
            if len(set(urls)) != len(urls):
                c["dup_url_in_product"] += 1
                rep.flag(s, pid, "url_anh_lap_trong_san_pham")
            if r.get("text") and r.get("text_group") != calc_text_group(r["text"]):
                c["tg_mismatch"] += 1
                rep.flag(s, pid, "text_group_khong_khop_text")
        rows_out.append([s, f"{len(rows):,}", c["missing"], c["bad_cat"], c["bad_img_meta"], c["bad_pos"],
                         c["dup_url_in_product"], c["tg_mismatch"], c["no_shop"]])
        tot.update(c)
    table(["split", "dòng", "thiếu field", "category lạ", "ảnh meta lỗi", "position sai", "URL lặp/sp",
           "text_group lệch", "shop rỗng"], rows_out)
    dup_id = [k for k, n in ids.items() if n > 1]
    dup_path = [k for k, n in paths.items() if n > 1]
    print(f"\n  id trùng: {len(dup_id)}   local_path trùng giữa các ảnh: {len(dup_path)}")
    n_all = sum(len(r) for r in data.values())
    rep.check("Field bắt buộc đầy đủ", tot["missing"], must_be_zero(tot["missing"]))
    rep.check("Category ∈ {áo, quần, phụ kiện}", tot["bad_cat"], must_be_zero(tot["bad_cat"]))
    rep.check("Id không trùng", len(dup_id), must_be_zero(len(dup_id)))
    rep.check("local_path không trùng", len(dup_path), must_be_zero(len(dup_path)), "2 ảnh trỏ vào cùng một file")
    rep.check("Metadata ảnh hợp lệ (url, local_path images/*.jpg|png)", tot["bad_img_meta"], must_be_zero(tot["bad_img_meta"]))
    rep.check("position ảnh liên tục từ 1", tot["bad_pos"], must_be_zero(tot["bad_pos"]))
    rep.check("URL ảnh không lặp trong sản phẩm", tot["dup_url_in_product"], must_be_zero(tot["dup_url_in_product"]))
    rep.check("text_group = md5(text) (khoá chống rò rỉ đáng tin)", tot["tg_mismatch"], must_be_zero(tot["tg_mismatch"]),
              "text_group lệch thì build_splits gom nhóm sai")
    rep.check("Shop không rỗng (khoá gom anh em)", fp(pct(tot["no_shop"], n_all)), grade(pct(tot["no_shop"], n_all), 0.02),
              "shop rỗng làm nhóm anh em gộp qua nhiều shop")

    if manifest is None:
        rep.check("split.json khớp với 3 file jsonl", "không có split.json", "WARN")
        return
    mism = sum(1 for s, rows in data.items() for r in rows if (manifest.get(str(r.get("id"))) or {}).get("split") != s)
    extra = len(set(manifest) - set(ids))
    print(f"  split.json: {len(manifest):,} id; id lệch split so với jsonl: {mism}; id chỉ có trong split.json: {extra}")
    rep.check("split.json khớp với 3 file jsonl", mism + extra, must_be_zero(mism + extra))
    gvals = {v.get("group_id") for v in manifest.values()}
    useless = gvals <= set(SPLITS)
    if useless:
        print("  LƯU Ý: group_id trong split.json chỉ là tên split, không phải id nhóm "
              "(build_splits.py dùng assignment[...] đã là tên split).")
    rep.check("group_id trong split.json là id nhóm thật", "chỉ là tên split" if useless else "ok",
              "WARN" if useless else "PASS", "không truy ngược được nhóm anh em; sửa: group_id = row_to_gid[id]")


# ---------------------------------------------------------------- 2. split
def sec_split(data, rep, args):
    head(rep, "2. CHẤT LƯỢNG SPLIT VÀ RÒ RỈ THEO KHOÁ GOM NHÓM")
    n_all = sum(len(r) for r in data.values())

    print("2.1 Tỉ lệ")
    dev, rows_out = 0.0, []
    for s, rows in data.items():
        share = pct(len(rows), n_all)
        dev = max(dev, abs(share - EXPECTED_RATIO[s]))
        rows_out.append([s, f"{len(rows):,}", fp(share), fp(EXPECTED_RATIO[s], 0)])
    table(["split", "sản phẩm", "tỉ lệ", "mục tiêu"], rows_out)
    rep.check("Tỉ lệ split ≈ 80/10/10", fp(dev, 2) + " lệch tối đa", grade(dev, RATIO_TOL))

    print("\n2.2 Mỗi (nguồn, category) được chia thế nào (% của chính nhóm đó)")
    strata = {s: Counter(f"{r.get('source')}|{r.get('category')}" for r in rows) for s, rows in data.items()}
    keys = sorted(set().union(*strata.values()))
    rows_out, worst = [], 0.0
    for k in keys:
        tot = sum(strata[s][k] for s in SPLITS)
        shares = {s: pct(strata[s][k], tot) for s in SPLITS}
        worst = max(worst, *(abs(shares[s] - EXPECTED_RATIO[s]) for s in SPLITS))
        rows_out.append([k, f"{tot:,}"] + [f"{strata[s][k]:>5,} ({fp(shares[s])})" for s in SPLITS])
    table(["nguồn|category", "tổng"] + list(SPLITS), rows_out)
    rep.check("Mỗi (nguồn, category) chia ≈ 80/10/10", fp(worst, 1) + " lệch tối đa", grade(worst, STRATUM_TOL))
    small = [(s, k, strata[s][k]) for s in ("val", "test") for k in keys if strata[s][k] < MIN_STRATUM_EVAL]
    if small:
        print(f"  Nhóm quá nhỏ trong val/test (<{MIN_STRATUM_EVAL}): {small}")
    rep.check(f"Mỗi (nguồn, category) ≥{MIN_STRATUM_EVAL} sp trong val/test", len(small), "WARN" if small else "PASS",
              "metric theo nhóm sẽ nhiễu nếu nhóm quá nhỏ")
    src_train = {r.get("shop") for r in data["train"]}
    for s in ("val", "test"):
        seen = sum(1 for r in data[s] if r.get("shop") in src_train)
        print(f"  {s}: {fp(pct(seen, len(data[s])))} sản phẩm thuộc shop đã có trong train")
    rep.data["strata"] = {s: dict(c) for s, c in strata.items()}

    print("\n2.3 Rò rỉ theo khoá gom nhóm (sản phẩm val/test có khoá trùng split trước)")
    leak_defs = {
        "id": lambda r: [str(r.get("id"))],
        "text_group": lambda r: [r.get("text_group")],
        "URL ảnh": lambda r: [i["url"] for i in r.get("images") or [] if isinstance(i, dict) and i.get("url")],
        "anh em (nguồn|shop|tên bỏ SKU)": lambda r: [k for k in [sib_key(r)] if k],
    }
    rows_out = []
    for label, fn in leak_defs.items():
        seen = {k for r in data["train"] for k in fn(r)}
        row = [label]
        for s in ("val", "test"):
            hit = 0
            for r in data[s]:
                if any(k in seen for k in fn(r)):
                    hit += 1
                    rep.flag(s, r.get("id"), f"ro_ri_{label}")
            row.append(hit)
            seen |= {k for r in data[s] for k in fn(r)}
            rep.check(f"Không rò rỉ {label} ({s})", hit, must_be_zero(hit))
        rows_out.append(row)
    table(["khoá", "val ∩ train", "test ∩ (train+val)"], rows_out)

    print("\n2.4 Tên sản phẩm trùng split trước (bỏ số/dấu câu, tên ≥4 từ) - rò rỉ ngữ nghĩa qua shop khác")
    names = {s: {norm_name(r.get("name")) for r in data[s] if len(norm_name(r.get("name")).split()) >= 4} for s in SPLITS}
    rows_out = []
    for q, ref in (("val", ("train",)), ("test", ("train", "val"))):
        seen = set().union(*(names[s] for s in ref))
        hit = [r for r in data[q] if norm_name(r.get("name")) in seen]
        rate = pct(len(hit), len(data[q]))
        rows_out.append([f"{q} → {'+'.join(ref)}", len(hit), fp(rate)])
        for r in hit:
            rep.flag(q, r.get("id"), "ten_trung_split_truoc", str(r.get("name"))[:80])
        rep.check(f"Tên trùng split trước ({q})", fp(rate), grade(rate, NAME_LEAK_WARN),
                  "cùng tên có thể là cùng mẫu bán ở shop khác")
    table(["cặp", "sản phẩm", "tỉ lệ"], rows_out)


# ---------------------------------------------------------------- 3. ảnh
_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * j + 1) * k / (2 * _N)) for j in range(_N)] for k in range(_N)])


def phash64(gray_img):
    """pHash 64-bit (DCT 32x32, lấy khối 8x8 tần số thấp, so với trung vị)."""
    from PIL import Image
    a = np.asarray(gray_img.resize((_N, _N), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ a @ _DCT.T)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def inspect_image(path):
    from PIL import Image
    if not path.exists():
        return {"error": "missing"}
    try:
        with Image.open(path) as im:
            fmt, size, mode = im.format, im.size, im.mode
            im.load()
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)
            gray = im.convert("L")
    except Exception as e:  # file hỏng / cắt cụt
        return {"error": f"unreadable: {type(e).__name__}"}
    g = np.asarray(gray, dtype=np.uint8)
    mask = g < 245
    if mask.any():
        ys, xs = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        content = float((ys[-1] - ys[0] + 1) * (xs[-1] - xs[0] + 1) / g.size)
    else:
        content = 0.0
    std = float(g.std())
    return {"fmt": fmt, "size": size, "mode": mode, "bytes": path.stat().st_size, "std": std, "content": content,
            "mean": rgb.mean(0), "sq": (rgb ** 2).mean(0), "phash": phash64(gray) if std >= 3 else None}


def sec_images(data, rep, args):
    head(rep, "3. ẢNH (đầu vào nhánh ảnh)")
    if args.skip_images:
        print("  Bỏ qua (--skip-images).")
        rep.check("Kiểm tra file ảnh", "chưa kiểm tra", "WARN", "chạy lại không có --skip-images trước khi train")
        return None
    root, rng = Path(args.image_root), random.Random(0)
    size = args.image_size
    rows_out, arr, train_moments = [], {}, []
    for s, rows in data.items():
        jobs = [(i, r.get("id"), root / im["local_path"], im.get("position")) for i, r in enumerate(rows)
                for im in r.get("images") or [] if isinstance(im, dict) and im.get("local_path")]
        total = len(jobs)
        if args.max_images and total > args.max_images:
            jobs = rng.sample(jobs, args.max_images)
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(tqdm(ex.map(lambda j: inspect_image(j[2]), jobs), total=len(jobs), desc=f"ảnh {s}"))
        c, contents, nbytes = Counter(), [], []
        hs, own, pos = [], [], []
        for (i, pid, path, position), x in zip(jobs, res):
            if "error" in x:
                kind = x["error"].split(":")[0]
                c[kind] += 1
                rep.flag(s, pid, f"anh_{kind}", str(path))
                continue
            if FORMAT_OF_EXT.get(path.suffix.lower()) != x["fmt"]:
                c["bad_format"] += 1
                rep.flag(s, pid, "anh_sai_dinh_dang", f"{path.suffix} / {x['fmt']}: {path}")
            if x["size"] != (size, size):
                c["wrong_size"] += 1
                rep.flag(s, pid, "anh_sai_kich_thuoc", f"{x['size']}: {path}")
            if x["mode"] != "RGB":
                c["not_rgb"] += 1
            if x["std"] < 3:
                c["flat"] += 1
                rep.flag(s, pid, "anh_trang_hoac_dong_mau", str(path))
            if x["content"] < 0.10:
                c["tiny"] += 1
                rep.flag(s, pid, "san_pham_qua_nho_trong_anh", f"{x['content']:.2f}: {path}")
            contents.append(x["content"])
            nbytes.append(x["bytes"])
            if s == "train":
                train_moments.append((x["mean"], x["sq"]))
            if x["phash"] is not None:
                hs.append(x["phash"])
                own.append(i)
                pos.append(position)
        arr[s] = {"h": np.array(hs, dtype=np.uint64), "own": np.array(own, dtype=np.int64), "pos": pos}
        n = len(jobs)
        ok = n - c["missing"] - c["unreadable"]
        rows_out.append([s, f"{n:,}" + ("*" if n < total else ""), c["missing"], c["unreadable"], c["bad_format"],
                         c["wrong_size"], c["not_rgb"], f"{c['flat']} ({fp(pct(c['flat'], ok), 2)})",
                         f"{c['tiny']} ({fp(pct(c['tiny'], ok))})",
                         f"{np.median(nbytes) / 1024:.0f}KB" if nbytes else "-"])
        for key, name in (("missing", "Ảnh có file"), ("unreadable", "Ảnh đọc được"),
                          ("bad_format", "Đuôi file khớp định dạng thật (.jpg=JPEG, .png=PNG)")):
            rep.check(f"{name} ({s})", c[key], must_be_zero(c[key]))
        rep.check(f"Ảnh đúng {size}×{size} ({s})", c["wrong_size"], must_be_zero(c["wrong_size"]),
                  "to_square luôn xuất đúng kích thước")
        rep.check(f"Ảnh RGB ({s})", c["not_rgb"], must_be_zero(c["not_rgb"]))
        rep.check(f"Ảnh trắng/đồng màu ({s})", fp(pct(c["flat"], ok), 2), grade(pct(c["flat"], ok), FLAT_WARN))
        rep.check(f"Sản phẩm quá nhỏ trong khung ({s})", fp(pct(c["tiny"], ok)), grade(pct(c["tiny"], ok), TINY_CONTENT_WARN),
                  "vùng không-trắng < 10% khung ảnh")
        rep.data.setdefault("images", {})[s] = {"inspected": n, "total": total, **dict(c),
                                                 "content_ratio": pcts(contents, (5, 50))}
    table(["split", "ảnh đọc", "thiếu", "hỏng", "sai định dạng", "sai size", "không RGB", "trắng", "quá nhỏ",
           "dung lượng"], rows_out)
    if args.max_images:
        print(f"  (* = chỉ đọc mẫu {args.max_images} ảnh/split; mục 4 cũng chỉ phủ phần mẫu này)")
    if train_moments:
        m = np.mean([a for a, _ in train_moments], axis=0) / 255
        sd = np.sqrt(np.maximum(np.mean([b for _, b in train_moments], axis=0) / 255 ** 2 - m ** 2, 0))
        print(f"\n  Mean/std RGB của train (mẫu {len(train_moments):,} ảnh, thang 0-1; ảnh đệm trắng nên mean cao hơn ImageNet):")
        print(f"    mean = {np.round(m, 3).tolist()}   std = {np.round(sd, 3).tolist()}")
        rep.data["image_mean_std"] = {"mean": m.tolist(), "std": sd.tolist()}
    return arr


# ---------------------------------------------------------------- 4. rò rỉ theo nội dung ảnh
def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    pop8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return pop8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1, dtype=np.int16)


def min_hamming(q, ref, chunk=128):
    d = np.full(len(q), 64, dtype=np.int16)
    a = np.zeros(len(q), dtype=np.int64)
    if len(ref) == 0:
        return d, a
    for i in range(0, len(q), chunk):
        pc = popcount64(np.bitwise_xor(q[i:i + chunk, None], ref[None, :]))
        j = pc.argmin(1)
        a[i:i + chunk] = j
        d[i:i + chunk] = pc[np.arange(len(j)), j]
    return d, a


def sec_image_leak(data, rep, arr, args):
    head(rep, "4. RÒ RỈ VÀ TRÙNG LẶP THEO NỘI DUNG ẢNH (pHash 64-bit tính từ file ảnh)")
    if arr is None:
        print("  Bỏ qua (--skip-images).")
        rep.check("Rò rỉ ảnh gần trùng giữa split", "chưa kiểm tra", "WARN",
                  "build_splits không gom theo nội dung ảnh nên đây là chốt chặn duy nhất")
        return
    n_hash = sum(len(arr[s]["h"]) for s in SPLITS)
    n_img = sum(len(r.get("images") or []) for rows in data.values() for r in rows)
    print(f"  Ảnh có pHash: {n_hash:,}/{n_img:,} (thiếu = ảnh lỗi, đồng màu, hoặc chỉ đọc mẫu)")
    rep.check("Độ phủ pHash", fp(pct(n_hash, n_img)), "PASS" if args.max_images else grade(1 - pct(n_hash, n_img), 0.01, 0.5))

    print(f"\n4.1 Sản phẩm val/test có ảnh gần trùng ảnh của split trước (khoảng cách Hamming; ≤4 bit thường là cùng một ảnh)")
    thresholds = (0, 4, 8)
    rows_out = []
    for q, refs in (("val", ("train",)), ("test", ("train", "val"))):
        qh, qo = arr[q]["h"], arr[q]["own"]
        rh = np.concatenate([arr[s]["h"] for s in refs])
        ref_meta = [(s, data[s][i].get("id")) for s in refs for i in arr[s]["own"].tolist()]
        d, a = min_hamming(qh, rh)
        row = [f"{q} → {'+'.join(refs)}"]
        for t in thresholds:
            prods = set(qo[d <= t].tolist())
            row.append(f"{len(prods)} ({fp(pct(len(prods), len(data[q])), 2)})")
        rows_out.append(row)
        best = {}
        for k in np.where(d <= args.near_hamming)[0]:
            o = int(qo[k])
            if o not in best or d[k] < best[o][0]:
                best[o] = (int(d[k]), ref_meta[int(a[k])])
        for o, (dist, (rs, rid)) in best.items():
            rep.flag(q, data[q][o].get("id"), "anh_gan_trung_split_truoc", f"d={dist} với {rs} id={rid}")
        rate = pct(len(best), len(data[q]))
        rep.check(f"Ảnh gần trùng (≤{args.near_hamming} bit) {q} → {'+'.join(refs)}", fp(rate, 2),
                  grade(rate, NEAR_DUP_WARN, NEAR_DUP_FAIL), "R@k cao ảo vì model đã thấy ảnh này lúc train")
    table(["cặp", "= 0 bit", "≤ 4 bit", "≤ 8 bit"], rows_out)

    print("\n4.2 Trùng ảnh giữa các sản phẩm CÙNG split và ảnh lặp trong cùng sản phẩm (pHash trùng hệt)")
    rows_out = []
    for s in SPLITS:
        hs, ow = arr[s]["h"].tolist(), arr[s]["own"].tolist()
        by_hash = defaultdict(set)
        for h, o in zip(hs, ow):
            by_hash[h].add(o)
        tied = {o for v in by_hash.values() if len(v) > 1 for o in v}
        in_prod = len({o for (o, _), n in Counter(zip(ow, hs)).items() if n > 1})
        groups = Counter(r.get("text_group") for r in data[s])
        tied_txt = sum(n for n in groups.values() if n > 1)
        rows_out.append([s, len(tied), fp(pct(len(tied), len(data[s])), 2), tied_txt,
                         fp(pct(tied_txt, len(data[s])), 2), in_prod, fp(pct(in_prod, len(data[s])), 2)])
        if s != "train":
            rate = pct(len(tied), len(data[s]))
            rep.check(f"Ảnh trùng hệt sản phẩm khác cùng split ({s})", fp(rate, 2), grade(rate, TIE_WARN),
                      "query ảnh có 2 đáp án; train: 2 sản phẩm cùng ảnh nhưng text khác nhau")
            rate = pct(tied_txt, len(data[s]))
            rep.check(f"Text trùng hệt sản phẩm khác cùng split ({s})", fp(rate, 2), grade(rate, TIE_WARN),
                      "query text có nhiều đáp án đúng")
    table(["split", "sp trùng ảnh", "%", "sp trùng text", "%", "sp có ảnh lặp", "%"], rows_out)
    print("  (train: sản phẩm trùng ảnh/text khác nhau tạo negative sai; ảnh lặp trong sản phẩm làm lệch lấy mẫu positive)")


# ---------------------------------------------------------------- 5. text
def token_lengths(texts, tokenizer_name, use_tokenizer):
    if use_tokenizer:
        try:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            lens = []
            for i in tqdm(range(0, len(texts), 512), desc="tokenize"):
                enc = tok(texts[i:i + 512], add_special_tokens=True, truncation=False)["input_ids"]
                lens += [len(x) for x in enc]
            return np.array(lens), f"tokenizer {tokenizer_name}"
        except Exception as e:
            print(f"  (không nạp được tokenizer: {type(e).__name__}; dùng ước lượng 1.3 token/từ, KẾT QUẢ CHỈ GẦN ĐÚNG)")
    return np.array([int(len(t.split()) * 1.3) + 2 for t in texts]), "ƯỚC LƯỢNG 1.3 token/từ"


def sec_text(data, rep, args):
    head(rep, "5. TEXT (đầu vào nhánh PhoBERT: text_seg)")
    all_rows = [(s, r) for s, rows in data.items() for r in rows]
    lens, how = token_lengths([r.get("text_seg") or "" for _, r in all_rows], args.tokenizer, not args.no_tokenizer)
    limit = args.max_text_length
    by_split = defaultdict(list)
    for (s, _), n in zip(all_rows, lens):
        by_split[s].append(n)
    print(f"5.1 Độ dài token của text_seg ({how}); giới hạn model = {limit}")
    rows_out = []
    for s in SPLITS:
        v = np.array(by_split[s])
        q = pcts(v)
        trunc = pct(int((v > limit).sum()), len(v))
        rows_out.append([s, f"{np.mean(v):.0f}", f"{q['p50']:.0f}", f"{q['p90']:.0f}", f"{q['p99']:.0f}",
                         f"{q['max']:.0f}", fp(trunc, 2)])
        rep.check(f"Text bị cắt ở {limit} token ({s})", fp(trunc, 2), grade(trunc, TRUNC_WARN),
                  "tên và màu nằm đầu nên không mất; chỉ mất đuôi mô tả" + ("" if "tokenizer" in how else " (số ước lượng)"))
        rep.data.setdefault("text", {})[s] = {"tokens": q, "truncated_rate": trunc, "method": how}
    table(["split", "TB", "p50", "p90", "p99", "max", f"> {limit}"], rows_out)

    print("\n5.2 Nhất quán giữa name / text / text_seg")
    rows_out = []
    for s in SPLITS:
        c = Counter()
        for r in data[s]:
            t, seg, name = str(r.get("text") or ""), str(r.get("text_seg") or ""), str(r.get("name") or "")
            if not t.startswith(name.rstrip(".!?… ")):
                c["no_name"] += 1
                rep.flag(s, r.get("id"), "text_khong_bat_dau_bang_ten", t[:80])
            if seg != seg.casefold():
                c["not_lower"] += 1
            if fold(seg) != fold(" ".join(t.casefold().split()[:MAX_SEG_SYLLABLES])):
                c["seg_mismatch"] += 1
                rep.flag(s, r.get("id"), "text_seg_khong_khop_text", t[:80])
            if "_" not in seg and len(t.split()) >= 8:
                c["no_us"] += 1
                rep.flag(s, r.get("id"), "text_seg_khong_co_tu_ghep", seg[:80])
        rows_out.append([s] + [f"{c[k]} ({fp(pct(c[k], len(data[s])), 2)})" for k in
                               ("no_name", "not_lower", "seg_mismatch", "no_us")])
        rep.check(f"text bắt đầu bằng tên ({s})", c["no_name"], must_be_zero(c["no_name"]), "build_text luôn đặt tên đầu tiên")
        rep.check(f"text_seg là chữ thường ({s})", c["not_lower"], must_be_zero(c["not_lower"]))
        rep.check(f"text_seg khớp text, chỉ khác dấu cách/'_'/vị trí dấu ({s})", fp(pct(c["seg_mismatch"], len(data[s])), 2),
                  grade(pct(c["seg_mismatch"], len(data[s])), 0.005, 0.05), "text_seg lệch text = tách từ sai hoặc dữ liệu cũ")
        rep.check(f"text_seg có từ ghép '_' khi text ≥8 từ ({s})", fp(pct(c["no_us"], len(data[s])), 2),
                  grade(pct(c["no_us"], len(data[s])), 0.02, 0.2), "nhiều bản ghi không có '_' = tách từ không chạy")
    table(["split", "text không bắt đầu bằng tên", "text_seg không lowercase", "text_seg lệch text",
           "text_seg không có '_'"], rows_out)

    print("\n5.3 Rác còn sót (số sản phẩm; kiểm tra trên text, riêng mã SKU/quảng cáo/giặt ủi/cỡ chỉ xét phần sau tên)")
    hard = {
        "HTML": lambda n, b, t: _RE_HTML.search(t),
        "URL": lambda n, b, t: _RE_URL.search(t),
        "SĐT": lambda n, b, t: _RE_PHONE.search(t),
        "emoji": lambda n, b, t: _RE_EMOJI.search(t),
        "ký tự ngoại ngữ": lambda n, b, t: _RE_FOREIGN.search(t),
        "quảng cáo/liên hệ": lambda n, b, t: _RE_PROMO.search(b),
    }
    soft = {
        "hướng dẫn giặt/ủi": lambda n, b, t: _RE_CARE.search(b),
        "cỡ/cân nặng/bảng size": lambda n, b, t: _RE_SIZE.search(b),
        "mã sản phẩm trong mô tả": lambda n, b, t: any(not _NOT_CODE.fullmatch(m) for m in _RE_CODE.findall(b)),
    }
    checks = {**hard, **soft}
    counts = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for r in data[s]:
            n, b = split_text(r)
            t = str(r.get("text") or "")
            for label, fn in checks.items():
                if fn(n, b, t):
                    counts[s][label] += 1
                    if label in hard:
                        rep.flag(s, r.get("id"), f"rac_{label.replace(' ', '_')}", t[:80])
    table(["loại"] + list(SPLITS), [[k] + [f"{counts[s][k]} ({fp(pct(counts[s][k], len(data[s])))})" for s in SPLITS]
                                     for k in checks])
    for label in checks:
        warn = HARD_JUNK_WARN if label in hard else SOFT_JUNK_WARN
        for s in SPLITS:
            rate = pct(counts[s][label], len(data[s]))
            rep.check(f"Rác '{label}' ({s})", fp(rate, 2), grade(rate, warn))

    print("\n5.4 Độ giàu thông tin của text (train+val+test, theo nguồn)")
    stat = defaultdict(Counter)
    for s in SPLITS:
        for r in data[s]:
            has_color, extra = text_parts(r)
            c = stat[r.get("source")]
            c["n"] += 1
            c["name_only"] += extra < SHORT_TEXT_WORDS
            c["has_color"] += has_color
            c["color_field_no_text"] += bool(r.get("colors")) and not has_color
            if extra < SHORT_TEXT_WORDS:
                rep.flag(s, r.get("id"), "text_chi_co_ten_va_mau", str(r.get("text"))[:80])
    table(["nguồn", "sản phẩm", f"chỉ tên+màu (<{SHORT_TEXT_WORDS} từ mô tả)", "text có màu",
           "có field colors nhưng text không màu"],
          [[k, f"{c['n']:,}", fp(pct(c['name_only'], c['n'])), fp(pct(c['has_color'], c['n'])),
            fp(pct(c['color_field_no_text'], c['n']))] for k, c in sorted(stat.items())])
    n_all = sum(c["n"] for c in stat.values())
    name_only = pct(sum(c["name_only"] for c in stat.values()), n_all)
    nocolor = pct(sum(c["color_field_no_text"] for c in stat.values()), n_all)
    rep.check("Text chỉ có tên + màu (không mô tả)", fp(name_only), grade(name_only, NAME_ONLY_WARN),
              "text nghèo thông tin: model chỉ học từ tên; xem bảng theo nguồn")
    rep.check("Có field colors nhưng text không có màu", fp(nocolor), grade(nocolor, NO_COLOR_WARN),
              "giá trị màu không chuẩn hoá được (vd 'Ngẫu nhiên'); mất tín hiệu phân biệt anh em")
    rep.data["text_richness"] = {k: dict(c) for k, c in stat.items()}

    print("\n5.5 Câu lặp nhiều trong train (boilerplate làm mọi text giống nhau)")
    n_tr = len(data["train"])
    df = Counter()
    docs = []
    for r in data["train"]:
        ps = {p.strip().casefold() for p in split_text(r)[1].split(". ") if len(p.split()) >= 4 and not p.startswith("Màu ")}
        docs.append(ps)
        df.update(ps)
    thr = max(20, BOILER_DF * n_tr)
    common_sent = {k: n for k, n in df.items() if n >= thr}
    hit = sum(1 for ps in docs if ps & common_sent.keys())
    for k, n in sorted(common_sent.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n:>5} ({fp(pct(n, n_tr))})  {k[:90]}")
    print(f"  Câu xuất hiện ≥{fp(BOILER_DF, 0)} train: {len(common_sent)}; sản phẩm train chứa ≥1 câu như vậy: {fp(pct(hit, n_tr))}")
    rep.check("Sản phẩm train chứa câu boilerplate", fp(pct(hit, n_tr)), grade(pct(hit, n_tr), BOILER_WARN),
              "câu lặp ở nhiều sản phẩm không giúp phân biệt; cân nhắc thêm vào filter trong common.py")

    train_words = Counter(w for r in data["train"] for w in (r.get("text_seg") or "").split())
    print(f"\n  Từ vựng train: {len(train_words):,} từ. 20 từ phổ biến nhất (soát nhiễu bằng mắt):")
    print("  " + ", ".join(f"{w}({n})" for w, n in train_words.most_common(20)))
    for s in ("val", "test"):
        oov = sum(n for w, n in Counter(w for r in data[s] for w in (r.get("text_seg") or "").split()).items()
                  if w not in train_words)
        tot = sum(len((r.get("text_seg") or "").split()) for r in data[s])
        print(f"  {s}: {fp(pct(oov, tot), 2)} từ không có trong từ vựng train (PhoBERT vẫn xử lý bằng subword)")


# ---------------------------------------------------------------- 6. nhãn và cặp
def suspicious_category(r):
    name = str(r.get("name") or "").casefold()
    cat = r.get("category")
    has_ao, has_quan, has_acc = bool(KW_AO.search(name)), bool(KW_QUAN.search(name)), bool(KW_ACC.search(name))
    if cat == "áo":
        return (has_quan or has_acc) and not has_ao
    if cat == "quần":
        return (has_ao or has_acc) and not has_quan
    if cat == "phụ kiện":
        return (has_ao or has_quan) and not has_acc
    return False


def sec_labels(data, rep):
    head(rep, "6. NHÃN VÀ CẶP (ảnh, text)")
    print("6.1 Category so với `type` (nguồn) và so với từ khoá trong tên")
    type_cat = defaultdict(Counter)
    for rows in data.values():
        for r in rows:
            type_cat[r.get("type")][r.get("category")] += 1
    major = {t: c.most_common(1)[0] for t, c in type_cat.items() if sum(c.values()) >= 10}
    rows_out = []
    for s in SPLITS:
        bad_type = [r for r in data[s] if r.get("type") in major and r.get("category") != major[r["type"]][0]
                    and major[r["type"]][1] / sum(type_cat[r["type"]].values()) >= 0.9]
        bad_kw = [r for r in data[s] if suspicious_category(r)]
        for r in bad_type:
            rep.flag(s, r.get("id"), "category_lech_type", f"{r.get('category')} | type={r.get('type')} | {r.get('name')}")
        for r in bad_kw:
            rep.flag(s, r.get("id"), "category_nghi_sai_theo_ten", f"{r.get('category')} | {r.get('name')}")
        rows_out.append([s, f"{len(bad_type)} ({fp(pct(len(bad_type), len(data[s])), 2)})",
                         f"{len(bad_kw)} ({fp(pct(len(bad_kw), len(data[s])), 2)})"])
        rep.check(f"Category lệch so với type ({s})", fp(pct(len(bad_type), len(data[s])), 2),
                  grade(pct(len(bad_type), len(data[s])), 0.005), "type chiếm ≥90% một category nhưng sản phẩm này khác")
        rep.check(f"Category nghi sai theo tên ({s})", fp(pct(len(bad_kw), len(data[s])), 2),
                  grade(pct(len(bad_kw), len(data[s])), SUSPECT_CAT_WARN), "heuristic từ khoá, cần soát mắt (validation_flags.csv)")
        for r in bad_kw[:2]:
            print(f"    [{s}] {r.get('category')} | {r.get('name')}")
    table(["split", "category lệch type", "category nghi sai theo tên"], rows_out)

    print("\n6.2 Số ảnh mỗi sản phẩm (mỗi ảnh ghép với text của sản phẩm tạo thành một cặp positive)")
    rows_out = []
    for s, rows in data.items():
        cnt = Counter(min(len(r.get("images") or []), 6) for r in rows)
        n_imgs = [len(r.get("images") or []) for r in rows]
        total = sum(n_imgs)
        heavy = sum(n for n in n_imgs if n > 10)
        top1 = sum(sorted(n_imgs, reverse=True)[:max(1, len(n_imgs) // 100)])
        rows_out.append([s] + [f"{cnt[k]:,}" for k in range(1, 7)] +
                        [f"{total:,}", f"{total / max(len(rows), 1):.2f}", fp(pct(heavy, total)), fp(pct(top1, total))])
        rep.data.setdefault("pairs", {})[s] = {"products": len(rows), "images": total, "share_images_from_gt10": pct(heavy, total)}
    table(["split", "1", "2", "3", "4", "5", "6+", "ảnh", "ảnh/sp", "ảnh thuộc sp >10 ảnh", "ảnh thuộc 1% sp nhiều ảnh nhất"],
          rows_out)
    heavy_train = rep.data["pairs"]["train"]["share_images_from_gt10"]
    rep.check("Ảnh train thuộc sản phẩm >10 ảnh", fp(heavy_train), "WARN" if heavy_train > 0.25 else "PASS",
              "sản phẩm nhiều ảnh chiếm nhiều cặp positive; nên lấy mẫu theo sản phẩm hoặc giới hạn số ảnh mỗi epoch")


# ---------------------------------------------------------------- 7. giá trị cho retrieval
def cluster_sizes(rows):
    """Cỡ cụm "anh em" của từng sản phẩm: gộp theo text_group và nhóm (nguồn|shop|tên bỏ SKU)."""
    uf = UnionFind()
    for i, r in enumerate(rows):
        uf.union(f"r{i}", f"tg:{r.get('text_group')}")
        k = sib_key(r)
        if k:
            uf.union(f"r{i}", f"sib:{k}")
    roots = [uf.find(f"r{i}") for i in range(len(rows))]
    cnt = Counter(roots)
    return np.array([cnt[x] for x in roots])


def sec_retrieval(data, rep, args):
    head(rep, "7. GIÁ TRỊ DỮ LIỆU CHO BÀI TOÁN RETRIEVAL ẢNH ↔ TEXT")
    B = args.batch_size
    rows_out = []
    for s, rows in data.items():
        n = len(rows)
        m = cluster_sizes(rows)
        multi = sum(1 for r in rows if len(r.get("images") or []) >= 2)
        in_cluster = pct(int((m > 1).sum()), n)
        ceiling = float(np.mean(1 / m)) if n else 0.0
        p_fn = float(np.mean(1 - (1 - (m - 1) / max(n - 1, 1)) ** (B - 1))) if n else 0.0
        pairs_batch = float(((m - 1).sum() / 2) * B * (B - 1) / max(n * (n - 1), 1))
        rows_out.append([s, f"{n:,}", f"{multi:,}", fp(in_cluster), f"{np.median(m):.0f}/{m.max() if n else 0}",
                         fp(ceiling), f"{pairs_batch:.2f}", fp(p_fn, 1), fp(1 / max(n, 1), 3), fp(min(10, n) / max(n, 1), 2)])
        rep.data.setdefault("retrieval", {})[s] = {
            "products": n, "queries_img2img": multi, "share_in_sibling_cluster": in_cluster, "r1_ceiling": ceiling,
            "false_negative_pairs_per_batch": pairs_batch, "share_with_false_negative": p_fn}
    table(["split", "sản phẩm", "sp ≥2 ảnh", "sp có anh em", "cỡ cụm p50/max", "trần R@1*",
           f"cặp nhầm/batch{B}", f"sp có negative sai", "R@1 ngẫu nhiên", "R@10 ngẫu nhiên"], rows_out)
    print(f"""
  Cách đọc:
  - 'anh em' = cùng shop và cùng tên sau khi bỏ mã SKU (hoặc cùng text). build_splits giữ cả cụm trong một split,
    nên trong val/test chúng là các sản phẩm rất giống nhau đứng cạnh nhau trong gallery.
  - *trần R@1 = R@1 tối đa nếu model hoàn toàn không phân biệt được các anh em (trung bình 1/cỡ cụm).
    Thực tế ảnh/màu giúp phân biệt nên đây là mức khó nhất; nên báo cáo thêm R@k mức cụm.
  - 'sp có negative sai' (train) = % sản phẩm có ≥1 anh em rơi vào cùng batch {B}: InfoNCE sẽ đẩy xa hai sản phẩm
    gần như giống nhau. Nếu cao: dùng sampler không xếp anh em chung batch hoặc che chúng khỏi loss.
  - 'sp ≥2 ảnh' = số query ảnh→ảnh đánh giá được (query là ảnh phụ, gallery là ảnh đại diện).
  - R@k ngẫu nhiên = mốc tối thiểu model phải vượt xa trên gallery val/test.""")
    p_fn = rep.data["retrieval"]["train"]["share_with_false_negative"]
    rep.check(f"Sản phẩm train có negative sai trong batch {B}", fp(p_fn), "WARN" if p_fn > FN_BATCH_WARN else "PASS",
              "dùng sampler/mask theo cụm anh em để tránh phạt oan")
    for s in ("val", "test"):
        r = rep.data["retrieval"][s]
        rep.check(f"Số query ảnh→ảnh đánh giá được ({s})", r["queries_img2img"],
                  "WARN" if r["queries_img2img"] < MIN_EVAL_QUERIES else "PASS", f"ít hơn {MIN_EVAL_QUERIES} thì R@k dao động mạnh")
        rep.check(f"Kích thước gallery ({s})", r["products"], "WARN" if r["products"] < MIN_GALLERY else "PASS")


# ---------------------------------------------------------------- tổng kết
def summary(data, rep, elapsed):
    head(rep, "TỔNG KẾT")
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    cnt = Counter(c["status"] for c in rep.checks)
    for status in ("FAIL", "WARN"):
        items = [c for c in rep.checks if c["status"] == status]
        if not items:
            continue
        print(f"\n  {status}:")
        last = None
        for c in sorted(items, key=lambda c: rep.checks.index(c)):
            if c["section"] != last:
                print(f"   - {c['section']}")
                last = c["section"]
            note = f"  — {c['note']}" if c["note"] else ""
            print(f"       {c['name']}: {c['value']}{note}")
    print(f"\n  {cnt['PASS']} PASS | {cnt['WARN']} WARN | {cnt['FAIL']} FAIL  (tổng {len(rep.checks)} mục)")
    n_all = sum(len(r) for r in data.values())
    n_img = sum(len(r.get("images") or []) for rows in data.values() for r in rows)
    print(f"  Dữ liệu: {n_all:,} sản phẩm, {n_img:,} ảnh; nguồn: "
          + ", ".join(f"{k}={v}" for k, v in Counter(r.get('source') for rows in data.values() for r in rows).items()))
    if cnt["FAIL"]:
        verdict = "KHÔNG ĐẠT: sửa các mục FAIL (chi tiết trong validation_flags.csv) trước khi train."
    elif cnt["WARN"]:
        verdict = "ĐẠT, CÓ CẢNH BÁO: train được; các mục WARN là rủi ro khi diễn giải kết quả."
    else:
        verdict = "ĐẠT: dữ liệu sạch theo mọi tiêu chí đã kiểm tra."
    print(f"  Kết luận: {verdict}\n  Thời gian chạy: {elapsed:.1f}s")
    return cnt["FAIL"]


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"{type(o)} không serialize được")


def save_outputs(rep, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump({"checks": rep.checks, "details": rep.data,
                   "flag_counts": Counter(x["issue"] for x in rep.flags)}, f, ensure_ascii=False, indent=2, default=_jd)
    with (out / "validation_flags.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "id", "issue", "detail"])
        w.writeheader()
        w.writerows(rep.flags)
    print(f"\n  Đã lưu: {out / 'validation_report.json'}  và  {out / 'validation_flags.csv'} ({len(rep.flags)} dòng)")


def parse_args():
    ap = argparse.ArgumentParser(description="Kiểm định dữ liệu đầu vào mô hình retrieval ảnh ↔ text (ViMFD)")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--manifest", default=None, help="đường dẫn split.json (mặc định <splits-dir>/split.json)")
    ap.add_argument("--image-root", default="data/processed", help="thư mục chứa images/ (local_path tính từ đây)")
    ap.add_argument("--out-dir", default="data/validation")
    ap.add_argument("--image-size", type=int, default=224, help="IMAGE_SIZE trong common.py")
    ap.add_argument("--max-text-length", type=int, default=256, help="MAX_TEXT_LENGTH của model")
    ap.add_argument("--tokenizer", default="vinai/phobert-base")
    ap.add_argument("--no-tokenizer", action="store_true", help="không nạp tokenizer, ước lượng độ dài token")
    ap.add_argument("--near-hamming", type=int, default=4, help="ngưỡng Hamming (pHash 64-bit) coi là ảnh gần trùng")
    ap.add_argument("--batch-size", type=int, default=64, help="batch size lúc train (ước lượng negative sai)")
    ap.add_argument("--skip-images", action="store_true", help="bỏ qua đọc file ảnh (mục 3, 4)")
    ap.add_argument("--max-images", type=int, default=0, help="chỉ đọc mẫu N ảnh mỗi split (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    data, manifest = load_data(args)
    rep = Report()
    sec_schema(data, manifest, rep)
    sec_split(data, rep, args)
    arr = sec_images(data, rep, args)
    sec_image_leak(data, rep, arr, args)
    sec_text(data, rep, args)
    sec_labels(data, rep)
    sec_retrieval(data, rep, args)
    n_fail = summary(data, rep, time.time() - t0)
    save_outputs(rep, args.out_dir)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()