"""Đánh giá toàn bộ dữ liệu trên data/splits/{train,val,test}.jsonl.

Chạy:
    python data_validation.py
    python data_validation.py --splits-dir data/splits --image-root data/processed
    python data_validation.py --skip-images          # bỏ qua đọc ảnh (nhanh)
    python data_validation.py --max-images 2000      # chỉ đọc mẫu 2000 ảnh / split

Gồm 6 nhóm kiểm tra:
    1. Toàn vẹn      : field bắt buộc, id trùng, text_seg, cấu trúc ảnh
    2. Chất lượng split: tỉ lệ, phân bố nguồn/category, rò rỉ giữa các split
                         (id, text, URL ảnh, phash trùng và gần trùng, tên trùng)
    3. Ảnh           : file tồn tại / đọc được, định dạng, kích thước, ảnh trắng, sản phẩm quá nhỏ
    4. Text          : độ dài token (có bị cắt ở MAX_TEXT_LENGTH không), rác còn sót,
                       tách từ có chạy không, category lệch với tên sản phẩm
    5. Metadata      : giá, màu, shop
    6. Giá trị cho bài toán retrieval: dữ liệu train, khả năng đánh giá, nguy cơ negative sai
Kết quả: in ra màn hình + <out-dir>/validation_report.json + <out-dir>/validation_flags.csv
Mã thoát: 1 nếu có mục FAIL, ngược lại 0.

Cần: numpy, Pillow. Tuỳ chọn: tqdm, transformers (đo token chính xác bằng tokenizer PhoBERT).
"""
import argparse
import csv
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
REQUIRED = ("id", "name", "source", "category", "text", "text_seg", "text_group", "images")
ZERO_HASH = "0" * 16
VALID_FORMATS = {"JPEG", "PNG"}
VALID_EXT = {".jpg", ".png"}

# ---- Ngưỡng cảnh báo (tỉ lệ xấu cao hơn ngưỡng thì WARN / FAIL) ----
RATIO_TOL = 0.015          # lệch tỉ lệ split so với 80/10/10
TVD_WARN = 0.03            # lệch phân bố (source/category) so với train
MIN_STRATUM_EVAL = 20      # số sản phẩm tối thiểu của mỗi (nguồn, category) trong val/test
NEAR_DUP_WARN, NEAR_DUP_FAIL = 0.005, 0.03   # tỉ lệ sản phẩm val/test có ảnh gần trùng với train
NAME_LEAK_WARN = 0.05
TIE_WARN = 0.005           # sản phẩm trong val/test trùng ảnh/text với sản phẩm khác cùng split
TRUNC_WARN = 0.10          # tỉ lệ text bị cắt
SHORT_TEXT_WORDS = 5
FLAT_WARN = 0.005
TINY_CONTENT_WARN = 0.05   # tỉ lệ ảnh mà sản phẩm chiếm < 10% khung hình
SUSPECT_CAT_WARN = 0.02
JUNK_WARN = 0.01
MISSING_META_WARN = 0.05
MAX_SEG_SYLLABLES = 300    # giống common.py

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://|www\.", re.I)
_RE_PHONE = re.compile(r"(?<!\d)0\d{8,10}(?!\d)")
_RE_EMOJI = re.compile("[\u2600-\u27BF\U0001F000-\U0001FAFF\uFE0F\u200d]")
_RE_FOREIGN = re.compile("[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\uac00-\ud7af]")
_RE_CODE = re.compile(r"\b(?=[A-Za-z0-9\-/]*\d)(?=[A-Za-z0-9\-/]*[A-Za-z])[A-Za-z0-9\-/]{6,}\b")
_RE_NONWORD = re.compile(r"[\W\d_]+", re.U)

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
        self.checks, self.flags, self.data = [], [], {}

    def check(self, name, value, status, note=""):
        self.checks.append({"name": name, "value": value, "status": status, "note": note})

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
        return {f"p{q}": 0.0 for q in qs} | {"max": 0.0}
    a = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"max": float(a.max())}


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).rjust(w[i]) if i else str(h).ljust(w[i]) for i, h in enumerate(headers)))
    for r in rows:
        print("  " + "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)))


def norm_name(s):
    return _RE_NONWORD.sub(" ", unicodedata.normalize("NFC", str(s or "").casefold())).strip()


def norm_text(s):
    return " ".join(str(s or "").replace("_", " ").casefold().split())


def tvd(p, q):
    """Total variation distance giữa hai phân bố đếm."""
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    return 0.5 * sum(abs(p[k] / sp - q[k] / sq) for k in set(p) | set(q))


def load_splits(split_dir):
    out = {}
    for s in SPLITS:
        path = Path(split_dir) / f"{s}.jsonl"
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
    return out


# ---------------------------------------------------------------- 1. toàn vẹn
def sec_integrity(data, rep):
    head("1. TOÀN VẸN DỮ LIỆU")
    all_ids = Counter()
    rows_out, totals = [], Counter()
    for s, rows in data.items():
        miss, bad_pos, bad_img, no_img, empty_seg, no_us = Counter(), 0, 0, 0, 0, 0
        for r in rows:
            pid = r.get("id")
            all_ids[str(pid)] += 1
            for k in REQUIRED:
                v = r.get(k)
                if v is None or (isinstance(v, (str, list)) and len(v) == 0):
                    miss[k] += 1
                    rep.flag(s, pid, f"thieu_{k}")
            imgs = r.get("images") or []
            if not imgs:
                no_img += 1
            else:
                if any(not isinstance(im, dict) or not im.get("local_path") for im in imgs):
                    bad_img += 1
                    rep.flag(s, pid, "anh_thieu_local_path")
                pos = [im.get("position") for im in imgs if isinstance(im, dict)]
                if sorted(p for p in pos if isinstance(p, int)) != list(range(1, len(imgs) + 1)):
                    bad_pos += 1
                    rep.flag(s, pid, "position_khong_lien_tuc", str(pos))
            seg, txt = r.get("text_seg") or "", r.get("text") or ""
            if not seg.strip():
                empty_seg += 1
            elif "_" not in seg and len(txt.split()) >= 8:
                no_us += 1  # text dài mà không có từ ghép nào -> nghi ngờ tách từ không chạy
        n_miss = sum(miss.values())
        rows_out.append([s, f"{len(rows):,}", n_miss, no_img, bad_img, bad_pos, empty_seg, no_us])
        totals.update(miss=n_miss, no_img=no_img, bad_img=bad_img, bad_pos=bad_pos,
                      empty_seg=empty_seg, no_us=no_us)
        rep.data.setdefault("integrity", {})[s] = {"rows": len(rows), "missing_fields": dict(miss),
                                                     "no_image": no_img, "bad_image_meta": bad_img,
                                                     "bad_positions": bad_pos, "empty_text_seg": empty_seg,
                                                     "text_seg_without_underscore": no_us}
    table(["split", "dòng", "thiếu field", "không ảnh", "ảnh lỗi meta", "position sai",
           "text_seg rỗng", "seg không '_'"], rows_out)
    dup = [k for k, n in all_ids.items() if n > 1]
    print(f"\n  id trùng trong toàn bộ dữ liệu: {len(dup)}" + (f"  (vd {dup[:3]})" if dup else ""))
    rep.check("Field bắt buộc đầy đủ", totals["miss"], must_be_zero(totals["miss"]))
    rep.check("Id không trùng", len(dup), must_be_zero(len(dup)))
    rep.check("Sản phẩm đều có ảnh", totals["no_img"], must_be_zero(totals["no_img"]))
    rep.check("Metadata ảnh hợp lệ", totals["bad_img"], must_be_zero(totals["bad_img"]))
    rep.check("position ảnh liên tục từ 1", totals["bad_pos"], grade(totals["bad_pos"], 0))
    rep.check("text_seg không rỗng", totals["empty_seg"], must_be_zero(totals["empty_seg"]))
    n_all = sum(len(r) for r in data.values())
    rep.check("text_seg có từ ghép (tách từ đã chạy)", fp(pct(totals["no_us"], n_all)),
              grade(pct(totals["no_us"], n_all), 0.02, 0.2), "text dài ≥8 từ mà không có '_'")


# ---------------------------------------------------------------- 2. chất lượng split + rò rỉ
def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    pop8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return pop8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1, dtype=np.int16)


def min_hamming(q, ref, chunk=256):
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


def hash_arrays(rows):
    hs, owners = [], []
    for i, r in enumerate(rows):
        for im in r.get("images") or []:
            h = im.get("phash") if isinstance(im, dict) else None
            if h and h != ZERO_HASH:
                try:
                    hs.append(int(h, 16))
                    owners.append(i)
                except ValueError:
                    pass
    return np.array(hs, dtype=np.uint64), np.array(owners, dtype=np.int64)


def sec_split_quality(data, rep, args):
    head("2. CHẤT LƯỢNG SPLIT VÀ RÒ RỈ DỮ LIỆU")
    n_all = sum(len(r) for r in data.values())

    # --- 2.1 tỉ lệ
    print("2.1 Kích thước và tỉ lệ")
    dev = 0.0
    rows_out = []
    for s, rows in data.items():
        share = pct(len(rows), n_all)
        dev = max(dev, abs(share - EXPECTED_RATIO[s]))
        rows_out.append([s, f"{len(rows):,}", fp(share), fp(EXPECTED_RATIO[s], 0)])
    table(["split", "sản phẩm", "tỉ lệ", "mục tiêu"], rows_out)
    rep.check("Tỉ lệ split ≈ 80/10/10", fp(dev, 2) + " lệch tối đa", grade(dev, RATIO_TOL))

    # --- 2.2 phân bố
    print("\n2.2 Phân bố theo nguồn / category (% trong mỗi split)")
    dims = {
        "source": lambda r: r.get("source", "?"),
        "category": lambda r: r.get("category", "?"),
        "source|category": lambda r: f"{r.get('source', '?')}|{r.get('category', '?')}",
    }
    dist = {d: {s: Counter(fn(r) for r in rows) for s, rows in data.items()} for d, fn in dims.items()}
    for d, per in dist.items():
        keys = sorted(set().union(*per.values()))
        table([d] + list(SPLITS),
              [[k] + [f"{per[s][k]:>5,} ({fp(pct(per[s][k], len(data[s])))})" for s in SPLITS] for k in keys])
        print()
    worst = max(tvd(dist["source|category"]["train"], dist["source|category"][s]) for s in ("val", "test"))
    print(f"  TVD (nguồn×category) val/test so với train, lớn nhất: {worst:.4f}  (0 = giống hệt)")
    rep.check("Phân bố nguồn×category val/test ≈ train", f"TVD={worst:.4f}", grade(worst, TVD_WARN))
    small = [(s, k, n) for s in ("val", "test") for k, n in dist["source|category"][s].items() if n < MIN_STRATUM_EVAL]
    if small:
        print(f"  Nhóm nguồn×category quá nhỏ trong val/test (<{MIN_STRATUM_EVAL}): {small}")
    rep.check(f"Mỗi nhóm nguồn×category ≥{MIN_STRATUM_EVAL} sp trong val/test", len(small),
              "WARN" if small else "PASS", "metric theo nhóm sẽ nhiễu nếu nhóm quá nhỏ")
    rep.data["distribution"] = {d: {s: dict(c) for s, c in per.items()} for d, per in dist.items()}

    # --- 2.3 rò rỉ chính xác
    print("\n2.3 Rò rỉ chính xác giữa các split (số sản phẩm val/test có khoá trùng với split trước)")
    leak_defs = {
        "id": lambda r: [str(r.get("id"))],
        "text_group": lambda r: [r.get("text_group")],
        "text_seg (chuẩn hoá)": lambda r: [norm_text(r.get("text_seg"))],
        "URL ảnh": lambda r: [i["url"] for i in r.get("images") or [] if isinstance(i, dict) and i.get("url")],
        "phash trùng hệt": lambda r: [i["phash"] for i in r.get("images") or []
                                       if isinstance(i, dict) and i.get("phash") and i["phash"] != ZERO_HASH],
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

    # --- 2.4 ảnh gần trùng (Hamming trên phash)
    print(f"\n2.4 Ảnh gần trùng giữa các split (Hamming trên phash 64-bit)")
    arr = {s: hash_arrays(data[s]) for s in SPLITS}
    n_hash = sum(len(v[0]) for v in arr.values())
    n_img = sum(len(r.get("images") or []) for rows in data.values() for r in rows)
    print(f"  Ảnh có phash: {n_hash:,}/{n_img:,}")
    rep.check("Ảnh có phash", fp(pct(n_hash, n_img)), grade(1 - pct(n_hash, n_img), 0.01, 0.5),
              "thiếu phash = không gom được ảnh gần trùng khi chia split")
    thresholds = (0, 4, 8)
    rows_out = []
    for q, ref_splits in (("val", ("train",)), ("test", ("train",)), ("test", ("val",))):
        qh, qo = arr[q]
        refh = np.concatenate([arr[s][0] for s in ref_splits]) if ref_splits else np.array([], dtype=np.uint64)
        refo = [(s, i) for s in ref_splits for i in arr[s][1]]
        d, a = min_hamming(qh, refh)
        row = [f"{q} → {'+'.join(ref_splits)}"]
        for t in thresholds:
            prods = set(qo[d <= t].tolist())
            row.append(f"{len(prods)} ({fp(pct(len(prods), len(data[q])), 2)})")
        rows_out.append(row)
        if ref_splits == ("train",):
            near = pct(len(set(qo[d <= args.near_hamming].tolist())), len(data[q]))
            rep.check(f"Ảnh gần trùng (≤{args.near_hamming} bit) {q}→train", fp(near, 2),
                      grade(near, NEAR_DUP_WARN, NEAR_DUP_FAIL), "làm R@k cao ảo")
            for k in np.where(d <= args.near_hamming)[0]:
                rs, ri = refo[int(a[k])]
                rep.flag(q, data[q][int(qo[k])].get("id"), "anh_gan_trung_train",
                         f"d={int(d[k])} với train id={data['train'][ri].get('id')}")
    table(["cặp", "= 0 bit", "≤ 4 bit", "≤ 8 bit"], rows_out)
    print("  (số sản phẩm có ít nhất 1 ảnh gần trùng; ≤4 bit thường là cùng một ảnh)")

    # --- 2.5 tên trùng
    print("\n2.5 Tên sản phẩm trùng giữa các split (sau chuẩn hoá bỏ số, dấu câu)")
    names = {s: Counter(norm_name(r.get("name")) for r in data[s]) for s in SPLITS}
    rows_out = []
    for q, ref in (("val", ("train",)), ("test", ("train", "val"))):
        seen = set().union(*(set(names[s]) for s in ref))
        hit = sum(1 for r in data[q] if norm_name(r.get("name")) in seen)
        rows_out.append([f"{q} → {'+'.join(ref)}", hit, fp(pct(hit, len(data[q])))])
        rep.check(f"Tên trùng với split trước ({q})", fp(pct(hit, len(data[q]))),
                  grade(pct(hit, len(data[q])), NAME_LEAK_WARN),
                  "cùng tên có thể là cùng mẫu bán ở shop khác")
    table(["cặp", "sản phẩm", "tỉ lệ"], rows_out)

    # --- 2.6 trùng trong cùng split (làm nhiễu đánh giá)
    print("\n2.6 Sản phẩm trùng ảnh hoặc trùng text với sản phẩm khác CÙNG split")
    rows_out = []
    for s in SPLITS:
        hs, ow = arr[s]
        owners_by_hash = defaultdict(set)
        for h, o in zip(hs.tolist(), ow.tolist()):
            owners_by_hash[h].add(o)
        tied_img = {o for v in owners_by_hash.values() if len(v) > 1 for o in v}
        groups = Counter(r.get("text_group") for r in data[s])
        tied_txt = sum(n for n in groups.values() if n > 1)
        rows_out.append([s, len(tied_img), fp(pct(len(tied_img), len(data[s])), 2), tied_txt,
                         fp(pct(tied_txt, len(data[s])), 2)])
        if s != "train":
            rate = pct(len(tied_img | set()), len(data[s]))
            rep.check(f"Trùng ảnh trong cùng split ({s})", fp(rate, 2), grade(rate, TIE_WARN),
                      "2 sản phẩm cùng ảnh → query ảnh không phân biệt được")
            rate = pct(tied_txt, len(data[s]))
            rep.check(f"Trùng text trong cùng split ({s})", fp(rate, 2), grade(rate, TIE_WARN))
    table(["split", "trùng ảnh", "%", "trùng text_group", "%"], rows_out)


# ---------------------------------------------------------------- 3. ảnh
def inspect_image(path):
    from PIL import Image
    path = Path(path)
    if not path.exists():
        return {"error": "missing"}
    try:
        with Image.open(path) as im:
            fmt, size, mode = im.format, im.size, im.mode
            im.load()
            g = np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception as e:  # file hỏng / cắt cụt
        return {"error": f"unreadable: {type(e).__name__}"}
    mask = g < 245
    if mask.any():
        ys, xs = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        content = float((ys[-1] - ys[0] + 1) * (xs[-1] - xs[0] + 1) / g.size)
    else:
        content = 0.0
    return {"fmt": fmt, "size": size, "mode": mode, "bytes": path.stat().st_size,
            "std": float(g.std()), "content": content}


def sec_images(data, rep, args):
    head("3. ẢNH")
    root = Path(args.image_root)
    rng = random.Random(0)
    rows_out = []
    for s, rows in data.items():
        jobs = [(r.get("id"), root / im["local_path"], im) for r in rows
                for im in r.get("images") or [] if isinstance(im, dict) and im.get("local_path")]
        total = len(jobs)
        if args.max_images and total > args.max_images:
            jobs = rng.sample(jobs, args.max_images)
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(tqdm(ex.map(lambda j: inspect_image(j[1]), jobs), total=len(jobs), desc=f"ảnh {s}"))
        c = Counter()
        sizes, contents, nbytes = [], [], []
        for (pid, path, im), x in zip(jobs, res):
            if "error" in x:
                c[x["error"].split(":")[0]] += 1
                rep.flag(s, pid, f"anh_{x['error'].split(':')[0]}", str(path))
                continue
            if path.suffix.lower() not in VALID_EXT or x["fmt"] not in VALID_FORMATS:
                c["bad_format"] += 1
                rep.flag(s, pid, "anh_sai_dinh_dang", f"{path.suffix} / {x['fmt']}: {path}")
            if x["size"] != (args.image_size, args.image_size):
                c["wrong_size"] += 1
                sizes.append(x["size"])
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
        n = len(jobs)
        ok = n - c["missing"] - c["unreadable"]
        rows_out.append([s, f"{n:,}" + ("*" if n < total else ""), c["missing"], c["unreadable"], c["bad_format"],
                         c["wrong_size"], c["not_rgb"], f"{c['flat']} ({fp(pct(c['flat'], ok), 2)})",
                         f"{c['tiny']} ({fp(pct(c['tiny'], ok))})",
                         f"{np.median(nbytes) / 1024:.0f}KB" if nbytes else "-"])
        for key, name in (("missing", "Ảnh không thiếu file"), ("unreadable", "Ảnh đọc được"),
                          ("bad_format", "Ảnh chỉ jpg/png (không webp)")):
            rep.check(f"{name} ({s})", c[key], must_be_zero(c[key]))
        rep.check(f"Ảnh đúng {args.image_size}×{args.image_size} ({s})", c["wrong_size"], grade(c["wrong_size"], 0))
        rep.check(f"Ảnh trắng/đồng màu ({s})", fp(pct(c["flat"], ok), 2), grade(pct(c["flat"], ok), FLAT_WARN))
        rep.check(f"Sản phẩm quá nhỏ trong khung ({s})", fp(pct(c["tiny"], ok)),
                  grade(pct(c["tiny"], ok), TINY_CONTENT_WARN), "vùng không-trắng < 10% khung ảnh")
        rep.data.setdefault("images", {})[s] = {"inspected": n, "total": total, **dict(c),
                                                 "content_ratio": pcts(contents, (5, 50))}
    table(["split", "ảnh đọc", "thiếu", "hỏng", "sai định dạng", "sai size", "không RGB", "trắng", "quá nhỏ",
           "dung lượng"], rows_out)
    if args.max_images:
        print(f"  (* = chỉ đọc mẫu {args.max_images} ảnh/split)")

    print("\nSố ảnh mỗi sản phẩm (từ metadata, toàn bộ):")
    rows_out = []
    for s, rows in data.items():
        cnt = Counter(min(len(r.get("images") or []), 6) for r in rows)
        multi = sum(v for k, v in cnt.items() if k >= 2)
        rows_out.append([s] + [f"{cnt[k]:,}" for k in range(1, 7)] + [f"{multi:,} ({fp(pct(multi, len(rows)))})"])
    table(["split", "1", "2", "3", "4", "5", "6+", "≥2 ảnh"], rows_out)


# ---------------------------------------------------------------- 4. text
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
            print(f"  (không nạp được tokenizer: {type(e).__name__}; dùng ước lượng 1.3 token/từ)")
    return np.array([int(len(t.split()) * 1.3) + 2 for t in texts]), "ƯỚC LƯỢNG 1.3 token/từ"


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


def sec_text(data, rep, args):
    head("4. TEXT")
    all_rows = [(s, r) for s, rows in data.items() for r in rows]
    lens, how = token_lengths([r.get("text_seg") or "" for _, r in all_rows], args.tokenizer, not args.no_tokenizer)
    limit = args.max_text_length
    print(f"Độ dài token của text_seg ({how}), giới hạn model = {limit}")
    rows_out, by_split = [], defaultdict(list)
    for (s, _), n in zip(all_rows, lens):
        by_split[s].append(n)
    for s in SPLITS:
        v = np.array(by_split[s])
        words = [len((r.get("text") or "").split()) for r in data[s]]
        q = pcts(v)
        trunc = pct(int((v > limit).sum()), len(v))
        rows_out.append([s, f"{np.mean(v):.0f}", f"{q['p50']:.0f}", f"{q['p90']:.0f}", f"{q['p95']:.0f}",
                         f"{q['p99']:.0f}", f"{q['max']:.0f}", fp(trunc), f"{np.median(words):.0f}"])
        rep.check(f"Text bị cắt ở {limit} token ({s})", fp(trunc), grade(trunc, TRUNC_WARN),
                  "phần cuối mô tả bị bỏ; tên luôn nằm đầu nên không mất" if trunc else "")
        rep.data.setdefault("text", {})[s] = {"tokens": q, "truncated_rate": trunc, "method": how}
    table(["split", "TB", "p50", "p90", "p95", "p99", "max", "bị cắt", "từ (p50)"], rows_out)

    # ---- rác còn sót, text ngắn, tách từ
    print("\nRác còn sót trong `text` và chất lượng text (số sản phẩm):")
    checks = {
        "HTML": lambda r, t: _RE_HTML.search(t),
        "URL": lambda r, t: _RE_URL.search(t),
        "SĐT": lambda r, t: _RE_PHONE.search(t),
        "emoji": lambda r, t: _RE_EMOJI.search(t),
        "ký tự ngoại ngữ": lambda r, t: _RE_FOREIGN.search(t),
        "mã SKU sót": lambda r, t: _RE_CODE.search(t),
        "VIẾT HOA quá nhiều": lambda r, t: (lambda L: len(L) >= 10 and sum(c.isupper() for c in L) / len(L) > 0.6)(
            [c for c in t if c.isalpha()]),
        f"quá ngắn (<{SHORT_TEXT_WORDS} từ)": lambda r, t: len(t.split()) < SHORT_TEXT_WORDS,
        "tách từ lệch": lambda r, t: _seg_mismatch(r, t),
    }
    rows_out = []
    for s in SPLITS:
        c = Counter()
        for r in data[s]:
            t = str(r.get("text") or "")
            for label, fn in checks.items():
                if fn(r, t):
                    c[label] += 1
                    if label in ("quá ngắn (<%d từ)" % SHORT_TEXT_WORDS, "tách từ lệch"):
                        rep.flag(s, r.get("id"), label.replace(" ", "_"), t[:80])
        rows_out.append([s] + [f"{c[k]} ({fp(pct(c[k], len(data[s])), 1)})" for k in checks])
        for label in ("HTML", "URL", "SĐT", "emoji", "ký tự ngoại ngữ", "mã SKU sót"):
            rep.check(f"Rác '{label}' trong text ({s})", fp(pct(c[label], len(data[s])), 2),
                      grade(pct(c[label], len(data[s])), JUNK_WARN))
        short = pct(c[f"quá ngắn (<{SHORT_TEXT_WORDS} từ)"], len(data[s]))
        rep.check(f"Text quá ngắn ({s})", fp(short, 2), grade(short, JUNK_WARN))
        mis = pct(c["tách từ lệch"], len(data[s]))
        rep.check(f"text_seg khớp text ({s})", fp(mis, 2), grade(mis, JUNK_WARN, 0.2))
    print()
    for i, label in enumerate(checks):
        print(f"  [{i + 1}] {label}")
    table(["split"] + [f"[{i + 1}]" for i in range(len(checks))], rows_out)

    # ---- mô tả, từ phổ biến
    train_words = Counter(w for r in data["train"] for w in (r.get("text_seg") or "").split())
    print(f"\n  Từ vựng train: {len(train_words):,} từ khác nhau. 20 từ phổ biến nhất (kiểm tra nhiễu bằng mắt):")
    print("  " + ", ".join(f"{w}({n})" for w, n in train_words.most_common(20)))
    desc_empty = {s: sum(1 for r in rows if not str(r.get("description") or "").strip()) for s, rows in data.items()}
    print("  Sản phẩm không có mô tả: " + ", ".join(f"{s}={n} ({fp(pct(n, len(data[s])))})" for s, n in desc_empty.items()))

    # ---- category lệch tên
    print("\nCategory có thể sai so với tên sản phẩm (heuristic từ khoá, cần kiểm tra mắt):")
    rows_out = []
    for s in SPLITS:
        bad = [r for r in data[s] if suspicious_category(r)]
        rows_out.append([s, len(bad), fp(pct(len(bad), len(data[s])), 2)])
        for r in bad:
            rep.flag(s, r.get("id"), "category_nghi_sai", f"{r.get('category')} | {r.get('name')}")
        rep.check(f"Category nghi sai theo tên ({s})", fp(pct(len(bad), len(data[s])), 2),
                  grade(pct(len(bad), len(data[s])), SUSPECT_CAT_WARN))
        for r in bad[:3]:
            print(f"    [{s}] {r.get('category')} | {r.get('name')}")
    table(["split", "nghi sai", "%"], rows_out)


def _seg_mismatch(r, t):
    seg = str(r.get("text_seg") or "")
    n_txt = min(len(t.split()), MAX_SEG_SYLLABLES)
    n_seg = len(seg.replace("_", " ").split())
    return n_txt >= 8 and not (0.7 <= n_seg / n_txt <= 1.6)


# ---------------------------------------------------------------- 5. metadata
def to_price(p):
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return float(p)
    if isinstance(p, str):
        digits = re.sub(r"[^\d]", "", p)
        return float(digits) if digits else None
    return None


def sec_metadata(data, rep):
    head("5. METADATA (giá, màu, shop)")
    rows_out = []
    for s, rows in data.items():
        prices = [to_price(r.get("price")) for r in rows]
        bad_price = sum(1 for p in prices if p is None or p <= 0)
        no_color = sum(1 for r in rows if not r.get("colors"))
        shops = Counter(r.get("shop") for r in rows)
        top_share = pct(shops.most_common(1)[0][1], len(rows)) if shops else 0
        rows_out.append([s, f"{bad_price} ({fp(pct(bad_price, len(rows)))})", f"{no_color} ({fp(pct(no_color, len(rows)))})",
                         len(shops), fp(top_share)])
        rep.check(f"Giá thiếu hoặc ≤0 ({s})", fp(pct(bad_price, len(rows))), grade(pct(bad_price, len(rows)), MISSING_META_WARN))
        rep.check(f"Không có màu ({s})", fp(pct(no_color, len(rows))), grade(pct(no_color, len(rows)), 0.5),
                  "màu thiếu thì text kém thông tin")
    table(["split", "giá lỗi", "không màu", "số shop", "shop lớn nhất"], rows_out)

    print("\nGiá (VND, bỏ giá lỗi) theo nguồn, train:")
    by_src = defaultdict(list)
    for r in data["train"]:
        p = to_price(r.get("price"))
        if p and p > 0:
            by_src[r.get("source", "?")].append(p)
    table(["nguồn", "n", "p1", "p50", "p99", "max"],
          [[k, f"{len(v):,}"] + [f"{np.percentile(v, q):,.0f}" for q in (1, 50, 99)] + [f"{max(v):,.0f}"]
           for k, v in sorted(by_src.items())])
    shops_train = {r.get("shop") for r in data["train"]}
    for s in ("val", "test"):
        seen = sum(1 for r in data[s] if r.get("shop") in shops_train)
        print(f"  {s}: {fp(pct(seen, len(data[s])))} sản phẩm thuộc shop đã có trong train")


# ---------------------------------------------------------------- 6. giá trị cho retrieval
def expected_pairs(keys, batch):
    n = len(keys)
    if n < 2:
        return 0.0
    pairs = sum(v * (v - 1) / 2 for v in Counter(keys).values())
    return pairs * batch * (batch - 1) / (n * (n - 1))


def conf_key(r):
    w = norm_name(r.get("name")).split()
    return " ".join(w[:5]) if len(w) >= 3 else " ".join(w)


def sec_task_value(data, rep, args):
    head("6. GIÁ TRỊ DỮ LIỆU CHO BÀI TOÁN RETRIEVAL (ảnh ↔ text)")
    rows_out = []
    for s, rows in data.items():
        n_img = sum(len(r.get("images") or []) for r in rows)
        multi = sum(1 for r in rows if len(r.get("images") or []) >= 2)
        keys = [conf_key(r) for r in rows]
        share = pct(sum(v for v in Counter(keys).values() if v > 1), len(rows))
        pairs_batch = expected_pairs(keys, args.batch_size)
        rows_out.append([s, f"{len(rows):,}", f"{n_img:,}", f"{n_img / max(len(rows), 1):.2f}", f"{multi:,}",
                         fp(share), f"{pairs_batch:.2f}", f"{1 / max(len(rows), 1) * 100:.3f}%",
                         f"{min(10, len(rows)) / max(len(rows), 1) * 100:.2f}%"])
        rep.data.setdefault("task_value", {})[s] = {
            "products": len(rows), "images": n_img, "products_with_2plus_images": multi,
            "confusable_share": share, "expected_confusable_pairs_per_batch": pairs_batch}
    table(["split", "sản phẩm", "ảnh", "ảnh/sp", "sp ≥2 ảnh", "tên giống nhau", "cặp nhầm/batch",
           "R@1 ngẫu nhiên", "R@10 ngẫu nhiên"], rows_out)
    print(f"""
  Cách đọc:
  - 'sp ≥2 ảnh' ở val/test = số query ảnh→ảnh có thể đánh giá (query là ảnh phụ, gallery là ảnh đại diện).
  - 'tên giống nhau' = tỉ lệ sản phẩm có 5 từ đầu của tên trùng sản phẩm khác cùng split
    (loại sản phẩm gần giống nhau: ứng viên negative sai trong InfoNCE).
  - 'cặp nhầm/batch' = số cặp sản phẩm gần giống kỳ vọng rơi vào cùng 1 batch {args.batch_size} khi train.
    Nếu ≳1 thì contrastive loss thường xuyên phạt oan các cặp này.
  - R@k ngẫu nhiên = mốc tối thiểu: model phải vượt xa con số này trên gallery val/test.""")
    train_pairs = rep.data["task_value"]["train"]["expected_confusable_pairs_per_batch"]
    rep.check(f"Cặp sản phẩm gần giống / batch {args.batch_size} (train)", f"{train_pairs:.2f}",
              "WARN" if train_pairs > 1.0 else "PASS", "negative sai trong InfoNCE")
    for s in ("val", "test"):
        n_eval = rep.data["task_value"][s]["products_with_2plus_images"]
        rep.check(f"Số query ảnh→ảnh đánh giá được ({s})", n_eval, "WARN" if n_eval < 200 else "PASS",
                  "ít hơn 200 query thì R@k dao động mạnh")
        rep.check(f"Kích thước gallery ({s})", len(data[s]), "WARN" if len(data[s]) < 500 else "PASS")


# ---------------------------------------------------------------- tổng kết
def summary(data, rep, args, elapsed):
    head("TỔNG KẾT CÁC MỤC KIỂM TRA")
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    shown = sorted(rep.checks, key=lambda c: order[c["status"]])
    non_pass = [c for c in shown if c["status"] != "PASS"]
    for c in non_pass:
        note = f"  — {c['note']}" if c["note"] else ""
        print(f"  [{c['status']}] {c['name']}: {c['value']}{note}")
    cnt = Counter(c["status"] for c in rep.checks)
    print(f"\n  {cnt['PASS']} PASS | {cnt['WARN']} WARN | {cnt['FAIL']} FAIL  (tổng {len(rep.checks)} mục)")

    n_all = sum(len(r) for r in data.values())
    n_img = sum(len(r.get("images") or []) for rows in data.values() for r in rows)
    print(f"\n  Dữ liệu: {n_all:,} sản phẩm, {n_img:,} ảnh, nguồn: "
          + ", ".join(f"{k}={v}" for k, v in Counter(r.get('source') for rows in data.values() for r in rows).items()))
    if cnt["FAIL"]:
        verdict = "KHÔNG ĐẠT: sửa các mục FAIL (xem validation_flags.csv) trước khi train."
    elif cnt["WARN"]:
        verdict = "ĐẠT, CÓ CẢNH BÁO: dùng được; xem các mục WARN để biết rủi ro khi diễn giải kết quả."
    else:
        verdict = "ĐẠT: dữ liệu sạch theo mọi tiêu chí đã kiểm tra."
    print(f"  Kết luận: {verdict}")
    print(f"  Thời gian chạy: {elapsed:.1f}s")
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
    ap = argparse.ArgumentParser(description="Đánh giá chất lượng dữ liệu data/splits")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--image-root", default="data/processed", help="thư mục chứa images/ (local_path tính từ đây)")
    ap.add_argument("--out-dir", default="data/validation")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-text-length", type=int, default=256, help="MAX_TEXT_LENGTH của model")
    ap.add_argument("--tokenizer", default="vinai/phobert-base")
    ap.add_argument("--no-tokenizer", action="store_true", help="không nạp tokenizer, ước lượng độ dài token")
    ap.add_argument("--near-hamming", type=int, default=4, help="ngưỡng Hamming coi là ảnh gần trùng")
    ap.add_argument("--batch-size", type=int, default=64, help="batch size lúc train (ước lượng negative sai)")
    ap.add_argument("--skip-images", action="store_true", help="bỏ qua phần đọc file ảnh")
    ap.add_argument("--max-images", type=int, default=0, help="chỉ đọc mẫu N ảnh mỗi split (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    data = load_splits(args.splits_dir)
    rep = Report()
    sec_integrity(data, rep)
    sec_split_quality(data, rep, args)
    if args.skip_images:
        head("3. ẢNH")
        print("  Bỏ qua (--skip-images).")
    else:
        sec_images(data, rep, args)
    sec_text(data, rep, args)
    sec_metadata(data, rep)
    sec_task_value(data, rep, args)
    n_fail = summary(data, rep, args, time.time() - t0)
    save_outputs(rep, args.out_dir)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()