"""
V7b: Kiểm tra lập luận "dịch máy làm mất ngữ nghĩa" (không cần GPU, không cần ảnh)
Đầu vào : data/annotation/benchmark/queries_200.csv   (query_id, category, attrs_in_query, query_text)
          data/annotation/benchmark/translations.csv  (do V7 tạo: query_id, query_vi, query_en)
Đầu ra  : data/annotation/benchmark/translation_audit.csv   (từng truy vấn: thuộc tính giữ/mất, cờ mt_loss_flag)
          results/logs/log_B.md                              (tóm tắt)
Chạy    : python v7b_translation_audit.py

Cách đo: với mỗi thuộc tính trong attrs_in_query (vd. "chất liệu=da bò; màu=xanh navy") và danh từ chính của sản phẩm
(áo thun, ví, dép...), kiểm tra bản dịch tiếng Anh có chứa từ tương đương không (từ điển regex ở dưới).
  kept          bản dịch có từ tương đương đúng nghĩa
  partial       chỉ có từ gần nghĩa/rộng hơn (vd. "da bò" → "leather")
  mistranslated có từ sai nghĩa đã biết (vd. "vải dù" → "umbrella", "xanh dương" → "green")
  lost          không thấy gì tương đương
  unchecked     giá trị thuộc tính chưa có trong từ điển (được liệt kê ở cuối để bạn bổ sung)

GIỚI HẠN (ghi vào báo cáo): đây là phép kiểm tra bằng từ điển, không phải đánh giá của người. Nó đo "bản dịch có giữ từ khóa
thuộc tính không", không đo "CLIP có hiểu từ đó không". Nên đọc bằng mắt toàn bộ các dòng mt_loss_flag=1 (và một mẫu
dòng =0) trước khi dựa vào con số. Cột mt_loss_flag dùng để tách điểm theo từng truy vấn của V11 thành hai nhóm
(bản dịch giữ đủ / bản dịch mất ý) khi so clip_vi2en, fashionclip_vi2en với viclip_ot.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()

# value_vi (viết thường) -> (regex_đúng_nghĩa, regex_gần_nghĩa | None, regex_sai_nghĩa | None)
ATTR_RULES = {
    # ── chất liệu ──
    "polyester": (r"\bpoly(ester|amide)?\b", None, None),
    "nylon": (r"\bnylon\b|\bpolyamide\b", None, None),
    "cotton": (r"\bcotton\b", None, None),
    "len": (r"\bwool(en|len)?\b|\bknit(ted|wear)?\b|\bsweater\b|\bcashmere\b|\bjumper\b", None, None),
    "da": (r"\bleather\b|\bhide\b", None, None),
    "da bò": (r"\bcow(hide|\s*leather)?\b|\bcattle\b|\bbovine\b|\boxhide\b", r"\bleather\b", None),
    "da pu": (r"\bpu\b|faux leather|artificial leather|synthetic leather|vegan leather|imitation leather", r"\bleather\b", None),
    "da lộn": (r"\bsuede\b|\bnubuck\b", r"\bleather\b", None),
    "vải dù": (r"parachute|\bnylon\b|taslan|ripstop", r"windbreaker|waterproof|lightweight", r"umbrella"),
    "kaki": (r"\bkha?ki\b|\bchinos?\b", None, None),
    "denim": (r"\bdenim\b|\bjeans?\b", None, r"\bcowboy\b|\bcow\b|\bcowhide\b|\bbuffalo\b"),
    "nỉ": (r"\bfleece|\bfelt\b|\bflannel|\bsweat", r"\bwool|\bjersey|\bknit", None),
    "lụa": (r"\bsilk|\bsatin", None, None),
    "nhung tăm": (r"corduroy|\bcord\b", r"velvet|velour", None),
    "linen": (r"\blinen\b|\bflax\b", None, r"\bhemp\b"),
    "bamboo": (r"\bbamboo\b", None, None),
    "acrylic": (r"\bacrylic\b", None, None),
    "lưới": (r"\bmesh\b|\bnet(ting)?\b", None, None),
    "canvas": (r"\bcanvas\b", r"\bcloth\b|\bfabric\b", None),
    "nhựa": (r"\bplastic\b|\bresin\b|\bacetate\b|\bpc\b|\btr90\b", None, None),
    # ── màu ──
    "đen": (r"\bblack\b", None, None),
    "trắng": (r"\bwhite\b", None, None),
    "xám": (r"\bgr[ae]y\b|\bsilver\b|\bash\b|\bcharcoal\b", None, None),
    "nâu": (r"\bbrown\b|\bcoffee\b|\bcamel\b|\btan\b|\bchocolate\b", None, None),
    "đỏ": (r"\bred\b|\bburgundy\b|\bmaroon\b|\bcrimson\b", None, None),
    "cam": (r"\borange\b", None, None),
    "be": (r"\bbeige\b|\bcream\b|\bkhaki\b|\bnude\b|\btan\b|\boff-?white\b|\bivory\b", None, None),
    "kem": (r"\bcream\b|\bivory\b|\boff-?white\b|\bbeige\b", None, None),
    "xanh dương": (r"\bblue\b", None, r"\bgreen\b"),
    "xanh navy": (r"\bnavy\b|dark[- ]blue|deep blue|midnight|indigo|marine", r"\bblue\b", None),
    "xanh đậm": (r"dark[- ]blue|deep blue|navy|\bdark\b|\bdeep\b|indigo", r"\bblue\b", None),
    "xanh lá": (r"\bgreen\b", None, None),
    "xanh rêu": (r"\bmoss\b|\bolive\b|\barmy\b|\bmilitary\b|\bforest\b|dark green|khaki green", r"\bgreen\b", None),
    # ── phong cách ──
    "thể thao": (r"\bsport(s|y|swear)?\b|\bathletic\b|\bactive\b|\bdynamic\b|\benergetic\b|\bgym\b", None, None),
    "casual": (r"\bcasual\b|\beveryday\b|\bdaily\b|\bordinary\b|\bnormal\b|\bsimple\b|\brelaxed\b|\bleisure\b|\blaid-?back\b", None, None),
    "basic": (r"\bbasic\b|\bplain\b|\bsimple\b|\bessential\b|\bsolid\b", None, None),
    "vintage": (r"\bvintage\b|\bretro\b|\bclassic\b|\bold[- ]school\b|\bantique\b|\bnostalgic\b|\bold\b", None, None),
    "minimalist": (r"\bminimal(ist|ism)?\b|\bsimple\b|\bsimplicity\b|\bsimplistic\b", None, None),
    "công sở": (r"\boffice\b|\bbusiness\b|\bformal\b|\bprofessional\b|\bwork(place|wear)?\b|\bcorporate\b", None, None),
    "streetwear": (r"\bstreet(wear)?\b|\burban\b|\bhip-?hop\b", None, None),
    # ── kiểu dáng ──
    "oversize": (r"\boversize(d)?\b|\bloose\b|\bwide\b|\bbaggy\b|\brelaxed\b|\bboxy\b|\blarge\b|\bbig\b", None, None),
    "slim fit": (r"\bslim\b|\bfitt?ed\b|\btight\b|\bskinny\b|\bbody-?fit\b|\bsnug\b|\btailored\b|close[- ]fitting", None, None),
    "regular fit": (r"\bregular\b|\bstandard\b|\bstraight\b|\bclassic\b|\bnormal\b|\bmedium\b|\bmoderate\b", None, None),
}

# (regex tiếng Việt trên truy vấn, regex tiếng Anh cần thấy, nhãn) — khớp luật đầu tiên, cụ thể trước chung sau
HEAD_RULES = [
    (r"áo (thun|phông)", r"t-?shirt|\btees?\b|shirt|\btop\b", "áo thun"),
    (r"áo ba lỗ|áo tanktop|tank ?top", r"tank|singlet|sleeveless|vest|undershirt|\btop\b|t-?shirt", "áo ba lỗ"),
    (r"áo sơ mi|sơ mi", r"shirt|blouse", "áo sơ mi"),
    (r"áo polo", r"polo", "áo polo"),
    (r"áo hoodie", r"hood|sweatshirt|sweater", "áo hoodie"),
    (r"áo bomber", r"bomber|jacket|coat", "áo bomber"),
    (r"áo (khoác|gió)", r"jacket|coat|windbreaker|outerwear|cardigan|windcheater|parka|blazer|hoodie", "áo khoác"),
    (r"áo (vest|len)", r"vest|suit|blazer|jacket|waistcoat|sweater|knit|jumper|pullover|cardigan|wool", "áo vest/len"),
    (r"quần (short|đùi)", r"short", "quần short"),
    (r"quần (boxer|lót|sịp)", r"boxer|brief|underwear|panties|trunks|underpants", "quần lót"),
    (r"quần (jean|jeans|bò)", r"jean|denim|pants|trousers", "quần jean"),
    (r"quần kaki", r"kha?ki|chino|pants|trousers", "quần kaki"),
    (r"quần jogger", r"jogger|sweatpants|pants|trousers", "quần jogger"),
    (r"quần (âu|tây|dài|thể thao)", r"trousers|pants|slacks|bottoms", "quần dài"),
    (r"(^|\s)(ví|bóp)(\s|$)", r"wallet|purse|billfold", "ví"),
    (r"túi", r"bag|pouch|purse|satchel|messenger|tote|fanny|waist|sling", "túi"),
    (r"ba lô|balo", r"backpack|rucksack|knapsack|bag", "ba lô"),
    (r"giày", r"shoe|sneaker|loafer|boot|footwear|trainer", "giày"),
    (r"dép", r"sandal|slipper|flip|slides|footwear|shoe", "dép"),
    (r"thắt lưng|dây nịt", r"belt", "thắt lưng"),
    (r"cà vạt", r"\btie\b|necktie", "cà vạt"),
    (r"nón|mũ", r"\bhat\b|\bcap\b|beanie", "nón"),
    (r"kính", r"glass|spectacle|eyewear", "kính"),
]

VI_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
STATUSES = ["kept", "partial", "mistranslated", "lost", "unchecked"]


def parse_attrs(s):
    out = []
    for part in (s or "").split(";"):
        if "=" in part:
            t, v = part.split("=", 1)
            out.append((t.strip(), v.strip()))
    return out


def check_attr(en, value):
    rule = ATTR_RULES.get(value.lower())
    if rule is None:
        return "unchecked"
    strict, loose, bad = rule
    if re.search(strict, en):
        return "kept"
    if loose and re.search(loose, en):
        return "partial"
    if bad and re.search(bad, en):
        return "mistranslated"
    return "lost"


def check_head(vi, en):
    """Trả (nhãn, 'kept'|'lost') hoặc (None, 'na') nếu không có luật nào khớp."""
    for vi_re, en_re, label in HEAD_RULES:
        if re.search(vi_re, vi):
            return label, ("kept" if re.search(en_re, en) else "lost")
    return None, "na"


def audit_one(vi, en, attrs, category=""):
    vi_l, en_l = vi.lower(), (en or "").lower()
    res = [(t, v, check_attr(en_l, v)) for t, v in attrs]
    head_label, head = check_head(vi_l, en_l)
    untranslated = 1 if VI_CHARS.search(en_l) else 0
    n_bad = sum(1 for _, _, s in res if s in ("lost", "mistranslated"))
    flag = 1 if (n_bad > 0 or head == "lost" or untranslated) else 0
    return res, head_label, head, untranslated, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "data/annotation/benchmark/queries_200.csv"))
    ap.add_argument("--translations", default=str(ROOT / "data/annotation/benchmark/translations.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/annotation/benchmark/translation_audit.csv"))
    ap.add_argument("--log", default=str(ROOT / "results/logs/log_B.md"))
    args = ap.parse_args()

    if not Path(args.translations).exists():
        raise SystemExit(f"Chưa có {args.translations}. Chạy V7 với clip_vi2en (hoặc fashionclip_vi2en) trước để tạo bản dịch.")
    with open(args.queries, encoding="utf-8-sig", newline="") as f:
        queries = list(csv.DictReader(f))
    with open(args.translations, encoding="utf-8-sig", newline="") as f:
        tr = {r["query_id"].strip(): r["query_en"] for r in csv.DictReader(f)}
    missing = [q["query_id"] for q in queries if q["query_id"].strip() not in tr]
    if missing:
        raise SystemExit(f"translations.csv thiếu {len(missing)} query_id (vd. {missing[:5]}). Chạy lại V7 để dịch đủ.")

    rows, by_type, by_cat, head_stat = [], defaultdict(Counter), defaultdict(lambda: [0, 0]), Counter()
    unchecked_vals = Counter()
    for q in queries:
        qid, vi, en = q["query_id"].strip(), q["query_text"], tr[q["query_id"].strip()]
        attrs = parse_attrs(q.get("attrs_in_query", ""))
        res, head_label, head, untr, flag = audit_one(vi, en, attrs)
        for t, v, s in res:
            by_type[t][s] += 1
            if s == "unchecked":
                unchecked_vals[f"{t}={v}"] += 1
        head_stat[head] += 1
        c = by_cat[q.get("category", "")]
        c[0] += 1
        c[1] += flag
        cnt = Counter(s for _, _, s in res)
        rows.append({
            "query_id": qid, "category": q.get("category", ""),
            "n_attrs": len(res), **{s: cnt.get(s, 0) for s in STATUSES},
            "problems": "; ".join(f"{t}={v}:{s}" for t, v, s in res if s in ("partial", "mistranslated", "lost")),
            "head_noun": head_label or "", "head_status": head, "untranslated_vi": untr,
            "mt_loss_flag": flag, "query_vi": vi, "query_en": en,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot = Counter()
    for c in by_type.values():
        tot.update(c)
    n_checked = sum(tot[s] for s in ("kept", "partial", "mistranslated", "lost"))
    pct = lambda k, n: f"{100 * k / n:.1f}%" if n else "n/a"
    n_flag = sum(r["mt_loss_flag"] for r in rows)
    n_all_ok = sum(1 for r in rows if r["n_attrs"] and r["kept"] == r["n_attrs"])

    lines = [f"\n## V7b — kiểm tra mất ngữ nghĩa khi dịch ({datetime.now():%Y-%m-%d %H:%M})",
             f"- {len(rows)} truy vấn, {n_checked} thuộc tính đã kiểm (chưa có trong từ điển: {tot['unchecked']})",
             f"- thuộc tính giữ nguyên nghĩa: {pct(tot['kept'], n_checked)}; chỉ gần nghĩa: {pct(tot['partial'], n_checked)}; "
             f"sai nghĩa: {pct(tot['mistranslated'], n_checked)}; mất: {pct(tot['lost'], n_checked)}",
             f"- truy vấn giữ đủ mọi thuộc tính: {n_all_ok}/{len(rows)}; truy vấn có cờ mt_loss_flag=1: {n_flag}/{len(rows)} "
             f"(mất/sai thuộc tính, mất danh từ chính hoặc còn chữ Việt chưa dịch)",
             f"- danh từ chính trong bản dịch: giữ {head_stat['kept']}, mất {head_stat['lost']}, không có luật {head_stat['na']}"]
    print("\n".join(lines))
    print("\nTheo loại thuộc tính (kept / partial / mistranslated / lost):")
    for t, c in sorted(by_type.items()):
        n = sum(c[s] for s in ("kept", "partial", "mistranslated", "lost"))
        print(f"  {t:<12} n={n:<4} " + " / ".join(pct(c[s], n) for s in ("kept", "partial", "mistranslated", "lost")))
    print("\nTheo nhóm sản phẩm (tỉ lệ truy vấn có cờ):")
    for cat, (n, k) in sorted(by_cat.items()):
        print(f"  {cat:<10} {k}/{n}  ({pct(k, n)})")
    if unchecked_vals:
        print("\nGiá trị thuộc tính CHƯA có trong ATTR_RULES (bổ sung rồi chạy lại):")
        for k, n in unchecked_vals.most_common():
            print(f"  {k}  ×{n}")
    print("\nVí dụ bị cờ (đọc bằng mắt tất cả trước khi dùng số liệu):")
    for r in [r for r in rows if r["mt_loss_flag"]][:15]:
        print(f"  {r['query_id']}  VI: {r['query_vi']}\n        EN: {r['query_en']}\n        {r['problems'] or ''} head={r['head_status']}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLưu: {out}")


if __name__ == "__main__":
    main()
