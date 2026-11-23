"""
attr_utils.py — Thư viện dùng chung cho V1, V3, V4, V5
Cung cấp:
  normalize(text)             → chuẩn hóa chuỗi (NFC, chữ thường, strip)
  as_list(value)              → ép kiểu về list[str]
  map_type(raw_type, name)    → loại sản phẩm chuẩn hóa tiếng Việt
  map_color(raw_color)        → màu chuẩn hóa, None nếu không hợp lệ
  detect_kieu_dang(name, desc) → list kiểu dáng tìm được
  detect_phong_cach(name, desc) → list phong cách tìm được
  KIEU_DANG_CATEGORIES        → set danh mục được phép trích kiểu dáng
"""

import re
import unicodedata

# ═══════════════════════════════════════════════════════════════════
# 1. NORMALIZE
# ═══════════════════════════════════════════════════════════════════
def normalize(text: str | None) -> str:
    """Unicode NFC, chữ thường, bỏ khoảng trắng thừa. None → ''."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ═══════════════════════════════════════════════════════════════════
# 2. AS_LIST
# ═══════════════════════════════════════════════════════════════════
def as_list(value) -> list[str]:
    """Ép value về list[str], lọc bỏ phần tử rỗng."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    return [str(value).strip()]


# ═══════════════════════════════════════════════════════════════════
# 3. MAP_TYPE — chuẩn hóa loại sản phẩm về tiếng Việt chữ thường
# ═══════════════════════════════════════════════════════════════════
_TYPE_MAP: dict[str, str] = {
    # áo sơ mi
    "ao-so-mi": "áo sơ mi", "ao-so-mi-nam": "áo sơ mi",
    "áo sơ mi": "áo sơ mi",
    # áo thun / polo
    "ao-thun": "áo thun", "ao-thun-nam": "áo thun",
    "áo phông & áo ba lỗ": "áo thun", "áo thun": "áo thun",
    "ao-polo": "áo polo", "áo polo": "áo polo",
    "áo thun có cổ (polo)": "áo polo",
    # áo hoodie / sweater / nỉ / len
    "ao-hoodie-nam": "áo hoodie", "áo hoodies & sweatshirts": "áo hoodie",
    "ao-sweater": "áo sweater", "ao-len": "áo len", "ao-ni": "áo nỉ",
    "ao-ni-ao-len-nam": "áo nỉ",
    # áo tanktop / giữ nhiệt
    "ao-tanktop": "áo tanktop", "do-giu-nhiet": "áo giữ nhiệt",
    # áo khoác / vest
    "ao-khoac": "áo khoác", "ao-vest-ao-khoac-nam": "áo khoác",
    "áo khoác bomber": "áo khoác bomber", "áo khoác nhẹ": "áo khoác nhẹ",
    "ao-vest-blazer": "áo vest", "suits-blazer": "áo vest",
    "áo vest": "áo vest",
    # quần âu / dài
    "quan-au": "quần âu", "quan-tay": "quần âu", "quan-dai-nam": "quần dài",
    # quần jean
    "quan-jean": "quần jean", "quan-jeans": "quần jean",
    "quần jeans": "quần jean", "quần jean": "quần jean",
    # quần kaki / jogger
    "quan-kaki": "quần kaki", "quan-jogger": "quần jogger",
    # quần short / bơi / lót / boxer
    "quan-short": "quần short", "quan-short-nam": "quần short",
    "quần ngắn (shorts)": "quần short",
    "quần bơi": "quần bơi", "do-boi-do-di-bien-nam": "quần bơi",
    "quan-boxer": "quần boxer", "quan-brief": "quần brief",
    "quần đùi & quần lót": "quần lót", "quan-lot": "quần lót",
    # bộ / đồ mặc nhà / đồ lót
    "bo-trang-phuc-nam": "bộ trang phục", "set-quan-ao": "bộ trang phục",
    "bộ đồ": "bộ trang phục",
    "do-ngu-do-mac-nha-nam": "bộ mặc nhà",
    "do-lot": "đồ lót", "do-lot-nam": "đồ lót",
    # giày dép
    "giay": "giày", "giay-dep": "giày dép",
    "giày sneaker": "giày sneaker", "giày chạy bộ": "giày chạy bộ",
    "giày công sở": "giày công sở",
    "bốt cổ ngắn": "bốt", "xăng đan": "xăng đan",
    "dép xỏ ngón (flip flops)": "dép", "phụ kiện giày": "phụ kiện giày",
    # túi / ví / balo
    "balo": "balo", "ba lô nam": "balo", "balo-tui-xach": "balo",
    "tui": "túi", "túi đeo chéo & đeo vai": "túi đeo chéo",
    "cặp công sở": "cặp công sở",
    "vi": "ví", "vi-da": "ví da", "ví nam": "ví", "ví đựng thẻ": "ví đựng thẻ",
    "tui-vi-balo": "túi ví balo",
    # thắt lưng
    "that-lung": "thắt lưng", "that-lung-da": "thắt lưng da",
    "thắt lưng": "thắt lưng", "cap-da": "thắt lưng da", "day-nit": "thắt lưng",
    # mũ
    "mu": "mũ", "non": "mũ", "non-luoi-trai": "mũ lưỡi trai",
    # kính / cà vạt / khăn / phụ kiện khác
    "kinh-mat": "kính mắt", "mat-kinh": "kính mắt", "mắt kính": "kính mắt",
    "ca-vat": "cà vạt", "cà vạt": "cà vạt",
    "khan-quang-co": "khăn quàng cổ",
    "mang-set": "măng sét", "gang-tay": "găng tay",
    "tat": "tất", "vo": "vớ",
    "moc-khoa": "móc khóa",
    "dây chuyền": "dây chuyền", "nhẫn": "nhẫn",
    "vòng tay & vòng chân": "vòng tay", "bông tai": "bông tai",
    "đồng hồ thời trang": "đồng hồ",
    "quan-ao-nam-trung-nien": "quần áo",
    "đồ bảo hộ & suits": "bộ trang phục",
}

# Từ khóa trong name giúp đoán loại khi type không map được
_NAME_TYPE_HINTS: list[tuple[str, str]] = [
    ("áo sơ mi", "áo sơ mi"), ("sơ mi", "áo sơ mi"),
    ("áo polo", "áo polo"), ("polo", "áo polo"),
    ("áo thun", "áo thun"), ("áo hoodie", "áo hoodie"), ("hoodie", "áo hoodie"),
    ("áo khoác", "áo khoác"), ("áo vest", "áo vest"), ("blazer", "áo vest"),
    ("áo len", "áo len"), ("áo nỉ", "áo nỉ"), ("sweater", "áo sweater"),
    ("quần jean", "quần jean"), ("jeans", "quần jean"),
    ("quần kaki", "quần kaki"), ("khaki", "quần kaki"),
    ("quần jogger", "quần jogger"), ("jogger", "quần jogger"),
    ("quần short", "quần short"), ("short", "quần short"),
    ("quần âu", "quần âu"),
    ("balo", "balo"), ("ba lô", "balo"),
    ("ví", "ví"), ("thắt lưng", "thắt lưng"),
    ("mũ", "mũ"), ("nón", "mũ"),
    ("giày", "giày"), ("dép", "dép"),
]


def map_type(raw_type: str, name: str = "") -> str:
    """
    Trả về loại sản phẩm chuẩn hóa tiếng Việt chữ thường.
    Thử theo thứ tự: bảng _TYPE_MAP → normalize(raw_type) → gợi ý từ name.
    """
    key = normalize(raw_type)
    if key in _TYPE_MAP:
        return _TYPE_MAP[key]
    # Thử tìm key gốc (không normalize) phòng trường hợp có dấu
    if raw_type in _TYPE_MAP:
        return _TYPE_MAP[raw_type]
    # Gợi ý từ name
    name_norm = normalize(name)
    for hint, loai in _NAME_TYPE_HINTS:
        if hint in name_norm:
            return loai
    # Fallback: trả về normalize(raw_type)
    return key


# ═══════════════════════════════════════════════════════════════════
# 4. MAP_COLOR — chuẩn hóa màu, trả về None nếu không hợp lệ
# ═══════════════════════════════════════════════════════════════════
_COLOR_NORM_MAP: dict[str, str] = {
    # đen
    "black": "đen", "màu đen": "đen", "den": "đen",
    "full đen": "đen", "đen trơn": "đen", "đen tuyền": "đen",
    "deep black": "đen",
    # trắng / kem / be
    "white": "trắng", "màu trắng": "trắng", "trắng sáng": "trắng",
    "trắng kem": "kem", "antique white": "kem", "cream": "kem", "kem": "kem",
    "beige": "be", "beige st": "be", "begie": "be", "be": "be",
    # xám
    "grey": "xám", "gray": "xám", "màu xám": "xám",
    "ghi": "xám", "light grey": "xám nhạt", "light gr": "xám nhạt",
    "dark grey": "xám đậm", "dark gre": "xám đậm",
    "l.grey": "xám nhạt", "steel grey": "xám",
    "xám ghi": "xám", "ghi sáng": "xám nhạt", "ghi đậm": "xám đậm",
    "xám nhạt": "xám nhạt", "xám đậm": "xám đậm",
    "xám khói": "xám", "khói": "xám",
    # xanh navy / tím than
    "navy": "xanh navy", "navy blue": "xanh navy",
    "dark navy": "xanh navy", "dark nav": "xanh navy",
    "xanh navy": "xanh navy", "xanh tím than": "xanh tím than",
    "xanh chàm đậm": "xanh tím than", "midnight navy": "xanh navy",
    "indigo": "xanh indigo", "xanh indigo": "xanh indigo",
    "xanh đen": "xanh đen", "xanh đậm": "xanh đậm",
    # xanh dương
    "blue": "xanh dương", "màu xanh da trời": "xanh dương",
    "xanh da trời": "xanh dương", "xanh dương": "xanh dương",
    "xanh biển": "xanh dương",
    "light blue": "xanh nhạt", "light bl": "xanh nhạt",
    "lght blu": "xanh nhạt", "sky blue": "xanh nhạt",
    "xanh nhạt": "xanh nhạt", "xanh sáng": "xanh nhạt",
    "dark blue": "xanh đậm", "dark blu": "xanh đậm",
    "xanh": "xanh dương", "màu xanh": "xanh dương",
    # xanh lá
    "green": "xanh lá", "xanh lá": "xanh lá",
    "xanh rêu": "xanh rêu", "rêu": "xanh rêu",
    "xanh cổ vịt": "xanh cổ vịt",
    "xanh aqua": "xanh ngọc", "xanh bích": "xanh ngọc", "xanh ngọc": "xanh ngọc",
    "army green": "xanh lá", "dark green": "xanh lá đậm",
    "mint": "xanh mint", "xanh coban": "xanh dương",
    # nâu
    "brown": "nâu", "màu nâu": "nâu", "dark brown": "nâu đậm",
    "nâu đậm": "nâu đậm", "coffee": "nâu", "cinnamon": "nâu",
    "chestnut": "nâu", "mahogany": "nâu đỏ", "nâu đỏ": "nâu đỏ",
    "bò": "nâu", "da bò": "nâu",
    # đỏ / cam / hồng / tím
    "red": "đỏ", "màu đỏ": "đỏ",
    "đỏ đô": "đỏ đô", "boóc-đô": "đỏ đô", "đỏ đậm": "đỏ đô",
    "orange": "cam", "màu cam": "cam",
    "yellow": "vàng", "màu vàng": "vàng",
    "pink": "hồng", "hồng": "hồng", "hồng cánh sen": "hồng",
    "purple": "tím", "tím": "tím", "tím than": "tím than",
    # bạc / vàng kim
    "silver": "bạc", "màu bạc": "bạc", "bạc": "bạc",
    "gold": "vàng kim", "vàng kim": "vàng kim",
    # kaki
    "khaki": "kaki", "light khaki": "kaki nhạt", "màu kaki": "kaki",
}

# Pattern lọc bỏ — nếu khớp bất kỳ pattern nào → không phải màu hợp lệ
_COLOR_REJECT = re.compile(
    r"\bkẻ\b|\bsọc\b|\bjacquard\b|\bdobby\b"
    r"|\bhọa tiết\b|\bhoạ tiết\b|\bdệt\b|\bthêu\b|\bchấm\b|\bcaro\b"
    r"|\bin\b"                       # "trắng in", "xanh in họa tiết"
    r"|\bsolid\b"                    # "trắng solid"
    r"|\bjean\b"                     # "jean" = chất liệu, không phải màu
    r"|\bck\b|\bst\b|\bmf\b"         # mã thương hiệu / nội bộ
    r"|\bcan pha\b|\bwash\b|\bmelange\b|\bslub\b|\bseersucker\b"
    r"|nhiều màu|multicolor|ngẫu nhiên|như hình|như được hiển thị"
    r"|as the picture|không phân biệt|mặc định|default|many|none|not specified"
    r"|mã |mã số|size|cỡ|\bkg\b|\bml\b|lít|\bcái\b|\bpair\b|combo|hộp|\bbộ\b|\bset\b"
    r"|nữ\b|nam\b|áo\b|quần\b|giày\b|ví\b"
    r"|\s\d+$"                       # màu + mã số cuối: "đen 09", "xám 38"
    r"|^\d",                         # bắt đầu bằng số
    re.IGNORECASE,
)


def map_color(raw_color: str | None) -> str | None:
    """
    Chuẩn hóa một giá trị màu.
    Trả về chuỗi màu chuẩn hóa, hoặc None nếu không hợp lệ.
    """
    if not raw_color:
        return None
    c = normalize(raw_color)

    # 1. Tra bảng chuẩn hóa
    if c in _COLOR_NORM_MAP:
        return _COLOR_NORM_MAP[c]

    # 2. Lọc bỏ
    if _COLOR_REJECT.search(c):
        return None

    # 3. Quá dài (> 30 ký tự) → nhiều khả năng là mô tả
    if len(c) > 30:
        return None

    # 4. Giữ lại màu tiếng Việt đơn giản chưa có trong bảng
    return c


# ═══════════════════════════════════════════════════════════════════
# 5. DETECT_KIEU_DANG
# ═══════════════════════════════════════════════════════════════════
# Danh mục được phép trích kiểu dáng (chỉ Áo và Quần, không phụ kiện)
KIEU_DANG_CATEGORIES: set[str] = {
    "áo", "quần",
    # thêm các giá trị category thực tế trong dữ liệu (normalize)
    "ao", "quan",
}

# Từ điển kiểu dáng: (pattern tìm kiếm, giá trị chuẩn)
# Dùng re để khớp cả biến thể không dấu / viết liền
_KIEU_DANG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bslim[\s\-]?fit\b", re.I),       "slim fit"),
    (re.compile(r"\bregular[\s\-]?fit\b", re.I),    "regular fit"),
    (re.compile(r"\boversize[d]?\b", re.I),          "oversize"),
    (re.compile(r"\bstraight[\s\-]?fit\b", re.I),   "straight fit"),
    (re.compile(r"\bskinny\b", re.I),                "skinny"),
    (re.compile(r"\brelaxed[\s\-]?fit\b", re.I),    "relaxed fit"),
    (re.compile(r"\bloose[\s\-]?fit\b", re.I),      "loose fit"),
    (re.compile(r"\bcomfort[\s\-]?fit\b", re.I),    "comfort fit"),
    (re.compile(r"\btapered\b", re.I),               "tapered"),
    (re.compile(r"\bbaggy\b", re.I),                 "baggy"),
    (re.compile(r"\bbootcut\b", re.I),               "bootcut"),
    (re.compile(r"\bwide[\s\-]?leg\b", re.I),        "wide leg"),
    # tiếng Việt
    (re.compile(r"form\s+ôm|dáng\s+ôm|ôm\s+sát|ôm\s+vừa|ôm\s+người"),  "slim fit"),
    (re.compile(r"form\s+slim|dáng\s+slim"),                              "slim fit"),
    (re.compile(r"form\s+suông|suông\s+vừa|dáng\s+suông|form\s+regular|form\s+vừa"), "regular fit"),
    (re.compile(r"form\s+rộng|dáng\s+rộng|form\s+oversize"),             "oversize"),
    (re.compile(r"form\s+thẳng|dáng\s+thẳng"),                           "straight fit"),
    (re.compile(r"form\s+thoải\s+mái"),                                   "relaxed fit"),
    (re.compile(r"dáng\s+skinny|ôm\s+chặt"),                             "skinny"),
    (re.compile(r"dáng\s+jogger"),                                        "relaxed fit"),
    # từ đơn thường gặp trong name sản phẩm
    (re.compile(r"\bslim\b", re.I),                 "slim fit"),
    (re.compile(r"\bregular\b", re.I),              "regular fit"),
    (re.compile(r"\boversize\b", re.I),             "oversize"),
]


def detect_kieu_dang(name: str, desc: str) -> list[str]:
    """
    Dò kiểu dáng trong name và desc (đã normalize).
    Trả về list giá trị chuẩn, không trùng lặp, theo thứ tự xuất hiện.
    """
    text = name + " " + desc
    found = []
    for pattern, value in _KIEU_DANG_PATTERNS:
        if pattern.search(text) and value not in found:
            found.append(value)
    return found


# ═══════════════════════════════════════════════════════════════════
# 6. DETECT_PHONG_CACH
# ═══════════════════════════════════════════════════════════════════
_PHONG_CACH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcasual\b", re.I),                           "casual"),
    (re.compile(r"đời thường|hàng ngày|năng động|\bbasic\b", re.I), "casual"),
    (re.compile(r"công\s+sở|văn\s+phòng|\bformal\b|\bbusiness\b", re.I), "công sở"),
    (re.compile(r"business\s+casual", re.I),                    "công sở"),
    (re.compile(r"thể\s+thao|\bsporty\b|\bsport\b", re.I),     "thể thao"),
    (re.compile(r"streetwear|đường\s+phố|\bstreet\b", re.I),   "streetwear"),
    (re.compile(r"\bvintage\b", re.I),                          "vintage"),
    (re.compile(r"\bminimalist\b|tối\s+giản", re.I),            "minimalist"),
    (re.compile(r"\bpreppy\b", re.I),                           "preppy"),
    (re.compile(r"\bworkwear\b", re.I),                         "công sở"),
    (re.compile(r"dạo\s+phố|đi\s+chơi", re.I),                 "casual"),
]

# Giá trị được gộp về chuẩn (tránh trùng "casual" và "casual")
_PHONG_CACH_NORM: dict[str, str] = {
    "sporty":    "thể thao",
    "sport":     "thể thao",
    "business":  "công sở",
    "formal":    "công sở",
    "workwear":  "công sở",
    "đường phố": "streetwear",
    "street":    "streetwear",
}


def detect_phong_cach(name: str, desc: str) -> list[str]:
    """
    Dò phong cách trong name và desc (đã normalize).
    Trả về list giá trị chuẩn, không trùng lặp, theo thứ tự xuất hiện.
    """
    text = name + " " + desc
    found = []
    for pattern, value in _PHONG_CACH_PATTERNS:
        if pattern.search(text):
            # chuẩn hóa nếu có trong _PHONG_CACH_NORM
            value = _PHONG_CACH_NORM.get(value, value)
            if value not in found:
                found.append(value)
    return found


# ═══════════════════════════════════════════════════════════════════
# 7. KIỂM TRA NHANH (chạy trực tiếp: python attr_utils.py)
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        # normalize
        ("normalize", normalize("  Áo Sơ Mi  "), "áo sơ mi"),
        ("normalize", normalize(None), ""),
        # as_list
        ("as_list", as_list(None), []),
        ("as_list", as_list("cotton"), ["cotton"]),
        ("as_list", as_list(["cotton", "polyester"]), ["cotton", "polyester"]),
        # map_type
        ("map_type", map_type("ao-so-mi"), "áo sơ mi"),
        ("map_type", map_type("Áo sơ mi"), "áo sơ mi"),
        ("map_type", map_type("quan-jean"), "quần jean"),
        ("map_type", map_type("unknown_type", "áo polo form slim"), "áo polo"),
        # map_color
        ("map_color", map_color("black"), "đen"),
        ("map_color", map_color("Navy"), "xanh navy"),
        ("map_color", map_color("multicolor"), None),
        ("map_color", map_color("trắng in họa tiết"), None),
        ("map_color", map_color("đen 09"), None),   # màu + mã số
        ("map_color", map_color("xanh kẻ vàng"), None),
        ("map_color", map_color("đỏ"), "đỏ"),
        # detect_kieu_dang
        ("detect_kieu_dang", detect_kieu_dang("áo sơ mi slim fit", "form ôm vừa"), ["slim fit"]),
        ("detect_kieu_dang", detect_kieu_dang("áo polo", "oversize thoải mái"), ["oversize"]),
        # detect_phong_cach
        ("detect_phong_cach", detect_phong_cach("áo thun casual", "phong cách đường phố"), ["casual", "streetwear"]),
        ("detect_phong_cach", detect_phong_cach("áo sơ mi công sở business", ""), ["công sở"]),
    ]

    passed = failed = 0
    for fn, got, expected in tests:
        ok = got == expected
        status = "✅" if ok else "❌"
        if not ok:
            print(f"{status} {fn}: expected {expected!r}, got {got!r}")
            failed += 1
        else:
            passed += 1
    print(f"\n{passed} passed, {failed} failed")