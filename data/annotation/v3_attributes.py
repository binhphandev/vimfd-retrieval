"""
V3 Thuộc tính và từ điển
Đầu vào:
  - data/splits/train.jsonl, data/splits/val.jsonl
Đầu ra:
  - data/annotation/cross_encoder/attr_dictionary.xlsx
        sheet attr_dictionary : value_raw, value_norm, attr_type, synonyms   (theo kế hoạch 5.1)
        sheet synonyms        : value_norm, attr_type, synonym               (V4 đọc sheet này, GIỮ 3 cột)
        sheet type_map        : raw_type, loai_norm, n_train                 (bảng type_map)
        sheet color_map       : raw_color, color_norm, n_train               (để rà soát)
        sheet color_unmapped  : raw_color, n_train                           (màu chưa map -> bổ sung COLOR_MAP nếu cần)
  - data/annotation/cross_encoder/product_attrs.json  (id -> {LOAI, MAU, CHAT_LIEU, KIEU_DANG, PHONG_CACH})
  - results/logs/log_A.md (nối thêm thống kê V3)
Chạy: python v3_attributes.py

Khác bản cũ (cần ghi vào mục 12 / log_A.md):
  1. MAU: chuẩn hóa về ~20 màu tiếng Việt chuẩn (black/đen/màu đen -> "đen"), bỏ giá trị rác
     (số, "style 1", "như hình", "multicolor"...). Chỉ gán MAU khi MỌI giá trị trong colors cùng quy về
     một màu chuẩn (đúng quy tắc "đúng 1 màu").
  2. KIEU_DANG / PHONG_CACH: dò bằng regex có ranh giới từ (không dò chuỗi con), gộp đồng nghĩa về giá trị
     chuẩn (oversized -> oversize), che cụm đã khớp để "casual" không khớp lại trong "smart casual".
     KIEU_DANG chỉ gán cho Áo/Quần và chỉ khi xác định được đúng 1 giá trị (mơ hồ thì bỏ).
     PHONG_CACH tiếng Việt dễ gây nhiễu (thể thao, công sở, đường phố) chỉ nhận từ description khi có ngữ cảnh
     "phong cách ...", còn trong name thì nhận trực tiếp.
  3. Bỏ thuộc tính trùng với loại sản phẩm (vd LOAI "giày công sở" thì không gán PHONG_CACH "công sở").
  4. LOAI: với vài type nguồn lẫn lộn (Lazada "Quần", "Đồ bảo hộ & Suits"...) tinh chỉnh theo name
     (bật/tắt bằng USE_NAME_REFINEMENT).
  5. Bảng đồng nghĩa giới hạn đúng <= 30 cặp (value_norm, synonym), chờ B duyệt.
  6. CHAT_LIEU: dữ liệu KHÔNG có trường `material` (đã xác nhận trên train.jsonl), nên chất liệu được dò bằng regex
     từ name (nhận trực tiếp) và description (chỉ nhận khi có ngữ cảnh "chất liệu/chất vải/làm từ/vải ..."), giá trị
     chuẩn tiếng Việt (da, jean, len, nỉ, tre, lanh, cotton, polyester, dạ, nylon, lụa). Chỉ gán khi xác định được
     ĐÚNG 1 giá trị. Bản cũ đọc trường material nên CHAT_LIEU luôn rỗng và mẫu câu "chat" của V4 không bao giờ chạy.
  7. SKIP_ATTR_IN_LOAI so khớp theo ranh giới từ (bản cũ dò chuỗi con, dễ loại nhầm giá trị ngắn như "da", "be").
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# ── Đường dẫn ─────────────────────────────────────────────────────
ROOT       = Path(".")
TRAIN_PATH = ROOT / "data/splits/train.jsonl"
VAL_PATH   = ROOT / "data/splits/val.jsonl"
OUT_DIR    = ROOT / "data/annotation/cross_encoder"
LOG_DIR    = ROOT / "results/logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

USE_NAME_REFINEMENT = True   # tinh chỉnh LOAI theo name cho các type nguồn lẫn lộn (xem REFINE_RULES)
SKIP_ATTR_IN_LOAI   = True   # bỏ thuộc tính nếu đã nằm trong tên loại (vd "quần kaki" + màu "kaki")


# ── Chuẩn hóa văn bản (mục 3.2) ───────────────────────────────────
def normalize(text) -> str:
    """Unicode NFC, chữ thường, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


# ══════════════════════════════════════════════════════════════════
# 1. LOAI  (type_map)
# ══════════════════════════════════════════════════════════════════
TYPE_MAP = {
    # --- áo ---
    "ao-so-mi": "áo sơ mi", "ao-so-mi-nam": "áo sơ mi", "Áo sơ mi": "áo sơ mi",
    "ao-thun": "áo thun", "ao-thun-nam": "áo thun", "Áo phông & Áo ba lỗ": "áo thun",
    "ao-polo": "áo polo", "Áo thun có cổ (Polo)": "áo polo",
    "ao-hoodie-nam": "áo hoodie", "Áo hoodies & Sweatshirts": "áo hoodie",
    "ao-sweater": "áo sweater", "ao-len": "áo len", "ao-ni": "áo nỉ",
    "ao-ni-ao-len-nam": "áo nỉ", "ao-tanktop": "áo tanktop",
    "Áo polo": "áo polo", "Áo hoodie & Sweatshirt": "áo hoodie",
    "ao-khoac": "áo khoác", "ao-vest-ao-khoac-nam": "áo khoác",
    "Áo khoác bomber": "áo khoác bomber", "Áo khoác nhẹ": "áo khoác nhẹ",
    "ao-vest-blazer": "áo vest", "suits-blazer": "áo vest",
    "do-giu-nhiet": "áo giữ nhiệt",
    # --- quần ---
    "Quần": "quần", "quan-au": "quần âu", "quan-tay": "quần âu",
    "quan-dai-nam": "quần dài", "Quần dài": "quần dài", "Quần shorts": "quần short",
    "quan-jean": "quần jean", "quan-jeans": "quần jean", "Quần jeans": "quần jean",
    "quan-kaki": "quần kaki", "quan-jogger": "quần jogger",
    "quan-short": "quần short", "quan-short-nam": "quần short", "Quần ngắn (Shorts)": "quần short",
    "Quần bơi": "quần bơi", "do-boi-do-di-bien-nam": "quần bơi",
    "quan-boxer": "quần boxer", "quan-brief": "quần brief",
    "Quần đùi & Quần lót": "quần lót", "quan-lot": "quần lót",
    "quan-ao-nam-trung-nien": "quần",
    # --- bộ / đồ lót ---
    "bo-trang-phuc-nam": "bộ trang phục", "set-quan-ao": "bộ trang phục",
    "Bộ đồ": "bộ trang phục", "Đồ bảo hộ & Suits": "bộ trang phục",
    "do-ngu-do-mac-nha-nam": "bộ mặc nhà",
    "do-lot": "đồ lót", "do-lot-nam": "đồ lót", "ao-lot": "áo lót",
    # --- giày dép ---
    "giay": "giày", "giay-dep": "giày dép", "Dép xỏ ngón": "dép", 
    "Giày sneaker": "giày sneaker", "Giày chạy bộ": "giày chạy bộ", "Giày công sở": "giày công sở",
    "Bốt cổ ngắn": "bốt", "Xăng đan": "xăng đan", "Dép xỏ ngón (Flip Flops)": "dép",
    "Phụ kiện giày": "phụ kiện giày",
    # --- túi, ví, balo ---
    "balo": "balo", "Ba lô nam": "balo", "balo-tui-xach": "balo",
    "tui": "túi", "Túi đeo chéo & đeo vai": "túi đeo chéo", "Cặp công sở": "cặp công sở",
    "vi": "ví", "vi-da": "ví da", "Ví nam": "ví", "Ví đựng thẻ": "ví đựng thẻ",
    "tui-vi-balo": "túi ví balo",
    # --- thắt lưng ---
    "that-lung": "thắt lưng", "that-lung-da": "thắt lưng da", "Thắt lưng": "thắt lưng",
    "cap-da": "thắt lưng da", "day-nit": "thắt lưng",
    # --- phụ kiện khác ---
    "mu": "mũ", "non": "mũ", "non-luoi-trai": "mũ lưỡi trai",
    "kinh-mat": "kính mắt", "mat-kinh": "kính mắt", "Mắt kính": "kính mắt",
    "ca-vat": "cà vạt", "Cà vạt": "cà vạt",
    "khan-quang-co": "khăn quàng cổ", "mang-set": "măng sét", "gang-tay": "găng tay",
    "tat": "tất", "vo": "vớ", "moc-khoa": "móc khóa",
    "Dây chuyền": "dây chuyền", "Nhẫn": "nhẫn", "Vòng tay & vòng chân": "vòng tay",
    "Bông tai": "bông tai", "Đồng hồ thời trang": "đồng hồ",
}

# Type nguồn lẫn lộn nhiều loại -> tinh chỉnh theo name. Quy tắc duyệt từ trên xuống, khớp đầu tiên thắng.
# Không khớp quy tắc nào thì giữ giá trị trong TYPE_MAP.
REFINE_RULES = {
    "Quần": [
        (r"jean|denim|\bbò\b", "quần jean"),
        (r"kaki|khaki", "quần kaki"),
        (r"\btây\b|\bâu\b|công sở", "quần âu"),
        (r"short|đùi|sooc", "quần short"),
        (r"jogger", "quần jogger"),
        (r"thể thao|chạy bộ|gym", "quần thể thao"),
    ],
    "quan-dai-nam": [
        (r"jean|denim", "quần jean"),
        (r"kaki|khaki", "quần kaki"),
        (r"\btây\b|\bâu\b", "quần âu"),
        (r"jogger", "quần jogger"),
        (r"thể thao", "quần thể thao"),
    ],
    "Đồ bảo hộ & Suits": [
        (r"vest|blazer|gile|gilê|suit|comple", "áo vest"),
        (r"quần", "quần âu"),
    ],
    "ao-vest-ao-khoac-nam": [
        (r"vest|blazer|gile|gilê", "áo vest"),
    ],
}
_REFINE_COMPILED = {k: [(re.compile(p), v) for p, v in rules] for k, rules in REFINE_RULES.items()}

unmapped_types = Counter()


def map_type(raw_type: str, name: str = "") -> str:
    """Loại sản phẩm đã chuẩn hóa (tiếng Việt, chữ thường)."""
    if raw_type in TYPE_MAP:
        loai = TYPE_MAP[raw_type]
    else:
        unmapped_types[raw_type] += 1
        loai = normalize(raw_type).replace("-", " ")
    if USE_NAME_REFINEMENT and raw_type in _REFINE_COMPILED:
        nm = normalize(name)
        for pat, new_loai in _REFINE_COMPILED[raw_type]:
            if pat.search(nm):
                return new_loai
    return loai


# ══════════════════════════════════════════════════════════════════
# 2. MAU  (chuẩn hóa màu)
# ══════════════════════════════════════════════════════════════════
# Giá trị chuẩn -> danh sách cách viết thô (đã normalize, bỏ "màu " ở đầu và dấu chấm cuối).
# Chủ ý gộp sắc độ (dark/light, đậm/nhạt) về màu gốc: truy vấn "màu xám" đúng với cả xám đậm lẫn xám nhạt.
# Giá trị mơ hồ ("xanh", "xanh đậm", "xanh kẻ"...) và màu hiếm không có trong bảng -> không gán MAU.
_COLOR_GROUPS = {
    "đen":       ["black", "đen", "den", "đen tuyền", "jet black"],
    "trắng":     ["white", "trắng", "trang", "off white", "trắng sữa", "ivory"],
    "xám":       ["grey", "gray", "xám", "ghi", "dark grey", "dark gray", "light grey", "light gray",
                  "xám đậm", "xám nhạt", "xám khói", "ghi sáng", "ghi đậm", "ghi xám", "charcoal", "xám tro",
                  "xám ghi"],
    "xanh navy": ["navy", "navy blue", "dark blue", "dark navy", "deep blue", "xanh navy", "xanh than",
                  "xanh tím than", "tím than", "xanh hải quân", "xanh đen", "xanh đen đậm"],
    "xanh dương": ["blue", "light blue", "sky blue", "lake blue", "royal blue", "xanh dương", "xanh biển",
                   "xanh da trời", "xanh lam", "xanh nhạt", "xanh dương đậm", "xanh dương nhạt", "xanh hoàng gia",
                   "xanh coban", "xanh bích"],
    "xanh lá":   ["green", "dark green", "light green", "xanh lá", "xanh lá cây", "xanh cốm"],
    "xanh rêu":  ["army green", "olive", "olive green", "rêu", "xanh rêu", "xanh quân đội"],
    "nâu":       ["brown", "dark brown", "light brown", "coffee", "nâu", "nau", "nâu đậm", "nâu nhạt", "nâu sẫm",
                  "cà phê", "cafe", "nâu cafe", "chocolate"],
    "be":        ["beige", "be", "nude"],
    "kem":       ["cream", "kem", "champagne"],
    "kaki":      ["khaki", "kaki"],
    "đỏ":        ["red", "đỏ", "do"],
    "đỏ đô":     ["đỏ đô", "burgundy", "wine red", "dark red", "rượu vang đỏ", "maroon"],
    "vàng":      ["yellow", "vàng", "mustard", "vàng nghệ"],
    "cam":       ["orange", "cam"],
    "hồng":      ["pink", "hồng", "hong", "light pink", "hồng phấn"],
    "tím":       ["purple", "tím", "violet"],
    "bạc":       ["silver", "bạc"],
}
COLOR_MAP = {raw: norm for norm, raws in _COLOR_GROUPS.items() for raw in raws}

# Một số cụm chắc chắn không phải màu đơn: bỏ nhanh để thống kê rõ hơn (không bắt buộc, giá trị không nằm
# trong COLOR_MAP thì vốn đã không có MAU).
_JUNK_COLOR = re.compile(
    r"^\d+$|^[a-z]\d*$|^\d|style|mẫu|^màu \d|as the picture|như hình|hình ảnh|ngẫu nhiên|multicolor|nhiều màu|"
    r"transparent|random|\bxl\b|\bsize\b"
)


def clean_color(raw: str) -> str:
    c = normalize(raw)
    c = re.sub(r"^màu\s+", "", c)          # "màu đen" -> "đen"
    c = re.sub(r"\s*color$", "", c)         # "apricot color" -> "apricot"
    c = re.sub(r"[\.\,;:!]+$", "", c).strip()
    # bóc mã số / đuôi nhiễu nhưng giữ màu gốc: "đen-0032", "đen 09", "đen1", "xanh tím than 24"
    c = re.sub(r"^\d+\s*[-.]\s*", "", c)   # mã đứng đầu: "01-trắng" -> "trắng"
    c = re.sub(r"-\d+$", "", c)
    c = re.sub(r"(?<=\D)\s*\d+$", "", c)
    c = re.sub(r"\s+(?:solid|wash)$", "", c)  # "trắng solid" (trơn), "đen wash" (denim wash)
    return c.strip()


def map_color(raw: str):
    """Trả về màu chuẩn hoặc None nếu không map được."""
    c = clean_color(raw)
    if not c or _JUNK_COLOR.search(c):
        return None
    return COLOR_MAP.get(c)


# ══════════════════════════════════════════════════════════════════
# 3. KIEU_DANG và PHONG_CACH  (regex có ranh giới từ, gộp đồng nghĩa)
# ══════════════════════════════════════════════════════════════════
def _b(expr: str) -> str:
    """Bọc ranh giới từ theo Unicode (không khớp chuỗi con giữa từ)."""
    return rf"(?<!\w)(?:{expr})(?!\w)"


# (giá_trị_chuẩn, regex). Thứ tự = độ ưu tiên; cụm khớp được che để mẫu sau không khớp lại.
KIEU_DANG_PATTERNS = [
    ("slim fit",     _b(r"slim\s?fit|(?:form|phom|dáng)\s+slim|skinny\s?fit")),
    ("regular fit",  _b(r"regular\s?fit|(?:form|phom|dáng)\s+regular|classic\s?fit|standard\s?fit")),
    ("relaxed fit",  _b(r"relaxed?\s?fit|(?:form|phom|dáng)\s+relax(?:ed)?")),
    ("comfort fit",  _b(r"comfort\s?fit")),
    ("loose fit",    _b(r"loose\s?fit|(?:form|phom)\s+loose")),
    ("oversize",     _b(r"oversize[d]?|(?:form|phom|dáng)\s+rộng")),
    ("straight fit", _b(r"straight\s?fit|(?:form|phom)\s+straight|ống\s+suông|ống\s+đứng|dáng\s+thẳng")),
    ("skinny",       _b(r"skinny")),
    ("tapered",      _b(r"tapered(?:\s?fit)?|ống\s+côn")),
    ("baggy",        _b(r"baggy")),
    ("wide leg",     _b(r"wide[\s\-]?leg|ống\s+rộng")),
]
KIEU_DANG_VALUES = [v for v, _ in KIEU_DANG_PATTERNS]
KIEU_DANG_CATEGORIES = {"áo", "quần"}   # phụ kiện gần như không có kiểu dáng, bỏ để khỏi nhiễu

# (giá_trị_chuẩn, regex, can_ngu_canh_trong_description)
# can_ngu_canh=True: từ tiếng Việt thông dụng, trong description chỉ nhận khi đứng sau "phong cách"/"style"...
PHONG_CACH_PATTERNS = [
    ("smart casual", _b(r"smart\s+casual|business\s+casual"), False),
    ("casual",       _b(r"casual"), False),
    ("công sở",      _b(r"công\s+sở|văn\s+phòng|office"), True),
    ("thể thao",     _b(r"thể\s+thao|sporty|sportswear|athleisure"), True),
    ("streetwear",   _b(r"streetwear|street\s?style|đường\s+phố"), True),
    ("vintage",      _b(r"vintage|retro|cổ\s+điển"), True),
    ("minimalist",   _b(r"minimalist|minimal|tối\s+giản"), True),
    ("basic",        _b(r"basic"), False),
    ("preppy",       _b(r"preppy"), False),
    ("workwear",     _b(r"workwear"), False),
]
PHONG_CACH_VALUES = [v for v, _, _ in PHONG_CACH_PATTERNS]

# Cụm "loại + phong cách" thực chất là TÊN LOẠI (giày thể thao, giày/cặp công sở): che trước khi dò phong cách
_TYPE_PHRASE_MASK = re.compile(
    _b(r"(?:giày|dép|sandal|xăng\s?đan|bốt)\s+thể\s+thao|(?:giày|cặp|túi|balo|ba\s?lô|ví)\s+công\s+sở")
)
_STYLE_CTX = r"(?:phong\s+cách|style|dòng|hơi\s+hướng|tinh\s+thần|kiểu)\s+(?:\w+\s+){0,2}?"

_KD_COMPILED = [(v, re.compile(p)) for v, p in KIEU_DANG_PATTERNS]
_PC_COMPILED = [
    (v, re.compile(p), re.compile(_STYLE_CTX + "(?:" + p + ")") if ctx else re.compile(p))
    for v, p, ctx in PHONG_CACH_PATTERNS
]


def _find_all(text: str, compiled) -> list:
    """Dò lần lượt các mẫu (v, regex), che cụm đã khớp; trả về list giá trị chuẩn theo thứ tự ưu tiên."""
    found = []
    for v, rx in compiled:
        if rx.search(text):
            found.append(v)
            text = rx.sub(" ", text)
    return found


def detect_kieu_dang(name: str, desc: str) -> list:
    """Chỉ trả về khi xác định được ĐÚNG 1 kiểu dáng. Ưu tiên name; name không có thì dùng description."""
    in_name = _find_all(name, _KD_COMPILED)
    if len(in_name) == 1:
        return in_name
    if len(in_name) > 1:
        return []                       # name nêu nhiều kiểu dáng -> mơ hồ
    in_desc = _find_all(desc, _KD_COMPILED)
    return in_desc if len(in_desc) == 1 else []


def detect_phong_cach(name: str, desc: str) -> list:
    name = _TYPE_PHRASE_MASK.sub(" ", name)
    desc = _TYPE_PHRASE_MASK.sub(" ", desc)
    found = []
    # name: nhận trực tiếp
    for v, rx_plain, _ in _PC_COMPILED:
        if rx_plain.search(name):
            found.append(v)
            name = rx_plain.sub(" ", name)
    # description: từ thông dụng cần ngữ cảnh
    for v, rx_plain, rx_ctx in _PC_COMPILED:
        if v in found:
            continue
        if rx_ctx.search(desc):
            found.append(v)
            desc = rx_ctx.sub(" ", desc)
    return found


# --- CHAT_LIEU: dò từ name/description (dữ liệu không có trường material) ---
# (giá_trị_chuẩn, regex). Không đưa "kaki" vào đây để khỏi lẫn với MAU "kaki" và LOAI "quần kaki".
CHAT_LIEU_PATTERNS = [
    ("jean",      _b(r"jeans?|denim|vải\s+bò")),
    ("da",        _b(r"da(?:\s+(?:bò|thật|cừu|lộn|pu))?")),
    ("dạ",        _b(r"dạ")),
    ("len",       _b(r"len|wool|cashmere")),
    ("nỉ",        _b(r"nỉ|fleece")),
    ("lụa",       _b(r"lụa|silk")),
    ("tre",       _b(r"tre|sợi\s+tre|bamboo")),
    ("lanh",      _b(r"lanh|linen")),
    ("cotton",    _b(r"cotton|cô\s?tông")),
    ("polyester", _b(r"polyester|poly")),
    ("nylon",     _b(r"nylon|ni\s?lông")),
]
CHAT_LIEU_VALUES = [v for v, _ in CHAT_LIEU_PATTERNS]
# Trong description chỉ nhận khi đứng sau cụm nêu chất liệu (tránh "da" trong "phù hợp mọi loại da", "dạ" = vâng...)
_MAT_CTX = r"(?:chất\s+liệu|chất\s+vải|làm\s+từ|được\s+làm|vải|chất)\s*[:\-]?\s+(?:\w+\s+){0,1}?"
_CL_NAME = [(v, re.compile(p)) for v, p in CHAT_LIEU_PATTERNS]
_CL_DESC = [(v, re.compile(_MAT_CTX + "(?:" + p + ")")) for v, p in CHAT_LIEU_PATTERNS]


def detect_chat_lieu(name: str, desc: str) -> list:
    """Chỉ trả về khi xác định được ĐÚNG 1 chất liệu. Ưu tiên name; name không có thì dùng description (cần ngữ cảnh)."""
    in_name = _find_all(name, _CL_NAME)
    if len(in_name) == 1:
        return in_name
    if len(in_name) > 1:
        return []                       # name nêu nhiều chất liệu -> mơ hồ
    in_desc = _find_all(desc, _CL_DESC)
    return in_desc if len(in_desc) == 1 else []


# ══════════════════════════════════════════════════════════════════
# 4. BẢNG ĐỒNG NGHĨA (tối đa 30 cặp, cần B duyệt) — V4 đọc sheet "synonyms"
# ══════════════════════════════════════════════════════════════════
# (value_norm, attr_type, synonym). value_norm PHẢI trùng giá trị trong product_attrs.json.
SYNONYM_PAIRS = [
    # kiểu dáng
    ("slim fit",     "KIEU_DANG",  "dáng ôm"),
    ("regular fit",  "KIEU_DANG",  "dáng tiêu chuẩn"),
    ("oversize",     "KIEU_DANG",  "dáng rộng"),
    ("straight fit", "KIEU_DANG",  "dáng suông thẳng"),
    ("relaxed fit",  "KIEU_DANG",  "dáng rộng thoải mái"),
    ("skinny",       "KIEU_DANG",  "dáng bó"),
    # phong cách
    ("casual",       "PHONG_CACH", "thường ngày"),
    ("casual",       "PHONG_CACH", "đời thường"),
    ("công sở",      "PHONG_CACH", "văn phòng"),
    ("thể thao",     "PHONG_CACH", "sporty"),
    ("thể thao",     "PHONG_CACH", "năng động"),
    ("streetwear",   "PHONG_CACH", "đường phố"),
    ("vintage",      "PHONG_CACH", "retro"),
    ("minimalist",   "PHONG_CACH", "tối giản"),
    ("smart casual", "PHONG_CACH", "thanh lịch"),
    # chất liệu (value_norm = giá trị chuẩn tiếng Việt của detect_chat_lieu)
    ("da",           "CHAT_LIEU",  "chất da"),
    ("jean",         "CHAT_LIEU",  "vải bò"),
    ("jean",         "CHAT_LIEU",  "vải denim"),
    ("len",          "CHAT_LIEU",  "vải len"),
    ("tre",          "CHAT_LIEU",  "sợi tre"),
    ("lanh",         "CHAT_LIEU",  "vải lanh"),
    ("cotton",       "CHAT_LIEU",  "vải bông"),
    ("polyester",    "CHAT_LIEU",  "vải poly"),
    # màu
    ("xanh navy",    "MAU",        "xanh than"),
    ("xám",          "MAU",        "ghi"),
]
assert len(SYNONYM_PAIRS) <= 30, f"Bảng đồng nghĩa vượt 30 cặp: {len(SYNONYM_PAIRS)}"


# ══════════════════════════════════════════════════════════════════
# 5. LOAD DỮ LIỆU
# ══════════════════════════════════════════════════════════════════
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


print("Đang load dữ liệu...")
train_data = load_jsonl(TRAIN_PATH)
val_data   = load_jsonl(VAL_PATH)
print(f"  Train: {len(train_data)} | Val: {len(val_data)}")


# ══════════════════════════════════════════════════════════════════
# 6. THỐNG KÊ TRÊN TRAIN (type, material, màu)
# ══════════════════════════════════════════════════════════════════
def as_list(x):
    if not x:
        return []
    return [x] if isinstance(x, str) else list(x)


type_counter = Counter(d.get("type", "") for d in train_data)
mat_counter  = Counter(x for d in train_data
                       for x in detect_chat_lieu(normalize(d.get("name")), normalize(d.get("description"))))
raw_color_counter = Counter(clean_color(c) for d in train_data for c in as_list(d.get("colors")) if clean_color(c))

print(f"\nTop 20 chất liệu dò được (train): {len(mat_counter)} giá trị khác nhau")
for mat, cnt in mat_counter.most_common(20):
    print(f"  {mat}: {cnt}")


# ══════════════════════════════════════════════════════════════════
# 7. TRÍCH THUỘC TÍNH CHO MỖI SẢN PHẨM
# ══════════════════════════════════════════════════════════════════
def extract_mau(product: dict) -> list:
    """Có MAU khi mọi giá trị trong colors đều map được và cùng quy về 1 màu chuẩn."""
    raws = [c for c in as_list(product.get("colors")) if normalize(c)]
    if not raws:
        return []
    mapped = [map_color(c) for c in raws]
    if any(m is None for m in mapped):
        return []                       # có màu lạ/rác -> không chắc là đúng 1 màu
    distinct = set(mapped)
    return [distinct.pop()] if len(distinct) == 1 else []


def extract_attrs(product: dict) -> dict:
    """Trả về {LOAI, MAU, CHAT_LIEU, KIEU_DANG, PHONG_CACH}; mỗi key là list (rỗng nếu không có)."""
    name = normalize(product.get("name"))
    desc = normalize(product.get("description"))
    category = normalize(product.get("category"))

    loai = map_type(product.get("type", ""), product.get("name", ""))
    attrs = {"LOAI": [loai] if loai else [], "MAU": [], "CHAT_LIEU": [], "KIEU_DANG": [], "PHONG_CACH": []}

    attrs["MAU"] = extract_mau(product)

    attrs["CHAT_LIEU"] = detect_chat_lieu(name, desc)     # dữ liệu không có trường material

    if category in KIEU_DANG_CATEGORIES:
        attrs["KIEU_DANG"] = detect_kieu_dang(name, desc)
    attrs["PHONG_CACH"] = detect_phong_cach(name, desc)

    # Bỏ thuộc tính đã nằm trong tên loại (vd "quần kaki" + màu/chất liệu "kaki")
    if SKIP_ATTR_IN_LOAI and loai:
        loai_words = f" {loai} "
        for key in ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]:
            attrs[key] = [v for v in attrs[key] if f" {v} " not in loai_words]
    return attrs


print("\nĐang trích thuộc tính cho Train và Val...")
product_attrs = {}
for split_name, split_data in [("train", train_data), ("val", val_data)]:
    n_no_attr = 0
    for d in split_data:
        attrs = extract_attrs(d)
        product_attrs[d["id"]] = attrs
        if not any(attrs[k] for k in ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]):
            n_no_attr += 1
    print(f"  {split_name}: {len(split_data)} sản phẩm, {n_no_attr} không có thuộc tính ngoài loại (không dùng sinh truy vấn giả)")

if unmapped_types:
    print(f"\n[CẢNH BÁO] type chưa có trong TYPE_MAP: {dict(unmapped_types)}")

# Lưu ý: V3 không đọc test.jsonl nên không trích thuộc tính cho sản phẩm Test (đúng quy định mục 5.1).

out_attrs = OUT_DIR / "product_attrs.json"
with open(out_attrs, "w", encoding="utf-8") as f:
    json.dump(product_attrs, f, ensure_ascii=False, indent=2)
print(f"\nĐã lưu: {out_attrs}")


# ══════════════════════════════════════════════════════════════════
# 8. XUẤT attr_dictionary.xlsx
# ══════════════════════════════════════════════════════════════════
print("Đang tạo attr_dictionary.xlsx...")

# tổng hợp đồng nghĩa theo (value_norm, attr_type)
syn_by_value = {}
for norm, atype, syn in SYNONYM_PAIRS:
    syn_by_value.setdefault((norm, atype), []).append(syn)


def syn_str(norm, atype):
    return "; ".join(syn_by_value.get((norm, atype), []))


def style_header(ws, widths):
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


wb = openpyxl.Workbook()

# --- sheet chính: value_raw, value_norm, attr_type, synonyms ---
ws = wb.active
ws.title = "attr_dictionary"
ws.append(["value_raw", "value_norm", "attr_type", "synonyms"])
for mat in CHAT_LIEU_VALUES:
    ws.append([mat, mat, "CHAT_LIEU", syn_str(mat, "CHAT_LIEU")])
for v in KIEU_DANG_VALUES:
    ws.append([v, v, "KIEU_DANG", syn_str(v, "KIEU_DANG")])
for v in PHONG_CACH_VALUES:
    ws.append([v, v, "PHONG_CACH", syn_str(v, "PHONG_CACH")])
for norm, raws in _COLOR_GROUPS.items():
    ws.append([norm, norm, "MAU", syn_str(norm, "MAU")])
style_header(ws, [28, 28, 14, 40])

# --- sheet synonyms (V4 đọc: giữ đúng 3 cột value_norm, attr_type, synonym) ---
ws2 = wb.create_sheet("synonyms")
ws2.append(["value_norm", "attr_type", "synonym"])
for row in SYNONYM_PAIRS:
    ws2.append(list(row))
style_header(ws2, [20, 14, 24])

# --- sheet type_map ---
ws3 = wb.create_sheet("type_map")
ws3.append(["raw_type", "loai_norm", "n_train"])
for raw, cnt in type_counter.most_common():
    ws3.append([raw, TYPE_MAP.get(raw, normalize(raw).replace("-", " ")), cnt])
style_header(ws3, [32, 24, 10])

# --- sheet color_map (raw -> chuẩn) và color_unmapped ---
ws4 = wb.create_sheet("color_map")
ws4.append(["raw_color", "color_norm", "n_train"])
ws5 = wb.create_sheet("color_unmapped")
ws5.append(["raw_color", "n_train"])
n_map = n_unmap = 0
for raw, cnt in raw_color_counter.most_common():
    norm = None if _JUNK_COLOR.search(raw) else COLOR_MAP.get(raw)
    if norm and n_map < 150:
        ws4.append([raw, norm, cnt]); n_map += 1
    elif not norm and n_unmap < 150:
        ws5.append([raw, cnt]); n_unmap += 1
style_header(ws4, [30, 16, 10])
style_header(ws5, [40, 10])

out_xlsx = OUT_DIR / "attr_dictionary.xlsx"
wb.save(out_xlsx)
print(f"Đã lưu: {out_xlsx}")


# ══════════════════════════════════════════════════════════════════
# 9. THỐNG KÊ + KIỂM TRA
# ══════════════════════════════════════════════════════════════════
total = len(product_attrs)
lines = [f"\n## V3 — thuộc tính ({total} sản phẩm Train+Val)"]
for key in ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]:
    n = sum(1 for v in product_attrs.values() if v[key])
    lines.append(f"- Có {key}: {n} ({n/total:.1%})")
n_extra = sum(1 for v in product_attrs.values() if any(v[k] for k in ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]))
lines.append(f"- Dùng được để sinh truy vấn giả (có ≥1 thuộc tính ngoài loại): {n_extra} ({n_extra/total:.1%})")
lines.append(f"- Số giá trị MAU khác nhau: {len({x for v in product_attrs.values() for x in v['MAU']})}")
lines.append(f"- Số giá trị LOAI khác nhau: {len({x for v in product_attrs.values() for x in v['LOAI']})}")
lines.append(f"- Số cặp đồng nghĩa: {len(SYNONYM_PAIRS)}/30 (chờ B duyệt)")
lines.append(f"- USE_NAME_REFINEMENT={USE_NAME_REFINEMENT}, SKIP_ATTR_IN_LOAI={SKIP_ATTR_IN_LOAI}")

for key in ["MAU", "KIEU_DANG", "PHONG_CACH", "CHAT_LIEU"]:
    top = Counter(x for v in product_attrs.values() for x in v[key]).most_common(12)
    lines.append(f"- Top {key}: " + ", ".join(f"{k} ({c})" for k, c in top))

# Kiểm tra: mọi value_norm trong bảng đồng nghĩa phải xuất hiện trong product_attrs
present = {k: {x for v in product_attrs.values() for x in v[k]} for k in ["MAU", "CHAT_LIEU", "KIEU_DANG", "PHONG_CACH"]}
missing = [(n, t) for n, t, _ in SYNONYM_PAIRS if n not in present[t]]
if missing:
    lines.append(f"- [CẢNH BÁO] value_norm trong bảng đồng nghĩa không xuất hiện ở sản phẩm nào: {missing}")

report = "\n".join(lines)
print(report)
with open(LOG_DIR / "log_A.md", "a", encoding="utf-8") as f:
    f.write(report + "\n")

print("\nV3 hoàn thành! Nhờ B rà sheet 'synonyms' (10 phút) trước khi chạy V4.")
print("Xem sheet 'color_unmapped' để biết màu nào còn bị bỏ; muốn giữ thì thêm vào _COLOR_GROUPS rồi chạy lại.")