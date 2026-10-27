"""file này giúp làm sạch text trong description dùng cho train/index/value
loại bỏ các thông tin thừa như "chào_mừng mọi người", "thank you so much", "giá sản phẩm trên tiki" và các ký tự html
Mục tiêu: 
- làm sạch name, description, tags
- loại bỏ mã sản phẩm, HTML,URL, số điện thoại, emoji, hashtag, đoạn khuyến mãi rác...
- ghép các trường theo thứ tự ưu tiên: name -> type -> tags -> description
"""
import random
import re

#giới hạn số lượng từ tối đa cho description
MAX_DESC_WORDS = 60
#tên của các trường (ví dụ: lazada)
MAX_NAME_WORDS = 40
MAX_TAGS = 5

#dictionary chuyển loại sản phẩm (ví dụ "ao-so-mi") sang tiếng việt đã tách từ ("áo sơ_mi")
TYPE_VI = {
    "ao-so-mi": "áo sơ_mi", "giay-dep": "giày_dép", "ao-polo": "áo polo", "quan-jean": "quần jean",
    "ao-thun": "áo thun", "quan-short": "quần short", "vi": "ví", "ao-khoac": "áo_khoác",
    "quan-au": "quần âu", "tui": "túi", "ao-tanktop": "áo tanktop", "quan-brief": "quần brief",
    "quan-lot": "quần_lót", "cap-da": "cặp da", "day-chuyen": "dây_chuyền", "nhan": "nhẫn",
    "suits-blazer": "vest blazer", "that-lung": "thắt_lưng", "phu-kien-giay": "phụ_kiện giày",
    "ca-vat": "cà_vạt", "vong-tay": "vòng_tay", "dong-ho": "đồng_hồ", "day-deo-quan": "dây_đeo quần",
    "mang-set": "măng_sét", "balo": "balo", "kinh-mat": "kính_mát", "quan-kaki": "quần kaki",
    "quan-boi": "quần bơi", "ao-len": "áo len", "quan-boxer": "quần boxer", "tat": "tất",
    "bong-tai": "bông_tai", "trang-suc": "trang_sức", "non": "nón", "ao-ni": "áo nỉ",
    "ao-sweater": "áo sweater", "gang-tay": "găng_tay", "quan-jogger": "quần jogger",
    "set-quan-ao": "set quần_áo", "moc-khoa": "móc_khóa", "tui-vi-balo": "túi ví balo",
    "balo-tui-xach": "balo túi_xách", "ao-dai": "áo_dài", "nuoc-hoa": "nước_hoa", "khan": "khăn",
    "phu-kien-khac": "phụ_kiện khác", "do-lot": "đồ_lót", "do-giu-nhiet": "đồ giữ_nhiệt",
    "ao-hoodies": "áo hoodie",
}


#regex cho các từ dư thừa 
BAD_DESC_PREFIX = ("< html >", "Chào_mừng mọi người", "Thank you so much", "Giá sản_phẩm trên Tiki")
#loại bỏ HTML, link, số điện thoại, hashtag, emoji, chuẩn hóa khoảng trắng
_RE_HTML    = re.compile(r"<[^>]+>")
_RE_URL     = re.compile(r"https?://\S+|www\.\S+", re.I)
_RE_PHONE   = re.compile(r"\b0\d{8,10}\b")
_RE_HASHTAG = re.compile(r"#\S+")
_RE_EMOJI   = re.compile("[\u2600-\u27BF\U0001F000-\U0001FAFF\uFE0F\u200d]")
_RE_SPACE   = re.compile(r"\s+")
#loại bỏ đoạn giới thiệu/khuyến mãi đặc thù của shop (aristino)
_RE_HEN_PROMO = re.compile(r"^Chương_trình.*?Mô_tả sản_phẩm\s*:?\s*", re.I)       # henstore
_RE_ARISTINO  = re.compile(r"^Tên sản_phẩm\s*:\s*(?:\S+\s)*?(?=\S+\s:\s)", re.I)        # aristino: nhắc lại tên SP
#xóa nội dung trong dấu []
_RE_BRACKET   = re.compile(r"\[[^\]]*\]")  
#phát hiện token rác chỉ toàn ký tự đặc biệt
_RE_JUNK_TOK  = re.compile(r"^[_\W]+$") 
#hỗ trợ nhận diện mã sản phẩm 
_RE_CODE_PART = re.compile(r"^[A-Za-z0-9\-/]+$")
 
#danh sách từ khóa khuyến mãi / nội bộ cần loại khỏi tags
_TAG_BLACKLIST =(
        "voucher", "branday", "donggia", "nguyengia", "hàng cũ", "hàng mới", "tồn", "kiểm_kê",
    "không phân", "hàng tại", "hàng chưa", "khung", "sale", "sptmdt", "o2o", "ecom", "new", "top", "may_mặc",
)

#kiểm tra xem một đoạn text có phải là mã sản phẩm hay không
"""điều kiện: khớp pattern chữ-số-gạch, độ dài >= 5, có ít nhất một chữ số và có chữ cái, giúp model tránh học thuộc mã sản phẩm nếu ảnh không chứa mã """
def _is_code(part: str) -> bool:
    """Mã sản phẩm kiểu AKK0100Z, SP25FH27T-AKJN, 00002906 (>=5 ký tự, có số)."""
    if not _RE_CODE_PART.match(part) or len(part) < 5:
        return False
    has_digit = any(c.isdigit() for c in part)
    has_alpha = any(c.isalpha() for c in part)
    return has_digit and (has_alpha or len(part) >= 6)
 
# loại bỏ mã sản phẩm khỏi text 
# cách hoạt động: tách theo khoảng trắng, bỏ token rác
def strip_codes(text: str) -> str:
    """Bỏ mã sản phẩm (kể cả khi dính trong từ ghép bằng '_'). Mã là định danh duy nhất,
    model dễ học thuộc nó (overfit) mà ảnh thì không hề chứa mã."""
    out = []
    for tok in text.split():
        if _RE_JUNK_TOK.match(tok) and tok not in {".", ",", "-", "–"}:
            continue
        tok = tok.strip("_") or tok
        parts = [p for p in tok.split("_") if not _is_code(p)]
        if parts:
            out.append("_".join(parts))
    return " ".join(out)
 
# cắt text theo tối đa n (giữ từ phần đầu)
def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])
 
#làm sạch tên sản phẩm: xóa nội dung trong [], emoji, chuẩn hóa khoảng trắng
def clean_name(name: str) -> str:
    name = _RE_BRACKET.sub(" ", _RE_EMOJI.sub(" ", name or ""))
    name = strip_codes(_RE_SPACE.sub(" ", name).strip())
    return _truncate_words(name, MAX_NAME_WORDS)
 
#làm sạch mô tả sản phẩm
def clean_description(desc: str, max_words: int = MAX_DESC_WORDS, rng=None) -> str:
    desc = (desc or "").strip()
    if not desc or desc.startswith(BAD_DESC_PREFIX):
        return ""
    desc = _RE_HTML.sub(" ", desc)
    desc = _RE_URL.sub(" ", desc)
    desc = _RE_PHONE.sub(" ", desc)
    desc = _RE_HASHTAG.sub(" ", desc)
    desc = _RE_EMOJI.sub(" ", desc)
    desc = _RE_SPACE.sub(" ", desc).strip()
    desc = _RE_HEN_PROMO.sub("", desc)
    desc = _RE_ARISTINO.sub("", desc)
    desc = strip_codes(desc)
    words = desc.split()
    if rng is not None and len(words) > max_words:      # augmentation khi train: lấy đoạn ngẫu nhiên
        n = rng.randint(max_words // 2, max_words)
        s = rng.randint(0, len(words) - n)
        words = words[s:s + n]
    return " ".join(words[:max_words])
 
"""Làm sạch danh sách tags:
Bỏ tag rỗng, trùng lặp.
Bỏ tag chứa số hoặc % (thường là mã giảm giá, size, năm…).
Bỏ tag viết hoa toàn bộ (thường là mã nội bộ).
Bỏ tag chứa từ trong _TAG_BLACKLIST.
Giữ tối đa max_tags tag."""`
def clean_tags(tags, max_tags: int = MAX_TAGS) -> list:
    out, seen = [], set()
    for t in tags or []:
        t = _RE_SPACE.sub(" ", str(t)).strip()
        low = t.lower()
        if not t or low in seen:
            continue
        if any(c.isdigit() for c in t) or "%" in t:        # mã khuyến mãi, năm, size...
            continue
        letters = [c for c in t if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):   # mã viết hoa: HNG, CHIEC, NGUYENGIA
            continue
        if any(b in low for b in _TAG_BLACKLIST):
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= max_tags:
            break
    return out
 

# xây dựng chuỗi text để đưa vào model theo thứ tự clean_name -> type_vi-> tags đã sạch -> description sạch 
def build_text(product: dict, train: bool = False, rng=None) -> str:
    """Thứ tự: name . type . tags . description  (name đứng đầu vì giàu thông tin nhất)."""
    rng = rng or (random if train else None)
    parts = [
        clean_name(product.get("name", "")),
        TYPE_VI.get(product.get("type"), ""),
        " ".join(clean_tags(product.get("tags"))),
        clean_description(product.get("description"), rng=rng if train else None),
    ]
    return " . ".join(p for p in parts if p)
    