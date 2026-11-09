"""Clean product text and build the model input string."""

import random
import re
from typing import Any

# Maximum text lengths.
MAX_DESC_WORDS = 60
MAX_NAME_WORDS = 40
MAX_TAGS = 5

# Vietnamese labels for product types.
TYPE_VI = {
    "ao-so-mi": "áo sơ_mi",
    "giay-dep": "giày_dép",
    "ao-polo": "áo polo",
    "quan-jean": "quần jean",
    "ao-thun": "áo thun",
    "quan-short": "quần short",
    "vi": "ví",
    "ao-khoac": "áo_khoác",
    "quan-au": "quần âu",
    "tui": "túi",
    "ao-tanktop": "áo tanktop",
    "quan-brief": "quần brief",
    "quan-lot": "quần_lót",
    "cap-da": "cặp da",
    "day-chuyen": "dây_chuyền",
    "nhan": "nhẫn",
    "suits-blazer": "vest blazer",
    "that-lung": "thắt_lưng",
    "phu-kien-giay": "phụ_kiện giày",
    "ca-vat": "cà_vạt",
    "vong-tay": "vòng_tay",
    "dong-ho": "đồng_hồ",
    "day-deo-quan": "dây_đeo quần",
    "mang-set": "măng_sét",
    "balo": "balo",
    "kinh-mat": "kính_mát",
    "quan-kaki": "quần kaki",
    "quan-boi": "quần bơi",
    "ao-len": "áo len",
    "quan-boxer": "quần boxer",
    "tat": "tất",
    "bong-tai": "bông_tai",
    "trang-suc": "trang_sức",
    "non": "nón",
    "ao-ni": "áo nỉ",
    "ao-sweater": "áo sweater",
    "gang-tay": "găng_tay",
    "quan-jogger": "quần jogger",
    "set-quan-ao": "set quần_áo",
    "moc-khoa": "móc_khóa",
    "tui-vi-balo": "túi ví balo",
    "balo-tui-xach": "balo túi_xách",
    "ao-dai": "áo_dài",
    "nuoc-hoa": "nước_hoa",
    "khan": "khăn",
    "phu-kien-khac": "phụ_kiện khác",
    "do-lot": "đồ_lót",
    "do-giu-nhiet": "đồ giữ_nhiệt",
    "ao-hoodies": "áo hoodie",
}


# Text cleanup rules.
BAD_DESC_PREFIX = (
    "< html >",
    "Chào_mừng mọi người",
    "Thank you so much",
    "Giá sản_phẩm trên Tiki",
)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_RE_PHONE = re.compile(r"\b0\d{8,10}\b")
_RE_HASHTAG = re.compile(r"#\S+")
_RE_EMOJI = re.compile(
    "[\u2600-\u27BF\U0001F000-\U0001FAFF\uFE0F\u200d]"
)
_RE_SPACE = re.compile(r"\s+")
_RE_HEN_PROMO = re.compile(r"^Chương_trình.*?Mô_tả sản_phẩm\s*:?\s*", re.I)
_RE_ARISTINO = re.compile(
    r"^Tên sản_phẩm\s*:\s*(?:\S+\s)*?(?=\S+\s:\s)", re.I
)
_RE_BRACKET = re.compile(r"\[[^\]]*\]")
_RE_JUNK_TOK = re.compile(r"^[_\W]+$")
_RE_CODE_PART = re.compile(r"^[A-Za-z0-9\-/]+$")

_TAG_BLACKLIST = (
    "voucher",
    "branday",
    "donggia",
    "nguyengia",
    "hàng cũ",
    "hàng mới",
    "tồn",
    "kiểm_kê",
    "không phân",
    "hàng tại",
    "hàng chưa",
    "khung",
    "sale",
    "sptmdt",
    "o2o",
    "ecom",
    "new",
    "top",
    "may_mặc",
)


def _is_code(part: str) -> bool:
    """Return whether a token part looks like a product code."""
    if not _RE_CODE_PART.match(part) or len(part) < 5:
        return False
    has_digit = any(c.isdigit() for c in part)
    has_alpha = any(c.isalpha() for c in part)
    return has_digit and (has_alpha or len(part) >= 6)


def strip_codes(text: str) -> str:
    """Remove product-code parts and punctuation-only tokens."""
    cleaned_tokens = []
    for token in text.split():
        if _RE_JUNK_TOK.match(token) and token not in {".", ",", "-", "–"}:
            continue
        token = token.strip("_") or token
        parts = [part for part in token.split("_") if not _is_code(part)]
        if parts:
            cleaned_tokens.append("_".join(parts))
    return " ".join(cleaned_tokens)


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])


def clean_name(name: str) -> str:
    name = _RE_BRACKET.sub(" ", _RE_EMOJI.sub(" ", name or ""))
    name = strip_codes(_RE_SPACE.sub(" ", name).strip())
    return _truncate_words(name, MAX_NAME_WORDS)


def clean_description(
    desc: str,
    max_words: int = MAX_DESC_WORDS,
    rng: Any = None,
) -> str:
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
    if rng is not None and len(words) > max_words:
        n = rng.randint(max_words // 2, max_words)
        s = rng.randint(0, len(words) - n)
        words = words[s:s + n]
    return " ".join(words[:max_words])


def clean_tags(tags: Any, max_tags: int = MAX_TAGS) -> list[str]:
    """Remove empty, duplicate, promotional, and code-like tags."""
    cleaned_tags = []
    seen = set()
    for tag in tags or []:
        tag = _RE_SPACE.sub(" ", str(tag)).strip()
        lowered = tag.lower()
        if not tag or lowered in seen:
            continue
        if any(char.isdigit() for char in tag) or "%" in tag:
            continue
        letters = [char for char in tag if char.isalpha()]
        if len(letters) >= 2 and all(char.isupper() for char in letters):
            continue
        if any(keyword in lowered for keyword in _TAG_BLACKLIST):
            continue
        seen.add(lowered)
        cleaned_tags.append(tag)
        if len(cleaned_tags) >= max_tags:
            break
    return cleaned_tags


def build_text(
    product: dict[str, Any],
    train: bool = False,
    rng: Any = None,
) -> str:
    """Join cleaned name, type, tags, and description in model-input order."""
    rng = rng or (random if train else None)
    product_type = product.get("type")
    name = product.get("name")
    description = product.get("description")
    parts = [
        clean_name(name if isinstance(name, str) else ""),
        TYPE_VI.get(product_type if isinstance(product_type, str) else "", ""),
        " ".join(clean_tags(product.get("tags"))),
        clean_description(
            description if isinstance(description, str) else "",
            rng=rng if train else None,
        ),
    ]
    return " . ".join(p for p in parts if p)


def segment_text(text: str) -> str:
    """Word-segment Vietnamese text for PhoBERT."""
    if not text:
        return text
    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise ImportError(
            "Thiếu underthesea. Cài đặt: pip install underthesea"
        ) from exc
    return str(word_tokenize(text, format="text"))


def get_model_text(
    product: dict[str, Any],
    model_text_field: str = "retrieval_text",
) -> str:
    """Return stored model text or build it consistently for legacy records."""
    model_text = product.get(model_text_field)
    if isinstance(model_text, str) and model_text.strip():
        return model_text
    return segment_text(build_text(product))


def preprocess_query_text(query: str) -> str:
    """Apply the product-name cleanup and PhoBERT segmentation to a query."""
    return segment_text(clean_name(query))
    