"""Build a catalog-grounded, stratified draft of 200 test queries.

The seed product documents query evidence only; it is not a relevance label.
Review the generated workbook before using it as a benchmark, then pool and
re-annotate results for every query.
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from functools import lru_cache
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.text_processing.text_utils import clean_colors, clean_description, clean_name

DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "test.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "annotation" / "query_list_200_v2_single_color_review.xlsx"
CATEGORY_LABELS = {"ao": "Áo", "quan": "Quần", "phu-kien": "Phụ kiện"}
QUOTAS = {
    "ao": {1: 30, 2: 30, 3: 30},
    "quan": {1: 10, 2: 10, 3: 24},
    "phu-kien": {1: 27, 2: 27, 3: 12},
}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")
_EXCEL_ILLEGAL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

TYPE_RULES = {
    "ao-so-mi": (("áo sơ mi", ("áo sơ mi", "sơ mi", "shirt")),),
    "ao-polo": (("áo polo", ("áo polo", "polo")),),
    "ao-thun": (("áo thun", ("áo thun", "áo phông", "t-shirt", "t shirt", "tee")),),
    "ao-khoac": (("áo khoác", ("áo khoác", "jacket", "coat")),),
    "ao-tanktop": (("áo tanktop", ("tanktop", "tank top", "áo ba lỗ")),),
    "ao-hoodies": (("áo hoodie", ("áo hoodie", "hoodie", "hoodies")),),
    "ao-ni": (("áo nỉ", ("áo nỉ", "nỉ")),),
    "ao-sweater": (("áo sweater", ("áo sweater", "sweater", "sweatshirt")),),
    "ao-len": (("áo len", ("áo len", "len")),),
    "suits-blazer": (("áo vest", ("áo vest", "vest", "blazer", "suit")),),
    "quan-jean": (("quần jean", ("quần jean", "quần jeans", "quần bò", "jeans")),),
    "quan-short": (("quần short", ("quần short", "quần đùi", "shorts")),),
    "quan-au": (("quần âu", ("quần âu", "quần tây", "trousers")),),
    "quan-kaki": (("quần kaki", ("quần kaki", "quần khaki", "kaki", "khaki")),),
    "quan-brief": (("quần brief", ("quần brief", "briefs")),),
    "quan-lot": (("quần lót", ("quần lót", "quần sịp", "quần sip", "underwear")),),
    "quan-boxer": (("quần boxer", ("quần boxer", "boxer shorts")),),
    "quan-boi": (("quần bơi", ("quần bơi", "swim shorts")),),
    "quan-jogger": (("quần jogger", ("quần jogger", "jogger")),),
    "giay-dep": (
        ("giày", ("giày", "shoes", "shoe", "boots", "boot")),
        ("dép", ("dép", "sandals", "sandal", "clog")),
    ),
    "vi": (("ví", ("ví", "bóp", "wallet")),),
    "cap-da": (("cặp", ("cặp", "briefcase")),),
    "tui": (("túi", ("túi", "bag", "handbag")),),
    "balo": (("balo", ("balo", "ba lô", "backpack")),),
    "ca-vat": (("cà vạt", ("cà vạt", "cravat", "necktie")),),
    "mang-set": (("măng sét", ("măng sét", "măng-sét", "cufflink", "cuff links")),),
    "that-lung": (("thắt lưng", ("thắt lưng", "dây lưng", "belt")),),
    "day-deo-quan": (
        ("dây đeo quần", ("dây đeo quần", "dây treo quần", "suspenders")),
    ),
    "nhan": (("nhẫn", ("nhẫn", "ring")),),
    "dong-ho": (("đồng hồ", ("đồng hồ", "watch")),),
    "day-chuyen": (("dây chuyền", ("dây chuyền", "dây đeo cổ", "necklace")),),
    "vong-tay": (("vòng tay", ("vòng tay", "bracelet")),),
    "bong-tai": (("bông tai", ("bông tai", "hoa tai", "earrings")),),
    "phu-kien-giay": (
        ("phụ kiện giày", ("phụ kiện giày", "shoe accessory", "shoe accessories")),
    ),
    "khan": (("khăn", ("khăn", "scarf", "shawl")),),
    "kinh-mat": (("kính", ("kính", "glasses", "sunglasses", "eyewear")),),
    "non": (
        ("nón", ("nón", "cap")),
        ("mũ", ("mũ", "hat")),
    ),
    "tat": (
        ("tất", ("tất", "vớ", "socks")),
    ),
    "gang-tay": (("găng tay", ("găng tay", "gloves")),),
    "set-quan-ao": (
        ("bộ quần áo", ("bộ quần áo", "set quần áo", "set đồ")),
    ),
}
SIZE_TYPES = {
    "ao-so-mi", "ao-polo", "ao-thun", "ao-khoac", "ao-tanktop",
    "ao-hoodies", "ao-ni", "ao-sweater", "ao-len", "quan-jean",
    "quan-short", "quan-au", "quan-kaki", "quan-brief", "quan-lot",
    "quan-boxer", "quan-boi", "quan-jogger", "giay-dep", "nhan",
    "gang-tay", "suits-blazer", "set-quan-ao",
}
COLOR_TERMS = (
    ("xanh navy", ("xanh navy", "xanh than", "navy blue", "navy")),
    ("xanh dương", (
        "xanh dương", "xanh da trời", "xanh jeans", "xanh jean",
        "xanh biển", "xanh lơ", "xanh nhạt", "xanh đậm", "blue",
    )),
    ("xanh lá", ("xanh lá", "xanh lá cây", "green", "rêu", "olive")),
    ("đen", ("đen", "black")),
    ("trắng", ("trắng", "white")),
    ("xám", ("xám", "ghi", "grey", "gray")),
    ("nâu", ("nâu", "brown")),
    ("be", ("beige", "be", "cream", "kem")),
    ("đỏ", ("đỏ", "red")),
    ("hồng", ("hồng", "pink")),
    ("vàng", ("vàng", "yellow", "gold")),
    ("tím", ("tím", "purple")),
    ("cam", ("cam", "orange")),
    ("bạc", ("bạc", "silver")),
    ("kaki", ("kaki", "khaki")),
)
COMPOUND_COLOR_TERMS = (
    "xanh tim than", "vang hong", "xanh den", "xam den", "den trang",
    "den xam", "hong do",
)
MATERIAL_TERMS = (
    ("da thật", ("da thật", "da that", "leather")),
    ("da lộn", ("da lộn", "da lon", "suede")),
    ("cotton", ("cotton",)),
    ("linen", ("linen",)),
    ("denim", ("denim", "jean")),
    ("kaki", ("kaki",)),
    ("polyester", ("polyester",)),
    ("nylon", ("nylon",)),
    ("ramie", ("ramie",)),
    ("lụa", ("lụa", "lua", "silk")),
    ("len", ("len", "wool")),
    ("canvas", ("canvas",)),
    ("spandex", ("spandex",)),
    ("rayon", ("rayon",)),
    ("voan", ("voan",)),
)
MATERIAL_INCOMPATIBLE_TYPES = {
    "lụa": {"nhan", "giay-dep", "mang-set"},
    "len": {"nhan", "giay-dep", "mang-set"},
    "denim": {"nhan", "giay-dep", "mang-set"},
}
_SIZE_PREFIX = re.compile(r"^\s*(?:int|eu|size|số|so)\s*:?\s*", re.I)
_SIZE_ALPHA = re.compile(
    r"^(?:XXXXS|XXXXL|XXXS|XXS|XS|S|M|L|XL|XXL|XXXL|[2-9]XL)"
    r"(?:\s*(?:\[[^\]]*\]|\([^)]*\)))?$",
    re.I,
)
_SIZE_NUMERIC = re.compile(r"^\d{1,2}(?:\.5)?$")


@lru_cache(maxsize=256)
def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return _SPACE.sub(" ", _NON_ALNUM.sub(" ", value)).strip()


def _has_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize(text)} "
    normalized_phrase = f" {_normalize(phrase)} "
    return normalized_phrase in normalized_text


def _name_overlap(query: str, name: str) -> float:
    query_tokens = set(_normalize(query).split())
    name_tokens = set(_normalize(name).split())
    if not query_tokens:
        return 0.0
    return len(query_tokens & name_tokens) / len(query_tokens)


def _excel_safe_text(value: str) -> str:
    return _EXCEL_ILLEGAL_CHARACTERS.sub("", value)


def _extract_terms(
    text: str,
    term_table: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[dict[str, str]]:
    normalized_text = f" {_normalize(text)} "
    matches = []
    for display, aliases in term_table:
        for alias in aliases:
            if f" {_normalize(alias)} " not in normalized_text:
                continue
            span = _find_phrase_span(text, alias)
            if span is None:
                continue
            context = _normalize(
                text[max(0, span[0] - 32):min(len(text), span[1] + 32)]
            )
            if display == "trơn" and re.search(
                r"\b(?:chong\s+tron|tron\s+truot|chong\s+truot)\b",
                context,
            ):
                continue
            matches.append({
                "term": display,
                "alias": alias,
                "excerpt": _description_excerpt(text, span),
            })
            break
    return matches


def _find_phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    target = _normalize(phrase).split()
    tokens = list(re.finditer(r"[^\W_]+", text, flags=re.UNICODE))
    normalized_tokens = [_normalize(token.group()) for token in tokens]
    for index in range(len(tokens) - len(target) + 1):
        if normalized_tokens[index:index + len(target)] == target:
            return tokens[index].start(), tokens[index + len(target) - 1].end()
    return None


def _description_excerpt(
    text: str,
    span: tuple[int, int],
    context: int = 90,
) -> str:
    start = max(0, span[0] - context)
    end = min(len(text), span[1] + context)
    excerpt = text[start:end].strip()
    if start:
        excerpt = "... " + excerpt
    if end < len(text):
        excerpt += " ..."
    return excerpt


def _clean_full_description(value: Any) -> str:
    text = str(value or "")
    return clean_description(text, max_words=max(60, len(text) + 1))


def _match_product_type(product_type: str, name: str) -> str | None:
    for display, aliases in TYPE_RULES.get(product_type, ()):
        if any(_has_phrase(name, alias) for alias in aliases):
            return display
    return None


def _single_color(color: str) -> str | None:
    normalized = _normalize(color)
    if any(f" {term} " in f" {normalized} " for term in COMPOUND_COLOR_TERMS):
        return None

    matches = {
        display
        for display, aliases in COLOR_TERMS
        if any(_has_phrase(color, alias) for alias in aliases)
    }
    if "xanh navy" in matches:
        matches.discard("xanh dương")

    if "xanh" in normalized.split() and not matches.intersection(
        {"xanh navy", "xanh dương", "xanh lá"}
    ):
        matches.add("xanh")
    return next(iter(matches)) if len(matches) == 1 else None


def _extract_colors(product: dict[str, Any]) -> list[tuple[str, str]]:
    colors = clean_colors(product.get("colors"), str(product.get("name") or ""))
    extracted = []
    seen = set()
    for color in colors:
        display = _single_color(color)
        if display and display not in seen:
            extracted.append((display, color))
            seen.add(display)
    return extracted


def _extract_sizes(product: dict[str, Any]) -> list[tuple[str, str]]:
    raw_sizes = product.get("sizes")
    if isinstance(raw_sizes, str):
        values = [raw_sizes]
    elif isinstance(raw_sizes, list):
        values = raw_sizes
    else:
        return []

    extracted = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        raw_value = _SPACE.sub(" ", value).strip()
        without_prefix = _SIZE_PREFIX.sub("", raw_value, count=1)
        for option in without_prefix.split("|"):
            option = option.strip()
            if _SIZE_ALPHA.fullmatch(option):
                normalized = option.split("[", 1)[0].split("(", 1)[0].strip().upper()
                if normalized == "XXXXL":
                    normalized = "4XL"
            elif _SIZE_NUMERIC.fullmatch(option):
                numeric_size = float(option)
                if numeric_size == 0:
                    continue
                normalized = (
                    str(numeric_size).removesuffix(".0")
                    if "." in option
                    else str(int(option))
                )
            else:
                continue
            if normalized not in seen:
                extracted.append((normalized, raw_value))
                seen.add(normalized)
    return extracted


def _material_is_compatible(
    product_type: str,
    material: str,
    product_name: str,
    description: str,
) -> bool:
    if product_type in MATERIAL_INCOMPATIBLE_TYPES.get(material, set()):
        return False

    normalized_text = _normalize(f"{product_name} {description}")
    material_terms = dict(MATERIAL_TERMS)
    for other_material, aliases in MATERIAL_TERMS:
        if other_material == material:
            continue
        for alias in aliases:
            alias_tokens = _normalize(alias).split()
            alias_pattern = r"\s+".join(re.escape(token) for token in alias_tokens)
            if re.search(
                rf"\b(?:5[0-9]|[6-9][0-9]|100)\s+{alias_pattern}\b",
                normalized_text,
            ):
                return False

    aliases = material_terms.get(material, ())
    for alias in aliases:
        alias_tokens = _normalize(alias).split()
        alias_pattern = r"\s+".join(re.escape(token) for token in alias_tokens)
        if re.search(
            rf"\b(?:khong|not)\s+(?:(?:phai|dung|co)\s+)?{alias_pattern}\b",
            normalized_text,
        ):
            return False
    return True


def _query_text(
    product_type: str,
    color: str,
    size: str,
    description_terms: list[str],
) -> str:
    parts = [product_type, *description_terms, "màu", color]
    if size:
        parts.extend(("cỡ", size))
    return " ".join(parts)


def _build_candidates(
    products: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        category = product.get("category")
        product_type = product.get("type")
        product_id = product.get("id")
        if category not in QUOTAS or not isinstance(product_type, str) or not product_id:
            continue

        product_name = str(product.get("name") or "")
        type_text = _match_product_type(product_type, product_name)
        if not type_text:
            continue
        raw_colors = product.get("colors")
        if not isinstance(raw_colors, list) or len(raw_colors) != 1:
            continue
        colors = _extract_colors(product)
        description = _clean_full_description(product.get("description"))
        materials = [
            term for term in _extract_terms(description, MATERIAL_TERMS)
            if _material_is_compatible(
                product_type,
                term["term"],
                product_name,
                description,
            )
        ]
        sizes = _extract_sizes(product) if product_type in SIZE_TYPES else []
        size_based = bool(sizes)

        if not colors or not product_id:
            continue
        source = str(product_id).split("_", 1)[0].casefold()
        images = product.get("images")
        first_image = images[0] if isinstance(images, list) and images else {}
        image_url = (
            str(first_image.get("url") or "")
            if isinstance(first_image, dict)
            else ""
        )
        base_values = {
            "group": CATEGORY_LABELS[category],
            "category_code": category,
            "product_type_code": product_type,
            "product_type": type_text,
            "source": source,
            "seed_product_id": str(product_id),
            "evidence_product_name": _excel_safe_text(clean_name(product_name)),
            "image_url": image_url,
        }

        def add_candidate(
            level: int,
            color: str,
            raw_color: str,
            *,
            size: str = "",
            raw_size: str = "",
            material: dict[str, str] | None = None,
            second_material: dict[str, str] | None = None,
        ) -> None:
            terms = [
                term["term"]
                for term in (material, second_material)
                if term is not None
            ]
            query = _query_text(type_text, color, size, terms)
            if level == 1:
                intent = "màu"
            elif size_based and level == 2:
                intent = "màu + cỡ"
            elif size_based:
                intent = "từ khóa mô tả + màu + cỡ"
            elif level == 2:
                intent = "từ khóa mô tả + màu"
            else:
                intent = "hai chất liệu + màu"

            bundle = (product_type, color, size)
            material_bundle = tuple(sorted(terms))
            excerpts = [
                term["excerpt"]
                for term in (material, second_material)
                if term is not None
            ]
            candidates[(category, level)].append({
                **base_values,
                "color": color,
                "evidence_color": raw_color,
                "size": size,
                "evidence_size": raw_size,
                "description_terms": " + ".join(terms),
                "evidence_excerpt": " ... ".join(excerpts),
                "query": query,
                "intent": intent,
                "attribute_count": level,
                "bundle": (*bundle, material_bundle),
                "type_color_pair": (product_type, color),
            })

        for color, raw_color in colors:
            add_candidate(1, color, raw_color)
            if size_based:
                for size, raw_size in sizes:
                    add_candidate(2, color, raw_color, size=size, raw_size=raw_size)
                    for material in materials:
                        add_candidate(
                            3,
                            color,
                            raw_color,
                            size=size,
                            raw_size=raw_size,
                            material=material,
                        )
            else:
                for material in materials:
                    add_candidate(2, color, raw_color, material=material)
                for first_material, second_material in combinations(materials, 2):
                    add_candidate(
                        3,
                        color,
                        raw_color,
                        material=first_material,
                        second_material=second_material,
                    )
    return candidates


def _allocate_source_quotas(
    products: list[dict[str, Any]],
    total_queries: int,
    source_capacity: dict[str, int] | None = None,
) -> dict[str, int]:
    source_counts = Counter(
        str(product.get("id") or "").split("_", 1)[0].casefold()
        for product in products
        if product.get("id")
    )
    product_total = sum(source_counts.values())
    if product_total == 0:
        raise ValueError("Không thể phân bổ quota nguồn: test không có ID sản phẩm.")

    quotas = {
        source: total_queries * count // product_total
        for source, count in source_counts.items()
    }
    remaining = total_queries - sum(quotas.values())
    remainders = sorted(
        source_counts,
        key=lambda source: (
            -(total_queries * source_counts[source] % product_total),
            source,
        ),
    )
    for source in remainders[:remaining]:
        quotas[source] += 1

    if source_capacity is None:
        return quotas

    capacities = {
        source: source_capacity.get(source, 0)
        for source in source_counts
    }
    if sum(capacities.values()) < total_queries:
        raise ValueError(
            "Không đủ ID nguồn có ứng viên hợp lệ để sinh "
            f"{total_queries} query: sức chứa={capacities}."
        )
    for source in quotas:
        quotas[source] = min(quotas[source], capacities[source])
    remaining = total_queries - sum(quotas.values())
    while remaining:
        eligible = [
            source for source in source_counts
            if quotas[source] < capacities[source]
        ]
        eligible_weight = sum(source_counts[source] for source in eligible)
        ideal_additions = {
            source: remaining * source_counts[source] / eligible_weight
            for source in eligible
        }
        added = 0
        for source in eligible:
            addition = min(
                capacities[source] - quotas[source],
                int(ideal_additions[source]),
            )
            quotas[source] += addition
            added += addition
        remaining -= added
        if not remaining:
            break
        if not added:
            source = max(
                eligible,
                key=lambda key: (
                    ideal_additions[key] % 1,
                    source_counts[key],
                    key,
                ),
            )
            quotas[source] += 1
            remaining -= 1
    return quotas


def _select_candidates(
    candidates_by_cell: dict[tuple[str, int], list[dict[str, Any]]],
    source_quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining_cells = {
        (category, level): quota
        for category, levels in QUOTAS.items()
        for level, quota in levels.items()
    }
    selected_sources: Counter[str] = Counter()
    used_queries: set[str] = set()
    used_bundles: set[tuple[Any, ...]] = set()
    type_color_counts: Counter[tuple[str, str]] = Counter()
    selected = []

    while any(remaining_cells.values()):
        by_cell_source: dict[
            tuple[str, int], dict[str, list[dict[str, Any]]]
        ] = {}
        for cell, remaining in remaining_cells.items():
            if not remaining:
                continue
            source_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates_by_cell.get(cell, []):
                source = candidate["source"]
                if _normalize(candidate["query"]) in used_queries:
                    continue
                if candidate["bundle"] in used_bundles:
                    continue
                if type_color_counts[candidate["type_color_pair"]] >= 3:
                    continue
                source_candidates[source].append(candidate)
            if source_candidates:
                by_cell_source[cell] = source_candidates

        if not by_cell_source:
            unfilled_cells = {
                f"{CATEGORY_LABELS[category]}:{level}": count
                for (category, level), count in remaining_cells.items()
                if count
            }
            unfilled_sources = {
                source: target - selected_sources[source]
                for source, target in source_quotas.items()
                if selected_sources[source] < target
            }
            raise ValueError(
                "Không đủ query thỏa các quota/bộ lọc; "
                f"ô còn thiếu={unfilled_cells}, nguồn còn thiếu={unfilled_sources}"
            )

        def cell_scarcity(
            item: tuple[tuple[str, int], dict[str, list[dict[str, Any]]]],
        ) -> tuple[float, int]:
            cell, by_source = item
            pair_bundles: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
            for rows in by_source.values():
                for candidate in rows:
                    pair_bundles[candidate["type_color_pair"]].add(
                        candidate["bundle"]
                    )
            capacity = sum(
                min(
                    len(bundles),
                    3 - type_color_counts[pair],
                )
                for pair, bundles in pair_bundles.items()
            )
            return capacity / remaining_cells[cell], capacity

        cell = min(by_cell_source.items(), key=cell_scarcity)[0]
        source_options = by_cell_source[cell]
        source = max(
            source_options,
            key=lambda key: source_quotas.get(key, 0) - selected_sources[key],
        )
        candidates = source_options[source]
        lowest_pair_count = min(
            type_color_counts[candidate["type_color_pair"]]
            for candidate in candidates
        )
        candidates = [
            candidate for candidate in candidates
            if type_color_counts[candidate["type_color_pair"]] == lowest_pair_count
        ]
        candidate = rng.choice(candidates)
        selected.append(candidate)
        selected_sources[source] += 1
        used_queries.add(_normalize(candidate["query"]))
        used_bundles.add(candidate["bundle"])
        type_color_counts[candidate["type_color_pair"]] += 1
        remaining_cells[cell] -= 1

    return selected


def build_query_rows(
    products: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    candidates = _build_candidates(products)
    total_queries = sum(sum(levels.values()) for levels in QUOTAS.values())
    source_capacity = {
        source: len({
            (candidate["type_color_pair"], candidate["bundle"])
            for rows in candidates.values()
            for candidate in rows
            if candidate["source"] == source
        })
        for source in {
            candidate["source"]
            for rows in candidates.values()
            for candidate in rows
        }
    }
    eligible_seed_ids = {
        candidate["seed_product_id"]
        for rows in candidates.values()
        for candidate in rows
    }
    source_quotas = _allocate_source_quotas(
        [{"id": product_id} for product_id in eligible_seed_ids],
        total_queries,
        source_capacity,
    )
    last_error = None
    rows = []
    for attempt in range(20):
        rng = random.Random(seed + attempt)
        try:
            rows = _select_candidates(candidates, source_quotas, rng)
            break
        except ValueError as error:
            last_error = error
    else:
        raise ValueError(f"Không thể cân bằng query theo bộ lọc/quota: {last_error}")

    rng.shuffle(rows)
    for number, row in enumerate(rows, start=1):
        row["stt"] = number
        row["name_overlap"] = _name_overlap(
            row["query"],
            row["evidence_product_name"],
        )
    if len(rows) != total_queries:
        raise AssertionError(f"Expected {total_queries} queries, got {len(rows)}.")
    if len({_normalize(row["query"]) for row in rows}) != total_queries:
        raise AssertionError("Generated query text contains normalized duplicates.")
    return rows


def _write_workbook(rows: list[dict[str, Any]], output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing workbook: {output_path}")
    workbook = openpyxl.Workbook()
    query_sheet = workbook.active
    query_sheet.title = "Query List"
    headers = [
        "STT",
        "Nhóm",
        "Truy vấn",
        "Ý định",
        "Số thuộc tính",
        "Nguồn",
        "ID mẫu",
        "Tên sản phẩm (đã làm sạch)",
        "Ảnh đại diện (URL)",
        "Màu",
        "Màu nguồn",
        "Cỡ",
        "Cỡ nguồn",
        "Từ khóa mô tả",
        "Đoạn trích mô tả",
    ]
    query_sheet.append(headers)
    for row in rows:
        values = [
            row["stt"],
            row["group"],
            row["query"],
            row["intent"],
            row["attribute_count"],
            row["source"],
            row["seed_product_id"],
            row["evidence_product_name"],
            row["image_url"],
            row["color"],
            row["evidence_color"],
            row["size"],
            row["evidence_size"],
            row["description_terms"],
            row["evidence_excerpt"],
        ]
        query_sheet.append([
            _excel_safe_text(value) if isinstance(value, str) else value
            for value in values
        ])
        image_cell = query_sheet.cell(row=query_sheet.max_row, column=9)
        if row["image_url"].startswith(("http://", "https://")):
            image_cell.hyperlink = row["image_url"]
            image_cell.style = "Hyperlink"

    query_sheet.freeze_panes = "D2"
    query_sheet.auto_filter.ref = query_sheet.dimensions
    widths = [8, 16, 48, 34, 16, 16, 30, 48, 64, 16, 24, 12, 24, 28, 72]
    for index, width in enumerate(widths, start=1):
        query_sheet.column_dimensions[openpyxl.utils.get_column_letter(index)].width = width
    for cell in query_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E79")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in query_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    criteria = workbook.create_sheet("Hướng dẫn")
    notes = [
        ["Mục", "Hướng dẫn"],
        ["Mục đích", "Đây là danh sách truy vấn dự thảo, chưa phải nhãn liên quan."],
        ["Hợp lệ khi", "Seed có đúng một phần tử trong metadata colors; đúng loại sản phẩm; câu tự nhiên; màu/cỡ/từ khóa mô tả có bằng chứng; không mâu thuẫn hoặc trùng ý."],
        ["Kiểm tra nguồn", "Đối chiếu màu/cỡ với metadata; từ khóa mô tả phải khớp đoạn trích. Mở URL ảnh để xác nhận màu. ID mẫu chỉ là nguồn tạo query, không phải sản phẩm đúng."],
        ["Trùng tên", "Tỷ lệ trên 0.8 được liệt kê bên dưới để xem lại; đây là cảnh báo, không tự động loại truy vấn."],
        ["Số thuộc tính", "Đếm Màu, Cỡ và Từ khóa mô tả; không tính loại sản phẩm."],
        ["Phân bổ quota", "Mục tiêu 200 query; quota nhóm Áo 90, Quần 44, Phụ kiện 66 để phù hợp số cặp loại-màu đơn sắc hợp lệ. Số thuộc tính mục tiêu 1/2/3 là 67/67/66. Nếu không đủ ứng viên, workbook báo số thực tế; không nới bộ lọc."],
        ["ID mẫu", "Một sản phẩm có thể làm seed cho nhiều query nếu các query cùng cặp loại-màu khác nhau về cỡ hoặc chất liệu. ID seed chỉ là bằng chứng sinh query, không phải nhãn liên quan."],
        ["Phân bổ nguồn", "Số lượng theo nguồn phản ánh các ứng viên còn lại sau bộ lọc đơn màu và quota nhóm/thuộc tính; không nhất thiết bằng tỷ lệ nguồn thô trong toàn bộ test."],
        ["Chuẩn hóa cỡ", "Chỉ giữ cỡ chữ cái, cỡ số 1-2 chữ số hoặc .5, và dạng có tiền tố Int/EU/Size rõ ràng. Bỏ one size, EU đứng riêng, khoảng số, cân nặng và giá trị lẫn màu."],
        ["Từ khóa mô tả", "Trích theo ranh giới từ trên toàn bộ description đã làm sạch; đoạn trích hiển thị vùng có từ khóa."],
        ["Kiểm tra tự động", "Seed chỉ lấy từ sản phẩm có đúng một giá trị colors; query không trùng; tối đa ba query mỗi cặp loại-màu và các query trong cặp phải khác cỡ hoặc chất liệu."],
        ["Nếu không hợp lệ", "Sửa câu nếu giữ được ý định; nếu không, thay truy vấn và kiểm tra lại tính duy nhất."],
        ["Sau khi duyệt", "Lưu thành query_list_200_v2_schema_approved.xlsx; chạy mọi baseline và ViMFR trên cùng 200 truy vấn, cùng gallery; lưu top-40 để tạo pool."],
        ["Gán nhãn", "Gán nhãn lại các cặp trong pool bởi hai người. Không chuyển nhãn từ query list v1 sang v2."],
    ]
    for row in notes:
        criteria.append(row)
    criteria.append([])
    criteria.append(["STT", "Truy vấn", "Tên sản phẩm", "Trùng tên"])
    for row in rows:
        if row["name_overlap"] > 0.8:
            criteria.append([
                row["stt"],
                row["query"],
                row["evidence_product_name"],
                row["name_overlap"],
            ])
    criteria.column_dimensions["A"].width = 24
    criteria.column_dimensions["B"].width = 112
    criteria.column_dimensions["C"].width = 64
    criteria.column_dimensions["D"].width = 16
    for row in criteria.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    workbook.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a catalog-grounded draft of 200 retrieval queries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        products = json.load(file)
    if not isinstance(products, list):
        raise ValueError(f"{args.input} must contain a JSON array.")

    rows = build_query_rows(products, seed=args.seed)
    _write_workbook(rows, args.output)
    print(f"Draft created: {args.output}")
    print(f"Queries: {len(rows)}")
    print(json.dumps({
        "category_counts": dict(Counter(row["group"] for row in rows)),
        "attribute_counts": dict(Counter(row["attribute_count"] for row in rows)),
        "category_attribute_counts": {
            group: dict(Counter(
                row["attribute_count"] for row in rows if row["group"] == group
            ))
            for group in CATEGORY_LABELS.values()
        },
        "source_counts": dict(Counter(row["source"] for row in rows)),
    }, ensure_ascii=True))
    print("Review every query using the workbook instructions before pooling.")


if __name__ == "__main__":
    main()
