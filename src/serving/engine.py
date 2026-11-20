"""
SearchEngine — khớp với train_colab.ipynb.

Mọi truy vấn đều so với IMAGE index (product_image.index), giống notebook:
    Text  -> Image : cos(encode_text(segment(query)), embedding ảnh sản phẩm)   (search_text)
    Image -> Image : cos(encode_image(ảnh upload),    embedding ảnh sản phẩm)   (search_image)
Text index (product_text.index) KHÔNG còn được API dùng.

Quy tắc segment cho query chữ: underthesea.word_tokenize(query, format="text"),
đúng như embed_texts(segment=True) trong notebook (dữ liệu train đã tách từ kiểu "Áo_Thun").

Khi có cả chữ và ảnh: hai điểm cosine nằm ở hai thang khác nhau (T->I thấp hơn I->I nhiều),
nên mỗi tín hiệu được chuẩn hóa z-score trên toàn gallery trước khi trộn:
    score = (1 - w) * z(T->I) + w * z(I->I)
Chỉ có một loại truy vấn thì score = cosine thô (xếp hạng y hệt notebook).

Khi khởi động engine kiểm tra index có khớp checkpoint và dữ liệu hiện tại hay không.
"""

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from underthesea import word_tokenize

import config as cfg
from src.models.model import ViMFRModel, load_state_dict_compat
from src.serving.catalog import (
    ckpt_fingerprint,
    get_eval_transform,
    load_products,
    meta_path,
    primary_image,
    resolve_image,
)

log = logging.getLogger(__name__)

CANDIDATES = 200  # số kết quả tối đa giữ lại sau xếp hạng (rồi mới phân trang / sắp xếp giá)

BRAND_FIELD = getattr(cfg, "BRAND_FIELD", "shop")
CATEGORY_FIELD = getattr(cfg, "CATEGORY_FIELD", "category")

KEYWORDS = [
    "áo sơ mi", "áo thun", "áo khoác", "quần", "trắng", "đen", "xanh", "be", "xám",
    "slim fit", "slim", "oxford", "linen", "polo", "bomber", "kaki",
]


def parse_query(text: str) -> list[str]:
    t = (text or "").lower()
    return [k for k in KEYWORDS if re.search(rf"(^|\s){re.escape(k)}(\s|$)", t)]


def segment(text: str) -> str:
    """Đúng như notebook: word_tokenize(t, format='text')."""
    return word_tokenize(text.strip(), format="text")


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _price(p: dict) -> float:
    """Schema: "price": {"current": 59000, "original": null, "currency": "VND"}."""
    v = p.get("price")
    if isinstance(v, dict):
        v = v.get("current") or v.get("original")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _load_index(index_file, mapping_file) -> tuple[np.ndarray, list[str]]:
    index = faiss.read_index(str(index_file))
    emb = index.reconstruct_n(0, index.ntotal).astype(np.float32)  # chỉ dùng được với IndexFlat*
    with open(mapping_file, encoding="utf-8") as f:
        mapping = json.load(f)
    return emb, [str(mapping[str(i)]) for i in range(index.ntotal)]


class SearchEngine:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.dev}")
        allow_stale = bool(getattr(cfg, "ALLOW_STALE_INDEX", False))

        # --- model: cùng checkpoint với index ---
        ckpt = torch.load(cfg.CHECKPOINT_BEST, map_location=self.dev)
        self.model = ViMFRModel()
        load_state_dict_compat(self.model, ckpt[cfg.CKPT_MODEL_KEY])
        self.model.to(self.dev).eval()
        self.tok = AutoTokenizer.from_pretrained(cfg.PHOBERT_MODEL)
        self.tf = get_eval_transform()
        segment("áo sơ mi")  # khởi động sẵn underthesea

        # --- index ảnh ---
        emb, ids = _load_index(cfg.FAISS_IMAGE_INDEX_FILE, cfg.FAISS_IMAGE_MAPPING_FILE)
        self.meta = self._check_checkpoint(allow_stale)

        # --- dữ liệu hiện tại (SPLIT_DIR) ---
        catalog = {str(p[cfg.PRODUCT_ID_FIELD]): p for p in load_products()}
        stale = [i for i in ids if i not in catalog]
        if stale:
            msg = (f"Image index có {len(stale)}/{len(ids)} id không nằm trong dữ liệu hiện tại "
                   f"({cfg.SPLIT_DIR}), ví dụ {stale[:3]}. Hãy build lại index: python -m src.indexing.build_index")
            if not allow_stale:
                raise RuntimeError(msg)
            log.warning(msg)
        rows = [r for r, i in enumerate(ids) if i in catalog]
        self.emb = emb[rows]
        self.pids = [ids[r] for r in rows]
        self.pos = {p: n for n, p in enumerate(self.pids)}
        self.items = [catalog[p] for p in self.pids]
        not_indexed = len(catalog) - len(self.pids)
        if not_indexed:
            log.warning(f"{not_indexed} sản phẩm không có trong image index (thiếu ảnh) nên không tìm được")

        self.price = np.array([_price(it) for it in self.items], dtype=np.float32)
        self.hay = [f"{it.get('name', '')} {it.get(CATEGORY_FIELD, '')}".lower() for it in self.items]
        self.hay_color = [
            f"{it.get('name', '')} {' '.join(map(str, it.get('colors') or []))}".lower()
            for it in self.items
        ]
        self.img_url = [self._image_url(it) for it in self.items]
        log.info(f"Engine ready: {len(self.pids)} products (image index)")

    # ---------- kiểm tra index ----------
    @staticmethod
    def _check_checkpoint(allow_stale: bool) -> dict:
        mp = meta_path(cfg.FAISS_IMAGE_INDEX_FILE)
        if not mp.exists():
            log.warning(f"Không thấy {mp.name}: index được build bằng bản build_index cũ, "
                        "không kiểm tra được checkpoint. Nên build lại index.")
            return {}
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint") != ckpt_fingerprint(cfg.CHECKPOINT_BEST):
            msg = (f"Image index được build từ checkpoint khác với {cfg.CHECKPOINT_BEST} "
                   f"(index: {meta.get('checkpoint')}). Build lại index hoặc đặt đúng CHECKPOINT_BEST.")
            if not allow_stale:
                raise RuntimeError(msg)
            log.warning(msg)
        return meta

    # ---------- encode ----------
    @torch.inference_mode()
    def _enc_text(self, q: str) -> np.ndarray:
        enc = self.tok(segment(q), max_length=cfg.MAX_TEXT_LENGTH, padding="max_length",
                       truncation=True, return_tensors="pt")
        e = self.model.encode_text(enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev))
        return e[0].float().cpu().numpy()

    @torch.inference_mode()
    def _enc_image(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.dev)
        return self.model.encode_image(x)[0].float().cpu().numpy()

    # ---------- helpers ----------
    @staticmethod
    def _image_url(p: dict) -> str:
        """Ưu tiên ảnh local (phục vụ qua /media), không có thì dùng URL gốc của shop."""
        img = primary_image(p) or {}
        rp = resolve_image(img.get(cfg.IMAGE_PATH_FIELD, ""))
        if rp:
            try:
                return "/media/" + quote(rp.relative_to(Path(cfg.IMAGE_DIR)).as_posix())
            except ValueError:
                pass
        return img.get("url", "")

    def _card(self, n: int, score=None, st=None, si=None) -> dict:
        p = self.items[n]
        f = lambda v: None if v is None else round(float(v), 4)
        return {
            "id": self.pids[n],
            "name": p.get("name", ""),
            "price": self.price[n].item(),
            "brand": p.get(BRAND_FIELD, ""),
            "category": p.get(CATEGORY_FIELD, ""),
            "image_url": self.img_url[n],
            "url": p.get("url", ""),
            "source": p.get("source", ""),
            "colors": [str(c) for c in (p.get("colors") or [])],
            "score": f(score),
            "score_text": f(st),     # cosine text -> image
            "score_image": f(si),    # cosine image -> image
        }

    def _filter_mask(self, types, colors, price_min, price_max) -> np.ndarray:
        m = np.ones(len(self.pids), dtype=bool)
        if price_min is not None:
            m &= self.price >= price_min
        if price_max is not None:
            m &= self.price <= price_max
        for words, hay in ((types, self.hay), (colors, self.hay_color)):
            words = [w.lower() for w in words if w]
            if words:
                m &= np.array([any(w in h for w in words) for h in hay])
        return m

    # ---------- public ----------
    def search(self, *, text: str = "", image: Image.Image | None = None, w: float = 0.5,
               types=(), colors=(), price_min=None, price_max=None,
               sort: str = "rel", page: int = 1, page_size: int = 12) -> dict:
        t0 = time.perf_counter()
        q = text.strip() if text else ""
        st = self.emb @ self._enc_text(q) if q else None          # T -> I
        si = self.emb @ self._enc_image(image) if image is not None else None   # I -> I
        if st is None and si is None:
            raise ValueError("Cần nhập mô tả hoặc tải ảnh")

        if st is not None and si is not None:
            score = (1 - w) * _z(st) + w * _z(si)
        else:
            score = st if st is not None else si

        idx = np.flatnonzero(self._filter_mask(types, colors, price_min, price_max))
        idx = idx[np.argsort(-score[idx])][:CANDIDATES]
        if sort in ("asc", "desc"):
            order = np.argsort(self.price[idx], kind="stable")
            idx = idx[order if sort == "asc" else order[::-1]]

        sel = idx[(page - 1) * page_size: page * page_size]
        g = lambda arr, n: None if arr is None else arr[n]
        return {
            "items": [self._card(n, score[n], g(st, n), g(si, n)) for n in sel],
            "total": len(idx),
            "page": page,
            "parsed": parse_query(text),
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def get(self, pid: str) -> dict | None:
        n = self.pos.get(pid)
        if n is None:
            return None
        card = self._card(n)
        card["description"] = self.items[n].get("description", "")
        card["sizes"] = [str(x) for x in (self.items[n].get("sizes") or [])]
        return card

    def similar(self, pid: str, k: int = 4) -> list[dict] | None:
        """Sản phẩm giống nhất về ảnh (I -> I, như search_image trong notebook)."""
        n = self.pos.get(pid)
        if n is None:
            return None
        s = self.emb @ self.emb[n]
        s[n] = -np.inf  # bỏ chính nó
        top = np.argsort(-s)[:k]
        return [self._card(m, s[m], None, s[m]) for m in top]