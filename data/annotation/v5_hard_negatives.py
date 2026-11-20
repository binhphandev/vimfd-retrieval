"""
V5 Khai thác âm khó (Hard Negative Mining)
Đầu vào:
  - data/annotation/cross_encoder/pseudo_train.csv, pseudo_val.csv   (V4)
  - data/annotation/cross_encoder/product_attrs.json                  (V3)
  - data/annotation/cross_encoder/attr_dictionary.xlsx                (V3, sheet synonyms + color_map)
  - data/splits/train.jsonl, val.jsonl
  - checkpoints/best_18112026.pt   (dual-encoder B1)
  - config.py                      (MODEL_TEXT_FIELD)
Đầu ra:
  - data/annotation/cross_encoder/ce_pairs_train.csv, ce_pairs_val.csv   (pq_id, query_text, product_id, doc_text, label)
  - index/experiments/train/{product_image.index, id_mapping_image.json}   (chỉ mục Train, KHÔNG ghi đè index serving)
  - index/experiments/val/{product_image.index, id_mapping_image.json}     (chỉ mục Val)
  - results/logs/log_A.md (nối thêm thống kê V5)

Chạy thật : python v5_hard_negatives.py
Thử luồng : python v5_hard_negatives.py --dry-run      (KHÔNG cần checkpoint/ảnh/GPU; dùng TF-IDF thay dual-encoder
                                                         để kiểm tra bộ lọc, chọn âm, kiểm tra 20 cặp, ghi file)
Tùy chọn  : --strict-n1        áp dụng biện pháp khắc phục của kế hoạch 5.3 khi kiểm tra 20 cặp < 18:
                               loại thêm sản phẩm cùng loại khớp từ (n_attrs - 1) thuộc tính trở lên
            --rebuild-index    mã hóa lại ảnh dù đã có chỉ mục lưu sẵn
            --top-k N          (mặc định 50)

Khác bản cũ (ghi vào mục 12 / log_A.md):
  1. Lọc âm giả "tính cả đồng nghĩa" (kế hoạch 5.3): ngoài thuộc tính trong product_attrs, còn dò name/description bằng
     giá trị chuẩn + từ đồng nghĩa; màu thì dò thêm danh sách colors thô và name (sản phẩm nhiều màu có chứa màu truy
     vấn KHÔNG được làm âm). Chủ ý lọc rộng tay: thà mất bớt âm khó còn hơn để âm giả lọt vào.
  2. "Cùng loại" tính theo quan hệ bao hàm (áo khoác ⊂ áo khoác bomber, quần ⊂ quần jean, ví ⊂ ví da), vì sản phẩm
     "áo khoác bomber đen" thật sự liên quan tới truy vấn "áo khoác đen".
  3. Chỉ mục Train và Val lưu riêng dưới index/experiments/{train,val}/ (mục 3.1, 3.5); lần chạy sau dùng lại.
  4. Sản phẩm không có file ảnh KHÔNG đưa vào chỉ mục (bản cũ nạp ảnh đen làm các vector này dính vào nhau).
  5. Ảnh đại diện: is_featured nếu có, nếu không lấy position nhỏ nhất (mục 0).
  6. Mã hóa ảnh bằng DataLoader nhiều luồng; toàn bộ code đặt trong main() nên chạy được cả trên Windows.
  7. Kiểm tra 20 cặp: in bảng để ĐỌC bằng mắt (bước bắt buộc); kiểm tra tự động chỉ xác nhận bộ lọc chạy đúng.
"""

import argparse
import csv
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import openpyxl
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.model import ViMFRModel, load_state_dict_compat


import config
SEED = 42

ROOT         = Path(".")
PSEUDO_TRAIN = ROOT / "data/annotation/cross_encoder/pseudo_train.csv"
PSEUDO_VAL   = ROOT / "data/annotation/cross_encoder/pseudo_val.csv"
ATTRS_PATH   = ROOT / "data/annotation/cross_encoder/product_attrs.json"
DICT_PATH    = ROOT / "data/annotation/cross_encoder/attr_dictionary.xlsx"
TRAIN_PATH   = ROOT / "data/splits/train.jsonl"
VAL_PATH     = ROOT / "data/splits/val.jsonl"
CKPT_PATH    = ROOT / "checkpoints/best_18112026.pt"
OUT_DIR      = ROOT / "data/annotation/cross_encoder"
INDEX_DIR    = ROOT / "index/experiments"
LOG_DIR      = ROOT / "results/logs"

TOP_K        = 50
N_NEG        = 4
DIM          = 256      # ← KIỂM TRA: chiều embedding của dual-encoder B1
IMG_SIZE     = 224    # ← KIỂM TRA: phải đúng với tiền xử lý ảnh lúc B1 đánh giá
IMG_MEAN     = [0.48145466, 0.4578275, 0.40821073]   # ← KIỂM TRA
IMG_STD      = [0.26862954, 0.26130258, 0.27577711]   # ← KIỂM TRA
BATCH_TEXT   = 64
BATCH_IMAGE  = 64
NUM_WORKERS  = 4

try:
    import config                      # config.py của project (ROOT_DIR)
except ImportError:                    # chỉ chấp nhận khi --dry-run
    config = None


# ══════════════════════════════════════════════════════════════════
# Tiện ích chung
# ══════════════════════════════════════════════════════════════════
def normalize(text) -> str:
    """Unicode NFC, chữ thường, bỏ khoảng trắng thừa (mục 3.2)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def load_jsonl_ordered(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_text(product: dict) -> str:
    """Văn bản sản phẩm theo ĐÚNG config của B1 (MODEL_TEXT_FIELD → text → name). Không tự viết lại."""
    field = getattr(config, "MODEL_TEXT_FIELD", "text_seg") if config else "text_seg"
    return product.get(field) or product.get("text") or product.get("name", "")


def featured_image_path(product: dict):
    """is_featured nếu có, nếu không thì position nhỏ nhất. Trả về Path tồn tại trên đĩa hoặc None."""
    images = product.get("images") or []
    if not images:
        return None
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None
    local_path = Path(lp)
    candidates = [local_path, ROOT / local_path]
    if config is not None and not local_path.is_absolute():
        candidates.append(Path(config.PROCESSED_DIR) / local_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


class ImagePathDataset:
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


# ══════════════════════════════════════════════════════════════════
# Bộ so khớp thuộc tính: dùng để lọc âm giả (tính cả đồng nghĩa)
# ══════════════════════════════════════════════════════════════════
# Biến thể viết khác của giá trị chuẩn (ngoài bảng đồng nghĩa)
VARIANTS = {
    "oversize": ["oversized"], "slim fit": ["slimfit"], "regular fit": ["regularfit"],
    "relaxed fit": ["relax fit", "relax"], "straight fit": ["straight"], "skinny": ["skinny fit"],
    "tapered": ["tapered fit", "ống côn"], "wide leg": ["wide-leg", "ống rộng"], "loose fit": ["loose"],
    "smart casual": ["business casual"], "streetwear": ["street style"], "vintage": ["retro", "cổ điển"],
    "minimalist": ["minimal"], "công sở": ["office"], "thể thao": ["sportswear", "sporty"],
    "leather": ["da thật"], "denim": ["jeans"],
}


def _boundary(alts):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


def _clean_color(raw: str) -> str:
    """Cùng quy tắc làm sạch màu như V3."""
    c = normalize(raw)
    c = re.sub(r"^màu\s+", "", c)
    c = re.sub(r"\s*color$", "", c)
    c = re.sub(r"[\.\,;:!]+$", "", c).strip()
    c = re.sub(r"^\d+\s*[-.]\s*", "", c)
    c = re.sub(r"-\d+$", "", c)
    c = re.sub(r"(?<=\D)\s*\d+$", "", c)
    c = re.sub(r"\s+(?:solid|wash)$", "", c)
    return c.strip()


def _tokens(loai: str) -> set:
    return set(loai.split())


def loai_related(a: str, b: str) -> bool:
    """Cùng loại hoặc bao hàm nhau (token của loại này ⊂ loại kia): 'áo khoác' ~ 'áo khoác bomber'."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


class Matcher:
    def __init__(self, products_by_id, product_attrs, synonym_map, color_aliases):
        self.products = products_by_id
        self.pa = product_attrs
        self.syn = synonym_map                # {(attr_type, value_norm): [synonym, ...]}
        self.color_aliases = color_aliases    # {màu chuẩn: {cách viết thô, ...}}
        self._doc, self._name, self._rx, self._color_rx, self._colors = {}, {}, {}, {}, {}

    # --- văn bản chuẩn hóa của sản phẩm (cache) ---
    def _doc_text(self, pid):
        if pid not in self._doc:
            p = self.products[pid]
            self._doc[pid] = normalize((p.get("name") or "") + " . " + (p.get("description") or ""))
        return self._doc[pid]

    def _name_text(self, pid):
        if pid not in self._name:
            self._name[pid] = normalize(self.products[pid].get("name"))
        return self._name[pid]

    def _raw_colors(self, pid):
        if pid not in self._colors:
            cols = self.products[pid].get("colors") or []
            if isinstance(cols, str):
                cols = [cols]
            self._colors[pid] = [_clean_color(c) for c in cols if normalize(c)]
        return self._colors[pid]

    # --- regex theo (nhóm, giá trị) ---
    def _phrase_rx(self, group, value):
        key = (group, value)
        if key not in self._rx:
            alts = {value} | set(self.syn.get(key, [])) | set(VARIANTS.get(value, []))
            self._rx[key] = _boundary(sorted(alts, key=len, reverse=True))
        return self._rx[key]

    def _color_rx_for(self, value):
        if value not in self._color_rx:
            alts = {value} | {a for a in self.color_aliases.get(value, set()) if len(a) >= 2}
            alts |= set(self.syn.get(("MAU", value), []))
            self._color_rx[value] = _boundary(sorted(alts, key=len, reverse=True))
        return self._color_rx[value]

    # --- so khớp một thuộc tính ---
    def attr_match(self, pid, group, value) -> bool:
        if value in (self.pa.get(pid, {}).get(group) or []):
            return True
        if group == "MAU":
            # chỉ dùng colors thô và name (description hay nhắc màu của đồ phối kèm nên không dùng)
            rx = self._color_rx_for(value)
            if any(rx.search(c) for c in self._raw_colors(pid)):
                return True
            return bool(rx.search(self._name_text(pid)))
        return bool(self._phrase_rx(group, value).search(self._doc_text(pid)))

    def n_match(self, pid, query_attrs) -> int:
        return sum(self.attr_match(pid, g, v) for g, v in query_attrs.items())

    def is_false_negative(self, query_attrs, cand_pid, pos_id, strict_n1=False) -> bool:
        """True nếu cand_pid KHÔNG được dùng làm âm.
        Mặc định (kế hoạch 5.3): cùng/bao hàm LOAI với dương VÀ mọi thuộc tính truy vấn đều khớp (tính cả đồng nghĩa).
        strict_n1: khớp từ (n_attrs - 1) thuộc tính trở lên (biện pháp khắc phục khi kiểm tra 20 cặp < 18)."""
        if cand_pid == pos_id:
            return True
        pos_loai = self.pa.get(pos_id, {}).get("LOAI", [])
        cand_loai = self.pa.get(cand_pid, {}).get("LOAI", [])
        if not any(loai_related(a, b) for a in pos_loai for b in cand_loai):
            return False
        need = len(query_attrs)
        if strict_n1:
            need = max(1, len(query_attrs) - 1)
        return self.n_match(cand_pid, query_attrs) >= need


# ══════════════════════════════════════════════════════════════════
# Bộ truy xuất: dual-encoder + FAISS (chạy thật) hoặc TF-IDF (dry-run)
# ══════════════════════════════════════════════════════════════════
class TfidfStandIn:
    """Thay dual-encoder khi --dry-run để thử luồng mà không cần mô hình/ảnh. Không dùng cho kết quả thật."""
    def __init__(self):
        self.pids = []

    def build(self, products, split_name, rebuild=False):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.pids = [p["id"] for p in products]
        self.vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform([get_text(p) for p in products])
        return len(self.pids), 0

    def search(self, query_texts, k):
        Q = self.vec.transform(query_texts)
        out = np.zeros((len(query_texts), k), dtype=np.int64)
        for s in range(0, Q.shape[0], 500):
            sims = (Q[s:s + 500] @ self.X.T).toarray()
            part = np.argpartition(-sims, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
            out[s:s + 500] = np.take_along_axis(part, order, axis=1)
        return out


class DualEncoderRetriever:
    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = torch.cuda.is_available()
        print(f"  Device: {self.device}")
        self._load_model()
        self.pids = []
        self.index = None

    # ---- ← SỬA CHỖ NÀY cho khớp code B1 -------------------------------------------------------------
    def _load_model(self):
        torch = self.torch
        from transformers import AutoTokenizer
        from src.models.model import ViMFRModel, load_state_dict_compat            # ← SỬA tên module

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get(config.CKPT_MODEL_KEY, ckpt)
        model = ViMFRModel()
        load_state_dict_compat(model, state)
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_MODEL)
        print(f"  Checkpoint epoch={ckpt.get(config.CKPT_EPOCH_KEY)}, val_loss={ckpt.get(config.CKPT_VAL_LOSS_KEY)}")

    def _text_forward(self, enc):
        return self.model.encode_text(enc["input_ids"], enc["attention_mask"])                    # ← SỬA nếu tên method khác

    def _image_forward(self, batch):
        return self.model.encode_image(batch)                # ← SỬA nếu tên method khác
    # -------------------------------------------------------------------------------------------------

    def encode_texts(self, texts):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_TEXT):
                enc = self.tokenizer(texts[i:i + BATCH_TEXT], return_tensors="pt", padding=True,
                                     truncation=True, max_length=256)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", enabled=self.use_fp16):
                    v = self._text_forward(enc)
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.vstack(out).astype("float32")

    def encode_images(self, paths):
        torch = self.torch
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(IMG_MEAN, IMG_STD)])

        loader = torch.utils.data.DataLoader(ImagePathDataset(paths, tf), batch_size=BATCH_IMAGE, shuffle=False,
                                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        out = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                with torch.autocast(device_type="cuda", enabled=self.use_fp16):
                    v = self._image_forward(batch)
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.vstack(out).astype("float32")

    def build(self, products, split_name, rebuild=False):
        """Lập (hoặc nạp lại) chỉ mục ảnh của split. Trả về (số sản phẩm trong chỉ mục, số sản phẩm thiếu ảnh)."""
        import faiss
        idx_dir = INDEX_DIR / split_name
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_file, map_file = idx_dir / "product_image.index", idx_dir / "id_mapping_image.json"

        with_img = [(p["id"], featured_image_path(p)) for p in products]
        usable = [(pid, path) for pid, path in with_img if path is not None]
        n_missing = len(products) - len(usable)
        self.pids = [pid for pid, _ in usable]

        if not usable:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh nào cho split {split_name}. "
                f"Kiểm tra local_path và thư mục ảnh {getattr(config, 'PROCESSED_DIR', ROOT)}."
            )

        if idx_file.exists() and map_file.exists() and not rebuild:
            saved = json.load(open(map_file, encoding="utf-8"))
            if saved == self.pids:
                self.index = faiss.read_index(str(idx_file))
                print(f"  Nạp lại chỉ mục {split_name}: {self.index.ntotal} vector ({idx_file}). "
                      "Dùng --rebuild-index nếu đã đổi checkpoint.")
                return len(self.pids), n_missing
            print("  Mapping lưu sẵn không khớp danh sách sản phẩm hiện tại → mã hóa lại ảnh.")

        print(f"  Mã hóa {len(usable)} ảnh {split_name} (thiếu ảnh: {n_missing})...")
        vecs = self.encode_images([path for _, path in usable])
        assert vecs.shape[1] == DIM, f"DIM={DIM} nhưng embedding có {vecs.shape[1]} chiều"
        self.index = faiss.IndexFlatIP(DIM)
        self.index.add(vecs)
        faiss.write_index(self.index, str(idx_file))
        json.dump(self.pids, open(map_file, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  Đã lưu chỉ mục: {idx_file}")
        return len(self.pids), n_missing

    def search(self, query_texts, k):
        qv = self.encode_texts(query_texts)
        _, idx = self.index.search(qv, k)
        return idx


# ══════════════════════════════════════════════════════════════════
# Khai thác âm khó cho một split
# ══════════════════════════════════════════════════════════════════
def mine_hard_negatives(split_name, pseudo_rows, products, matcher, retriever, strict_n1, rebuild, top_k):
    print(f"\n--- {split_name} ---")
    products_by_id = {p["id"]: p for p in products}
    n_idx, n_missing = retriever.build(products, split_name, rebuild)
    pids_in_index = retriever.pids

    print(f"  Tìm top-{top_k} cho {len(pseudo_rows)} truy vấn giả...")
    idx = retriever.search([r["query_text"] for r in pseudo_rows], top_k)

    pairs, neg_info = [], []
    st = Counter()
    for i, row in enumerate(pseudo_rows):
        pos_id, pq_id = row["pos_id"], row["pq_id"]
        qattrs = json.loads(row["attrs"])
        pos = products_by_id.get(pos_id)
        if pos is None:
            st["thiếu_dương"] += 1
            continue

        cands, n_fn = [], 0
        for j in idx[i]:
            if j < 0:
                continue
            cid = pids_in_index[j]
            if matcher.is_false_negative(qattrs, cid, pos_id, strict_n1):
                n_fn += 1
                continue
            cands.append(cid)
        st["ứng_viên_top_k"] += len(idx[i])
        st["bị_loại_(dương+âm_giả)"] += n_fn

        if not cands:
            st["bỏ_vì_0_âm"] += 1
            continue

        rng = random.Random(f"{SEED}:{pq_id}")
        negs = rng.sample(cands, min(N_NEG, len(cands)))
        pairs.append({"pq_id": pq_id, "query_text": row["query_text"], "product_id": pos_id,
                      "doc_text": get_text(pos), "label": 1})
        for nid in negs:
            pairs.append({"pq_id": pq_id, "query_text": row["query_text"], "product_id": nid,
                          "doc_text": get_text(products_by_id[nid]), "label": 0})
            nm = matcher.n_match(nid, qattrs)
            neg_info.append({"pq_id": pq_id, "query_raw": row.get("query_raw", row["query_text"]),
                             "attrs": qattrs, "neg_id": nid, "n_match": nm, "n_attrs": len(qattrs),
                             "same_type": any(loai_related(a, b)
                                              for a in matcher.pa[pos_id]["LOAI"]
                                              for b in matcher.pa[nid]["LOAI"])})
        st["truy_vấn_giữ"] += 1

    n_pos = sum(1 for r in pairs if r["label"] == 1)
    n_neg = len(pairs) - n_pos
    print(f"  Kết quả: {n_pos} dương, {n_neg} âm, tỉ lệ 1:{n_neg / max(n_pos, 1):.2f} | "
          f"bỏ vì 0 âm: {st['bỏ_vì_0_âm']} | thiếu ảnh (không vào chỉ mục): {n_missing}")
    return pairs, neg_info, st, n_missing


def audit_negatives(neg_info, products_by_id, n=20):
    """Bước kiểm tra bắt buộc (5.3): in 20 cặp âm ngẫu nhiên để ĐỌC bằng mắt.
    Kiểm tra tự động chỉ xác nhận bộ lọc đã loại cặp 'cùng loại và khớp đủ thuộc tính'."""
    sample = random.Random(SEED).sample(neg_info, min(n, len(neg_info)))
    print(f"\n=== Kiểm tra bắt buộc: {len(sample)} cặp âm ngẫu nhiên (đọc bằng mắt) ===")
    n_ok = 0
    for k, r in enumerate(sample, 1):
        name = (products_by_id[r["neg_id"]].get("name") or "")[:70]
        ok = not (r["same_type"] and r["n_match"] >= r["n_attrs"])
        n_ok += ok
        print(f"{k:>2}. [{'OK ' if ok else 'ÂM GIẢ?'}] q='{r['query_raw']}' attrs={json.dumps(r['attrs'], ensure_ascii=False)}\n"
              f"      âm: {name} | khớp {r['n_match']}/{r['n_attrs']} thuộc tính | cùng loại: {r['same_type']}")
    verdict = "ĐẠT" if n_ok >= 18 else "KHÔNG ĐẠT → chạy lại với --strict-n1 rồi mới sang V6"
    print(f"\n  Tự động: {n_ok}/{len(sample)} ({verdict}).")
    print("  LƯU Ý: kết quả tự động ở trên chỉ chứng minh bộ lọc chạy đúng. Hãy đọc 20 cặp trên: ít nhất 18 cặp phải "
          "thật sự KHÔNG khớp thuộc tính truy vấn (nhìn cả name sản phẩm). Nếu <18: --strict-n1.")
    return n_ok


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Thử luồng bằng TF-IDF, không cần mô hình/ảnh/GPU")
    ap.add_argument("--strict-n1", action="store_true", help="Loại thêm sản phẩm cùng loại khớp ≥ n_attrs-1 thuộc tính")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.dry_run:
        print("=== DRY RUN: dùng TF-IDF thay dual-encoder; KẾT QUẢ KHÔNG DÙNG ĐỂ HUẤN LUYỆN ===")
    if config is None and not args.dry_run:
        raise SystemExit("Không import được config.py. Chạy từ thư mục ROOT_DIR (chứa config.py).")

    print("Load dữ liệu...")
    train_products = load_jsonl_ordered(TRAIN_PATH)
    val_products = load_jsonl_ordered(VAL_PATH)
    product_attrs = json.load(open(ATTRS_PATH, encoding="utf-8"))
    pseudo_train, pseudo_val = load_csv(PSEUDO_TRAIN), load_csv(PSEUDO_VAL)
    print(f"  Train {len(train_products)} | Val {len(val_products)} | "
          f"pseudo_train {len(pseudo_train)} | pseudo_val {len(pseudo_val)}")

    # từ điển: đồng nghĩa + bảng màu thô -> chuẩn
    wb = openpyxl.load_workbook(DICT_PATH, read_only=True)
    synonym_map = defaultdict(list)
    for row in wb["synonyms"].iter_rows(min_row=2, values_only=True):
        norm, atype, syn = row[:3]
        if norm and atype and syn:
            synonym_map[(atype, normalize(norm))].append(normalize(syn))
    color_aliases = defaultdict(set)
    for row in wb["color_map"].iter_rows(min_row=2, values_only=True):
        raw, norm = row[:2]
        if raw and norm:
            color_aliases[normalize(norm)].add(normalize(raw))
    print(f"  Đồng nghĩa: {sum(len(v) for v in synonym_map.values())} cặp | màu thô đã map: "
          f"{sum(len(v) for v in color_aliases.values())}")

    all_by_id = {p["id"]: p for p in train_products + val_products}
    matcher = Matcher(all_by_id, product_attrs, synonym_map, color_aliases)

    if args.dry_run:
        retriever_train, retriever_val = TfidfStandIn(), TfidfStandIn()
    else:
        retriever_train = DualEncoderRetriever()
        retriever_val = retriever_train          # cùng mô hình, hai chỉ mục tách riêng theo split

    results = {}
    for name, rows, prods in [("train", pseudo_train, train_products), ("val", pseudo_val, val_products)]:
        retr = retriever_train if name == "train" else retriever_val
        results[name] = mine_hard_negatives(name, rows, prods, matcher, retr,
                                            args.strict_n1, args.rebuild_index, args.top_k)

    # kiểm tra bắt buộc (mục 5.3) trên Train
    train_pairs, train_negs, train_st, train_missing = results["train"]
    n_ok = audit_negatives(train_negs, all_by_id) if train_negs else 0

    # lưu CSV
    fieldnames = ["pq_id", "query_text", "product_id", "doc_text", "label"]
    suffix = "_DRYRUN" if args.dry_run else ""
    for name in ("train", "val"):
        pairs = results[name][0]
        out_path = OUT_DIR / f"ce_pairs_{name}{suffix}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(pairs)
        print(f"Đã lưu: {out_path}  ({len(pairs)} dòng)")

    # log
    lines = ["\n## V5 — khai thác âm khó" + (" (DRY RUN – không dùng)" if args.dry_run else ""),
             f"- Tham số: seed={SEED}, top_k={args.top_k}, n_neg={N_NEG}, strict_n1={args.strict_n1}, "
             f"thời gian={time.time() - t0:.0f}s"]
    for name in ("train", "val"):
        pairs, negs, st, miss = results[name]
        n_pos = sum(1 for r in pairs if r["label"] == 1)
        n_neg = len(pairs) - n_pos
        n_q = max(st["truy_vấn_giữ"], 1)
        same_type = sum(1 for r in negs if r["same_type"]) / max(len(negs), 1)
        n1 = sum(1 for r in negs if r["same_type"] and r["n_match"] == r["n_attrs"] - 1) / max(len(negs), 1)
        lines.append(f"- {name}: {n_pos} dương, {n_neg} âm (1:{n_neg / max(n_pos, 1):.2f}); bỏ vì 0 âm: {st['bỏ_vì_0_âm']}; "
                     f"thiếu ảnh: {miss}; TB bị loại/truy vấn: {st['bị_loại_(dương+âm_giả)'] / n_q:.1f}/{args.top_k}; "
                     f"âm cùng loại: {same_type:.0%}; âm cùng loại và khớp n−1 thuộc tính: {n1:.0%}")
    lines.append(f"- Kiểm tra 20 cặp (tự động): {n_ok}/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log")
    report = "\n".join(lines)
    print(report)
    with open(LOG_DIR / "log_A.md", "a", encoding="utf-8") as f:
        f.write(report + "\n")

    if args.dry_run:
        print("\nDRY RUN xong. Chạy thật: python v5_hard_negatives.py (sau khi sửa 3 chỗ '← SỬA' của B1).")
    else:
        print("\nV5 hoàn thành! Đọc 20 cặp ở trên: nếu ĐẠT → chạy ngay v6_train_cross_encoder.py "
              "(nếu <18 → chạy lại với --strict-n1).")


if __name__ == "__main__":
    main()