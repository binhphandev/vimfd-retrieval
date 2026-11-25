"""
V8: Chạy dual-encoder và ViMFR trên 200 truy vấn kiểm thử
Đầu vào:
  - data/annotation/benchmark/queries_200.csv
  - data/splits/test.jsonl
  - checkpoint dual-encoder B1: config.CHECKPOINT_BEST, hoặc biến môi trường VIMFD_DE_CKPT (PHẢI là checkpoint
    train trên split hiện tại; checkpoint train trên split cũ có thể đã thấy sản phẩm Test mới -> kết quả bị thổi phồng)
  - checkpoints/cross_encoder_best/     (Cross-Encoder V6)
Đầu ra:
  - results/runs/dual_encoder.run
  - results/runs/vimfr.run
  - results/logs/latency.csv
Chạy: python v8_run_vimfr.py              (đầy đủ 200 truy vấn)
      python v8_run_vimfr.py --sample5    (thử 5 truy vấn, ghi vào results/runs_sample/, không đè kết quả thật)
      python v8_run_vimfr.py --rebuild-index   (mã hóa lại ảnh Test dù đã có chỉ mục lưu sẵn)

BẢN SỬA (so với v8_run_vimfr_1.py) — ghi vào mục 12 / log:
  1. Truy vấn được CHUẨN HÓA + TÁCH TỪ (seg_utils: VnCoreNLP, cùng bộ tách từ với text_seg) trước khi mã hóa/chấm điểm: dual-encoder và
     Cross-Encoder đều được huấn luyện trên truy vấn đã tách từ (mục 3.2, 7).
  2. Ảnh đại diện lấy bằng đúng hàm của V5 (is_featured/position, thử local_path, ROOT/local_path,
     config.PROCESSED_DIR/local_path). Bản cũ chỉ thử đường dẫn tương đối theo thư mục chạy rồi âm thầm thay bằng
     ẢNH ĐEN khi không thấy: vector ảnh đen làm hỏng chỉ mục. Nay sản phẩm thiếu ảnh KHÔNG vào chỉ mục (như V5/V7).
  3. Đo thời gian đủ 3 bước theo mục 7: mã hóa truy vấn (gồm tách từ), FAISS, rerank; có đồng bộ GPU trước khi bấm giờ.
  4. Chỉ mục Test lưu ở index/experiments/test/ và được nạp lại ở lần chạy sau.
  5. Kiểm tra đầy đủ: product_id thuộc Test, không trùng, score giảm dần, hai file cùng tập 50 ứng viên.
"""

import argparse
import csv
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import torch
import faiss
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

SEED = 42
ROOT = Path(__file__).resolve().parents[2]

QUERIES_CSV  = ROOT / "data/annotation/benchmark/queries_200.csv"
TEST_PATH    = ROOT / "data/splits/test.jsonl"
DE_CKPT      = None   # xác định sau khi import config (xem _resolve_de_ckpt bên dưới)
CE_CKPT_DIR  = ROOT / "checkpoints/cross_encoder_best"
RUNS_DIR     = ROOT / "results/runs"
LOG_DIR      = ROOT / "results/logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50
INDEX_DIR = ROOT / "index/experiments/test"

_ap = argparse.ArgumentParser()
_ap.add_argument("--sample5", action="store_true")
_ap.add_argument("--rebuild-index", action="store_true")
ARGS = _ap.parse_args()
if ARGS.sample5:
    RUNS_DIR = ROOT / "results/runs_sample"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load queries ──────────────────────────────────────────────────
queries = []
with open(QUERIES_CSV, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        queries.append({"query_id": row["query_id"].strip(), "query_text": row["query_text"]})
if ARGS.sample5:
    queries = queries[:5]
print(f"Số truy vấn: {len(queries)}" + ("  [SAMPLE 5]" if ARGS.sample5 else ""))

# ── Load test products ────────────────────────────────────────────
test_products = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            test_products.append(json.loads(line))
test_pids = [p["id"] for p in test_products]
print(f"Test corpus: {len(test_products)} sản phẩm")

import os
import sys
sys.path.insert(0, str(ROOT))
import config

def _resolve_de_ckpt():
    """VIMFD_DE_CKPT > config.CHECKPOINT_BEST > checkpoints/best_24112026.pt."""
    p = os.environ.get("VIMFD_DE_CKPT") or getattr(config, "CHECKPOINT_BEST", None) or "checkpoints/best_24112026.pt"
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def ckpt_fingerprint(p):
    """Dấu vân tay checkpoint (tên:kích thước:thời điểm sửa) để biết chỉ mục lưu sẵn có thuộc checkpoint này không."""
    st = Path(p).stat()
    return f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}"


DE_CKPT = _resolve_de_ckpt()
if not DE_CKPT.exists():
    raise SystemExit(f"Không thấy checkpoint dual-encoder: {DE_CKPT}. Đặt VIMFD_DE_CKPT hoặc sửa config.CHECKPOINT_BEST.")

def get_text(product):
    field = getattr(config, "MODEL_TEXT_FIELD", "text_seg")
    return product.get(field) or product.get("text") or product.get("name", "")

def featured_image_path(product):
    """Giống V5: is_featured trước, rồi position nhỏ nhất; trả về Path tồn tại trên đĩa hoặc None."""
    images = product.get("images") or []
    if not images:
        return None
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None
    local_path = Path(lp)
    candidates = [local_path, ROOT / local_path]
    if getattr(config, "PROCESSED_DIR", None) and not local_path.is_absolute():
        candidates.append(Path(config.PROCESSED_DIR) / local_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def normalize(text):
    """Mục 3.2: Unicode NFC, chữ thường, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


from seg_utils import segment_text, segmenter_name


def segment_query(text):
    """Chuẩn hóa rồi tách từ bằng bộ tách từ thống nhất (đúng như truy vấn giả V4 và như text_seg)."""
    return segment_text(text)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

# ── Load dual-encoder ─────────────────────────────────────────────
import datetime as _dt
print(f"\nLoad dual-encoder: {DE_CKPT}  (sửa lần cuối {_dt.datetime.fromtimestamp(DE_CKPT.stat().st_mtime):%Y-%m-%d %H:%M})")
print("  LƯU Ý: checkpoint này PHẢI được train trên split hiện tại (11.571/1.450/1.449).")
print(f"  Bộ tách từ truy vấn: {segmenter_name()}")
from src.models.model import ViMFRModel, load_state_dict_compat

de_ckpt = torch.load(DE_CKPT, map_location=device, weights_only=False)
dual_encoder = ViMFRModel()
load_state_dict_compat(
    dual_encoder,
    de_ckpt.get(config.CKPT_MODEL_KEY, de_ckpt),
)
dual_encoder.to(device).eval()

de_tokenizer = AutoTokenizer.from_pretrained(
    getattr(config, "PHOBERT_MODEL", "vinai/phobert-base"))

DIM        = 256
IMG_SIZE   = 224
BATCH_SIZE = 64
USE_FP16   = torch.cuda.is_available()

img_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean=getattr(config, "IMAGE_MEAN", [0.48145466, 0.4578275, 0.40821073]),
        std=getattr(config, "IMAGE_STD", [0.26862954, 0.26130254, 0.27577711]),
    ),
])

@torch.no_grad()
def encode_texts_de(texts):
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = de_tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", enabled=USE_FP16):
            v = dual_encoder.encode_text(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.cpu().float().numpy())
    return np.vstack(vecs).astype("float32")

@torch.no_grad()
def encode_images_de(paths):
    vecs = []
    for i in range(0, len(paths), BATCH_SIZE):
        imgs = [img_transform(Image.open(p).convert("RGB")) for p in paths[i:i+BATCH_SIZE]]
        tensor = torch.stack(imgs).to(device)
        with torch.autocast(device_type="cuda", enabled=USE_FP16):
            v = dual_encoder.encode_image(tensor)   # ← SỬA nếu khác
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.cpu().float().numpy())
    return np.vstack(vecs).astype("float32")

# ── Build index Test (chỉ sản phẩm có file ảnh) ───────────────────
usable = [(p, featured_image_path(p)) for p in test_products]
n_missing = sum(1 for _, path in usable if path is None)
usable = [(p, path) for p, path in usable if path is not None]
index_products = [p for p, _ in usable]
index_pids = [p["id"] for p in index_products]
print(f"\nSản phẩm Test có ảnh: {len(index_products)}/{len(test_products)} (thiếu ảnh: {n_missing})")
if n_missing:
    from collections import Counter
    print("  Thiếu ảnh theo shop:", dict(Counter(p["id"].split("_")[0] for p in test_products
                                                 if featured_image_path(p) is None)))
    print("  [CẢNH BÁO] Các sản phẩm này không thể được tìm thấy bởi hệ thống dùng ảnh. Kiểm tra config.PROCESSED_DIR.")

idx_file, map_file = INDEX_DIR / "product_image.index", INDEX_DIR / "id_mapping_image.json"
fp_file = INDEX_DIR / "ckpt_fingerprint.txt"
_fp = ckpt_fingerprint(DE_CKPT)
index = None
if idx_file.exists() and map_file.exists() and not ARGS.rebuild_index:
    if not (fp_file.exists() and fp_file.read_text(encoding="utf-8").strip() == _fp):
        print("  Chỉ mục Test lưu sẵn thuộc checkpoint khác (hoặc chưa có dấu vân tay) -> mã hóa lại.")
    elif json.load(open(map_file, encoding="utf-8")) == index_pids:
        index = faiss.read_index(str(idx_file))
        print(f"Nạp lại chỉ mục Test: {index.ntotal} vector (--rebuild-index nếu đã đổi checkpoint B1)")
if index is None:
    print("Mã hóa ảnh sản phẩm Test → FAISS index...")
    image_vecs = encode_images_de([path for _, path in usable])
    assert image_vecs.shape[1] == DIM, f"DIM={DIM} nhưng embedding có {image_vecs.shape[1]} chiều"
    index = faiss.IndexFlatIP(DIM)
    index.add(image_vecs)
    faiss.write_index(index, str(idx_file))
    json.dump(index_pids, open(map_file, "w", encoding="utf-8"), ensure_ascii=False)
    fp_file.write_text(_fp, encoding="utf-8")
    print(f"Index: {index.ntotal} vectors (đã lưu {idx_file})")

# ── Load Cross-Encoder ────────────────────────────────────────────
print(f"\nLoad Cross-Encoder: {CE_CKPT_DIR}")

class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert   = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]
        logits = self.linear(cls).squeeze(-1)
        return torch.sigmoid(logits)   # điểm 0–1

ce_model    = CrossEncoder(str(CE_CKPT_DIR)).to(device).eval()
try:
    linear_data = torch.load(CE_CKPT_DIR / "linear_head.pt", map_location=device, weights_only=False)
except TypeError:
    linear_data = torch.load(CE_CKPT_DIR / "linear_head.pt", map_location=device)
ce_model.linear.load_state_dict(linear_data["linear_state_dict"])
ce_tokenizer = AutoTokenizer.from_pretrained(str(CE_CKPT_DIR))
print(f"  Best epoch: {linear_data['best_epoch']}, val_loss={linear_data['val_loss']:.4f}")

@torch.no_grad()
def rerank_with_ce(query: str, candidates: list[dict]) -> list[float]:
    """Chấm điểm 50 cặp trong 1 batch, trả về list score."""
    doc_texts = [get_text(p) for p in candidates]
    enc = ce_tokenizer(
        [query] * len(doc_texts),
        doc_texts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast(device_type="cuda", enabled=USE_FP16):
        scores = ce_model(**enc)
    return scores.cpu().float().tolist()

# ── Chạy retrieval + rerank, đo latency (từng truy vấn một, như lúc phục vụ thật) ─────────────
print("\nChạy mã hóa truy vấn + FAISS + rerank...")
latency_rows, de_run_rows, vimfr_run_rows = [], [], []
WARMUP_N = 5   # 5 truy vấn đầu làm nóng, không tính latency

for i, query in enumerate(queries):
    qid, qraw = query["query_id"], query["query_text"]

    # --- mã hóa truy vấn (gồm chuẩn hóa + tách từ) ---
    sync(); t0 = time.perf_counter()
    qseg = segment_query(qraw)
    q_vec = encode_texts_de([qseg])
    sync(); t_encode = time.perf_counter() - t0

    # --- FAISS ---
    t0 = time.perf_counter()
    scores_faiss, indices = index.search(q_vec, TOP_K)
    t_faiss = time.perf_counter() - t0

    keep = [(j, float(scores_faiss[0][k])) for k, j in enumerate(indices[0]) if j >= 0]
    cand_pids     = [index_pids[j] for j, _ in keep]
    cand_scores   = [sc for _, sc in keep]
    cand_products = [index_products[j] for j, _ in keep]

    for rank, (pid, score) in enumerate(zip(cand_pids, cand_scores), start=1):
        de_run_rows.append(f"{qid} Q0 {pid} {rank} {score:.6f} dual_encoder")

    # --- Cross-Encoder rerank ---
    sync(); t0 = time.perf_counter()
    ce_scores = rerank_with_ce(qseg, cand_products)
    sync(); t_ce = time.perf_counter() - t0

    reranked = sorted(zip(cand_pids, ce_scores), key=lambda x: -x[1])
    for rank, (pid, score) in enumerate(reranked, start=1):
        vimfr_run_rows.append(f"{qid} Q0 {pid} {rank} {score:.6f} vimfr")

    if i >= WARMUP_N:
        latency_rows.append({
            "query_id": qid,
            "t_encode_ms": t_encode * 1000,
            "t_faiss_ms":  t_faiss * 1000,
            "t_rerank_ms": t_ce * 1000,
            "total_ms":    (t_encode + t_faiss + t_ce) * 1000,
        })
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(queries)} truy vấn xong")

# ── Kiểm tra ─────────────────────────────────────────────────────
NQ = len(queries)
assert len(de_run_rows) == NQ * TOP_K, f"DE run thiếu dòng: {len(de_run_rows)}"
assert len(vimfr_run_rows) == NQ * TOP_K, f"ViMFR run thiếu dòng: {len(vimfr_run_rows)}"
test_id_set = set(test_pids)

def check_run(rows, name, descending=True):
    per = {}
    for r in rows:
        qid, _, pid, rank, score, sysname = r.split()
        assert pid in test_id_set and sysname == name, f"{name}: dòng lỗi {r}"
        per.setdefault(qid, []).append((pid, float(score)))
    for qid, items in per.items():
        pids = [p for p, _ in items]
        sc = [s for _, s in items]
        assert len(set(pids)) == len(pids), f"{name}: trùng sản phẩm ở {qid}"
        assert all(sc[k] >= sc[k + 1] - 1e-9 for k in range(len(sc) - 1)), f"{name}: score không giảm ở {qid}"
    print(f"  ✅ {name}: {len(rows)} dòng hợp lệ")

check_run(de_run_rows, "dual_encoder")
check_run(vimfr_run_rows, "vimfr")

# Hai file phải cùng tập 50 ứng viên
for i, q in enumerate(queries):
    de_pids    = set(r.split()[2] for r in de_run_rows[i*TOP_K:(i+1)*TOP_K])
    vimfr_pids = set(r.split()[2] for r in vimfr_run_rows[i*TOP_K:(i+1)*TOP_K])
    assert de_pids == vimfr_pids, f"Tập product_id khác nhau tại {q['query_id']}!"

# Cảnh báo nếu thứ tự giống nhau > 90%
n_same_order = 0
for i in range(NQ):
    de_order    = [r.split()[2] for r in de_run_rows[i*TOP_K:(i+1)*TOP_K]]
    vimfr_order = [r.split()[2] for r in vimfr_run_rows[i*TOP_K:(i+1)*TOP_K]]
    if de_order == vimfr_order:
        n_same_order += 1
if n_same_order > 0.9 * NQ:
    print(f"⚠️  CẢNH BÁO: {n_same_order}/{NQ} truy vấn có thứ tự giống hệt DE! Kiểm tra Cross-Encoder checkpoint.")
else:
    print(f"✅ Rerank hợp lệ: {n_same_order}/{NQ} truy vấn giữ nguyên thứ tự")

# Đọc bằng mắt top-3 của vài truy vấn
names = {p["id"]: (p.get("name") or "") for p in test_products}
print("\nTop-3 (đọc bằng mắt):")
for i in range(min(5, NQ)):
    qid = queries[i]["query_id"]
    print(f"  {qid} '{queries[i]['query_text']}'")
    for tag, rows in (("DE   ", de_run_rows), ("ViMFR", vimfr_run_rows)):
        top = [names[r.split()[2]][:45] for r in rows[i*TOP_K:i*TOP_K+3]]
        print(f"     {tag}: " + " | ".join(top))

# ── Lưu run files ─────────────────────────────────────────────────
for name, rows in [("dual_encoder", de_run_rows), ("vimfr", vimfr_run_rows)]:
    out = RUNS_DIR / f"{name}.run"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Lưu: {out}  ({len(rows)} dòng)")

# ── Lưu latency ──────────────────────────────────────────────────
lat_path = LOG_DIR / "latency.csv"
with open(lat_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["query_id","t_encode_ms","t_faiss_ms","t_rerank_ms","total_ms"])
    writer.writeheader()
    writer.writerows(latency_rows)

totals = [r["total_ms"] for r in latency_rows]
totals_sorted = sorted(totals)
n = len(totals_sorted)
if n:
    median_ms = totals_sorted[n // 2]
    p95_ms    = totals_sorted[min(int(n * 0.95), n - 1)]
else:                                   # --sample5: chưa đủ truy vấn sau 5 truy vấn làm nóng
    median_ms = p95_ms = float("nan")
print(f"\nLatency (không tính warmup 5 truy vấn đầu):")
print(f"  Trung vị: {median_ms:.1f} ms")
print(f"  P95:      {p95_ms:.1f} ms")
print(f"  Lưu: {lat_path}")

log_lines = [f"\n## V8 — chạy dual-encoder và ViMFR ({datetime.now():%Y-%m-%d %H:%M})" + (" [SAMPLE 5 – không dùng]" if ARGS.sample5 else ""),
             f"- thiết bị: {device}" + (f", GPU {torch.cuda.get_device_name(0)}, VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else " (KHÔNG có GPU: độ trễ là của CPU)"),
             f"- torch {torch.__version__}; chỉ mục Test {index.ntotal}/{len(test_products)} sản phẩm (thiếu ảnh {n_missing}); fp16={USE_FP16}",
             f"- {NQ} truy vấn, top_k={TOP_K}; thứ tự giữ nguyên sau rerank: {n_same_order}/{NQ}",
             f"- độ trễ (bỏ {WARMUP_N} truy vấn đầu; gồm tách từ + mã hóa + FAISS + rerank): trung vị {median_ms:.1f} ms, P95 {p95_ms:.1f} ms"]
with open(LOG_DIR / "log_A.md", "a", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")

print("\nV8 hoàn thành!")
