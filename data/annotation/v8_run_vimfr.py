"""
V8: Chạy dual-encoder và ViMFR trên 200 truy vấn kiểm thử
Đầu vào:
  - data/annotation/benchmark/queries_200.csv
  - data/splits/test.jsonl
  - checkpoints/best_18112026.pt        (dual-encoder B1)
  - checkpoints/cross_encoder_best/     (Cross-Encoder V6)
Đầu ra:
  - results/runs/dual_encoder.run
  - results/runs/vimfr.run
  - results/logs/latency.csv
Chạy: python v8_run_vimfr.py
"""

import csv
import json
import time
from pathlib import Path

import torch
import faiss
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

SEED = 42
ROOT = Path(".")

QUERIES_CSV  = ROOT / "data/annotation/benchmark/queries_200.csv"
TEST_PATH    = ROOT / "data/splits/test.jsonl"
DE_CKPT      = ROOT / "checkpoints/best_18112026.pt"
CE_CKPT_DIR  = ROOT / "checkpoints/cross_encoder_best"
RUNS_DIR     = ROOT / "results/runs"
LOG_DIR      = ROOT / "results/logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load queries ──────────────────────────────────────────────────
queries = []
with open(QUERIES_CSV, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        queries.append({"query_id": row["query_id"], "query_text": row["query_text"]})
print(f"Số truy vấn: {len(queries)}")

# ── Load test products ────────────────────────────────────────────
test_products = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            test_products.append(json.loads(line))
test_pids = [p["id"] for p in test_products]
print(f"Test corpus: {len(test_products)} sản phẩm")

import sys
sys.path.insert(0, str(ROOT))
import config

def get_text(product):
    field = getattr(config, "MODEL_TEXT_FIELD", "text_seg")
    return product.get(field) or product.get("text") or product.get("name", "")

def get_featured_image_path(product):
    images = sorted(product.get("images") or [], key=lambda x: x.get("position", 9999))
    return images[0].get("local_path") if images else None

# ── Load dual-encoder ─────────────────────────────────────────────
print(f"\nLoad dual-encoder: {DE_CKPT}")
from your_model_module import DualEncoder   # ← SỬA tên module
de_ckpt = torch.load(DE_CKPT, map_location=device)
dual_encoder = DualEncoder(...)   # ← SỬA tham số
dual_encoder.load_state_dict(de_ckpt.get("model_state_dict", de_ckpt))
dual_encoder.to(device).eval()

de_tokenizer = AutoTokenizer.from_pretrained(
    getattr(config, "TEXT_MODEL_NAME", "vinai/phobert-base"))

DIM        = 256
IMG_SIZE   = 224
BATCH_SIZE = 64
USE_FP16   = torch.cuda.is_available()

img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def encode_texts_de(texts):
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = de_tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            v = dual_encoder.encode_text(**enc)   # ← SỬA nếu khác
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.cpu().float().numpy())
    return np.vstack(vecs).astype("float32")

@torch.no_grad()
def encode_images_de(products):
    vecs = []
    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i+BATCH_SIZE]
        imgs = []
        for p in batch:
            path = get_featured_image_path(p)
            if path and Path(path).exists():
                img = Image.open(path).convert("RGB")
            else:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            imgs.append(img_transform(img))
        tensor = torch.stack(imgs).to(device)
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            v = dual_encoder.encode_image(tensor)   # ← SỬA nếu khác
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.cpu().float().numpy())
    return np.vstack(vecs).astype("float32")

# ── Build index Test ──────────────────────────────────────────────
print("\nMã hóa ảnh sản phẩm Test → FAISS index...")
image_vecs = encode_images_de(test_products)   # (1339, DIM)
index = faiss.IndexFlatIP(DIM)
index.add(image_vecs)
print(f"Index: {index.ntotal} vectors")

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
    with torch.cuda.amp.autocast(enabled=USE_FP16):
        scores = ce_model(**enc)
    return scores.cpu().float().tolist()

# ── Chạy retrieval + rerank, đo latency ──────────────────────────
print("\nMã hóa 200 truy vấn...")
query_texts = [q["query_text"] for q in queries]
query_vecs  = encode_texts_de(query_texts)   # (200, DIM)

print("Chạy FAISS + rerank...")
latency_rows  = []
de_run_rows   = []
vimfr_run_rows = []

WARMUP_N = 5   # 5 truy vấn đầu làm nóng, không tính latency

for i, (query, q_vec) in enumerate(zip(queries, query_vecs)):
    qid   = query["query_id"]
    qtext = query["query_text"]
    is_warmup = (i < WARMUP_N)

    # --- FAISS search ---
    t_faiss_start = time.perf_counter()
    scores_faiss, indices = index.search(q_vec.reshape(1, -1), TOP_K)
    t_faiss = time.perf_counter() - t_faiss_start

    cand_pids     = [test_pids[j] for j in indices[0] if j >= 0]
    cand_scores   = [float(scores_faiss[0][k]) for k, j in enumerate(indices[0]) if j >= 0]
    cand_products = [test_products[j] for j in indices[0] if j >= 0]

    # --- Dual-encoder run ---
    for rank, (pid, score) in enumerate(zip(cand_pids, cand_scores), start=1):
        de_run_rows.append(f"{qid} Q0 {pid} {rank} {score:.6f} dual_encoder")

    # --- Cross-Encoder rerank ---
    t_ce_start = time.perf_counter()
    ce_scores  = rerank_with_ce(qtext, cand_products)
    t_ce = time.perf_counter() - t_ce_start

    # Sắp lại theo ce_score giảm dần
    reranked = sorted(zip(cand_pids, ce_scores), key=lambda x: -x[1])
    for rank, (pid, score) in enumerate(reranked, start=1):
        vimfr_run_rows.append(f"{qid} Q0 {pid} {rank} {score:.6f} vimfr")

    if not is_warmup:
        latency_rows.append({
            "query_id": qid,
            "t_encode_ms":  0,         # mã hóa truy vấn đã batch trước
            "t_faiss_ms":   t_faiss * 1000,
            "t_rerank_ms":  t_ce * 1000,
            "total_ms":     (t_faiss + t_ce) * 1000,
        })

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/200 truy vấn xong")

# ── Kiểm tra ─────────────────────────────────────────────────────
assert len(de_run_rows) == 200 * TOP_K, f"DE run thiếu dòng: {len(de_run_rows)}"
assert len(vimfr_run_rows) == 200 * TOP_K, f"ViMFR run thiếu dòng: {len(vimfr_run_rows)}"

# Kiểm tra tập product_id giống nhau giữa DE và ViMFR
for i, q in enumerate(queries):
    qid = q["query_id"]
    de_pids    = set(r.split()[2] for r in de_run_rows[i*50:(i+1)*50])
    vimfr_pids = set(r.split()[2] for r in vimfr_run_rows[i*50:(i+1)*50])
    assert de_pids == vimfr_pids, f"Tập product_id khác nhau tại {qid}!"

# Cảnh báo nếu thứ tự giống nhau > 90%
n_same_order = 0
for i in range(200):
    de_order    = [r.split()[2] for r in de_run_rows[i*50:(i+1)*50]]
    vimfr_order = [r.split()[2] for r in vimfr_run_rows[i*50:(i+1)*50]]
    if de_order == vimfr_order:
        n_same_order += 1
if n_same_order > 180:   # > 90%
    print(f"⚠️  CẢNH BÁO: {n_same_order}/200 truy vấn có thứ tự giống hệt DE! "
          "Kiểm tra Cross-Encoder checkpoint.")
else:
    print(f"✅ Rerank hợp lệ: {n_same_order}/200 truy vấn giữ nguyên thứ tự")

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
median_ms = totals_sorted[n // 2]
p95_ms    = totals_sorted[int(n * 0.95)]
print(f"\nLatency (không tính warmup 5 truy vấn đầu):")
print(f"  Trung vị: {median_ms:.1f} ms")
print(f"  P95:      {p95_ms:.1f} ms")
print(f"  Lưu: {lat_path}")

print("\nV8 hoàn thành!")
