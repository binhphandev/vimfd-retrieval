"""
V6: Huấn luyện Cross-Encoder (vinai/phobert-base)
Đầu vào:
  - data/annotation/cross_encoder/ce_pairs_train.csv  (từ V5)
  - data/annotation/cross_encoder/ce_pairs_val.csv    (từ V5)
Đầu ra:
  - checkpoints/cross_encoder_best/   (checkpoint + tokenizer)
  - results/logs/ce_train.md          (log mỗi epoch)
Chạy: python v6_train_cross_encoder.py
Yêu cầu: torch, transformers, scikit-learn
"""

import csv
import json
import math
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(".")
TRAIN_CSV = ROOT / "data/annotation/cross_encoder/ce_pairs_train.csv"
VAL_CSV   = ROOT / "data/annotation/cross_encoder/ce_pairs_val.csv"
CKPT_DIR  = ROOT / "checkpoints/cross_encoder_best"
LOG_DIR   = ROOT / "results/logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Siêu tham số (theo kế hoạch) ─────────────────────────────────
MODEL_NAME   = "vinai/phobert-base"
MAX_LEN      = 256
BATCH_SIZE   = 32
MAX_EPOCHS   = 10
PATIENCE     = 3          # early stopping theo val loss
LR_PHOBERT   = 2e-5       # learning rate cho PhoBERT
LR_LINEAR    = 1e-4       # learning rate cho lớp Linear đầu ra
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10       # 10% số bước warmup tuyến tính
MAX_NORM     = 1.0        # gradient clipping
TIME_LIMIT   = 2.25 * 3600  # 2 giờ 15 phút (giây)
USE_FP16     = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Dataset ───────────────────────────────────────────────────────
class CEPairDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.samples = []
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "query":   row["query_text"],
                    "doc":     row["doc_text"],
                    "label":   float(row["label"]),
                })
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["query"],
            s["doc"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(s["label"], dtype=torch.float),
        }

# ── Model: PhoBERT + Linear(768→1) ───────────────────────────────
class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert   = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]   # vector [CLS]
        logits  = self.linear(cls_vec).squeeze(-1)     # (batch,)
        return logits   # BCEWithLogitsLoss tự áp Sigmoid

# ── Log ───────────────────────────────────────────────────────────
log_path = LOG_DIR / "ce_train.md"
def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log(f"=== Bắt đầu V6 ===")
log(f"Model: {MODEL_NAME}, MAX_LEN={MAX_LEN}, BATCH={BATCH_SIZE}")
log(f"LR_phobert={LR_PHOBERT}, LR_linear={LR_LINEAR}, WD={WEIGHT_DECAY}")
log(f"MAX_EPOCHS={MAX_EPOCHS}, PATIENCE={PATIENCE}, WARMUP={WARMUP_RATIO}")
log(f"FP16={USE_FP16}, seed={SEED}")

# ── Load tokenizer và data ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds  = CEPairDataset(TRAIN_CSV, tokenizer, MAX_LEN)
val_ds    = CEPairDataset(VAL_CSV,   tokenizer, MAX_LEN)

# Nếu hết VRAM với batch=32 thì chuyển sang 16 + gradient_accumulation=2
ACTUAL_BATCH    = BATCH_SIZE
ACCUMULATE_GRAD = 1
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_vram_gb < 10:
        ACTUAL_BATCH    = 16
        ACCUMULATE_GRAD = 2
        log(f"VRAM thấp ({total_vram_gb:.0f}GB) → batch=16, accumulate=2")

train_loader = DataLoader(train_ds, batch_size=ACTUAL_BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=ACTUAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)
log(f"Train: {len(train_ds)} cặp, Val: {len(val_ds)} cặp")

# ── Khởi tạo model ───────────────────────────────────────────────
model = CrossEncoder(MODEL_NAME).to(device)

# Hai learning rate khác nhau: PhoBERT vs lớp Linear
optimizer = AdamW(
    [
        {"params": model.bert.parameters(),   "lr": LR_PHOBERT},
        {"params": model.linear.parameters(), "lr": LR_LINEAR},
    ],
    weight_decay=WEIGHT_DECAY,
)

total_steps  = (len(train_loader) // ACCUMULATE_GRAD) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

criterion = nn.BCEWithLogitsLoss()
scaler    = torch.cuda.amp.GradScaler(enabled=USE_FP16)

log(f"Total steps={total_steps}, warmup={warmup_steps}")

# ── Hàm đánh giá val ─────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.extend(logits.cpu().float().tolist())
            all_labels.extend(labels.cpu().float().tolist())
    avg_loss = total_loss / len(loader.dataset)
    # Accuracy và ROC-AUC tại ngưỡng 0.5 (chỉ theo dõi, không dùng để stop)
    preds = [1 if x > 0 else 0 for x in all_logits]   # sigmoid(0) = 0.5
    acc   = sum(p == int(l) for p, l in zip(preds, all_labels)) / len(all_labels)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_logits)
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc

# ── Vòng lặp huấn luyện ──────────────────────────────────────────
best_val_loss   = float("inf")
patience_count  = 0
best_epoch      = 0
start_time      = time.time()
stop_reason     = "max epochs"

for epoch in range(1, MAX_EPOCHS + 1):
    # Kiểm tra trần thời gian
    elapsed = time.time() - start_time
    if elapsed >= TIME_LIMIT:
        stop_reason = f"trần thời gian ({TIME_LIMIT/3600:.2f}h)"
        log(f"⚠️  Hết trần thời gian tại epoch {epoch}. Dừng.")
        break

    model.train()
    t0 = time.time()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=USE_FP16):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels) / ACCUMULATE_GRAD

        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATE_GRAD == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item() * ACCUMULATE_GRAD * len(labels)

    train_loss /= len(train_loader.dataset)
    val_loss, val_acc, val_auc = evaluate(val_loader)
    epoch_time = time.time() - t0

    log(f"Epoch {epoch}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_auc={val_auc:.4f} | "
        f"time={epoch_time:.0f}s")

    # Lưu checkpoint tốt nhất theo val loss
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_epoch     = epoch
        patience_count = 0
        model.bert.save_pretrained(CKPT_DIR)
        tokenizer.save_pretrained(CKPT_DIR)
        # Lưu thêm lớp Linear và config
        torch.save({
            "linear_state_dict": model.linear.state_dict(),
            "best_epoch":        epoch,
            "val_loss":          val_loss,
            "config": {
                "model_name": MODEL_NAME,
                "max_len":    MAX_LEN,
            }
        }, CKPT_DIR / "linear_head.pt")
        log(f"  → Lưu checkpoint tốt nhất (epoch {epoch}, val_loss={val_loss:.4f})")
    else:
        patience_count += 1
        log(f"  → Không cải thiện ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            stop_reason = f"early stopping (patience={PATIENCE})"
            log(f"⏹  Early stopping tại epoch {epoch}.")
            break

log(f"\n=== Kết thúc V6 ===")
log(f"Epoch tốt nhất: {best_epoch}, val_loss={best_val_loss:.4f}")
log(f"Lý do dừng: {stop_reason}")
log(f"Tổng thời gian: {(time.time()-start_time)/60:.1f} phút")
log(f"Checkpoint lưu tại: {CKPT_DIR}")
print("\nV6 hoàn thành!")
