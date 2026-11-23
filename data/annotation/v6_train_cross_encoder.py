"""
V6 (bản viết lại): Huấn luyện Cross-Encoder (vinai/phobert-base + Linear(768 -> 1))
Đầu vào:
  - data/annotation/cross_encoder/ce_pairs_train.csv, ce_pairs_val.csv   (từ V5)
Đầu ra:
  - checkpoints/cross_encoder_best/   (PhoBERT + tokenizer + linear_head.pt; V8 đọc đúng định dạng này)
  - results/logs/ce_train.md          (log mỗi epoch)

Chạy thật (cần GPU)  : python v6_train_cross_encoder.py
Thử nhanh trên CPU   : python v6_train_cross_encoder.py --smoke      (~200 cặp, 1 epoch; ghi vào thư mục *_smoke, không đè kết quả thật)
Giảm dữ liệu (rủi ro "11:20 chưa kịp" trong kế hoạch): --train-frac 0.5   (giữ ngẫu nhiên 50% truy vấn Train, seed 42)
Tùy chọn khác        : --max-epochs N  --time-limit-hours H  --allow-cpu

Khác bản cũ (ghi vào mục 12 / log nếu cần):
  1. Sửa lỗi Windows "An attempt has been made to start a new process...": toàn bộ code nằm trong main() và DataLoader
     không dùng worker (dữ liệu được tách token sẵn một lần, nên không cần).
  2. Lịch learning rate: khởi động tuyến tính 10% rồi COSINE (kế hoạch 5.4). Bản cũ dùng giảm tuyến tính.
  3. Tách token một lần ở đầu và pad theo batch (dynamic padding) thay vì pad cứng 256: nhanh hơn rõ rệt.
  4. Trần thời gian tính từ lúc khởi chạy; kiểm tra cả trong epoch và dừng sớm nếu không đủ thời gian cho epoch kế tiếp.
  5. Chặn chạy trên CPU (sẽ mất nhiều chục giờ); dùng --smoke để thử luồng, hoặc --allow-cpu nếu thật sự muốn.
  6. Thêm val_top1: tỉ lệ truy vấn mà cặp dương có điểm cao nhất trong nhóm (1 dương + 4 âm). Chỉ để theo dõi.
     Lưu ý val_acc có mốc 0,80 nếu mô hình đoán toàn "âm" (tỉ lệ 1:4).
  7. Bỏ token_type_ids (PhoBERT không dùng); dùng API autocast/GradScaler mới, tự lùi về API cũ nếu torch cũ.
"""

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_CSV = ROOT / "data/annotation/cross_encoder/ce_pairs_train.csv"
VAL_CSV   = ROOT / "data/annotation/cross_encoder/ce_pairs_val.csv"
LOG_DIR   = ROOT / "results/logs"

# ── Siêu tham số (theo kế hoạch 5.4) ─────────────────────────────
MODEL_NAME   = "vinai/phobert-base"
MAX_LEN      = 256
BATCH_SIZE   = 32
MAX_EPOCHS   = 10
PATIENCE     = 3            # early stopping theo val loss
LR_PHOBERT   = 2e-5
LR_LINEAR    = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10         # khởi động tuyến tính 10% số bước, sau đó cosine
MAX_NORM     = 1.0
TIME_LIMIT_H = 2.25         # 2 giờ 15 phút kể từ lúc khởi chạy


# ══════════════════════════════════════════════════════════════════
# Dữ liệu
# ══════════════════════════════════════════════════════════════════
def load_pairs(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [{"pq_id": r["pq_id"], "query": r["query_text"], "doc": r["doc_text"], "label": float(r["label"])}
                for r in csv.DictReader(f)]


def subsample_queries(pairs, frac, seed=SEED, max_queries=None):
    """Giữ ngẫu nhiên một phần TRUY VẤN (kèm đủ cặp dương/âm của chúng), seed cố định."""
    pq_ids = sorted({p["pq_id"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(pq_ids)
    k = int(len(pq_ids) * frac)
    if max_queries is not None:
        k = min(k, max_queries)
    keep = set(pq_ids[:max(k, 1)])
    return [p for p in pairs if p["pq_id"] in keep]


def tokenize_pairs(pairs, tokenizer, max_len, log, name):
    """Tách token một lần cho cả tập; chưa pad (pad theo batch trong collate)."""
    ids = []
    for i in range(0, len(pairs), 2000):
        chunk = pairs[i:i + 2000]
        enc = tokenizer([p["query"] for p in chunk], [p["doc"] for p in chunk],
                        truncation=True, max_length=max_len)
        ids.extend(enc["input_ids"])
        log(f"  tách token {name}: {min(i + 2000, len(pairs))}/{len(pairs)}")
    return ids


class PairDataset(Dataset):
    def __init__(self, pairs, input_ids):
        self.ids = input_ids
        self.labels = [p["label"] for p in pairs]
        self.pq_ids = [p["pq_id"] for p in pairs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :len(ids)] = 1
        labels = torch.tensor([l for _, l in batch], dtype=torch.float)
        return input_ids, attn, labels
    return collate


# ══════════════════════════════════════════════════════════════════
# Mô hình: PhoBERT + Linear(768 -> 1)
# ══════════════════════════════════════════════════════════════════
class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]        # vector [CLS]
        return self.linear(cls_vec).squeeze(-1)         # logit; BCEWithLogitsLoss tự áp Sigmoid


def amp_autocast(use_amp):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)


def make_scaler(use_amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)      # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def evaluate(model, loader, dataset, criterion, device, use_amp):
    model.eval()
    total_loss, logits_all, labels_all = 0.0, [], []
    for input_ids, attn, labels in loader:
        input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
        with amp_autocast(use_amp):
            logits = model(input_ids, attn)
        loss = criterion(logits.float(), labels)
        total_loss += loss.item() * len(labels)
        logits_all.extend(logits.float().cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(labels_all)

    acc = sum((1 if x > 0 else 0) == int(l) for x, l in zip(logits_all, labels_all)) / len(labels_all)  # ngưỡng 0,5
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels_all, logits_all)
    except Exception:
        auc = float("nan")

    # val_top1: trong mỗi truy vấn, cặp dương có điểm cao hơn mọi cặp âm không
    groups = defaultdict(lambda: {"pos": None, "neg": []})
    for pq, s, l in zip(dataset.pq_ids, logits_all, labels_all):
        if l == 1:
            groups[pq]["pos"] = s
        else:
            groups[pq]["neg"].append(s)
    ok = [g["pos"] > max(g["neg"]) for g in groups.values() if g["pos"] is not None and g["neg"]]
    top1 = sum(ok) / len(ok) if ok else float("nan")
    return avg_loss, acc, auc, top1


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Thử luồng nhanh (200 cặp, 1 epoch), cho phép CPU")
    ap.add_argument("--allow-cpu", action="store_true", help="Cho phép huấn luyện thật trên CPU (rất chậm)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="Tỉ lệ truy vấn Train giữ lại (seed 42)")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--time-limit-hours", type=float, default=TIME_LIMIT_H)
    ap.add_argument("--model-name", default=MODEL_NAME)
    args = ap.parse_args()

    t_launch = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    suffix = "_smoke" if args.smoke else ""
    ckpt_dir = ROOT / f"checkpoints/cross_encoder_best{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ce_train{suffix}.md"

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Thiết bị ──
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if not cuda and not (args.smoke or args.allow_cpu):
        print("\n[DỪNG] Không phát hiện GPU (torch.cuda.is_available() = False).\n"
              "Huấn luyện PhoBERT 25.030 cặp x 10 epoch trên CPU sẽ mất nhiều chục giờ, không kịp trần 2 giờ 15.\n"
              "  - Kiểm tra: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"\n"
              "  - Có card NVIDIA nhưng ra False: thường là cài bản torch chỉ chạy CPU -> cài lại bản có CUDA.\n"
              "  - Không có GPU: chạy trên Google Colab / Kaggle (T4 miễn phí), đưa 2 file ce_pairs_*.csv lên.\n"
              "  - Chỉ muốn thử luồng: python v6_train_cross_encoder.py --smoke")
        sys.exit(1)
    use_amp = cuda
    max_len = 128 if args.smoke else MAX_LEN
    max_epochs = 1 if args.smoke else args.max_epochs
    limit = args.time_limit_hours * 3600

    log("=== Bắt đầu V6" + (" (SMOKE TEST)" if args.smoke else "") + " ===")
    log(f"Model: {args.model_name}, MAX_LEN={max_len}, BATCH={BATCH_SIZE}")
    log(f"LR_phobert={LR_PHOBERT}, LR_linear={LR_LINEAR}, WD={WEIGHT_DECAY}, lịch=warmup {WARMUP_RATIO:.0%} + cosine")
    log(f"MAX_EPOCHS={max_epochs}, PATIENCE={PATIENCE}, trần thời gian={args.time_limit_hours}h")
    log(f"FP16={use_amp}, seed={SEED}, device={device}")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {torch.cuda.get_device_name(0)}, VRAM {props.total_memory / 1e9:.1f} GB")

    # ── Dữ liệu ──
    train_pairs, val_pairs = load_pairs(TRAIN_CSV), load_pairs(VAL_CSV)
    if args.smoke:
        train_pairs = subsample_queries(train_pairs, 1.0, max_queries=40)
        val_pairs = subsample_queries(val_pairs, 1.0, max_queries=20)
    elif args.train_frac < 1.0:
        n0 = len({p["pq_id"] for p in train_pairs})
        train_pairs = subsample_queries(train_pairs, args.train_frac)
        log(f"Giảm dữ liệu Train: giữ {len({p['pq_id'] for p in train_pairs})}/{n0} truy vấn (seed {SEED})")
    log(f"Train: {len(train_pairs)} cặp, Val: {len(val_pairs)} cặp")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    log("Tách token (một lần)...")
    train_ds = PairDataset(train_pairs, tokenize_pairs(train_pairs, tokenizer, max_len, log, "train"))
    val_ds = PairDataset(val_pairs, tokenize_pairs(val_pairs, tokenizer, max_len, log, "val"))

    batch, accum = BATCH_SIZE, 1
    if cuda and props.total_memory / 1e9 < 10:
        batch, accum = 16, 2
        log(f"VRAM thấp ({props.total_memory / 1e9:.0f}GB) → batch=16, accumulate=2")

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=cuda)
    val_loader = DataLoader(val_ds, batch_size=batch * 2, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=cuda)

    # ── Mô hình, tối ưu, lịch learning rate ──
    model = CrossEncoder(args.model_name).to(device)
    optimizer = AdamW([{"params": model.bert.parameters(), "lr": LR_PHOBERT},
                       {"params": model.linear.parameters(), "lr": LR_LINEAR}], weight_decay=WEIGHT_DECAY)
    total_steps = max((len(train_loader) // accum) * max_epochs, 1)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = make_scaler(use_amp)
    log(f"Total steps={total_steps}, warmup={warmup_steps}, steps/epoch={len(train_loader) // accum}")

    # ── Vòng lặp huấn luyện ──
    best_val_loss, best_epoch, patience_count = float("inf"), 0, 0
    stop_reason, last_epoch_time = "max epochs", 0.0

    for epoch in range(1, max_epochs + 1):
        elapsed = time.time() - t_launch
        if epoch > 1 and elapsed + last_epoch_time > limit:
            stop_reason = f"trần thời gian ({args.time_limit_hours}h): không đủ thời gian cho epoch {epoch}"
            log(f"⚠️  {stop_reason}. Dừng.")
            break

        model.train()
        t0 = time.time()
        running, seen, timed_out = 0.0, 0, False
        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, attn, labels) in enumerate(train_loader):
            if time.time() - t_launch >= limit:
                timed_out = True
                break
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            with amp_autocast(use_amp):
                logits = model(input_ids, attn)
            loss = criterion(logits.float(), labels) / accum
            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * accum * len(labels)
            seen += len(labels)
            if (step + 1) % 100 == 0:
                log(f"  epoch {epoch} step {step + 1}/{len(train_loader)} loss={running / seen:.4f} "
                    f"({(time.time() - t_launch) / 60:.1f} phút từ lúc khởi chạy)")

        if timed_out:
            stop_reason = f"trần thời gian ({args.time_limit_hours}h): dừng giữa epoch {epoch} (epoch dở không đánh giá)"
            log(f"⚠️  {stop_reason}.")
            break

        train_loss = running / max(seen, 1)
        val_loss, val_acc, val_auc, val_top1 = evaluate(model, val_loader, val_ds, criterion, device, use_amp)
        last_epoch_time = time.time() - t0
        log(f"Epoch {epoch}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f} | val_top1={val_top1:.4f} | time={last_epoch_time:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_count = val_loss, epoch, 0
            model.bert.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save({"linear_state_dict": model.linear.state_dict(), "best_epoch": epoch, "val_loss": val_loss,
                        "config": {"model_name": args.model_name, "max_len": max_len}}, ckpt_dir / "linear_head.pt")
            log(f"  → Lưu checkpoint tốt nhất (epoch {epoch}, val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            log(f"  → Không cải thiện ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                stop_reason = f"early stopping (patience={PATIENCE})"
                log(f"⏹  Early stopping tại epoch {epoch}.")
                break

    log("\n=== Kết thúc V6 ===")
    if best_epoch == 0:
        log("CHƯA có checkpoint nào (không hoàn thành được epoch đầu tiên). Giảm dữ liệu: --train-frac 0.5")
    else:
        log(f"Epoch tốt nhất: {best_epoch}, val_loss={best_val_loss:.4f}")
    log(f"Lý do dừng: {stop_reason}")
    log(f"Tổng thời gian: {(time.time() - t_launch) / 60:.1f} phút")
    log(f"Checkpoint lưu tại: {ckpt_dir}")
    log("Lưu ý: val_auc/val_top1 trên truy vấn giả thường rất cao; chất lượng thật chỉ đo được trên 200 truy vấn kiểm thử.")
    print("\nV6 hoàn thành!")


if __name__ == "__main__":
    main()