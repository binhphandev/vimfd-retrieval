# src/training/train.py
# Hướng đánh giá: Text → Image  và  Image → Image
#   - Gallery  : ảnh đại diện của mọi sản phẩm (val set)
#   - T→I query: text mô tả sản phẩm
#   - I→I query: MỘT ẢNH KHÁC (ảnh phụ) của cùng sản phẩm  (không dùng lại ảnh đại diện,
#                nếu không query sẽ khớp chính nó và R@1 ≈ 100%)

import os
import sys
import csv
import json
import math
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from src.models.model import ViMFRModel
from src.models.loss import InfoNCELoss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
from dataset import build_dataloaders  # noqa: E402

import config as cfg  # noqa: E402
from config import (  # noqa: E402
    BATCH_SIZE, NUM_EPOCHS, WARMUP_EPOCHS,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
    GRAD_CLIP_NORM, LOGIT_SCALE_MAX,
    CHECKPOINT_BEST, CHECKPOINT_LATEST,
    CKPT_MODEL_KEY, CKPT_LOGIT_SCALE_KEY, CKPT_EPOCH_KEY, CKPT_VAL_LOSS_KEY,
    PROCESSED_DIR, IMAGE_DIR,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    IMAGE_LIST_FIELD, IMAGE_PRIMARY_FIELD, IMAGE_POSITION_FIELD,
    PRODUCT_ID_FIELD,
)

# config.py bạn gửi không có IMAGE_PATH_FIELD (build_index.py thì import nó) -> dùng mặc định
IMAGE_PATH_FIELD = getattr(cfg, "IMAGE_PATH_FIELD", "local_path")
NUM_WORKERS = 2  # đặt 0 nếu gặp lỗi DataLoader trên Windows


# ------------------------------------------------------------------
# Dataset ảnh phụ: query cho Image → Image
# ------------------------------------------------------------------
def _pos(img: dict) -> float:
    p = img.get(IMAGE_POSITION_FIELD)
    return p if isinstance(p, (int, float)) else float("inf")


def _pick_alt_image(product: dict):
    """Trả về 1 ảnh KHÔNG phải ảnh đại diện (None nếu sản phẩm chỉ có 1 ảnh).
    Cách chọn ảnh đại diện giống build_index.py: is_featured -> position nhỏ nhất."""
    images = product.get(IMAGE_LIST_FIELD)
    if not isinstance(images, list):
        return None
    imgs = [im for im in images if isinstance(im, dict) and im.get(IMAGE_PATH_FIELD)]
    if len(imgs) < 2:
        return None
    primary = next((im for im in imgs if im.get(IMAGE_PRIMARY_FIELD)), None)
    if primary is None:
        primary = min(imgs, key=_pos)
    alts = sorted((im for im in imgs if im is not primary), key=_pos)
    return alts[0]


def _resolve_path(rel: str):
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    for root in (Path(PROCESSED_DIR), Path(IMAGE_DIR).parent, Path(IMAGE_DIR)):
        if (root / p).exists():
            return root / p
    return None


class AltImageDataset(Dataset):
    """Mỗi phần tử: 1 ảnh phụ của 1 sản phẩm, trả {"image", "product_id"}
    (cùng format với dataset chính nên dùng lại được trong evaluate.py)."""

    def __init__(self, json_path):
        with open(json_path, encoding="utf-8") as f:
            products = json.load(f)
        # QUAN TRỌNG: transform này phải GIỐNG transform ảnh của val/test trong dataset.py
        self.transform = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
        self.samples = []
        for p in products:
            alt = _pick_alt_image(p)
            if alt is None:
                continue
            path = _resolve_path(alt[IMAGE_PATH_FIELD])
            if path is not None:
                self.samples.append((str(p[PRODUCT_ID_FIELD]), path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pid, path = self.samples[i]
        with Image.open(path) as im:
            return {"image": self.transform(im.convert("RGB")), "product_id": pid}


# ------------------------------------------------------------------
# Optimizer / scheduler
# ------------------------------------------------------------------
def get_optimizer(model: ViMFRModel) -> AdamW:
    backbone_params = [
        p for p in list(model.visual_encoder.parameters())
        + list(model.text_encoder.parameters())
        if p.requires_grad
    ]
    head_params = (
        list(model.visual_proj.parameters())
        + list(model.text_proj.parameters())
    )
    return AdamW([
        {"params": backbone_params,     "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params,         "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
        {"params": [model.logit_scale], "lr": LR_HEAD,     "weight_decay": 0.0},
    ])


def get_scheduler(optimizer, steps_per_epoch: int) -> LambdaLR:
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = min(WARMUP_EPOCHS * steps_per_epoch, total_steps // 2)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Validation: loss + Recall@K cho T→I và I→I
# ------------------------------------------------------------------
def _to_str_list(pids):
    if torch.is_tensor(pids):
        pids = pids.tolist()
    return [str(x) for x in pids]


def _recall(rank: torch.Tensor, ks):
    return {k: (rank < k).float().mean().item() for k in ks}


@torch.no_grad()
def validate(model, loader, alt_loader, loss_fn, device, use_amp, ks=(1, 5, 10)):
    model.eval()
    val_loss, n_batches = 0.0, 0
    all_img, all_txt, gallery_ids = [], [], []

    for batch in loader:
        pixel_values   = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            img, txt, logit_scale = model(pixel_values, input_ids, attention_mask)
        img, txt = img.float(), txt.float()

        val_loss += loss_fn(img, txt, logit_scale).item()
        n_batches += 1
        all_img.append(img.cpu())
        all_txt.append(txt.cpu())
        gallery_ids.extend(_to_str_list(batch["product_id"]))

    gallery = torch.cat(all_img)   # [N, 256] ảnh đại diện = gallery
    txt = torch.cat(all_txt)

    # --- Text → Image ---
    sim = txt @ gallery.T
    rank = (sim > sim.diag().unsqueeze(1)).sum(dim=1)
    t2i = _recall(rank, ks)

    # --- Image → Image (query = ảnh phụ của cùng sản phẩm) ---
    i2i = None
    if alt_loader is not None:
        q_embs, q_ids = [], []
        for batch in alt_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                e = model.encode_image(batch["image"].to(device))
            q_embs.append(e.float().cpu())
            q_ids.extend(_to_str_list(batch["product_id"]))
        q = torch.cat(q_embs)

        id2idx = {pid: i for i, pid in enumerate(gallery_ids)}
        keep = [j for j, pid in enumerate(q_ids) if pid in id2idx]
        if keep:
            gt = torch.tensor([id2idx[q_ids[j]] for j in keep]).unsqueeze(1)
            sim_q = q[keep] @ gallery.T
            rank = (sim_q > sim_q.gather(1, gt)).sum(dim=1)
            i2i = _recall(rank, ks)

    return val_loss / max(1, n_batches), t2i, i2i


def save_ckpt(path, model, epoch, val_loss, extra=None):
    ckpt = {
        CKPT_MODEL_KEY:       model.state_dict(),
        CKPT_LOGIT_SCALE_KEY: model.logit_scale.data,
        CKPT_EPOCH_KEY:       epoch,
        CKPT_VAL_LOSS_KEY:    val_loss,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp} | batch_size: {BATCH_SIZE}")

    model = ViMFRModel().to(device)
    loss_fn = InfoNCELoss()

    loaders = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        split_dir=Path(PROCESSED_DIR),
        image_dir=Path(IMAGE_DIR),
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # Query ảnh phụ cho I→I
    alt_ds = AltImageDataset(Path(PROCESSED_DIR) / "val.json")
    print(f"I→I: {len(alt_ds)} sản phẩm val có ảnh phụ làm query")
    alt_loader = None
    if len(alt_ds) > 0:
        alt_loader = DataLoader(alt_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)
    else:
        print("  ⚠ Không tìm được ảnh phụ nào -> bỏ qua I→I. "
              "Kiểm tra IMAGE_PATH_FIELD / đường dẫn ảnh trong config.")

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_log_scale = math.log(LOGIT_SCALE_MAX)
    print(f"Sanity: loss ban đầu nên ≈ log({BATCH_SIZE}) = {math.log(BATCH_SIZE):.3f}")

    os.makedirs("results", exist_ok=True)
    log_path = "results/loss_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "logit_scale",
            "t2i_R1", "t2i_R5", "t2i_R10", "i2i_R1", "i2i_R5", "i2i_R10",
        ])

    best_score = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss, n_batches = 0.0, 0

        for batch in train_loader:
            pixel_values   = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if pixel_values.size(0) < 2:
                continue

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                image_embeds, text_embeds, logit_scale = model(
                    pixel_values, input_ids, attention_mask
                )
            loss = loss_fn(image_embeds.float(), text_embeds.float(), logit_scale)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                model.logit_scale.clamp_(0, max_log_scale)

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        # --- Validation ---
        val_loss, t2i, i2i = validate(model, val_loader, alt_loader, loss_fn, device, use_amp)
        scale_val = model.logit_scale.exp().item()

        msg = (f"Epoch {epoch:03d}/{NUM_EPOCHS} | train: {train_loss:.4f} | "
               f"val: {val_loss:.4f} | scale: {scale_val:.1f}\n"
               f"   T→I R@1/5/10: {t2i[1]:.3f}/{t2i[5]:.3f}/{t2i[10]:.3f}")
        if i2i is not None:
            msg += f" | I→I R@1/5/10: {i2i[1]:.3f}/{i2i[5]:.3f}/{i2i[10]:.3f}"
        print(msg)

        i2i_row = i2i if i2i is not None else {1: 0.0, 5: 0.0, 10: 0.0}
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 4), round(val_loss, 4), round(scale_val, 2),
                round(t2i[1], 4), round(t2i[5], 4), round(t2i[10], 4),
                round(i2i_row[1], 4), round(i2i_row[5], 4), round(i2i_row[10], 4),
            ])

        save_ckpt(CHECKPOINT_LATEST, model, epoch, val_loss)

        # Best theo Recall@10 trung bình của 2 hướng
        score = (t2i[10] + i2i[10]) / 2 if i2i is not None else t2i[10]
        if score > best_score:
            best_score = score
            save_ckpt(CHECKPOINT_BEST, model, epoch, val_loss, extra={"val_recall10": score})
            print(f"  ✓ Saved best checkpoint (val R@10 avg = {score:.4f})")


if __name__ == "__main__":
    train()