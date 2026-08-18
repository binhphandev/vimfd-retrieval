# src/training/train.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from pathlib import Path
import os
import csv

from src.models.model import ViMFRModel
from src.models.loss import InfoNCELoss
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
from dataset import build_dataloaders
from config import (
    BATCH_SIZE_CPU, NUM_EPOCHS,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
    GRAD_CLIP_NORM, CHECKPOINT_BEST, CHECKPOINT_LATEST,
    CKPT_MODEL_KEY, CKPT_LOGIT_SCALE_KEY, CKPT_EPOCH_KEY, CKPT_VAL_LOSS_KEY,
    PROCESSED_DIR,IMAGE_DIR
)


# ------------------------------------------------------------------
# Tách param groups: backbone lr nhỏ, head lr lớn
# ------------------------------------------------------------------
def get_optimizer(model: ViMFRModel):
    backbone_params = (
        list(model.visual_encoder.parameters()) +
        list(model.text_encoder.parameters())
    )
    head_params = (
        list(model.visual_proj.parameters()) +
        list(model.text_proj.parameters()) +
        [model.logit_scale]
    )
    return AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = ViMFRModel().to(device)
    loss_fn = InfoNCELoss()
    optimizer = get_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    loaders = build_dataloaders(
        batch_size=BATCH_SIZE_CPU,
        num_workers = 0,
        split_dir = Path(PROCESSED_DIR),
        image_dir = Path(IMAGE_DIR),
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    best_val_loss = float("inf")

    # Log loss ra CSV
    log_path = "results/loss_log.csv"
    os.makedirs("results", exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, NUM_EPOCHS + 1):

        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch  in train_loader:
            pixel_values   = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_embeds, text_embeds, logit_scale = model(
                pixel_values, input_ids, attention_mask
            )
            loss = loss_fn(image_embeds, text_embeds, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values   = batch["image"].to(device)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                image_embeds, text_embeds, logit_scale = model(
                    pixel_values, input_ids, attention_mask
                )
                loss = loss_fn(image_embeds, text_embeds, logit_scale)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # --- Log CSV ---
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(train_loss, 4), round(val_loss, 4)])

        # --- Checkpoint latest ---
        torch.save({
            CKPT_MODEL_KEY:       model.state_dict(),
            CKPT_LOGIT_SCALE_KEY: model.logit_scale.data,
            CKPT_EPOCH_KEY:       epoch,
            CKPT_VAL_LOSS_KEY:    val_loss,
        }, CHECKPOINT_LATEST)

        # --- Checkpoint best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                CKPT_MODEL_KEY:       model.state_dict(),
                CKPT_LOGIT_SCALE_KEY: model.logit_scale.data,
                CKPT_EPOCH_KEY:       epoch,
                CKPT_VAL_LOSS_KEY:    val_loss,
            }, CHECKPOINT_BEST)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    train()