# src/training/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv
from src.models.model import ViMFRModel
from src.models.loss import InfoNCELoss
from config import (
    BATCH_SIZE_CPU, NUM_EPOCHS, WARMUP_EPOCHS,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
    GRAD_CLIP_NORM, CHECKPOINT_BEST, CHECKPOINT_LATEST,
    CKPT_MODEL_KEY, CKPT_LOGIT_SCALE_KEY, CKPT_EPOCH_KEY, CKPT_VAL_LOSS_KEY
)


# ------------------------------------------------------------------
# Dummy DataLoader — thay bằng DataLoader thật từ Người B sau này
# ------------------------------------------------------------------
def get_dummy_dataloader(batch_size=4, num_samples=32):
    pixel_values   = torch.randn(num_samples, 3, 224, 224)
    input_ids      = torch.randint(0, 1000, (num_samples, 128))
    attention_mask = torch.ones(num_samples, 128, dtype=torch.long)
    dataset = TensorDataset(pixel_values, input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

    # Dummy dataloader — đổi thành real dataloader sau
    train_loader = get_dummy_dataloader(batch_size=BATCH_SIZE_CPU)
    val_loader   = get_dummy_dataloader(batch_size=BATCH_SIZE_CPU, num_samples=16)

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
        for pixel_values, input_ids, attention_mask in train_loader:
            pixel_values   = pixel_values.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

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
            for pixel_values, input_ids, attention_mask in val_loader:
                pixel_values   = pixel_values.to(device)
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)

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