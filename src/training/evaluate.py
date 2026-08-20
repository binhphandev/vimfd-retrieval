import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

import torch
import faiss
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.model import ViMFRModel
from dataset import build_dataloaders
from config import (
    CHECKPOINT_BEST, PROCESSED_DIR, IMAGE_DIR,
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE,
    CKPT_MODEL_KEY, TOP_K
)


def load_model(checkpoint_path: str) -> ViMFRModel:
    model = ViMFRModel()
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[CKPT_MODEL_KEY])
    model.eval()
    return model


def load_faiss(index_path: str, mapping_path: str):
    index   = faiss.read_index(index_path)
    with open(mapping_path) as f:
        mapping = json.load(f)  # {"0": "product_id", ...}
    return index, mapping


def compute_recall_at_k(retrieved: list, relevant: str, k: int) -> float:
    return 1.0 if relevant in retrieved[:k] else 0.0


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(CHECKPOINT_BEST).to(device)

    # Load FAISS index
    index, mapping = load_faiss(FAISS_INDEX_FILE, FAISS_MAPPING_FILE)
    print(f"FAISS index: {index.ntotal} vectors")

    # Load test set
    loaders = build_dataloaders(
        batch_size=32,
        num_workers=0,
        split_dir=Path(PROCESSED_DIR),
        image_dir=Path(IMAGE_DIR),
    )
    test_loader = loaders["test"]

    # --- Evaluate Text→Image ---
    print("\n=== Text → Image ===")
    recall_1 = recall_5 = recall_10 = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            product_ids    = batch["product_id"]

            # Encode text → 256D
            text_embeds = model.encode_text(input_ids, attention_mask)
            text_embeds = text_embeds.cpu().numpy().astype("float32")

            # Search FAISS
            _, indices = index.search(text_embeds, 10)

            for i, pid in enumerate(product_ids):
                retrieved = [mapping[str(idx)] for idx in indices[i] if str(idx) in mapping]
                recall_1  += compute_recall_at_k(retrieved, pid, 1)
                recall_5  += compute_recall_at_k(retrieved, pid, 5)
                recall_10 += compute_recall_at_k(retrieved, pid, 10)
                total += 1

    print(f"Recall@1  : {recall_1  / total:.4f}")
    print(f"Recall@5  : {recall_5  / total:.4f}")
    print(f"Recall@10 : {recall_10 / total:.4f}")


    # ---Evaluate Image -> Image---
    print("\n=== Image -> Image ===")
    recall_1_img = recall_5_img = recall_10_img =0.0
    total_img =0 

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pixel_values = batch["image"].to(device)
            product_ids  = batch["product_id"]

            # Encode image → 256D
            image_embeds = model.encode_image(pixel_values)
            image_embeds = image_embeds.cpu().numpy().astype("float32")

            # Search FAISS
            _, indices = index.search(image_embeds, 10)

            for i, pid in enumerate(product_ids):
                retrieved      = [mapping[str(idx)] for idx in indices[i] if str(idx) in mapping]
                recall_1_img  += compute_recall_at_k(retrieved, pid, 1)
                recall_5_img  += compute_recall_at_k(retrieved, pid, 5)
                recall_10_img += compute_recall_at_k(retrieved, pid, 10)
                total_img += 1

    print(f"Recall@1  : {recall_1_img  / total_img:.4f}")
    print(f"Recall@5  : {recall_5_img  / total_img:.4f}")
    print(f"Recall@10 : {recall_10_img / total_img:.4f}")

    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.txt", "w") as f:
        f.write(f"=== Text → Image ===\n")
        f.write(f"Recall@1  : {recall_1  / total:.4f}\n")
        f.write(f"Recall@5  : {recall_5  / total:.4f}\n")
        f.write(f"Recall@10 : {recall_10 / total:.4f}\n")
        f.write("\n=== Image -> Image ===\n")
        f.write(f"Recall@1  : {recall_1_img  / total_img:.4f}\n")
        f.write(f"Recall@5  : {recall_5_img  / total_img:.4f}\n")
        f.write(f"Recall@10 : {recall_10_img / total_img:.4f}\n")

    print("\nKết quả đã lưu vào results/metrics.txt")


if __name__ == "__main__":
    evaluate()