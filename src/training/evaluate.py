# src/evaluation/evaluate.py
# Đánh giá 2 hướng:
#   Text  → Image : query = text mô tả (test set)         , gallery = image index
#   Image → Image : query = ẢNH PHỤ của sản phẩm (test set), gallery = image index
# Mỗi hướng đo trên 2 kiểu gallery:
#   (a) full      : toàn bộ sản phẩm trong index (train+val+test)
#   (b) test-only : chỉ các sản phẩm của test set  (cách so sánh chuẩn trong paper)

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent)) 

import json
import math
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import ViMFRModel, load_state_dict_compat
from src.training.train import AltImageDataset
from dataset import build_dataloaders
from config import (
    CHECKPOINT_BEST, PROCESSED_DIR, IMAGE_DIR,
    FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE,
    CKPT_MODEL_KEY, PRODUCT_ID_FIELD,
)

    
# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load_model(checkpoint_path: str) -> ViMFRModel:
    model = ViMFRModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_state_dict_compat(model, ckpt[CKPT_MODEL_KEY])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return model


def load_faiss(index_path: str, mapping_path: str):
    index = faiss.read_index(index_path)
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)  # {"0": "product_id", ...}
    print(f"Loaded index: {index_path} ({index.ntotal} vectors)")
    return index, mapping


def make_subset_index(index, mapping: dict, keep_ids: set):
    """Tạo index con chỉ chứa các sản phẩm trong keep_ids (không cần encode lại).
    Chỉ dùng được với IndexFlat* (reconstruct_n)."""
    n = index.ntotal
    vecs = index.reconstruct_n(0, n)
    pos = [i for i in range(n) if mapping[str(i)] in keep_ids]
    sub = faiss.IndexFlatIP(vecs.shape[1])
    sub.add(np.ascontiguousarray(vecs[pos]))
    sub_mapping = {str(j): mapping[str(i)] for j, i in enumerate(pos)}
    return sub, sub_mapping


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def recall_at_k(retrieved: list, relevant: str, k: int) -> float:
    return 1.0 if relevant in retrieved[:k] else 0.0


def ndcg_at_k(retrieved: list, relevant: str, k: int) -> float:
    # 1 đáp án đúng duy nhất -> NDCG = 1/log2(rank+1)
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid == relevant:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def _ids(pids) -> list:
    if torch.is_tensor(pids):
        pids = pids.tolist()
    return [str(x) for x in pids]


def evaluate_direction(direction, search_index, mapping, loader, device,
                       encode_fn, input_keys):
    print(f"\n=== {direction} | gallery = {search_index.ntotal} sản phẩm ===")
    gallery_ids = set(mapping.values())
    r1 = r5 = r10 = ndcg10 = total = skipped = 0.0

    with torch.no_grad():
        for batch in tqdm(loader):
            pids = _ids(batch["product_id"])
            inputs = [batch[k].to(device) for k in input_keys]
            embeds = encode_fn(*inputs).cpu().numpy().astype("float32")
            _, indices = search_index.search(embeds, 10)

            for i, pid in enumerate(pids):
                # Sản phẩm không có trong gallery (vd. thiếu ảnh) -> không tính
                if pid not in gallery_ids:
                    skipped += 1
                    continue
                retrieved = [mapping[str(idx)] for idx in indices[i] if str(idx) in mapping]
                r1 += recall_at_k(retrieved, pid, 1)
                r5 += recall_at_k(retrieved, pid, 5)
                r10 += recall_at_k(retrieved, pid, 10)
                ndcg10 += ndcg_at_k(retrieved, pid, 10)
                total += 1

    if skipped:
        print(f"  (bỏ qua {int(skipped)} query vì sản phẩm không có trong gallery)")
    total = max(total, 1)
    results = {
        "Recall@1":  round(r1 / total, 4),
        "Recall@5":  round(r5 / total, 4),
        "Recall@10": round(r10 / total, 4),
        "NDCG@10":   round(ndcg10 / total, 4),
        "n_queries": int(total),
        "gallery":   int(search_index.ntotal),
    }
    print(f"  Queries   : {results['n_queries']}")
    for k in ("Recall@1", "Recall@5", "Recall@10", "NDCG@10"):
        print(f"  {k:<10}: {results[k]:.4f}")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cảnh báo nếu index cũ hơn checkpoint (index không khớp model)
    if os.path.getmtime(CHECKPOINT_BEST) > os.path.getmtime(FAISS_IMAGE_INDEX_FILE):
        print("⚠ CẢNH BÁO: checkpoint MỚI HƠN image index. "
              "Hãy chạy lại build_index.py trước khi evaluate!")

    model = load_model(CHECKPOINT_BEST).to(device)
    image_index, image_mapping = load_faiss(FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE)

    # Query T→I: test loader chính (text)
    loaders = build_dataloaders(
        batch_size=32, num_workers=0,
        split_dir=Path(PROCESSED_DIR), image_dir=Path(IMAGE_DIR),
    )
    test_loader = loaders["test"]

    # Query I→I: ảnh phụ của sản phẩm test
    alt_ds = AltImageDataset(Path(PROCESSED_DIR) / "test.json")
    print(f"I→I: {len(alt_ds)} sản phẩm test có ảnh phụ làm query")
    alt_loader = DataLoader(alt_ds, batch_size=32, shuffle=False, num_workers=0)

    # Gallery test-only
    with open(Path(PROCESSED_DIR) / "test.json", encoding="utf-8") as f:
        test_ids = {str(p[PRODUCT_ID_FIELD]) for p in json.load(f)}
    sub_index, sub_mapping = make_subset_index(image_index, image_mapping, test_ids)

    galleries = {
        "full":      (image_index, image_mapping),
        "test-only": (sub_index, sub_mapping),
    }

    all_results = {}
    for g_name, (g_index, g_mapping) in galleries.items():
        all_results[g_name] = {
            "Text → Image": evaluate_direction(
                "Text → Image", g_index, g_mapping, test_loader, device,
                model.encode_text, ["input_ids", "attention_mask"],
            ),
            "Image → Image": evaluate_direction(
                "Image → Image", g_index, g_mapping, alt_loader, device,
                model.encode_image, ["image"],
            ),
        }

    # ---- Lưu kết quả ----
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.txt", "w", encoding="utf-8") as f:
        for g_name, dirs in all_results.items():
            for d_name, r in dirs.items():
                f.write(f"=== {d_name} | gallery: {g_name} ({r['gallery']}) ===\n")
                f.write(f"Queries   : {r['n_queries']}\n")
                f.write(f"Recall@1  : {r['Recall@1']:.4f}\n")
                f.write(f"Recall@5  : {r['Recall@5']:.4f}\n")
                f.write(f"Recall@10 : {r['Recall@10']:.4f}\n")
                f.write(f"NDCG@10   : {r['NDCG@10']:.4f}\n\n")
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nKết quả đã lưu vào results/metrics.txt và results/metrics.json")


if __name__ == "__main__":
    evaluate()