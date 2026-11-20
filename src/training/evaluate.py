# Đánh giá 2 hướng:
#   Text  → Image : query = text mô tả (test set)         , gallery = image index
#   Image → Image : query = MỌI ảnh phụ của sản phẩm (test set), gallery = image index
# Mỗi hướng đo trên 2 kiểu gallery:
#   (a) full      : toàn bộ sản phẩm trong index (train+val+test)
#   (b) test-only : chỉ các sản phẩm của test set  (cách so sánh chuẩn trong paper)
#
# THAY ĐỔI so với bản cũ (để khớp với AltImageDataset mới trong train.py, mỗi phần tử
# giờ là MỘT ảnh phụ thay vì một sản phẩm):
#   1. Lọc bỏ ảnh đọc lỗi (cờ "ok"=0) — trước đây các ảnh này lọt vào kết quả mà không báo.
#   2. Loại query mà kết quả top-1 trùng hệt chính nó (điểm ≥ IDENTICAL_THR) — tránh
#      Recall@1 bị đội giả tạo khi ảnh phụ thực chất là bản sao ảnh đại diện.
#   3. Báo cả MICRO (mỗi ảnh phụ một phiếu) và MACRO (mỗi sản phẩm một phiếu, không bị
#      lệch bởi sản phẩm có nhiều ảnh phụ hơn). Với Text→Image, mỗi sản phẩm vẫn chỉ có
#      1 query nên micro = macro (dùng để tự kiểm tra code không viết sai).
#   4. Thêm MRR@10 (1/(hạng+1) nếu đáp án lọt top-10, ngược lại 0) bên cạnh Recall/NDCG@10.
#   5. Dòng in "N sản phẩm có ảnh phụ" được thay bằng alt_ds.describe() cho đúng nghĩa
#      (len(alt_ds) giờ đếm ẢNH, không phải SẢN PHẨM).
#
# Phân tích sâu hơn (theo shop, theo thứ tự ảnh phụ, khoảng tin cậy bootstrap) nằm ở


import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import math
from collections import defaultdict
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
    CHECKPOINT_BEST, SPLIT_DIR, IMAGE_DIR, EVAL_BATCH_SIZE,
    FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE,
    CKPT_MODEL_KEY, PRODUCT_ID_FIELD, EVALUATION_SPLIT,
)

try:  # dùng ĐÚNG ngưỡng mà train.py dùng để loại query trùng hệt (nếu bản cũ không có thì dùng mặc định)
    from src.training.train import IDENTICAL_THR
except ImportError:
    IDENTICAL_THR = 0.9995

METRIC_NAMES = ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"]


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
# Tính điểm một query, rồi gộp theo micro/macro
# ------------------------------------------------------------------
def _ids(pids) -> list:
    if torch.is_tensor(pids):
        pids = pids.tolist()
    return [str(x) for x in pids]


def score_query(pid: str, retrieved: list, top1_score: float, identical_thr: float):
    """retrieved: product_id theo thứ tự top-10 (điểm giảm dần), đã lọc khỏi id rỗng.
    Trả về ("identical", None) nếu kết quả đầu trùng hệt chính nó (bị loại khỏi phép đo),
    hoặc ("ok", (hit@1, hit@5, hit@10, reciprocal_rank, ndcg)) — mọi giá trị =0 nếu đáp án
    không lọt top-10 (MRR/NDCG ở đây là bản giới hạn trong top-10, không phải hạng thật)."""
    if retrieved and retrieved[0] == pid and top1_score >= identical_thr:
        return "identical", None
    rank = retrieved.index(pid) if pid in retrieved else None  # 0-based trong top-10
    hit1 = 1.0 if rank == 0 else 0.0
    hit5 = 1.0 if (rank is not None and rank < 5) else 0.0
    hit10 = 1.0 if rank is not None else 0.0
    rr = 1.0 / (rank + 1) if rank is not None else 0.0
    ndcg = (1.0 / math.log2(rank + 2)) if rank is not None else 0.0
    return "ok", (hit1, hit5, hit10, rr, ndcg)


def _mean_rows(rows) -> dict:
    """Trung bình cộng theo từng cột. rows rỗng -> None (không có query nào để tính)."""
    if not rows:
        return None
    cols = list(zip(*rows))
    return {name: float(sum(c) / len(c)) for name, c in zip(METRIC_NAMES, cols)}


def evaluate_direction(direction, search_index, mapping, loader, device,
                       encode_fn, input_keys, require_ok=False, dedupe_identical=False,
                       identical_thr=IDENTICAL_THR):
    """Chấm một hướng (Text→Image hoặc Image→Image) trên một gallery.
    require_ok=True    : bỏ các mẫu có cờ "ok"=0 (ảnh đọc lỗi; chỉ AltImageDataset mới có cờ này).
    dedupe_identical=True: bỏ query mà kết quả top-1 trùng hệt chính nó (xem score_query)."""
    print(f"\n=== {direction} | gallery = {search_index.ntotal} sản phẩm ===")
    gallery_ids = set(mapping.values())
    per_product = defaultdict(list)          # product_id -> [hàng chỉ số của từng query]
    n_not_in_gallery = n_unreadable = n_identical = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            pids = _ids(batch["product_id"])
            ok = batch.get("ok")
            ok = ok.tolist() if torch.is_tensor(ok) else [1] * len(pids)
            inputs = [batch[k].to(device) for k in input_keys]
            embeds = encode_fn(*inputs).cpu().numpy().astype("float32")
            scores, indices = search_index.search(embeds, 10)   # scores: inner product, giảm dần

            for i, pid in enumerate(pids):
                if require_ok and not ok[i]:
                    n_unreadable += 1
                    continue
                if pid not in gallery_ids:
                    n_not_in_gallery += 1
                    continue
                retrieved = [mapping[str(idx)] for idx in indices[i] if str(idx) in mapping]
                status, vals = score_query(pid, retrieved, float(scores[i][0]) if len(scores[i]) else -1.0,
                                           identical_thr if dedupe_identical else 2.0)  # 2.0 = không bao giờ loại
                if status == "identical":
                    n_identical += 1
                    continue
                per_product[pid].append(vals)

    all_rows = [row for rows in per_product.values() for row in rows]      # mỗi query 1 dòng
    product_means = [_mean_rows(rows) for rows in per_product.values()]    # mỗi sản phẩm 1 dòng (đã trung bình)
    micro = _mean_rows(all_rows)
    macro = _mean_rows([list(m.values()) for m in product_means]) if product_means else None

    n_queries, n_products = len(all_rows), len(per_product)
    print(f"  Query dùng được : {n_queries}  (trên {n_products} sản phẩm, "
          f"bình quân {n_queries / max(1, n_products):.2f} ảnh/sản phẩm)")
    skip_bits = []
    if n_not_in_gallery: skip_bits.append(f"không có trong gallery: {n_not_in_gallery}")
    if n_unreadable:     skip_bits.append(f"ảnh đọc lỗi: {n_unreadable}")
    if n_identical:      skip_bits.append(f"trùng hệt ảnh gallery (bị loại): {n_identical}")
    if skip_bits:
        print("  Bỏ qua: " + " | ".join(skip_bits))

    if micro is None:
        print("  Không có query nào dùng được.")
        return {"gallery": int(search_index.ntotal), "n_queries": 0, "n_products": 0,
                "micro": None, "macro": None}

    same = macro is not None and all(abs(micro[m] - macro[m]) < 1e-9 for m in METRIC_NAMES)
    print(f"  {'Chỉ số':<10}{'micro (mỗi ảnh)':>18}{'macro (mỗi sản phẩm)':>22}")
    for m in METRIC_NAMES:
        mv = macro[m] if macro else float("nan")
        print(f"  {m:<10}{micro[m]:>18.4f}{mv:>22.4f}")
    if same:
        print("  (micro = macro: mỗi sản phẩm đúng 1 query — như Text→Image)")

    return {
        "gallery": int(search_index.ntotal),
        "n_queries": n_queries, "n_products": n_products,
        "n_not_in_gallery": n_not_in_gallery, "n_unreadable": n_unreadable, "n_identical": n_identical,
        "micro": micro, "macro": macro,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.getmtime(CHECKPOINT_BEST) > os.path.getmtime(FAISS_IMAGE_INDEX_FILE):
        print("⚠ CẢNH BÁO: checkpoint MỚI HƠN image index. "
              "Hãy chạy lại build_index.py trước khi evaluate!")

    model = load_model(CHECKPOINT_BEST).to(device)
    image_index, image_mapping = load_faiss(FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE)

    # Query T→I: test loader chính (text) — mỗi sản phẩm đúng 1 query, không đổi so với bản cũ.
    loaders = build_dataloaders(
        batch_size=EVAL_BATCH_SIZE, num_workers=0,
        split_dir=Path(SPLIT_DIR), image_dir=Path(IMAGE_DIR),
    )
    test_loader = loaders[EVALUATION_SPLIT]

    # Use the configured evaluation split (normally data/splits/test.jsonl).
    split_path = Path(SPLIT_DIR) / f"{EVALUATION_SPLIT}.jsonl"

    # Query I→I: every alternate image for products in the evaluation split.
    alt_ds = AltImageDataset(split_path)
    print(f"I→I: {len(alt_ds)} ảnh phụ từ split {EVALUATION_SPLIT}")
    alt_loader = DataLoader(
        alt_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Gallery subset for the configured evaluation split.
    with open(split_path, encoding="utf-8") as f:
        test_ids = {str(json.loads(line)[PRODUCT_ID_FIELD]) for line in f if line.strip()}
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
                require_ok=False, dedupe_identical=False,
            ),
            "Image → Image": evaluate_direction(
                "Image → Image", g_index, g_mapping, alt_loader, device,
                model.encode_image, ["image"],
                require_ok=True, dedupe_identical=True,
            ),
        }

    # ---- Lưu kết quả ----
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.txt", "w", encoding="utf-8") as f:
        for g_name, dirs in all_results.items():
            for d_name, r in dirs.items():
                f.write(f"=== {d_name} | gallery: {g_name} ({r['gallery']}) ===\n")
                f.write(f"Queries   : {r['n_queries']} (sản phẩm: {r.get('n_products', r['n_queries'])})\n")
                if r["micro"] is None:
                    f.write("(không có query nào dùng được)\n\n")
                    continue
                f.write("[micro — mỗi ảnh một phiếu]\n")
                for m in METRIC_NAMES:
                    f.write(f"{m:<10}: {r['micro'][m]:.4f}\n")
                if r["macro"] is not None:
                    f.write("[macro — mỗi sản phẩm một phiếu]\n")
                    for m in METRIC_NAMES:
                        f.write(f"{m:<10}: {r['macro'][m]:.4f}\n")
                f.write("\n")
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nKết quả đã lưu vào results/metrics.txt và results/metrics.json")
    print("Lưu ý: số liệu I→I không so sánh trực tiếp được với các lần evaluate.py trước đây "
          "(giờ dùng mọi ảnh phụ thay vì một ảnh, và đã loại query trùng hệt ảnh gallery).")


if __name__ == "__main__":
    evaluate()
