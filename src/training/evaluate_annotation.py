# src/training/evaluate_annotation.py
#
# Evaluate model trên bộ annotation thủ công 200 query
# Dùng graded relevance: exact=2, partial=1, irrelevant=0
# Metrics: NDCG@K, Precision@K, Recall@K

import sys
import os
import math
import json
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

import torch
import faiss
import numpy as np
import openpyxl
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

from src.models.model import ViMFRModel
from config import (
    CHECKPOINT_BEST,
    FAISS_INDEX_FILE, FAISS_MAPPING_FILE,
    FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE,
    CKPT_MODEL_KEY, PHOBERT_MODEL, MAX_TEXT_LENGTH,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Relevance score mapping
# ------------------------------------------------------------------
RELEVANCE_SCORE = {
    "exact":      2,
    "partial":    1,
    "irrelevant": 0,
}

# ------------------------------------------------------------------
# Đường dẫn file annotation — sửa lại nếu cần
# ------------------------------------------------------------------
ANNOTATION_DIR       = Path("data/annotation")
QUERY_LIST_FILE      = ANNOTATION_DIR / "query_list_200.xlsx"
ANNOTATION_BINH_FILE = ANNOTATION_DIR / "annotation_pairs_Binh.xlsx"
ANNOTATION_TRI_FILE  = ANNOTATION_DIR / "annotation_pairs_Tri.xlsx"


# ------------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------------

def load_queries(query_file: Path) -> list[dict]:
    """Load 200 query từ query_list_200.xlsx"""
    wb = openpyxl.load_workbook(query_file)
    ws = wb.active
    queries = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and isinstance(row[0], int):
            queries.append({
                "stt":    row[0],
                "nhom":   row[1],
                "query":  row[2],
                "intent": row[3],
            })
    log.info(f"Loaded {len(queries)} queries từ {query_file}")
    return queries


def load_annotation(annotation_file: Path) -> dict[str, dict[str, int]]:
    """
    Load annotation từ 1 file xlsx.
    Returns: {query_text: {product_url: relevance_score}}
    """
    wb = openpyxl.load_workbook(annotation_file)
    ws = wb.active
    annotation = defaultdict(dict)
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        query       = row[1]   # Query (của user)
        product_url = row[9]   # Link sản phẩm
        label       = row[11]  # NHÃN (exact/partial/irrelevant)
        if query and product_url and label in RELEVANCE_SCORE:
            annotation[query][product_url] = RELEVANCE_SCORE[label]
    log.info(f"Loaded annotation cho {len(annotation)} queries từ {annotation_file}")
    return dict(annotation)


def merge_annotations(ann_binh: dict, ann_tri: dict) -> dict[str, dict[str, int]]:
    """
    Merge 2 annotator bằng cách lấy trung bình relevance score.
    Chỉ giữ các cặp (query, product) mà cả 2 annotator đều annotate.
    """
    merged = {}
    all_queries = set(ann_binh.keys()) | set(ann_tri.keys())
    for query in all_queries:
        products_binh = ann_binh.get(query, {})
        products_tri  = ann_tri.get(query, {})
        all_products  = set(products_binh.keys()) | set(products_tri.keys())
        merged_products = {}
        for url in all_products:
            score_binh = products_binh.get(url, 0)
            score_tri  = products_tri.get(url, 0)
            # Lấy trung bình, làm tròn
            merged_products[url] = (score_binh + score_tri) / 2
        merged[query] = merged_products
    log.info(f"Merged annotation: {len(merged)} queries")
    return merged


def build_url_to_product_id(data_dir: Path) -> dict[str, str]:
    """
    Build mapping: product_url → product_id
    Đọc từ train.json + val.json + test.json
    """
    url_to_id = {}
    for split in ("train", "val", "test"):
        json_path = data_dir / f"{split}.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            products = json.load(f)
        for p in products:
            url = p.get("url", "")
            pid = p.get("id", "")
            if url and pid:
                url_to_id[url] = pid
    log.info(f"Built url→id mapping: {len(url_to_id)} products")
    return url_to_id


def convert_annotation_to_product_id(
    annotation: dict[str, dict[str, int]],
    url_to_id: dict[str, str],
) -> dict[str, dict[str, float]]:
    """
    Đổi key từ product_url sang product_id.
    Bỏ qua các URL không tìm thấy trong data.
    """
    converted = {}
    not_found = 0
    for query, products in annotation.items():
        converted_products = {}
        for url, score in products.items():
            pid = url_to_id.get(url)
            if pid:
                converted_products[pid] = score
            else:
                not_found += 1
        if converted_products:
            converted[query] = converted_products
    if not_found:
        log.warning(f"{not_found} URLs không tìm thấy trong data — bỏ qua")
    return converted


# ------------------------------------------------------------------
# Load model + index
# ------------------------------------------------------------------

def load_model() -> ViMFRModel:
    model = ViMFRModel()
    ckpt  = torch.load(CHECKPOINT_BEST, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[CKPT_MODEL_KEY])
    model.eval()
    log.info(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def load_faiss(index_path: str, mapping_path: str):
    index = faiss.read_index(index_path)
    with open(mapping_path) as f:
        mapping = json.load(f)
    log.info(f"Loaded index: {index_path} ({index.ntotal} vectors)")
    return index, mapping


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """
    NDCG@K với graded relevance.
    retrieved : list product_id theo thứ tự search
    relevance : {product_id: relevance_score}
    """
    dcg = 0.0
    for rank, pid in enumerate(retrieved[:k], 1):
        rel = relevance.get(pid, 0.0)
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG — sắp xếp relevance score giảm dần
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, 1)
        if rel > 0
    )

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """Recall@K — tỉ lệ sản phẩm relevant tìm được trong top K"""
    relevant_set = {pid for pid, score in relevance.items() if score > 0}
    if not relevant_set:
        return 0.0
    retrieved_relevant = sum(1 for pid in retrieved[:k] if pid in relevant_set)
    return retrieved_relevant / len(relevant_set)


def precision_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """Precision@K — tỉ lệ sản phẩm relevant trong top K kết quả"""
    if k == 0:
        return 0.0
    retrieved_relevant = sum(
        1 for pid in retrieved[:k]
        if relevance.get(pid, 0) > 0
    )
    return retrieved_relevant / k


# ------------------------------------------------------------------
# Main evaluate
# ------------------------------------------------------------------

def evaluate_annotation():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    model     = load_model().to(device)
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    # Load FAISS index — dùng image index cho Text→Image
    image_index, image_mapping = load_faiss(FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE)

    # Load queries
    queries = load_queries(QUERY_LIST_FILE)

    # Load + merge annotation
    ann_binh = load_annotation(ANNOTATION_BINH_FILE)
    ann_tri  = load_annotation(ANNOTATION_TRI_FILE)
    merged   = merge_annotations(ann_binh, ann_tri)

    # Build url → product_id mapping
    data_dir   = Path("data/processed")
    url_to_id  = build_url_to_product_id(data_dir)
    annotation = convert_annotation_to_product_id(merged, url_to_id)

    log.info(f"Queries có annotation: {len(annotation)}/{len(queries)}")

    # ------------------------------------------------------------------
    # Evaluate từng query
    # ------------------------------------------------------------------
    results_per_query = []
    skipped = 0

    for q in tqdm(queries, desc="Evaluating"):
        query_text = q["query"]

        if query_text not in annotation:
            skipped += 1
            continue

        relevance = annotation[query_text]

        # Tokenize query
        encoding = tokenizer(
            query_text,
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_emb = model.encode_text(
                encoding["input_ids"].to(device),
                encoding["attention_mask"].to(device),
            ).cpu().numpy().astype("float32")

        # Search FAISS top-10
        _, indices = image_index.search(text_emb, 10)
        retrieved  = [
            image_mapping[str(idx)]
            for idx in indices[0]
            if str(idx) in image_mapping
        ]

        results_per_query.append({
            "query":      query_text,
            "nhom":       q["nhom"],
            "intent":     q["intent"],
            "ndcg@5":     ndcg_at_k(retrieved, relevance, 5),
            "ndcg@10":    ndcg_at_k(retrieved, relevance, 10),
            "recall@5":   recall_at_k(retrieved, relevance, 5),
            "recall@10":  recall_at_k(retrieved, relevance, 10),
            "precision@5": precision_at_k(retrieved, relevance, 5),
        })

    # ------------------------------------------------------------------
    # Tổng hợp kết quả
    # ------------------------------------------------------------------
    n = len(results_per_query)
    if n == 0:
        log.error("Không có query nào được evaluate!")
        return

    avg = {
        "NDCG@5":      round(sum(r["ndcg@5"]      for r in results_per_query) / n, 4),
        "NDCG@10":     round(sum(r["ndcg@10"]     for r in results_per_query) / n, 4),
        "Recall@5":    round(sum(r["recall@5"]    for r in results_per_query) / n, 4),
        "Recall@10":   round(sum(r["recall@10"]   for r in results_per_query) / n, 4),
        "Precision@5": round(sum(r["precision@5"] for r in results_per_query) / n, 4),
    }

    print(f"\n{'='*40}")
    print(f"  Annotation Evaluation — Text → Image")
    print(f"  Queries evaluated : {n} / {len(queries)}")
    print(f"  Queries skipped   : {skipped}")
    print(f"{'='*40}")
    for metric, value in avg.items():
        print(f"  {metric:15}: {value:.4f}")
    print(f"{'='*40}")

    # ------------------------------------------------------------------
    # Phân tích theo nhóm category
    # ------------------------------------------------------------------
    print("\n=== Kết quả theo nhóm ===")
    groups = defaultdict(list)
    for r in results_per_query:
        groups[r["nhom"]].append(r)

    for nhom, items in sorted(groups.items()):
        avg_ndcg10 = sum(r["ndcg@10"] for r in items) / len(items)
        print(f"  {nhom:20} | n={len(items):3} | NDCG@10={avg_ndcg10:.4f}")

    # ------------------------------------------------------------------
    # Lưu kết quả
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    with open("results/annotation_metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Annotation Evaluation — Text → Image ===\n")
        f.write(f"Queries evaluated : {n} / {len(queries)}\n")
        f.write(f"Queries skipped   : {skipped}\n\n")
        for metric, value in avg.items():
            f.write(f"{metric:15}: {value:.4f}\n")
        f.write("\n=== Kết quả theo nhóm ===\n")
        for nhom, items in sorted(groups.items()):
            avg_ndcg10 = sum(r["ndcg@10"] for r in items) / len(items)
            f.write(f"{nhom:20} | n={len(items):3} | NDCG@10={avg_ndcg10:.4f}\n")

    log.info("Kết quả đã lưu vào results/annotation_metrics.txt")


if __name__ == "__main__":
    evaluate_annotation()
