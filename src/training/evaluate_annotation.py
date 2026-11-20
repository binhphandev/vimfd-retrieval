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
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

import torch
import faiss
import numpy as np
import openpyxl
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from underthesea import word_tokenize

from src.models.model import ViMFRModel
from src.validation.annotations import normalize_query, normalize_url
from src.training.evaluation_metrics import bootstrap_mean_ci, reciprocal_rank_at_k
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
ANNOTATION_DIR       = PROJECT_ROOT / "data" / "annotation"
QUERY_LIST_FILE      = ANNOTATION_DIR / "query_list_200.xlsx"
ANNOTATION_BINH_FILE = ANNOTATION_DIR / "annotation_pairs_Binh.xlsx"
ANNOTATION_TRI_FILE  = ANNOTATION_DIR / "annotation_pairs_Tri.xlsx"


# ------------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------------

def load_queries(query_file: Path) -> list[dict]:
    """Load 200 query từ query_list_200.xlsx"""
    wb = openpyxl.load_workbook(query_file, read_only=True, data_only=True)
    ws = wb["Query List"] if "Query List" in wb.sheetnames else wb.active
    queries = []
    try:
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and isinstance(row[0], int):
                queries.append({
                    "stt":    row[0],
                    "nhom":   row[1],
                    "query":  row[2],
                    "intent": row[3],
                })
    finally:
        wb.close()
    log.info(f"Loaded {len(queries)} queries từ {query_file}")
    return queries


def load_annotation(annotation_file: Path) -> dict[str, dict[str, int]]:
    """
    Load annotation từ 1 file xlsx.
    Returns: {query_text: {product_url: relevance_score}}
    """
    wb = openpyxl.load_workbook(annotation_file, read_only=True, data_only=True)
    ws = wb["Annotation"] if "Annotation" in wb.sheetnames else wb.active
    annotation = defaultdict(dict)
    try:
        for row_number, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            if not row[0]:
                continue
            query       = normalize_query(row[1])   # Query (của user)
            product_url = normalize_url(row[9])     # Link sản phẩm
            label       = normalize_query(row[11])  # NHÃN (exact/partial/irrelevant)
            if query and product_url and label in RELEVANCE_SCORE:
                if product_url in annotation[query]:
                    raise ValueError(
                        f"Duplicate annotation pair at row {row_number} in {annotation_file}: "
                        f"{query!r}, {product_url!r}"
                    )
                annotation[query][product_url] = RELEVANCE_SCORE[label]
    finally:
        wb.close()
    log.info(f"Loaded annotation cho {len(annotation)} queries từ {annotation_file}")
    return dict(annotation)


def merge_annotations(
    ann_binh: dict[str, dict[str, int]],
    ann_tri: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """
    Merge 2 annotator bằng cách lấy trung bình relevance score.
    Chỉ giữ các cặp (query, product) mà cả 2 annotator đều annotate.
    """
    merged = {}
    all_queries = set(ann_binh.keys()) & set(ann_tri.keys())
    for query in all_queries:
        products_binh = ann_binh.get(query, {})
        products_tri  = ann_tri.get(query, {})
        all_products  = set(products_binh.keys()) & set(products_tri.keys())
        merged_products = {}
        for url in all_products:
            score_binh = products_binh[url]
            score_tri  = products_tri[url]
            merged_products[url] = (score_binh + score_tri) / 2
        if merged_products:
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
            raise FileNotFoundError(f"Không tìm thấy catalog split: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            products = json.load(f)
        for p in products:
            url = p.get("url", "")
            pid = p.get("id", "")
            if url and pid:
                url_to_id[normalize_url(url)] = pid
    log.info(f"Built url→id mapping: {len(url_to_id)} products")
    return url_to_id


def convert_annotation_to_product_id(
    annotation: dict[str, dict[str, float]],
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
            pid = url_to_id.get(normalize_url(url))
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


def validate_annotation_coverage(
    queries: list[dict],
    annotation: dict[str, dict[str, float]],
    unmapped_pair_count: int,
    allow_partial: bool,
) -> int:
    annotated_queries = {
        normalize_query(query["query"])
        for query in queries
        if normalize_query(query["query"]) in annotation
    }
    missing_query_count = len(queries) - len(annotated_queries)
    if not annotated_queries:
        raise ValueError("Không có query nào có nhãn hợp lệ sau khi ánh xạ sang catalog.")
    if (missing_query_count or unmapped_pair_count) and not allow_partial:
        raise ValueError(
            f"Chỉ có {len(annotated_queries)}/{len(queries)} query có nhãn hợp lệ "
            f"({missing_query_count} query thiếu); "
            f"{unmapped_pair_count} cặp annotation không ánh xạ được sang catalog. "
            "Bổ sung/sửa nhãn hoặc dùng --allow-partial cho phân tích thăm dò."
        )
    return missing_query_count


# ------------------------------------------------------------------
# Main evaluate
# ------------------------------------------------------------------

def evaluate_annotation(
    query_file: Path = QUERY_LIST_FILE,
    annotation_binh_file: Path = ANNOTATION_BINH_FILE,
    annotation_tri_file: Path = ANNOTATION_TRI_FILE,
    output_dir: Path | None = None,
    allow_partial: bool = False,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load queries
    queries = load_queries(query_file)

    # Load + merge annotation
    ann_binh = load_annotation(annotation_binh_file)
    ann_tri  = load_annotation(annotation_tri_file)
    merged   = merge_annotations(ann_binh, ann_tri)

    # Build url → product_id mapping
    data_dir   = PROJECT_ROOT / "data" / "processed"
    url_to_id  = build_url_to_product_id(data_dir)
    unmapped_pair_count = sum(
        1
        for products in merged.values()
        for url in products
        if normalize_url(url) not in url_to_id
    )
    annotation = convert_annotation_to_product_id(merged, url_to_id)

    log.info(f"Queries có annotation: {len(annotation)}/{len(queries)}")
    missing_query_count = validate_annotation_coverage(
        queries,
        annotation,
        unmapped_pair_count,
        allow_partial,
    )

    # Load model and index only after verifying annotation coverage.
    model = load_model().to(device)
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
    image_index, image_mapping = load_faiss(
        FAISS_IMAGE_INDEX_FILE,
        FAISS_IMAGE_MAPPING_FILE,
    )

    # ------------------------------------------------------------------
    # Evaluate từng query
    # ------------------------------------------------------------------
    results_per_query = []
    skipped = 0

    for q in tqdm(queries, desc="Evaluating"):
        query_text = q["query"]

        normalized_query = normalize_query(query_text)
        if normalized_query not in annotation:
            skipped += 1
            continue

        relevance = annotation[normalized_query]
        model_query = word_tokenize(query_text, format="text")

        # Tokenize query
        encoding = tokenizer(
            model_query,
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
            "query_id":    q["stt"],
            "query":      query_text,
            "nhom":       q["nhom"],
            "intent":     q["intent"],
            "retrieved_ids": retrieved,
            "relevance": relevance,
            "ndcg@5":     ndcg_at_k(retrieved, relevance, 5),
            "ndcg@10":    ndcg_at_k(retrieved, relevance, 10),
            "recall@5":   recall_at_k(retrieved, relevance, 5),
            "recall@10":  recall_at_k(retrieved, relevance, 10),
            "precision@5": precision_at_k(retrieved, relevance, 5),
            "mrr@10": reciprocal_rank_at_k(
                retrieved,
                {pid for pid, score in relevance.items() if score > 0},
                10,
            ),
        })

    # ------------------------------------------------------------------
    # Tổng hợp kết quả
    # ------------------------------------------------------------------
    n = len(results_per_query)
    if n == 0:
        raise ValueError("Không có query nào đủ nhãn và sản phẩm hợp lệ để đánh giá.")

    avg = {
        "NDCG@5":      round(sum(r["ndcg@5"]      for r in results_per_query) / n, 4),
        "NDCG@10":     round(sum(r["ndcg@10"]     for r in results_per_query) / n, 4),
        "Recall@5":    round(sum(r["recall@5"]    for r in results_per_query) / n, 4),
        "Recall@10":   round(sum(r["recall@10"]   for r in results_per_query) / n, 4),
        "Precision@5": round(sum(r["precision@5"] for r in results_per_query) / n, 4),
        "MRR@10":      round(sum(r["mrr@10"]       for r in results_per_query) / n, 4),
    }
    confidence_intervals = {
        "NDCG@10": bootstrap_mean_ci([row["ndcg@10"] for row in results_per_query]),
        "MRR@10": bootstrap_mean_ci([row["mrr@10"] for row in results_per_query]),
        "Recall@10": bootstrap_mean_ci([row["recall@10"] for row in results_per_query]),
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
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / f"annotation_eval_{datetime.now():%Y%m%d_%H%M%S}"
        )
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "query_file": str(query_file),
        "n_queries_in_list": len(queries),
        "n_queries_evaluated": n,
        "n_queries_missing_annotations": missing_query_count,
        "n_annotation_pairs_unmapped": unmapped_pair_count,
        "allow_partial": allow_partial,
        "metrics": avg,
        "confidence_intervals": confidence_intervals,
        "bootstrap": {"n_resamples": 10_000, "seed": 42, "confidence_level": 0.95},
    }
    with (output_dir / "annotation_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("=== Annotation Evaluation — Text → Image ===\n")
        f.write(f"Queries evaluated : {n} / {len(queries)}\n")
        f.write(f"Queries skipped   : {skipped}\n\n")
        for metric, value in avg.items():
            f.write(f"{metric:15}: {value:.4f}\n")
        f.write("\n95% confidence intervals (query-level bootstrap):\n")
        for metric, ci in confidence_intervals.items():
            f.write(
                f"{metric:15}: [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]\n"
            )
        f.write("\n=== Kết quả theo nhóm ===\n")
        for nhom, items in sorted(groups.items()):
            avg_ndcg10 = sum(r["ndcg@10"] for r in items) / len(items)
            f.write(f"{nhom:20} | n={len(items):3} | NDCG@10={avg_ndcg10:.4f}\n")

    with (output_dir / "annotation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, ensure_ascii=False, indent=2)
    with (output_dir / "annotation_per_query.json").open("w", encoding="utf-8") as file:
        json.dump(results_per_query, file, ensure_ascii=False, indent=2)

    log.info("Kết quả đã lưu vào %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViMFR on pooled manual annotations.")
    parser.add_argument("--query-file", type=Path, default=QUERY_LIST_FILE)
    parser.add_argument("--annotation-binh", type=Path, default=ANNOTATION_BINH_FILE)
    parser.add_argument("--annotation-tri", type=Path, default=ANNOTATION_TRI_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow exploratory evaluation when not all listed queries have shared labels.",
    )
    args = parser.parse_args()
    evaluate_annotation(
        query_file=args.query_file,
        annotation_binh_file=args.annotation_binh,
        annotation_tri_file=args.annotation_tri,
        output_dir=args.output_dir,
        allow_partial=args.allow_partial,
    )
