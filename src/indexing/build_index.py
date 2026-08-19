"""
Pipeline:
    1. Load checkpoint từ Người A
    2. Encode toàn bộ text sản phẩm → 256D embeddings
    3. Build FAISS IndexFlatIP
    4. Lưu index + id_mapping.json

Cách dùng:
    python build_index.py
    python build_index.py --checkpoint checkpoints/best.pt
    python build_index.py --batch-size 128
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from config import (
    CHECKPOINT_BEST,
    CKPT_MODEL_KEY,
    EMBED_DIM,
    FAISS_INDEX_FILE,
    FAISS_MAPPING_FILE,
    MAX_TEXT_LENGTH,
    PHOBERT_MODEL,
    PROCESSED_DIR,
    PRODUCT_ID_FIELD,
    TEXT_FIELDS,
)



log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(product: dict) -> str:
    """Ghép các field text thành 1 chuỗi — giống dataset.py."""
    parts = []
    for field in TEXT_FIELDS:
        value = product.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            joined = " ".join(str(v) for v in value if v)
            if joined.strip():
                parts.append(joined.strip())
    return " ".join(parts)


def load_all_products(processed_dir: Path) -> list[dict]:
    """Load toàn bộ sản phẩm từ train + val + test."""
    all_products = []
    for split in ("train", "val", "test"):
        json_path = processed_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {json_path}. Hãy chạy data_preprocess.py trước."
            )
        with open(json_path, encoding="utf-8") as f:
            all_products.extend(json.load(f))

    log.info(f"Loaded {len(all_products)} products từ train/val/test")
    return all_products


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load ViMFRModel từ checkpoint.
    [CONTRACT] Checkpoint phải có key 'model_state_dict'.
    """
    # Import model từ Người A
    try:
        from src.models.model import ViMFRModel
    except ImportError:
        raise ImportError(
            "Không tìm thấy model.py. "
            "Hãy chắc chắn Người A đã cung cấp file model.py."
        )

    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = ViMFRModel()
    model.load_state_dict(ckpt[CKPT_MODEL_KEY])
    model.to(device)
    model.eval()

    log.info(f"  epoch={ckpt.get('epoch', '?')} | val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all_products(
    products: list[dict],
    model,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode toàn bộ text sản phẩm → 256D embeddings đã L2-norm.

    Returns:
        embeddings : np.ndarray [N, 256], float32
        product_ids: list[str], len = N
    """
    all_embeddings = []
    all_ids = []

    for i in tqdm(range(0, len(products), batch_size), desc="Encoding"):
        batch = products[i : i + batch_size]

        texts = [_build_text(p) for p in batch]
        ids   = [str(p[PRODUCT_ID_FIELD]) for p in batch]

        encoding = tokenizer(
            texts,
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # [CONTRACT] encode_text trả về [B, 256] đã L2-norm
        embeddings = model.encode_text(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())
        all_ids.extend(ids)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    log.info(f"Encoded {len(all_ids)} products → shape {embeddings.shape}")
    return embeddings, all_ids


# ---------------------------------------------------------------------------
# Build & save FAISS index
# ---------------------------------------------------------------------------

def build_and_save_index(
    embeddings: np.ndarray,
    product_ids: list[str],
    index_path: Path,
    mapping_path: Path,
) -> None:
    """
    Build FAISS IndexFlatIP và lưu ra disk kèm id_mapping.json.

    IndexFlatIP + L2-normed vectors = cosine similarity search.
    """
    n, dim = embeddings.shape
    assert dim == EMBED_DIM, f"Embedding dim {dim} != EMBED_DIM {EMBED_DIM}"

    log.info(f"Building FAISS IndexFlatIP (dim={dim}, n={n}) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  Index built — ntotal={index.ntotal}")

    # Lưu index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    log.info(f"  Saved index → {index_path}")

    # Lưu mapping: { "0": "product_id_A", "1": "product_id_B", ... }
    mapping = {str(i): pid for i, pid in enumerate(product_ids)}
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved mapping → {mapping_path} ({len(mapping)} entries)")


# ---------------------------------------------------------------------------
# Smoke test index
# ---------------------------------------------------------------------------

def smoke_test(index_path: Path, mapping_path: Path, model, tokenizer, device) -> None:
    """Thử search 1 query để xác nhận index hoạt động."""
    log.info("Smoke test index ...")

    index = faiss.read_index(str(index_path))
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    query = "áo sơ mi nam trắng"
    encoding = tokenizer(
        query,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        q_emb = model.encode_text(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        ).cpu().numpy()

    scores, indices = index.search(q_emb, k=5)

    log.info(f"  Query: '{query}'")
    log.info(f"  Top-5 results:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        pid = mapping.get(str(idx), "?")
        log.info(f"    {rank}. product_id={pid} | score={score:.4f}")

    log.info("Smoke test PASSED ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index cho ViMFD")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(CHECKPOINT_BEST),
        help=f"Path tới checkpoint (default: {CHECKPOINT_BEST})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size khi encode (default: 64)",
    )
    parser.add_argument(
        "--no-smoke-test", action="store_true",
        help="Bỏ qua smoke test sau khi build",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Load data ---
    products = load_all_products(Path(PROCESSED_DIR))

    # --- Load model ---
    model = load_model(args.checkpoint, device)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    # --- Encode ---
    embeddings, product_ids = encode_all_products(
        products, model, tokenizer, device, batch_size=args.batch_size
    )

    # --- Build & save index ---
    build_and_save_index(
        embeddings,
        product_ids,
        index_path=Path(FAISS_INDEX_FILE),
        mapping_path=Path(FAISS_MAPPING_FILE),
    )

    # --- Smoke test ---
    if not args.no_smoke_test:
        smoke_test(
            Path(FAISS_INDEX_FILE),
            Path(FAISS_MAPPING_FILE),
            model, tokenizer, device,
        )

    log.info("Build index complete.")
    log.info(f"  Index   : {FAISS_INDEX_FILE}")
    log.info(f"  Mapping : {FAISS_MAPPING_FILE}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()