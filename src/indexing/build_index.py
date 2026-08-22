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
from typing import Optional, cast

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from config import (
    CHECKPOINT_BEST,
    CKPT_MODEL_KEY,
    EMBED_DIM,
    FAISS_IMAGE_INDEX_FILE,
    FAISS_IMAGE_MAPPING_FILE,
    FAISS_INDEX_FILE,
    FAISS_MAPPING_FILE,
    IMAGE_LIST_FIELD,
    IMAGE_MEAN,
    IMAGE_PATH_FIELD,
    IMAGE_POSITION_FIELD,
    IMAGE_PRIMARY_FIELD,
    IMAGE_SIZE,
    IMAGE_STD,
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


def _select_primary_image(product: dict) -> Optional[dict]:
    """
    Chọn ảnh đại diện của sản phẩm từ IMAGE_LIST_FIELD.
    Ưu tiên: is_featured=True → position nhỏ nhất → phần tử đầu tiên.
    """
    images = product.get(IMAGE_LIST_FIELD)
    if not isinstance(images, list) or not images:
        return None

    for img in images:
        if isinstance(img, dict) and img.get(IMAGE_PRIMARY_FIELD):
            return img

    try:
        return sorted(
            (img for img in images if isinstance(img, dict)),
            key=lambda img: img.get(IMAGE_POSITION_FIELD, float("inf")),
        )[0]
    except IndexError:
        return None


def build_image_transform() -> T.Compose:
    """Transform ảnh dùng chung cho encode — resize/crop + normalize theo config."""
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def _load_image_tensor(product: dict, transform: T.Compose) -> Optional[torch.Tensor]:
    """Load + preprocess ảnh đại diện của 1 sản phẩm. Trả None nếu thiếu/lỗi ảnh."""
    img_meta = _select_primary_image(product)
    if img_meta is None:
        return None

    rel_path = img_meta.get(IMAGE_PATH_FIELD)
    if not rel_path:
        return None

    img_path = Path(rel_path)
    if not img_path.is_absolute():
        # Ảnh thực tế nằm ở data/processed/images/<brand>/... (xem cây thư mục),
        # nên phải nối local_path với PROCESSED_DIR (data/processed), KHÔNG phải
        # DATA_DIR (data) — nối với DATA_DIR sẽ thiếu mất "processed/" và không
        # tìm thấy file (=> mọi sản phẩm bị skip).
        img_path = Path(PROCESSED_DIR) / rel_path

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            # transform (Resize → CenterCrop → ToTensor → Normalize) trả về
            # torch.Tensor lúc runtime, nhưng type stub của Compose là generic
            # theo input nên Pylance suy luận nhầm ra kiểu Image — cast lại cho đúng.
            return cast(torch.Tensor, transform(im))
    except (FileNotFoundError, OSError) as e:
        log.warning(f"  Bỏ qua ảnh lỗi/không tồn tại: {img_path} ({e})")
        return None


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


@torch.no_grad()
def encode_all_products_images(
    products: list[dict],
    model,
    transform: T.Compose,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode ảnh đại diện của toàn bộ sản phẩm → 256D embeddings đã L2-norm.

    Sản phẩm không có ảnh hợp lệ sẽ bị bỏ qua (không đưa vào image_index).

    Returns:
        embeddings : np.ndarray [M, 256], float32   (M <= len(products))
        product_ids: list[str], len = M
    """
    # Bước 1: load + preprocess ảnh trước, bỏ qua sản phẩm thiếu ảnh
    valid_ids: list[str] = []
    valid_tensors: list[torch.Tensor] = []
    skipped = 0
    sample_failed_paths: list[str] = []

    for p in tqdm(products, desc="Loading images"):
        img_meta = _select_primary_image(p)
        rel_path = img_meta.get(IMAGE_PATH_FIELD) if img_meta else None

        tensor = _load_image_tensor(p, transform)
        if tensor is None:
            skipped += 1
            if rel_path and len(sample_failed_paths) < 3:
                resolved = rel_path if Path(rel_path).is_absolute() else str(Path(PROCESSED_DIR) / rel_path)
                sample_failed_paths.append(resolved)
            continue
        valid_ids.append(str(p[PRODUCT_ID_FIELD]))
        valid_tensors.append(tensor)

    if skipped:
        log.warning(f"  Bỏ qua {skipped}/{len(products)} sản phẩm không có ảnh hợp lệ")
        if sample_failed_paths:
            log.warning(f"  Ví dụ path đã thử (không tồn tại/lỗi): {sample_failed_paths}")

    # Bước 2: encode theo batch
    all_embeddings = []
    for i in tqdm(range(0, len(valid_tensors), batch_size), desc="Encoding images"):
        batch = valid_tensors[i : i + batch_size]
        pixel_values = torch.stack(batch, dim=0).to(device)

        # [CONTRACT] encode_image trả về [B, 256] đã L2-norm
        embeddings = model.encode_image(pixel_values)
        all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        raise RuntimeError("Không encode được ảnh nào — kiểm tra lại IMAGE_PATH_FIELD / IMAGE_DIR trong config.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    log.info(f"Encoded {len(valid_ids)} product images → shape {embeddings.shape}")
    return embeddings, valid_ids


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


def image_smoke_test(
    index_path: Path,
    mapping_path: Path,
    products: list[dict],
    transform: T.Compose,
    model,
    device,
) -> None:
    """
    Round-trip test cho image_index: lấy ảnh của 1 sản phẩm bất kỳ đã có trong
    index, encode lại, search → kỳ vọng top-1 chính là sản phẩm đó (score ~1.0).
    """
    log.info("Smoke test image_index ...")

    index = faiss.read_index(str(index_path))
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    query_tensor, query_pid = None, None
    for p in products:
        t = _load_image_tensor(p, transform)
        if t is not None:
            query_tensor, query_pid = t, str(p[PRODUCT_ID_FIELD])
            break

    if query_tensor is None:
        log.warning("  Không tìm được ảnh hợp lệ nào để smoke test — bỏ qua.")
        return

    with torch.no_grad():
        q_emb = model.encode_image(query_tensor.unsqueeze(0).to(device)).cpu().numpy()

    scores, indices = index.search(q_emb, k=5)

    log.info(f"  Query: product_id={query_pid} (ảnh đại diện)")
    log.info(f"  Top-5 results:")
    top1_pid = None
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        pid = mapping.get(str(idx), "?")
        if rank == 1:
            top1_pid = pid
        log.info(f"    {rank}. product_id={pid} | score={score:.4f}")

    if top1_pid == query_pid:
        log.info("Smoke test image_index PASSED ✓ (top-1 khớp chính nó)")
    else:
        log.warning(
            f"Smoke test image_index CẢNH BÁO: top-1={top1_pid} != query={query_pid}. "
            "Kiểm tra lại encode_image / preprocessing nếu điều này bất thường."
        )


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

    # --- Build & save TEXT index ---
    build_and_save_index(
        embeddings,
        product_ids,
        index_path=Path(FAISS_INDEX_FILE),
        mapping_path=Path(FAISS_MAPPING_FILE),
    )

    if not args.no_smoke_test:
        smoke_test(
            Path(FAISS_INDEX_FILE),
            Path(FAISS_MAPPING_FILE),
            model, tokenizer, device,
        )

    # --- Encode + build & save IMAGE index ---
    # QUAN TRỌNG: đây là index riêng, không gian embedding khác với text index
    # ở trên (encode_image vs encode_text). Mọi query dạng ảnh (Image→Image,
    # Image→Text) phải search vào index này, KHÔNG được search vào FAISS_INDEX_FILE.
    transform = build_image_transform()
    image_embeddings, image_product_ids = encode_all_products_images(
        products, model, transform, device, batch_size=args.batch_size
    )

    build_and_save_index(
        image_embeddings,
        image_product_ids,
        index_path=Path(FAISS_IMAGE_INDEX_FILE),
        mapping_path=Path(FAISS_IMAGE_MAPPING_FILE),
    )

    if not args.no_smoke_test:
        image_smoke_test(
            Path(FAISS_IMAGE_INDEX_FILE),
            Path(FAISS_IMAGE_MAPPING_FILE),
            products, transform, model, device,
        )

    log.info("Build index complete.")
    log.info(f"  Text index    : {FAISS_INDEX_FILE}")
    log.info(f"  Text mapping  : {FAISS_MAPPING_FILE}")
    log.info(f"  Image index   : {FAISS_IMAGE_INDEX_FILE}")
    log.info(f"  Image mapping : {FAISS_IMAGE_MAPPING_FILE}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()