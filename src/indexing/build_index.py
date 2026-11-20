"""
Pipeline:
    1. Load checkpoint 
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
    IMAGE_DIR,
    IMAGE_PATH_FIELD,
    IMAGE_POSITION_FIELD,
    IMAGE_PRIMARY_FIELD,
    IMAGE_SIZE,
    IMAGE_STD,
    MAX_TEXT_LENGTH,
    MODEL_TEXT_FIELD,
    PHOBERT_MODEL,
    PRODUCT_ID_FIELD,
    SPLIT_DIR,
    CKPT_EPOCH_KEY,
    CKPT_VAL_LOSS_KEY,
)
# Quy tắc dữ liệu/ảnh dùng CHUNG với API (src/serving/engine.py) và khớp train_colab.ipynb
from src.serving.catalog import (
    ckpt_fingerprint,
    get_eval_transform,
    load_products,
    primary_image,
    resolve_image,
    write_index_meta,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_primary_image(product: dict) -> Optional[dict]:
    """Ảnh đại diện: is_featured -> position nhỏ nhất (chỉ xét ảnh có local_path). Dùng chung với API."""
    return primary_image(product)


def build_image_transform() -> T.Compose:
    """Đúng transform eval lúc train (data/dataset.py: Resize BICUBIC -> CenterCrop -> Normalize)."""
    return get_eval_transform()


def _load_image_tensor(product: dict, transform: T.Compose) -> Optional[torch.Tensor]:
    """Load + preprocess ảnh đại diện của 1 sản phẩm. Trả None nếu thiếu/lỗi ảnh."""
    img_meta = _select_primary_image(product)
    if img_meta is None:
        return None

    # resolve_image chịu được local_path ghi .png nhưng file thật là .jpg
    img_path = resolve_image(img_meta.get(IMAGE_PATH_FIELD, ""))
    if img_path is None:
        log.warning(f"  Bỏ qua ảnh không tồn tại: {img_meta.get(IMAGE_PATH_FIELD)}")
        return None

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            return cast(torch.Tensor, transform(im))
    except OSError as e:
        log.warning(f"  Bỏ qua ảnh lỗi: {img_path} ({e})")
        return None


def load_all_products(split_dir: Path) -> list[dict]:
    """Toàn bộ sản phẩm từ SPLIT_DIR/{train,val,test}.jsonl (nguồn dữ liệu hiện tại)."""
    return load_products(split_dir)


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load ViMFRModel từ checkpoint.
    [CONTRACT] Checkpoint phải có key 'model_state_dict'.
    """
    # Import model từ Người A
    try:
        from src.models.model import ViMFRModel, load_state_dict_compat
    except ImportError:
        raise ImportError(
            "Không tìm thấy model.py. "
            "Hãy chắc chắn Người A đã cung cấp file model.py."
        )

    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = ViMFRModel()
    load_state_dict_compat(model, ckpt[CKPT_MODEL_KEY])
    model.to(device)
    model.eval()

    val_loss = ckpt.get(CKPT_VAL_LOSS_KEY)
    val_loss = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "?"
    log.info(f"  epoch={ckpt.get(CKPT_EPOCH_KEY, '?')} | val_loss={val_loss}")
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

        # Current split records already contain text_seg from the external
        # preprocessing pipeline. Fall back for older records without that field.
        texts = [
            str(p.get(MODEL_TEXT_FIELD) or p.get("text") or p.get("name") or "")
            for p in batch
        ]
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
    Encode ảnh đại diện theo kiểu streaming: mỗi lần chỉ giữ 1 batch tensor
    trong RAM, encode xong thì giải phóng. Sản phẩm thiếu ảnh bị bỏ qua.
    """
    valid_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []
    skipped = 0
    sample_failed_paths: list[str] = []

    def flush() -> None:
        if not batch_tensors:
            return
        pixel_values = torch.stack(batch_tensors, dim=0).to(device)
        # [CONTRACT] encode_image trả về [B, 256] đã L2-norm
        emb = model.encode_image(pixel_values)
        all_embeddings.append(emb.cpu().numpy())
        valid_ids.extend(batch_ids)
        batch_tensors.clear()
        batch_ids.clear()

    for p in tqdm(products, desc="Encoding images"):
        tensor = _load_image_tensor(p, transform)
        if tensor is None:
            skipped += 1
            img_meta = _select_primary_image(p)
            rel_path = img_meta.get(IMAGE_PATH_FIELD) if img_meta else None
            if rel_path and len(sample_failed_paths) < 3:
                rp = Path (rel_path)
                if rp.is_absolute():
                    resolved = str(rp)
                else:
                    rp = Path(*rp.parts[1:]) if rp.parts and rp.parts[0] == "images" else rp
                    resolved = str(Path(IMAGE_DIR) / rp)
                sample_failed_paths.append(resolved)
            continue

        batch_tensors.append(tensor)
        batch_ids.append(str(p[PRODUCT_ID_FIELD]))
        if len(batch_tensors) >= batch_size:
            flush()

    flush()  # batch cuối

    if skipped:
        log.warning(f"  Bỏ qua {skipped}/{len(products)} sản phẩm không có ảnh hợp lệ")
        if sample_failed_paths:
            log.warning(f"  Ví dụ path đã thử (không tồn tại/lỗi): {sample_failed_paths}")

    if not all_embeddings:
        raise RuntimeError("Không encode được ảnh nào, kiểm tra lại IMAGE_PATH_FIELD / IMAGE_DIR / SPLIT_DIR trong config.")

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
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Bỏ qua bước build text index (dùng khi text index đã có sẵn)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Load data ---
    products = load_all_products(Path(SPLIT_DIR))

    # --- Load model ---
    model = load_model(args.checkpoint, device)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

  

    # --- Build & save TEXT index ---
    if not args.skip_text:
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        embeddings, product_ids = encode_all_products(
            products, model, tokenizer, device, batch_size=args.batch_size
        )
        build_and_save_index(
            embeddings, product_ids,
            index_path=Path(FAISS_INDEX_FILE),
            mapping_path=Path(FAISS_MAPPING_FILE),
        )
        if not args.no_smoke_test:
            smoke_test(Path(FAISS_INDEX_FILE), Path(FAISS_MAPPING_FILE), model, tokenizer, device)


    # --- Encode + build & save IMAGE index ---
    # QUAN TRỌNG: API chỉ dùng index này. Mọi query (Text→Image và Image→Image,
    # như search_text/search_image trong train_colab.ipynb) đều search vào image index.
    # Text index ở trên không còn được API dùng: có thể thêm --skip-text để bỏ qua.
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
    # Dấu vân tay để API từ chối chạy nếu index và checkpoint/dữ liệu không khớp nhau
    write_index_meta(
        FAISS_IMAGE_INDEX_FILE,
        checkpoint=str(args.checkpoint),
        fingerprint=ckpt_fingerprint(args.checkpoint),
        split_dir=str(SPLIT_DIR),
        image_dir=str(IMAGE_DIR),
        n_products=len(products),
        n_indexed=len(image_product_ids),
        transform=repr(transform),
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
