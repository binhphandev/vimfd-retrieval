"""
Người A import 3 thứ từ file này:
    from dataset import get_image_transform, ViMFDDataset, build_dataloaders
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Any, Literal, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from data_preprocess import _pick_primary_image

from config import (
    IMAGE_LIST_FIELD,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    MAX_TEXT_LENGTH,
    PHOBERT_MODEL,
    PROCESSED_DIR,
    PRODUCT_ID_FIELD,
    TEXT_FIELDS,
)

log = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


# --- Image transform ---

def get_image_transform(split: SplitType) -> transforms.Compose:
    """
    - train     : RandomHorizontalFlip + ColorJitter nhẹ + Normalize
    - val / test: Resize + CenterCrop + Normalize (không augment)
    """
    normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# --- Helpers ---

def _build_text(product: dict) -> str:
    """Ghép TEXT_FIELDS thành 1 chuỗi đầu vào cho PhoBERT."""
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


# --- Dataset ---

class ViMFDDataset(Dataset):
    """
    Mỗi item trả về:
        {
            "image"          : FloatTensor [3, 224, 224],
            "input_ids"      : LongTensor  [MAX_TEXT_LENGTH],
            "attention_mask" : LongTensor  [MAX_TEXT_LENGTH],
            "product_id"     : str,
        }
    """

    def __init__(
        self,
        split: SplitType,
        tokenizer: Any,
        split_dir: Path | None = None,
        image_dir: Path | None = None,
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.transform = get_image_transform(split)

        split_dir = Path(split_dir) if split_dir else Path(PROCESSED_DIR)
        self.image_dir = Path(image_dir) if image_dir else split_dir / "images"

        json_path = split_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {json_path}. Hãy chạy data_preprocess.py trước."
            )

        with open(json_path, encoding="utf-8") as f:
            self.data: list[dict] = json.load(f)

        log.info(f"ViMFDDataset [{split}]: {len(self.data)} products")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        product = self.data[idx]

        img_meta = _pick_primary_image(product.get(IMAGE_LIST_FIELD, []))
        image_tensor = self._load_image(img_meta)

        text = _build_text(product)
        encoding = self.tokenizer(  # type: ignore
            text,
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image":          image_tensor,
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "product_id":     str(product[PRODUCT_ID_FIELD]),
        }

    def _load_image(self, img_meta: dict | None) -> torch.Tensor:
        if img_meta is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        rel = Path(img_meta.get("local_path", ""))
        if rel.parts and rel.parts[0] == "images":
            rel = Path(*rel.parts[1:])

        img_path = self.image_dir / rel

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                return cast(torch.Tensor, self.transform(img))
        except Exception as e:
            log.warning(f"Không đọc được ảnh {img_path}: {e} — dùng tensor zero")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


# --- DataLoader builder ---

def build_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    split_dir: Path | None = None,
    image_dir: Path | None = None,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho cả 3 split, dùng chung 1 tokenizer instance."""
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = ViMFDDataset(
            split=split,
            tokenizer=tokenizer,
            split_dir=split_dir,
            image_dir=image_dir,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders


# --- Smoke test ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("=== Smoke test dataset.py ===")

    loaders = build_dataloaders(batch_size=4, num_workers=0,split_dir = Path("data/processed"), image_dir = Path("data/images"),)

    for split, loader in loaders.items():
        batch = next(iter(loader))
        log.info(f"\n[{split}]")
        log.info(f"  image          : {batch['image'].shape}")
        log.info(f"  input_ids      : {batch['input_ids'].shape}")
        log.info(f"  attention_mask : {batch['attention_mask'].shape}")
        log.info(f"  product_id     : {batch['product_id']}")

    log.info("\nSmoke test PASSED ✓")