import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    IMAGE_LIST_FIELD,
    IMAGE_PATH_FIELD,
    IMAGE_POSITION_FIELD,
    IMAGE_PRIMARY_FIELD,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    MAX_TEXT_LENGTH,
    MODEL_TEXT_FIELD,
    PHOBERT_MODEL,
    PROCESSED_DIR,
    PRODUCT_ID_FIELD,
)
from src.text_processing.text_utils import get_model_text

log = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


def _pick_primary_image(images: list[dict]) -> dict | None:
    if not images:
        return None
    for image in images:
        if image.get(IMAGE_PRIMARY_FIELD):
            return image
    for image in images:
        if image.get(IMAGE_POSITION_FIELD) == 0:
            return image
    return images[0]


# --- Image transform ---

def get_image_transform(split: SplitType) -> transforms.Compose:
    """
    - train     : RandomHorizontalFlip + ColorJitter nhẹ + Normalize
    - val / test: Resize + CenterCrop + Normalize (không augment)
    """
    normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(
                    IMAGE_SIZE,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


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

        self.split_dir = Path(split_dir) if split_dir else Path(PROCESSED_DIR)
        self.image_dir = Path(image_dir) if image_dir else self.split_dir / "images"

        json_path = self.split_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {json_path}. Hãy chạy data_preprocess.py trước."
            )

        with json_path.open(encoding="utf-8") as file:
            self.data: list[dict] = json.load(file)

        log.info("ViMFDDataset [%s]: %s products", split, len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        product = self.data[idx]

        img_meta = _pick_primary_image(product.get(IMAGE_LIST_FIELD, []))
        product_id = str(product.get(PRODUCT_ID_FIELD, "<missing id>"))
        image_tensor = self._load_image(img_meta, product_id)

        text = get_model_text(product, MODEL_TEXT_FIELD)
        encoding = self.tokenizer(
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
            "product_id":     product_id,
        }

    def _load_image(self, img_meta: dict | None, product_id: str) -> torch.Tensor:
        if img_meta is None:
            raise ValueError(f"Sản phẩm {product_id} không có metadata ảnh đại diện.")

        local_path = img_meta.get(IMAGE_PATH_FIELD)
        if not isinstance(local_path, str) or not local_path.strip():
            raise ValueError(f"Sản phẩm {product_id} không có local_path ảnh hợp lệ.")

        relative_path = Path(local_path)
        if relative_path.parts and relative_path.parts[0] == "images":
            relative_path = Path(*relative_path.parts[1:])

        image_path = self.image_dir / relative_path

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                return cast(torch.Tensor, self.transform(image))
        except OSError as error:
            raise RuntimeError(
                f"Không đọc được ảnh của sản phẩm {product_id}: {image_path}"
            ) from error


# --- DataLoader builder ---

def build_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    split_dir: Path | None = None,
    image_dir: Path | None = None,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho cả 3 split, dùng chung 1 tokenizer instance."""
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    loaders: dict[str, DataLoader] = {}
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
    log.info("=== Dataset smoke test ===")

    loaders = build_dataloaders(
        batch_size=4,
        num_workers=0,
        split_dir=Path(PROCESSED_DIR),
    )

    for split, loader in loaders.items():
        batch = next(iter(loader))
        log.info("[%s]", split)
        log.info("  image          : %s", batch["image"].shape)
        log.info("  input_ids      : %s", batch["input_ids"].shape)
        log.info("  attention_mask : %s", batch["attention_mask"].shape)
        log.info("  product_id     : %s", batch["product_id"])

    log.info("Dataset smoke test passed")