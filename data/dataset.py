import heapq
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
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
    SPLIT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_DIR,
    PRODUCT_ID_FIELD,
)

import config as _config  # noqa: E402

# Xác suất thay text đầy đủ bằng text NGẮN khi train (mô phỏng truy vấn người dùng chỉ gõ vài từ).
SHORT_TEXT_PROB = getattr(_config, "SHORT_TEXT_PROB", 0.5)

log = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from error
    return records


def _valid_images(images: list[dict]) -> list[dict]:
    return [im for im in images if isinstance(im, dict) and im.get(IMAGE_PATH_FIELD)]


def _pos(image: dict) -> float:
    p = image.get(IMAGE_POSITION_FIELD)
    return p if isinstance(p, (int, float)) else float("inf")


def _pick_primary_image(images: list[dict]) -> dict | None:
    """is_featured -> position nhỏ nhất (position bắt đầu từ 1, không phải 0).
    Khớp catalog.primary_image để ảnh train/val/index là cùng một ảnh."""
    images = _valid_images(images)
    if not images:
        return None
    for image in images:
        if image.get(IMAGE_PRIMARY_FIELD):
            return image
    return min(images, key=_pos)


# --- Text ngắn (augmentation lúc train + truy vấn ngắn lúc validate) ---
# text_seg có dạng: "<tên> . màu <màu> . <mô tả> ..." (đã tách từ, chữ thường).
# Người dùng thật chỉ gõ vài từ ("quần jeans xanh"), còn lúc train text dài ~100 từ -> lệch phân phối.
# Các biến thể ngắn dưới đây được cắt từ chính text_seg nên không cần VnCoreNLP lúc train.
SHORT_VARIANTS = ("name", "head4", "head3_color")   # dùng để validate (xác định, không ngẫu nhiên)

_SENT_SPLIT = re.compile(r"\s*\.(?:\s+|$)")
_COLOR_WORDS = frozenset({
    "đen", "trắng", "xám", "xanh", "đỏ", "vàng", "nâu", "hồng", "cam", "tím", "kem", "be", "ghi", "bạc",
    "navy", "kaki", "rêu", "dương", "lá", "đậm", "nhạt", "biển", "hải_quân", "tím_than", "đô", "cổ_vịt",
    "ngọc", "lam", "da", "bò", "tro", "nude", "xanh_lá", "xanh_dương", "xám_đậm", "xám_nhạt",
})


def split_name_color(text_seg: str) -> tuple[str, str]:
    """Tách (tên, màu) từ text_seg. Màu chỉ giữ các từ thuộc từ vựng màu, tối đa 3 từ; không có -> ''."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text_seg) if p.strip()]
    name = parts[0] if parts else text_seg.strip()
    color = ""
    for p in parts[1:3]:
        if p.startswith("màu "):
            words = [w for w in p[4:].split(",")[0].split() if w in _COLOR_WORDS][:3]
            color = " ".join(words)
            break
    return name, color


def make_short_text(text_seg: str, variant: str = "random", rng: Any = None) -> str:
    """Tạo text ngắn từ text_seg. variant: name | head4 | head3_color (xác định) hoặc random (lúc train)."""
    rng = rng or random
    name, color = split_name_color(text_seg)
    toks = name.split()
    if variant == "random":
        variant = rng.choice(("name", "name_color", "head", "head_color", "head_color_mau"))
    k = rng.randint(2, 7)
    if variant == "name":
        out = name
    elif variant == "name_color":
        out = f"{name} màu {color}" if color else name
    elif variant == "head4":
        out = " ".join(toks[:4])
    elif variant == "head3_color":
        out = " ".join(toks[:3] + color.split())
    elif variant == "head":
        out = " ".join(toks[:k])
    elif variant == "head_color":
        out = " ".join(toks[:k] + color.split())
    elif variant == "head_color_mau":
        out = " ".join(toks[:k] + (["màu"] + color.split() if color else []))
    else:
        raise ValueError(f"variant không hợp lệ: {variant!r}")
    return out or name or text_seg


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

    Mọi split đều dùng ảnh đại diện + text sản phẩm (MODEL_TEXT_FIELD), khớp với index và
    giao thức đánh giá.
    """

    def __init__(
        self,
        split: SplitType,
        tokenizer: Any,
        split_dir: Path | None = None,
        image_dir: Path | None = None,
        short_text_prob: float = 0.0,
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        # Chỉ áp dụng cho train: val/test luôn dùng text đầy đủ để số đo so sánh được.
        self.short_text_prob = float(short_text_prob) if split == "train" else 0.0
        self.transform = get_image_transform(split)

        self.split_dir = Path(split_dir) if split_dir else Path(SPLIT_DIR)
        self.image_dir = Path(image_dir) if image_dir else Path(IMAGE_DIR)

        jsonl_path = self.split_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {jsonl_path}. Hãy chạy common.py và build_splits.py trước."
            )

        self.data = load_jsonl(jsonl_path)

        log.info("ViMFDDataset [%s]: %s products", split, len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def _pick_pair(self, product: dict) -> tuple[dict | None, Any]:
        primary = _pick_primary_image(product.get(IMAGE_LIST_FIELD) or [])
        return primary, product.get(MODEL_TEXT_FIELD)

    def __getitem__(self, idx: int) -> dict:
        product = self.data[idx]
        product_id = str(product.get(PRODUCT_ID_FIELD, "<missing id>"))

        img_meta, text = self._pick_pair(product)
        image_tensor = self._load_image(img_meta, product_id)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Sản phẩm {product_id} thiếu trường text hợp lệ "
                f"{MODEL_TEXT_FIELD!r}."
            )
        if self.short_text_prob > 0 and random.random() < self.short_text_prob:
            text = make_short_text(text, "random")
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


class ShortQueryDataset(Dataset):
    """Query văn bản NGẮN xác định (không ngẫu nhiên) cho validate: chỉ trả text + product_id.
    variant: xem SHORT_VARIANTS / make_short_text."""

    def __init__(self, split: SplitType, tokenizer: Any, variant: str,
                 split_dir: Path | None = None) -> None:
        self.tokenizer = tokenizer
        self.variant = variant
        path = (Path(split_dir) if split_dir else Path(SPLIT_DIR)) / f"{split}.jsonl"
        self.data = [p for p in load_jsonl(path)
                     if isinstance(p.get(MODEL_TEXT_FIELD), str) and p[MODEL_TEXT_FIELD].strip()]

    def __len__(self) -> int:
        return len(self.data)

    def text(self, idx: int) -> str:
        return make_short_text(self.data[idx][MODEL_TEXT_FIELD], self.variant)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(self.text(idx), max_length=MAX_TEXT_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "product_id":     str(self.data[idx].get(PRODUCT_ID_FIELD, "<missing id>")),
        }


# --- Sampler: không xếp sản phẩm cùng nhóm vào một batch ---

class GroupBatchSampler:
    """Batch sampler cho InfoNCE: mỗi batch có TỐI ĐA MỘT sản phẩm từ mỗi group_id.

    Sản phẩm cùng nhóm (cùng mẫu khác màu/SKU, ảnh trùng hệt) là "negative sai"; nếu chung batch thì
    loss đẩy chúng ra xa nhau. Một nhóm lớn hơn số batch/epoch chỉ góp tối đa số_batch sản phẩm mỗi
    epoch (chọn ngẫu nhiên, khác nhau giữa các epoch), phần còn lại nghỉ epoch đó.
    Số batch cố định (len) để scheduler biết tổng số bước.
    """

    def __init__(self, groups, batch_size: int, seed: int = 42):
        self.groups = list(groups)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        sizes = Counter(self.groups)
        b = max(1, len(self.groups) // batch_size)
        for _ in range(10):  # nhóm lớn bị cắt -> ít sản phẩm hơn -> ít batch hơn
            nb = max(1, sum(min(n, b) for n in sizes.values()) // batch_size)
            if nb == b:
                break
            b = nb
        self.num_batches = b
        self.n_used = sum(min(n, b) for n in sizes.values())

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        B, bs = self.num_batches, self.batch_size
        by_group = defaultdict(list)
        for i, g in enumerate(self.groups):
            by_group[g].append(i)
        items = []
        for idx in by_group.values():
            rng.shuffle(idx)
            items.append(idx[:B])
        rng.shuffle(items)
        items.sort(key=len, reverse=True)       # nhóm lớn xếp trước (ổn định: giữ thứ tự ngẫu nhiên)

        batches = [[] for _ in range(B)]
        heap = [(0, rng.random(), b) for b in range(B)]   # (độ đầy, hoà ngẫu nhiên, batch)
        heapq.heapify(heap)
        for idx in items:
            taken = []
            for i in idx:
                if not heap:
                    break
                ln, _, b = heapq.heappop(heap)
                if ln >= bs:                    # batch ít đầy nhất đã đầy -> mọi batch đã đầy
                    heap.clear()
                    break
                batches[b].append(i)
                taken.append((ln + 1, rng.random(), b))
            for t in taken:                      # chỉ trả lại sau khi xong nhóm -> 1 sản phẩm/nhóm/batch
                heapq.heappush(heap, t)
        order = [b for b in batches if len(b) >= 2]
        rng.shuffle(order)
        yield from order


# --- DataLoader builder ---

def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    split_dir: Path | None = None,
    image_dir: Path | None = None,
    group_map: dict | None = None,
    short_text_prob: float | None = None,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho cả 3 split, dùng chung 1 tokenizer instance.

    group_map = {product_id: group_id}: nếu có, train loader dùng GroupBatchSampler.
    short_text_prob: xác suất dùng text ngắn ở train (None -> config.SHORT_TEXT_PROB, mặc định 0.5)."""
    if short_text_prob is None:
        short_text_prob = getattr(_config, "SHORT_TEXT_PROB", SHORT_TEXT_PROB)
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = ViMFDDataset(
            split=split,
            tokenizer=tokenizer,
            split_dir=split_dir,
            image_dir=image_dir,
            short_text_prob=short_text_prob,
        )
        if split == "train" and group_map:
            groups = [group_map.get(str(p.get(PRODUCT_ID_FIELD)), "pid:" + str(p.get(PRODUCT_ID_FIELD)))
                      for p in dataset.data]
            sampler = GroupBatchSampler(groups, batch_size)
            log.info("GroupBatchSampler: %d nhóm, %d batch/epoch, %d/%d sản phẩm mỗi epoch",
                     len(set(groups)), len(sampler), sampler.n_used, len(groups))
            loaders[split] = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            continue
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
        split_dir=Path(SPLIT_DIR),
    )

    for split, loader in loaders.items():
        batch = next(iter(loader))
        log.info("[%s]", split)
        log.info("  image          : %s", batch["image"].shape)
        log.info("  input_ids      : %s", batch["input_ids"].shape)
        log.info("  attention_mask : %s", batch["attention_mask"].shape)
        log.info("  product_id     : %s", batch["product_id"])

    sample = loaders["val"].dataset.data[0][MODEL_TEXT_FIELD]
    for v in SHORT_VARIANTS + ("random",):
        log.info("  short[%s]: %r", v, make_short_text(sample, v))
    log.info("Dataset smoke test passed")