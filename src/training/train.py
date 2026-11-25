# src/training/train.py
# Hướng đánh giá: Text → Image (text đầy đủ + text NGẮN)  và  Image → Image
#   - Gallery  : ảnh đại diện của mọi sản phẩm (val set)
#   - T→I query: text mô tả sản phẩm
#   - I→I query: MỘT ẢNH KHÁC (ảnh phụ) của cùng sản phẩm  (không dùng lại ảnh đại diện,
#                nếu không query sẽ khớp chính nó và R@1 ≈ 100%)

import os
import sys
import csv
import json
import math
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "data"))

from src.models.model import ViMFRModel
from src.models.loss import InfoNCELoss

from dataset import (  # noqa: E402
    build_dataloaders, load_jsonl, ShortQueryDataset, SHORT_VARIANTS,
)

import config as cfg  # noqa: E402
from config import (  # noqa: E402
    BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, WARMUP_EPOCHS,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
    GRAD_CLIP_NORM, LOGIT_SCALE_MAX,
    CHECKPOINT_BEST, CHECKPOINT_LATEST,
    CKPT_MODEL_KEY, CKPT_LOGIT_SCALE_KEY, CKPT_EPOCH_KEY, CKPT_VAL_LOSS_KEY,
    PROCESSED_DIR, SPLIT_DIR, IMAGE_DIR,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    IMAGE_LIST_FIELD, IMAGE_PRIMARY_FIELD, IMAGE_POSITION_FIELD,
    PRODUCT_ID_FIELD,
)

IMAGE_PATH_FIELD = cfg.IMAGE_PATH_FIELD


# ------------------------------------------------------------------
# Dataset ảnh phụ: query cho Image → Image
# ------------------------------------------------------------------
def _pos(img: dict) -> float:
    p = img.get(IMAGE_POSITION_FIELD)
    return p if isinstance(p, (int, float)) else float("inf")


def _pick_alt_image(product: dict):
    """Trả về 1 ảnh KHÔNG phải ảnh đại diện (None nếu sản phẩm chỉ có 1 ảnh).
    Cách chọn ảnh đại diện giống build_index.py: is_featured -> position nhỏ nhất."""
    images = product.get(IMAGE_LIST_FIELD)
    if not isinstance(images, list):
        return None
    imgs = [im for im in images if isinstance(im, dict) and im.get(IMAGE_PATH_FIELD)]
    if len(imgs) < 2:
        return None
    primary = next((im for im in imgs if im.get(IMAGE_PRIMARY_FIELD)), None)
    if primary is None:
        primary = min(imgs, key=_pos)
    alts = sorted((im for im in imgs if im is not primary), key=_pos)
    return alts[0]


def _resolve_path(rel: str):
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    for root in (Path(PROCESSED_DIR), Path(IMAGE_DIR).parent, Path(IMAGE_DIR)):
        if (root / p).exists():
            return root / p
    return None


class AltImageDataset(Dataset):
    """Mỗi phần tử: 1 ảnh phụ của 1 sản phẩm, trả {"image", "product_id"}
    (cùng format với dataset chính nên dùng lại được trong evaluate.py)."""

    def __init__(self, json_path):
        products = load_jsonl(Path(json_path))
        # QUAN TRỌNG: transform này phải GIỐNG transform ảnh của val/test trong dataset.py
        self.transform = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
        self.samples = []
        for p in products:
            alt = _pick_alt_image(p)
            if alt is None:
                continue
            path = _resolve_path(alt[IMAGE_PATH_FIELD])
            if path is not None:
                self.samples.append((str(p[PRODUCT_ID_FIELD]), path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pid, path = self.samples[i]
        with Image.open(path) as im:
            return {"image": self.transform(im.convert("RGB")), "product_id": pid}


# ------------------------------------------------------------------
# Optimizer / scheduler
# ------------------------------------------------------------------
def get_optimizer(model: ViMFRModel) -> AdamW:
    backbone_params = [
        p for p in list(model.visual_encoder.parameters())
        + list(model.text_encoder.parameters())
        if p.requires_grad
    ]
    head_params = (
        list(model.visual_proj.parameters())
        + list(model.text_proj.parameters())
    )
    return AdamW([
        {"params": backbone_params,     "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params,         "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
        {"params": [model.logit_scale], "lr": LR_HEAD,     "weight_decay": 0.0},
    ])


def get_scheduler(
    optimizer,
    steps_per_epoch: int,
    num_epochs: int = NUM_EPOCHS,
) -> LambdaLR:
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(WARMUP_EPOCHS * steps_per_epoch, total_steps // 2)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Validation: loss + Recall@K cho T→I và I→I
# ------------------------------------------------------------------
def _to_str_list(pids):
    if torch.is_tensor(pids):
        pids = pids.tolist()
    return [str(x) for x in pids]


def load_group_map(split_dir):
    """{product_id: group_id} từ split.json do build_splits.py ghi: {id: {"split":..., "group_id":...}}."""
    path = Path(split_dir) / "split.json"
    if not path.exists():
        print(f"  ⚠ Không thấy {path}: mức nhóm sẽ trùng mức sản phẩm.")
        return {}
    gm = {}
    for pid, v in json.load(open(path, encoding="utf-8")).items():
        g = v.get("group_id") if isinstance(v, dict) else None
        if g is not None:
            gm[str(pid)] = str(g)
    if gm and set(gm.values()) <= {"train", "val", "test"}:
        print("  ⚠ group_id trong split.json chỉ là tên split (chạy build_splits.py bản mới): bỏ qua.")
        return {}
    return gm


def _group_codes(ids, group_map):
    codes = {}
    return torch.tensor([codes.setdefault(group_map.get(p, "pid:" + p), len(codes)) for p in ids])


def _ranks(sim, tgt):
    """Hạng 0-based của cột đích; điểm HÒA tính là thua (ảnh trùng hệt cho cùng điểm)."""
    s_t = sim.gather(1, tgt)
    return (sim > s_t).sum(1) + ((sim == s_t).sum(1) - 1)


def _group_ranks(sim, q_grp, g_grp):
    """Hạng 0-based của sản phẩm cùng nhóm đứng cao nhất (sản phẩm cùng nhóm đều là đáp án đúng)."""
    same = q_grp[:, None] == g_grp[None, :]
    best = sim.masked_fill(~same, float("-inf")).max(1).values[:, None]
    return ((sim >= best) & ~same).sum(1)


def _metrics(sim, tgt, q_grp, g_grp, ks):
    out = {}
    for tag, r in (("p", _ranks(sim, tgt)), ("g", _group_ranks(sim, q_grp, g_grp))):
        for k in ks:
            out[(tag, k)] = (r < k).float().mean().item()
    return out


@torch.no_grad()
def validate(
    model,
    loader,
    alt_loader,
    loss_fn,
    device,
    use_amp,
    group_map=None,
    ks=(1, 5, 10),
    max_batches=None,
    short_loaders=None,
    cat_map=None,
):
    """Trả (val_loss, t2i, i2i); mỗi dict có khoá ('p', k) = mức sản phẩm, ('g', k) = mức nhóm.
    Nếu truyền short_loaders ({tên_biến_thể: DataLoader text ngắn}) thì trả thêm phần tử thứ 4:
    {tên_biến_thể: dict metric T→I với query ngắn, thêm ('c', 10) = tỉ lệ top-10 đúng danh mục
    khi có cat_map {product_id: category}}."""
    group_map = group_map or {}
    model.eval()
    val_loss, n_batches = 0.0, 0
    all_img, all_txt, gallery_ids = [], [], []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pixel_values   = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            img, txt, logit_scale = model(pixel_values, input_ids, attention_mask)
        img, txt = img.float(), txt.float()

        val_loss += loss_fn(img, txt, logit_scale).item()
        n_batches += 1
        all_img.append(img.cpu())
        all_txt.append(txt.cpu())
        gallery_ids.extend(_to_str_list(batch["product_id"]))

    gallery = torch.cat(all_img)   # [N, 256] ảnh đại diện = gallery
    txt = torch.cat(all_txt)
    g_grp = _group_codes(gallery_ids, group_map)

    # --- Text → Image ---
    sim = txt @ gallery.T
    tgt = torch.arange(len(gallery_ids))[:, None]
    t2i = _metrics(sim, tgt, g_grp, g_grp, ks)

    id2idx = {pid: i for i, pid in enumerate(gallery_ids)}

    # --- Image → Image (query = ảnh phụ của cùng sản phẩm) ---
    i2i = None
    if alt_loader is not None:
        q_embs, q_ids = [], []
        for batch_index, batch in enumerate(alt_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                e = model.encode_image(batch["image"].to(device))
            q_embs.append(e.float().cpu())
            q_ids.extend(_to_str_list(batch["product_id"]))
        q = torch.cat(q_embs)

        keep = [j for j, pid in enumerate(q_ids) if pid in id2idx]
        if keep:
            gt = torch.tensor([id2idx[q_ids[j]] for j in keep]).unsqueeze(1)
            sim_q = q[keep] @ gallery.T
            i2i = _metrics(sim_q, gt, g_grp[gt.squeeze(1)], g_grp, ks)

    if short_loaders is None:
        return val_loss / max(1, n_batches), t2i, i2i

    # --- Text NGẮN → Image: query = tên / 4 từ đầu / 3 từ đầu + màu (giống người dùng thật) ---
    cat_codes = None
    if cat_map:
        codes = {}
        cat_codes = torch.tensor([codes.setdefault(cat_map.get(p), len(codes)) for p in gallery_ids])
    short = {}
    for name, sl in short_loaders.items():
        q_embs, q_ids = [], []
        for batch_index, batch in enumerate(sl):
            if max_batches is not None and batch_index >= max_batches:
                break
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                e = model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            q_embs.append(e.float().cpu())
            q_ids.extend(_to_str_list(batch["product_id"]))
        if not q_embs:
            continue
        q = torch.cat(q_embs)
        keep = [j for j, pid in enumerate(q_ids) if pid in id2idx]
        if not keep:
            continue
        gt = torch.tensor([id2idx[q_ids[j]] for j in keep]).unsqueeze(1)
        sim_q = q[keep] @ gallery.T
        m = _metrics(sim_q, gt, g_grp[gt.squeeze(1)], g_grp, ks)
        if cat_codes is not None:
            top = sim_q.topk(min(10, sim_q.size(1)), dim=1).indices
            m[("c", 10)] = (cat_codes[top] == cat_codes[gt.squeeze(1)][:, None]).float().mean().item()
        short[name] = m

    return val_loss / max(1, n_batches), t2i, i2i, short


def save_ckpt(path, model, epoch, val_loss, extra=None):
    ckpt = {
        CKPT_MODEL_KEY:       model.state_dict(),
        CKPT_LOGIT_SCALE_KEY: model.logit_scale.data,
        CKPT_EPOCH_KEY:       epoch,
        CKPT_VAL_LOSS_KEY:    val_loss,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_train_batches=None,
    max_val_batches=None,
    output_dir=None,
    patience=5,
    select="mix",
    use_group_sampler=True,
):
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if batch_size < 2:
        raise ValueError("InfoNCE training requires batch_size >= 2.")
    if max_train_batches is not None and max_train_batches < 1:
        raise ValueError("max_train_batches must be at least 1.")
    if max_val_batches is not None and max_val_batches < 1:
        raise ValueError("max_val_batches must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(
        f"Device: {device} | AMP: {use_amp} | batch_size: {batch_size} "
        f"| epochs: {num_epochs}"
    )
    if max_train_batches is not None or max_val_batches is not None:
        print(
            "PILOT mode: "
            f"max_train_batches={max_train_batches}, "
            f"max_val_batches={max_val_batches}"
        )

    model = ViMFRModel().to(device)
    loss_fn = InfoNCELoss()

    group_map = load_group_map(Path(SPLIT_DIR))
    print(f"group_id: {len(set(group_map.values())):,} nhóm cho {len(group_map):,} sản phẩm")
    loaders = build_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        split_dir=Path(SPLIT_DIR),
        image_dir=Path(IMAGE_DIR),
        group_map=group_map if use_group_sampler else None,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]
    print(f"Train: {len(train_loader)} batch/epoch "
          f"({'GroupBatchSampler' if use_group_sampler and group_map else 'shuffle thường'})"
          f" | text ngắn khi train: p={train_loader.dataset.short_text_prob}")

    # Query văn bản NGẮN cho validate (giống người dùng thật) + nhãn danh mục để đo Category P@10
    val_ds = val_loader.dataset
    cat_field = getattr(cfg, "CATEGORY_FIELD", "category")
    cat_map = {str(p.get(PRODUCT_ID_FIELD)): p.get(cat_field) for p in val_ds.data}
    short_loaders = {}
    for v in SHORT_VARIANTS:
        sds = ShortQueryDataset("val", val_ds.tokenizer, v, split_dir=Path(SPLIT_DIR))
        short_loaders[v] = DataLoader(sds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
        print(f"  short[{v}] ví dụ: {sds.text(0)!r}")

    # Query ảnh phụ cho I→I
    alt_ds = AltImageDataset(Path(SPLIT_DIR) / "val.jsonl")
    print(f"I→I: {len(alt_ds)} sản phẩm val có ảnh phụ làm query")
    alt_loader = None
    if len(alt_ds) > 0:
        alt_loader = DataLoader(alt_ds, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS)
    else:
        print("  ⚠ Không tìm được ảnh phụ nào -> bỏ qua I→I. "
              "Kiểm tra IMAGE_PATH_FIELD / đường dẫn ảnh trong config.")

    optimizer = get_optimizer(model)
    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    scheduler = get_scheduler(
        optimizer,
        steps_per_epoch=max(1, steps_per_epoch),
        num_epochs=num_epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_log_scale = math.log(LOGIT_SCALE_MAX)
    print(f"Sanity: loss ban đầu nên ≈ log({batch_size}) = {math.log(batch_size):.3f}")

    if output_dir is None:
        log_path = Path("results/loss_log.csv")
        best_checkpoint = Path(CHECKPOINT_BEST)
        latest_checkpoint = Path(CHECKPOINT_LATEST)
    else:
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Run output directory is not empty: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "loss_log.csv"
        best_checkpoint = output_dir / "best.pt"
        latest_checkpoint = output_dir / "latest.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "logit_scale",
            "t2i_pR1", "t2i_pR5", "t2i_pR10", "t2i_gR1", "t2i_gR5", "t2i_gR10",
            "i2i_pR1", "i2i_pR5", "i2i_pR10", "i2i_gR1", "i2i_gR5", "i2i_gR10",
        ] + [c for v in SHORT_VARIANTS for c in (f"short_{v}_gR10", f"short_{v}_catP10")])

    best_score, bad_epochs = -1.0, 0

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss, n_batches = 0.0, 0

        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            pixel_values   = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if pixel_values.size(0) < 2:
                continue

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                image_embeds, text_embeds, logit_scale = model(
                    pixel_values, input_ids, attention_mask
                )
            loss = loss_fn(image_embeds.float(), text_embeds.float(), logit_scale)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                model.logit_scale.clamp_(0, max_log_scale)

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        # --- Validation ---
        val_loss, t2i, i2i, short = validate(
            model,
            val_loader,
            alt_loader,
            loss_fn,
            device,
            use_amp,
            group_map=group_map,
            max_batches=max_val_batches,
            short_loaders=short_loaders,
            cat_map=cat_map,
        )
        scale_val = model.logit_scale.exp().item()

        def fmt(m):
            return (f"sp {m[('p',1)]:.3f}/{m[('p',5)]:.3f}/{m[('p',10)]:.3f}"
                    f" | nhóm {m[('g',1)]:.3f}/{m[('g',5)]:.3f}/{m[('g',10)]:.3f}")
        msg = (f"Epoch {epoch:03d}/{num_epochs} | train: {train_loss:.4f} | "
               f"val: {val_loss:.4f} | scale: {scale_val:.1f}\n"
               f"   T→I R@1/5/10: {fmt(t2i)}")
        if i2i is not None:
            msg += f"\n   I→I R@1/5/10: {fmt(i2i)}"
        if short:
            msg += "\n   T→I NGẮN (nhóm R@10 / đúng danh mục P@10): " + " | ".join(
                f"{v} {m[('g', 10)]:.3f}/{m.get(('c', 10), float('nan')):.3f}" for v, m in short.items())
        print(msg)

        zero = {(t, k): 0.0 for t in "pg" for k in (1, 5, 10)}
        i2i_row = i2i if i2i is not None else zero
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(train_loss, 4), round(val_loss, 4), round(scale_val, 2)]
                + [round(t2i[(t, k)], 4) for t in "pg" for k in (1, 5, 10)]
                + [round(i2i_row[(t, k)], 4) for t in "pg" for k in (1, 5, 10)]
                + [round(short.get(v, {}).get(key, 0.0), 4) for v in SHORT_VARIANTS
                   for key in (("g", 10), ("c", 10))])

        save_ckpt(latest_checkpoint, model, epoch, val_loss)

        # Chọn best: mặc định T→I R@10 mức NHÓM (I→I gần như phẳng từ epoch 1 vì CLIP ảnh đã tốt sẵn,
        # và val lệch phân phối làm nó nhiễu). --select avg = trung bình T→I và I→I mức nhóm.
        # mix (mặc định) = trung bình T→I text đầy đủ và T→I text NGẮN (mức nhóm R@10); short = chỉ text ngắn.
        short_score = (sum(m[("g", 10)] for m in short.values()) / len(short)) if short else t2i[("g", 10)]
        if select == "avg" and i2i is not None:
            score = (t2i[("g", 10)] + i2i[("g", 10)]) / 2
        elif select == "short":
            score = short_score
        elif select == "mix":
            score = (t2i[("g", 10)] + short_score) / 2
        else:
            score = t2i[("g", 10)]
        if score > best_score:
            best_score, bad_epochs = score, 0
            save_ckpt(best_checkpoint, model, epoch, val_loss, extra={"val_recall10": score})
            print(f"  ✓ Saved best checkpoint (val {select} R@10 nhóm = {score:.4f})")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"  Early stopping: {bad_epochs} epoch không cải thiện (best = {best_score:.4f}).")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViMFR.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-group-sampler", action="store_true",
                        help="tắt sampler theo nhóm (shuffle thường)")
    parser.add_argument("--patience", type=int, default=5, help="early stopping; 0 = tắt")
    parser.add_argument("--select", choices=["t2i_group", "avg", "short", "mix"], default="mix",
                        help="tiêu chí chọn best checkpoint: t2i_group = T→I text đầy đủ; avg = T→I và I→I; "
                             "short = T→I text ngắn; mix = trung bình T→I đầy đủ và ngắn (mặc định)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Separate directory for loss_log.csv and checkpoints.",
    )
    args = parser.parse_args()
    train(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        output_dir=args.output_dir,
        patience=args.patience,
        select=args.select,
        use_group_sampler=not args.no_group_sampler,
    )