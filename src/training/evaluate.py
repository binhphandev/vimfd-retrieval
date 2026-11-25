# Đánh giá trên EVALUATION_SPLIT (mặc định test), gallery = image index (ảnh đại diện).
#
# Hướng đánh giá:
#   Text → Image : query = text_seg đầy đủ ("full") và 3 biến thể NGẮN của dataset.SHORT_VARIANTS
#                  (name / head4 / head3_color), giống cách người dùng gõ thật.
#   Image → Image: query = MỘT ảnh phụ của mỗi sản phẩm (đúng AltImageDataset trong train.py).
#                  Báo hai bản: tất cả query, và bỏ query có ảnh phụ trùng hệt ảnh đại diện
#                  (điểm cosine với chính sản phẩm đó >= IDENTICAL_THR) vì các query này quá dễ.
# Hai kiểu gallery: "full" (toàn bộ index) và "test-only" (chỉ sản phẩm của split đánh giá).
#
# Cách tính khớp với validate() trong train.py:
#   - Mức SẢN PHẨM: hạng của đúng sản phẩm đó; điểm HÒA tính là THUA (ảnh trùng hệt cho cùng điểm).
#   - Mức NHÓM    : sản phẩm cùng group_id (split.json) đều là đáp án đúng; hạng = hạng của sản phẩm
#                   cùng nhóm đứng cao nhất. Đây là số nên dùng để báo cáo, vì có sản phẩm trùng ảnh.
#   - Chỉ số: Recall@1/5/10, MRR@10, NDCG@10 (MRR và NDCG giới hạn trong top-10).
#   - Khoảng tin cậy 95%: bootstrap theo NHÓM (evaluation_metrics.cluster_bootstrap_mean_ci).
#
# Trước khi chạy: build lại index từ ĐÚNG checkpoint (build_index.py). Script so fingerprint
# checkpoint với index_meta và dừng nếu lệch (dùng --allow-stale để bỏ qua).
#
# Chạy từ thư mục gốc dự án (đặt cùng thư mục với evaluation_metrics.py):
#     python -m <đường.dẫn.module>.evaluate [--checkpoint ...] [--bootstrap 2000]

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.append(str(Path(__file__).resolve().parent))  # evaluation_metrics.py nằm cạnh file này

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import config as cfg
from dataset import SHORT_VARIANTS, ShortQueryDataset, load_jsonl
from evaluation_metrics import cluster_bootstrap_mean_ci, rank_metrics
from src.models.model import ViMFRModel, load_state_dict_compat
from src.serving.catalog import ckpt_fingerprint, meta_path
from src.training.train import AltImageDataset, load_group_map

TOP_K = 10
IDENTICAL_THR = 0.9995   # ảnh phụ coi là trùng hệt ảnh đại diện nếu cosine với chính sản phẩm >= ngưỡng
TIE_EPS = 1e-6           # sai số float khi coi hai điểm là hòa (ảnh trùng hệt)
METRICS = ["Recall@1", "Recall@5", "Recall@10", f"MRR@{TOP_K}", f"NDCG@{TOP_K}"]


# ------------------------------------------------------------------
# Dataset query
# ------------------------------------------------------------------
class FullTextQueryDataset(ShortQueryDataset):
    """Query text ĐẦY ĐỦ (text_seg nguyên bản), chỉ text, không tải ảnh."""

    def text(self, idx: int) -> str:
        return self.data[idx][cfg.MODEL_TEXT_FIELD]


class SafeAltImageDataset(AltImageDataset):
    """AltImageDataset + cờ "ok": ảnh hỏng không làm crash mà bị bỏ khỏi phép đo (và được đếm)."""

    def __getitem__(self, i):
        try:
            item = super().__getitem__(i)
            item["ok"] = 1
        except Exception:
            pid, _ = self.samples[i]
            item = {"image": torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), "product_id": pid, "ok": 0}
        return item


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load_model(checkpoint_path: Path, device) -> ViMFRModel:
    model = ViMFRModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_state_dict_compat(model, ckpt[cfg.CKPT_MODEL_KEY])
    model.to(device).eval()
    val_loss = ckpt.get(cfg.CKPT_VAL_LOSS_KEY)
    vl = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "?"
    print(f"Loaded checkpoint: {Path(checkpoint_path).name} | epoch={ckpt.get(cfg.CKPT_EPOCH_KEY, '?')}, val_loss={vl}")
    return model


def check_index(checkpoint_path: Path, allow_stale: bool) -> dict:
    """Index phải được build từ đúng checkpoint này (so fingerprint sha256, giống engine.py)."""
    mp = meta_path(cfg.FAISS_IMAGE_INDEX_FILE)
    if not Path(cfg.FAISS_IMAGE_INDEX_FILE).exists():
        raise SystemExit(f"Chưa có {cfg.FAISS_IMAGE_INDEX_FILE}. Chạy build_index.py trước.")
    problem = None
    meta = {}
    if not mp.exists():
        problem = f"Không thấy {mp.name}: không kiểm tra được index build từ checkpoint nào. Hãy build lại index."
    else:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint") != ckpt_fingerprint(checkpoint_path):
            problem = (f"Index được build từ checkpoint khác ({meta.get('checkpoint')}) so với "
                       f"{Path(checkpoint_path).name}. Hãy build lại index.")
    if problem:
        if not allow_stale:
            raise SystemExit("✗ " + problem + "  (--allow-stale để bỏ qua, số liệu sẽ KHÔNG đáng tin)")
        print("⚠ " + problem)
    return meta


def load_gallery(index_path, mapping_path):
    """Embedding ảnh đại diện + id sản phẩm từ FAISS index (chỉ dùng được với IndexFlat*)."""
    index = faiss.read_index(str(index_path))
    emb = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    ids = [str(mapping[str(i)]) for i in range(index.ntotal)]
    print(f"Loaded index: {Path(index_path).name} ({index.ntotal} vectors)")
    return emb, ids


def group_codes(ids, group_map) -> np.ndarray:
    """Mã số nhóm cho từng id. Không có group_id thì mỗi sản phẩm là một nhóm riêng (mức nhóm = mức sản phẩm)."""
    codes: dict = {}
    return np.array([codes.setdefault(group_map.get(p, "pid:" + p), len(codes)) for p in ids], dtype=np.int64)


@torch.no_grad()
def encode(loader, encode_fn, keys, device, desc):
    embs, ids, oks = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        e = encode_fn(*[batch[k].to(device) for k in keys])
        embs.append(e.float().cpu().numpy())
        pids = [str(x) for x in batch["product_id"]]
        ids.extend(pids)
        ok = batch.get("ok")
        oks.extend(ok.tolist() if torch.is_tensor(ok) else [1] * len(pids))
    emb = np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, cfg.EMBED_DIM), np.float32)
    return {"emb": emb, "ids": ids, "ok": oks}


# ------------------------------------------------------------------
# Xếp hạng + tổng hợp
# ------------------------------------------------------------------
def rank_queries(q, gal, tgt, g_grp, chunk: int = 256):
    """Trả về (hạng mức sản phẩm, hạng mức nhóm, cosine với đúng sản phẩm); hạng 0-based.
    q: [Q,D], gal: [G,D], tgt: [Q] chỉ số hàng đích trong gal, g_grp: [G] mã nhóm của gallery."""
    q_grp = g_grp[tgt]
    prod, grp, self_sim = [], [], []
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ gal.T
        t = tgt[s:s + chunk]
        s_t = sim[np.arange(len(t)), t][:, None]
        # mức sản phẩm: số điểm cao hơn + số điểm hòa (trừ chính nó) -> hòa tính là thua
        prod.append((sim > s_t + TIE_EPS).sum(1) + (np.abs(sim - s_t) <= TIE_EPS).sum(1) - 1)
        # mức nhóm: số sản phẩm KHÁC nhóm có điểm >= điểm tốt nhất trong nhóm đúng
        same = q_grp[s:s + chunk][:, None] == g_grp[None, :]
        best = np.where(same, sim, -np.inf).max(1)[:, None]
        grp.append(((sim >= best - TIE_EPS) & ~same).sum(1))
        self_sim.append(s_t[:, 0])
    return np.concatenate(prod), np.concatenate(grp), np.concatenate(self_sim)


def summarize(rank_prod, rank_grp, clusters, n_bootstrap, seed):
    """Trung bình + CI95 (bootstrap theo nhóm) cho từng chỉ số, ở cả mức sản phẩm và mức nhóm."""
    out = {"n_queries": int(len(rank_prod)), "n_groups": int(len(set(clusters.tolist())))}
    for level, ranks in (("product", rank_prod), ("group", rank_grp)):
        out[level] = {}
        for name, vals in rank_metrics(ranks, TOP_K).items():
            if n_bootstrap > 0:
                r = cluster_bootstrap_mean_ci(vals, clusters, n_bootstrap=n_bootstrap, seed=seed)
                out[level][name] = {k: r[k] for k in ("mean", "ci95_low", "ci95_high")}
            else:
                out[level][name] = {"mean": float(vals.mean())}
    return out


def evaluate_queries(queries, gallery, *, identical_view, n_bootstrap, seed):
    """Chấm một tập query trên một gallery. identical_view=True thì thêm bản bỏ query trùng hệt ảnh đại diện."""
    pos = {p: i for i, p in enumerate(gallery["ids"])}
    keep, tgt = [], []
    n_unreadable = n_missing = 0
    for j, pid in enumerate(queries["ids"]):
        if not queries["ok"][j]:
            n_unreadable += 1
        elif pid not in pos:
            n_missing += 1
        else:
            keep.append(j)
            tgt.append(pos[pid])
    result = {"gallery": len(gallery["ids"]),
              "skipped": {"not_in_gallery": n_missing, "unreadable": n_unreadable}, "views": {}}
    if not keep:
        return result
    keep, tgt = np.array(keep), np.array(tgt)
    rp, rg, self_sim = rank_queries(queries["emb"][keep], gallery["emb"], tgt, gallery["grp"])
    clusters = gallery["grp"][tgt]
    result["views"]["all"] = summarize(rp, rg, clusters, n_bootstrap, seed)
    if identical_view:
        m = self_sim < IDENTICAL_THR
        result["n_identical"] = int((~m).sum())
        if m.any():
            result["views"]["no_identical"] = summarize(rp[m], rg[m], clusters[m], n_bootstrap, seed)
    return result


# ------------------------------------------------------------------
# Báo cáo
# ------------------------------------------------------------------
def _fmt(d):
    s = f"{d['mean']:.4f}"
    return s + (f" [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]" if "ci95_low" in d else "")


def format_block(title, res, view_names):
    lines = [f"=== {title} | gallery = {res['gallery']} sản phẩm ==="]
    sk = res["skipped"]
    skip = [f"{k}: {v}" for k, v in (("không có trong gallery", sk["not_in_gallery"]),
                                     ("ảnh đọc lỗi", sk["unreadable"])) if v]
    if skip:
        lines.append("  Bỏ qua: " + " | ".join(skip))
    if not res["views"]:
        lines.append("  Không có query nào dùng được.")
        return lines
    for key, label in view_names:
        v = res["views"].get(key)
        if v is None:
            continue
        extra = f", đã bỏ {res['n_identical']} query trùng hệt ảnh đại diện" if key == "no_identical" else ""
        lines.append(f"  [{label}] {v['n_queries']} query / {v['n_groups']} nhóm{extra}")
        lines.append(f"  {'Chỉ số':<10}{'mức sản phẩm':>34}{'mức NHÓM (nên báo cáo)':>40}")
        for m in METRICS:
            lines.append(f"  {m:<10}{_fmt(v['product'][m]):>34}{_fmt(v['group'][m]):>40}")
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=cfg.CHECKPOINT_BEST)
    ap.add_argument("--bootstrap", type=int, default=2000, help="số lần bootstrap theo nhóm cho CI95; 0 = tắt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--allow-stale", action="store_true", help="bỏ qua kiểm tra fingerprint index (không khuyên dùng)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt_path = Path(args.checkpoint)
    split = cfg.EVALUATION_SPLIT
    split_dir = Path(cfg.SPLIT_DIR)
    split_path = split_dir / f"{split}.jsonl"
    print(f"Split đánh giá: {split} ({split_path})")

    meta = check_index(ckpt_path, args.allow_stale)
    model = load_model(ckpt_path, device)
    gal_emb, gal_ids = load_gallery(cfg.FAISS_IMAGE_INDEX_FILE, cfg.FAISS_IMAGE_MAPPING_FILE)

    group_map = load_group_map(split_dir)
    grp = group_codes(gal_ids, group_map)
    full = {"emb": gal_emb, "ids": gal_ids, "grp": grp}

    eval_ids = {str(p[cfg.PRODUCT_ID_FIELD]) for p in load_jsonl(split_path)}
    rows = np.array([i for i, p in enumerate(gal_ids) if p in eval_ids], dtype=np.int64)
    test_only = {"emb": gal_emb[rows], "ids": [gal_ids[i] for i in rows], "grp": grp[rows]}
    galleries = {"full": full, "test-only": test_only}
    print(f"Gallery full={len(gal_ids)} | test-only={len(rows)}/{len(eval_ids)} sản phẩm của split | "
          f"{len(set(grp.tolist()))} nhóm")

    # ---- encode query (một lần, dùng cho cả hai gallery) ----
    tok = AutoTokenizer.from_pretrained(cfg.PHOBERT_MODEL)
    bs = cfg.EVAL_BATCH_SIZE
    directions = []   # (tên, query, có bản bỏ trùng hệt?)
    text_sets = {"full": FullTextQueryDataset(split, tok, "full", split_dir=split_dir)}
    for v in SHORT_VARIANTS:
        text_sets[v] = ShortQueryDataset(split, tok, v, split_dir=split_dir)
    for name, ds in text_sets.items():
        q = encode(DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0),
                   model.encode_text, ["input_ids", "attention_mask"], device, f"T→I [{name}]")
        directions.append((f"Text[{name}] → Image", q, False))

    alt_ds = SafeAltImageDataset(split_path)
    print(f"I→I: {len(alt_ds)} sản phẩm có ảnh phụ (mỗi sản phẩm một ảnh phụ)")
    if len(alt_ds):
        q = encode(DataLoader(alt_ds, batch_size=bs, shuffle=False, num_workers=0),
                   model.encode_image, ["image"], device, "I→I")
        directions.append(("Image → Image", q, True))
    else:
        print("⚠ Không có ảnh phụ nào: bỏ qua Image → Image.")

    # ---- chấm ----
    header = [
        f"Checkpoint : {ckpt_path.name} (fingerprint {meta.get('fingerprint', '?')}, index build {meta.get('built_at', '?')})",
        f"Split      : {split} | bootstrap theo nhóm: {args.bootstrap} lần (seed {args.seed})",
        "Hạng       : điểm hòa tính là thua; mức nhóm = sản phẩm cùng group_id đều đúng",
    ]
    report, results = list(header), {}
    for g_name, gallery in galleries.items():
        results[g_name] = {}
        for d_name, q, ident in directions:
            res = evaluate_queries(q, gallery, identical_view=ident, n_bootstrap=args.bootstrap, seed=args.seed)
            results[g_name][d_name] = res
            views = [("all", "tất cả query")] + ([("no_identical", "bỏ ảnh phụ trùng ảnh đại diện")] if ident else [])
            block = format_block(f"{d_name} | gallery: {g_name}", res, views)
            report += [""] + block

    print("\n" + "\n".join(report))

    # ---- lưu (giữ bản cũ thành .bak) ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": ckpt_path.name, "fingerprint": meta.get("fingerprint"),
               "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "split": split,
               "bootstrap": args.bootstrap, "seed": args.seed, "results": results}
    for name, content in (("metrics.json", json.dumps(payload, ensure_ascii=False, indent=2)),
                          ("metrics.txt", "\n".join(report) + "\n")):
        p = out / name
        if p.exists():
            p.replace(p.with_name(name + ".bak"))
        p.write_text(content, encoding="utf-8")
    print(f"\nĐã lưu: {out / 'metrics.txt'} và {out / 'metrics.json'} (bản cũ, nếu có, đổi tên thành .bak)")
    print("Lưu ý: không so trực tiếp với evaluate.py cũ — cách tính hòa điểm, bộ query và gallery đã đổi.")


if __name__ == "__main__":
    main()