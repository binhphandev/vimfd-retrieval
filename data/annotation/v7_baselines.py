"""
V7: Bốn phương pháp cơ sở trên 200 truy vấn kiểm thử (kế hoạch mục 6)
Đầu vào : data/annotation/benchmark/queries_200.csv  (cột query_id, query_text)
          data/splits/test.jsonl (corpus 1.339 sản phẩm Test), config.py (MODEL_TEXT_FIELD, PROCESSED_DIR)
Đầu ra  : results/runs/{tfidf,bm25,clip_vi2en,bienc_text}.run     (qid Q0 product_id rank score hệ_thống)
          data/annotation/benchmark/translations.csv               (bản dịch vi→en, dịch MỘT lần)
          results/logs/log_B.md                                    (phiên bản thư viện, tên mô hình, thời gian)
Chạy    : python v7_baselines.py                      (cả 4 hệ thống)
          python v7_baselines.py --sample5            (thử trên 5 truy vấn, ghi vào results/runs_sample/)
          python v7_baselines.py --only tfidf,bm25    (chọn hệ thống)

Yêu cầu : pip install scikit-learn rank_bm25 underthesea
          clip_vi2en : pip install torch torchvision transformers pillow sentencepiece
          bienc_text : pip install sentence-transformers pyvi
Không tinh chỉnh tham số bằng nhãn hay kết quả. Chạy lúc V8 KHÔNG đang đo độ trễ.

Ghi chú thiết kế:
  - Văn bản sản phẩm lấy bằng get_text() theo config.MODEL_TEXT_FIELD (text_seg → text → name), giống V5/V8.
  - Truy vấn: chuẩn hóa mục 3.2 rồi tách từ bằng underthesea (như V4) cho TF-IDF và BM25.
  - Hạng bằng điểm được phá bằng thứ tự sản phẩm trong test.jsonl (ổn định, tái lập được).
  - Sản phẩm không có file ảnh KHÔNG vào chỉ mục của hệ thống dùng ảnh (clip_vi2en); số lượng được ghi vào log.
"""

import argparse
import csv
import importlib
import json
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
try:
    import config
except ImportError:
    config = None

SEED = 42
TOP_K = 50
TEST_PATH    = ROOT / "data/splits/test.jsonl"
QUERIES_CSV  = ROOT / "data/annotation/benchmark/queries_200.csv"
TRANSL_CSV   = ROOT / "data/annotation/benchmark/translations.csv"
LOG_DIR      = ROOT / "results/logs"

# Tham số cố định (mục 6): KHÔNG chỉnh
BM25_K1, BM25_B = 1.2, 0.75
TRANSLATE_MODEL = "vinai/vinai-translate-vi2en"
TRANSLATE_BEAMS = 4
CLIP_NAME  = "openai/clip-vit-base-patch32"
CLIP_MEAN  = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD   = (0.26862954, 0.26130258, 0.27577711)
BIENC_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
ALL_SYSTEMS = ["tfidf", "bm25", "clip_vi2en", "bienc_text"]


# ══════════════════════════════════════════════════════════════════
# Tiện ích
# ══════════════════════════════════════════════════════════════════
def normalize(text) -> str:
    """Mục 3.2: Unicode NFC, chữ thường, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def get_text(product: dict) -> str:
    """Văn bản sản phẩm theo config.MODEL_TEXT_FIELD (không tự định nghĩa hàm mới)."""
    field = getattr(config, "MODEL_TEXT_FIELD", "text_seg") if config else "text_seg"
    return product.get(field) or product.get("text") or product.get("name", "")


def featured_image_path(product: dict):
    images = product.get("images") or []
    if not images:
        return None
    images = sorted(images, key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    lp = images[0].get("local_path")
    if not lp:
        return None
    local_path = Path(lp)
    candidates = [local_path, ROOT / local_path]
    if config is not None and getattr(config, "PROCESSED_DIR", None) and not local_path.is_absolute():
        candidates.append(Path(config.PROCESSED_DIR) / local_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_queries(limit=None):
    with open(QUERIES_CSV, encoding="utf-8-sig", newline="") as f:
        rows = [{"query_id": r["query_id"].strip(), "query_text": r["query_text"]} for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def segment_underthesea(text: str) -> str:
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


def top_k_indices(scores: np.ndarray, k: int = TOP_K):
    """Top-k theo điểm giảm dần; điểm bằng nhau thì theo thứ tự sản phẩm (ổn định)."""
    return np.argsort(-scores, kind="stable")[:k]


def run_lines(qids, pids, score_matrix, system):
    """score_matrix: (n_query, n_product). Trả về các dòng run theo định dạng mục 3.3."""
    lines = []
    for i, qid in enumerate(qids):
        idx = top_k_indices(score_matrix[i])
        for rank, j in enumerate(idx, start=1):
            lines.append(f"{qid} Q0 {pids[j]} {rank} {float(score_matrix[i][j]):.6f} {system}")
    return lines


def validate_run(lines, qids, test_ids, names, system, k=TOP_K):
    """Mục 6.6: đủ dòng, product_id thuộc Test, không trùng trong truy vấn, score giảm dần; in top-3 của 5 truy vấn."""
    assert len(lines) == len(qids) * k, f"{system}: {len(lines)} dòng, cần {len(qids) * k}"
    per_q = {}
    for line in lines:
        qid, _, pid, rank, score, sysname = line.split()
        assert sysname == system and pid in test_ids, f"{system}: dòng lỗi {line}"
        per_q.setdefault(qid, []).append((pid, int(rank), float(score)))
    for qid, items in per_q.items():
        pids = [p for p, _, _ in items]
        scores = [s for _, _, s in items]
        assert len(set(pids)) == len(pids), f"{system}: trùng sản phẩm ở {qid}"
        assert [r for _, r, _ in items] == list(range(1, len(items) + 1)), f"{system}: rank sai ở {qid}"
        assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)), f"{system}: score không giảm ở {qid}"
    print(f"  ✅ {system}: {len(lines)} dòng hợp lệ ({len(per_q)} truy vấn)")


def eyeball(lines, queries, names, n=5):
    qtext = {q["query_id"]: q["query_text"] for q in queries}
    rng = random.Random(SEED)
    picked = rng.sample([q["query_id"] for q in queries], min(n, len(queries)))
    top = {}
    for line in lines:
        qid, _, pid, rank, _, _ = line.split()
        if qid in picked and int(rank) <= 3:
            top.setdefault(qid, []).append(names[pid][:70])
    print("  Đọc bằng mắt top-3 (mục 6.6):")
    for qid in picked:
        print(f"   {qid} '{qtext[qid]}'")
        for j, nm in enumerate(top.get(qid, []), 1):
            print(f"      {j}. {nm}")


def pip_version(mod):
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return "không có"


# ══════════════════════════════════════════════════════════════════
# 1. TF-IDF  (mục 6.2)
# ══════════════════════════════════════════════════════════════════
def run_tfidf(queries, products):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [get_text(p) for p in products]
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs)                                  # fit trên corpus Test
    Q = vec.transform([segment_underthesea(normalize(q["query_text"])) for q in queries])
    scores = (Q @ X.T).toarray()                                 # vector đã chuẩn hóa L2 → tích vô hướng = cosine
    return scores, {"vocab_size": len(vec.vocabulary_)}


# ══════════════════════════════════════════════════════════════════
# 2. BM25  (mục 6.3)
# ══════════════════════════════════════════════════════════════════
def run_bm25(queries, products):
    from rank_bm25 import BM25Okapi
    corpus = [get_text(p).lower().split() for p in products]
    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    scores = np.vstack([
        bm25.get_scores(segment_underthesea(normalize(q["query_text"])).lower().split()) for q in queries
    ])
    return scores, {"k1": BM25_K1, "b": BM25_B}


# ══════════════════════════════════════════════════════════════════
# 3. CLIP-only + dịch máy  (mục 6.4)
# ══════════════════════════════════════════════════════════════════
def get_translations(queries, device):
    """Dịch MỘT lần, lưu translations.csv; các lần chạy sau đọc từ file."""
    if TRANSL_CSV.exists():
        with open(TRANSL_CSV, encoding="utf-8-sig", newline="") as f:
            saved = {r["query_id"]: r["query_en"] for r in csv.DictReader(f)}
        if all(q["query_id"] in saved for q in queries):
            print(f"  Đọc bản dịch từ {TRANSL_CSV} (không dịch lại)")
            return {q["query_id"]: saved[q["query_id"]] for q in queries}, "đọc từ file"
        print("  translations.csv thiếu một số query_id → dịch lại toàn bộ")

    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"  Dịch {len(queries)} truy vấn bằng {TRANSLATE_MODEL} (beam={TRANSLATE_BEAMS})...")
    tok = AutoTokenizer.from_pretrained(
        TRANSLATE_MODEL,
        src_lang="vi_VN",
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL).to(device).eval()
    en_id = getattr(tok, "lang_code_to_id", {}).get("en_XX") if hasattr(tok, "lang_code_to_id") else None
    if en_id is None:
        en_id = tok.convert_tokens_to_ids("en_XX")
    out = {}
    with torch.no_grad():
        for i in range(0, len(queries), 16):
            batch = queries[i:i + 16]
            enc = tok([normalize(q["query_text"]) for q in batch], padding=True, return_tensors="pt").to(device)
            gen = model.generate(**enc, decoder_start_token_id=en_id, num_beams=TRANSLATE_BEAMS,
                                 num_return_sequences=1, early_stopping=True, max_length=64)
            for q, text in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                out[q["query_id"]] = text.strip()
    TRANSL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSL_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_vi", "query_en", "model", "num_beams", "transformers"])
        for q in queries:
            w.writerow([q["query_id"], q["query_text"], out[q["query_id"]], TRANSLATE_MODEL,
                        TRANSLATE_BEAMS, transformers.__version__])
    del model
    return out, f"dịch mới bằng {TRANSLATE_MODEL}"


def run_clip(queries, products, device):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from transformers import CLIPModel, CLIPTokenizer

    en, note = get_translations(queries, device)

    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize(CLIP_MEAN, CLIP_STD)])

    usable = [(i, featured_image_path(p)) for i, p in enumerate(products)]
    usable = [(i, path) for i, path in usable if path is not None]
    print(f"  Mã hóa {len(usable)} ảnh (thiếu ảnh: {len(products) - len(usable)})...")
    feats = []
    with torch.no_grad():
        for s in range(0, len(usable), 32):
            batch = torch.stack([tf(Image.open(path).convert("RGB")) for _, path in usable[s:s + 32]]).to(device)
            v = model.visual_projection(model.vision_model(pixel_values=batch).pooler_output)
            feats.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    img = np.vstack(feats)

    texts = [en[q["query_id"]] for q in queries]
    enc = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        t = model.text_projection(model.text_model(**enc).pooler_output)
    txt = torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()

    sims = txt @ img.T                                           # (n_query, n_usable)
    scores = np.full((len(queries), len(products)), -1e9, dtype=np.float32)   # sản phẩm không ảnh không bao giờ lọt top
    cols = [i for i, _ in usable]
    scores[:, cols] = sims
    return scores, {"translation": note, "n_images": len(usable), "n_missing_images": len(products) - len(usable)}


# ══════════════════════════════════════════════════════════════════
# 4. Bi-encoder văn bản–văn bản  (mục 6.5)
# ══════════════════════════════════════════════════════════════════
def run_bienc(queries, products, device):
    from sentence_transformers import SentenceTransformer
    try:
        from pyvi.ViTokenizer import tokenize as seg        # hướng dẫn của mô hình: tách từ bằng pyvi
        seg_name = "pyvi"
    except ImportError:
        seg = segment_underthesea
        seg_name = "underthesea (KHÔNG có pyvi — cài pyvi để đúng hướng dẫn của mô hình)"
        print("  [CẢNH BÁO] Chưa cài pyvi: tách từ truy vấn bằng underthesea. pip install pyvi")
    model = SentenceTransformer(BIENC_NAME, device=device)
    q_emb = model.encode([seg(normalize(q["query_text"])) for q in queries], batch_size=32,
                         normalize_embeddings=True, convert_to_numpy=True)
    d_emb = model.encode([get_text(p) for p in products], batch_size=32, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
    return q_emb @ d_emb.T, {"query_segmenter": seg_name, "max_seq_length": model.max_seq_length}


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(ALL_SYSTEMS), help="Danh sách hệ thống, cách nhau bằng dấu phẩy")
    ap.add_argument("--sample5", action="store_true", help="Thử trên 5 truy vấn đầu, ghi vào results/runs_sample/")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    systems = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"Hệ thống không hợp lệ: {unknown}. Chọn trong {ALL_SYSTEMS}")

    out_dir = ROOT / ("results/runs_sample" if args.sample5 else "results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    queries = load_queries(5 if args.sample5 else None)
    products = load_jsonl(TEST_PATH)
    pids = [p["id"] for p in products]
    names = {p["id"]: (p.get("name") or "") for p in products}
    test_ids = set(pids)
    print(f"Truy vấn: {len(queries)} | corpus Test: {len(products)}" + ("  [SAMPLE 5]" if args.sample5 else ""))

    n_dup = len(queries) - len({normalize(q['query_text']) for q in queries})
    if n_dup:
        print(f"  [CẢNH BÁO] {n_dup} truy vấn trùng chữ trong queries_200.csv (mục 4.4 cấm trùng) — sửa trước khi chạy chính thức")
    if any("_" in q["query_text"] for q in queries):
        print("  [CẢNH BÁO] query_text đã có dấu '_' — file này đã tách từ? V7 tự tách từ nên sẽ tách hai lần.")

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                    # chưa cài torch, hoặc cài lỗi (OSError/DLL): tfidf và bm25 vẫn chạy được
            device = "cpu"
    print(f"Device: {device}")

    qids = [q["query_id"] for q in queries]
    log = [f"\n## V7 — baseline ({datetime.now():%Y-%m-%d %H:%M})" + (" [SAMPLE 5 – không dùng]" if args.sample5 else ""),
           f"- {len(queries)} truy vấn, corpus {len(products)}, device={device}, seed={SEED}, top_k={TOP_K}",
           f"- thư viện: sklearn {pip_version('sklearn')}, rank_bm25 {pip_version('rank_bm25')}, "
           f"torch {pip_version('torch')}, transformers {pip_version('transformers')}, "
           f"sentence_transformers {pip_version('sentence_transformers')}, underthesea {pip_version('underthesea')}"]

    runners = {
        "tfidf": lambda: run_tfidf(queries, products),
        "bm25": lambda: run_bm25(queries, products),
        "clip_vi2en": lambda: run_clip(queries, products, device),
        "bienc_text": lambda: run_bienc(queries, products, device),
    }
    model_note = {"tfidf": "TfidfVectorizer(token_pattern=\\S+, ngram=(1,2), sublinear_tf)",
                  "bm25": f"BM25Okapi(k1={BM25_K1}, b={BM25_B})",
                  "clip_vi2en": f"{TRANSLATE_MODEL} (beam {TRANSLATE_BEAMS}) + {CLIP_NAME}",
                  "bienc_text": BIENC_NAME}

    for system in systems:
        print(f"\n=== {system} ===")
        t0 = time.time()
        scores, extra = runners[system]()
        elapsed = time.time() - t0
        lines = run_lines(qids, pids, scores, system)
        validate_run(lines, qids, test_ids, names, system)
        eyeball(lines, queries, names)
        out = out_dir / f"{system}.run"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Lưu: {out}  ({elapsed:.0f}s)")
        log.append(f"- {system}: {model_note[system]}; {elapsed:.0f}s; {extra}")

    with open(LOG_DIR / "log_B.md", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\nV7 hoàn thành." + (" Đây là chạy thử 5 truy vấn: đọc top-3 ở trên rồi chạy lại không có --sample5." if args.sample5 else ""))


if __name__ == "__main__":
    main()