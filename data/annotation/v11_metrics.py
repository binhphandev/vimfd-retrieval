"""
V11: Tính chỉ số đánh giá cho 6 hệ thống
Đầu vào:
  - results/runs/{tfidf,bm25,clip_vi2en,bienc_text,dual_encoder,vimfr}.run
  - data/annotation/benchmark/qrels.txt
  - data/annotation/benchmark/queries_200.csv
Đầu ra:
  - results/benchmark/metrics_per_query.csv
  - results/benchmark/metrics_summary.csv
  - results/benchmark/metrics_by_group.csv
  - results/benchmark/worst30.csv
Chạy: python v11_metrics.py
      (Thử trên dữ liệu giả trước: python v11_metrics.py --dry-run)
"""

import csv
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

ROOT    = Path(".")
RUNS_DIR  = ROOT / "results/runs"
OUT_DIR   = ROOT / "results/benchmark"
QRELS_PATH   = ROOT / "data/annotation/benchmark/qrels.txt"
QUERIES_PATH = ROOT / "data/annotation/benchmark/queries_200.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["tfidf", "bm25", "clip_vi2en", "bienc_text", "dual_encoder", "vimfr"]

# ── Tạo dữ liệu giả để thử script trước khi có qrels thật ────────
def make_fake_data():
    """Tạo qrels và run giả để kiểm tra logic tính chỉ số."""
    fake_qids    = [f"Q{i:03d}" for i in range(1, 21)]
    fake_pids    = [f"P{j:04d}" for j in range(1, 101)]
    fake_qrels   = {}
    for qid in fake_qids:
        # Mỗi truy vấn có 5–10 sản phẩm liên quan
        relevant = random.sample(fake_pids, random.randint(5, 10))
        fake_qrels[qid] = {pid: random.choice([1, 2]) for pid in relevant}
        # Các sản phẩm khác nhãn 0 (không cần lưu, mặc định 0)
    
    fake_runs = {}
    for sys in SYSTEMS:
        fake_runs[sys] = {}
        for qid in fake_qids:
            # Top-50 ngẫu nhiên
            ranked = random.sample(fake_pids, 50)
            fake_runs[sys][qid] = [(pid, 50 - i) for i, pid in enumerate(ranked)]
    
    return fake_qrels, fake_runs, {q: {"n_attrs": random.choice([1,2,3]),
                                       "category": random.choice(["áo","quần","phụ kiện"])}
                                   for q in fake_qids}

# ── Load qrels ────────────────────────────────────────────────────
def load_qrels(path):
    """qrels[query_id][product_id] = label (0/1/2)"""
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, pid, label = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][pid] = label
    return dict(qrels)

# ── Load run files ─────────────────────────────────────────────────
def load_run(path):
    """run[query_id] = [(product_id, score), ...] sắp theo rank"""
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, pid, rank, score, sys = parts
            run[qid].append((pid, float(score)))
    # Đã sắp theo score giảm dần trong file, giữ nguyên
    return dict(run)

# ── Hàm tính chỉ số ───────────────────────────────────────────────
def recall_at_k(ranked_pids, qrels_q, k):
    """Recall@K = số liên quan trong top-K / tổng liên quan trong pool"""
    r_q = sum(1 for label in qrels_q.values() if label >= 1)
    if r_q == 0:
        return None   # loại khỏi trung bình
    hits = sum(1 for pid in ranked_pids[:k] if qrels_q.get(pid, 0) >= 1)
    return hits / r_q

def ndcg_at_k(ranked_pids, qrels_q, k=10):
    """NDCG@10 với gain = 2^label - 1"""
    def gain(label):
        return 2 ** label - 1
    # DCG
    dcg = sum(
        gain(qrels_q.get(pid, 0)) / math.log2(i + 2)
        for i, pid in enumerate(ranked_pids[:k])
    )
    # IDCG: sắp các nhãn trong pool giảm dần, cắt ở k
    ideal_labels = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum(gain(l) / math.log2(i + 2) for i, l in enumerate(ideal_labels))
    if idcg == 0:
        return None
    return dcg / idcg

def mrr_at_k(ranked_pids, qrels_q, k=10):
    """MRR@10: 1/rank của sản phẩm liên quan (nhãn ≥ 1) đầu tiên trong top-K"""
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels_q.get(pid, 0) >= 1:
            return 1.0 / (i + 1)
    return 0.0

# ── Bootstrap CI ─────────────────────────────────────────────────
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Khoảng tin cậy bootstrap theo truy vấn."""
    n = len(values)
    rng = random.Random(SEED)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return lo, hi

# ── Tính chỉ số cho tất cả hệ thống ─────────────────────────────
def compute_all_metrics(runs, qrels, queries_meta):
    """
    runs: {sys_name: {qid: [(pid, score), ...]}}
    qrels: {qid: {pid: label}}
    queries_meta: {qid: {n_attrs, category}}
    Trả về per_query_rows, summary_rows, group_rows
    """
    # Truy vấn có R_q = 0 → loại khỏi mọi trung bình
    valid_qids = {
        qid for qid, q_qrels in qrels.items()
        if any(label >= 1 for label in q_qrels.values())
    }
    invalid_qids = set(qrels.keys()) - valid_qids
    if invalid_qids:
        print(f"  ⚠️  Loại {len(invalid_qids)} truy vấn có R_q=0: {sorted(invalid_qids)}")

    per_query_rows = []

    for sys_name, run in runs.items():
        for qid in valid_qids:
            ranked = [pid for pid, _ in run.get(qid, [])]
            q_qrels = qrels.get(qid, {})
            meta    = queries_meta.get(qid, {})

            r5  = recall_at_k(ranked, q_qrels, 5)
            r10 = recall_at_k(ranked, q_qrels, 10)
            nd  = ndcg_at_k(ranked, q_qrels, 10)
            mrr = mrr_at_k(ranked, q_qrels, 10)

            per_query_rows.append({
                "system":   sys_name,
                "query_id": qid,
                "n_attrs":  meta.get("n_attrs", ""),
                "category": meta.get("category", ""),
                "recall_5":  r5  if r5  is not None else "",
                "recall_10": r10 if r10 is not None else "",
                "ndcg_10":   nd  if nd  is not None else "",
                "mrr":       mrr,
            })

    # --- Summary (trung bình + CI) ---
    summary_rows = []
    for sys_name in SYSTEMS:
        sys_rows = [r for r in per_query_rows if r["system"] == sys_name]
        def avg(key):
            vals = [r[key] for r in sys_rows if r[key] != ""]
            return sum(vals) / len(vals) if vals else 0.0
        def vals(key):
            return [r[key] for r in sys_rows if r[key] != ""]

        nd_vals  = vals("ndcg_10")
        r10_vals = vals("recall_10")
        nd_lo,  nd_hi  = bootstrap_ci(nd_vals)  if nd_vals  else (0.0, 0.0)
        r10_lo, r10_hi = bootstrap_ci(r10_vals) if r10_vals else (0.0, 0.0)

        summary_rows.append({
            "system":        sys_name,
            "recall_5":      f"{avg('recall_5'):.4f}",
            "recall_10":     f"{avg('recall_10'):.4f}",
            "recall_10_ci":  f"[{r10_lo:.4f}, {r10_hi:.4f}]",
            "ndcg_10":       f"{avg('ndcg_10'):.4f}",
            "ndcg_10_ci":    f"[{nd_lo:.4f}, {nd_hi:.4f}]",
            "mrr":           f"{avg('mrr'):.4f}",
            "n_queries":     len(sys_rows),
        })

    # --- Nhóm theo n_attrs và category ---
    group_rows = []
    for sys_name in SYSTEMS:
        sys_rows = [r for r in per_query_rows if r["system"] == sys_name]
        for group_key in ["n_attrs", "category"]:
            groups = defaultdict(list)
            for r in sys_rows:
                groups[r[group_key]].append(r)
            for group_val, grp in sorted(groups.items()):
                nd_vals = [r["ndcg_10"] for r in grp if r["ndcg_10"] != ""]
                r10_vals = [r["recall_10"] for r in grp if r["recall_10"] != ""]
                group_rows.append({
                    "system":    sys_name,
                    "group_key": group_key,
                    "group_val": group_val,
                    "ndcg_10":   f"{sum(nd_vals)/len(nd_vals):.4f}" if nd_vals else "",
                    "recall_10": f"{sum(r10_vals)/len(r10_vals):.4f}" if r10_vals else "",
                    "n_queries": len(grp),
                })

    return per_query_rows, summary_rows, group_rows

# ── So sánh có cặp (bootstrap) ───────────────────────────────────
def paired_comparison(per_query_rows, sys_a, sys_b, metric="ndcg_10"):
    """Bootstrap khoảng tin cậy của hiệu trung bình sys_a - sys_b."""
    a_map = {r["query_id"]: r[metric] for r in per_query_rows
             if r["system"] == sys_a and r[metric] != ""}
    b_map = {r["query_id"]: r[metric] for r in per_query_rows
             if r["system"] == sys_b and r[metric] != ""}
    common = sorted(set(a_map) & set(b_map))
    diffs  = [a_map[q] - b_map[q] for q in common]
    if not diffs:
        return 0.0, (0.0, 0.0)
    mean_diff = sum(diffs) / len(diffs)
    lo, hi    = bootstrap_ci(diffs)
    return mean_diff, (lo, hi)

# ── Main ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Dùng dữ liệu giả để kiểm tra script")
args = parser.parse_args()

if args.dry_run:
    print("=== DRY RUN (dữ liệu giả) ===")
    qrels, runs, queries_meta = make_fake_data()
else:
    print("=== Chạy thật ===")
    # Load qrels
    qrels = load_qrels(QRELS_PATH)
    print(f"Qrels: {len(qrels)} truy vấn")

    # Load 6 run files
    runs = {}
    for sys in SYSTEMS:
        run_path = RUNS_DIR / f"{sys}.run"
        if not run_path.exists():
            print(f"  ⚠️  Thiếu: {run_path}")
            continue
        runs[sys] = load_run(run_path)
        print(f"  {sys}: {sum(len(v) for v in runs[sys].values())} dòng")

    # Load queries_meta
    queries_meta = {}
    with open(QUERIES_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            queries_meta[row["query_id"]] = {
                "n_attrs":  int(row.get("n_attrs", 1)),
                "category": row.get("category", ""),
            }

# ── Tính chỉ số ───────────────────────────────────────────────────
print("\nTính chỉ số...")
per_query_rows, summary_rows, group_rows = compute_all_metrics(runs, qrels, queries_meta)

# ── So sánh có cặp ────────────────────────────────────────────────
print("\nSo sánh có cặp:")
# ViMFR vs baseline tốt nhất (theo NDCG@10)
best_baseline = max(
    [s for s in summary_rows if s["system"] != "vimfr"],
    key=lambda x: float(x["ndcg_10"])
)["system"]
mean_diff, ci = paired_comparison(per_query_rows, "vimfr", best_baseline)
sig = "✅ có ý nghĩa" if ci[0] > 0 else "❌ không có ý nghĩa"
print(f"  ViMFR vs {best_baseline}: Δ={mean_diff:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] {sig}")

mean_diff2, ci2 = paired_comparison(per_query_rows, "vimfr", "dual_encoder")
sig2 = "✅ có ý nghĩa" if ci2[0] > 0 else "❌ không có ý nghĩa"
print(f"  ViMFR vs dual_encoder: Δ={mean_diff2:.4f} CI=[{ci2[0]:.4f},{ci2[1]:.4f}] {sig2}")

# ── Lưu kết quả ──────────────────────────────────────────────────
# metrics_per_query.csv
pq_fields = ["system","query_id","n_attrs","category",
             "recall_5","recall_10","ndcg_10","mrr"]
with open(OUT_DIR/"metrics_per_query.csv","w",encoding="utf-8",newline="") as f:
    w = csv.DictWriter(f, fieldnames=pq_fields)
    w.writeheader(); w.writerows(per_query_rows)

# metrics_summary.csv
sm_fields = ["system","recall_5","recall_10","recall_10_ci",
             "ndcg_10","ndcg_10_ci","mrr","n_queries"]
with open(OUT_DIR/"metrics_summary.csv","w",encoding="utf-8",newline="") as f:
    w = csv.DictWriter(f, fieldnames=sm_fields)
    w.writeheader(); w.writerows(summary_rows)

# metrics_by_group.csv
gr_fields = ["system","group_key","group_val","ndcg_10","recall_10","n_queries"]
with open(OUT_DIR/"metrics_by_group.csv","w",encoding="utf-8",newline="") as f:
    w = csv.DictWriter(f, fieldnames=gr_fields)
    w.writeheader(); w.writerows(group_rows)

# worst30.csv: 30 truy vấn có NDCG@10 thấp nhất của ViMFR
vimfr_rows = [r for r in per_query_rows if r["system"] == "vimfr" and r["ndcg_10"] != ""]
worst30    = sorted(vimfr_rows, key=lambda x: float(x["ndcg_10"]))[:30]
with open(OUT_DIR/"worst30.csv","w",encoding="utf-8",newline="") as f:
    w = csv.DictWriter(f, fieldnames=pq_fields)
    w.writeheader(); w.writerows(worst30)

print("\n=== Tóm tắt kết quả ===")
print(f"{'System':<15} {'R@5':>6} {'R@10':>6} {'NDCG@10':>8} {'MRR':>6}")
print("-" * 45)
for r in summary_rows:
    print(f"{r['system']:<15} {r['recall_5']:>6} {r['recall_10']:>6} {r['ndcg_10']:>8} {r['mrr']:>6}")

print(f"\nĐã lưu kết quả vào {OUT_DIR}/")
print("V11 hoàn thành!")
