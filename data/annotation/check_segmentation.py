"""
check_segmentation.py — Kiểm tra truy vấn sau khi tách từ có KHỚP cách tách của text_seg không (làm ở Giai đoạn A).

Cách làm: lấy từ vựng (token) trong text_seg của Train+Val+Test; tách truy vấn bằng bộ tách từ thống nhất
(seg_utils) rồi đếm tỉ lệ từ ghép (token có "_") của truy vấn xuất hiện trong từ vựng sản phẩm. Nếu có
underthesea thì in thêm cùng chỉ số cho underthesea để so. Bộ tách từ nào có tỉ lệ cao hơn là bộ khớp dữ liệu hơn.

Chạy : python check_segmentation.py                    (dùng queries_200.csv nếu đã có, không thì 12 câu mẫu)
       python check_segmentation.py --queries file.csv  (cột query_text)
Không cần GPU. Không ghi file.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import seg_utils

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()

SAMPLES = ["áo sơ mi nam dài tay màu trắng", "quần jean ống suông xanh đậm", "áo polo cotton form regular",
           "thắt lưng da bò khóa tự động", "áo khoác bomber phong cách đường phố", "ví da nam mini",
           "quần short kaki thể thao", "áo thun oversize cổ tròn", "balo laptop chống nước",
           "mũ lưỡi trai màu đen", "áo len cổ lọ giữ ấm", "quần tây công sở ôm vừa"]


def load_vocab():
    vocab = Counter()
    for split in ("train", "val", "test"):
        p = ROOT / f"data/splits/{split}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    vocab.update((row.get("text_seg") or "").lower().split())
    return vocab


def rate(queries, fn, vocab):
    compounds = [t for q in queries for t in fn(q).split() if "_" in t]
    found = [t for t in compounds if t in vocab]
    missing = Counter(t for t in compounds if t not in vocab)
    return len(compounds), len(found), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "data/annotation/benchmark/queries_200.csv"))
    args = ap.parse_args()

    qpath = Path(args.queries)
    if qpath.exists():
        with open(qpath, encoding="utf-8-sig", newline="") as f:
            queries = [r["query_text"] for r in csv.DictReader(f) if r.get("query_text")]
        print(f"Đọc {len(queries)} truy vấn từ {qpath}")
    else:
        queries = SAMPLES
        print(f"Chưa có {qpath}: dùng {len(queries)} câu mẫu")

    vocab = load_vocab()
    if not vocab:
        raise SystemExit("Không đọc được text_seg trong data/splits/*.jsonl")
    print(f"Từ vựng text_seg: {len(vocab)} token; từ ghép (có '_'): {sum(1 for t in vocab if '_' in t)}")

    print(f"\n[1] Bộ tách từ thống nhất: {seg_utils.segmenter_name()}")
    n, ok, miss = rate(queries, seg_utils.segment_text, vocab)
    print(f"    từ ghép trong truy vấn: {n}; có trong từ vựng sản phẩm: {ok} ({ok / max(n, 1):.1%})")
    print("    10 từ ghép KHÔNG có trong từ vựng:", miss.most_common(10))

    try:
        from underthesea import word_tokenize
    except ImportError:
        print("\n[2] underthesea chưa cài: bỏ qua phần so sánh.")
        return
    fn = lambda t: word_tokenize(seg_utils.normalize(t), format="text")        # noqa: E731
    n2, ok2, miss2 = rate(queries, fn, vocab)
    print("\n[2] underthesea (để so)")
    print(f"    từ ghép trong truy vấn: {n2}; có trong từ vựng sản phẩm: {ok2} ({ok2 / max(n2, 1):.1%})")
    print("    10 từ ghép KHÔNG có trong từ vựng:", miss2.most_common(10))
    diff = [(q, seg_utils.segment_text(q), fn(q)) for q in queries if seg_utils.segment_text(q) != fn(q)]
    print(f"\nSố truy vấn hai bộ tách khác nhau: {len(diff)}/{len(queries)}. 5 ví dụ:")
    for q, a, b in diff[:5]:
        print(f"  {q}\n    thống nhất : {a}\n    underthesea: {b}")


if __name__ == "__main__":
    main()
