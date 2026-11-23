"""
V10: Tính kappa và ghép nhãn (kế hoạch mục 9.6)
Hai bước, chạy theo thứ tự:

  python v10_kappa_merge.py kappa      # sau khi cả hai xong file overlap: κ, ma trận nhầm lẫn, disagreements.csv
  python v10_kappa_merge.py merge      # sau khi gán xong mọi file: labels_merged.csv, qrels.txt

Thư mục nhãn (đặt các file .xlsx ĐÃ GÁN, giữ nguyên tên file):
  data/annotation/benchmark/annotation/filled_user/     file của bạn: annotation_A_01a/01b, A_02..05, B_02..05
  data/annotation/benchmark/annotation/filled_helper/   file của người thứ hai: annotation_B_01a/01b
                                                        (và annotation_R_helper.xlsx nếu phải làm bước 2b)
Đầu ra (data/annotation/benchmark/): kappa.txt, disagreements.csv, labels_merged.csv, qrels.txt

Cách hoạt động:
  - Cặp được HAI người cùng gán: tập overlap (Q086–Q100, Q186–Q200) và, nếu có, tập R (Q071–Q080, Q171–Q180).
  - kappa: κ trọng số TUYẾN TÍNH là giá trị báo cáo chính; κ bậc hai chỉ tham khảo; kèm tỉ lệ đồng thuận chính xác và
    ma trận nhầm lẫn 3×3. κ ≥ 0,61 → qua G4; ngược lại → bước 2b (họp 20 phút, làm tập R, tính κ lần 2).
  - disagreements.csv sắp theo |label_user − label_helper| giảm dần; cột final_label để TRỐNG, hai người điền khi họp
    (mỗi dòng thảo luận tối đa 1 phút). Dòng bỏ trống final_label sẽ lấy NHÃN THẤP HƠN (bước 3).
  - merge: dòng hai nhãn giống nhau giữ nguyên; dòng khác nhau lấy final_label (hoặc nhãn thấp hơn); phần còn lại lấy
    nhãn của người gán duy nhất. KHÔNG sửa nhãn vòng 1 để tăng kappa.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/splits").exists():
    ROOT = Path(".").resolve()

BENCH = ROOT / "data/annotation/benchmark"
ANN = BENCH / "annotation"
USER_DIR = ANN / "filled_user"
HELPER_DIR = ANN / "filled_helper"
POOL_SOURCES = ANN / "_private/pool_sources.csv"       # chỉ để kiểm tra ĐỦ cặp, không đọc nguồn hệ thống
LOG_DIR = ROOT / "results/logs"
OVERLAP = [(86, 100), (186, 200)]
RESERVE_R = [(71, 80), (171, 180)]
THRESHOLD = 0.61


def qnum(qid):
    return int("".join(ch for ch in qid if ch.isdigit()))


def in_ranges(n, ranges):
    return any(a <= n <= b for a, b in ranges)


def read_labels(folder):
    """Đọc mọi file .xlsx trong thư mục: {(query_id, product_id): (label, note, file, query_text, name, attrs)}.
    label = None nếu để trống; báo lỗi nếu giá trị không thuộc {0,1,2} hoặc cặp bị lặp."""
    import openpyxl
    out, problems = {}, []
    if not folder.exists():
        return out, [f"Không có thư mục {folder}"]
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["annotation"]
        rows = ws.iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
        col = {h: i for i, h in enumerate(header)}
        for need in ("query_id", "product_id", "label", "note"):
            if need not in col:
                problems.append(f"{path.name}: thiếu cột {need}")
                break
        else:
            for r in rows:
                if r[col["query_id"]] is None:
                    continue
                key = (str(r[col["query_id"]]).strip(), str(r[col["product_id"]]).strip())
                raw = r[col["label"]]
                if raw is None or str(raw).strip() == "":
                    label = None
                else:
                    try:
                        label = int(float(str(raw).strip()))
                    except ValueError:
                        label = -1
                    if label not in (0, 1, 2):
                        problems.append(f"{path.name}: nhãn không hợp lệ {raw!r} ở {key}")
                        label = None
                if key in out:
                    problems.append(f"{path.name}: cặp {key} lặp (đã có ở {out[key]['file']})")
                    continue
                out[key] = {"label": label, "note": r[col["note"]] or "", "file": path.name,
                            "query_text": r[col.get("query_text", 0)] if "query_text" in col else "",
                            "name": r[col["name"]] if "name" in col else "",
                            "attrs": r[col["attrs_in_query"]] if "attrs_in_query" in col else ""}
    return out, problems


def kappa_stats(a, b):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    labels = [0, 1, 2]
    lin = cohen_kappa_score(a, b, labels=labels, weights="linear")
    quad = cohen_kappa_score(a, b, labels=labels, weights="quadratic")
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    cm = confusion_matrix(a, b, labels=labels)
    return lin, quad, agree, cm


def double_labeled(user, helper):
    keys = sorted(set(user) & set(helper))
    return keys


def split_sets(keys):
    ov = [k for k in keys if in_ranges(qnum(k[0]), OVERLAP)]
    rs = [k for k in keys if in_ranges(qnum(k[0]), RESERVE_R)]
    return ov, rs


def report_round(name, keys, user, helper, lines):
    missing = [k for k in keys if user[k]["label"] is None or helper[k]["label"] is None]
    if missing:
        raise SystemExit(f"{name}: còn {len(missing)} cặp chưa có nhãn của một trong hai người (ví dụ {missing[:3]}).")
    a = [user[k]["label"] for k in keys]
    b = [helper[k]["label"] for k in keys]
    lin, quad, agree, cm = kappa_stats(a, b)
    qs = len({k[0] for k in keys})
    lines += [f"{name}: {len(keys)} cặp, {qs} truy vấn",
              f"  κ trọng số TUYẾN TÍNH (giá trị báo cáo chính): {lin:.3f}",
              f"  κ trọng số bậc hai (tham khảo): {quad:.3f}",
              f"  tỉ lệ đồng thuận chính xác: {agree:.3f}",
              "  ma trận nhầm lẫn (hàng = bạn, cột = người thứ hai; nhãn 0,1,2):"]
    lines += ["    " + " ".join(f"{int(v):5d}" for v in row) for row in cm]
    lines.append(f"  → {'QUA G4 (κ ≥ 0,61)' if lin >= THRESHOLD else 'CHƯA QUA G4 (κ < 0,61): làm bước 2b'}")
    return lin


def cmd_kappa():
    user, p1 = read_labels(USER_DIR)
    helper, p2 = read_labels(HELPER_DIR)
    for p in p1 + p2:
        print("  [CẢNH BÁO]", p)
    keys = double_labeled(user, helper)
    ov, rs = split_sets(keys)
    if not ov:
        raise SystemExit("Chưa có cặp overlap nào được cả hai gán. Kiểm tra tên file và thư mục filled_user/filled_helper.")
    # đủ cặp overlap?
    expected_ov = {(qid, pid) for (qid, pid) in user if in_ranges(qnum(qid), OVERLAP)}
    lines = ["# Kappa (kế hoạch 9.6)", ""]
    k1 = report_round("Vòng 1 — tập overlap", ov, user, helper, lines)
    k2 = None
    if rs:
        lines.append("")
        k2 = report_round("Vòng 2 — tập dự phòng R", rs, user, helper, lines)
    only_user = len(expected_ov - set(ov))
    if only_user:
        lines.append(f"\nCẢNH BÁO: {only_user} cặp overlap có trong file của bạn nhưng chưa thấy ở file người thứ hai.")
    lines.append("\nKhông sửa nhãn vòng 1 để tăng kappa. Báo cáo ghi cả hai giá trị κ (vòng 1 và vòng 2 nếu có) "
                 "và loại trọng số.")
    text = "\n".join(lines)
    print(text)
    (BENCH / "kappa.txt").write_text(text + "\n", encoding="utf-8")

    # disagreements.csv: cả overlap và R, sắp theo |chênh lệch| giảm dần
    dis = [k for k in ov + rs if user[k]["label"] != helper[k]["label"]]
    dis.sort(key=lambda k: (-abs(user[k]["label"] - helper[k]["label"]), k))
    with open(BENCH / "disagreements.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "product_id", "label_user", "label_helper", "abs_diff", "final_label",
                    "query_text", "attrs_in_query", "name", "note_user", "note_helper"])
        for k in dis:
            u, h = user[k], helper[k]
            w.writerow([k[0], k[1], u["label"], h["label"], abs(u["label"] - h["label"]), "",
                        u["query_text"], u["attrs"], u["name"], u["note"], h["note"]])
    print(f"\nĐã lưu: {BENCH / 'kappa.txt'}, {BENCH / 'disagreements.csv'} ({len(dis)} dòng bất đồng)")
    print("Họp: duyệt dòng bất đồng lớn nhất trước, điền cột final_label. Để trống = lấy nhãn thấp hơn.")
    with open(LOG_DIR / "log_B.md", "a", encoding="utf-8") as f:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        f.write(f"\n## V10 — kappa\n- vòng 1: κ tuyến tính {k1:.3f}" + (f"; vòng 2: {k2:.3f}" if k2 is not None else "") + "\n")


def cmd_merge():
    user, p1 = read_labels(USER_DIR)
    helper, p2 = read_labels(HELPER_DIR)
    problems = p1 + p2
    # nhãn chốt từ cuộc họp
    resolved = {}
    dis_path = BENCH / "disagreements.csv"
    if dis_path.exists():
        for r in csv.DictReader(open(dis_path, encoding="utf-8", newline="")):
            fl = (r.get("final_label") or "").strip()
            if fl:
                assert fl in ("0", "1", "2"), f"final_label không hợp lệ: {fl}"
                resolved[(r["query_id"], r["product_id"])] = int(fl)

    both = set(user) & set(helper)
    rows, missing = [], []
    for key in sorted(user, key=lambda k: (qnum(k[0]), k[1])):
        u = user[key]
        h = helper.get(key)
        lu, lh = u["label"], (h["label"] if h else None)
        if key in both:
            if lu is None or lh is None:
                missing.append(key)
                continue
            final = lu if lu == lh else resolved.get(key, min(lu, lh))
            src = "double" if lu == lh else ("họp" if key in resolved else "thấp hơn")
        else:
            if lu is None:
                missing.append(key)
                continue
            final, src = lu, "single"
        rows.append({"query_id": key[0], "product_id": key[1], "label_user": lu, "label_helper": lh if lh is not None else "",
                     "final_label": final, "source": src, "note_user": u["note"],
                     "note_helper": h["note"] if h else ""})
    # file của người thứ hai không có trong file của bạn (ví dụ B_01) vẫn phải có trong file của bạn: A_01 tương ứng
    only_helper = set(helper) - set(user)
    for p in problems:
        print("  [CẢNH BÁO]", p)
    if only_helper:
        print(f"  [CẢNH BÁO] {len(only_helper)} cặp chỉ có ở file người thứ hai (không có ở file của bạn): kiểm tra overlap.")
    # so với pool: mọi cặp trong pool phải có nhãn
    if POOL_SOURCES.exists():
        pool = {(r["query_id"], r["product_id"]) for r in csv.DictReader(open(POOL_SOURCES, encoding="utf-8", newline=""))}
        labeled = {(r["query_id"], r["product_id"]) for r in rows}
        no_label = pool - labeled
        if no_label:
            missing += sorted(no_label - set(missing))
    if missing:
        print(f"\nCHƯA ĐỦ NHÃN: còn {len(set(missing))} cặp chưa có nhãn hợp lệ, ví dụ {sorted(set(missing))[:5]}")
        print("Gán nốt rồi chạy lại. Chưa ghi labels_merged.csv / qrels.txt.")
        sys.exit(1)

    with open(BENCH / "labels_merged.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(BENCH / "qrels.txt", "w", encoding="utf-8") as f:                 # qid 0 pid label (mục 3.3)
        for r in rows:
            f.write(f"{r['query_id']} 0 {r['product_id']} {r['final_label']}\n")

    dist = Counter(r["final_label"] for r in rows)
    per_q = defaultdict(int)
    for r in rows:
        if r["final_label"] >= 1:
            per_q[r["query_id"]] += 1
    n_q = len({r["query_id"] for r in rows})
    zero_rq = sorted({r["query_id"] for r in rows} - set(per_q))
    print(f"\nĐã lưu {BENCH / 'labels_merged.csv'} và {BENCH / 'qrels.txt'}: {len(rows)} cặp, {n_q} truy vấn")
    print(f"  phân bố nhãn: " + ", ".join(f"{k}: {dist[k]}" for k in (0, 1, 2)))
    print(f"  truy vấn có R_q = 0 (sẽ bị V11 loại khỏi trung bình): {len(zero_rq)} {zero_rq[:10]}")
    print(f"  nguồn nhãn: {dict(Counter(r['source'] for r in rows))}")
    with open(LOG_DIR / "log_B.md", "a", encoding="utf-8") as f:
        f.write(f"\n## V10 — ghép nhãn\n- {len(rows)} cặp, {n_q} truy vấn; nhãn {dict(dist)}; R_q=0: {len(zero_rq)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["kappa", "merge"])
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd_kappa() if args.command == "kappa" else cmd_merge()