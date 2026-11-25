"""Tiền xử lý truy vấn cho API, sao chép NGUYÊN VĂN từ vimfd-v4/common.py
(clean_text, segment, preprocess_query) để truy vấn đi qua đúng pipeline của text_seg lúc train.

Đặt tại: vimfd-retrieval/src/serving/text_utils.py
Cần: Java, `pip install py_vncorenlp`, và thư mục model VnCoreNLP.
Mặc định dùng <gốc dự án>/vncorenlp; đặt biến môi trường VNCORENLP_DIR nếu muốn dùng thư mục khác.

Kiểm tra khớp với dữ liệu train (nên chạy sau mỗi lần đổi cấu hình):
    python src/serving/text_utils.py --train data/splits/train.jsonl
"""
import html
import json
import os
import re
import threading
import unicodedata
from pathlib import Path

MAX_SEG_SYLLABLES = 300  # phải bằng common.MAX_SEG_SYLLABLES
# Mặc định: thư mục vncorenlp nằm ở gốc dự án (src/serving/text_utils.py -> parents[2]); biến môi trường ghi đè.
VNCORENLP_DIR = os.environ.get("VNCORENLP_DIR") or str(Path(__file__).resolve().parents[2] / "vncorenlp")

_HTML = re.compile(r"<[^>]+>")
_JUNK = re.compile("[\u200b-\u200d\ufeff\u00ad\ufe0f]")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def clean_text(s):
    """NFKC (also folds fancy Unicode fonts), unescape HTML, drop tags / zero-width / emoji / 【】, collapse spaces."""
    s = unicodedata.normalize("NFKC", html.unescape(s or ""))
    s = _EMOJI.sub(" ", _JUNK.sub("", _HTML.sub(" ", s)))
    s = re.sub(r"[=\-_*~#+|]{3,}", " ", s)
    return re.sub(r"\s+", " ", re.sub("[\u00a0【】]", " ", s)).strip()


_RDR = None
_LOCK = threading.Lock()  # VnCoreNLP (Java) không an toàn khi gọi song song từ nhiều thread của API


def segment(text):
    global _RDR
    with _LOCK:
        if _RDR is None:
            import py_vncorenlp
            cwd = os.getcwd()
            try:
                _RDR = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_DIR)
            finally:
                os.chdir(cwd)  # py_vncorenlp chdir vào save_dir và không đổi lại
        text = " ".join((text or "").lower().split()[:MAX_SEG_SYLLABLES])
        return " ".join(_RDR.word_segment(text)) if text else ""


def preprocess_query(query):
    """Search query -> model input, same pipeline as product `text_seg` (clean + segment)."""
    return segment(clean_text(query))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="đường dẫn train.jsonl")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    bad = total = 0
    with open(a.train, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if segment(row["text"]) != row["text_seg"]:
                bad += 1
                if bad <= 3:
                    print("LỆCH:", row["id"], "\n  ", segment(row["text"]), "\n  ", row["text_seg"])
            if total >= a.n:
                break
    print(f"{total - bad}/{total} dòng khớp text_seg")