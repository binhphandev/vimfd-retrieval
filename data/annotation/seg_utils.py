"""
seg_utils.py — Bộ tách từ truy vấn DÙNG CHUNG cho V4, V7, V8 (và check_segmentation.py).

Vì sao cần: text_seg của sản phẩm (đầu vào PhoBERT, TF-IDF, BM25) được tạo bằng VnCoreNLP (RDRSegmenter) theo
báo cáo xử lý dữ liệu. Kế hoạch gốc (mục 3.2) ghi underthesea; nếu truy vấn tách bằng underthesea còn tài liệu
tách bằng VnCoreNLP thì từ ghép có thể khác dạng ("áo_sơ_mi" so với "áo sơ_mi"), làm TF-IDF/BM25 khớp sai và
PhoBERT nhìn thấy truy vấn lệch phân phối so với lúc huấn luyện. => Mọi truy vấn phải qua CÙNG MỘT hàm này.

Thứ tự thử (biến môi trường VIMFD_SEGMENTER, mặc định "auto"):
  1. src.serving.text_utils.preprocess_query   (đúng pipeline của API serving, VnCoreNLP)
  2. py_vncorenlp                              (cần Java + thư mục model; đặt VNCORENLP_DIR hoặc config.VNCORENLP_DIR)
  3. KHÔNG tự lùi về underthesea. Chỉ dùng underthesea khi đặt VIMFD_SEGMENTER=underthesea
     hoặc VIMFD_ALLOW_UNDERTHESEA=1 (và khi đó phải ghi vào mục 12 của kế hoạch).

Dùng:
    from seg_utils import segment_text, segmenter_name
    segment_text("Áo sơ mi nam màu đen")     # -> "áo_sơ_mi nam màu đen" (ví dụ)
Thử nhanh:
    python seg_utils.py "áo sơ mi nam màu đen" "quần jean ống suông"
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

# Để import được config.py và src.* dù chạy từ thư mục nào
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, *list(_HERE.parents)[:3], Path.cwd()]:
    if (_cand / "config.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

_state = {"fn": None, "name": None}


def normalize(text) -> str:
    """Mục 3.2: Unicode NFC, chữ thường, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def _try_serving():
    from src.serving.text_utils import preprocess_query      # pipeline API: chuẩn hóa + VnCoreNLP
    out = preprocess_query("Áo sơ mi nam màu đen")
    if not isinstance(out, str) or not out.strip():
        raise RuntimeError(f"preprocess_query trả về {type(out).__name__}, không phải chuỗi")
    return preprocess_query, "src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)"


def _try_vncorenlp():
    import py_vncorenlp
    try:
        import config
    except ImportError:
        config = None
    d = os.environ.get("VNCORENLP_DIR") or (getattr(config, "VNCORENLP_DIR", None) if config else None)
    if not d:
        raise RuntimeError("Chưa đặt VNCORENLP_DIR (thư mục model VnCoreNLP, đường dẫn tuyệt đối)")
    cwd = os.getcwd()
    seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(Path(d).resolve()))
    os.chdir(cwd)                                             # py_vncorenlp có thể đổi thư mục làm việc

    def fn(t):
        return " ".join(seg.word_segment(t))
    return fn, "py_vncorenlp (RDRSegmenter)"


def _use_underthesea():
    from underthesea import word_tokenize

    def fn(t):
        return word_tokenize(t, format="text")
    return fn, "underthesea (KHÔNG khớp text_seg nếu dữ liệu tách bằng VnCoreNLP — ghi mục 12)"


def get_segmenter():
    if _state["fn"] is not None:
        return _state["fn"], _state["name"]
    mode = os.environ.get("VIMFD_SEGMENTER", "auto").lower()
    errors = []
    attempts = []
    if mode in ("auto", "serving"):
        attempts.append(("serving", _try_serving))
    if mode in ("auto", "vncorenlp"):
        attempts.append(("vncorenlp", _try_vncorenlp))
    if mode == "underthesea" or os.environ.get("VIMFD_ALLOW_UNDERTHESEA") == "1":
        attempts.append(("underthesea", _use_underthesea))
    for label, loader in attempts:
        try:
            fn, name = loader()
            _state["fn"], _state["name"] = fn, name
            if label == "underthesea":
                print(f"  [CẢNH BÁO] Bộ tách từ: {name}")
            return fn, name
        except Exception as e:                                # noqa: BLE001 - muốn gom mọi lỗi để báo rõ
            errors.append(f"  - {label}: {type(e).__name__}: {e}")
    raise SystemExit(
        "Không khởi tạo được bộ tách từ khớp text_seg (VnCoreNLP).\n" + "\n".join(errors) + "\n"
        "Cách xử lý: chạy từ thư mục ROOT (chứa config.py) để import được src.serving.text_utils, hoặc đặt\n"
        "VNCORENLP_DIR=<thư mục model VnCoreNLP> (cần Java). Chỉ khi chấp nhận lệch so với text_seg mới đặt\n"
        "VIMFD_ALLOW_UNDERTHESEA=1 và ghi vào mục 12.")


def segmenter_name() -> str:
    return get_segmenter()[1]


def segment_text(text: str) -> str:
    """Chuẩn hóa (mục 3.2) rồi tách từ bằng bộ tách từ thống nhất."""
    fn, _ = get_segmenter()
    return fn(normalize(text))


if __name__ == "__main__":
    samples = sys.argv[1:] or ["áo sơ mi nam màu đen", "quần jean ống suông", "thắt lưng da bò"]
    print("Bộ tách từ:", segmenter_name())
    for s in samples:
        print(f"  {s!r} -> {segment_text(s)!r}")
