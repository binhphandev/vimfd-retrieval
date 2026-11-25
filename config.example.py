import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")
# train/val/test.jsonl + split.json (split có gom nhóm theo ảnh gần trùng; group_id nằm trong split.json).
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
INDEX_DIR = os.path.join(ROOT_DIR, "index")
# Model VnCoreNLP (VnCoreNLP-1.2.jar + models/) để tách từ truy vấn trong API.
# Mặc định là thư mục vncorenlp ở gốc dự án; đặt biến môi trường VNCORENLP_DIR để dùng nơi khác.
VNCORENLP_DIR = os.environ.get("VNCORENLP_DIR") or os.path.join(ROOT_DIR, "vncorenlp")

for _dir in [DATA_DIR, IMAGE_DIR, PROCESSED_DIR, SPLIT_DIR, CHECKPOINT_DIR, INDEX_DIR]:
    os.makedirs(_dir, exist_ok=True)

# --- Product schema ---
PRODUCT_ID_FIELD = "id"
BRAND_FIELD = "shop"          # engine.py: trường hiển thị là "brand" trong kết quả
CATEGORY_FIELD = "category"   # engine.py: áo / quần / phụ kiện

# --- Image ---
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP ViT-B/32 — đổi nếu dùng encoder khác
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_PRIMARY_FIELD = "is_featured"   # dữ liệu mới không có trường này -> dùng position nhỏ nhất (từ 1)
IMAGE_POSITION_FIELD = "position"
IMAGE_LIST_FIELD = "images"
IMAGE_PATH_FIELD = "local_path"

# --- Text / PhoBERT ---
PHOBERT_MODEL = "vinai/phobert-base"  # hoặc phobert-large
MAX_TEXT_LENGTH = 256
# Text sản phẩm đã tách từ VnCoreNLP, chữ thường. Truy vấn API đi qua src/serving/text_utils.preprocess_query.
MODEL_TEXT_FIELD = "text_seg"

# --- Model architecture ---
CLIP_MODEL = "openai/clip-vit-base-patch32"
VISUAL_BACKBONE_DIM = 768
TEXT_BACKBONE_DIM = 768
EMBED_DIM = 256  # ProjectionHead output, L2-normalized
UNFREEZE_LAST_N = 2  # thử 4, 6 sau khi smoke test ổn
PROJECTION_DROPOUT = 0.1
LOGIT_SCALE_INIT = 2.65926  # log(1/0.07); model.exp() gives an initial scale near 14.29
LOGIT_SCALE_MAX = 100.0

# --- Training ---
# train.py: early stopping (patience 5, --patience 0 để tắt); checkpoint tốt nhất theo T→I R@10 mức nhóm; train loader dùng GroupBatchSampler.
# train_colab.ipynb ghi đè NUM_EPOCHS = 20 lúc chạy.
BATCH_SIZE = 64
NUM_EPOCHS = 30
NUM_WORKERS = 2
WARMUP_EPOCHS = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
# Xác suất thay text đầy đủ bằng text ngắn (tên / vài từ đầu / + màu) khi train, để khớp truy vấn thật.
# 0 = tắt. dataset.build_dataloaders đọc giá trị này; best checkpoint chọn theo train.py --select mix.
SHORT_TEXT_PROB = 0.5

# --- Evaluation ---
EVAL_BATCH_SIZE = 64
EVALUATION_SPLIT = "test"   # evaluate.py luôn chấm cả gallery "full" và "test-only"

# --- Checkpoint ---
# Format: { "model_state_dict", "logit_scale", "epoch", "val_loss" }
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "best_short_24112026.pt")
CHECKPOINT_LATEST = os.path.join(CHECKPOINT_DIR, "latest.pt")
CKPT_MODEL_KEY = "model_state_dict"
CKPT_LOGIT_SCALE_KEY = "logit_scale"
CKPT_EPOCH_KEY = "epoch"
CKPT_VAL_LOSS_KEY = "val_loss"

# --- FAISS index ---
# id_mapping.json format: { "0": "id_A", "1": "id_B", ... }
# Index text: build_index.py vẫn tạo, nhưng API KHÔNG dùng.
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "product.index")
FAISS_MAPPING_FILE = os.path.join(INDEX_DIR, "id_mapping.json")
# Index ảnh (chỉ ảnh đại diện): API và evaluate.py dùng cho mọi truy vấn (Text→Image, Image→Image).
FAISS_IMAGE_INDEX_FILE = os.path.join(INDEX_DIR, "product_image.index")
FAISS_IMAGE_MAPPING_FILE = os.path.join(INDEX_DIR, "id_mapping_image.json")

# --- API ---
# True = cho phép khởi động khi index lệch checkpoint/dữ liệu (chỉ cảnh báo). Mặc định từ chối.
ALLOW_STALE_INDEX = False

# --- Hiện không được mã nguồn đã kiểm tra sử dụng (giữ lại để tương thích) ---
BATCH_SIZE_CPU = 4
GALLERY_MODE = "test-only"
RANDOM_SEED = 42           # seed chia split nằm ở build_splits.py (vimfd-v4), train.py không đặt seed
API_HOST = "0.0.0.0"       # uvicorn nhận host/port từ dòng lệnh
API_PORT = 8000
TOP_K = 10
MAX_TOP_K = 50             # engine.py dùng hằng số CANDIDATES = 200
PRODUCT_META_FIELDS = ["id", "name", "price", "images"]