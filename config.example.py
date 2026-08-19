# config.example.py — ViMFD Project
# Copy file này thành config.py và chỉnh lại nếu cần.

import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
INDEX_DIR = os.path.join(ROOT_DIR, "index")

for _dir in [DATA_DIR, IMAGE_DIR, PROCESSED_DIR, CHECKPOINT_DIR, INDEX_DIR]:
    os.makedirs(_dir, exist_ok=True)

# --- Product schema ---
PRODUCT_ID_FIELD = "id"

# --- Image ---
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP ViT-B/32 — đổi nếu dùng encoder khác
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_PRIMARY_FIELD = "is_featured"
IMAGE_POSITION_FIELD = "position"
IMAGE_LIST_FIELD = "images"

# --- Text / PhoBERT ---
PHOBERT_MODEL = "vinai/phobert-base"  # hoặc phobert-large
MAX_TEXT_LENGTH = 256
TEXT_FIELDS = ["name", "description", "category", "tags"]

# --- Model architecture ---
CLIP_MODEL = "openai/clip-vit-base-patch32"
VISUAL_BACKBONE_DIM = 768
TEXT_BACKBONE_DIM = 768
EMBED_DIM = 256  # ProjectionHead output, L2-normalized
UNFREEZE_LAST_N = 2  # thử 4, 6 sau khi smoke test ổn
PROJECTION_DROPOUT = 0.1
LOGIT_SCALE_INIT = 14.29  # log(1/0.07)
LOGIT_SCALE_MAX = 100.0

# --- Training ---
BATCH_SIZE = 64  # giảm xuống BATCH_SIZE_CPU nếu chạy CPU
BATCH_SIZE_CPU = 4
NUM_EPOCHS = 30
WARMUP_EPOCHS = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# --- Data split ---
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

# --- Checkpoint ---
# Format: { "model_state_dict", "logit_scale", "epoch", "val_loss" }
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "best.pt")
CHECKPOINT_LATEST = os.path.join(CHECKPOINT_DIR, "latest.pt")
CKPT_MODEL_KEY = "model_state_dict"
CKPT_LOGIT_SCALE_KEY = "logit_scale"
CKPT_EPOCH_KEY = "epoch"
CKPT_VAL_LOSS_KEY = "val_loss"

# --- FAISS index ---
# id_mapping.json format: { "0": "id_A", "1": "id_B", ... }
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "product.index")
FAISS_MAPPING_FILE = os.path.join(INDEX_DIR, "id_mapping.json")

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000
TOP_K = 10
MAX_TOP_K = 50
PRODUCT_META_FIELDS = ["id", "name", "price", "images"]