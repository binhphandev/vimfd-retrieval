import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
INDEX_DIR = os.path.join(ROOT_DIR, "index")

# --- Product schema ---
PRODUCT_ID_FIELD = "id"

# --- Image ---
IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275,  0.40821073)  # CLIP ViT-B/32
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)  # Đổi nếu dùng encoder khác

IMAGE_PRIMARY_FIELD = "is_featured"
IMAGE_POSITION_FIELD = "position"
IMAGE_LIST_FIELD = "images"

# --- Text / PhoBERT ---
PHOBERT_MODEL = "vinai/phobert-base"  # hoặc phobert-large
MAX_TEXT_LENGTH = 256
TEXT_FIELDS = ["name", "description", "category", "tags"]

# --- Embedding ---
EMBED_DIM = 256  # ProjectionHead output, L2-normalized

# --- Data split ---
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

# --- Checkpoint ---
# Format: { "model_state_dict", "logit_scale", "epoch", "val_loss" }
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "best.pt")
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