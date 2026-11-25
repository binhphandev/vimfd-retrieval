"""
Chạy từ thư mục gốc dự án:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/health
    POST /api/search                 (multipart/form-data)
    GET  /api/products/{id}
    GET  /api/products/{id}/similar
    /                                (phục vụ thư mục frontend/ nếu có)
"""

import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.api.schemas import Item, ProductOut, SearchOut
from src.serving.engine import SearchEngine
import config as cfg

log = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SearchEngine()
    yield


app = FastAPI(title="ViMFD Fashion Retrieval API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _engine(request: Request) -> SearchEngine:
    return request.app.state.engine


def _read_image(f: UploadFile) -> Image.Image:
    if f.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Ảnh phải là JPG, PNG hoặc WEBP")
    data = f.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ảnh tối đa 5 MB")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh")


# Dùng `def` (không phải async def) để FastAPI chạy inference trong threadpool,
# không chặn event loop.
@app.get("/api/health")
def health(request: Request):
    e = _engine(request)
    return {
        "status": "ok",
        "products": len(e.pids),
        "device": str(e.dev),
        "checkpoint": Path(cfg.CHECKPOINT_BEST).name,
        "split_dir": Path(cfg.SPLIT_DIR).name,
        "index": e.meta,
    }


@app.post("/api/search", response_model=SearchOut)
def search(
    request: Request,
    q: str = Form(""),
    image: UploadFile | None = File(None),
    w: float = Form(0.5, ge=0, le=1),          # 0 = chỉ theo mô tả, 1 = chỉ theo ảnh
    colors: list[str] = Form(default=[]),
    price_min: float | None = Form(None),
    price_max: float | None = Form(None),
    sort: str = Form("rel", pattern="^(rel|asc|desc)$"),
    page: int = Form(1, ge=1),
    page_size: int = Form(12, ge=1, le=48),
):
    img = _read_image(image) if image is not None and image.filename else None
    try:
        return _engine(request).search(
            text=q, image=img, w=w, colors=colors,
            price_min=price_min, price_max=price_max,
            sort=sort, page=page, page_size=page_size,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/products/{pid}", response_model=ProductOut)
def product(request: Request, pid: str):
    p = _engine(request).get(pid)
    if p is None:
        raise HTTPException(404, "Không tìm thấy sản phẩm")
    return p


@app.get("/api/products/{pid}/similar", response_model=list[Item])
def similar(request: Request, pid: str, limit: int = 4):
    r = _engine(request).similar(pid, k=min(max(limit, 1), 24))
    if r is None:
        raise HTTPException(404, "Không tìm thấy sản phẩm")
    return r


# Ảnh sản phẩm local (IMAGE_DIR) -> /media/<đường dẫn tương đối>
_media = Path(cfg.IMAGE_DIR)
if _media.is_dir():
    app.mount("/media", StaticFiles(directory=_media), name="media")

# Phải đặt SAU các route /api để không che chúng.
_web = ROOT / "frontend"
if _web.is_dir():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")