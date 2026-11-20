from pydantic import BaseModel


class Item(BaseModel):
    id: str
    name: str
    price: float
    brand: str = ""
    category: str = ""
    image_url: str = ""
    url: str = ""          # trang sản phẩm trên cửa hàng
    source: str = ""       # tiki / lazada / tên shop local
    colors: list[str] = []
    score: float | None = None
    score_text: float | None = None    # None nếu query không có chữ / sản phẩm thiếu text
    score_image: float | None = None   # None nếu query không có ảnh / sản phẩm thiếu ảnh


class SearchOut(BaseModel):
    items: list[Item]
    total: int
    page: int
    parsed: list[str]   # chip "Hệ thống hiểu"
    took_ms: float


class ProductOut(Item):
    description: str = ""
    sizes: list[str] = []
