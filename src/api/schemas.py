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
    score: float | None = None         # điểm xếp hạng: cosine thô, hoặc z-score trộn nếu có cả chữ lẫn ảnh
    score_text: float | None = None    # cosine text -> image (None nếu query không có chữ)
    score_image: float | None = None   # cosine image -> image (None nếu query không có ảnh)


class Facet(BaseModel):
    name: str
    count: int


class SearchOut(BaseModel):
    items: list[Item]
    total: int
    browse: bool = False            # True = không có truy vấn, đang duyệt toàn bộ sản phẩm
    page: int
    parsed: list[str]   # chip "Hệ thống hiểu"
    facets: list[Facet] = []        # màu có trong kết quả hiện tại, kèm số lượng
    query_colors: list[str] = []    # màu chuẩn tìm thấy trong mô tả
    dropped: list[str] = []         # màu trong mô tả bị bỏ vì xung đột với bộ lọc màu
    took_ms: float


class ProductOut(Item):
    description: str = ""
    sizes: list[str] = []