export const PRICE_MIN = 0;
export const PRICE_MAX = 2000000;   // thanh trượt giá: chạm hai đầu nghĩa là không lọc

export const S = {
  q: '',
  file: null,            // File gốc để gửi lên server
  preview: null,         // blob URL chỉ để xem trước
  items: [],             // kết quả đã tải (nhiều trang)
  total: 0,
  page: 1,
  parsed: [],            // chip "Hệ thống hiểu" do backend trả
  st: 'ok',              // load | ok | empty | err
  err: '',
  ok: false,             // đã tìm ít nhất một lần
  cur: null,             // id sản phẩm đang xem chi tiết
  dcache: null,          // [product, similar, full] của trang chi tiết
  byId: new Map(),       // mọi sản phẩm đã thấy, để mở chi tiết không cần gọi lại
  hiIds: new Set(),      // top 4 điểm cao nhất -> badge "Rất phù hợp"
  res: false,            // chế độ nghiên cứu
  w: 55,                 // thanh trượt 0–100 (100 = chỉ theo ảnh)
  sort: 'rel',
  F: { t: new Set(), c: new Set(), min: PRICE_MIN, max: PRICE_MAX },
};

export const nF = () =>
  S.F.t.size + S.F.c.size + (S.F.min > PRICE_MIN || S.F.max < PRICE_MAX ? 1 : 0);
