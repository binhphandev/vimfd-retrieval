// Gọi backend. Nhớ giữ đối tượng File gốc của ảnh (không phải blob URL) để gửi lên.
export async function search({ q = '', image = null, w = 55, types = [], colors = [],
                               priceMin = null, priceMax = null, sort = 'rel', page = 1 } = {}) {
  const fd = new FormData();
  fd.append('q', q);
  if (image) fd.append('image', image);
  fd.append('w', String(w / 100));            // thanh trượt 0–100 -> 0–1
  types.forEach(t => fd.append('types', t));
  colors.forEach(c => fd.append('colors', c));
  if (priceMin != null) fd.append('price_min', priceMin);
  if (priceMax != null) fd.append('price_max', priceMax);
  fd.append('sort', sort);
  fd.append('page', page);
  return request('/api/search', { method: 'POST', body: fd });
}

export const getProduct = id => request(`/api/products/${encodeURIComponent(id)}`);
export const getSimilar = (id, limit = 4) =>
  request(`/api/products/${encodeURIComponent(id)}/similar?limit=${limit}`);

async function request(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || r.statusText);   // hiển thị ở trạng thái lỗi / toast
  }
  return r.json();
}
