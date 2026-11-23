import { S, nF } from './state.js';

export const $ = s => document.querySelector(s);
export const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const money = n => Number(n).toLocaleString('vi-VN') + 'đ';
export const cap = w => w ? w[0].toUpperCase() + w.slice(1) : w;
export const kre = w => new RegExp('(^|\\s)' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(\\s|$)', 'i');
const num = v => (v == null ? '—' : v.toFixed(2));

let tt;
export function toast(m) {
  const t = $('#toast');
  t.textContent = m;
  t.classList.add('on');
  clearTimeout(tt);
  tt = setTimeout(() => t.classList.remove('on'), 3200);
}

export const cardH = (p, hi, res) =>
  `<a class="card" href="#detail" data-id="${esc(p.id)}"><div class="im">Ảnh sản phẩm${
    p.image_url ? `<img src="${esc(p.image_url)}" alt="" loading="lazy" decoding="async">` : ''}</div>
  <div class="bd"><h3>${esc(p.name)}</h3>
  <div class="pr"><b>${money(p.price)}</b><span>${esc(p.brand)}</span></div>
  ${p.score == null ? '' : `<span class="bg${hi ? ' hi' : ''}">${hi ? 'Rất phù hợp' : 'Phù hợp'}</span>`}
  ${res && p.score != null ? `<span class="sco">Chữ ${num(p.score_text)} · Ảnh ${num(p.score_image)} · Tổng ${num(p.score)}</span>` : ''}
  </div></a>`;

const chipsH = () =>
  (S.parsed.length || S.file ? '<span class="lb">Hệ thống hiểu:</span>' : '') +
  S.parsed.map(w => `<button class="chip" data-k="${esc(w)}">${esc(cap(w))} ×</button>`).join('') +
  (S.file ? '<span class="chip st">Ảnh mẫu</span>' : '');

const SKELETON = Array(8).fill('<div class="card sk"><div class="im"></div><div class="bd"><i></i><i></i></div></div>').join('');

export function renderResults() {
  const { items, st, total } = S, g = $('#grid');
  $('#cnt').textContent = st === 'load' ? 'Đang tải…'
    : st === 'err' ? 'Không tải được kết quả'
    : `${(st === 'empty' ? 0 : total).toLocaleString('vi-VN')} sản phẩm${S.browse ? '' : ' phù hợp'}`;
  $('#und').innerHTML = chipsH();
  const nt = $('#note');
  nt.hidden = !S.dropped.length;
  nt.textContent = S.dropped.length ? `Đã bỏ “${S.dropped.join(', ')}” khỏi mô tả vì bạn đang lọc màu ${[...S.F.c].join(', ')}.` : '';
  $('#wt').hidden = !(S.q && S.file);   // thanh ưu tiên chỉ có nghĩa khi truy vấn cả chữ lẫn ảnh
  $('#fbtn').textContent = 'Bộ lọc' + (nF() ? ` (${nF()})` : '');

  if (st === 'load') g.innerHTML = SKELETON;
  else if (st === 'ok') {
    S.hiIds = S.browse ? new Set() : new Set([...items].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 4).map(p => p.id));
    g.innerHTML = items.map(p => cardH(p, S.hiIds.has(p.id), S.res)).join('');
  } else if (st === 'err') {
    g.innerHTML = `<div class="msg"><div class="ic2">!</div>
      <b>${S.err ? 'Không tải được kết quả' : 'Không kết nối được tới hệ thống'}</b>
      <p>${S.err ? esc(S.err) : 'Kiểm tra kết nối mạng rồi thử lại. Truy vấn của bạn vẫn được giữ nguyên.'}</p>
      <div class="acts"><button class="btn p" data-act="retry">Thử lại</button></div></div>`;
  } else {
    g.innerHTML = `<div class="msg"><div class="ic2">?</div>
      <b>Không tìm thấy sản phẩm phù hợp</b>
      <p>Thử bỏ bớt bộ lọc, đổi cách mô tả, hoặc thêm ảnh tham chiếu để hệ thống hiểu rõ hơn.</p>
      <div class="acts"><button class="btn p" data-act="clearF"${nF() ? '' : ' disabled'}>Xóa bộ lọc</button><button class="btn" data-act="addimg">Thêm ảnh tham chiếu</button></div>
      <div class="sg"><span class="lb">Có thể bạn muốn tìm</span><div class="chips">
        ${['Áo khoác nam', 'Áo khoác gió', 'Áo phao nhẹ'].map(t => `<button class="chip o" data-ex="${t}">${t}</button>`).join('')}</div></div></div>`;
  }
  $('#more').hidden = !(st === 'ok' && items.length < total);
}

export function renderDetail(p, sim = [], full = null) {
  S.dcache = [p, sim, full];
  const d = full || p;
  const why = p.score_text != null && p.score_image != null ? 'Mô tả và hình ảnh của sản phẩm đều gần với truy vấn của bạn.'
    : p.score_image != null ? 'Hình ảnh sản phẩm gần với ảnh mẫu bạn đã tải lên.'
    : p.score_text != null ? 'Mô tả sản phẩm khớp với từ khóa của bạn.'
    : 'Sản phẩm có mô tả và hình ảnh tương tự.';
  const chips = [p.category, p.brand, p.source, ...(p.colors || []).slice(0, 3)].filter(Boolean)
    .map(c => `<span class="chip">${esc(c)}</span>`).join('');
  const sizes = (d.sizes || []).length ? `<div class="chips">${d.sizes.map(s => `<span class="chip">${esc(s)}</span>`).join('')}</div>` : '';
  const rs = S.res
    ? `<div class="rt"><span>Điểm theo mô tả</span><b>${num(p.score_text)}</b><span>Điểm theo ảnh</span><b>${num(p.score_image)}</b><span>Tổng hợp</span><b>${num(p.score)}</b></div>`
    : '<button class="lk u" data-act="score">Xem điểm chi tiết (chế độ nghiên cứu)</button>';
  $('#dt').innerHTML = `<div class="dg">
  <div class="pim">Ảnh sản phẩm${p.image_url ? `<img src="${esc(p.image_url)}" alt="${esc(p.name)}" decoding="async">` : ''}</div>
  <div class="inf"><h1>${esc(p.name)}</h1>
   <div class="pr big"><b>${money(p.price)}</b><span>${esc(p.brand)}</span></div>
   ${S.hiIds.has(p.id) ? '<span class="bg hi">Rất phù hợp với tìm kiếm của bạn</span>' : ''}
   ${p.score == null ? '' : `<div class="why"><h2>Vì sao ra kết quả này</h2><p>${why}</p><div class="chips">${chips}</div>${rs}</div>`}
   ${d.description ? `<div class="why"><h2>Mô tả</h2><p class="desc">${esc(d.description)}</p>${sizes}</div>` : ''}
   <button class="btn p" data-act="shop">Xem trên cửa hàng</button>
   <button class="btn" data-act="sim">Tìm tương tự</button></div></div>
  ${sim.length ? `<h2 class="sh">Sản phẩm tương tự</h2><div class="grid sim">${sim.map(s => cardH(s, false, S.res)).join('')}</div>` : ''}`;
}