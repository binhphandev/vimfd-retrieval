import { search, getProduct, getSimilar } from './api.js';
import { S, PRICE_MIN, PRICE_MAX } from './state.js';
import { $, esc, money, cap, kre, toast, renderResults, renderDetail } from './ui.js';

const HEX = { 'trắng': '#fff', 'đen': '#1b1b1b', 'xám': '#8a8f98', 'xanh navy': '#1f2f55', 'xanh dương': '#3b6fd4', 'xanh rêu': '#6b7a3a',
  'xanh lá': '#3f9a5a', 'xanh': '#5b8fb9', 'be': '#d9c7a5', 'nâu': '#7a5a3a', 'đỏ': '#c0392b', 'vàng': '#e0b43c', 'hồng': '#e8a0b4', 'tím': '#7b5ea7', 'cam': '#e67e22' };
const MAX_IMG = 5 * 1024 * 1024;
const hasQuery = () => !!(S.q || S.file);

// ---------- tìm kiếm ----------
let seq = 0;   // bỏ qua phản hồi cũ nếu người dùng đã tìm lại

async function run({ append = false, silent = false } = {}) {
  const my = ++seq;
  if (append) S.page++;
  else {
    S.page = 1;
    if (!silent) { S.st = 'load'; renderResults(); }
  }
  try {
    const r = await search({
      q: S.q, image: S.file, w: S.w, sort: S.sort, page: S.page,
      colors: [...S.F.c],
      priceMin: S.F.min > PRICE_MIN ? S.F.min : null,
      priceMax: S.F.max < PRICE_MAX ? S.F.max : null,
    });
    if (my !== seq) return;
    r.items.forEach(p => S.byId.set(p.id, p));
    S.items = append ? S.items.concat(r.items) : r.items;
    S.total = r.total;
    S.parsed = r.parsed;
    S.browse = r.browse;
    S.facets = r.facets;
    S.qcolors = r.query_colors;
    S.dropped = r.dropped;
    S.st = S.items.length ? 'ok' : 'empty';
  } catch (e) {
    if (my !== seq) return;
    if (append) { S.page--; toast(e.message); }
    else { S.st = 'err'; S.err = e instanceof TypeError ? '' : e.message; }   // TypeError = mất kết nối
  }
  renderResults();
}

function go() {
  S.ok = true;
  nav('results');
  run();
}

// ---------- chi tiết ----------
async function showDetail() {
  const id = S.cur, p = S.byId.get(id);
  if (!p) return nav('results');
  renderDetail(p);
  scrollTo(0, 0);
  try {
    const [full, sim] = await Promise.all([getProduct(id), getSimilar(id, 4)]);
    sim.forEach(s => S.byId.set(s.id, s));
    if (S.cur === id && !$('#v-detail').hidden) renderDetail(p, sim, full);
  } catch { /* giữ phần thông tin cơ bản đã hiển thị */ }
}

// ---------- routing theo hash (#results, #detail) ----------
function route() {
  const h = location.hash.slice(1);
  let v = 'home';
  if (h === 'results' && S.ok) v = 'results';
  else if (h === 'detail' && S.cur != null) v = 'detail';
  for (const x of ['home', 'results', 'detail']) $('#v-' + x).hidden = x !== v;
  document.body.dataset.v = v;
  $('#sf').hidden = v === 'detail';
  if (v !== 'detail') {
    const focused = document.activeElement === $('#q');   // di chuyển ô tìm kiếm sẽ làm mất focus
    $(v === 'results' ? '#sr' : '#sh').appendChild($('#sf'));
    if (focused) $('#q').focus();
  }
  if (v === 'detail') showDetail();
  else scrollTo(0, 0);
}
const nav = h => (location.hash === '#' + h ? route() : (location.hash = h));

function setRes(v) {
  S.res = v;
  $('#sw').setAttribute('aria-checked', v);
  if (!$('#v-results').hidden) renderResults();
  if (!$('#v-detail').hidden && S.dcache) renderDetail(...S.dcache);
}

function setFile(f) {
  if (S.preview) URL.revokeObjectURL(S.preview);
  S.file = f;
  S.preview = f ? URL.createObjectURL(f) : null;
  $('#th').style.backgroundImage = S.preview ? `url(${S.preview})` : '';
  $('#sf').classList.toggle('has-img', !!f);
}

// ---------- bộ lọc ----------
const pct = v => ((v - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)) * 100 + '%';
function fsync() {
  const F = S.F;
  renderColors();
  for (const [id, v] of [['pmin', F.min], ['pmax', F.max]]) {
    const e = $('#' + id);
    e.value = v;
    e.style.setProperty('--v', pct(v));
  }
  $('#vmin').textContent = money(F.min);
  $('#vmax').textContent = money(F.max) + (F.max >= PRICE_MAX ? '+' : '');
}
// Chỉ hiện màu thực sự có trong kết quả hiện tại (kèm số lượng), không dùng danh sách cố định
function renderColors() {
  $('#fcol').innerHTML = S.facets.length
    ? S.facets.map(f => `<button class="cl" data-c="${esc(f.name)}" aria-pressed="${S.F.c.has(f.name)}"><i style="background:${HEX[f.name] || '#ccc'}"></i>${esc(cap(f.name))}<small>${f.count}</small></button>`).join('')
    : '<span class="hint">Chưa có dữ liệu màu, hãy tìm kiếm trước.</span>';
  const h = $('#fhint');
  h.hidden = !S.qcolors.length;
  h.textContent = `Mô tả của bạn đã có màu: ${S.qcolors.join(', ')}. Chọn màu khác sẽ thay cho màu trong mô tả.`;
}
const fkey = () => JSON.stringify([[...S.F.c].sort(), S.F.min, S.F.max]);
let fOpen = '';

$('#fd').addEventListener('click', e => {
  const b = e.target.closest('[data-c]');
  if (!b) return;
  const set = S.F.c, k = b.dataset.c;
  set.has(k) ? set.delete(k) : set.add(k);
  fsync();
});
$('#pmin').oninput = e => { S.F.min = Math.min(+e.target.value, S.F.max); fsync(); };
$('#pmax').oninput = e => { S.F.max = Math.max(+e.target.value, S.F.min); fsync(); };
$('#fd').onclose = () => { if (fkey() !== fOpen && S.ok) run(); else renderResults(); };
for (const d of document.querySelectorAll('dialog'))
  d.addEventListener('click', e => { if (e.target === d) d.close(); });

// ---------- điều khiển ----------
let wt;
$('#w').oninput = e => {
  S.w = +e.target.value;
  e.target.style.setProperty('--v', S.w + '%');
  clearTimeout(wt);
  if (S.ok && S.q && S.file) wt = setTimeout(() => run({ silent: true }), 300);   // gộp điểm ở server
};
$('#w').style.setProperty('--v', '55%');
$('#sort').onchange = e => { S.sort = e.target.value; if (S.ok) run(); };
$('#sw').onclick = () => setRes(!S.res);
$('#thx').onclick = () => {
  setFile(null);
  const t = $('#q').value.trim();
  if (!t) return S.ok ? clearResults() : undefined;   // hết cả chữ lẫn ảnh -> xóa kết quả, ở lại màn hình kết quả
  S.q = t;
  go();                                            // còn chữ -> tìm lại chỉ bằng chữ
};
$('#file').onchange = e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  if (!/^image\/(jpeg|png|webp)$/.test(f.type) || f.size > MAX_IMG)
    return toast('Ảnh cần là JPG, PNG hoặc WEBP và tối đa 5 MB.');
  setFile(f);
};
$('#sf').onsubmit = e => {
  e.preventDefault();
  S.q = $('#q').value.trim();
  if (!hasQuery()) return S.ok ? clearResults() : $('#q').focus();
  go();
};

document.addEventListener('click', e => {
  const c = e.target.closest('.card[data-id]');
  if (c) {
    S.cur = c.dataset.id;
    if (location.hash === '#detail') { e.preventDefault(); showDetail(); }   // bấm thẻ "tương tự" ngay trong trang chi tiết
    return;
  }
  const k = e.target.closest('[data-k]');
  if (k) {
    S.q = S.q.replace(kre(k.dataset.k), ' ').replace(/\s+/g, ' ').trim();
    $('#q').value = S.q;
    return hasQuery() ? go() : clearResults();
  }
  const x = e.target.closest('[data-ex]');
  if (x) { $('#q').value = x.dataset.ex; return $('#sf').requestSubmit(); }
  const a = e.target.closest('[data-act]');
  if (!a) return;
  switch (a.dataset.act) {
    case 'home': e.preventDefault(); S.ok = false; location.hash = ''; route(); break;
    case 'back': nav('results'); break;
    case 'more': run({ append: true }); break;
    case 'retry': run(); break;
    case 'filter': fsync(); fOpen = fkey(); $('#fd').showModal(); break;
    case 'how': $('#hd').showModal(); break;
    case 'close': a.closest('dialog').close(); break;
    case 'clearF':
      S.F = { c: new Set(), min: PRICE_MIN, max: PRICE_MAX };
      fsync();
      if (!$('#fd').open && S.ok) run();
      break;
    case 'addimg': $('#file').click(); break;
    case 'score': setRes(true); break;
    case 'shop': {
      const u = S.byId.get(S.cur)?.url;
      u ? window.open(u, '_blank', 'noopener') : toast('Sản phẩm này chưa có liên kết cửa hàng.');
      break;
    }
    case 'sim': {
      const p = S.byId.get(S.cur);
      if (!p) break;
      setFile(null);
      S.q = p.name;
      $('#q').value = S.q;
      go();
      break;
    }
  }
});

// Ảnh sản phẩm lỗi (link hỏng) thì bỏ thẻ <img>, giữ khung xám bên dưới
document.addEventListener('error', e => { if (e.target.tagName === 'IMG') e.target.remove(); }, true);

// Xóa hết truy vấn (chữ + ảnh) = bỏ điều kiện tìm kiếm: ở lại màn hình kết quả và hiển thị
// toàn bộ sản phẩm (truy vấn rỗng -> backend trả chế độ "duyệt"). Bộ lọc và sắp xếp được giữ lại.
function clearResults() {
  clearTimeout(wt);
  Object.assign(S, { q: '', parsed: [], qcolors: [], dropped: [] });
  run();
}
$('#q').addEventListener('input', e => {
  if (S.ok && !$('#v-results').hidden && !e.target.value.trim() && !S.file) clearResults();
});

addEventListener('hashchange', route);
route();