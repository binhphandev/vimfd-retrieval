import { search, getProduct, getSimilar } from './api.js';
import { S, PRICE_MIN, PRICE_MAX } from './state.js';
import { $, money, cap, kre, toast, renderResults, renderDetail } from './ui.js';

const TYPES = ['Áo sơ mi', 'Áo thun', 'Áo khoác', 'Quần'];
const COLORS = [['trắng', '#fff'], ['đen', '#1b1b1b'], ['xanh', '#2f4a7a'], ['be', '#d9c7a5'], ['xám', '#8a8f98'], ['nâu', '#7a5a3a']];
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
      types: [...S.F.t], colors: [...S.F.c],
      priceMin: S.F.min > PRICE_MIN ? S.F.min : null,
      priceMax: S.F.max < PRICE_MAX ? S.F.max : null,
    });
    if (my !== seq) return;
    r.items.forEach(p => S.byId.set(p.id, p));
    S.items = append ? S.items.concat(r.items) : r.items;
    S.total = r.total;
    S.parsed = r.parsed;
    S.st = S.items.length ? 'ok' : 'empty';
  } catch (e) {
    if (my !== seq) return;
    if (append) { S.page--; toast(e.message); }
    else { S.st = 'err'; S.err = e.message; }
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
  $('#sf').hidden = v === 'detail';
  if (v !== 'detail') $(v === 'results' ? '#sr' : '#sh').appendChild($('#sf'));
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
  document.querySelectorAll('#ftype .chip').forEach(b => b.classList.toggle('on', F.t.has(b.dataset.t)));
  document.querySelectorAll('#fcol .cl').forEach(b => b.setAttribute('aria-pressed', F.c.has(b.dataset.c)));
  for (const [id, v] of [['pmin', F.min], ['pmax', F.max]]) {
    const e = $('#' + id);
    e.value = v;
    e.style.setProperty('--v', pct(v));
  }
  $('#vmin').textContent = money(F.min);
  $('#vmax').textContent = money(F.max) + (F.max >= PRICE_MAX ? '+' : '');
}
const fkey = () => JSON.stringify([[...S.F.t].sort(), [...S.F.c].sort(), S.F.min, S.F.max]);
let fOpen = '';

$('#ftype').innerHTML = TYPES.map(t => `<button class="chip" data-t="${t}">${t}</button>`).join('');
$('#fcol').innerHTML = COLORS.map(c =>
  `<button class="cl" data-c="${c[0]}" style="background:${c[1]}" aria-label="${cap(c[0])}" aria-pressed="false"></button>`).join('');
$('#fd').addEventListener('click', e => {
  const b = e.target.closest('[data-t],[data-c]');
  if (!b) return;
  const set = b.dataset.t ? S.F.t : S.F.c, k = b.dataset.t || b.dataset.c;
  set.has(k) ? set.delete(k) : set.add(k);
  fsync();
});
$('#pmin').oninput = e => { S.F.min = Math.min(+e.target.value, S.F.max); fsync(); };
$('#pmax').oninput = e => { S.F.max = Math.max(+e.target.value, S.F.min); fsync(); };
$('#fd').onclose = () => { if (fkey() !== fOpen && hasQuery()) run(); else renderResults(); };
for (const d of document.querySelectorAll('dialog'))
  d.addEventListener('click', e => { if (e.target === d) d.close(); });

// ---------- điều khiển ----------
let wt;
$('#w').oninput = e => {
  S.w = +e.target.value;
  e.target.style.setProperty('--v', S.w + '%');
  clearTimeout(wt);
  if (S.ok && hasQuery()) wt = setTimeout(() => run({ silent: true }), 300);   // gộp điểm ở server
};
$('#w').style.setProperty('--v', '55%');
$('#sort').onchange = e => { S.sort = e.target.value; if (hasQuery()) run(); };
$('#sw').onclick = () => setRes(!S.res);
$('#th').onclick = () => setFile(null);
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
  if (!hasQuery()) return $('#q').focus();
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
    return hasQuery() ? go() : nav('');
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
      S.F = { t: new Set(), c: new Set(), min: PRICE_MIN, max: PRICE_MAX };
      fsync();
      if (!$('#fd').open && hasQuery()) run();
      break;
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

addEventListener('hashchange', route);
route();
