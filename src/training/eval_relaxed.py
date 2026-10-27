import sys, os, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # for src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

import torch
from src.training.evaluate import load_model, load_faiss, make_subset_index
from dataset import build_dataloaders
from config import (CHECKPOINT_BEST, PROCESSED_DIR, IMAGE_DIR, PRODUCT_ID_FIELD,
                    FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE)

BAD = ('< html >', 'Chào_mừng mọi người', 'Thank you so much', 'Giá sản_phẩm trên Tiki')
norm = lambda s: re.sub(r'[\s_]+', ' ', s.lower()).strip()

def group_key(p):
    d = (p.get('description') or '').strip()
    return f"{p['shop']}|{norm(d if d and not d.startswith(BAD) else p['name'])}"

meta, test_ids = {}, set()
for sp in ('train', 'val', 'test'):
    for p in json.load(open(Path(PROCESSED_DIR) / f'{sp}.json', encoding='utf-8')):
        pid = str(p[PRODUCT_ID_FIELD])
        meta[pid] = {'g': group_key(p), 'type': p.get('type'), 'shop': p['shop']}
        if sp == 'test': test_ids.add(pid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(CHECKPOINT_BEST).to(device)
index, mapping = load_faiss(FAISS_IMAGE_INDEX_FILE, FAISS_IMAGE_MAPPING_FILE)
sub, sub_map = make_subset_index(index, mapping, test_ids)
loader = build_dataloaders(batch_size=32, num_workers=0, split_dir=Path(PROCESSED_DIR),
                           image_dir=Path(IMAGE_DIR))['test']

for name, (idx, mp) in {'full': (index, mapping), 'test-only': (sub, sub_map)}.items():
    h = dict(n=0, strict1=0, strict10=0, group10=0, type1=0, shop1=0)
    with torch.no_grad():
        for b in loader:
            pids = [str(x) for x in b['product_id']]
            e = model.encode_text(b['input_ids'].to(device),
                                  b['attention_mask'].to(device)).cpu().numpy().astype('float32')
            _, I = idx.search(e, 10)
            for pid, row in zip(pids, I):
                ret = [mp[str(j)] for j in row if str(j) in mp]
                if not ret: continue
                q = meta[pid]; h['n'] += 1
                h['strict1']  += ret[0] == pid
                h['strict10'] += pid in ret
                h['group10']  += any(meta[r]['g'] == q['g'] for r in ret)
                h['type1']    += meta[ret[0]]['type'] == q['type']
                h['shop1']    += meta[ret[0]]['shop'] == q['shop']
    n = h['n']
    print(f"[{name}] n={n} | strict R@1={h['strict1']/n:.3f} R@10={h['strict10']/n:.3f} | "
          f"group R@10={h['group10']/n:.3f} | top1 đúng type={h['type1']/n:.3f} | top1 đúng shop={h['shop1']/n:.3f}")