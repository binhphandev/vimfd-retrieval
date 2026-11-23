import json, zipfile
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config   # để lấy PROCESSED_DIR, giống V7

def featured_path(p):
    imgs = sorted(p.get("images") or [],
                  key=lambda x: (not x.get("is_featured", False), x.get("position", 9999)))
    if not imgs or not imgs[0].get("local_path"):
        return None
    lp = Path(imgs[0]["local_path"].replace("\\", "/"))
    cands = [lp, ROOT / lp, Path(config.PROCESSED_DIR) / lp, Path(config.IMAGE_DIR) / lp.name]
    return next((c for c in cands if c.exists()), None)

prods = [json.loads(l) for l in open("data/splits/test.jsonl", encoding="utf-8") if l.strip()]
found, missing = [], []
for p in prods:
    f = featured_path(p)
    (found if f else missing).append((p["id"], f))

print(f"Tìm thấy {len(found)}/{len(prods)} | thiếu {len(missing)}")
Path("results").mkdir(exist_ok=True)
with zipfile.ZipFile("test_images.zip", "w") as z:
    manifest = []
    for pid, f in found:
        try:
            arc = f.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            arc = f"data/processed/images/{f.name}"          # ảnh nằm ngoài project
        z.write(f, arc)
        manifest.append(f"{pid}\t{arc}")
    z.writestr("test_images_manifest.tsv", "\n".join(manifest))
print("Missing ids:", [m[0] for m in missing][:20])