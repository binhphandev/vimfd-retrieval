# ViMFD Retrieval — Tổng quan dự án

> Tài liệu này được cập nhật ngày 25/11/2026, đối chiếu với bộ mã nguồn, báo cáo xử lý dữ liệu (`Báo_cáo_xử_lý_dữ_liệu_ViMFD.md`) và log chạy thật của `v0_bad_products.py`, `v1_seeds.py`, `build_queries200.py`, `g1_check_queries200.py`, `v3_attributes.py`, `v4_pseudo_queries.py`, `check_segmentation.py`, `v10_kappa_merge.py`, `v11_metrics.py` (kèm hai file kết quả `metrics_summary.csv`, `metrics_by_group.csv`) và kết quả chạy `evaluate.py` bản mới (`results/metrics.txt`, mục 8). Các script `v7_baselines.py`, `v7b_translation_audit.py`, `v8_run_vimfr.py` và hai notebook Colab đã được đọc; **log chạy của V5/V6/V7/V7b/V8 đã được đọc từ output cell của hai notebook** (`train_colab.ipynb` cho dual-encoder, `vimfd_cross_encoder_colab.ipynb` cho V5/V6, `vimfr_colab_v7_v8.ipynb` cho V7/V7b/V8) — xem chi tiết ở mục 6, 9.5, 9.7. Mã `v9_pooling.py` cùng log chạy V9 đã được đọc (mục 9.8); log chạy `build_index` (index ảnh 14.470 sản phẩm) và mã `build_index.py` cũng đã được đọc (mục 7). Mã `main.py`, `schemas.py` và frontend (`api.js`, `main.js`, `state.js`, `ui.js`) đã được đọc nên phần API (mục 2, 10) được đối chiếu với mã; `engine.py`, `catalog.py`, `model.py`, `text_utils.py`, `train.py` và `dataset.py` cũng đã được đọc. `evaluate.py` và `evaluation_metrics.py` đã được viết lại (mục 8) và chạy ra `results/metrics.txt`. Nhiều số liệu trong bản trước đã lỗi thời so với dữ liệu hiện tại (xem mục 4) và đã được sửa hoặc gỡ bỏ. Số liệu được dẫn từ artifact và log có trong repo; chúng không mặc nhiên chứng minh rằng toàn bộ pipeline có thể tái lập trên máy khác.

## 1. Tóm tắt

ViMFD Retrieval là nguyên mẫu tìm kiếm sản phẩm thời trang nam bằng tiếng Việt, tập trung vào áo, quần và phụ kiện. Giao diện/API hỗ trợ truy vấn bằng văn bản, ảnh hoặc kết hợp cả hai. Hệ thống mã hóa truy vấn và ảnh sản phẩm thành vector, tìm ứng viên theo cosine similarity, rồi áp dụng bộ lọc metadata và phân trang.

Các thành phần chính:

- Dual-encoder gồm CLIP ViT-B/32 cho ảnh và PhoBERT-base cho văn bản; hai projection head đưa vector về 256 chiều.
- FAISS `IndexFlatIP` cho tìm kiếm vector chính xác trên vector đã L2-normalize.
- Backend FastAPI, catalog sản phẩm và giao diện HTML/CSS/JavaScript.
- Pipeline xử lý dữ liệu riêng (làm sạch theo nguồn → gộp/chia split theo nhóm → kiểm định 109 mục) được mô tả chi tiết trong báo cáo xử lý dữ liệu.
- Bộ đánh giá dual-encoder (`evaluate.py`, bản viết lại: R@k mức sản phẩm và mức nhóm, truy vấn text đầy đủ/ngắn và ảnh, CI bootstrap theo nhóm; kết quả ở mục 8).
- Pipeline annotation (V0–V11) để xây benchmark truy vấn có gán nhãn và huấn luyện cross-encoder rerank thử nghiệm. Đã có: `queries_200.csv` (200 truy vấn, qua G1 với 0 lỗi, 1 cảnh báo), pool gán nhãn V9 (5.063 cặp), nhãn đã ghép V10 (`qrels.txt`, κ tuyến tính 0,712 giữa hai người, qua G4) và bảng chỉ số V11 trên **tám hệ thống** (mục 9.10). Các kết quả này **chưa phải bằng chứng cuối cùng**: pool gán nhãn lệch (dual_encoder/vimfr có 100% top-10 đã gán nhãn, các baseline chỉ 30–51%), nhãn người dùng là nhãn sơ bộ có máy hỗ trợ, và checkpoint dual-encoder chưa được xác nhận là train trên split hiện tại — xem mục 9 và 11.

Tên model retrieval trong mã là `ViMFRModel`; tên dự án/repository là **ViMFD Retrieval**.

## 2. Phạm vi và luồng hệ thống

```text
Truy vấn văn bản ─ PhoBERT ─ projection ─┐
                                         ├─ điểm cosine / trộn điểm ─ lọc ─ kết quả
Truy vấn ảnh ───── CLIP Vision ─ projection┘

Ảnh sản phẩm ─ CLIP Vision ─ projection ─ FAISS image index
Văn bản sản phẩm ─ PhoBERT ─ projection ─ FAISS text index (được build; API hiện không dùng)
```

Khi có cả văn bản và ảnh, `SearchEngine.search` chuẩn hóa riêng hai mảng điểm (z-score) rồi trộn theo trọng số `w`: `(1-w) * z(text_score) + w * z(image_score)`. Tìm kiếm văn bản hoặc ảnh đơn lẻ dùng điểm cosine thô của nhánh tương ứng (xếp hạng y hệt notebook). Không có cả chữ lẫn ảnh = chế độ duyệt, trả toàn bộ sản phẩm theo một hoán vị cố định, không xếp hạng theo độ liên quan. Bộ lọc giá, danh mục/loại và màu được thực hiện trong engine; engine giữ tối đa 200 ứng viên sau xếp hạng (hằng số `CANDIDATES` khai báo trực tiếp trong `engine.py`, không lấy từ `config.MAX_TOP_K` — biến này hiện không được dùng). Sắp xếp theo giá được áp dụng sau khi chọn tập ứng viên theo độ liên quan.

Truy vấn văn bản đi qua `preprocess_query` (`src/serving/text_utils.py`): chuẩn hóa Unicode/HTML/emoji rồi tách từ bằng **VnCoreNLP (RDRSegmenter, qua thư viện `py_vncorenlp`)**, đúng pipeline tạo `text_seg` lúc xử lý dữ liệu — **không phải `underthesea`**. Ảnh truy vấn đi qua `prepare_query_image`/`to_square` (đệm nền trắng thành ảnh vuông) trước khi vào transform, để cùng phép biến đổi với ảnh gallery.

API chính: `GET /api/health`, `POST /api/search`, `GET /api/products/{pid}`, `GET /api/products/{pid}/similar`. Similarity endpoint dùng embedding ảnh. API phục vụ frontend gốc nếu thư mục đó tồn tại. Theo `main.py` và `schemas.py` (đã đọc): `POST /api/search` nhận `multipart/form-data` (`q`, `image`, `w`, `colors`, `price_min`, `price_max`, `sort`, `page`, `page_size`) và trả JSON theo schema Pydantic `SearchOut`. Ảnh phải là JPG/PNG/WEBP và tối đa 5 MB (lỗi 415/413); ảnh không đọc được → 400; không thấy sản phẩm → 404. Các ràng buộc `w` ∈ [0, 1], `page` ≥ 1, `page_size` ∈ [1, 48] và `sort` ∈ {rel, asc, desc} do FastAPI kiểm trên `Form`, nên vi phạm trả **422** (bản trước ghi nhầm là 400); `limit` của `/similar` được cắt về [1, 24]. Frontend gọi bằng `fetch('/api/search', ...)` (`api.js`); khi chạy theo lệnh ở mục 10 server bind `0.0.0.0` và FastAPI tự sinh tài liệu tại `/docs`. Đây là một API REST thật, hiện **chỉ chạy local, chưa triển khai production** (mục 10).

## 3. Cấu trúc repository và thành phần

| Đường dẫn | Vai trò đã xác nhận |
|---|---|
| `config.py` | Đường dẫn và tham số model, huấn luyện, index, API |
| `data/splits/` | JSONL train/val/test + `split.json` (chứa `group_id` thật từ union-find) |
| `data/processed/` | JSONL đã làm sạch theo nguồn (`{nguồn}_clean.jsonl`) và cây ảnh 224×224 đã chuẩn hóa |
| `data/dataset.py` | Dataset/transform PyTorch, `GroupBatchSampler` (tránh xếp sản phẩm cùng nhóm vào một batch InfoNCE) |
| `src/models/` | Encoder (CLIP + PhoBERT), projection head, model và loss |
| `src/training/train.py`, `evaluate.py`, `evaluation_metrics.py` | Huấn luyện; đánh giá dual-encoder (`evaluate.py`); chỉ số xếp hạng và bootstrap CI theo nhóm (`evaluation_metrics.py`, đặt cùng thư mục với `evaluate.py`) |
| `src/indexing/build_index.py` | Build index text và image cùng mapping/metadata |
| `src/serving/catalog.py`, `text_utils.py`, `engine.py`, `src/api/` | Quy tắc dữ liệu dùng chung, tiền xử lý truy vấn, search engine, endpoint FastAPI |
| `frontend/` | Giao diện web |
| `common.py`, `process_local.py`, `process_tiki.py`, `process_lazada.py`, `build_splits.py`, `data_validation.py` | Pipeline làm sạch theo nguồn, hợp nhất/chia split theo nhóm và kiểm định — mô tả chi tiết trong `Báo_cáo_xử_lý_dữ_liệu_ViMFD.md`. Các script này **có tồn tại và được tài liệu hóa đầy đủ** (khác với ghi nhận trước đây rằng chúng "không có trong source snapshot"); tuy vậy bản thân mã nguồn của các script này không nằm trong bộ file được rà soát ở lượt cập nhật này, nên chưa đọc trực tiếp được. Báo cáo cũng lưu ý `process_local.py` "chưa có bản mới trong bộ tệp" tại thời điểm viết báo cáo — cần đối chiếu `--root` khi chạy lại local để tránh lặp tiền tố đường dẫn ảnh. |
| `data/annotation/` | Pipeline annotation: `v0_bad_products.py`, `v1_seeds.py`, `attr_utils.py` (bộ trích thuộc tính dùng chung V1/V3 — chưa được đọc trực tiếp), `v3_attributes.py`, `v4_pseudo_queries.py`, `v5_hard_negatives.py`, `v6_train_cross_encoder.py`, `build_queries200.py`, `g1_check_queries200.py`, `v7_baselines.py`, `v7b_translation_audit.py`, `v8_run_vimfr.py`, `v9_pooling.py` (tạo pool gán nhãn), `v10_kappa_merge.py`, `v11_metrics.py`, `seg_utils.py` (bộ tách từ dùng chung), `check_segmentation.py`; notebook Colab **`train_colab.ipynb`** (dual-encoder, không phải V5/V6), **`vimfd_cross_encoder_colab.ipynb`** (V5/V6 cross-encoder) và **`vimfr_colab_v7_v8.ipynb`** (V7/V7b/V8); artifact trong `benchmark/` (V0: `bad_products.txt`; V1: `seeds_A.xlsx`, `seeds_B.xlsx`, `_private/seeds_meta.csv`; `queries_200.csv`; V7 tạo thêm `translations.csv`, V7b tạo `translation_audit.csv`) và `cross_encoder/` (V3–V5) — xem mục 9 |
| `checkpoints/` | Checkpoint dual-encoder: `best_grp_24112026.pt` (epoch 12, val_loss=2,2252) và `best_short_24112026.pt` (config mặc định trong `config.py`). Checkpoint cross-encoder: `cross_encoder_best/` (~540 MB, `model.safetensors`, epoch 3, val_loss=0,1726, val_acc=0,9374, val_auc=0,9769, val_top1=0,9209) — **đã xác nhận** từ log V6 trong `vimfd_cross_encoder_colab.ipynb`. |
| `index/`, `results/` | FAISS artifacts, metric, log và run files |

## 4. Dữ liệu và chất lượng

### 4.1 Quy mô

Theo `Báo_cáo_xử_lý_dữ_liệu_ViMFD.md` (24/11/2026), sau làm sạch dữ liệu gồm **14.470 sản phẩm** và **43.536 ảnh** (≈3,01 ảnh/sản phẩm), thu thập từ ba nguồn: thương hiệu nội địa (aristino, johnhenry, henstore, icondenim), Tiki và Lazada.

| Nguồn | Sản phẩm | Áo | Quần | Phụ kiện |
|---|---:|---:|---:|---:|
| local | 3.894 | 2.136 | 1.065 | 693 |
| tiki | 2.739 | 1.898 | 841 | 0 |
| lazada | 7.837 | 2.848 | 2.341 | 2.648 |
| **Tổng** | **14.470** | **6.882** | **4.247** | **3.341** |

Tiki không có sản phẩm `phụ kiện`; nhóm này chỉ đến từ Lazada và local.

Chia split theo nhóm (union-find trên `text_group`, (nguồn, shop, tên bỏ SKU), URL ảnh và pHash ≤4 bit), mục tiêu 80/10/10, kết quả từ lần kiểm định mới nhất (107 PASS, 2 WARN, 0 FAIL):

| Split | Sản phẩm | Tỉ lệ | Ảnh | Ảnh/sản phẩm |
|---|---:|---:|---:|---:|
| train | 11.571 | 80,0% | 34.412 | 2,97 |
| val | 1.450 | 10,0% | 5.267 | 3,63 |
| test | 1.449 | 10,0% | 3.857 | 2,66 |
| **Tổng** | **14.470** | **100%** | **43.536** | **3,01** |

Record chứa ID, nguồn/shop/URL, danh mục, tên/mô tả, giá, màu/kích cỡ/chất liệu và danh sách ảnh. Mô hình dùng `text_seg` làm đầu vào văn bản.

### 4.2 Tiền xử lý theo mã nguồn

- Ảnh đầu vào mô hình: ảnh gallery đã được xuất sẵn thành ảnh vuông 224×224 đệm nền trắng lúc xử lý dữ liệu (`to_square`), nên transform train/eval (`dataset.py`) chỉ resize (BICUBIC) + center-crop + chuẩn hóa theo mean/std của CLIP ViT-B/32, không cắt méo nội dung. Validation/test không augment; training có lật ngang ngẫu nhiên và ColorJitter nhẹ.
- Ảnh đại diện trong catalog/index: ưu tiên `is_featured`; nếu không có, chọn `position` nhỏ nhất trong các ảnh có `local_path` hợp lệ (cùng quy tắc ở `catalog.py`, `dataset.py` và `train.py`).
- Văn bản: `text_seg`, tối đa 256 token PhoBERT (chỉ dùng 300 âm tiết đầu trước khi tách từ); truy vấn API được word-segment bằng **VnCoreNLP (RDRSegmenter, qua `py_vncorenlp`)**, đúng pipeline tạo `text_seg` của dữ liệu train.
- Split chia theo nhóm (không phải ngẫu nhiên theo từng sản phẩm) để tránh rò rỉ giữa các split; tỉ lệ đạt gần đúng 80/10/10 ở mọi ô (nguồn, category), lệch không quá ~0,2 điểm phần trăm theo báo cáo kiểm định mới nhất.

### 4.3 Cảnh báo chất lượng (theo kiểm định mới nhất, 107 PASS / 2 WARN / 0 FAIL)

Dữ liệu đã qua ba lượt kiểm định. Lượt 1 có 2 FAIL do rò rỉ ảnh gần trùng (pHash cách nhau ≤4 bit không được gộp nhóm trước khi chia); lượt 2 hết FAIL nhưng một nhóm sản phẩm khổng lồ (1.615 sản phẩm, chủ yếu aristino) rơi trọn vào val làm lệch phân bố; lượt 3 (hiện tại) đã sửa cả hai vấn đề. Các cảnh báo/kết quả hiện còn cần lưu ý khi đọc kết quả:

- **Rò rỉ ảnh gần trùng giữa các split (đã sửa)**: ở ngưỡng pHash ≤4 bit — ngưỡng dùng để gộp nhóm khi chia split — hiện là **0,00%** giữa val/test và các split trước đó (trước đây là 26,36%/22,67% ở lượt 1 thất bại). Ở ngưỡng tham khảo rộng hơn 8 bit (không dùng để chia split) vẫn còn 38,14% (val→train) và 36,51% (test→train+val); con số 8-bit này chỉ để tham khảo, không phải lỗi.
- **Ảnh trùng hệt trong cùng split**: val 20,55% sản phẩm (298), test 16,15% (234) — khiến truy vấn ảnh có thể có nhiều đáp án đúng cùng lúc (nên đọc R@k mức *nhóm*, không chỉ mức *sản phẩm*, khi đánh giá).
- **Negative sai trong batch InfoNCE** (sản phẩm cùng nhóm anh em/ảnh gần trùng bị xếp chung batch train): trung bình 0,51 cặp/batch-64 ở train (1,4% sản phẩm train bị ảnh hưởng), 1,39 cặp ở val (3,4%), 2,04 cặp ở test (5,0%). Vấn đề này **đã được xử lý khi train** bằng `GroupBatchSampler` (`dataset.py`), dùng `group_id` thật trong `split.json` để không xếp quá một sản phẩm/nhóm vào cùng batch — đúng khuyến nghị của báo cáo dữ liệu.
- **Văn bản bị cắt ở giới hạn 256 token**: theo lần kiểm định mới nhất (ước lượng 1,3 token/từ vì thiếu thư viện `transformers` khi chạy), tỉ lệ vượt 256 chỉ **0,06% (train), 0,07% (val), 0,00% (test)**. Lần chạy trước có tokenizer PhoBERT thật cho p99 ≈170–175 và **không text nào bị cắt**. (Số liệu 21–24% bị cắt ghi ở bản trước của tài liệu này là của một lần chạy cũ hơn và không còn đúng — đã gỡ bỏ.)
- **Giá**: chưa có số liệu kiểm định hiện hành xác nhận tỉ lệ giá thiếu/không hợp lệ (claim "100% thiếu giá" ở bản trước không có căn cứ trong báo cáo hiện tại nên đã gỡ bỏ). `common.py` chỉ xóa giá trị sentinel 99999999 về rỗng, không loại sản phẩm vì giá; `engine.py` xử lý phòng thủ khi thiếu `price` (trả 0.0).
- **Mô tả rỗng ở tập Test**: theo `v0_bad_products.py`, **309/1.449 sản phẩm Test (21,3%)** có `description` rỗng — Lazada 257/784 (33%), Tiki 49/274 (18%), local 3/391 (johnhenry 1/186, icondenim 2/33, aristino 0/112, henstore 0/60). Cả 309 trường hợp bị loại chỉ vì mô tả rỗng; không sản phẩm Test nào thiếu ảnh đại diện trên đĩa (lý do `khong_co_images`/`khong_co_file_anh` đều bằng 0). Đây là lý do chính khiến benchmark bị lệch về nguồn local (xem mục 9.2 và 9.6). Lưu ý đây là số sản phẩm mô tả **rỗng** (chỉ ở Test), khác với 22,8% sản phẩm toàn bộ dữ liệu có mô tả **dưới 5 từ** nêu ngay dưới đây.
- **Các hạn chế khác đáng chú ý** (chi tiết ở mục 7 của báo cáo dữ liệu): 22,8% sản phẩm toàn bộ dữ liệu chỉ có tên + màu (dưới 5 từ mô tả, chủ yếu Lazada/Tiki); 54 sản phẩm nghi gắn sai category theo trường `type` gốc; một số sản phẩm lạc đề lọt bộ lọc phạm vi; ~2,3–2,8% sản phẩm val/test trùng tên với split trước (rò rỉ ngữ nghĩa nhẹ, cùng mẫu bán ở shop khác); val nghiêng về sản phẩm nhiều ảnh hơn test (3,63 ảnh/sp so với 2,66), nên R@k theo ảnh của hai split khó so trực tiếp.

Việc dedupe theo cụm ảnh/sản phẩm trước khi chia split — khuyến nghị ở bản trước của tài liệu này — **đã được thực hiện** (`build_splits.py` dùng union-find gộp theo `text_group`, (nguồn, shop, tên bỏ SKU), URL ảnh và pHash ≤4 bit, cộng bước gỡ "ảnh hub" trước khi gom nhóm). Việc còn tồn đọng là: cài `transformers` để đo token thật thay vì ước lượng, soát 54 nhãn category nghi sai, và dùng mean/std ảnh riêng của bộ dữ liệu thay vì mean/std của CLIP (xem mục 11).

## 5. Kiến trúc dual-encoder

| Nhánh | Backbone | Vector backbone | Đầu ra |
|---|---|---:|---:|
| Ảnh | `openai/clip-vit-base-patch32` | 768 | 256 |
| Văn bản | `vinai/phobert-base` | 768 | 256 |

Mỗi projection head: `Linear(768,256) → LayerNorm → GELU → Dropout(0.1) → Linear(256,256)`, sau đó L2-normalize. Cấu hình chỉ mở hai block cuối của mỗi backbone (`UNFREEZE_LAST_N = 2`) cùng lớp layer-norm liên quan; phần còn lại được đóng băng. Model có `logit_scale` khởi tạo `log(1/0.07)` (≈ scale 14,29) và giới hạn scale tối đa 100.

Loss huấn luyện là Symmetric InfoNCE: cross-entropy hai chiều image→text và text→image, lấy trung bình; các cặp cùng vị trí trong batch là positive, phần còn lại được xem là negative — đây chính là lý do negative sai (mục 4.3) cần `GroupBatchSampler` để giảm thiểu.

## 6. Huấn luyện và checkpoint

Cấu hình mặc định trong `config.py`: batch 64, 30 epoch, learning rate backbone `1e-5`, head `1e-4`, weight decay `0,01`, warmup 2 epoch rồi cosine decay, gradient clipping 1.0, 2 workers. `BATCH_SIZE_CPU = 4` cũng được khai báo trong `config.py` nhưng nằm trong nhóm "hiện không được mã nguồn đã kiểm tra sử dụng" — không có fallback CPU tự động trong `train.py`. AMP FP16 bật khi CUDA khả dụng; optimizer là AdamW với 3 nhóm tham số (backbone, head, `logit_scale`).

Train loader dùng `GroupBatchSampler` khi có `group_map` (đọc từ `group_id` trong `split.json`, có cơ chế tự tắt nếu phát hiện `group_id` chỉ là tên split do chạy `build_splits.py` bản cũ). Validation trong `train.py` tính loss cùng Recall@1/5/10 cho cả hai hướng Text→Image và Image→Image, ở **cả hai mức**: mức sản phẩm (hòa điểm tính là thua) và mức nhóm (`group_id`, thành công nếu bất kỳ sản phẩm cùng nhóm vào top-k) — khớp khuyến nghị của báo cáo dữ liệu về đọc R@k mức nhóm. Validation còn đo T→I với text **ngắn** (ba biến thể tất định `name`, `head4`, `head3_color` của `dataset.SHORT_VARIANTS`; mức nhóm R@10 và tỉ lệ đúng danh mục P@10 trong top-10). Mặc định (`--select mix`), script chọn checkpoint tốt nhất theo **trung bình của T→I R@10 mức nhóm với text đầy đủ và T→I R@10 mức nhóm với text ngắn** (early stopping patience 5); các tùy chọn khác: `t2i_group` (chỉ text đầy đủ), `avg` (trung bình T→I và I→I mức nhóm), `short` (chỉ text ngắn). Khi huấn luyện, text đầy đủ được thay bằng text ngắn với xác suất `SHORT_TEXT_PROB = 0,5`. Việc chọn checkpoint chỉ dùng split val, không dùng test.

Đường dẫn checkpoint mặc định hiện tại trong `config.py` là `checkpoints/best_short_24112026.pt` và `checkpoints/latest.pt` (hậu tố `short` khớp với việc huấn luyện có text ngắn, `SHORT_TEXT_PROB = 0,5`; các bản trước của tài liệu này ghi `best_24112026.pt` và trước đó `best_18112026.pt` — tên file đã đổi, xem mục 7). **Notebook  trong bộ file hiện tại là notebook huấn luyện dual-encoder** (có , , gọi ), không phải V5/V6. Log chạy từ cell output của notebook: checkpoint tốt nhất tại epoch 12 (val_loss=2,2252, T→I group R@10=0,7269), early stop tại epoch 17 (patience=5); checkpoint lưu tại  trên Drive. Kết quả test (cell-16):

- Gallery Test-only (1.449 sp): T→I product R@10=0,5721, group R@10=0,7391; I→I product R@10=0,8551, group R@10=0,9439
- Gallery Full (14.470 sp): T→I product R@1=0,0221, R@10=0,1573, group R@1=0,1097, R@10=0,3451
- Color P@10: Full=0,4554, Test-only=0,4322

Notebook lấy checkpoint dual-encoder từ biến môi trường , nếu không có thì từ ;  hiện trỏ tới . **Đã xác nhận** notebook này train trên split 11.571/1.450/1.449 hiện tại (cell-8 output). Notebook riêng cho cross-encoder là .

## 7. Chỉ mục và khả năng tái lập

`src/indexing/build_index.py` tạo index văn bản và ảnh (`IndexFlatIP`) cùng mapping ID. Vì vector được L2-normalize, inner product bằng cosine similarity.

Ghi chú quan trọng về tính thời sự: dữ liệu hiện tại có **14.470 sản phẩm** (11.571/1.450/1.449) và `config.py` hiện trỏ `CHECKPOINT_BEST` tới `best_short_24112026.pt`. Metadata index quan sát được ở các lần rà soát trước (13.368 vector, checkpoint khai báo `best_18112026.pt`, fingerprint `870967d596f8dbd6e138df5ca13443cf`, build lúc 2026-11-19 17:23:00, đường dẫn máy build cũ `C:\Users\DELL\Downloads\vimfd-retrieval\...`) rõ ràng **không còn khớp** với split và tên checkpoint hiện tại — số sản phẩm chênh hơn 1.100 và tên checkpoint đã đổi. Tài liệu này không có quyền truy cập vào file metadata index hiện hành nên không khẳng định số liệu mới; nhưng engine.py tự kiểm tra việc này khi khởi động (so fingerprint checkpoint, so ID index với catalog) và sẽ từ chối chạy (trừ khi `ALLOW_STALE_INDEX=True`) nếu lệch — do đó **gần như chắc chắn cần build lại index** trước khi dùng API hoặc evaluate.py với checkpoint/dữ liệu hiện tại.

**Cập nhật (log `build_index` mới):** index ảnh đã được build lại trên catalog hiện tại — mã hóa **14.470/14.470** ảnh đại diện (23 phút 21 giây), `IndexFlatIP` có `ntotal = 14470`, mapping 14.470 mục, không có cảnh báo bỏ qua ảnh, smoke test ảnh đạt (sản phẩm truy vấn nằm trong top-5, score ≈ 1,0000). Checkpoint được nạp ở epoch 17, `val_loss = 2,1441`; log **không in đường dẫn checkpoint** (mặc định của script là `config.CHECKPOINT_BEST`, tức `best_short_24112026.pt`). Như vậy số vector index đã khớp catalog 14.470, nên tình trạng "index lệch 13.368 so với 14.470" ở trên không còn cho lần build này. Sau đó `evaluate.py` bản mới đã được chạy (mục 8): đầu `results/metrics.txt` ghi checkpoint `best_short_24112026.pt`, fingerprint `c9142e241cf3c72e3bc09310fa1bafd0` và index build lúc 2026-11-25 11:46:02. Script so fingerprint này với checkpoint và dừng nếu lệch (trừ khi dùng `--allow-stale`; giả định lần chạy không dùng cờ này), nên index hiện hành được coi là khớp checkpoint. Log epoch 17 / `val_loss = 2,1441` ở trên chưa được đối chiếu với fingerprint này (không rõ có phải cùng một lần build). **Vẫn chưa xác nhận** checkpoint được huấn luyện trên split 14.470. Lưu ý: `build_index.py` chỉ index ảnh đại diện và chỉ build text index khi có `--with-text`; index thí nghiệm của V5/V8 nằm riêng ở `index/experiments/`.

## 8. Đánh giá dual-encoder: cơ chế và kết quả

`evaluate.py` (và `evaluation_metrics.py`) được viết lại ngày 25/11/2026 để khớp với `train.py` và chạy ra `results/metrics.txt` (cùng `metrics.json`). Các bảng dưới đây lấy từ lần chạy đó: checkpoint `best_short_24112026.pt`, fingerprint `c9142e241cf3c72e3bc09310fa1bafd0`, index ảnh build lúc 2026-11-25 11:46:02, split `test`, bootstrap 2.000 lần theo nhóm (seed 42).

### 8.1 Cơ chế đánh giá

- **Gallery**: embedding ảnh đại diện lấy từ `product_image.index` (`reconstruct_n`, chỉ dùng được với `IndexFlat*`). Hai kiểu: **full** = 14.470 sản phẩm (cả train/val/test), gần với điều kiện API thật; **test-only** = 1.449 sản phẩm test, dùng để so sánh kiểu paper.
- **Truy vấn Text→Image** (1.449 query, 948 nhóm): `text_seg` đầy đủ của sản phẩm test, và ba biến thể ngắn tất định của `dataset.SHORT_VARIANTS` — `name` (tên), `head4` (4 từ đầu), `head3_color` (3 từ đầu + màu) — mô phỏng cách người dùng gõ thật. Truy vấn chỉ mã hóa text (không tải ảnh).
- **Truy vấn Image→Image** (766 query, 560 nhóm): **một** ảnh phụ cho mỗi sản phẩm test có ≥ 2 ảnh có `local_path` (đúng `AltImageDataset` của `train.py`), tức 766/1.449 sản phẩm test (≈ 53%). Báo hai bản: tất cả query, và bỏ query có ảnh phụ trùng hệt ảnh đại diện (cosine với chính sản phẩm đó ≥ 0,9995; chỉ 2 query bị bỏ). Ảnh đọc lỗi (nếu có) được đếm và bỏ qua thay vì làm crash; lần chạy này không ghi nhận ảnh nào bị bỏ.
- **Cách tính hạng** (khớp `validate()` trong `train.py`): *mức sản phẩm* — hạng của đúng sản phẩm đó, điểm hòa tính là thua (sai số float 1e-6); *mức nhóm* — mọi sản phẩm cùng `group_id` (`split.json`) đều là đáp án đúng, hạng = hạng của sản phẩm cùng nhóm cao nhất. Recall@k ở đây là tỉ lệ query có đáp án trong top-k (mỗi query một đáp án hoặc một nhóm đáp án); MRR@10 và NDCG@10 giới hạn trong top-10.
- **Khoảng tin cậy 95%**: bootstrap theo **nhóm** (lấy mẫu có hoàn lại các nhóm, không phải từng query, vì query trong cùng nhóm không độc lập); mọi chỉ số đều có CI trong `metrics.txt`/`metrics.json`.
- **Kiểm tra index**: script so fingerprint sha256 của checkpoint với `product_image.meta.json` và dừng nếu lệch (`--allow-stale` để bỏ qua). Cờ `"ok"` và ngưỡng `IDENTICAL_THR` nay do chính `evaluate.py` định nghĩa (`SafeAltImageDataset`, `IDENTICAL_THR = 0,9995`), nên hai điểm không khớp với `train.py` ở bản trước không còn; `AltImageDataset` trong `train.py` vẫn không có hai thứ này (không ảnh hưởng huấn luyện). Bản cũ báo micro/macro; bản mới bỏ vì mỗi sản phẩm đúng một query (micro = macro).

### 8.2 Kết quả — gallery full (14.470 sản phẩm)

| Query | Mức | R@1 | R@5 | R@10 [CI 95%] | MRR@10 | NDCG@10 |
|---|---|---:|---:|---|---:|---:|
| T→I, text đầy đủ | nhóm | 0,0801 | 0,2429 | 0,3423 [0,2553–0,4233] | 0,1510 | 0,1959 |
| | sản phẩm | 0,0166 | 0,0939 | 0,1636 [0,1380–0,1896] | 0,0510 | 0,0770 |
| T→I, `name` | nhóm | 0,0704 | 0,2174 | 0,2995 [0,2186–0,3760] | 0,1341 | 0,1730 |
| | sản phẩm | 0,0152 | 0,0780 | 0,1215 [0,0991–0,1468] | 0,0412 | 0,0598 |
| T→I, `head4` | nhóm | 0,0552 | 0,1387 | 0,1884 [0,1178–0,2646] | 0,0906 | 0,1136 |
| | sản phẩm | 0,0104 | 0,0428 | 0,0718 [0,0544–0,0888] | 0,0252 | 0,0360 |
| T→I, `head3_color` | nhóm | 0,0290 | 0,1201 | 0,1539 [0,0882–0,2208] | 0,0628 | 0,0843 |
| | sản phẩm | 0,0104 | 0,0442 | 0,0725 [0,0550–0,0894] | 0,0243 | 0,0354 |
| I→I, tất cả (766 query / 560 nhóm) | nhóm | 0,5117 | 0,6997 | 0,7728 [0,7243–0,8176] | 0,5932 | 0,6362 |
| | sản phẩm | 0,3433 | 0,5640 | 0,6540 [0,5661–0,7283] | 0,4433 | 0,4938 |
| I→I, bỏ 2 query trùng hệt (764 / 558) | nhóm | 0,5105 | 0,6990 | 0,7723 [0,7263–0,8169] | 0,5921 | 0,6352 |
| | sản phẩm | 0,3416 | 0,5628 | 0,6531 [0,5569–0,7277] | 0,4419 | 0,4925 |

### 8.3 Kết quả — gallery test-only (1.449 sản phẩm)

| Query | Mức | R@1 | R@5 | R@10 [CI 95%] | MRR@10 | NDCG@10 |
|---|---|---:|---:|---|---:|---:|
| T→I, text đầy đủ | nhóm | 0,3092 | 0,5928 | 0,7191 [0,6679–0,7651] | 0,4305 | 0,4991 |
| | sản phẩm | 0,1263 | 0,3727 | 0,5383 [0,4914–0,5836] | 0,2337 | 0,3050 |
| T→I, `name` | nhóm | 0,2698 | 0,5645 | 0,6798 [0,6244–0,7302] | 0,3930 | 0,4614 |
| | sản phẩm | 0,0994 | 0,3299 | 0,4838 [0,4409–0,5282] | 0,1968 | 0,2637 |
| T→I, `head4` | nhóm | 0,1643 | 0,3934 | 0,5183 [0,4353–0,5927] | 0,2644 | 0,3243 |
| | sản phẩm | 0,0552 | 0,1953 | 0,3209 [0,2833–0,3604] | 0,1180 | 0,1649 |
| T→I, `head3_color` | nhóm | 0,1256 | 0,3713 | 0,4810 [0,4032–0,5479] | 0,2249 | 0,2854 |
| | sản phẩm | 0,0497 | 0,2001 | 0,3168 [0,2806–0,3527] | 0,1148 | 0,1617 |
| I→I, tất cả (766 / 560) | nhóm | 0,7285 | 0,8982 | 0,9347 [0,9095–0,9550] | 0,7993 | 0,8323 |
| | sản phẩm | 0,5091 | 0,7768 | 0,8499 [0,7472–0,9236] | 0,6255 | 0,6799 |
| I→I, bỏ 2 query trùng hệt (764 / 558) | nhóm | 0,7277 | 0,8979 | 0,9346 [0,9118–0,9548] | 0,7987 | 0,8319 |
| | sản phẩm | 0,5079 | 0,7762 | 0,8495 [0,7468–0,9244] | 0,6245 | 0,6791 |

### 8.4 Cách đọc kết quả

- **Dùng mức nhóm làm số chính.** Mức nhóm cao hơn mức sản phẩm rõ rệt ở Text→Image (ví dụ text đầy đủ, R@10: 0,3423 so với 0,1636 ở gallery full; 0,7191 so với 0,5383 ở test-only). Điều này phù hợp với việc nhiều sản phẩm có ảnh trùng hệt/gần trùng (16,15% sản phẩm test có ảnh trùng hệt, mục 4.3): ở mức sản phẩm, các sản phẩm trùng ảnh cho cùng điểm nên bị tính là thua, dù đều là đáp án hợp lý.
- **Gallery full khó hơn nhiều so với test-only** (T→I text đầy đủ, R@10 mức nhóm 0,3423 so với 0,7191; R@1 mức nhóm 0,0801 so với 0,3092). API tìm trên toàn bộ 14.470 sản phẩm, nên số gallery full mới sát điều kiện sử dụng; số test-only chỉ nên dùng khi so sánh kiểu paper và phải nêu rõ cỡ gallery 1.449.
- **Text ngắn giảm chất lượng.** Theo thứ tự đầy đủ → `name` → `head4` → `head3_color`, R@10 mức nhóm giảm từ 0,3423 xuống 0,2995, 0,1884, 0,1539 (full) và từ 0,7191 xuống 0,6798, 0,5183, 0,4810 (test-only). `name` gần với text đầy đủ (CI chồng lên nhau nhiều); `head4` và `head3_color` thấp hơn rõ (ở test-only, CI của hai biến thể này không chồng với CI của text đầy đủ). Đây là so sánh CI riêng từng số, chưa phải kiểm định có cặp. Thêm màu (`head3_color`) không cho thấy lợi ích so với `head4` (CI chồng nhau; hai biến thể cũng khác nhau ở số từ giữ lại).
- **Image→Image cao hơn Text→Image nhiều** (gallery full, mức nhóm: R@1 0,5117 và R@10 0,7728; test-only: 0,7285 và 0,9347), nhưng không so trực tiếp được với Text→Image: tập query khác (766 sản phẩm có ảnh phụ so với 1.449), và query là một ảnh khác của cùng trang sản phẩm. Chỉ 2/766 query bị loại vì trùng hệt ảnh đại diện, nên số I→I không bị đội lên bởi bản sao ảnh; hai bản gần như trùng nhau (chênh ≤ 0,002).
- **CI khá rộng** (Text→Image mức nhóm thường rộng ±0,04–0,09) vì 1.449 query chỉ thuộc 948 nhóm. Chênh lệch nhỏ hơn khoảng đó giữa các biến thể không nên được diễn giải.
- **Giới hạn**: một checkpoint, một split test; chưa tách theo nguồn/shop/danh mục; 309/1.449 sản phẩm test có mô tả rỗng (mục 4.3) vẫn nằm trong query nhưng chưa được phân tích riêng (text đầy đủ của chúng chủ yếu là tên và màu). Các số này **không so trực tiếp** với bảng cũ của `evaluate.py` (gallery 13.368/1.339, tính micro/macro, hòa điểm không tính là thua) — bảng cũ đã bị gỡ — và cũng không so với mục 9.10, nơi điểm tính trên 200 truy vấn do người viết, nhãn 0/1/2 và pool gán nhãn.

## 9. Pipeline annotation, benchmark và cross-encoder (V0–V11): tình trạng thực hiện

Mục này thay cho mục 9 cũ (benchmark "200 truy vấn, 187 hợp lệ, ViMFR dẫn đầu Recall@10/NDCG@10, BM25 dẫn đầu MRR@10"). Các số liệu của bản cũ **đã bị gỡ bỏ** vì không có run file hay log kiểm chứng. Bộ số liệu ở mục 9.10 là **lần chạy mới**: `queries_200.csv` (V1 → gộp → G1), pool V9, nhãn V10 và chỉ số V11 trên tám file run. Cross-encoder là thành phần thí nghiệm; `engine.py` không nạp hay gọi nó. Những phần không có log (V5, V6, V7b, latency, mã V9) được ghi rõ là chưa xác minh.

### 9.1 Trạng thái từng bước

| Bước | Script | Trạng thái | Ghi chú |
|---|---|---|---|
| V0 | `v0_bad_products.py` | Đã chạy | Kiểm tra đầu vào, tạo `bad_products.txt` (309 id Test) |
| V1 | `v1_seeds.py` | Đã chạy | Chọn hạt giống từ 1.140 ứng viên Test; xuất `seeds_A.xlsx`/`seeds_B.xlsx` (mỗi người 100 slot chính + dự phòng); 4 cảnh báo thiếu dự phòng (mục 9.6) |
| (thủ công) | Người viết A, B điền `query_text`, `attrs_in_query` vào `seeds_*.xlsx` | Đã xong (suy ra từ bước gộp) | `build_queries200.py` đọc đủ 100 + 100 dòng có truy vấn; seeds_B có 41 dòng `skipped_reason` cần rà |
| Gộp | `build_queries200.py` | Đã chạy | Ghi `queries_200.csv` 200 dòng (A 100 / B 100; overlap 30) |
| G1 | `g1_check_queries200.py` | Đã chạy | **0 lỗi, 1 cảnh báo** (truy vấn dạng "loại, màu, …" ước lượng 43% > 40%); ghi `results/logs/log_A.md` |
| V3 | `v3_attributes.py` | Đã chạy (lần 2) | Trích thuộc tính Train+Val; còn 2 type chưa map (mục 9.3) |
| V4 | `v4_pseudo_queries.py` | Đã chạy lại bằng `seg_utils` | Truy vấn tách từ khớp `text_seg` (mục 9.4) |
| V5 | `v5_hard_negatives.py` | Đã có log (notebook cell output) | Train: 5.155 dương + 20.543 âm (1:3,99); val: 860 dương + 3.440 âm; 20/20 auto check passed; seed=42, top_k=50, n_neg=4; file `ce_pairs_train.csv` (~16,2 MB, 25.819 cặp) và `ce_pairs_val.csv` (~2,6 MB, 4.300 cặp) |
| V6 | `v6_train_cross_encoder.py` | Đã có log (notebook cell output) | PhoBERT-base, MAX_LEN=256, BATCH=32, LR=2e-5/1e-4, FP16, Tesla T4; best epoch 3 (val_loss=0,1726, val_acc=0,9374, val_auc=0,9769, val_top1=0,9209); early stop epoch 6; total 24,4 phút; checkpoint `cross_encoder_best/` (~540 MB) |
| V7 | `v7_baselines.py` | Đã chạy (có 6 run file; không có log) | Sáu hệ thống cơ sở trên 200 truy vấn, top-50 (mục 9.7) |
| V7b | `v7b_translation_audit.py` | Đã chạy (đã có log từ notebook cell-11) | 200 truy vấn, 401 thuộc tính kiểm tra; kept 91,5%, partial 1,7%, mistranslated 1,0%, lost 5,7%; 35/200 truy vấn có mt_loss_flag=1; 166/200 giữ toàn bộ thuộc tính (mục 9.7) |
| V8 | `v8_run_vimfr.py` | Đã chạy (có `dual_encoder.run`, `vimfr.run`; latency từ notebook cell-16) | Latency trên T4: median=154,8ms, P95=203,2ms; 0/200 truy vấn giữ nguyên thứ tự sau rerank; cần xác nhận checkpoint dual-encoder train trên split hiện tại (mục 11, điểm 14) |
| V9 | `v9_pooling.py` | Đã chạy (đã đọc mã và log) | Pool 5.063 cặp, K baseline = 2 (mục 9.8) |
| V10 | `v10_kappa_merge.py` | Đã chạy (`kappa`, `merge`) | κ tuyến tính 0,712, qua G4; `qrels.txt` 5.063 cặp (mục 9.9) |
| V11 | `v11_metrics.py` | Đã chạy (bản đã sửa: 8 hệ thống + `judged@10`) | Mục 9.10 |
| V2 | — | Không nằm trong bộ file rà soát | — |

### 9.2 V0 — kiểm tra đầu vào và `bad_products.txt`

- Kích thước ba tập khớp kỳ vọng: train 11.571, val 1.450, test 1.449; giao train∩val, train∩test, val∩test đều **0 id trùng và 0 `text_group` trùng**; không thiếu trường nào trong schema.
- Sản phẩm Test bị loại: **309/1.449 (21,3%)**, toàn bộ vì mô tả rỗng (`mo_ta_rong`). `bad_products.txt` có đúng 309 dòng: lazada 257, tiki 49, icondenim 2, johnhenry 1.
- **Còn 1.140 sản phẩm Test** làm ứng viên hạt giống truy vấn. Phân bố bị loại theo shop: lazada 257/784 (33%), tiki 49/274 (18%), johnhenry 1/186 (1%), icondenim 2/33 (6%), aristino 0/112, henstore 0/60.
- Hệ quả: các sản phẩm Lazada/Tiki bị loại nhiều hơn hẳn nguồn local, nên phân bố nguồn của hạt giống (và của benchmark) lệch về local so với catalog (local chiếm 391/1.449 = 27,0% sản phẩm Test nhưng 388/1.140 ≈ 34,0% ứng viên). Cần báo cáo rõ khi diễn giải kết quả benchmark. Bộ benchmark thực tế sau V1 có 76/200 (38%) hạt giống nội địa (mục 9.6), còn cao hơn cả 27,0% của catalog Test.

### 9.3 V3 — thuộc tính sản phẩm (Train+Val, 13.021 sản phẩm)

- Có thuộc tính: MAU 2.267 (17,4%), CHAT_LIEU 4.723 (36,3%), KIEU_DANG 3.621 (27,8%), PHONG_CACH 3.948 (30,3%). Dùng được để sinh truy vấn giả (có ≥1 thuộc tính ngoài loại): **9.540 (73,3%)** — train 8.515/11.571 (3.056 không có thuộc tính), val 1.025/1.450 (425 không có).
- 18 giá trị MAU, **51** giá trị LOAI, 11 giá trị CHAT_LIEU. `CHAT_LIEU` được dò bằng regex từ name/description vì dữ liệu **không có trường `material`**.
- V3 không đọc `test.jsonl`, nên Test không có thuộc tính.
- Bảng đồng nghĩa: 25/30 cặp, đang chờ người duyệt (sheet `synonyms`); V4 đọc trực tiếp sheet này.
- **`TYPE_MAP`**: lần chạy đầu có 6 type chưa map; sau khi bổ sung, lần chạy thứ hai còn 2: `Dép xỏ ngón` (108) và `ao-lot` (8). Type chưa map bị đổi thành tên gốc viết thường bỏ gạch ngang: `dép xỏ ngón` (đọc tự nhiên nhưng không nhất quán với `Dép xỏ ngón (Flip Flops)` đã map thành `dép`) và `ao lot` (không dấu, sai chính tả; 8 sản phẩm). Nên thêm `"ao-lot": "áo lót"`, `"Dép xỏ ngón": "dép"` rồi chạy lại V3 → V4 nếu muốn dứt điểm; không chặn V5.

### 9.4 V4 — truy vấn giả

Kết quả lần chạy hiện tại (sau khi chuyển sang `seg_utils`; seed 42, `MAX_DUP=3`, `MAX_TRIES=5`, `P_SYNONYM=0.3`, `P_SHUFFLE=0.5`):

| | Train | Val |
|---|---:|---:|
| Sản phẩm dùng được (có thuộc tính) | 8.515 | 1.025 |
| Truy vấn giả sinh ra | 5.174 | 860 |
| Bỏ vì trùng chữ quá 3 lần | 3.341 | 165 |
| Chữ truy vấn khác nhau | 3.254 | 604 |
| Số chữ xuất hiện 1/2/3 lần | 2.082/424/748 | 443/66/95 |
| n_attrs = 1 / 2 / 3 | 44% / 50% / 7% | 69% / 27% / 3% |
| Mẫu câu: generic / chat / mau / kd_pc / pc_mau | 2.919/1.105/674/365/111 | 547/202/64/35/12 |
| Giá trị thay bằng đồng nghĩa | 2.218 | 308 |

Điểm cần lưu ý khi đọc: (a) chỉ 5.174/11.571 (44,7%) sản phẩm Train và 860/1.450 (59,3%) sản phẩm Val có truy vấn giả; (b) 748 chữ Train được dùng đủ 3 lần, tức 2.244/5.174 (43%) truy vấn có chữ trùng với hai truy vấn khác trỏ tới sản phẩm dương khác nhau — hệ quả của không gian thuộc tính nhỏ, cần nhớ khi đọc loss/độ chính xác của cross-encoder; (c) phân bố `n_attrs` của Val lệch về truy vấn ngắn hơn Train.

**Tách từ.** Bản V4 đầu tiên gọi `underthesea.word_tokenize`, lệch `text_seg` (VnCoreNLP). `check_segmentation.py` trên 12 câu mẫu (từ vựng `text_seg`: 18.951 token, 3.369 từ ghép) cho: bộ tách thống nhất `seg_utils` 11/11 từ ghép có trong từ vựng (100%), underthesea 10/13 (76,9%); hai bộ khác nhau ở 5/12 câu; underthesea tạo `ống_suông`, `cổ_lọ`, `quần_tây` không có trong từ vựng. Trong `text_seg`, `ống_suông`, `cổ_lọ`, `quần_tây`, `áo_len` đều 0 lần, `mũ_lưỡi_trai` 44 lần; chỉ có dạng dấu thanh `khoá` (1.438), không có `khóa` (0).

V4 đã được sửa để dùng `seg_utils.segment_text` và chạy lại. Kiểm tra trực tiếp trên `pseudo_train.csv`: **3.876/3.876 từ ghép (100%) trong `query_text` có trong từ vựng `text_seg`** (chưa kiểm cho `pseudo_val.csv`). Hạn chế: chỉ đo từ ghép đã tạo có nằm trong từ vựng, không đo việc bỏ sót từ ghép. Quy tắc giữ nguyên: mọi truy vấn (V4, V7, V8 và API) phải qua `seg_utils.segment_text`. Docstring của V7 và V8 khai báo dùng `seg_utils` (V7: cho TF-IDF/BM25; V8: trước khi mã hóa/chấm điểm); cần xác nhận lại trong mã khi chạy thật. Ngoại lệ có chủ đích duy nhất đã khai báo: `viclip_ot` của V7 mặc định đưa truy vấn **không tách từ** (chỉ bật tách từ bằng `--viclip-seg`, chọn trước khi chạy).

### 9.5 V5–V6 — âm khó và cross-encoder (thiết kế; log không có trong bộ file)

- V5: với mỗi truy vấn giả, lấy top-50 bằng chỉ mục ảnh của dual-encoder (`config.CHECKPOINT_BEST`; dòng ghi `best_18112026.pt` trong docstring là lỗi thời), lọc "âm giả" theo thuộc tính + đồng nghĩa, chọn **4 âm khó** cho mỗi 1 dương (tối đa 5 cặp/truy vấn). Chỉ mục lưu ở `index/experiments/{train,val}/`, không ghi đè index serving. Bắt buộc **đọc bằng mắt 20 cặp âm** (cần ≥18 thật sự không khớp), nếu thấp hơn thì chạy lại với `--strict-n1`.
- V6: PhoBERT-base + `Linear(768→1)` ghép cặp query–document, `MAX_LEN=256`, batch 32, tối đa 10 epoch, early stopping patience 3, LR 2e-5 (PhoBERT) / 1e-4 (head), warmup 10% rồi cosine, trần thời gian 2,25 giờ. Do tỉ lệ 1 dương : 4 âm, val_acc có mốc 0,80 nếu mô hình đoán toàn "âm" — nên theo dõi `val_top1`.
- Notebook Colab `vimfd_cross_encoder_colab.ipynb` mặc định `RUN_V3_V4=False` (dùng sẵn `product_attrs.json`/`pseudo_*.csv`), `STRICT_N1=False`, `V6_SMOKE=True`, `V6_TRAIN_FRAC=1.0`; giải nén `vimfd-cross-encoder.zip` từ Drive. **Lưu ý**: với `RUN_V3_V4=False`, notebook dùng `product_attrs.json`/`pseudo_*.csv` có trong file zip trên Drive; phải nén và upload lại zip với bản V3/V4 đã chạy lại (mục 9.4), nếu không Colab sẽ dùng file cũ lỗi tách từ.
**Kết quả V5 (đã xác nhận từ notebook `vimfd_cross_encoder_colab.ipynb`):**
- Train: 5.155 cặp dương + 20.543 cặp âm (tỉ lệ 1:3,99); val: 860 cặp dương + 3.440 cặp âm (1:4,00)
- Tổng: `ce_pairs_train.csv` 25.819 cặp (~16,2 MB), `ce_pairs_val.csv` 4.300 cặp (~2,6 MB)
- 20 cặp âm tự động kiểm tra: **20/20 passed** (không cần chạy lại với `--strict-n1`)

**Kết quả V6 (đã xác nhận từ notebook):**

| Epoch | train_loss | val_loss | val_acc | val_auc | val_top1 | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0,4210 | 0,2674 | 0,8919 | 0,9324 | 0,8023 | best |
| 2 | — | 0,2071 | 0,9267 | 0,9696 | 0,9012 | best |
| 3 | — | **0,1726** | **0,9374** | **0,9769** | **0,9209** | **BEST (saved)** |
| 4–6 | — | Không cải thiện | — | — | — | early stop (patience=3) |

Tổng thời gian huấn luyện: **24,4 phút** trên Tesla T4 (15,6 GB VRAM). Checkpoint: `checkpoints/cross_encoder_best/` (~540 MB, `model.safetensors`). Input: 25.819 cặp train, 4.300 cặp val; FP16=True, seed=42.

### 9.6 V1 → `queries_200.csv` → G1: bộ truy vấn benchmark

Luồng: V1 chọn hạt giống → người viết A (Q001–Q100) và B (Q101–Q200) viết `query_text`, `attrs_in_query` vào `seeds_A.xlsx`/`seeds_B.xlsx` → `build_queries200.py` gộp → `g1_check_queries200.py` kiểm tra. (Mục này đặt sau 9.5 để không đổi số các mục đã được dẫn; thứ tự thời gian thực tế là V0 → V1 → … → G1.)

**V1 (`v1_seeds.py`, seed 42).** Ứng viên = Test − `bad_products.txt` = **1.140**; trong đó **878 (77,0%)** có ít nhất một nhóm thuộc tính (màu, chất liệu, kiểu dáng, phong cách) mà người viết xác nhận được trên thẻ hạt giống. Nhóm thuộc tính dò bằng bộ trích của V3 (`attr_utils.py`); kiểu dáng/phong cách chỉ dò trong `name` + 300 ký tự đầu của `description` (thẻ hạt giống chỉ hiện 300 ký tự), và kiểu dáng chỉ dò cho áo/quần. `n_target` = 1/2/≥3 cần ≥1/2/3 nhóm; `n_target ≥ 3` chỉ cho áo/quần. Số ứng viên đủ điều kiện (nội địa / sàn TMĐT):

| Danh mục | n_target ≥ 1 | n_target ≥ 2 | n_target ≥ 3 |
|---|---:|---:|---:|
| áo | 193 / 279 | 123 / 137 | 32 / 25 |
| quần | 104 / 146 | 75 / 86 | 19 / 25 |
| phụ kiện | 35 / 121 | 8 / 30 | 0 / 0 |

Kết quả chọn mẫu, mỗi người, chỉ slot chính (A và B không có sản phẩm chung):

| | Slot | n_target 1/2/3 | Áo/Quần/Phụ kiện | Nội địa | Dự phòng |
|---|---:|---|---|---:|---:|
| A | 100 | 35/35/30 | 40/30/30 | 40 | 23 |
| B | 100 | 35/35/30 | 40/30/30 | 36 | 22 |

Đạt chỉ tiêu (100 truy vấn/người; 35/35/30; 40/30/30; nội địa ≥ 30). V1 in 4 cảnh báo **thiếu hạt giống dự phòng** do các lớp quá nhỏ (phụ kiện nội địa với n_target ≥ 2 chỉ có 8 ứng viên; áo sàn TMĐT với n_target ≥ 3 chỉ có 25): người A thiếu 2 ở (phụ kiện, n=2, nội địa) và 1 ở (áo, n=3, sàn TMĐT); người B thiếu 1 ở (phụ kiện, n=2, nội địa) và 3 ở (áo, n=3, sàn TMĐT). Hậu quả chỉ là ít phương án thay thế nếu một hạt giống chính không viết được truy vấn.

**Gộp (`build_queries200.py`).** Ghi `queries_200.csv` 200 dòng: cột `query_id`, `seed_id`, `category`, `name`, `n_attrs`, `attrs_in_query`, `query_text`, `author`, `is_overlap`; `author` = A cho Q001–Q100, B cho Q101–Q200 (100/100); `is_overlap` = 1 cho Q086–Q100 và Q186–Q200 (**30** truy vấn). Script chỉ cảnh báo, không chặn: `seeds_B.xlsx` có **41 dòng `skipped_reason` khác rỗng** (ví dụ Q106, Q110, Q112, Q114, Q120); không có cảnh báo `reserve=1` nên có vẻ không dòng nào dùng hạt giống dự phòng. Ý nghĩa chính xác của `skipped_reason` không thể suy ra từ bộ file; **G1 không kiểm cột này**, cần B rà lại.

**G1 (`g1_check_queries200.py`).** Kết quả: **0 lỗi, 1 cảnh báo.**

| Kiểm tra | Kết quả |
|---|---|
| Đủ cột; đủ 200 dòng; `query_id` không trùng, đúng dạng Q001–Q200 | Đạt |
| `query_text` 2–12 từ; không trùng sau chuẩn hóa | Đạt |
| `n_attrs` 1 / 2 / 3+ (chỉ tiêu 70/70/60, sai số ≤ 8) | **70 / 70 / 60** — đạt |
| `attrs_in_query` đúng định dạng; số nhóm khớp `n_attrs` | Đạt |
| Danh mục áo / quần / phụ kiện (chỉ tiêu 80/60/60) | **80 / 60 / 60** — đạt |
| `seed_id` không trùng, thuộc `test.jsonl`, không nằm trong `bad_products` | Đạt |
| Không chép cụm ≥ 4 từ liên tiếp từ `name` (tra `name` trong CSV) | Đạt |
| Không có tên thương hiệu (danh sách 17 từ khóa); không có giá/size/khuyến mãi/mã | Đạt |
| `is_overlap` đúng Q086–Q100, Q186–Q200 | Đạt |
| Tỉ lệ truy vấn dùng từ đồng nghĩa (≥ 30%) | **103/200 (52%)** — đạt |
| Truy vấn theo thứ tự "loại, màu, …" (≤ 40%) | **86/200 (43%)** — **cảnh báo** |

Điểm cần lưu ý khi đọc G1:

- Cảnh báo 43% là **ước lượng** bằng cách tìm vị trí giá trị thuộc tính trong chuỗi truy vấn; chính script khuyến nghị đếm lại bằng mắt vì sát ngưỡng. Chưa xử lý.
- Con số 52% đồng nghĩa dựa trên một **danh sách từ khóa cố định** (`SYN_WORDS`, có cả mục `"len "` — không phải cặp đồng nghĩa theo nghĩa thông thường), nên có thể cao hơn thực tế; chỉ nên coi là mức ước lượng, không trích như số đo chính xác.
- G1 kiểm **hình thức và chỉ tiêu**, không kiểm nội dung: không xác nhận `attrs_in_query` thật sự đúng với thẻ hạt giống, cũng không đo chất lượng diễn đạt. Ngoài ra `is_overlap` chỉ được kiểm theo dải `query_id`.
- Phân bố nguồn bị lệch về nội địa: **76/200 = 38%** hạt giống nội địa (A 40 + B 36, theo log V1) so với 27,0% sản phẩm Test (391/1.449) và 34,0% ứng viên (388/1.140). Cần nêu rõ khi diễn giải kết quả (liên hệ mục 9.2).
- `queries_200.csv` chỉ chứa truy vấn và hạt giống, không chứa nhãn liên quan; nhãn nằm ở `qrels.txt` (mục 9.9) và chỉ số ở mục 9.10.

### 9.7 V7, V7b, V8 — thiết kế đọc từ mã (V7, V8 đã chạy ra run file)

Mô tả dưới đây lấy từ docstring/hằng số trong script và notebook. Log chạy không nằm trong bộ file; tám file run (V7: tfidf, bm25, clip_vi2en, bienc_text, fashionclip_vi2en, viclip_ot; V8: dual_encoder, vimfr) đã tồn tại, mỗi file 10.000 dòng. Kết quả ở mục 9.10.

- **Corpus**: V7 và V8 đều nạp toàn bộ `data/splits/test.jsonl`, tức **1.449 sản phẩm**. Docstring của V7 còn ghi "corpus 1.339" — con số cũ của split trước, **đã lỗi thời**. Cả hai không lọc `bad_products`, nên 309 sản phẩm mô tả rỗng vẫn nằm trong corpus (chỉ không được dùng làm hạt giống). Sản phẩm thiếu file ảnh không vào chỉ mục của hệ thống dùng ảnh (số lượng ghi vào log; V0 cho thấy không sản phẩm Test nào thiếu ảnh đại diện, nên dự kiến là 0).
- **V7 — 6 hệ thống** (`results/runs/{tfidf,bm25,clip_vi2en,bienc_text,fashionclip_vi2en,viclip_ot}.run`, định dạng `qid Q0 product_id rank score hệ_thống`, `TOP_K=50`): TF-IDF, BM25 (`k1=1,2`, `b=0,75`), CLIP ViT-B/32 gốc với truy vấn dịch vi→en (`vinai/vinai-translate-vi2en`, 4 beam), bi-encoder văn bản (`sentence-transformers` + `pyvi`), FashionCLIP (CLIP tinh chỉnh trên ảnh thời trang Farfetch) dùng **chung** `translations.csv` và chỉ mục ảnh với `clip_vi2en`, và ViCLIP-OT (truy vấn tiếng Việt gốc, `trust_remote_code=True`, giấy phép GPL-3.0, cần `transformers>=4.57,<5` và `torch>=2.8,<2.10`). Văn bản sản phẩm lấy theo `config.MODEL_TEXT_FIELD` (`text_seg` → `text` → `name`); hạng bằng điểm phá bằng thứ tự trong `test.jsonl`; tham số cố định, **không** chỉnh theo nhãn hay kết quả.
- **V7b — kiểm tra mất nghĩa khi dịch** (**đã chạy**, kết quả từ notebook cell-11): đọc `queries_200.csv` và `translations.csv`, ghi `translation_audit.csv` và `log_B.md`. Mỗi thuộc tính được xếp `kept / partial / mistranslated / lost / unchecked` bằng từ điển regex (`ATTR_RULES`), cộng kiểm tra danh từ chính (`HEAD_RULES`) và chữ Việt còn sót; cờ `mt_loss_flag` = 1 nếu mất/sai thuộc tính, mất danh từ chính hoặc còn chữ Việt.

  **Kết quả V7b (200 truy vấn, 401 thuộc tính kiểm tra, 0 ngoài từ điển):**
  - Kept: 91,5% | Partial: 1,7% | Mistranslated: 1,0% | Lost: 5,7%
  - Queries giữ toàn bộ thuộc tính: **166/200**; queries có mt_loss_flag=1: **35/200**
  - Theo loại thuộc tính: màu 97,0% kept; phong cách 98,5% kept; chất liệu 90,9% kept; kiểu dáng 78,7% kept (mất nhiều nhất)
  - Theo nhóm sản phẩm bị gắn cờ: phụ kiện 5,0%; quần 26,7%; áo 20,0%
  - Head (danh từ chính): kept 189, lost 9

  Giới hạn do chính script nêu: đây là **phép kiểm bằng từ điển, không phải đánh giá của người**, đo "bản dịch có giữ từ khóa" chứ không đo "CLIP có hiểu từ đó"; giá trị ngoài từ điển (ví dụ nhiều kiểu dáng như `wide leg`, `tapered`) được xếp `unchecked`. Cần đọc bằng mắt mọi dòng `mt_loss_flag=1` và một mẫu dòng `=0` trước khi dùng số liệu.
- **V8**: chạy dual-encoder và ViMFR (dual-encoder lấy top-50, cross-encoder rerank) trên 200 truy vấn; ghi `dual_encoder.run`, `vimfr.run`, `latency.csv`; chỉ mục Test lưu ở `index/experiments/test/`. Yêu cầu: (1) checkpoint dual-encoder từ `VIMFD_DE_CKPT` hoặc `config.CHECKPOINT_BEST`, **phải train trên split hiện tại** (nếu không, kết quả có thể bị thổi phồng vì checkpoint có thể đã thấy sản phẩm Test mới); (2) `checkpoints/cross_encoder_best/` từ V6 — notebook kiểm tra `linear_head.pt`; `vimfr.run` đã được tạo nên điều kiện này đã được thỏa theo một cách nào đó, nhưng checkpoint và log V6 **chưa được xác minh** ở đây. Bản sửa: truy vấn được chuẩn hóa và tách từ; sản phẩm thiếu ảnh không vào chỉ mục (không còn thay bằng ảnh đen); đo độ trễ đủ ba bước (mã hóa truy vấn gồm tách từ, FAISS, rerank) có đồng bộ GPU; kiểm tra product_id thuộc Test, không trùng, điểm giảm dần, hai file cùng tập 50 ứng viên.
- **Notebook `vimfr_colab_v7_v8.ipynb`** (Colab T4): giải nén `vimfd-eval.zip` (tạo bởi `make_eval_colab_zip.py` — script này không nằm trong bộ file rà soát) → cài thư viện → thử `--sample5` → `clip_vi2en` (tạo `translations.csv`) → `v7b_translation_audit.py` → `tfidf,bm25,bienc_text,fashionclip_vi2en` → `viclip_ot` chạy riêng → V8 `--sample5` rồi `--rebuild-index` → rsync kết quả về Drive. **Không chạy V7 và V8 cùng lúc** vì V8 đo độ trễ. Nếu `config.py` còn đường dẫn Windows (`C:\...`) thì phải sửa trước khi chạy trên Colab.

### 9.8 V9 — pool gán nhãn (`v9_pooling.py`, đã đọc mã và log)

**Cách lập pool (theo mã).** Đầu vào: tám file run, `queries_200.csv`, `test.jsonl`. Mỗi truy vấn: lấy top-K của từng hệ thống rồi hợp nhất, loại trùng theo `product_id`, thêm hạt giống. K của `dual_encoder` và `vimfr` cố định **10**; K của sáu baseline là 5, hạ xuống 3 rồi 2 nếu N vượt ngưỡng G3 = 3.800 (cổng G3 **không** hạ K của dual_encoder và vimfr). Thứ tự cặp trong mỗi truy vấn được xáo bằng seed (42 cho A, 43 cho B) cộng số thứ tự truy vấn; file không có cột lộ hệ thống, hạt giống, thứ hạng, điểm hay nhãn của người kia; cột `label` chỉ nhận 0/1/2. Nguồn từng cặp nằm ở `_private/pool_sources.csv` (không được mở khi gán nhãn).

**Kết quả (log chạy):**

| K baseline | N (cặp) |
|---:|---:|
| 5 | 7.169 |
| 3 | 5.776 |
| 2 | **5.063** (được chọn) |

- N = 5.063 **vượt ngưỡng G3 (3.800) ngay cả khi K = 2**; script giữ K = 2 và chấp nhận buổi gán nhãn dài hơn. Cỡ pool mỗi truy vấn: 17–32, trung bình 25,3. Pool gồm **1.187 sản phẩm**, 0 sản phẩm thiếu ảnh.
- Hạt giống đã nằm sẵn trong top-K của ít nhất một hệ thống ở 152/200 truy vấn; 48/200 là do được thêm vào.
- Chia file: A gán Q101–Q185 và B gán Q001–Q085 (mỗi người 85 truy vấn chia thành 4 file `_02`–`_05`), cả hai cùng gán 30 truy vấn overlap (Q086–Q100, Q186–Q200) trong file `_01`. `A_01` và `B_01` chứa **cùng 776 dòng** nhưng thứ tự giống nhau chỉ 4,1%. Số cặp: `A_01`/`B_01` 776; `A_02`–`A_05` 2.174; `B_02`–`B_05` 2.113; tổng 5.063. Mã V9 tự ghi chú: file `_01` có 30 truy vấn (> 25 truy vấn/file ở mục 8.4 của kế hoạch) — cần ghi vào mục 12 nếu giữ nguyên.
- Công gán nhãn theo log V9 ước 4,9 giờ mỗi người (hai người) hoặc 9,7 giờ nếu một người gán cả hai phần (6 giây/cặp). Trong dự án này người dùng gán 5.063 cặp (A_01–A_05 và B_02–B_05), người thứ hai chỉ gán B_01 (776 cặp).
- Log V9 không cho biết có dùng `--skip-baselines` hay không; vì `judged@10` ở V11 của cả sáu baseline đều > 0 nên có vẻ không bỏ hệ thống nào.

**Hệ quả thiết kế đã được mã V9 tự ghi (pool bias).** Vì dual_encoder và vimfr được lấy top-10 còn mỗi baseline chỉ top-2, sản phẩm nằm trong top-10 của baseline nhưng ngoài pool bị tính là không liên quan, nên điểm của baseline có thể bị đánh thấp, có lợi cho ViMFR. Điều này khớp với `judged@10` ở V11: 1,0000 cho dual_encoder và vimfr so với 0,30–0,51 cho các baseline (mục 9.10). Đây không còn là suy luận mà là hệ quả trực tiếp của K = 10 / K = 2.

### 9.9 V10 — quy tắc gán nhãn, κ và ghép nhãn

**Quy tắc (đề xuất, cần ghi vào mục 12 của kế hoạch).** Quy trình 3 bước: (1) cổng loại — sai loại thì nhãn 0; (2) với mỗi thuộc tính trong `attrs_in_query` xét KHỚP / MÂU THUẪN / KHÔNG RÕ, không rõ không bao giờ tính là khớp; (3) gọi n là số thuộc tính, m số khớp, c số mâu thuẫn: c ≥ 1 → 0; m = n → 2; m ≥ ⌈n/2⌉ → 1; còn lại → 0.

**Nhãn của người dùng là nhãn sơ bộ có máy hỗ trợ**: bộ quy tắc chỉ đọc chữ trên thẻ, không đọc ảnh. Cần ghi rõ điều này trong báo cáo.

**Lần κ đầu (κ = 1,000) không hợp lệ và đã bị loại.** Hai bộ nhãn khi đó cùng do một bộ quy tắc sinh ra trên cùng 776 dòng nên giống nhau theo cách xây dựng; kết quả "QUA G4" không có giá trị. Người thứ hai thật sau đó đã gán lại B_01 độc lập.

**Kết quả κ hợp lệ (tập overlap, 776 cặp, 30 truy vấn):**

| Chỉ số | Giá trị |
|---|---:|
| κ trọng số tuyến tính (chỉ số chính) | **0,712** |
| κ trọng số bậc hai (tham khảo) | 0,800 |
| Tỉ lệ đồng thuận chính xác | 0,825 |
| Kết luận | Qua G4 (κ ≥ 0,61) |

Ma trận nhầm lẫn (hàng = người dùng, cột = người thứ hai; nhãn 0/1/2): hàng 0 = 501/121/11; hàng 1 = 0/33/0; hàng 2 = 0/4/106. **136 dòng bất đồng, trong đó 132 dòng (121 + 11) là người dùng gán thấp hơn**, chỉ 4 dòng ngược lại. Chiều lệch rõ ràng này phù hợp với giả thuyết rằng nhãn máy hỗ trợ chặt hơn nhãn người, vì không nhìn ảnh nên các thuộc tính "không rõ" bị tính là không khớp; nhận định của hai người gán cũng xác nhận nhiều dòng bất đồng nên thiên về nhãn 1. Số dòng đổi nhãn sau khi họp không được ghi lại.

**Ghép nhãn (`merge`):** 5.063 cặp, 200 truy vấn, **0 truy vấn có R_q = 0**. Phân bố nhãn 0/1/2 = **3.687 / 564 / 812** (72,8% / 11,1% / 16,0%). Nguồn nhãn: `single_user` 4.287, `double` (hai người giống nhau) 640, `họp` 134, `thấp hơn` 2 (dòng bất đồng để trống `final_label`). Đầu ra: `labels_merged.csv`, `qrels.txt`.

**Hạn chế cần nêu khi báo cáo:**
- κ chỉ đo trên 30/200 truy vấn overlap; 4.287 cặp còn lại (84,7% pool) chỉ có nhãn của một người.
- Thiên lệch thấp của nhãn người dùng ở overlap nhiều khả năng cũng có ở các file không overlap nếu chưa được soát lại bằng ảnh. Chưa có thông tin cho thấy phần này đã được soát đầy đủ.
- Nhãn 1 chỉ chiếm 11,1%, nên NDCG với gain 2^label − 1 chủ yếu phân biệt giữa nhãn 2 và còn lại.
- Script `v10_kappa_merge.py` đọc `disagreements.csv` bằng `encoding="utf-8"`; lưu file bằng Excel "CSV UTF-8" thêm BOM và làm `merge` lỗi `KeyError`. Nên dùng `utf-8-sig` hoặc lưu bằng trình soạn thảo không thêm BOM.

### 9.10 V11 — chỉ số trên 200 truy vấn, tám hệ thống

`v11_metrics.py` đã được sửa so với bản gốc: thêm `fashionclip_vi2en` và `viclip_ot` vào danh sách hệ thống (bản gốc chỉ có 6) và thêm `judged@10` (tỉ lệ sản phẩm trong top-10 đã có nhãn trong `qrels.txt`, kể cả nhãn 0). Cặp chưa có nhãn bị tính là không liên quan. Recall@K có mẫu số là số sản phẩm liên quan **trong pool** (R_q). CI là bootstrap 95% theo truy vấn (seed 42). Cả 200 truy vấn đều hợp lệ (R_q ≥ 1).

| Hệ thống | R@5 | R@10 [CI 95%] | NDCG@10 [CI 95%] | MRR@10 | judged@10 |
|---|---:|---|---|---:|---:|
| tfidf | 0,2623 | 0,3665 [0,3266–0,4072] | 0,3662 [0,3275–0,4049] | 0,5692 | 0,4795 |
| bm25 | 0,3402 | 0,4526 [0,4097–0,4969] | **0,4883** [0,4462–0,5293] | **0,7100** | 0,5015 |
| clip_vi2en | 0,1826 | 0,3093 [0,2695–0,3493] | 0,2676 [0,2335–0,3028] | 0,4709 | 0,5005 |
| bienc_text | 0,2534 | 0,3581 [0,3188–0,3985] | 0,3534 [0,3173–0,3894] | 0,5757 | 0,4325 |
| fashionclip_vi2en | 0,2439 | 0,3685 [0,3289–0,4097] | 0,3516 [0,3138–0,3906] | 0,5758 | 0,5050 |
| viclip_ot | 0,0439 | 0,0708 [0,0515–0,0914] | 0,0618 [0,0456–0,0792] | 0,1590 | 0,2965 |
| dual_encoder | 0,2023 | 0,3661 [0,3282–0,4040] | 0,3052 [0,2693–0,3416] | 0,4558 | **1,0000** |
| vimfr | 0,3118 | **0,5029** [0,4627–0,5431] | 0,4766 [0,4370–0,5168] | 0,6542 | **1,0000** |

**So sánh có cặp (bootstrap theo truy vấn, NDCG@10):**
- ViMFR so với baseline tốt nhất theo NDCG@10 (bm25): Δ = −0,0117, CI [−0,0658; 0,0414] — **không có ý nghĩa**.
- ViMFR so với dual_encoder: Δ = +0,1714, CI [0,1202; 0,2226] — **có ý nghĩa**; phần cải thiện này là hiệu quả của bước rerank bằng cross-encoder trên top-50 của dual_encoder.
- Chưa có so sánh có cặp cho Recall@10 và cho các baseline khác. CI của R@10 giữa vimfr (0,4627–0,5431) và bm25 (0,4097–0,4969) chồng lên nhau, nên **không được kết luận ViMFR vượt bm25 ở Recall@10** chỉ từ bảng này.

**Pool bias — điểm quan trọng nhất khi đọc bảng.** `judged@10` của dual_encoder và vimfr là 1,0000, còn các baseline chỉ 0,30–0,51 (viclip_ot thấp nhất, 0,2965). Sản phẩm top-10 của baseline chưa được gán nhãn bị tính là 0, nên điểm của baseline bị đánh thấp, còn của dual_encoder và vimfr không bị. Hệ quả: (a) khoảng cách giữa vimfr/dual_encoder và các baseline không so sánh trên cùng điều kiện; (b) dù có lợi thế này, ViMFR vẫn không vượt bm25 có ý nghĩa ở NDCG@10, tức là kết quả không ủng hộ khẳng định "ViMFR tốt hơn mọi baseline"; (c) các bảng nên luôn đặt cột `judged@10` cạnh điểm.

**Theo nhóm (NDCG@10; mỗi nhóm 60–80 truy vấn, không có CI, chỉ mang tính tham khảo):**

| Hệ thống | n_attrs = 1 | n_attrs = 2 | n_attrs = 3+ | phụ kiện | quần | áo |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 0,3588 | 0,4256 | 0,3056 | 0,4599 | 0,3064 | 0,3408 |
| bm25 | 0,4887 | 0,5603 | 0,4038 | 0,6638 | 0,3914 | 0,4294 |
| clip_vi2en | 0,3158 | 0,2987 | 0,1750 | 0,3788 | 0,2478 | 0,1990 |
| bienc_text | 0,3918 | 0,3956 | 0,2592 | 0,4626 | 0,3126 | 0,3020 |
| fashionclip_vi2en | 0,3595 | 0,4161 | 0,2671 | 0,5068 | 0,3121 | 0,2649 |
| viclip_ot | 0,0796 | 0,0727 | 0,0285 | 0,1044 | 0,0034 | 0,0737 |
| dual_encoder | 0,3689 | 0,3366 | 0,1943 | 0,4490 | 0,2414 | 0,2453 |
| vimfr | **0,6166** | 0,4171 | 0,3826 | 0,5490 | **0,4907** | 0,4117 |

Quan sát (thăm dò): vimfr cao nhất ở n_attrs = 1 và nhóm quần; bm25 cao nhất ở n_attrs = 2, 3+, phụ kiện và áo. Mọi hệ thống giảm khi số thuộc tính tăng, trừ vài trường hợp nhỏ; nhóm n_attrs = 3+ khó nhất với các hệ thống dựa trên ảnh (clip_vi2en, dual_encoder). Nhóm theo `n_attrs`/`category` đọc từ `queries_200.csv`; số truy vấn mỗi nhóm (70/70/60 và 80/60/60) **trùng đúng với số liệu G1 trước khi sửa dữ liệu truy vấn**, nên nhiều khả năng các sửa dữ liệu (Q120, Q168, Q127, Q076, Q135, Q153, Q171, `category` của Q034, Q049, Q079, Q083, Q130, Q148, Q198) **chưa được áp dụng**; cần xác nhận trước khi trích bảng nhóm.

**Điểm bất thường cần lưu ý:** `viclip_ot` thấp hơn hẳn mọi hệ thống khác (NDCG@10 0,0618; R@10 0,0708; quần 0,0034). **File run đã được xác nhận hợp lệ** (định dạng TREC đúng, 10.000 dòng, 200 truy vấn, điểm ~0,53–0,60 bình thường). Nguyên nhân điểm thấp nhiều khả năng là pool bias (judged@10 = 0,2965 — thấp nhất trong 8 hệ thống): sản phẩm viclip_ot trả về nhưng không có nhãn bị tính là 0, kéo điểm xuống mạnh hơn so với các baseline khác. Trước khi đưa số này vào báo cáo, nên xác nhận việc có dùng `--viclip-seg` và phiên bản `transformers`/`torch` đúng yêu cầu (mục 9.7). Ngoài ra `dual_encoder` (R@10 0,3661) ngang tfidf (0,3665) và thấp hơn bm25; nếu checkpoint dual-encoder không được train trên split 14.470 hiện tại (mục 11, điểm 14), kết quả của dual_encoder và vimfr có thể bị ảnh hưởng theo cả hai chiều.

## 10. Độ trễ và vận hành (chưa đối chiếu lại lần này)

Số liệu latency đã được xác nhận từ notebook `vimfr_colab_v7_v8.ipynb` cell-16 (Tesla T4, V8, `vimfr.run`): **median = 154,8 ms, P95 = 203,2 ms** (đo đủ ba bước: mã hóa truy vấn gồm tách từ + FAISS + rerank, có đồng bộ GPU). Đây là latency của ViMFR (dual-encoder + cross-encoder rerank) trên Colab T4, chưa phải trên cấu hình triển khai xác định. Khuyến nghị: đo lại end-to-end trên cấu hình triển khai thực tế, sau khi index/checkpoint đã được đồng bộ (mục 7).

Khởi chạy theo mã nguồn, từ thư mục gốc:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Trạng thái triển khai:** API chạy được trên máy local và truy cập được từ máy khác cùng mạng qua `http://<ip>:8000` (do bind `0.0.0.0`); chưa được triển khai production. Khi báo cáo nên ghi "API REST bằng FastAPI, chạy local, chưa triển khai production". Để đưa lên môi trường thật cần (theo đánh giá, chưa kiểm chứng trong mã): hosting chạy liên tục có tên miền và HTTPS (reverse proxy như Nginx/Caddy); môi trường có Java (VnCoreNLP), thư mục `vncorenlp/`, checkpoint, index và ảnh; giới hạn CORS (hiện `allow_origins=["*"]`); xác thực/rate limit (endpoint tìm kiếm có xử lý ảnh, nặng); chạy **một worker** (mỗi worker uvicorn nạp model, embedding và JVM riêng); tự khởi động lại và log/theo dõi `/api/health` (systemd hoặc Docker); build lại index và khởi động lại API mỗi khi đổi checkpoint hoặc split.

API dùng CUDA nếu khả dụng, nếu không dùng CPU. Upload ảnh giới hạn JPG/PNG/WEBP, tối đa 5 MB (đã đối chiếu với `main.py`). Khi khởi động, `SearchEngine` từ chối chạy nếu index lệch checkpoint hoặc dữ liệu hiện tại (trừ khi `ALLOW_STALE_INDEX = True`); `GET /api/health` trả số sản phẩm, thiết bị, tên checkpoint, tên thư mục split và metadata index.

## 11. Các điểm cần hoàn thiện trước khi công bố

1. **(Đã xong 25/11/2026)** Chạy lại `evaluate.py` trên index mới (14.470 vector) và checkpoint `best_short_24112026.pt`; số liệu mục 8 lấy từ `results/metrics.txt`. Còn lại: xác nhận lần chạy không dùng `--allow-stale` và lưu kèm `results/metrics.json` làm artifact.
2. **(Đã xong)** Mức nhóm (`group_id`) đã được thêm vào `evaluate.py`, song song với mức sản phẩm và cùng quy tắc tính hạng với `train.py`'s `validate()`.
3. **(Đã xong trong `evaluate.py`)** Hai điểm không khớp với `train.py` (`IDENTICAL_THR`, cờ `"ok"`) nay do `evaluate.py` tự định nghĩa. Tùy chọn: chuyển hai thứ này vào `train.py` để dùng chung; không bắt buộc.
4. Dùng mean/std RGB riêng của bộ dữ liệu (ví dụ mean ≈ (0,660; 0,641; 0,636), std ≈ (0,333; 0,337; 0,339) theo báo cáo dữ liệu trên tập train) thay cho mean/std của CLIP đang dùng trong `config.py`, vì ảnh đã được đệm trắng khác phân phối với ảnh gốc CLIP được huấn luyện.
5. Cài `transformers` và chạy lại `data_validation.py` để có độ dài token thật (thay vì ước lượng 1,3 token/từ); soát 54 nhãn category nghi sai và sản phẩm lạc đề còn lọt bộ lọc.
6. Xử lý trường giá rỗng/không hợp lệ một cách tường minh, hoặc ghi chú rõ ràng mức độ tin cậy của tính năng lọc/sắp xếp giá (hiện không có số liệu kiểm định mới để khẳng định mức độ nghiêm trọng).
7. Bổ sung manifest tái lập: nguồn, hash dữ liệu, thời điểm crawl, split seed, phiên bản preprocessing, checkpoint hash và lệnh build — đặc biệt để tránh lặp lại tình trạng mục 7 (index lệch checkpoint/dữ liệu).
8. Pipeline annotation: `queries_200.csv`, V5, V6, V7, V7b, V8, V9, V10, V11 đã chạy và có log từ notebook. Còn lại: (tùy chọn) bổ sung 2 type còn lại vào `TYPE_MAP` và chạy lại V3 → V4; đọc bằng mắt mọi dòng `mt_loss_flag=1` của V7b (35/200 truy vấn) và mẫu dòng `=0`; xác nhận latency trên cấu hình triển khai thực tế (số T4 đã có: median 154,8 ms, P95 203,2 ms). Không chạy V7 cùng lúc với V8.
9. Paired bootstrap CI: V11 mới so sánh ViMFR với baseline tốt nhất (bm25) và với dual_encoder, chỉ cho NDCG@10. Cần bổ sung so sánh có cặp cho Recall@10 và MRR, và so với các baseline khác nếu cần; công bố rõ pool bias (mục 9.10).
10. Bổ sung kiểm thử tự động cho catalog, fingerprint/index, API, evaluator và các trường hợp metadata thiếu.
11. Khi báo cáo benchmark, nêu rõ rằng 309/1.449 sản phẩm Test (mô tả rỗng) bị loại khỏi hạt giống (còn 1.140) nhưng vẫn nằm trong corpus xếp hạng của V7/V8 (1.449), và phân bố nguồn của hạt giống lệch về nội địa (38% so với 27,0% của Test; mục 9.2, 9.6).
12. Rà các điểm G1 chưa đóng: đếm lại bằng mắt tỉ lệ truy vấn dạng "loại, màu, …" (ước lượng 43% > ngưỡng 40%); rà 41 dòng `skipped_reason` của `seeds_B.xlsx`; coi 52% đồng nghĩa là ước lượng (danh sách từ khóa có `"len "`).
13. Sửa docstring `v7_baselines.py` ("corpus 1.339" → 1.449) và quyết định có giữ 309 sản phẩm mô tả rỗng trong corpus đánh giá hay không (hiện V7/V8 giữ).
14. Xác nhận checkpoint dual-encoder dùng cho V5 và V8 (`best_short_24112026.pt` hoặc `VIMFD_DE_CKPT`) được huấn luyện trên split 14.470 hiện tại, vì V8 yêu cầu điều này.
15. **Pool bias**: `judged@10` của dual_encoder/vimfr = 1,0 so với 0,30–0,51 của baseline. Khi báo cáo, đặt `judged@10` cạnh mọi bảng điểm; cân nhắc gán nhãn bổ sung cho top-10 của các baseline hoặc báo cáo thêm điểm tính trên danh sách đã rút gọn (chỉ các sản phẩm đã có nhãn), và không kết luận ViMFR vượt baseline khi chênh lệch chưa có ý nghĩa (NDCG@10 so với bm25 không có ý nghĩa).
16. **Nhãn**: ghi rõ nhãn người dùng là sơ bộ có máy hỗ trợ và được soát lại; ghi κ tuyến tính 0,712 và κ bậc hai 0,800 trên 30 truy vấn overlap, kèm chiều lệch (132/136 dòng bất đồng là người dùng gán thấp hơn) và số dòng đổi sau họp. Soát lại các cặp nhãn 0 chỉ vì "không rõ" (thiếu thông tin chữ, ảnh có thể xác nhận) trong 4.287 cặp không overlap nếu chưa làm.
17. **(Đã sửa)** `queries_200.csv` đã được sửa các lỗi dữ liệu truy vấn (Q120, Q168, Q127, Q076, Q135, Q153, Q171 và `category` ở Q034, Q049, Q079, Q083, Q130, Q148, Q198). Cần chạy lại V11 để bảng nhóm (mục 9.10) phản ánh đúng bản sửa; xác nhận cột `attrs_in_query` còn khớp với nhãn đã gán.
18. **`viclip_ot.run` đã được xác nhận hợp lệ** (định dạng TREC đúng, 10.000 dòng, 200 truy vấn, điểm ~0,53–0,60 bình thường). NDCG@10 thấp (0,0618) nhiều khả năng do pool bias (judged@10 = 0,2965 — thấp nhất trong 8 hệ thống), không phải file lỗi. Nếu muốn chẩn đoán sâu hơn: kiểm tra có dùng `--viclip-seg` và phiên bản `transformers`/`torch` đúng yêu cầu. Sửa `v10_kappa_merge.py` dùng `utf-8-sig` khi đọc/ghi `disagreements.csv`.
19. **Triển khai (nếu cần)**: API mới chạy local; chuẩn bị hosting/HTTPS, giới hạn CORS, xác thực/rate limit, một worker và giám sát như mô tả ở mục 10. Với đồ án hoặc bản demo, chạy local là chấp nhận được nếu báo cáo ghi rõ.
20. **Kiểm định có cặp cho mục 8**: hiện chỉ có CI riêng từng số. Bổ sung paired bootstrap theo nhóm giữa text đầy đủ và `name`/`head4`/`head3_color` (và giữa gallery full và test-only nếu cần) trước khi kết luận chênh lệch giữa các biến thể.
21. **Phân tích lỗi mục 8**: tách kết quả theo nguồn/shop, danh mục và nhóm 309 sản phẩm test có mô tả rỗng; tìm nguyên nhân R@1 mức nhóm ở gallery full thấp (T→I text đầy đủ 0,0801); có thể thêm đo tỉ lệ đúng danh mục P@10 như `train.py`.
22. **Chọn số báo cáo**: dùng gallery full và mức nhóm làm số chính, nêu kèm số mức sản phẩm và số test-only (ghi rõ cỡ gallery 1.449); ghi rõ Image→Image chỉ có 766/1.449 sản phẩm test có ảnh phụ.

## 12. Lệnh thường dùng

```powershell
python data_validation.py
python src/training/train.py
python -m src.indexing.build_index
python src/training/evaluate.py   # --bootstrap 2000 (mặc định), --checkpoint, --allow-stale; cần evaluation_metrics.py cùng thư mục
python data/annotation/v0_bad_products.py
python data/annotation/v1_seeds.py
python data/annotation/build_queries200.py   # sau khi A, B đã điền truy vấn vào seeds_*.xlsx
python data/annotation/g1_check_queries200.py
python data/annotation/v3_attributes.py
python data/annotation/v4_pseudo_queries.py
python data/annotation/check_segmentation.py
python data/annotation/v5_hard_negatives.py   # --dry-run để thử luồng bằng TF-IDF
python data/annotation/v6_train_cross_encoder.py   # cần GPU; --smoke để thử
python data/annotation/v7_baselines.py   # --sample5 để thử; --only tfidf,bm25,... để chọn hệ thống
python data/annotation/v7b_translation_audit.py   # cần translations.csv từ V7
python data/annotation/v8_run_vimfr.py   # --sample5, --rebuild-index; cần cross_encoder_best/
python data/annotation/v10_kappa_merge.py kappa   # sau khi A_01 và B_01 được gán độc lập bởi hai người
python data/annotation/v10_kappa_merge.py merge   # sau khi điền final_label và gán xong mọi file
python data/annotation/v11_metrics.py   # --dry-run để thử bằng dữ liệu giả; cần đủ 8 file run
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Các lệnh trên mô tả entry point trong repository; chạy thành công vẫn phụ thuộc dependencies (đặc biệt `transformers` cho đo token thật, và Java + `py_vncorenlp` + thư mục model VnCoreNLP cho tiền xử lý truy vấn; `pandas` + `openpyxl` cho V3–V5 và `build_queries200.py`; `rank_bm25`, `scikit-learn`, `sentence-transformers`, `pyvi`, `sentencepiece`, `faiss-cpu`, `timm` cho V7/V8 (ViCLIP-OT cần `transformers>=4.57,<5`, `torch>=2.8,<2.10`); `underthesea` chỉ còn dùng để so sánh trong `check_segmentation.py`). Các script annotation chạy từ thư mục gốc của repo; V4 dùng `seg_utils` nên phải chạy từ thư mục gốc để import được `src.serving.text_utils`, đường dẫn dữ liệu/ảnh và checkpoint phù hợp trên máy. Đặc biệt, sau thay đổi split hoặc checkpoint phải build lại index trước khi đánh giá hay chạy API.

## 13. Kết luận

Repository hiện có một nguyên mẫu retrieval đa phương thức ở mức nghiên cứu, kèm API REST bằng FastAPI chạy local (chưa triển khai production): dual-encoder 256 chiều (CLIP ViT-B/32 + PhoBERT-base), FAISS, pipeline xử lý dữ liệu đã được tài liệu hóa chi tiết, `GroupBatchSampler` xử lý negative sai lúc train, và bộ đánh giá dual-encoder. Dữ liệu đã qua ba lượt kiểm định và lần mới nhất đạt 107 PASS / 2 WARN / 0 FAIL trên 14.470 sản phẩm, 43.536 ảnh — một cải thiện đáng kể so với lượt đầu (rò rỉ ảnh gần trùng ~26–27%, nay còn 0% ở ngưỡng dùng để chia split).

Pipeline annotation đã chạy đến V11: V0 xác nhận ba split tách rời hoàn toàn và loại 309/1.449 sản phẩm Test mô tả rỗng (còn 1.140 hạt giống); V1 chọn 2 × 100 hạt giống đạt chỉ tiêu; `queries_200.csv` qua G1 với 0 lỗi, 1 cảnh báo; V3/V4 sinh truy vấn giả bằng bộ tách từ thống nhất `seg_utils`; V9 tạo pool 5.063 cặp; V10 cho κ tuyến tính 0,712 (bậc hai 0,800) giữa hai người gán sau khi loại lần κ = 1,000 không hợp lệ, và ghép 5.063 nhãn (0/1/2 = 3.687/564/812); V11 tính chỉ số cho tám hệ thống trên 200 truy vấn. Kết quả chính: BM25 có NDCG@10 cao nhất (0,4883) và MRR@10 cao nhất (0,7100); ViMFR có Recall@10 cao nhất (0,5029) và NDCG@10 0,4766; ViMFR vượt dual_encoder có ý nghĩa (ΔNDCG@10 = +0,1714, CI [0,1202; 0,2226]) nhưng **không khác BM25 có ý nghĩa** (Δ = −0,0117, CI [−0,0658; 0,0414]).

Các kết quả trên cần đọc kèm các hạn chế: pool gán nhãn lệch (`judged@10` = 1,0 cho dual_encoder/vimfr so với 0,30–0,51 cho baseline), nhãn người dùng là nhãn sơ bộ có máy hỗ trợ với thiên lệch thấp đã quan sát ở overlap, κ chỉ đo trên 30 truy vấn, `viclip_ot` thấp bất thường chưa được chẩn đoán, và bảng theo nhóm có thể dựa trên `queries_200.csv` chưa sửa. Ngoài ra, `evaluate.py` bản mới đã được chạy trên index 14.470 vector và checkpoint `best_short_24112026.pt` (mục 8): trên gallery full, Text→Image với text đầy đủ đạt R@10 mức nhóm 0,3423 (CI 95% 0,2553–0,4233) và mức sản phẩm 0,1636; Image→Image đạt R@10 mức nhóm 0,7728; trên gallery test-only các số này lần lượt là 0,7191 và 0,9347. Text ngắn giảm rõ ở biến thể `head4` và `head3_color`. CI mức nhóm khá rộng vì test chỉ có 948 nhóm. Chưa xác nhận checkpoint dual-encoder dùng cho V5/V8 được huấn luyện trên split hiện tại, nên số liệu mục 9.10 chưa nên được dùng làm bằng chứng hiệu năng cuối cùng khi chưa xử lý mục 11 (điểm 14–18); số đo độ trễ (mục 10) vẫn cần đo lại. Các hạn chế dữ liệu đã biết còn nguyên (ảnh trùng hệt trong cùng split, văn bản nghèo thông tin ở một phần dữ liệu, một số nhãn category nghi sai).
