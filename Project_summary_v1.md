# ViMFD Retrieval — Tài liệu tổng hợp

## 1. Mục tiêu và phạm vi

ViMFD Retrieval tìm kiếm sản phẩm thời trang theo hai luồng:

- **Text → Image:** mô tả văn bản → ảnh sản phẩm.
- **Image → Image:** ảnh truy vấn → ảnh sản phẩm tương tự.

Repository gồm tiền xử lý dữ liệu, huấn luyện embedding đa phương thức, FAISS index, đánh giá và API backend.

## 2. Kiến trúc

```text
Metadata + ảnh
  → tiền xử lý và chia train/val/test
  → CLIP Vision + PhoBERT
  → hai projection head 256 chiều, L2-normalized
  → symmetric InfoNCE
  → checkpoint
  → FAISS image/text index
  → truy vấn text hoặc ảnh
```

- CLIP: `openai/clip-vit-base-patch32`.
- PhoBERT: `vinai/phobert-base`.
- Embedding: 256 chiều.
- Tìm kiếm: `FAISS IndexFlatIP`; inner product tương đương cosine vì embedding đã chuẩn hóa.
- Mã chính: [src/models/](/C:/Users/DELL/Downloads/vimfd-retrieval/src/models), [config.py](/C:/Users/DELL/Downloads/vimfd-retrieval/config.py).

## 3. Dữ liệu

### 3.1. Pipeline tiền xử lý mới

Tiền xử lý hiện được tổ chức theo hai giai đoạn độc lập:

1. `common.py` làm sạch và xuất dữ liệu cho từng nguồn.
2. `build_splits.py` gộp các nguồn đã làm sạch rồi chia train/validation/test.

Luồng này thay cho pipeline tiền xử lý cũ đọc một metadata JSON duy nhất.

```text
Metadata từng nguồn
  → common.py.clean_dataset()
  → {source}_clean.jsonl
  → build_splits.py
  → train.jsonl, val.jsonl, test.jsonl, split.json
```

### 3.2. Làm sạch dữ liệu bằng `common.py`

`common.py` thực hiện chuỗi làm sạch theo thứ tự:

1. Chuẩn hóa Unicode NFKC, HTML và khoảng trắng.
2. Kiểm tra tên bằng định dạng tiếng Việt.
3. Loại sản phẩm ngoài phạm vi thời trang nam; phụ kiện nam được giữ lại.
4. Chuẩn hóa tên, loại bỏ mã SKU, từ khuyến mãi và tiền tố thương hiệu không phù hợp.
5. Chuẩn hóa category thành ba giá trị: `áo`, `quần`, `phụ kiện`.
6. Kiểm tra và loại ảnh thiếu file, kích thước bất thường, đọc được/thất bại hoặc trùng URL.
7. Làm sạch mô tả: bỏ HTML, câu trùng mẫu, câu khuyến mãi, câu lặp lại trong từng nhóm và câu boilerplate.
8. Trích chất liệu từ mô tả hoặc tên sản phẩm.
9. Tạo `text`, sau đó chạy word-segment thành `text_seg`.
10. Tạo `text_group` để xác định các bản ghi có cùng nội dung model input.

Tên và mô tả được làm sạch, nhưng dữ liệu gốc không bị ghi đè. Mỗi nguồn có thể chạy riêng với:

```powershell
python common.py --src <nguon> --out-dir data/processed --root <thu-muc-anh-goc>
```

Đầu ra là `<source>_clean.jsonl` trong `data/processed/` cùng thư mục ảnh `data/processed/images/`.

### 3.3. Chuẩn hóa ảnh

`common.py` không crop hoặc kéo méo ảnh. Mỗi ảnh được:

- đọc và chuyển theo EXIF;
- chuyển sang RGB;
- giữ tỷ lệ khung hình;
- resize cạnh dài xuống hoặc lên tới tối đa 224;
- đặt trên nền trắng vuông 224×224;
- lưu `.jpg` hoặc giữ `.png`.

Kiểm tra ảnh mặc định được bật. Tùy chọn `--no-check-files` bỏ kiểm tra kích thước và perceptual hash. Khi export, file ảnh mới được ghi sang thư mục processed thay vì ghi đè hoặc tạo hard link vào ảnh gốc.

Ảnh được kiểm tra theo:

- file tồn tại;
- ảnh mở và đọc được;
- cạnh ngắn tối thiểu 100 pixel;
- tỷ lệ cạnh dài/cạnh ngắn không vượt 3;
- URL không trùng trong cùng sản phẩm;
- không là placeholder toàn màu.

### 3.4. Tạo văn bản model

Chuỗi `text` được ghép theo thứ tự:

```text
tên đã làm sạch
+ chất liệu trích được
+ một màu duy nhất khi dữ liệu màu là một tên màu hợp lệ
+ các câu mô tả đã làm sạch
```

Các câu mô tả lặp lại tên sản phẩm hoặc quá giống tên sẽ bị bỏ. `text` không giới hạn độ dài thủ công; `text_seg` chỉ segment tối đa 300 syllables trước khi PhoBERT tokenization sử dụng `max_length = 256`.

Word segmentation dùng Underthesea:

```text
áo thun cổ tròn
→ áo_thun cổ_tròn
```

Query cũng cần đi qua `clean_text()` và `segment()` bằng cùng quy tắc trước khi đưa vào tokenizer:

```text
clean_text(query) → segment(text) → tokenizer
```

### 3.5. Chia train/validation/test bằng `build_splits.py`

`build_splits.py` gộp tất cả file `{source}_clean.jsonl`, sau đó kiểm tra:

- field bắt buộc: `id`, `source`, `category`, `text_seg`, `text_group`, `images`;
- ID trùng giữa các nguồn;
- ID không rỗng.

Dữ liệu được nhóm trước khi chia bằng:

- `text_group`: văn bản model input giống nhau;
- URL ảnh: cùng URL xuất hiện ở nhiều sản phẩm;
- perceptual hash: ảnh gần trùng hoặc trùng;
- union-find: hợp toàn bộ bản ghi liên quan thành một extended group.

Mỗi extended group được giữ nguyên trong một split. Không có thành phần nào của nhóm trúng nhiều split.

Thuật toán chia nhóm:

1. Tách theo stratum `(source, category)`.
2. Tính tỷ lệ mục tiêu mặc định: 80% train, 10% validation, 10% test.
3. Xếp các extended group từ lớn đến nhỏ.
4. Gán group sang split còn thiếu tỷ trọng của các stratum trong group.
5. Dùng seed `42` để rút ngẫu nhiên khi các split ngang nhau.

Đầu ra:

- `data/splits/train.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`
- `data/splits/split.json`, ánh xạ `product_id → train/val/test`

Sau khi chia, script báo tỷ lệ từng source/category và kiểm tra overlap theo text group, URL, perceptual hash và ID.

Lệnh chạy:

```powershell
python build_splits.py --in-dir data/processed --out-dir data/splits --seed 42
```

### 3.6. Quy tắc thống nhất

- Text dùng cùng giá trị `text_seg` đã tạo bởi `common.py`.
- Dataset, index và serving không được trộn quy tắc tiền xử lý cũ và mới.
- `text_group`, URL ảnh và phash phải được kiểm tra overlap trước huấn luyện.
- Không rebuild split sau khi train mà không đóng băng và ghi phiên bản snapshot.
- Nếu thay đổi `common.py`, phải làm sạch lại các file source, build lại split và build lại toàn bộ FAISS index.

### 3.7. Dataset

[data/dataset.py](/C:/Users/DELL/Downloads/vimfd-retrieval/data/dataset.py) trả:

```text
image [3, 224, 224]
input_ids [256]
attention_mask [256]
product_id
```

Transform train có crop, flip và color jitter nhẹ. Validation/test chỉ resize/crop và chuẩn hóa.

## 4. Huấn luyện

[src/training/train.py](/C:/Users/DELL/Downloads/vimfd-retrieval/src/training/train.py) dùng:

| Tham số | Giá trị |
|---|---:|
| Batch size | 64 |
| Epoch | 30 |
| Learning rate backbone | 1e-5 |
| Learning rate head | 1e-4 |
| Warmup | 2 epoch |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Embedding | 256 |
| `UNFREEZE_LAST_N` | 2 |

- Mỗi cặp ảnh-văn bản cùng sản phẩm là positive; cặp còn lại trong batch là negative.
- Loss: symmetric InfoNCE cho hai hướng.
- Optimizer: AdamW; scheduler: linear warmup rồi cosine decay.
- AMP và gradient scaling bật khi có CUDA.
- Checkpoint mới nhất và tốt nhất lưu thành `latest.pt` và `best.pt`.
- Checkpoint tốt nhất được chọn theo Recall@10 validation trung bình T→I và I→I.

Chạy:

```powershell
python src/training/train.py
```

## 5. FAISS và truy xuất

[src/indexing/build_index.py](/C:/Users/DELL/Downloads/vimfd-retrieval/src/indexing/build_index.py) tạo:

| Index | Nội dung | Mapping |
|---|---|---|
| `product.index` | Embedding văn bản | `id_mapping.json` |
| `product_image.index` | Embedding ảnh đại diện | `id_mapping_image.json` |

Luồng truy xuất:

```text
Text query → tokenize → PhoBERT → text projection → top-k image index
Image query → resize/normalize → CLIP Vision → image projection → top-k image index
```

Index phải được build lại sau khi đổi checkpoint, text preprocessing hoặc snapshot dữ liệu.

## 6. Đánh giá

[src/training/evaluate.py](/C:/Users/DELL/Downloads/vimfd-retrieval/src/training/evaluate.py) đánh giá trên tập test:

- Query T→I: mô tả của sản phẩm test.
- Query I→I: ảnh phụ của sản phẩm test, không trùng ảnh đại diện trong gallery.
- Gallery:
  - `full`: toàn bộ index.
  - `test-only`: chỉ sản phẩm test.
- Batch size evaluator: 64.
- Metric: Recall@1, Recall@5, Recall@10, NDCG@10, MRR@10.

Đánh giá chỉ tính query có sản phẩm đích trong gallery và lưu kết quả theo từng query để tính khoảng tin cậy.

```powershell
python src/training/evaluate.py
python src/training/evaluate_annotation.py
```

## 7. API và demo

[src/api.py](/C:/Users/DELL/Downloads/vimfd-retrieval/src/api.py) nạp model, tokenizer, index và metadata để phục vụ truy vấn. Hai luồng cần cùng kiểm tra pipeline:

- Text query: nạp text query/nhánh phù hợp và tìm trên image gallery.
- Image query: encode ảnh và tìm trên image gallery.

Tài nguyên startup phải khớp cùng một checkpoint, snapshot dữ liệu và mapping.

## 8. Trạng thái và công việc bắt buộc

### P0

- Chốt snapshot dữ liệu và chạy pipeline mới: làm sạch từng nguồn bằng `common.py`, gộp bằng `build_splits.py`.
- Xác nhận ảnh processed 224×224, không có ảnh thiếu/lỗi hoặc tensor rỗng.
- Kiểm tra split không overlap theo text, URL, perceptual hash hoặc ID.
- Dùng cùng `text_seg` cho dataset, index và serving; không dùng lại quy tắc `retrieval_text/text_seg` của pipeline cũ.
- Sửa/kiểm chứng `logit_scale` và train lại nếu checkpoint cũ dùng scale sai.
- Chạy đủ số epoch/batch đã chốt; lưu config, log và checkpoint.
- Xác nhận I→I có số query thực tế, không dùng kết quả `n_queries=1`.
- Build index từ checkpoint run cuối và chạy lại evaluator.
- Chỉ công bố baseline khi đã có lệnh chạy và artifact kết quả riêng.

### P1

- Hoàn thiện annotation 200 query và IAA; báo đúng số query thực sự có nhãn.
- Thêm MRR, khoảng tin cậy bootstrap, kiểm định paired test và phân tích lỗi.
- Xem xét reranker/Cross-Encoder và baseline bổ sung sau khi pipeline chính ổn định.
- Đo thời gian API median/P95 trên cùng phần cứng.

## 9. Lệnh chạy đầy đủ

```powershell
# Làm sạch từng nguồn
python common.py --src local --out-dir data/processed --root .
python common.py --src tiki --out-dir data/processed --root .
python common.py --src lazada --out-dir data/processed --root .

# Gộp và chia train/validation/test
python build_splits.py --in-dir data/processed --out-dir data/splits --seed 42

# Huấn luyện, build index và đánh giá
python src/training/train.py
python src/indexing/build_index.py
python src/training/evaluate.py
python src/training/evaluate_annotation.py
```

Thứ tự bắt buộc: **dữ liệu → huấn luyện → build index → đánh giá**. Khi thay preprocessing, checkpoint hoặc snapshot dữ liệu, phải build lại index trước khi đánh giá.

## 10. Kết luận đánh giá

Mô hình đã có pipeline end-to-end cho text và image retrieval. Trước khi báo cáo kết quả, repository cần dùng thống nhất một checkpoint, một snapshot dữ liệu, một quy tắc text, một batch size và một chế độ gallery. Không kết luận chất lượng từ metrics cũ, pilot hoặc run thiếu query I→I; số liệu hợp lệ phải đi kèm config, số query, checkpoint, index và log huấn luyện tương ứng.
