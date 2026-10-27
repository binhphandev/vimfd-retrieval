# Tóm tắt dự án ViMFD Retrieval

## 1. Tổng quan

Dự án này xây dựng một hệ thống truy xuất sản phẩm đa phương thức (multimodal retrieval) cho dữ liệu thương mại thời trang/marketplace, cho phép người dùng:

- nhập mô tả văn bản (text query)
- hoặc gửi ảnh sản phẩm (image query)
- tìm lại sản phẩm tương đồng trong kho dữ liệu bằng embedding của text và ảnh

Mô hình cốt lõi là một kiến trúc contrastive learning kết hợp:

- encoder ảnh: CLIP Vision Transformer
- encoder text: PhoBERT
- projection head: map hai embedding về cùng không gian 256 chiều
- loss: InfoNCE
- tìm kiếm tương đồng: FAISS (IndexFlatIP)

Project tập trung vào bài toán retrieval thông tin sản phẩm từ hình ảnh và mô tả văn bản, phục vụ tìm kiếm theo text-to-image và image-to-image.

---

## 2. Mục tiêu của dự án

- Xây dựng pipeline dữ liệu từ metadata và ảnh gốc
- Chuẩn hóa text và ảnh cho training
- Huấn luyện mô hình đa phương thức để học embedding chung giữa mô tả text và hình ảnh
- Chuyển đổi embedding thành index truy xuất hiệu quả bằng FAISS
- Đánh giá độ chính xác bằng Recall@K và NDCG@K

---

## 3. Kiến trúc tổng thể

Pipeline chính của dự án như sau:

1. Đọc metadata sản phẩm từ file JSON gốc
2. Kiểm tra, lọc và chuẩn hóa dữ liệu sản phẩm
3. Word-segment text bằng `underthesea`
4. Chia dữ liệu thành train/val/test theo stratified split
5. Resize và lưu ảnh ở kích thước 224x224
6. Huấn luyện model với cặp (text, image) theo contrastive loss
7. Encode toàn bộ sản phẩm thành embedding
8. Build FAISS index cho text và ảnh
9. Tìm kiếm và đánh giá trên tập test

---

## 4. Cấu trúc thư mục chính

- [config.py](./config.py): chứa hằng số toàn cục: đường dẫn, tham số model, training, checkpoint, FAISS, API
- [config.example.py](./config.example.py): file mẫu để copy sang config.py
- [requirements.txt](./requirements.txt): danh sách dependency cần thiết
- [data/](./data): dữ liệu và preprocessing
- [src/models/](./src/models): định nghĩa model, encoder, projection head và loss
- [src/training/](./src/training): training và evaluation
- [src/indexing/](./src/indexing): build FAISS index
- [checkpoints/](./checkpoints): checkpoint training
- [index/](./index): file index và mapping ID
- [results/](./results): log và metrics đánh giá

---

## 5. Mô tả từng module

### 5.1 [config.py](./config.py)

Đây là file trung tâm của dự án. Nó định nghĩa:

- đường dẫn gốc dữ liệu
- model text và hình ảnh (
  - `PHOBERT_MODEL = "vinai/phobert-base"`
  - `CLIP_MODEL = "openai/clip-vit-base-patch32"`
)
- kích thước ảnh (`IMAGE_SIZE = 224`)
- mean/std chuẩn hóa ảnh theo CLIP
- field dữ liệu: `name`, `description`, `category`, `tags`, `images`
- kích thước embedding đầu ra (`EMBED_DIM = 256`)
- tham số training: batch size, learning rate, warmup, epochs
- checkpoint paths
- FAISS index paths
- API host/port

Nói cách khác, hầu hết các tham số quan trọng đều được quản lý tập trung ở đây để không hard-code rải rác.

### 5.2 [data/data_preprocess.py](./data/data_preprocess.py)

Đây là module tiền xử lý dữ liệu chính.

Nhiệm vụ:

- load metadata JSON gốc
- validate dữ liệu đầu vào
- bỏ qua sản phẩm thiếu `id`, `category`, `shop`, hoặc `images`
- tách text theo field cần thiết
- thực hiện word segmentation bằng `underthesea`
- chia dữ liệu thành train / val / test theo stratify trên shop + category
- lựa chọn ảnh chính (`is_featured` hoặc `position` thấp nhất) để lưu vào split
- resize ảnh về đúng kích thước chuẩn để dùng cho encoder

Dùng kỹ thuật `train_test_split` để chia dữ liệu cân bằng theo strata, tránh tình trạng dữ liệu mất cân bằng giữa các danh mục.

### 5.3 [data/dataset.py](./data/dataset.py)

Module này định nghĩa `ViMFDDataset` và `build_dataloaders`.

Nó làm ba việc chính:

- đọc file JSON của từng split (`train.json`, `val.json`, `test.json`)
- chọn ảnh chính của sản phẩm
- tạo tensor cho ảnh và text (token IDs + attention mask)

Các output của dataset gồm:

- `image`: tensor ảnh 3x224x224
- `input_ids`: token ids cho PhoBERT
- `attention_mask`: mask cho token
- `product_id`: id sản phẩm

Nó cũng áp dụng data augmentation nhẹ cho phần train như:

- resize/crop
- random horizontal flip
- color jitter

### 5.4 [src/models/encoders.py](./src/models/encoders.py)

Định nghĩa hai encoder chính:

- `VisualEncoder`: dùng `CLIPVisionModel` từ Hugging Face
- `TextEncoder`: dùng `AutoModel` với mô hình PhoBERT

Điểm đặc biệt là cả hai encoder đều được:

- freeze toàn bộ backbone ban đầu
- unfreeze các block cuối (`UNFREEZE_LAST_N`) để fine-tune cho phù hợp với dữ liệu Việt Nam

`forward` của từng encoder trả về hidden state của token `[CLS]`, sau đó được đưa vào projection head.

### 5.5 [src/models/projection.py](./src/models/projection.py)

Projection head biến đầu vào 768 chiều thành embedding 256 chiều và L2-normalize.

Cấu trúc chính:

- Linear(768 -> 256)
- LayerNorm
- GELU
- Dropout
- Linear(256 -> 256)
- normalize theo chiều cuối

Kết quả là hai embedding ảnh và text cùng không gian 256 chiều để so sánh cosine similarity.

### 5.6 [src/models/loss.py](./src/models/loss.py)

Định nghĩa `InfoNCELoss`.

Loss này tính symmetric cross-entropy trên ma trận tương đồng giữa batch ảnh và batch text:

- ảnh i so với text i là positive pair
- phần còn lại trong batch là negative
- loss được tính theo cả hướng image-to-text và text-to-image

Điều này giúp mô hình học được việc ánh xạ text và image về cùng không gian embedding.

### 5.7 [src/models/model.py](./src/models/model.py)

Đây là class `ViMFRModel` tích hợp cả VisualEncoder, TextEncoder và ProjectionHead.

Phương thức chính:

- `encode_image(pixel_values)`
- `encode_text(input_ids, attention_mask)`
- `forward(pixel_values, input_ids, attention_mask)`

Mô hình còn có `logit_scale` là tham số learnable temperature để điều chỉnh độ lớn của logits khi tính similarity.

### 5.8 [src/training/train.py](./src/training/train.py)

Đây là file huấn luyện chính.

Chức năng:

- xây dựng `AltImageDataset` để lấy ảnh phụ của cùng sản phẩm làm query image-to-image
- định nghĩa optimizer và scheduler
- Validate cả hai hướng:
  - Text → Image
  - Image → Image
- đo metric Recall@1, Recall@5, Recall@10
- lưu checkpoint tốt nhất (`best.pt`) và checkpoint mới nhất (`latest.pt`)
- log dữ liệu epoch theo CSV

File này đánh giá mô hình qua cả text retrieval và image retrieval, rất phù hợp cho việc tối ưu độ mạnh của model.

### 5.9 [src/indexing/build_index.py](./src/indexing/build_index.py)

Đây là module tạo FAISS index cho hệ thống retrieval.

Nhiệm vụ:

- load tất cả product từ `train.json`, `val.json`, `test.json`
- encode text cho toàn bộ sản phẩm
- build `product.index` và `id_mapping.json`
- encode ảnh đại diện cho toàn bộ sản phẩm
- build `product_image.index` và `id_mapping_image.json`
- chạy smoke test để kiểm tra index hoạt động

FAISS được dùng để làm hệ thống tìm kiếm hàng triệu vector nhanh, thay vì brute-force khi query theo text hoặc ảnh.

### 5.10 [src/training/evaluate.py](./src/training/evaluate.py)

File này dùng checkpoint và FAISS index để đánh giá hiệu năng thực tế.

Phân tích gồm:

- Text → Image: dùng text của test set làm query
- Image → Image: dùng ảnh phụ của sản phẩm test làm query
- so sánh trên gallery toàn bộ dữ liệu và gallery chỉ test set

Metric được tính:

- Recall@1
- Recall@5
- Recall@10
- NDCG@10

Kết quả được ghi vào:

- [results/metrics.txt](./results/metrics.txt)
- [results/metrics.json](./results/metrics.json)

---

## 6. Luồng dữ liệu của dự án

### Bước 1: Tiền xử lý

- input: JSON metadata sản phẩm từ `data/raw` hoặc dữ liệu đã có
- output: `data/processed/train.json`, `val.json`, `test.json`

### Bước 2: Tạo dataset cho training

- mỗi record chứa:
  - ảnh đại diện
  - text tích hợp từ `name`, `description`, `category`, `tags`
  - `product_id`

### Bước 3: Training

- mô hình học tương quan giữa ảnh và text
- mỗi batch có các cặp positive giữa `image_i` và `text_i`
- loss `InfoNCE` giúp hàng xóm gần nhau hơn, các sản phẩm khác xa nhau hơn

### Bước 4: Build index

- encode toàn bộ sản phẩm sang embedding 256D
- lưu index riêng cho text và image
- mapping mỗi index row với `product_id`

### Bước 5: Retrieval

- query text: encode query text -> search trong text index
- query image: encode ảnh -> search trong image index
- trả về top-k sản phẩm tương đồng

---

## 7. Kết quả hiện tại

Kết quả đánh giá hiện có trong [results/metrics.txt](./results/metrics.txt):

- Text → Image
  - Recall@1: 0.0197
  - Recall@5: 0.0849
  - Recall@10: 0.1580
  - NDCG@10: 0.0751

- Image → Image
  - Recall@1: 0.0220
  - Recall@5: 0.0975
  - Recall@10: 0.1769
  - NDCG@10: 0.0842

Kết quả này cho thấy hệ thống đang trong giai đoạn phát triển và cần cải tiến thêm về chất lượng retrieval, ví dụ:

- tăng số epoch
- tinh chỉnh optimizer / scheduler
- thử backbone mạnh hơn
- xử lý dữ liệu thiếu ảnh và text thiếu thông tin
- xét lại chiến lược split và negative sampling

---

## 8. Cách chạy dự án

### Khởi tạo môi trường

```bash
pip install -r requirements.txt
```

### Tạo file cấu hình

```bash
cp config.example.py config.py
```

### Bước 1: tiền xử lý dữ liệu

```bash
python data/data_preprocess.py
```

### Bước 2: train mô hình

```bash
python src/training/train.py
```

### Bước 3: xây index FAISS

```bash
python src/indexing/build_index.py
```

### Bước 4: đánh giá

```bash
python src/training/evaluate.py
```

---

## 9. Điểm mạnh của dự án

- pipeline dữ liệu rõ ràng và có cấu trúc theo từng bước
- huấn luyện theo kiến trúc contrastive learning hiện đại
- hỗ trợ cả text-to-image và image-to-image retrieval
- tận dụng FAISS để truy xuất nhanh và hiệu quả
- dễ thay đổi model, checkpoint, batch size, learning rate qua config

---

## 10. Hạn chế / hướng phát triển tiếp theo

- dữ liệu và metric hiện tại chưa đạt hiệu suất cao; cần tinh chỉnh thêm
- model hiện có thể bị ảnh hưởng bởi dữ liệu thiếu hoặc không đồng nhất
- cần xử lý tính đa dạng của text và ảnh trong sản phẩm thực tế
- có thể mở rộng thêm:
  - reranking sau retrieval
  - query expansion
  - hard negative mining
  - tối ưu hóa index theo chiến lược approximate search
  - xây API phục vụ trực tiếp cho web/mobile

---

## 11. Kết luận

Dự án ViMFD Retrieval là một hệ thống retrieval sản phẩm đa phương thức, kết hợp CLIP + PhoBERT + FAISS để tìm sản phẩm qua văn bản hoặc hình ảnh. Dự án có kiến trúc rõ ràng, pipeline hoàn chỉnh từ preprocessing → training → indexing → evaluation, và đang là nền tảng tốt để phát triển thêm các tính năng tìm kiếm thương mại nâng cao.
