# ViMFD Retrieval — Tổng quan dự án

> Tài liệu này mô tả snapshot repository được kiểm tra ngày 23/11/2026. Số liệu được dẫn từ artifact có trong repo; chúng không mặc nhiên chứng minh rằng toàn bộ pipeline có thể tái lập trên máy khác.

## 1. Tóm tắt

ViMFD Retrieval là nguyên mẫu tìm kiếm sản phẩm thời trang bằng tiếng Việt, tập trung vào quần áo và phụ kiện. Giao diện/API hỗ trợ truy vấn bằng văn bản, ảnh hoặc kết hợp cả hai. Hệ thống mã hóa truy vấn và ảnh sản phẩm thành vector, tìm ứng viên theo cosine similarity, rồi áp dụng bộ lọc metadata và phân trang.

Các thành phần chính:

- Dual-encoder gồm CLIP ViT-B/32 cho ảnh và PhoBERT-base cho văn bản; hai projection head đưa vector về 256 chiều.
- FAISS `IndexFlatIP` cho tìm kiếm vector chính xác trên vector đã L2-normalize.
- Backend FastAPI, catalog sản phẩm và giao diện HTML/CSS/JavaScript.
- Bộ đánh giá dual-encoder, benchmark truy vấn có gán nhãn và checkpoint cross-encoder thử nghiệm.

Tên model retrieval trong mã là `ViMFRModel`; tên dự án/repository là **ViMFD Retrieval**. Cần thống nhất cách viết này trong báo cáo và hình minh họa.

## 2. Phạm vi và luồng hệ thống

```text
Truy vấn văn bản ─ PhoBERT ─ projection ─┐
                                         ├─ điểm cosine / trộn điểm ─ lọc ─ kết quả
Truy vấn ảnh ───── CLIP Vision ─ projection┘

Ảnh sản phẩm ─ CLIP Vision ─ projection ─ FAISS image index
Văn bản sản phẩm ─ PhoBERT ─ projection ─ FAISS text index (được build; API hiện không dùng)
```

Khi có cả văn bản và ảnh, `SearchEngine.search` chuẩn hóa riêng hai mảng điểm rồi trộn theo trọng số `w`: `(1-w) * text_score + w * image_score`. Tìm kiếm văn bản hoặc ảnh đơn lẻ dùng điểm của nhánh tương ứng. Bộ lọc giá, danh mục/loại và màu được thực hiện trong engine. Sắp xếp theo giá được áp dụng sau khi chọn tập ứng viên theo độ liên quan.

API chính: `GET /api/health`, `POST /api/search`, `GET /api/products/{pid}`, `GET /api/products/{pid}/similar`. Similarity endpoint dùng embedding ảnh. API phục vụ frontend gốc nếu thư mục đó tồn tại.

## 3. Cấu trúc repository và thành phần

| Đường dẫn | Vai trò đã xác nhận |
|---|---|
| `config.py` | Đường dẫn và tham số model, huấn luyện, index, API |
| `data/splits/` | JSONL train/val/test |
| `data/processed/` | JSONL đã làm sạch và cây ảnh xử lý |
| `data/dataset.py`, `data/data_validation.py` | Dataset/transform và kiểm tra chất lượng dữ liệu |
| `src/models/` | Encoder, projection, model và loss |
| `src/training/` | Huấn luyện và đánh giá dual-encoder |
| `src/indexing/build_index.py` | Build index text và image cùng mapping/metadata |
| `src/serving/`, `src/api/` | Catalog, tìm kiếm, endpoint FastAPI |
| `frontend/` | Giao diện web |
| `data/annotation/` | Pipeline benchmark truy vấn, pooling và gán nhãn |
| `checkpoints/` | Có checkpoint dual-encoder và cross-encoder trong snapshot |
| `index/`, `results/` | FAISS artifacts, metric, log và run files |

Các script tái tạo nguồn làm sạch/chia split được nhắc tới ở tài liệu cũ (chẳng hạn `common.py`, `build_splits.py`) không có trong source snapshot. Dữ liệu split có sẵn nhưng quy trình tạo lại từ dữ liệu thô chưa được đóng gói đầy đủ.

## 4. Dữ liệu và chất lượng

### 4.1 Quy mô

Các JSONL split và báo cáo kiểm định ghi nhận:

| Split | Sản phẩm | Tỷ lệ xấp xỉ |
|---|---:|---:|
| Train | 10.690 | 80% |
| Validation | 1.339 | 10% |
| Test | 1.339 | 10% |
| **Tổng** | **13.368** | **100%** |

Theo báo cáo phân bố, nguồn gồm local 3.823, Tiki 1.982 và Lazada 7.563 sản phẩm; danh mục gồm áo 4.746, quần 2.817 và phụ kiện 5.805. Đây là số đếm sản phẩm trong snapshot, không phải số lượng ảnh. Báo cáo kiểm định đếm 41.834 ảnh qua ba split (33.071 train, 4.247 val, 4.516 test).

Record chứa ID, nguồn/shop/URL, danh mục, tên/mô tả, giá, màu/kích cỡ/chất liệu và danh sách ảnh. Mô hình dùng `text_seg` làm đầu vào văn bản.

### 4.2 Tiền xử lý theo mã nguồn

- Ảnh đầu vào mô hình: resize/crop về 224×224, tensor hóa, chuẩn hóa theo mean/std của CLIP ViT-B/32. Validation/test không dùng augmentation; training có lật ngang ngẫu nhiên và ColorJitter nhẹ.
- Ảnh đại diện trong catalog/index: ưu tiên `is_featured`; nếu không có, chọn `position` nhỏ nhất trong các ảnh có `local_path` hợp lệ.
- Văn bản: `text_seg`, tối đa 256 token PhoBERT; truy vấn API được word-segment bằng `underthesea`.
- Split có tỷ lệ gần 80/10/10 và phân bố source×category khá cân bằng theo báo cáo kiểm định.

### 4.3 Cảnh báo chất lượng có ảnh hưởng đến kết luận

`data/validation/validation_report.json` ghi nhiều kiểm tra PASS, nhưng cũng có các cảnh báo/FAIL cần đưa vào mọi báo cáo kết quả:

- Ảnh gần trùng (perceptual hash distance ≤4) giữa validation và train: **27,48%**; giữa test và train: **27,11%**. Đây là rủi ro rò rỉ nội dung thị giác và có thể làm Recall@k cao hơn khả năng tổng quát hóa thực tế. Báo cáo hiện tại không chứng minh rằng ảnh gần trùng đã được loại khỏi split.
- Ảnh trùng trong cùng split: **20,84% validation**, **24,65% test**. Điều này làm truy vấn ảnh có thể mơ hồ giữa nhiều ID.
- Văn bản bị cắt ở giới hạn 256 token: **21,7% train**, **23,8% validation**, **21,4% test**.
- Giá thiếu hoặc không dương: **100% ở cả ba split**, theo kiểm tra validation. Vì vậy lọc/sắp xếp giá có thể không hữu ích với dữ liệu hiện tại; giao diện không nên được mô tả như thể giá đã được kiểm chứng đầy đủ.
- Cặp sản phẩm gần giống trung bình **1,69 cặp mỗi batch 64** trong train, có thể tạo false negatives cho InfoNCE.

Vì các vấn đề này, metric trên split ngẫu nhiên không nên được diễn giải là kết quả đánh giá trên tập ảnh/sản phẩm độc lập hoàn toàn. Ưu tiên tiếp theo là tạo split theo cụm ảnh/mẫu sản phẩm (dedup trước khi chia), sau đó build lại index và chạy lại đánh giá.

## 5. Kiến trúc dual-encoder

| Nhánh | Backbone | Vector backbone | Đầu ra |
|---|---|---:|---:|
| Ảnh | `openai/clip-vit-base-patch32` | 768 | 256 |
| Văn bản | `vinai/phobert-base` | 768 | 256 |

Mỗi projection head: `Linear(768,256) → LayerNorm → GELU → Dropout(0.1) → Linear(256,256)`, sau đó L2-normalize. Cấu hình chỉ mở hai block cuối của backbone cùng các lớp chuẩn hóa liên quan; phần còn lại được đóng băng. Model có `logit_scale` khởi tạo `log(1/0.07)` và giới hạn scale tối đa 100.

Loss huấn luyện là Symmetric InfoNCE: cross-entropy hai chiều image→text và text→image, lấy trung bình; các cặp cùng vị trí trong batch là positive, phần còn lại được xem là negative.

## 6. Huấn luyện và checkpoint

Cấu hình trong `config.py`: batch 64 (CPU fallback 4), 30 epoch, learning rate backbone `1e-5`, head `1e-4`, weight decay `0.01`, warmup 2 epoch rồi cosine decay, gradient clipping 1.0, 2 workers. AMP FP16 bật khi CUDA khả dụng. Optimizer là AdamW.

Validation trong training tính loss cùng Recall@1/5/10 cho text→image và image→image. Script lưu checkpoint tốt nhất theo điểm tổng hợp Recall@10 hai hướng; `latest.pt` là trạng thái epoch mới nhất. Cấu hình mặc định trỏ tới `checkpoints/best_18112026.pt` và `checkpoints/latest.pt`.

## 7. Chỉ mục và khả năng tái lập

`src/indexing/build_index.py` tạo index văn bản và ảnh (`IndexFlatIP`) cùng mapping ID. Vì vector được L2-normalize, inner product bằng cosine similarity. Metadata image index trong snapshot ghi:

- 13.368 sản phẩm và 13.368 vector được index.
- Checkpoint khai báo: `checkpoints/best_18112026.pt`.
- Fingerprint checkpoint: `870967d596f8dbd6e138df5ca13443cf`.
- Thời điểm build ghi trong metadata: **2026-11-19 17:23:00**.
- `split_dir` và `image_dir` trong metadata vẫn trỏ đến `C:\Users\DELL\Downloads\vimfd-retrieval\...`, tức đường dẫn của máy build trước.

API so sánh fingerprint checkpoint và kiểm tra ID index với catalog để phát hiện index cũ. Index và checkpoint hiện có trong repo, nhưng metadata đường dẫn cũ làm giảm khả năng tái lập nguyên trạng. Khi đổi dữ liệu, checkpoint hoặc transform, cần build lại index trong môi trường hiện tại và xác nhận fingerprint/ID.

## 8. Kết quả dual-encoder

Nguồn: `results/metrics.json`, evaluator `src/training/evaluate.py`. Gallery full gồm 13.368 sản phẩm; gallery test-only gồm 1.339 sản phẩm. Mỗi query văn bản là một mô tả sản phẩm test (1.339 query); image→image có 751 query ảnh phụ hợp lệ theo evaluator.

| Gallery | Hướng | N query | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Full (13.368) | Text→Image | 1.339 | 0,0157 | 0,0919 | 0,1538 | 0,0492 | 0,0734 |
| Full (13.368) | Image→Image | 751 | 0,4181 | 0,6245 | 0,6951 | 0,5026 | 0,5486 |
| Test-only (1.339) | Text→Image | 1.339 | 0,1621 | 0,4563 | 0,6169 | 0,2889 | 0,3663 |
| Test-only (1.339) | Image→Image | 751 | 0,5486 | 0,7803 | 0,8429 | 0,6460 | 0,6936 |

Full gallery là kịch bản tìm kiếm rộng hơn, còn test-only là gallery kín gồm chính các sản phẩm truy vấn; hai kết quả không nên so sánh như cùng một bài toán. Full-gallery text retrieval còn yếu (R@10 0,1538). Các chỉ số cũng cần được đọc cùng cảnh báo near-duplicate ở mục 4.3.

## 9. Benchmark truy vấn gán nhãn và cross-encoder

### 9.1 Benchmark sáu hệ thống

Có 200 truy vấn biên soạn/annotate; 13 truy vấn không có sản phẩm liên quan (nhãn relevance ≥1) bị loại khỏi trung bình, nên metric dưới đây dùng **187 truy vấn hợp lệ**. So sánh gồm TF-IDF, BM25, CLIP Vi→En, Bi-Encoder text, Dual-Encoder và ViMFR.

| Hệ thống | Recall@5 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0,2317 | 0,3139 | 0,3321 | 0,5942 |
| BM25 | 0,2907 | 0,3556 | 0,4076 | 0,7557 |
| CLIP Vi→En | 0,2233 | 0,2943 | 0,3106 | 0,6139 |
| Bi-Encoder text | 0,2318 | 0,2932 | 0,3176 | 0,6073 |
| Dual-Encoder | 0,1572 | 0,2896 | 0,2485 | 0,4309 |
| ViMFR | **0,2985** | **0,4619** | **0,4918** | 0,6830 |

Trong các số tổng hợp này, ViMFR có Recall@10 và NDCG@10 cao nhất; BM25 có MRR@10 cao nhất. Bootstrap 95% CI của ViMFR: Recall@10 **[0,4262; 0,4973]**, NDCG@10 **[0,4496; 0,5332]**. Đây là CI từng hệ thống theo truy vấn, không tự nó chứng minh khác biệt có ý nghĩa so với mọi baseline. Không diễn giải thứ hạng tổng hợp như ưu thế thống kê nếu chưa xem paired comparison.

Giới hạn quan trọng: qrels được tạo từ pool của hệ thống (run) và hạt giống. Sản phẩm ngoài pool chưa được đánh giá có thể bị coi là không liên quan khi tính metric. Quy trình V9 trong mã ghi rõ baseline top-6..10 có thể không nằm trong pool; điều này có thể làm metric baseline thấp giả tạo. Kết quả do đó là đánh giá trên pool hiện có, không phải judgment exhaustively pooled trên toàn corpus.

### 9.2 Gán nhãn và độ nhất quán

Tập truy vấn chia theo số thuộc tính: 70 truy vấn một thuộc tính, 70 truy vấn hai thuộc tính, 60 truy vấn từ ba thuộc tính; theo nhóm sản phẩm: áo 80, quần 60, phụ kiện 60. Kiểm tra query ghi nhận cảnh báo phân bố thuộc tính còn lệch nhẹ so với mục tiêu 35/35/30.

Hai annotator gán nhãn relevance 0/1/2 độc lập cho các cặp trong pool. Vòng 1 trên 707 cặp overlap/30 truy vấn đạt Cohen’s κ trọng số tuyến tính **0,743**, κ bậc hai **0,768**, đồng thuận chính xác **0,857**. Sau hợp nhất, artifact ghi 3.128 cặp nhãn 0, 944 nhãn 1, 572 nhãn 2 và 13 query không có relevance. Tổng nhãn là 4.644 cặp; không đồng nhất số cặp được gán nhãn với số query dùng tính metric.

### 9.3 Cross-encoder

Artifact V6 huấn luyện PhoBERT-base ghép cặp `(query, document)`, dùng CLS → Linear(768→1), BCEWithLogitsLoss. Log ghi 25.027 cặp train, 4.030 cặp validation; batch 32, tối đa 10 epoch, patience 3, LR PhoBERT `2e-5`, LR linear `1e-4`, weight decay `0.01`, warmup 10% rồi cosine, FP16 trên Tesla T4. Checkpoint nằm trong `checkpoints/cross_encoder_best/`.

Đây là thành phần thí nghiệm riêng. `src/serving/engine.py` hiện không nạp checkpoint hoặc gọi cross-encoder; API trả kết quả dual-encoder và bộ lọc/sắp xếp. Vì vậy không mô tả cross-encoder như reranker đang hoạt động trong sản phẩm. Chỉ nêu số validation của nó khi đối chiếu trực tiếp log đầy đủ và xác nhận tiêu chí/epoch chọn checkpoint.

## 10. Độ trễ và vận hành

`results/logs/latency.csv` có 200 truy vấn đo trên CPU; số tổng hợp ghi trong báo cáo log là median **17,7153 giây**, P95 **28,2738 giây**. Đo bao gồm encode truy vấn, FAISS và rerank theo benchmark latency đó; phần rerank chiếm phần lớn thời gian. Con số này không đại diện cho API serving hiện tại một cách trực tiếp, vì engine API trong snapshot chưa tích hợp cross-encoder. Cần đo lại endpoint end-to-end trên cấu hình triển khai xác định trước khi công bố SLA.

Khởi chạy theo mã nguồn, từ thư mục gốc:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API dùng CUDA nếu khả dụng, nếu không dùng CPU. Đáp ứng `/api/health` trả số sản phẩm từ image index và thiết bị đang dùng. Upload ảnh giới hạn JPG/PNG/WEBP, tối đa 5 MB.

## 11. Các điểm cần hoàn thiện trước khi công bố

1. Dedupe theo perceptual hash/cụm sản phẩm trước khi chia lại train/validation/test; build lại toàn bộ index và báo cáo metric.
2. Xử lý trường giá rỗng/không hợp lệ hoặc tắt/ghi chú rõ tính năng lọc giá.
3. Bổ sung manifest tái lập: nguồn, hash dữ liệu, thời điểm crawl, split seed, phiên bản preprocessing, checkpoint hash và lệnh build.
4. Khôi phục hoặc đóng gói script làm sạch/chia split để tái tạo JSONL từ dữ liệu nguồn.
5. Đo API latency thực tế sau khi chốt có/không có reranker; nếu tích hợp cross-encoder, giới hạn candidate count và benchmark lại.
6. Báo cáo paired bootstrap CI cho chênh lệch ViMFR với baseline trên cùng query; công bố rõ pool bias và 187/200 query hợp lệ.
7. Bổ sung kiểm thử tự động cho catalog, fingerprint/index, API, evaluator và các trường hợp metadata thiếu.

## 12. Lệnh thường dùng

```powershell
python data/data_validation.py
python src/training/train.py
python -m src.indexing.build_index
python src/training/evaluate.py
python data/annotation/v11_metrics.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Các lệnh trên mô tả entry point trong repository; chạy thành công vẫn phụ thuộc dependencies, đường dẫn dữ liệu/ảnh và checkpoint phù hợp trên máy. Đặc biệt, sau thay đổi split hoặc checkpoint phải build lại index trước khi đánh giá hay chạy API.

## 13. Kết luận

Repository hiện có một nguyên mẫu retrieval đa phương thức hoàn chỉnh ở mức nghiên cứu: dual-encoder 256 chiều, FAISS, API/UI, artifact đánh giá trên split và benchmark gán nhãn. Kết quả benchmark truy vấn cho thấy ViMFR dẫn đầu Recall@10 và NDCG@10 trong sáu hệ thống được so sánh, trong khi BM25 dẫn đầu MRR@10. Tuy nhiên, gần 27% ảnh validation/test có ảnh gần trùng với train, benchmark dựa trên pooling hạn chế, giá thiếu toàn bộ và latency CPU benchmark bị chi phối bởi rerank chưa nằm trong API. Các điểm này cần được xử lý hoặc nêu cạnh kết quả trước khi xem metric là bằng chứng tổng quát hóa hay hiệu năng production.
