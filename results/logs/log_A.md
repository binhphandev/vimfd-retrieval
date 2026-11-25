
## V8 — chạy dual-encoder và ViMFR (2026-11-24 19:54) [SAMPLE 5 – không dùng]
- thiết bị: cuda, GPU Tesla T4, VRAM 15.6 GB
- torch 2.14.1+cu130; chỉ mục Test 1449/1449 sản phẩm (thiếu ảnh 0); fp16=True
- 5 truy vấn, top_k=50; thứ tự giữ nguyên sau rerank: 0/5
- độ trễ (bỏ 5 truy vấn đầu; gồm tách từ + mã hóa + FAISS + rerank): trung vị nan ms, P95 nan ms

## V8 — chạy dual-encoder và ViMFR (2026-11-24 19:56)
- thiết bị: cuda, GPU Tesla T4, VRAM 15.6 GB
- torch 2.14.1+cu130; chỉ mục Test 1449/1449 sản phẩm (thiếu ảnh 0); fp16=True
- 200 truy vấn, top_k=50; thứ tự giữ nguyên sau rerank: 0/200
- độ trễ (bỏ 5 truy vấn đầu; gồm tách từ + mã hóa + FAISS + rerank): trung vị 154.8 ms, P95 203.2 ms
