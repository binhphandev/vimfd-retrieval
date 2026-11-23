
## V1 — chọn hạt giống
- seed=42; DESC_LIMIT=300; DOMESTIC_SHARE=0.35; RESERVE_RATE=0.2
- ứng viên: 952/1339 (bad_products 312)
- ô chỉ tiêu mỗi người: áo/n1=11, áo/n2=11, áo/n3=18, quần/n1=9, quần/n2=9, quần/n3=12, phụ kiện/n1=15, phụ kiện/n2=15
- A: 100 slot, nội địa 40, dự phòng 26
- B: 100 slot, nội địa 40, dự phòng 26

## V9 — pooling
- K: dual_encoder 10, vimfr 10, baseline 2 (mỗi baseline); thêm hạt giống; N = 4644 (ngưỡng 3800: VƯỢT 3800 ngay cả khi K=2 (giữ K=2, chấp nhận kéo dài buổi gán nhãn))
- N theo K baseline: K=5: 6104, K=3: 5106, K=2: 4644
- cỡ pool mỗi truy vấn: min 15, max 28, TB 23.2; cặp thuộc overlap: 707
- hạt giống đã nằm trong top-K của hệ thống: 122/200; chỉ có nhờ thêm: 78/200
- file annotation: annotation_A_01: 30 truy vấn/707 cặp; annotation_A_02: 22 truy vấn/511 cặp; annotation_A_03: 21 truy vấn/487 cặp; annotation_A_04: 21 truy vấn/494 cặp; annotation_A_05: 21 truy vấn/484 cặp; annotation_B_01: 30 truy vấn/707 cặp; annotation_B_02: 22 truy vấn/518 cặp; annotation_B_03: 21 truy vấn/487 cặp; annotation_B_04: 21 truy vấn/477 cặp; annotation_B_05: 21 truy vấn/479 cặp
- ước tính công gán nhãn: 4.5 giờ mỗi người (hai người); 8.9 giờ nếu một người gán cả hai phần
- Hệ quả cần ghi vào báo cáo: sản phẩm hạng 6–10 của baseline không nằm trong pool của hệ thống nào bị coi là không liên quan, nên chỉ số baseline có thể bị đánh giá thấp (có lợi cho ViMFR).
- file *_01 có 30 truy vấn (> 25 truy vấn/file ở mục 8.4): ghi mục 12 hoặc dùng --split-overlap

## V10 — kappa
- vòng 1: κ tuyến tính 0.743

## V10 — ghép nhãn
- 4644 cặp, 200 truy vấn; nhãn {0: 3128, 1: 944, 2: 572}; R_q=0: 13
