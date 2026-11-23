
## V3 — thuộc tính (12029 sản phẩm Train+Val)
- Có MAU: 2062 (17.1%)
- Có CHAT_LIEU: 5554 (46.2%)
- Có KIEU_DANG: 2382 (19.8%)
- Có PHONG_CACH: 3292 (27.4%)
- Dùng được để sinh truy vấn giả (có ≥1 thuộc tính ngoài loại): 8265 (68.7%)
- Số giá trị MAU khác nhau: 18
- Số giá trị LOAI khác nhau: 62
- Số cặp đồng nghĩa: 26/30 (chờ B duyệt)
- USE_NAME_REFINEMENT=True, SKIP_ATTR_IN_LOAI=True
- Top MAU: đen (750), trắng (254), xám (221), xanh navy (219), xanh dương (146), nâu (137), be (61), đỏ (57), vàng (44), bạc (41), xanh rêu (32), xanh lá (27)
- Top KIEU_DANG: regular fit (912), slim fit (567), oversize (499), straight fit (190), relaxed fit (63), wide leg (55), tapered (45), baggy (18), loose fit (17), skinny (9), comfort fit (7)
- Top PHONG_CACH: thể thao (900), vintage (608), casual (567), công sở (507), basic (468), minimalist (390), streetwear (226), smart casual (213), workwear (16), preppy (8)
- Top CHAT_LIEU: cotton (1879), polyester (1868), leather (861), spandex (848), denim (440), bamboo (348), viscose (310), fleece (237), nylon (229), linen (175), silk (169), wool (148)

## V4 — truy vấn giả
- Tham số: seed=42, MAX_DUP=3, MAX_TRIES=5, P_SYNONYM=0.3, P_SHUFFLE=0.5, n_attrs~{1: 0.3, 2: 0.4, 3: 0.3}
- train: 5006/7348 sản phẩm dùng được có truy vấn (68.1%); bỏ vì trùng: 2342; chữ truy vấn khác nhau: 3150
  n_attrs: n=1: 2511 (50%), n=2: 2116 (42%), n=3: 379 (8%)
  số truy vấn cùng chữ (1/2/3 lần): 2014/416/720
  mẫu câu: generic=2825, chat=1288, mau=607, kd_pc=205, pc_mau=81
  thay đồng nghĩa: 1602 giá trị
  5 truy vấn lặp nhiều nhất: áo sơ mi dáng tiêu chuẩn (3); áo sơ mi xanh navy (3); áo sơ mi cotton minimalist (3); áo sơ mi thanh lịch (3); áo sơ mi minimalist regular fit (3)
- val: 806/917 sản phẩm dùng được có truy vấn (87.9%); bỏ vì trùng: 111; chữ truy vấn khác nhau: 594
  n_attrs: n=1: 569 (71%), n=2: 190 (24%), n=3: 47 (6%)
  số truy vấn cùng chữ (1/2/3 lần): 456/64/74
  mẫu câu: generic=498, chat=212, mau=65, kd_pc=19, pc_mau=12
  thay đồng nghĩa: 196 giá trị
  5 truy vấn lặp nhiều nhất: áo sơ mi regular fit (3); áo sơ mi dáng tiêu chuẩn (3); áo sơ mi dáng ôm (3); tìm áo sơ mi polyester (3); áo sơ mi polyester (3)

## V5 — khai thác âm khó (DRY RUN – không dùng)
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=13s
- train: 5003 dương, 20002 âm (1:4.00); bỏ vì 0 âm: 3; thiếu ảnh: 0; TB bị loại/truy vấn: 8.1/50; âm cùng loại: 49%; âm cùng loại và khớp n−1 thuộc tính: 40%
- val: 806 dương, 3224 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 0; TB bị loại/truy vấn: 7.2/50; âm cùng loại: 35%; âm cùng loại và khớp n−1 thuộc tính: 28%
- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log

## V5 — khai thác âm khó
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=412s
- train: 5006 dương, 20024 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 7511; TB bị loại/truy vấn: 3.2/50; âm cùng loại: 35%; âm cùng loại và khớp n−1 thuộc tính: 23%
- val: 806 dương, 3224 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 955; TB bị loại/truy vấn: 3.1/50; âm cùng loại: 21%; âm cùng loại và khớp n−1 thuộc tính: 16%
- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log

## V5 — khai thác âm khó
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=111s
- train: 5006 dương, 20024 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 7511; TB bị loại/truy vấn: 3.5/50; âm cùng loại: 35%; âm cùng loại và khớp n−1 thuộc tính: 23%
- val: 806 dương, 3224 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 955; TB bị loại/truy vấn: 3.5/50; âm cùng loại: 21%; âm cùng loại và khớp n−1 thuộc tính: 17%
- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log

## V5 — khai thác âm khó
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=977s
- train: 5006 dương, 20021 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 0; TB bị loại/truy vấn: 4.7/50; âm cùng loại: 45%; âm cùng loại và khớp n−1 thuộc tính: 31%
- val: 806 dương, 3224 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 0; TB bị loại/truy vấn: 4.1/50; âm cùng loại: 24%; âm cùng loại và khớp n−1 thuộc tính: 19%
"""- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log
=== Kiểm tra bắt buộc: 20 cặp âm ngẫu nhiên (đọc bằng mắt) ===
 1. [OK ] q='giày dép đen' attrs={"MAU": "đen"}
      âm: Da Thật Chính Hãng Da Nam Túi Đeo Vai Kinh Doanh Du Lịch Cặp Da Bò Túi | khớp 1/1 thuộc tính | cùng loại: False
 2. [OK ] q='tìm áo sơ mi polyester slim fit' attrs={"KIEU_DANG": "slim fit", "CHAT_LIEU": "polyester"}
      âm: Áo khoác blazer vest mùa đông chất liệu tweed len xám, kiểu dáng một h | khớp 0/2 thuộc tính | cùng loại: False
 3. [OK ] q='tìm áo nỉ cotton xám' attrs={"CHAT_LIEU": "cotton", "MAU": "xám"}
      âm: Áo len dài tay Aligro | khớp 1/2 thuộc tính | cùng loại: True
 4. [OK ] q='tìm ví da' attrs={"CHAT_LIEU": "leather"}
      âm: Da Thật Chính Hãng Da Nam Túi Đeo Vai Kinh Doanh Du Lịch Cặp Da Bò Túi | khớp 1/1 thuộc tính | cùng loại: False
 5. [OK ] q='tìm quần âu polyester basic' attrs={"PHONG_CACH": "basic", "CHAT_LIEU": "polyester"}
      âm: [may đo] quần tây the suits house, chất liệu vải | khớp 0/2 thuộc tính | cùng loại: True
 6. [OK ] q='áo khoác đỏ đô cotton casual' attrs={"CHAT_LIEU": "cotton", "MAU": "đỏ đô", "PHONG_CACH": "casual"}
      âm: Áo khoác bông nam May 10 mã NAVY | khớp 0/3 thuộc tính | cùng loại: True
 7. [OK ] q='túi nâu' attrs={"MAU": "nâu"}
      âm: WQ WAIOU LIVELYS Ví tiền xu mini nhiều màu bằng da PU, móc khóa, đựng  | khớp 1/1 thuộc tính | cùng loại: False
 8. [OK ] q='giày chạy bộ màu trắng' attrs={"MAU": "trắng"}
      âm: Adidas | Giày Chạy Bộ Nhẹ Adidas Dành Cho Nam Giày Thể Thao Lưới Thoán | khớp 0/1 thuộc tính | cùng loại: True
 9. [OK ] q='quần short màu xám' attrs={"MAU": "xám"}
      âm: Quần Short Nam Chất Liệu Cotton Linen Thoáng Khí Mùa Hè 2025, Màu Trơn | khớp 0/1 thuộc tính | cùng loại: True
10. [OK ] q='túi đeo chéo vintage' attrs={"PHONG_CACH": "vintage"}
      âm: ≮1103≯Túi đeo chéo da PU dành cho nam, phù hợp cho công việc và đi du  | khớp 0/1 thuộc tính | cùng loại: True
11. [OK ] q='tìm áo polo silk' attrs={"CHAT_LIEU": "silk"}
      âm: Áo thun Polo trơn nam có cổ Phúc An 2033 có size lớn (over size) | khớp 0/1 thuộc tính | cùng loại: False
12. [OK ] q='áo polo regular fit phong cách minimalist polyester' attrs={"CHAT_LIEU": "polyester", "KIEU_DANG": "regular fit", "PHONG_CACH": "minimalist"}
      âm: Áo blazer nam trơn basic vải sợi Linen tự nhiên không bai xù, mặc thoả | khớp 0/3 thuộc tính | cùng loại: False
13. [OK ] q='tìm áo sơ mi polyester trắng slim fit' attrs={"CHAT_LIEU": "polyester", "MAU": "trắng", "KIEU_DANG": "slim fit"}
      âm: Bộ vest nam màu trơn, chất lượng cao, thời trang, xu hướng dự tiệc cướ | khớp 1/3 thuộc tính | cùng loại: False
14. [OK ] q='tìm quần short kaki dáng tiêu chuẩn' attrs={"KIEU_DANG": "regular fit", "CHAT_LIEU": "kaki"}
      âm: Quần short bóng rổ khô nhanh cho nam và nữ, quần thể thao nhẹ và thoán | khớp 0/2 thuộc tính | cùng loại: True
15. [OK ] q='áo len wool' attrs={"CHAT_LIEU": "wool"}
      âm: Áo thun dài tay nam dáng rộng họa tiết in thời trang cổ tròn phong các | khớp 0/1 thuộc tính | cùng loại: False
16. [OK ] q='tìm quần kaki cotton công sở slim fit' attrs={"KIEU_DANG": "slim fit", "PHONG_CACH": "công sở", "CHAT_LIEU": "cotton"}
      âm: G&MN Quần dài nam mùa hè phong cách Âu Mỹ, dáng rộng ống đứng, quần cô | khớp 1/3 thuộc tính | cùng loại: False
17. [OK ] q='cà vạt vintage polyester' attrs={"PHONG_CACH": "vintage", "CHAT_LIEU": "polyester"}
      âm: [VNXK] Quần tây trắng tinh, quần âu trắng ống côn dáng Hàn Quốc chất v | khớp 0/2 thuộc tính | cùng loại: False
18. [OK ] q='ví tối giản' attrs={"PHONG_CACH": "minimalist"}
      âm: ≮1105≯SEPTWOLVES | Ví da mềm lớp đầu tiên, kiểu dáng ngắn, ngang, thời | khớp 0/1 thuộc tính | cùng loại: True
19. [OK ] q='áo sơ mi đời thường regular fit' attrs={"PHONG_CACH": "casual", "KIEU_DANG": "regular fit"}
      âm: Áo Khoác Blazer Sọc Thời Trang Dành Cho Nam Giới Thông Minh, Giản Dị,  | khớp 0/2 thuộc tính | cùng loại: False
20. [OK ] q='giày chạy bộ màu trắng' attrs={"MAU": "trắng"}
      âm: Giày Chạy Bộ Nam Marathon Chất Liệu Carbon, Thoáng Khí, Nhẹ, Giày Thể  | khớp 0/1 thuộc tính | cùng loại: True"""

## V0 — kiểm tra đầu vào
- train: 10690 sản phẩm (kỳ vọng 10690)
- val: 1339 sản phẩm (kỳ vọng 1339)
- test: 1339 sản phẩm (kỳ vọng 1339)
- train ∩ val: id trùng 0, text_group trùng 0
- train ∩ test: id trùng 0, text_group trùng 0
- val ∩ test: id trùng 0, text_group trùng 0
- bad_products: 312/1339; lý do {'mo_ta_rong': 312}; theo shop {'icondenim': 1, 'tiki': 34, 'lazada': 277}
- ứng viên hạt giống còn lại: 1027
- đường dẫn ảnh thử: ROOT, ROOT/local_path, config.PROCESSED_DIR (D:\vimfd-retrieval\data\processed)


## G1 — Kiểm tra queries_200.csv
- Tổng: 200 dòng
- n_attrs: 1=70 / 2=70 / 3+=60
- Danh mục: áo=80 / quần=60 / phụ kiện=60
- Từ đồng nghĩa: 77/200
- Lỗi: 0 | Cảnh báo: 1
=== G1 Kiểm tra queries_200.csv (200 dòng) ===

Đủ 200 dòng
query_id không trùng
Định dạng query_id đúng (Q001–Q200)
Mọi query_text đều 2–12 từ

Phân bố n_attrs: 1=70 | 2=70 | 3+=60
attrs_in_query đúng định dạng

Phân bố danh mục: {'quần': 60, 'phụ kiện': 60, 'áo': 80}
Phân bố danh mục đạt chỉ tiêu
Không có hạt giống nào trong bad_products
Không có cụm 4 từ nào bị chép từ name
Không có tên thương hiệu

Dùng từ đồng nghĩa: 77/200 (38%)
Tỉ lệ từ đồng nghĩa đạt ≥ 30%

==================================================
KHÔNG CÓ LỖI — Có thể chạy V8

1 CẢNH BÁO (không bắt buộc sửa):
  Phân bố n_attrs lệch chỉ tiêu 35/35/30: 70/70/60
## V8 — chạy dual-encoder và ViMFR (2026-11-22 15:58)
- thiết bị: cpu (KHÔNG có GPU: độ trễ là của CPU)
- torch 2.1.0+cpu; chỉ mục Test 1339/1339 sản phẩm (thiếu ảnh 0); fp16=False
- 200 truy vấn, top_k=50; thứ tự giữ nguyên sau rerank: 0/200
- độ trễ (bỏ 5 truy vấn đầu; gồm tách từ + mã hóa + FAISS + rerank): trung vị 17715.3 ms, P95 28273.8 ms
