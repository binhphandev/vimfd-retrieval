
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
