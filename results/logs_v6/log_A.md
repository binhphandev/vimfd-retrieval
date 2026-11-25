
## V5 — khai thác âm khó (DRY RUN – không dùng)
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=12s
- train: 5155 dương, 20543 âm (1:3.99); bỏ vì 0 âm: 19; thiếu ảnh: 0; TB bị loại/truy vấn: 9.3/50; âm cùng loại: 56%; âm cùng loại và khớp n−1 thuộc tính: 45%
- val: 860 dương, 3440 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 0; TB bị loại/truy vấn: 8.5/50; âm cùng loại: 53%; âm cùng loại và khớp n−1 thuộc tính: 45%
- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log

## V5 — khai thác âm khó
- Tham số: seed=42, top_k=50, n_neg=4, strict_n1=False, thời gian=70s
- train: 5171 dương, 20648 âm (1:3.99); bỏ vì 0 âm: 3; thiếu ảnh: 0; TB bị loại/truy vấn: 8.2/50; âm cùng loại: 77%; âm cùng loại và khớp n−1 thuộc tính: 56%
- val: 860 dương, 3440 âm (1:4.00); bỏ vì 0 âm: 0; thiếu ảnh: 0; TB bị loại/truy vấn: 8.1/50; âm cùng loại: 65%; âm cùng loại và khớp n−1 thuộc tính: 53%
- Kiểm tra 20 cặp (tự động): 20/20 — nhớ ghi kết quả ĐỌC BẰNG MẮT của A vào log
