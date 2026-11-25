
## V7 — baseline (2026-11-24 19:37) [SAMPLE 5 – không dùng]
- 5 truy vấn, corpus 1449, device=cuda, seed=42, top_k=50
- thư viện: sklearn 1.6.1, rank_bm25 không có, torch 2.14.1+cu130, transformers 4.57.6, sentence_transformers 5.7.0, tách từ truy vấn: src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)
- tfidf: TfidfVectorizer(token_pattern=\S+, ngram=(1,2), sublinear_tf); 0s; {'vocab_size': 38727}
- bm25: BM25Okapi(k1=1.2, b=0.75); 0s; {'k1': 1.2, 'b': 0.75}
- clip_vi2en: vinai/vinai-translate-vi2en (beam 4) + openai/clip-vit-base-patch32; 61s; {'translation': 'dịch mới bằng vinai/vinai-translate-vi2en', 'n_images': 1449, 'n_missing_images': 0}
- bienc_text: bkai-foundation-models/vietnamese-bi-encoder; 21s; {'query_segmenter': 'pyvi', 'max_seq_length': 256}

## V7 — baseline (2026-11-24 19:39)
- 200 truy vấn, corpus 1449, device=cuda, seed=42, top_k=50
- thư viện: sklearn 1.6.1, rank_bm25 không có, torch 2.14.1+cu130, transformers 4.57.6, sentence_transformers 5.7.0, tách từ truy vấn: src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)
- clip_vi2en: vinai/vinai-translate-vi2en (beam 4) + openai/clip-vit-base-patch32; 23s; {'translation': 'dịch mới bằng vinai/vinai-translate-vi2en', 'n_images': 1449, 'n_missing_images': 0}

## V7b — kiểm tra mất ngữ nghĩa khi dịch (2026-11-24 19:40)
- 200 truy vấn, 401 thuộc tính đã kiểm (chưa có trong từ điển: 0)
- thuộc tính giữ nguyên nghĩa: 91.5%; chỉ gần nghĩa: 1.7%; sai nghĩa: 1.0%; mất: 5.7%
- truy vấn giữ đủ mọi thuộc tính: 166/200; truy vấn có cờ mt_loss_flag=1: 35/200 (mất/sai thuộc tính, mất danh từ chính hoặc còn chữ Việt chưa dịch)
- danh từ chính trong bản dịch: giữ 189, mất 9, không có luật 2

## V7 — baseline (2026-11-24 19:45)
- 200 truy vấn, corpus 1449, device=cuda, seed=42, top_k=50
- thư viện: sklearn 1.6.1, rank_bm25 không có, torch 2.14.1+cu130, transformers 4.57.6, sentence_transformers 5.7.0, timm 1.0.31, tách từ truy vấn: src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)
- tfidf: TfidfVectorizer(token_pattern=\S+, ngram=(1,2), sublinear_tf); 0s; {'vocab_size': 38727}
- bm25: BM25Okapi(k1=1.2, b=0.75); 1s; {'k1': 1.2, 'b': 0.75}
- bienc_text: bkai-foundation-models/vietnamese-bi-encoder; 17s; {'query_segmenter': 'pyvi', 'max_seq_length': 256}
- fashionclip_vi2en: vinai/vinai-translate-vi2en (beam 4) + patrickjohncyh/fashion-clip; 26s; {'model': 'patrickjohncyh/fashion-clip', 'commit': '7e3ba62ce16b379a1ab479346b66f192e76f51b7', 'translation': 'đọc từ file', 'n_images': 1449, 'n_missing_images': 0}

## V7 — baseline (2026-11-24 19:46)
- 200 truy vấn, corpus 1449, device=cuda, seed=42, top_k=50
- thư viện: sklearn 1.6.1, rank_bm25 không có, torch 2.14.1+cu130, transformers 4.57.6, sentence_transformers 5.7.0, timm 1.0.31, tách từ truy vấn: src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)
- viclip_ot: LỖI RuntimeError: Không nạp được minhnguyent546/ViCLIP-OT: Model name 'timm/vit_base_patch16_dinov3.lvd1689m' has no source prefix but looks like a Hugging Face Hub repo id or a local path. Use 'hf-hub:timm/vit_base_patch16_dinov3.lvd1689m' to load from the Hub or 'local-dir:timm/vit_base_patch16_dinov3.lvd1689m' to 

## V7 — baseline (2026-11-24 19:52)
- 200 truy vấn, corpus 1449, device=cuda, seed=42, top_k=50
- thư viện: sklearn 1.6.1, rank_bm25 không có, torch 2.14.1+cu130, transformers 4.57.6, sentence_transformers 5.7.0, timm 1.0.31, tách từ truy vấn: src.serving.text_utils.preprocess_query (VnCoreNLP, đúng pipeline serving)
- viclip_ot: minhnguyent546/ViCLIP-OT (truy vấn tiếng Việt, không tách từ); 36s; {'model': 'minhnguyent546/ViCLIP-OT', 'commit': '81a7a165655204899e63a2bfd1c27dcfd3acc8e2', 'query_segmented': False, 'dim': 768, 'n_images': 1449, 'n_missing_images': 0}

## V9 — pooling
- hệ thống trong pool: dual_encoder, vimfr + 6 baseline (tfidf, bm25, clip_vi2en, bienc_text, fashionclip_vi2en, viclip_ot)
- K: dual_encoder 10, vimfr 10, baseline 2 (mỗi baseline); thêm hạt giống; N = 5063 (ngưỡng 3800: VƯỢT 3800 ngay cả khi K=2 (giữ K=2, chấp nhận kéo dài buổi gán nhãn))
- N theo K baseline: K=5: 7169, K=3: 5776, K=2: 5063
- cỡ pool mỗi truy vấn: min 17, max 32, TB 25.3; cặp thuộc overlap: 776
- hạt giống đã nằm trong top-K của hệ thống: 152/200; chỉ có nhờ thêm: 48/200
- file annotation: annotation_A_01: 30 truy vấn/776 cặp; annotation_A_02: 22 truy vấn/568 cặp; annotation_A_03: 21 truy vấn/543 cặp; annotation_A_04: 21 truy vấn/529 cặp; annotation_A_05: 21 truy vấn/534 cặp; annotation_B_01: 30 truy vấn/776 cặp; annotation_B_02: 22 truy vấn/546 cặp; annotation_B_03: 21 truy vấn/524 cặp; annotation_B_04: 21 truy vấn/544 cặp; annotation_B_05: 21 truy vấn/499 cặp
- ước tính công gán nhãn: 4.9 giờ mỗi người (hai người); 9.7 giờ nếu một người gán cả hai phần
- Hệ quả cần ghi vào báo cáo: sản phẩm trong top-10 của baseline nhưng ngoài top-K pool và ngoài pool của mọi hệ thống khác bị coi là không liên quan, nên chỉ số baseline có thể bị đánh giá thấp (có lợi cho ViMFR); V11 báo thêm judged@10 để đo mức này.
- file *_01 có 30 truy vấn (> 25 truy vấn/file ở mục 8.4): ghi mục 12 hoặc dùng --split-overlap

## V10 — kappa
- vòng 1: κ tuyến tính 1.000

## V10 — kappa
- vòng 1: κ tuyến tính 0.712

## V10 — ghép nhãn
- 5063 cặp, 200 truy vấn; nhãn {0: 3687, 2: 812, 1: 564}; R_q=0: 0
