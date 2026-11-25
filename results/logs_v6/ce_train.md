[16:36:57] === Bắt đầu V6 ===
[16:36:57] Model: vinai/phobert-base, MAX_LEN=256, BATCH=32
[16:36:57] LR_phobert=2e-05, LR_linear=0.0001, WD=0.01, lịch=warmup 10% + cosine
[16:36:57] MAX_EPOCHS=10, PATIENCE=3, trần thời gian=2.25h
[16:36:57] FP16=True, seed=42, device=cuda
[16:36:57] GPU: Tesla T4, VRAM 15.6 GB
[16:36:58] Train: 25819 cặp, Val: 4300 cặp
[16:36:59] Tách token (một lần)...
[16:37:00]   tách token train: 2000/25819
[16:37:01]   tách token train: 4000/25819
[16:37:02]   tách token train: 6000/25819
[16:37:04]   tách token train: 8000/25819
[16:37:05]   tách token train: 10000/25819
[16:37:06]   tách token train: 12000/25819
[16:37:07]   tách token train: 14000/25819
[16:37:07]   tách token train: 16000/25819
[16:37:09]   tách token train: 18000/25819
[16:37:10]   tách token train: 20000/25819
[16:37:11]   tách token train: 22000/25819
[16:37:12]   tách token train: 24000/25819
[16:37:13]   tách token train: 25819/25819
[16:37:14]   tách token val: 2000/4300
[16:37:15]   tách token val: 4000/4300
[16:37:15]   tách token val: 4300/4300
[16:37:16] Total steps=8070, warmup=807, steps/epoch=807
[16:37:42]   epoch 1 step 100/807 loss=0.5705 (0.7 phút từ lúc khởi chạy)
[16:38:09]   epoch 1 step 200/807 loss=0.5371 (1.2 phút từ lúc khởi chạy)
[16:38:36]   epoch 1 step 300/807 loss=0.5213 (1.6 phút từ lúc khởi chạy)
[16:39:05]   epoch 1 step 400/807 loss=0.5124 (2.1 phút từ lúc khởi chạy)
[16:39:34]   epoch 1 step 500/807 loss=0.4930 (2.6 phút từ lúc khởi chạy)
[16:40:02]   epoch 1 step 600/807 loss=0.4679 (3.1 phút từ lúc khởi chạy)
[16:40:30]   epoch 1 step 700/807 loss=0.4422 (3.5 phút từ lúc khởi chạy)
[16:40:59]   epoch 1 step 800/807 loss=0.4225 (4.0 phút từ lúc khởi chạy)
[16:41:12] Epoch 1/10 | train_loss=0.4210 | val_loss=0.2674 | val_acc=0.8919 | val_auc=0.9324 | val_top1=0.8023 | time=236s
[16:41:15]   → Lưu checkpoint tốt nhất (epoch 1, val_loss=0.2674)
[16:41:43]   epoch 2 step 100/807 loss=0.2542 (4.8 phút từ lúc khởi chạy)
[16:42:12]   epoch 2 step 200/807 loss=0.2472 (5.2 phút từ lúc khởi chạy)
[16:42:40]   epoch 2 step 300/807 loss=0.2365 (5.7 phút từ lúc khởi chạy)
[16:43:09]   epoch 2 step 400/807 loss=0.2284 (6.2 phút từ lúc khởi chạy)
[16:43:37]   epoch 2 step 500/807 loss=0.2216 (6.7 phút từ lúc khởi chạy)
[16:44:06]   epoch 2 step 600/807 loss=0.2126 (7.1 phút từ lúc khởi chạy)
[16:44:34]   epoch 2 step 700/807 loss=0.2074 (7.6 phút từ lúc khởi chạy)
[16:45:02]   epoch 2 step 800/807 loss=0.2002 (8.1 phút từ lúc khởi chạy)
[16:45:15] Epoch 2/10 | train_loss=0.1999 | val_loss=0.2071 | val_acc=0.9267 | val_auc=0.9696 | val_top1=0.9012 | time=240s
[16:45:18]   → Lưu checkpoint tốt nhất (epoch 2, val_loss=0.2071)
[16:45:46]   epoch 3 step 100/807 loss=0.1293 (8.8 phút từ lúc khởi chạy)
[16:46:15]   epoch 3 step 200/807 loss=0.1317 (9.3 phút từ lúc khởi chạy)
[16:46:43]   epoch 3 step 300/807 loss=0.1280 (9.8 phút từ lúc khởi chạy)
[16:47:11]   epoch 3 step 400/807 loss=0.1342 (10.2 phút từ lúc khởi chạy)
[16:47:40]   epoch 3 step 500/807 loss=0.1333 (10.7 phút từ lúc khởi chạy)
[16:48:08]   epoch 3 step 600/807 loss=0.1302 (11.2 phút từ lúc khởi chạy)
[16:48:37]   epoch 3 step 700/807 loss=0.1280 (11.7 phút từ lúc khởi chạy)
[16:49:05]   epoch 3 step 800/807 loss=0.1262 (12.1 phút từ lúc khởi chạy)
[16:49:17] Epoch 3/10 | train_loss=0.1263 | val_loss=0.1726 | val_acc=0.9374 | val_auc=0.9769 | val_top1=0.9209 | time=240s
[16:49:22]   → Lưu checkpoint tốt nhất (epoch 3, val_loss=0.1726)
[16:49:51]   epoch 4 step 100/807 loss=0.0867 (12.9 phút từ lúc khởi chạy)
[16:50:20]   epoch 4 step 200/807 loss=0.0835 (13.4 phút từ lúc khởi chạy)
[16:50:48]   epoch 4 step 300/807 loss=0.0901 (13.8 phút từ lúc khởi chạy)
[16:51:16]   epoch 4 step 400/807 loss=0.0883 (14.3 phút từ lúc khởi chạy)
[16:51:44]   epoch 4 step 500/807 loss=0.0896 (14.8 phút từ lúc khởi chạy)
[16:52:13]   epoch 4 step 600/807 loss=0.0888 (15.3 phút từ lúc khởi chạy)
[16:52:41]   epoch 4 step 700/807 loss=0.0897 (15.7 phút từ lúc khởi chạy)
[16:53:10]   epoch 4 step 800/807 loss=0.0903 (16.2 phút từ lúc khởi chạy)
[16:53:22] Epoch 4/10 | train_loss=0.0906 | val_loss=0.1737 | val_acc=0.9523 | val_auc=0.9799 | val_top1=0.9349 | time=239s
[16:53:22]   → Không cải thiện (1/3)
[16:53:50]   epoch 5 step 100/807 loss=0.0630 (16.9 phút từ lúc khởi chạy)
[16:54:19]   epoch 5 step 200/807 loss=0.0545 (17.4 phút từ lúc khởi chạy)
[16:54:47]   epoch 5 step 300/807 loss=0.0627 (17.8 phút từ lúc khởi chạy)
[16:55:15]   epoch 5 step 400/807 loss=0.0632 (18.3 phút từ lúc khởi chạy)
[16:55:44]   epoch 5 step 500/807 loss=0.0644 (18.8 phút từ lúc khởi chạy)
[16:56:12]   epoch 5 step 600/807 loss=0.0641 (19.2 phút từ lúc khởi chạy)
[16:56:40]   epoch 5 step 700/807 loss=0.0643 (19.7 phút từ lúc khởi chạy)
[16:57:08]   epoch 5 step 800/807 loss=0.0653 (20.2 phút từ lúc khởi chạy)
[16:57:21] Epoch 5/10 | train_loss=0.0655 | val_loss=0.1936 | val_acc=0.9560 | val_auc=0.9813 | val_top1=0.9360 | time=239s
[16:57:21]   → Không cải thiện (2/3)
[16:57:49]   epoch 6 step 100/807 loss=0.0524 (20.9 phút từ lúc khởi chạy)
[16:58:17]   epoch 6 step 200/807 loss=0.0429 (21.3 phút từ lúc khởi chạy)
[16:58:46]   epoch 6 step 300/807 loss=0.0473 (21.8 phút từ lúc khởi chạy)
[16:59:14]   epoch 6 step 400/807 loss=0.0482 (22.3 phút từ lúc khởi chạy)
[16:59:42]   epoch 6 step 500/807 loss=0.0476 (22.7 phút từ lúc khởi chạy)
[17:00:10]   epoch 6 step 600/807 loss=0.0494 (23.2 phút từ lúc khởi chạy)
[17:00:39]   epoch 6 step 700/807 loss=0.0471 (23.7 phút từ lúc khởi chạy)
[17:01:07]   epoch 6 step 800/807 loss=0.0462 (24.2 phút từ lúc khởi chạy)
[17:01:19] Epoch 6/10 | train_loss=0.0462 | val_loss=0.2039 | val_acc=0.9567 | val_auc=0.9806 | val_top1=0.9360 | time=239s
[17:01:19]   → Không cải thiện (3/3)
[17:01:19] ⏹  Early stopping tại epoch 6.
[17:01:19] 
=== Kết thúc V6 ===
[17:01:19] Epoch tốt nhất: 3, val_loss=0.1726
[17:01:19] Lý do dừng: early stopping (patience=3)
[17:01:19] Tổng thời gian: 24.4 phút
[17:01:19] Checkpoint lưu tại: /content/vimfd-retrieval/checkpoints/cross_encoder_best
[17:01:19] Lưu ý: val_auc/val_top1 trên truy vấn giả thường rất cao; chất lượng thật chỉ đo được trên 200 truy vấn kiểm thử.
