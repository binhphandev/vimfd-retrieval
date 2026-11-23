[07:15:42] === Bắt đầu V6 ===
[07:15:42] Model: vinai/phobert-base, MAX_LEN=256, BATCH=32
[07:15:42] LR_phobert=2e-05, LR_linear=0.0001, WD=0.01, lịch=warmup 10% + cosine
[07:15:42] MAX_EPOCHS=10, PATIENCE=3, trần thời gian=2.25h
[07:15:42] FP16=True, seed=42, device=cuda
[07:15:42] GPU: Tesla T4, VRAM 15.6 GB
[07:15:43] Train: 25027 cặp, Val: 4030 cặp
[07:15:44] Tách token (một lần)...
[07:15:46]   tách token train: 2000/25027
[07:15:47]   tách token train: 4000/25027
[07:15:48]   tách token train: 6000/25027
[07:15:50]   tách token train: 8000/25027
[07:15:51]   tách token train: 10000/25027
[07:15:52]   tách token train: 12000/25027
[07:15:53]   tách token train: 14000/25027
[07:15:54]   tách token train: 16000/25027
[07:15:55]   tách token train: 18000/25027
[07:15:57]   tách token train: 20000/25027
[07:15:59]   tách token train: 22000/25027
[07:16:00]   tách token train: 24000/25027
[07:16:00]   tách token train: 25027/25027
[07:16:02]   tách token val: 2000/4030
[07:16:03]   tách token val: 4000/4030
[07:16:03]   tách token val: 4030/4030
[07:17:42] === Bắt đầu V6 ===
[07:17:42] Model: vinai/phobert-base, MAX_LEN=256, BATCH=32
[07:17:42] LR_phobert=2e-05, LR_linear=0.0001, WD=0.01, lịch=warmup 10% + cosine
[07:17:42] MAX_EPOCHS=10, PATIENCE=3, trần thời gian=2.25h
[07:17:42] FP16=True, seed=42, device=cuda
[07:17:42] GPU: Tesla T4, VRAM 15.6 GB
[07:17:43] Train: 25027 cặp, Val: 4030 cặp
[07:17:43] Tách token (một lần)...
[07:17:45]   tách token train: 2000/25027
[07:17:46]   tách token train: 4000/25027
[07:17:47]   tách token train: 6000/25027
[07:17:49]   tách token train: 8000/25027
[07:17:50]   tách token train: 10000/25027
[07:17:51]   tách token train: 12000/25027
[07:17:53]   tách token train: 14000/25027
[07:17:55]   tách token train: 16000/25027
[07:17:56]   tách token train: 18000/25027
[07:17:57]   tách token train: 20000/25027
[07:17:58]   tách token train: 22000/25027
[07:17:59]   tách token train: 24000/25027
[07:18:00]   tách token train: 25027/25027
[07:18:01]   tách token val: 2000/4030
[07:18:02]   tách token val: 4000/4030
[07:18:02]   tách token val: 4030/4030
[07:18:13] Total steps=7830, warmup=783, steps/epoch=783
[07:18:49]   epoch 1 step 100/783 loss=0.5582 (1.1 phút từ lúc khởi chạy)
[07:19:25]   epoch 1 step 200/783 loss=0.5018 (1.7 phút từ lúc khởi chạy)
[07:20:03]   epoch 1 step 300/783 loss=0.4617 (2.3 phút từ lúc khởi chạy)
[07:20:41]   epoch 1 step 400/783 loss=0.4402 (3.0 phút từ lúc khởi chạy)
[07:21:19]   epoch 1 step 500/783 loss=0.4215 (3.6 phút từ lúc khởi chạy)
[07:21:57]   epoch 1 step 600/783 loss=0.4027 (4.3 phút từ lúc khởi chạy)
[07:22:35]   epoch 1 step 700/783 loss=0.3861 (4.9 phút từ lúc khởi chạy)
[07:23:21] Epoch 1/10 | train_loss=0.3744 | val_loss=0.2844 | val_acc=0.8797 | val_auc=0.9272 | val_top1=0.7692 | time=308s
[07:23:31]   → Lưu checkpoint tốt nhất (epoch 1, val_loss=0.2844)
[07:24:09]   epoch 2 step 100/783 loss=0.2699 (6.4 phút từ lúc khởi chạy)
[07:24:47]   epoch 2 step 200/783 loss=0.2609 (7.1 phút từ lúc khởi chạy)
[07:25:25]   epoch 2 step 300/783 loss=0.2518 (7.7 phút từ lúc khởi chạy)
[07:26:03]   epoch 2 step 400/783 loss=0.2435 (8.3 phút từ lúc khởi chạy)
[07:26:41]   epoch 2 step 500/783 loss=0.2380 (9.0 phút từ lúc khởi chạy)
[07:27:19]   epoch 2 step 600/783 loss=0.2329 (9.6 phút từ lúc khởi chạy)
[07:27:57]   epoch 2 step 700/783 loss=0.2287 (10.3 phút từ lúc khởi chạy)
[07:28:42] Epoch 2/10 | train_loss=0.2237 | val_loss=0.1989 | val_acc=0.9256 | val_auc=0.9651 | val_top1=0.8734 | time=311s
[07:28:58]   → Lưu checkpoint tốt nhất (epoch 2, val_loss=0.1989)
[07:29:37]   epoch 3 step 100/783 loss=0.1514 (11.9 phút từ lúc khởi chạy)
[07:30:15]   epoch 3 step 200/783 loss=0.1530 (12.5 phút từ lúc khởi chạy)
[07:30:53]   epoch 3 step 300/783 loss=0.1618 (13.2 phút từ lúc khởi chạy)
[07:31:31]   epoch 3 step 400/783 loss=0.1553 (13.8 phút từ lúc khởi chạy)
[07:32:09]   epoch 3 step 500/783 loss=0.1540 (14.4 phút từ lúc khởi chạy)
[07:32:47]   epoch 3 step 600/783 loss=0.1536 (15.1 phút từ lúc khởi chạy)
[07:33:25]   epoch 3 step 700/783 loss=0.1543 (15.7 phút từ lúc khởi chạy)
[07:34:09] Epoch 3/10 | train_loss=0.1524 | val_loss=0.1655 | val_acc=0.9437 | val_auc=0.9806 | val_top1=0.9243 | time=311s
[07:34:12]   → Lưu checkpoint tốt nhất (epoch 3, val_loss=0.1655)
[07:34:50]   epoch 4 step 100/783 loss=0.1141 (17.1 phút từ lúc khởi chạy)
[07:35:28]   epoch 4 step 200/783 loss=0.1145 (17.8 phút từ lúc khởi chạy)
[07:36:06]   epoch 4 step 300/783 loss=0.1155 (18.4 phút từ lúc khởi chạy)
[07:36:44]   epoch 4 step 400/783 loss=0.1110 (19.0 phút từ lúc khởi chạy)
[07:37:22]   epoch 4 step 500/783 loss=0.1084 (19.7 phút từ lúc khởi chạy)
[07:38:01]   epoch 4 step 600/783 loss=0.1092 (20.3 phút từ lúc khởi chạy)
[07:38:39]   epoch 4 step 700/783 loss=0.1090 (20.9 phút từ lúc khởi chạy)
[07:39:23] Epoch 4/10 | train_loss=0.1080 | val_loss=0.1827 | val_acc=0.9469 | val_auc=0.9798 | val_top1=0.9318 | time=310s
[07:39:23]   → Không cải thiện (1/3)
[07:40:01]   epoch 5 step 100/783 loss=0.0681 (22.3 phút từ lúc khởi chạy)
[07:40:39]   epoch 5 step 200/783 loss=0.0781 (22.9 phút từ lúc khởi chạy)
[07:41:17]   epoch 5 step 300/783 loss=0.0797 (23.6 phút từ lúc khởi chạy)
[07:41:54]   epoch 5 step 400/783 loss=0.0775 (24.2 phút từ lúc khởi chạy)
[07:42:32]   epoch 5 step 500/783 loss=0.0777 (24.8 phút từ lúc khởi chạy)
[07:43:10]   epoch 5 step 600/783 loss=0.0803 (25.5 phút từ lúc khởi chạy)
[07:43:48]   epoch 5 step 700/783 loss=0.0821 (26.1 phút từ lúc khởi chạy)
[07:44:32] Epoch 5/10 | train_loss=0.0827 | val_loss=0.2090 | val_acc=0.9496 | val_auc=0.9806 | val_top1=0.9318 | time=310s
[07:44:32]   → Không cải thiện (2/3)
[07:45:10]   epoch 6 step 100/783 loss=0.0573 (27.5 phút từ lúc khởi chạy)
[07:45:48]   epoch 6 step 200/783 loss=0.0599 (28.1 phút từ lúc khởi chạy)
[07:46:26]   epoch 6 step 300/783 loss=0.0631 (28.7 phút từ lúc khởi chạy)
[07:47:04]   epoch 6 step 400/783 loss=0.0617 (29.4 phút từ lúc khởi chạy)
[07:47:42]   epoch 6 step 500/783 loss=0.0619 (30.0 phút từ lúc khởi chạy)
[07:48:19]   epoch 6 step 600/783 loss=0.0616 (30.6 phút từ lúc khởi chạy)
[07:48:57]   epoch 6 step 700/783 loss=0.0601 (31.2 phút từ lúc khởi chạy)
[07:49:41] Epoch 6/10 | train_loss=0.0613 | val_loss=0.1866 | val_acc=0.9571 | val_auc=0.9842 | val_top1=0.9529 | time=309s
[07:49:41]   → Không cải thiện (3/3)
[07:49:41] ⏹  Early stopping tại epoch 6.
[07:49:41] 
=== Kết thúc V6 ===
[07:49:41] Epoch tốt nhất: 3, val_loss=0.1655
[07:49:41] Lý do dừng: early stopping (patience=3)
[07:49:41] Tổng thời gian: 32.0 phút
[07:49:41] Checkpoint lưu tại: /content/vimfd-retrieval/checkpoints/cross_encoder_best
[07:49:41] Lưu ý: val_auc/val_top1 trên truy vấn giả thường rất cao; chất lượng thật chỉ đo được trên 200 truy vấn kiểm thử.
