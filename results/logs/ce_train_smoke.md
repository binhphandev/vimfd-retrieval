[04:02:41] === Bắt đầu V6 (SMOKE TEST) ===
[04:02:41] Model: vinai/phobert-base, MAX_LEN=128, BATCH=32
[04:02:41] LR_phobert=2e-05, LR_linear=0.0001, WD=0.01, lịch=warmup 10% + cosine
[04:02:41] MAX_EPOCHS=1, PATIENCE=3, trần thời gian=2.25h
[04:02:41] FP16=False, seed=42, device=cpu
[04:02:42] Train: 200 cặp, Val: 100 cặp
[04:02:44] Tách token (một lần)...
[04:02:44]   tách token train: 200/200
[04:02:44]   tách token val: 100/100
[04:02:46] Total steps=7, warmup=0, steps/epoch=7
[04:04:55] Epoch 1/1 | train_loss=0.5544 | val_loss=0.5031 | val_acc=0.8000 | val_auc=0.5444 | val_top1=0.1500 | time=129s
[04:04:55]   → Lưu checkpoint tốt nhất (epoch 1, val_loss=0.5031)
[04:04:55] 
=== Kết thúc V6 ===
[04:04:55] Epoch tốt nhất: 1, val_loss=0.5031
[04:04:55] Lý do dừng: max epochs
[04:04:55] Tổng thời gian: 2.2 phút
[04:04:55] Checkpoint lưu tại: D:\vimfd-retrieval\checkpoints\cross_encoder_best_smoke
[04:04:55] Lưu ý: val_auc/val_top1 trên truy vấn giả thường rất cao; chất lượng thật chỉ đo được trên 200 truy vấn kiểm thử.
