[16:36:38] === Bắt đầu V6 (SMOKE TEST) ===
[16:36:38] Model: vinai/phobert-base, MAX_LEN=128, BATCH=32
[16:36:38] LR_phobert=2e-05, LR_linear=0.0001, WD=0.01, lịch=warmup 10% + cosine
[16:36:38] MAX_EPOCHS=1, PATIENCE=3, trần thời gian=2.25h
[16:36:38] FP16=True, seed=42, device=cuda
[16:36:38] GPU: Tesla T4, VRAM 15.6 GB
[16:36:39] Train: 200 cặp, Val: 100 cặp
[16:36:40] Tách token (một lần)...
[16:36:40]   tách token train: 200/200
[16:36:40]   tách token val: 100/100
[16:36:42] Total steps=7, warmup=0, steps/epoch=7
[16:36:45] Epoch 1/1 | train_loss=0.5599 | val_loss=0.5056 | val_acc=0.8000 | val_auc=0.4388 | val_top1=0.1500 | time=3s
[16:36:48]   → Lưu checkpoint tốt nhất (epoch 1, val_loss=0.5056)
[16:36:48] 
=== Kết thúc V6 ===
[16:36:48] Epoch tốt nhất: 1, val_loss=0.5056
[16:36:48] Lý do dừng: max epochs
[16:36:48] Tổng thời gian: 0.2 phút
[16:36:48] Checkpoint lưu tại: /content/vimfd-retrieval/checkpoints/cross_encoder_best_smoke
[16:36:48] Lưu ý: val_auc/val_top1 trên truy vấn giả thường rất cao; chất lượng thật chỉ đo được trên 200 truy vấn kiểm thử.
