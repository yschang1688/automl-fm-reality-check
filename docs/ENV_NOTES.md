# 環境坑（macOS arm64／M1）

1. **LightGBM 與 PyTorch 的 OpenMP 衝突**：兩者各帶一份 OpenMP runtime，同一行程裡先後載入會 SIGSEGV（exit 139），沒有 Python 例外。解法：載入任何東西前設 `OMP_NUM_THREADS=1`，並讓 `import lightgbm` 早於 torch。三支 run 腳本開頭都已這樣寫。
2. **`autogluon.timeseries` 裝不起來**：它依賴 `autogluon.tabular[xgboost]`，後者指定 `xgboost-cpu`，而 xgboost-cpu 沒有 macOS arm64 wheel，會退回原始碼編譯並因缺 cmake 失敗。繞道：先裝一般的 `xgboost`，再逐一安裝 autogluon.timeseries 在 PyPI 上列的非 extra 相依，最後 `pip install --no-deps autogluon.timeseries==1.6.1`。
3. **表格基礎模型在 CPU 上跑不動**：TabICLv2 對 3,940 列的 in-context 推論在 M1 CPU 跑 2 小時 45 分未完成；同一件事在 Colab T4 上 4 秒。基礎模型的「免訓練」是用推論成本換來的。
