# automl-fm-reality-check

重現 KDD 2026「Taming Structured Data Foundation Models with AutoML」教學的全部九本筆記本，並記下**教學文案沒說、實跑才看得到**的事。

## In brief

- Re-ran all nine notebooks of the KDD 2026 hands-on tutorial on AutoML and structured-data foundation models (AutoGluon 1.6, TabICLv2, TabFM, TabPFN-3, Chronos-2, Toto-2) on an M1 laptop CPU and a Colab T4.
- **A random validation split was off by 26×** on grouped data, and it picked the worst model as the best.
- **Fine-tuning Chronos-2 moved validation and test error in opposite directions** (both within 0.4%), so it cannot be claimed as an improvement.
- **Two identical pretraining runs of nanoTabPFN gave downstream AUC 0.447 and 0.897** with indistinguishable loss curves.
- Every number here is a single split or a single run. Nothing in this repo supports "foundation models beat gradient boosting" in general.

## 三個反例

**1. 隨機切分把數字報錯 26 倍，還選錯模型**（第 8 本）

資料是 72 隻老鼠的重複量測。以「隻」為單位留 30% 當測試集後，比較兩種驗證方式：

| 驗證方式 | 最佳模型的驗證 log-loss | 同一模型的真實測試 log-loss | 偏差 |
|---|---|---|---|
| 隨機 holdout | 0.071 | 1.865 | 26× |
| `validation_structure={"group_on": "mouse"}` | 1.306 | 1.539 | 1.18× |

隨機切分下，驗證分數最差的 RandomForest 其實在測試集上最好。同一隻老鼠的其他量測留在訓練集，驗證分數量到的是記憶，不是泛化。時序資料同理：rolling-origin 加上與預測步長等長的間隔，是同一件事。

**2. 微調的效果在驗證與測試上方向相反**（第 9 本，Colab T4）

41 棟商業建築的逐時用電、日前 24 步預測，氣溫、露點、氣壓當已知協變量：

| 模型 | 驗證 MASE | 測試 MASE |
|---|---|---|
| Chronos-2 零樣本 | 0.817 | 0.696 |
| Chronos-2 LoRA 微調（267 秒） | 0.820 | 0.693 |

驗證說微調變差，測試說變好，兩邊的差距都在 0.4% 以內，AutoGluon 最後選的是零樣本版本。教學文案寫「微調再把準確度往上推」，在這份 41 條序列的資料上不成立，教學自己給的經驗法則也是 100 條序列以上才適合微調。

**3. 預訓練的 loss 看不出下游會不會崩**（第 5 本，Colab T4）

同一份程式、同一份 1 GB 合成先驗，nanoTabPFN 連續預訓練兩次，各約 280 秒：

| 次 | 預訓練 loss 走勢 | 下游乳癌資料零樣本 AUC | 邏輯迴歸 AUC |
|---|---|---|---|
| 1 | 約 65 秒後在 0.46–0.60 打平 | 0.447 | 0.994 |
| 2 | 同形狀，0.45–0.62 | 0.897 | 0.994 |

兩條 loss 曲線看不出差別，下游結果一次比隨機還差、一次接近教學宣稱的 0.96。要判斷一個預訓練有沒有成功，得用下游任務驗，不能看上游 loss。

## 各本結果

| 本 | 內容 | 結果 | 執行環境 |
|---|---|---|---|
| 01 | 同一份破產預測資料，四層方法 | 裸 XGBoost AUC 0.957 → AutoGluon 8 折 bagging 0.968 → TabICLv2 0.984 → TabFM 單成員 0.995 | 前兩層本機與 T4 一致；後兩層 T4 |
| 03 | AutoGluon 當基礎模型動物園 | 加權集成 0.986、TabPFN-3 0.985、TabICLv2 0.984、TabDPT-Turbo 0.965；只做迴歸的 Nori 被自動略過 | T4 |
| 04 | 預訓練用的合成先驗 | 兩張圖重現，另加四題延伸練習（見下） | 本機 CPU，12 秒 |
| 05 | 自己預訓練 nanoTabPFN | 見反例 3 | T4 |
| 06 | 高基數類別資料 | CatBoost 0.851 贏過 TabICLv2 0.839 與 TabPFN-3 0.832；TabPFN-3 thinking mode（雲端 API，190 秒）0.869 反超 | T4 |
| 07 | KV-cache 與 Shapley 解釋 | 快取後 20 次預測 8.8 秒降到 1.0 秒；最高風險 5 家的預測機率全部飽和在 1.0，「其餘 54 個特徵合計 +1.98」壓過任何單一財務比率 | T4 |
| 08 | 非 IID 驗證 | 見反例 1 | 本機 CPU |
| 09 | 時序預測 | 澳洲電力 5 區：LightGBM MASE 0.800（T4 重跑 0.790）、Chronos-2 零樣本 0.614；41 棟建築 Chronos-2 只看歷史 1.194（輸給季節性 naive），加氣象協變量後 0.696，氣溫的 shuffle 重要度 0.32 | 零樣本模型本機與 T4 逐位一致；有訓練的模型每次不同 |

原始數字在 [`results/`](results/)。

## 這份 repo 不宣稱什麼

- **不宣稱基礎模型比 LightGBM 或樹模型準。** 每個數字都是單一 fold 或單一測試窗，沒有雜訊帶，也沒有配對檢定。澳洲電力需求還很可能出現在 Chronos 的預訓練語料裡。
- **不宣稱這些結論適用於時序或分組資料。** 表格基礎模型在非 IID、時序與大型資料上的表現另有文獻，見 [Beyond IID（arXiv 2606.30410）](https://arxiv.org/abs/2606.30410)。
- **不宣稱任何商用可行性。** TabPFN 2.5 以後的權重與 TabFM 的權重都是非商用授權。

## 第 4 本延伸練習

把 TabPFN 與 TabICL 預訓練時用的「可學性過濾器」當成一道品質閘來驗收。題目在 [`docs/EXERCISES_04.md`](docs/EXERCISES_04.md)，建議先寫下預測再執行。

| 題 | 問題 | 結果 | 能不能宣稱 |
|---|---|---|---|
| E1 | 原始合成資料有多少被丟掉 | 300 次抽樣通過 48.3% | 二元與多元分類的差距在雜訊帶內 |
| E2 | 打亂標籤後閘門擋不擋得住 | 原樣 60/60 放行，打亂後 0/60 誤放行 | 可以，探針確實有效 |
| E3 | 列數少時真的結構會不會被判成學不到 | 50 列 22.7%、300 列 48.7%、1000 列 51.3% | 50 到 300 可以；300 到 1000 不行 |
| E4 | 高基數類別欄是否較難通過 | 數值 42.0%、200 類 32.7% | 不行，差距約 1.6 個標準誤 |

同一個設定只換隨機種子，E1 與 E4 第一組的通過率就差了 6 個百分點。比較任何兩個比率之前，先算雜訊帶。

## 重現

本機（macOS arm64，Python 3.12）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run08_noniid.py
.venv/bin/python scripts/run04_inside_the_prior.py
.venv/bin/python scripts/ex04_exercises.py all
```

`run09_timeseries.py` 需要 `autogluon.timeseries`，在 macOS arm64 上的安裝繞道見 [`docs/ENV_NOTES.md`](docs/ENV_NOTES.md)。第 1 本後兩層與第 3、5、6、7 本需要 GPU，在 Colab T4 上跑原教學筆記本，修改處見 [`docs/COLAB_NOTES.md`](docs/COLAB_NOTES.md)。

## 來源與授權

- 教學：[KDD 2026 tutorial site](https://kdd26-automl-hands-on.github.io/)，材料在 [Innixma/kdd2026_tutorial_materials](https://github.com/Innixma/kdd2026_tutorial_materials)。該 repo 沒有授權檔，因此**本 repo 不轉載任何筆記本內容**。`scripts/` 裡的 `run01`、`run08`、`run09`、`run04` 依對應筆記本的流程改寫成腳本，資料集、切分與模型設定與原教學相同。
- 合成先驗：`prior.py` 來自 [soda-inria/nanotabicl](https://github.com/soda-inria/nanotabicl)（BSD-3-Clause），執行時下載，不放進本 repo。
- 本 repo 自寫的腳本、結果表、圖與文件以 MIT 授權釋出，見 [LICENSE](LICENSE)。
