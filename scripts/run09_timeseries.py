"""KDD 2026 notebook 09 重現（本機 M1 CPU，去圖）。§1–3 澳洲電力 5 區半小時：基準／ML／零樣本 FM；§4 BuildingsBench Bull 41 棟商業建築小時負荷＋氣象協變量：Chronos-2 無／有協變量、feature importance、LoRA 微調。"""
import os; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import lightgbm as _lgb_preload  # 先於 torch 載入（macOS OpenMP 雙載 SIGSEGV）
import warnings; warnings.filterwarnings("ignore")
import time, pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
def lb(p, test, tag):
    df = p.leaderboard(test)[["model","score_test","score_val","fit_time_marginal","pred_time_test"]]
    print(f"\n== {tag}\n{df.to_string(index=False)}", flush=True); return df.assign(section=tag)
out = []
# §1–3
data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv")
print(f"AU: {data.num_items} series, freq {data.freq}", flush=True)
PL, NW = 48, 3
train, test = data.train_test_split(NW * PL)
p = TimeSeriesPredictor(prediction_length=PL, target="target", eval_metric="MASE", path="AutogluonModels/ts_base", verbosity=0).fit(
    train, hyperparameters={"Average": {}, "Naive": {}, "SeasonalNaive": {}}, enable_ensemble=False)
out.append(lb(p, test, "AU baselines"))
t0=time.time()
p = TimeSeriesPredictor(prediction_length=PL, target="target", eval_metric="MASE", path="AutogluonModels/ts_ml", verbosity=0).fit(
    train, hyperparameters={"SeasonalNaive": {}, "ETS": {}, "RecursiveTabular": {}, "PatchTST": {}}, time_limit=120)
out.append(lb(p, test, f"AU ML+ensemble ({time.time()-t0:.0f}s)"))
print(p.evaluate(test, metrics=["MASE","WQL","MQL"]), flush=True)
t0=time.time()
p = TimeSeriesPredictor(prediction_length=PL, target="target", eval_metric="MASE", path="AutogluonModels/ts_fm", verbosity=0).fit(
    train, hyperparameters={"Chronos2": {}, "Toto2": {"model_path": "Toto-2.0-22m"}}, enable_ensemble=False)
out.append(lb(p, test, f"AU zero-shot FM ({time.time()-t0:.0f}s incl. download)"))
# §4 Bull
data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/bull/test.parquet", id_column="id")
print(f"Bull: {data.num_items} buildings, freq {data.freq}, cols {list(data.columns)}", flush=True)
PL = 24; cov = ["airtemperature","dewtemperature","sealvlpressure"]
train, test = data.train_test_split(PL)
pu = TimeSeriesPredictor(prediction_length=PL, target="load", eval_metric="MASE", path="AutogluonModels/bull_uni", verbosity=0).fit(train[["load"]], hyperparameters={"Chronos2": {}})
pc = TimeSeriesPredictor(prediction_length=PL, target="load", known_covariates_names=cov, eval_metric="MASE", path="AutogluonModels/bull_cov", verbosity=0).fit(train, hyperparameters={"Chronos2": {}})
su = -pu.evaluate(test[["load"]])["MASE"]; sc = -pc.evaluate(test)["MASE"]
print(f"\nBull Chronos-2 MASE  without covariates: {su:.4f}\nBull Chronos-2 MASE  with    covariates: {sc:.4f}", flush=True)
fi = pc.feature_importance(test, model="Chronos2", relative_scores=True); print("\nfeature importance\n", fi.to_string(), flush=True)
t0=time.time()
pf = TimeSeriesPredictor(prediction_length=PL, target="load", known_covariates_names=cov, eval_metric="MASE", path="AutogluonModels/bull_ft", verbosity=0).fit(
    train, hyperparameters={"Chronos2": [{"ag_args": {"name_suffix": "ZeroShot"}}, {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}}]}, time_limit=300, enable_ensemble=False)
out.append(lb(pf, test, f"Bull Chronos-2 zero-shot vs LoRA fine-tune ({time.time()-t0:.0f}s)"))
pd.concat(out).to_csv("results/nb09_results.csv", index=False)
pd.DataFrame({"bull_uni_MASE":[su],"bull_cov_MASE":[sc]}).to_csv("results/nb09_bull_cov.csv", index=False)
print("DONE", flush=True)
