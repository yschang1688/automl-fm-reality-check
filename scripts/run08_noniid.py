"""KDD 2026 notebook 08 重現：MiceProtein 以「隻」為單位切 30% 為測試；naive 隨機 holdout vs validation_structure={'group_on':'mouse'}。"""
import os; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import lightgbm as _lgb_preload  # 先於 torch 載入，避免 macOS OpenMP 雙載 SIGSEGV
import numpy as np, openml, pandas as pd
from autogluon.tabular import TabularPredictor
ds = openml.datasets.get_dataset(40966)
df, *_ = ds.get_data(include_row_id=True)
df["mouse"] = df["MouseID"].astype(str).str.split("_").str[0]; df = df.drop(columns=["MouseID"])
rng = np.random.default_rng(0); mice = df["mouse"].unique()
test_mice = set(rng.choice(mice, size=int(0.3 * len(mice)), replace=False))
train = df[~df.mouse.isin(test_mice)].reset_index(drop=True); test = df[df.mouse.isin(test_mice)].reset_index(drop=True)
print(f"{df.mouse.nunique()} mice, {len(train)} train rows, {len(test)} test rows ({len(test_mice)} held-out mice)", flush=True)
hp = {"GBM": {}, "RF": {}, "XGB": {}}
naive = TabularPredictor(label="class", eval_metric="log_loss", path="AutogluonModels/noniid_naive", verbosity=0).fit(train.drop(columns=["mouse"]), hyperparameters=hp)
lb1 = naive.leaderboard(test.drop(columns=["mouse"]))[["model", "score_test", "score_val"]]
print("== naive\n", lb1.to_string(), flush=True)
grouped = TabularPredictor(label="class", eval_metric="log_loss", path="AutogluonModels/noniid_grouped", verbosity=0).fit(train, hyperparameters=hp, validation_structure={"group_on": "mouse"})
lb2 = grouped.leaderboard(test)[["model", "score_test", "score_val"]]
print("== grouped\n", lb2.to_string(), flush=True)
lb1.assign(run="naive").pipe(lambda a: pd.concat([a, lb2.assign(run="grouped")])).to_csv("results/nb08_results.csv", index=False)
