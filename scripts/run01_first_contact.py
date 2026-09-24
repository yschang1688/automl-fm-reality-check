"""KDD 2026 notebook 01 重現（本機 M1 CPU）：Tier1 裸 XGBoost／Tier2 AutoGluon bagged XGB／Tier3 TabICLv2。
Tier4 TabFM 需 GPU＋jax，本機跳過（如實記錄）。資料與切分照 notebook：OpenML task 363694 fold0。"""
import time, json, sys, os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import lightgbm as _lgb_preload  # 先於 torch 載入，避免 macOS OpenMP 雙載 SIGSEGV
import numpy as np, pandas as pd, openml
from sklearn.metrics import roc_auc_score
RANDOM_STATE = 0
results = {}
task = openml.tasks.get_task(363694)
X, y = task.get_X_and_y(dataset_format="dataframe")
y = (y == y.cat.categories[1]).astype(int) if hasattr(y, "cat") else y.astype(int)
tr, te = task.get_train_test_split_indices(repeat=0, fold=0)
X_train, y_train, X_test, y_test = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
print(f"train {X_train.shape} test {X_test.shape} pos_rate {y.mean():.3f}", flush=True)

from xgboost import XGBClassifier
t0 = time.time(); xgb = XGBClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
p = xgb.predict_proba(X_test)[:, 1]; results["Naive XGBoost"] = (roc_auc_score(y_test, p), time.time() - t0)
print("tier1", results["Naive XGBoost"], flush=True)

from autogluon.tabular import TabularPredictor
df = X_train.copy(); df["__label__"] = y_train.values
t0 = time.time()
pred = TabularPredictor(label="__label__", eval_metric="roc_auc", path="AutogluonModels/ag_xgb_bagged", verbosity=0).fit(
    df, hyperparameters={"XGB": {}}, num_bag_folds=8)
p = pred.predict_proba(X_test)[1]; results["XGBoost (AutoGluon, bagged)"] = (roc_auc_score(y_test, p), time.time() - t0)
print("tier2", results["XGBoost (AutoGluon, bagged)"], flush=True)

from tabicl import TabICLClassifier
ticl = TabICLClassifier()
warm = y_train.groupby(y_train).head(8).index
ticl.fit(X_train.loc[warm], y_train.loc[warm]); ticl.predict_proba(X_test.head(8))
t0 = time.time(); ticl.fit(X_train, y_train); p = ticl.predict_proba(X_test)[:, 1]
results["TabICLv2 (default, CPU)"] = (roc_auc_score(y_test, p), time.time() - t0)
print("tier3", results["TabICLv2 (default, CPU)"], flush=True)

comp = pd.DataFrame([(k, a, s) for k, (a, s) in results.items()], columns=["model", "test_auc", "seconds"]).set_index("model")
comp["error_vs_naive"] = (1 - comp.test_auc) / (1 - comp.loc["Naive XGBoost", "test_auc"])
print(comp.round(4).to_string())
comp.round(4).to_csv("results/nb01_results.csv")
