"""README 的關鍵數字必須與 results/ 的原始 CSV 一致。

README 是給人讀的摘要、CSV 是實跑產物；兩者各改各的，遲早會出現「摘要寫的數字沒有任何一次實跑產生過」。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
R = ROOT / "results"


def _in_readme(s: str):
    assert s in README, f"README 找不到「{s}」——數字改了嗎？"


def test_nb08_split_bias():
    df = pd.read_csv(R / "nb08_results.csv")
    naive = df[(df.run == "naive") & (df.model == "WeightedEnsemble_L2")].iloc[0]
    grouped = df[(df.run == "grouped") & (df.model == "WeightedEnsemble_L2")].iloc[0]
    _in_readme(f"{-naive.score_val:.3f}")
    _in_readme(f"{-naive.score_test:.3f}")
    _in_readme(f"{naive.score_test / naive.score_val:.0f}×")
    _in_readme(f"{grouped.score_test / grouped.score_val:.2f}×")


def test_nb09_finetune_disagreement():
    df = pd.read_csv(R / "nb09_colab_T4_finetune.csv")
    ft, zs = df.iloc[0], df.iloc[1]
    for v in (ft.score_test_MASE, ft.score_val_MASE, zs.score_test_MASE, zs.score_val_MASE):
        _in_readme(f"{v:.3f}")
    # 反例成立的前提：驗證與測試的方向真的相反
    assert (ft.score_val_MASE > zs.score_val_MASE) and (ft.score_test_MASE < zs.score_test_MASE)


def test_nb05_pretraining_runs():
    df = pd.read_csv(R / "nb05_colab_T4.csv")
    for v in df.nanoTabPFN_breast_cancer_AUC:
        _in_readme(f"{v:.3f}")


def test_ex04_sample_size():
    df = pd.read_csv(R / "ex04_e3_sample_size.csv")
    for _, r in df.iterrows():
        _in_readme(f"{int(r.n_samples)} 列 {r.pass_rate * 100:.1f}%")
