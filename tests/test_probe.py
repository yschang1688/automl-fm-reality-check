"""可學性閘門的行為測試：有結構的資料必須放行、打亂標籤的純雜訊必須擋下。

兩個方向都要測——只測「會放行」的閘門，跟一道永遠放行的壞閘門看起來一模一樣。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run04_inside_the_prior import learnability_pval, prior  # noqa: E402

N = 12


def _filtered_draws(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return [prior.rand_dataset_filtered([0, 0], [2], 300) for _ in range(N)]


def test_gate_passes_learnable_data():
    # 重算的統計量必須與產生器內建過濾器一致：已通過的資料集重算後仍須全數通過
    assert all(learnability_pval(t) < 0.05 for t in _filtered_draws(7))


def test_gate_rejects_shuffled_labels():
    # 探針：同一批資料打亂標籤後，特徵與標的無關，閘門必須擋下（容許至多 1 次誤放行）
    passed = 0
    for t in _filtered_draws(7):
        t2 = dict(t)
        y = t["y_0"].clone()
        t2["y_0"] = y[torch.randperm(len(y))]
        passed += learnability_pval(t2) < 0.05
    assert passed <= 1, f"打亂標籤後仍有 {passed}/{N} 個資料集被放行"
