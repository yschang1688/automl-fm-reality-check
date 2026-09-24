"""KDD 2026 notebook 04 重現（本機 CPU，存圖不顯示）。prior.py 取自 soda-inria/nanotabicl（BSD-3-Clause）。
圖 1：48 個通過可學性過濾的合成分類資料集；圖 2：同一產生器不過濾，被過濾器保留 vs 拒絕各 8 個。"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
import os, sys, urllib.request
_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR = os.path.join(_HERE, "prior.py")
if not os.path.exists(_PRIOR):  # BSD-3-Clause, soda-inria/nanotabicl；執行時下載、不入庫
    urllib.request.urlretrieve("https://raw.githubusercontent.com/soda-inria/nanotabicl/main/prior.py", _PRIOR)
sys.path.insert(0, _HERE)
import prior

def scatter(ax, tensors, n_classes):
    x = torch.cat([tensors["x_0"], tensors["x_1"]], dim=-1); y = tensors["y_0"].squeeze(-1)
    ax.set(xticks=[], yticks=[])
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=ListedColormap(plt.get_cmap("tab10").colors[:n_classes]),
               vmin=0, vmax=n_classes - 1, s=40, marker=".", linewidths=0)

def learnability_pval(tensors):
    """與 prior.rand_dataset_filtered 相同的統計量：ExtraTrees OOB 勝過平均基準的 bootstrap p 值。"""
    X = torch.cat([t.float() for k, t in tensors.items() if k.startswith("x")], dim=-1).numpy()
    y = tensors["y_0"].long().squeeze(-1)
    Y = F.one_hot(y, num_classes=int(y.max().item() + 1)).float()
    Y = (Y[:, :1] if Y.shape[1] == 2 else Y).numpy()
    et = ExtraTreesRegressor(n_estimators=25, bootstrap=True, oob_score=True, n_jobs=1, random_state=1,
                             max_depth=6).fit(X, Y[:, 0] if Y.shape[1] == 1 else Y)
    Yhat = et.oob_prediction_[:, None] if et.oob_prediction_.ndim == 1 else et.oob_prediction_
    m = ~np.isnan(Yhat).any(axis=1)
    imp = ((Y[m] - Y.mean(axis=0, keepdims=True)) ** 2 - (Y[m] - Yhat[m]) ** 2).sum(axis=1)
    idx = np.random.default_rng(0).integers(0, len(imp), size=(200, len(imp)))
    return float(np.mean(imp[idx].mean(axis=1) <= 0.0))

def n_cls():
    return 2 if np.random.rand() < 0.5 else np.random.randint(3, 11)

if __name__ == "__main__":
    np.random.seed(1); torch.manual_seed(1)
    fig, axs = plt.subplots(6, 8, figsize=(16, 12))
    for ax in axs.flat:
        k = n_cls(); scatter(ax, prior.rand_dataset_filtered([0, 0], [k], 300), k)
    plt.tight_layout(); plt.savefig("figs/nb04_gallery_filtered.png", dpi=80); plt.close()

    np.random.seed(3); torch.manual_seed(3)
    kept, rej = [], []
    while len(kept) < 8 or len(rej) < 8:
        k = n_cls(); t = prior.rand_dataset_plain([0, 0], [k], 300)
        b = kept if learnability_pval(t) < 0.05 else rej
        if len(b) < 8: b.append((t, k))
    fig, axs = plt.subplots(2, 8, figsize=(16, 4.4))
    for r, (title, draws) in enumerate([("kept (learnable)", kept), ("rejected", rej)]):
        for c, (t, k) in enumerate(draws):
            scatter(axs[r, c], t, k)
            if c == 0: axs[r, c].set_ylabel(title)
    plt.tight_layout(); plt.savefig("figs/nb04_kept_vs_rejected.png", dpi=80); plt.close()
    print("saved figs/nb04_gallery_filtered.png, figs/nb04_kept_vs_rejected.png")
