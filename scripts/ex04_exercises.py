"""第 4 本延伸練習：把基礎模型預訓練用的「可學性過濾器」當成一道品質閘來驗收。
用法（repo 根目錄）：python scripts/ex04_exercises.py e1|e2|e3|e4|all   （結果寫到 results/ex04_*.csv，圖寫到 figs/）
E1 通過率：生產者原始輸出有多少比例被擋？（對照 hvac P1 品質閘 Pass/Conditional/Reject 分布）
E2 探針要真的壞：把「已通過」資料集的標籤打亂，過濾器必須擋下；未打亂的必須放行（對照守門測試的反向探針）
E3 樣本數與檢定力：同一產生器、只改 n，通過率怎麼變？（對照 hvac「n<3 不判定」與 PPG MDE 反推）
E4 類別特徵：加入高基數類別欄後通過率與 TabPFN 系在 nb06 輸給 CatBoost 的關係"""
import sys, time, numpy as np, torch, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run04_inside_the_prior import learnability_pval, n_cls, prior

def draws(n, n_samples=300, x_cat=(0, 0), seed=0):
    np.random.seed(seed); torch.manual_seed(seed); out = []
    for _ in range(n):
        k = n_cls(); t = prior.rand_dataset_plain(list(x_cat), [k], n_samples)
        out.append({"k": k, "p": learnability_pval(t)})
    return pd.DataFrame(out)

def e1():
    df = draws(300); df["pass"] = df.p < 0.05
    by = df.assign(task=np.where(df.k == 2, "binary", "multiclass")).groupby("task")["pass"].agg(["mean", "size"])
    print("E1 overall pass rate", round(df["pass"].mean(), 3), "\n", by.round(3))
    plt.hist(df.p, bins=20); plt.axvline(0.05, color="r"); plt.xlabel("bootstrap p-value"); plt.ylabel("draws")
    plt.title("E1: learnability p-value of 300 raw prior draws"); plt.savefig("figs/ex04_e1_pvalues.png", dpi=80); plt.close()
    df.to_csv("results/ex04_e1_pass_rate.csv", index=False)

def e2():
    np.random.seed(7); torch.manual_seed(7); rows = []
    for i in range(60):
        k = n_cls(); t = prior.rand_dataset_filtered([0, 0], [k], 300)
        p_clean = learnability_pval(t)
        t2 = dict(t); y = t["y_0"].clone(); t2["y_0"] = y[torch.randperm(len(y))]
        rows.append({"k": k, "p_clean": p_clean, "p_shuffled": learnability_pval(t2)})
    df = pd.DataFrame(rows)
    print("E2 clean pass rate (must be 1.0):", (df.p_clean < 0.05).mean().round(3),
          "| shuffled false-pass rate (should be ~<=0.05):", (df.p_shuffled < 0.05).mean().round(3))
    df.to_csv("results/ex04_e2_probe.csv", index=False)

def e3():
    rows = []
    for n in (50, 150, 300, 1000):
        df = draws(150, n_samples=n, seed=11); rows.append({"n_samples": n, "pass_rate": (df.p < 0.05).mean(), "draws": len(df)})
        print("E3 n=", n, "pass rate", round(rows[-1]["pass_rate"], 3), flush=True)
    pd.DataFrame(rows).to_csv("results/ex04_e3_sample_size.csv", index=False)

def e4():
    rows = []
    for name, xc in [("2 numeric", (0, 0)), ("2 cat x 5", (5, 5)), ("2 cat x 50", (50, 50)), ("2 cat x 200", (200, 200))]:
        df = draws(150, x_cat=xc, seed=13); rows.append({"features": name, "pass_rate": (df.p < 0.05).mean(), "draws": len(df)})
        print("E4", name, "pass rate", round(rows[-1]["pass_rate"], 3), flush=True)
    pd.DataFrame(rows).to_csv("results/ex04_e4_categorical.csv", index=False)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)]:
        if which in (name, "all"):
            t0 = time.time(); fn(); print(f"[{name} {time.time()-t0:.0f}s]", flush=True)
