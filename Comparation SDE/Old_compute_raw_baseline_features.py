
# compute_raw_baseline_features.py
# -*- coding: utf-8 -*-

"""
从原始信号构造一个“传统信号处理”基线特征集（2s 窗口，步长可配），用于与“空间注意力转移”特征进行公平对比。

输出：
- raw_window_features.csv   （窗口级统计 + label + subject）
- raw_subject_features.csv  （对窗口级按 subject 聚合后的受试者级统计）

要求：
- 需要 500HZ_NIFEA_DB_processed_data.npz 与 nifea_data_utils.py 在同一目录或可导入。
- 该 NPZ 的组织与前文训练脚本一致（signals:list[(T,C)], labels, record_names 可选）。

运行示例：
  python compute_raw_baseline_features.py --data /path/to/500HZ_NIFEA_DB_processed_data.npz \
      --window-sec 2 --step-sec 5 --fs 500 \
      --out-win raw_window_features.csv --out-subj raw_subject_features.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch

def hjorth_params(x):
    # Activity, Mobility, Complexity
    x = np.asarray(x)
    dx = np.diff(x, prepend=x[0])
    var0 = np.var(x)
    var1 = np.var(dx)
    if var0 <= 1e-12:
        return float(var0), 0.0, 0.0
    mob = np.sqrt(var1 / var0)
    ddx = np.diff(dx, prepend=dx[0])
    var2 = np.var(ddx)
    comp = np.sqrt((var2/var1)) if var1>1e-12 else 0.0
    return float(var0), float(mob), float(comp)

def bandpowers(x, fs, bands):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    out = {}
    for (lo,hi,name) in bands:
        mask = (f>=lo) & (f<hi)
        out[f"bp_{name}"] = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
    # 归一化（相对功率）
    total = sum(v for k,v in out.items())
    if total > 1e-12:
        for k in list(out.keys()):
            out[k+"_rel"] = out[k]/total
    else:
        for (lo,hi,name) in bands:
            out[f"bp_{name}_rel"] = 0.0
    return out

def stats_1d(x):
    x = np.asarray(x)
    return {
        "mean": float(np.mean(x)),
        "std":  float(np.std(x)),
        "rms":  float(np.sqrt(np.mean(x**2))),
        "skew": float(((x - x.mean())**3).mean() / (x.std()**3 + 1e-12)),
        "kurt": float(((x - x.mean())**4).mean() / (x.std()**4 + 1e-12)),
        "zcr":  float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:]))),
        "linelen": float(np.sum(np.abs(np.diff(x))))
    }

def features_for_window(win, fs):
    # win: (T, C)
    feats = {}
    bands = [(0.5,4,"lf"), (4,8,"mf"), (8,15,"hf"), (15,30,"vhf")]
    for c in range(win.shape[1]):
        x = win[:,c]
        s = stats_1d(x); feats.update({f"c{c}_{k}":v for k,v in s.items()})
        a,m,b = hjorth_params(x); feats.update({f"c{c}_hj_act":a, f"c{c}_hj_mob":m, f"c{c}_hj_comp":b})
        bp = bandpowers(x, fs, bands); feats.update({f"c{c}_{k}":v for k,v in bp.items()})
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="500HZ_NIFEA_DB_processed_data.npz")
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--step-sec", type=float, default=5.0)
    ap.add_argument("--out-win", type=str, default="raw_window_features.csv")
    ap.add_argument("--out-subj", type=str, default="raw_subject_features.csv")
    args = ap.parse_args()


    data = np.load(args.data, allow_pickle=True)
    signals = data["signals"]   # list-like, each (T,C)
    labels  = data["labels"]    # (n,)
    names   = data["record_names"] if "record_names" in data.files else np.arange(len(signals))

    # --- 这里是修改的地方 ---
    # 将 args.window-sec 和 args.step-sec
    # 分别修改为 args.window_sec 和 args.step_sec
    fs = args.fs
    W = int(args.window_sec * fs)
    S = int(args.step_sec * fs)
    # --- 修改结束 ---

    rows = []
    for sid, (sig, lab, name) in enumerate(zip(signals, labels, names)):
        sig = np.nan_to_num(sig.astype(np.float32))
        T, C = sig.shape
        for start in range(0, T-W+1, S):
            win = sig[start:start+W,:]
            feats = features_for_window(win, fs)
            feats.update({"subject": str(name), "label": int(lab)})
            rows.append(feats)

    dfw = pd.DataFrame(rows)
    dfw.to_csv(args.out_win, index=False, encoding="utf-8")

    # 受试者级聚合
    numeric_cols = [c for c in dfw.columns if c not in ["subject","label"]]
    df_subj = dfw.groupby(["subject","label"], as_index=False)[numeric_cols].mean()
    df_subj.to_csv(args.out_subj, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
