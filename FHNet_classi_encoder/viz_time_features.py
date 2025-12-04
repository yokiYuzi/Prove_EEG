# viz_time_features.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_sequences_npz(
    base_dir, sample_tag,
    tat_only, gtu_only, mixed,
    sde_raw, sde_node_entropy, sde_node_inflow, sde_embedding,
    fs, label, window_signal
):
    """
    将所有过程数据分类保存为 .npz：
      base_dir/
        sequences/{tat_only,gtu_only,mixed}/{sample_tag}.npz    -> (N,T)
        sde/raw_sat_scores_seq/{sample_tag}.npz                 -> (T,K,N,N)
        sde/node_entropy_seq/{sample_tag}.npz                   -> (N,T)
        sde/node_inflow_seq/{sample_tag}.npz                    -> (N,T)
        sde/sde_embedding/{sample_tag}.npz                      -> (D,)
      额外保存 window_signal 供复现绘图（(T,C)，标准化后的窗口）
    """
    # 三类时间序列
    np.savez_compressed(
        os.path.join(base_dir, "sequences", "tat_only", f"{sample_tag}.npz"),
        seq=tat_only, fs=fs, label=label, window_signal=window_signal
    )
    np.savez_compressed(
        os.path.join(base_dir, "sequences", "gtu_only", f"{sample_tag}.npz"),
        seq=gtu_only, fs=fs, label=label, window_signal=window_signal
    )
    np.savez_compressed(
        os.path.join(base_dir, "sequences", "mixed", f"{sample_tag}.npz"),
        seq=mixed, fs=fs, label=label, window_signal=window_signal
    )
    # SDE 动态空间注意力与统计
    np.savez_compressed(
        os.path.join(base_dir, "sde", "raw_sat_scores_seq", f"{sample_tag}.npz"),
        sat_scores_seq=sde_raw, fs=fs, label=label
    )
    np.savez_compressed(
        os.path.join(base_dir, "sde", "node_entropy_seq", f"{sample_tag}.npz"),
        seq=sde_node_entropy, fs=fs, label=label
    )
    np.savez_compressed(
        os.path.join(base_dir, "sde", "node_inflow_seq", f"{sample_tag}.npz"),
        seq=sde_node_inflow, fs=fs, label=label
    )
    np.savez_compressed(
        os.path.join(base_dir, "sde", "sde_embedding", f"{sample_tag}.npz"),
        emb=sde_embedding, fs=fs, label=label
    )

def _zscore_along_time(x, eps=1e-6):
    # x: (T,) 一维
    mu = x.mean()
    std = x.std()
    std = std if std > eps else eps
    return (x - mu) / std

def plot_time_features(
    save_dir, sample_tag,
    window_signal,   # (T, C) —— 标准化后的 aECG 窗口
    seq_per_node,    # (N, T) —— 要画的“连续变化特征”之一（TAt-only / GTU-only / Mixed / SDE-Entropy / SDE-Inflow）
    fs, start_sec, end_sec,
    title_prefix="Feature"
):
    """
    生成一幅多子图（每导联一幅）：叠加原始信号(归一化)与指定的节点序列（归一化），
    并在标题标注窗口起止秒数。
    """
    ensure_dir(save_dir)
    T, C = window_signal.shape
    N, T2 = seq_per_node.shape
    assert T == T2, f"seq_per_node T={T2} 与窗口 T={T} 不一致"

    t_axis = np.arange(T) / float(fs) + start_sec  # [start_sec, end_sec]

    cols = 1
    rows = N
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.2 * rows), sharex=True)
    if N == 1:
        axes = [axes]

    for n in range(N):
        ax = axes[n]
        # 原始信号第 n 导联（若 C != N，优先截断到 N）
        ch_idx = n if n < C else C - 1
        sig = window_signal[:, ch_idx]
        sig_n = _zscore_along_time(sig)
        ax.plot(t_axis, sig_n, linewidth=1.0, label="aECG (z)")

        # 序列
        seq = seq_per_node[n, :]
        seq_n = _zscore_along_time(seq)
        ax.plot(t_axis, seq_n, linewidth=1.0, linestyle="--", label=f"{title_prefix} (z)")

        ax.set_ylabel(f"Lead {n+1}")
        ax.grid(True, linestyle=":")
        if n == 0:
            ax.set_title(f"{title_prefix} | Window: {start_sec:.2f}s → {end_sec:.2f}s (fs={fs}Hz)")
        if n == N - 1:
            ax.set_xlabel("Time (s)")

    # 统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])

    out_png = os.path.join(save_dir, f"{sample_tag}.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
