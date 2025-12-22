#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接可运行：EEG Filter Bank 分解 + 展示/保存“前 0.25s”的分解结果

说明：
1) 优先使用你项目里现有的 preprocess.get_data 读取 BCICIV_2a/2b 数据；
2) 为避免 sosfiltfilt 在超短片段(0.25s)上报 padlen 错误，本脚本会：
   - 先取更长的一段(默认 8s)做滤波
   - 再裁剪出前 0.25s 用于展示/保存
3) 输出：
   - raw_allch_first0.25s.png：原始(所有导联)前0.25s
   - band_*Hz_allch_first0.25s.png：各频带(所有导联)前0.25s
   - decomp_singlech_first0.25s.png：单导联 raw + 各频带分解(前0.25s)

依赖：
- numpy, matplotlib
- scipy(推荐)：用于 butter + sosfiltfilt；若没有 scipy，会自动退化到 FFT 理想带通(也能跑)

把 dataset_root 改成你本机数据路径即可直接运行。
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# ============ 可选：SciPy（推荐）============
try:
    from scipy.signal import butter, sosfiltfilt
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ============ 保证能找到 preprocess.py（沿用你原脚本的做法）============
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
if current_path not in sys.path:
    sys.path.append(current_path)
if root_path not in sys.path:
    sys.path.append(root_path)

# 只有当数据目录存在时才 import（否则给你 demo 也能直接跑）
_GET_DATA_OK = False
try:
    from preprocess import get_data
    _GET_DATA_OK = True
except Exception:
    _GET_DATA_OK = False


def parse_bands(bands_str: str):
    """
    解析频带字符串，例如 "0.5-4,4-8,8-13,13-30"
    返回 list[(lo, hi), ...]
    """
    out = []
    chunks = [c.strip() for c in bands_str.split(",") if c.strip()]
    for c in chunks:
        if "-" not in c:
            raise ValueError(f"频带格式错误：{c}，请用 low-high，例如 8-13")
        lo_s, hi_s = c.split("-", 1)
        lo = float(lo_s.strip())
        hi = float(hi_s.strip())
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError(f"频带上下限不合法：{lo}-{hi}")
        out.append((lo, hi))
    if not out:
        raise ValueError("未解析到任何频带，请检查 bands_str")
    return out


def build_continuous_segment(X, fs, seconds=8.0):
    """
    将多 trial 数据按时间顺序拼成一个连续片段，取前 seconds 秒。
    X: [n_trials, n_channels, n_samples_per_trial]
    返回: [n_channels, seconds * fs]
    """
    if X is None or len(X) == 0:
        raise ValueError("输入 X 为空，检查 get_data 的返回值。")
    if X.ndim != 3:
        raise ValueError(f"期望 X 的形状为 [trial, channel, time]，实际为 {X.shape}")

    n_trials, n_ch, n_s = X.shape
    if fs <= 0:
        raise ValueError(f"采样率 fs 必须为正数，当前 fs={fs}")

    total_samples = n_trials * n_s
    needed_samples = int(round(seconds * fs))
    if total_samples < needed_samples:
        total_seconds = total_samples / fs
        print(f"[警告] 数据总长仅 {total_seconds:.2f}s < {seconds:.2f}s，将使用全部数据。")
        needed_samples = total_samples

    # [trial, ch, time] -> [ch, trial*time]
    continuous = X.transpose(1, 0, 2).reshape(n_ch, -1)
    return continuous[:, :needed_samples]


def _filter_bank_scipy(data_chxt, fs, bands, order=4):
    """
    SciPy 版带通滤波器组
    data_chxt: [n_channels, n_samples]
    return: [n_bands, n_channels, n_samples]
    """
    nyq = fs / 2.0
    for lo, hi in bands:
        if hi >= nyq:
            raise ValueError(f"频带 {lo}-{hi}Hz 上限必须 < Nyquist={nyq}Hz。")

    data = np.asarray(data_chxt, dtype=np.float32)
    n_ch, n_s = data.shape

    out = np.zeros((len(bands), n_ch, n_s), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        # 注意：这里对“更长片段”做 filtfilt，避免 0.25s 太短导致 padlen 报错
        out[bi] = sosfiltfilt(sos, data, axis=-1).astype(np.float32)
    return out


def _filter_bank_fft_ideal(data_chxt, fs, bands):
    """
    无 SciPy 时的退化方案：FFT 理想带通（可运行，但边缘振铃更明显）
    data_chxt: [n_channels, n_samples]
    return: [n_bands, n_channels, n_samples]
    """
    data = np.asarray(data_chxt, dtype=np.float32)
    n_ch, n_s = data.shape

    freqs = np.fft.rfftfreq(n_s, d=1.0 / fs)
    Xf = np.fft.rfft(data, axis=-1)  # [ch, F]

    out = np.zeros((len(bands), n_ch, n_s), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi)
        Yf = Xf * mask[None, :]
        y = np.fft.irfft(Yf, n=n_s, axis=-1)
        out[bi] = y.astype(np.float32)
    return out


def filter_bank_multichannel(data_chxt, fs, bands, order=4):
    """
    data_chxt: [n_channels, n_samples]
    return: [n_bands, n_channels, n_samples]
    """
    if _SCIPY_OK:
        return _filter_bank_scipy(data_chxt, fs, bands, order=order)
    print("[警告] 未检测到 scipy，将使用 FFT 理想带通作为退化方案（仍可运行）。")
    return _filter_bank_fft_ideal(data_chxt, fs, bands)


def plot_eeg_segment_grid(data_seg, fs, title, channel_names=None):
    """
    data_seg: [n_channels, n_samples]
    返回 fig
    """
    if data_seg.ndim != 2:
        raise ValueError(f"期望 [channels, time]，实际 {data_seg.shape}")

    n_ch, n_s = data_seg.shape
    t = np.arange(n_s) / fs

    n_cols = 4 if n_ch > 4 else max(1, n_ch)
    n_rows = math.ceil(n_ch / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 2.3 * n_rows),
        sharex=True
    )
    axes = np.array(axes).reshape(-1)

    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(t, data_seg[ch])
        name = (channel_names[ch] if (channel_names is not None and ch < len(channel_names)) else f"Ch{ch+1}")
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=18)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if ch // n_cols == n_rows - 1:
            ax.set_xlabel("Time (s)")

    for ax in axes[n_ch:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_single_channel_decomp(raw_1d, band_2d, fs, bands, ch_name="C3"):
    """
    raw_1d: [n_samples]
    band_2d: [n_bands, n_samples]
    返回 fig（纵向子图：raw + 每个 band）
    """
    n_bands, n_s = band_2d.shape
    t = np.arange(n_s) / fs

    fig, axes = plt.subplots(n_bands + 1, 1, figsize=(12, 2.2 * (n_bands + 1)), sharex=True)
    axes = np.array(axes).reshape(-1)

    axes[0].plot(t, raw_1d)
    axes[0].set_title(f"{ch_name} - Raw (first {t[-1]:.3f}s)")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    for i, (lo, hi) in enumerate(bands):
        axes[i + 1].plot(t, band_2d[i])
        axes[i + 1].set_title(f"{ch_name} - Bandpass {lo:g}-{hi:g} Hz (first {t[-1]:.3f}s)")
        axes[i + 1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def get_channel_names(data_type: str, n_ch: int):
    if data_type == "2a" and n_ch == 22:
        return [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz",
        ]
    if data_type == "2b" and n_ch == 3:
        return ["C3", "Cz", "C4"]
    return None


def generate_demo_multichannel(fs=250, seconds=8.0, n_ch=22, seed=0):
    """
    用于“没有数据也能直接跑”的 demo：
    每个通道混合 δ/θ/α/β + 少量噪声
    返回：
      X_demo: [n_trials=1, n_ch, n_samples]
    """
    rng = np.random.default_rng(seed)
    n_s = int(round(seconds * fs))
    t = np.arange(n_s) / fs

    freqs = [2.0, 6.0, 10.0, 20.0]  # δ/θ/α/β
    amps = [1.0, 0.6, 0.4, 0.3]

    data = np.zeros((n_ch, n_s), dtype=np.float32)
    for ch in range(n_ch):
        phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
        sig = np.zeros(n_s, dtype=np.float32)
        for (f, a, p) in zip(freqs, amps, phase):
            sig += a * np.sin(2 * np.pi * f * t + p).astype(np.float32)
        sig += 0.2 * rng.standard_normal(n_s).astype(np.float32)
        data[ch] = sig

    return data[None, :, :]  # [1, n_ch, n_s]


def main():
    # ===================== 你只需要改这里 =====================
    dataset_root = r"C:/Prove_EEG/EISATC-Fusion-main/dataLoad/BCICIV_2a/"
    subject = 1
    data_type = "2a"          # "2a" or "2b"
    is_standard = False       # False:保留物理量级，更适合观察波形
    fs = 250                  # BCICIV_2a/2b 通常是 250Hz（不建议再像之前那样从 T 反推）

    # 你要观察的“前 0.25s”
    plot_seconds = 1

    # 为避免 0.25s 太短导致 filtfilt 报错：先拿更长片段做滤波，再裁剪前0.25s展示
    filter_seconds = 8.0

    # 频带（可自行调整）
    bands_str = "0.5-4,4-8,8-13,13-30"
    filter_order = 4

    # 想重点看的单导联（优先按名字找；找不到就用 0 号通道）
    target_channel_name = "C3"

    # 输出目录（自动创建）
    output_dir = os.path.join(current_path, "filterbank_plots_first0p25s")
    # ==========================================================

    os.makedirs(output_dir, exist_ok=True)
    bands = parse_bands(bands_str)

    # ============ 读取数据（若找不到数据就跑 demo，也能直接运行）============
    use_demo = True
    X_train = None

    if os.path.isdir(dataset_root) and _GET_DATA_OK:
        try:
            X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = get_data(
                path=dataset_root,
                subject=subject,
                LOSO=False,
                Transfer=False,
                onLine_2a=False,
                data_model="one_session",
                isStandard=is_standard,
                data_type=data_type,
            )
            if X_train is not None and len(X_train) > 0 and X_train.ndim == 3:
                use_demo = False
        except Exception as e:
            print(f"[警告] get_data 读取失败，将使用 demo 信号。错误：{repr(e)}")

    if use_demo:
        print("[INFO] 使用 demo 多通道信号（你仍可直接观察 filter bank 分解图）。")
        n_ch = 22 if data_type == "2a" else 3
        X_train = generate_demo_multichannel(fs=fs, seconds=max(filter_seconds, 8.0), n_ch=n_ch, seed=0)

    n_trials, n_ch, T = X_train.shape
    ch_names = get_channel_names(data_type, n_ch)

    print(f"[INFO] X_train shape = {X_train.shape}, fs={fs}Hz, scipy={'YES' if _SCIPY_OK else 'NO'}")
    print(f"[INFO] 频带 = {bands}")

    # ============ 取连续片段（更长，用于滤波）============
    long_seg = build_continuous_segment(X_train, fs, seconds=filter_seconds)  # [ch, n_samples_long]
    n_s_long = long_seg.shape[1]

    plot_samples = int(round(plot_seconds * fs))
    if plot_samples < 2:
        raise ValueError("plot_seconds 太小，导致采样点不足。")

    if plot_samples > n_s_long:
        print("[警告] 可用长度不足 plot_seconds，将使用全部可用长度绘制。")
        plot_samples = n_s_long

    raw_plot = long_seg[:, :plot_samples]  # [ch, plot_samples]

    # ============ Filter Bank 分解（对 long_seg 做滤波，再裁剪前0.25s展示）============
    fb_long = filter_bank_multichannel(long_seg, fs, bands, order=filter_order)  # [band, ch, n_samples_long]
    fb_plot = fb_long[:, :, :plot_samples]                                       # [band, ch, plot_samples]

    # ============ 保存：原始(所有导联)前0.25s ============
    fig_raw = plot_eeg_segment_grid(
        raw_plot, fs,
        title=f"RAW - first {plot_seconds:.2f}s (subject={subject}, {data_type}, fs={fs}Hz)",
        channel_names=ch_names
    )
    raw_path = os.path.join(output_dir, f"raw_allch_first{plot_seconds:.2f}s.png")
    fig_raw.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close(fig_raw)
    print(f"[SAVE] {raw_path}")

    # ============ 保存：各频带(所有导联)前0.25s ============
    for bi, (lo, hi) in enumerate(bands):
        fig_band = plot_eeg_segment_grid(
            fb_plot[bi], fs,
            title=f"Band {lo:g}-{hi:g} Hz - first {plot_seconds:.2f}s (subject={subject}, {data_type})",
            channel_names=ch_names
        )
        band_path = os.path.join(output_dir, f"band_{lo:g}-{hi:g}Hz_allch_first{plot_seconds:.2f}s.png")
        fig_band.savefig(band_path, dpi=300, bbox_inches="tight")
        plt.close(fig_band)
        print(f"[SAVE] {band_path}")

    # ============ 单导联：raw + 每个 band（前0.25s） ============
    if ch_names is not None and target_channel_name in ch_names:
        ch_idx = ch_names.index(target_channel_name)
        ch_name = target_channel_name
    else:
        ch_idx = 0
        ch_name = (ch_names[0] if ch_names else "Ch1")
        if ch_names is None or target_channel_name not in (ch_names or []):
            print(f"[提示] 未找到通道名 {target_channel_name}，将使用 {ch_name} (index=0) 进行单导联展示。")

    fig_decomp = plot_single_channel_decomp(
        raw_plot[ch_idx],
        fb_plot[:, ch_idx, :],
        fs=fs,
        bands=bands,
        ch_name=ch_name
    )
    decomp_path = os.path.join(output_dir, f"decomp_singlech_{ch_name}_first{plot_seconds:.2f}s.png")
    fig_decomp.savefig(decomp_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {decomp_path}")

    # 展示：只弹出最关键的单导联分解图（避免弹太多窗口）
    plt.show()
    plt.close(fig_decomp)

    print(f"\n[DONE] 全部图像已保存到：{output_dir}")


if __name__ == "__main__":
    main()
