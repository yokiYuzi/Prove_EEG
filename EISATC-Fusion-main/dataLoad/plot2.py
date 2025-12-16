import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline

# ==== 保证能找到 preprocess.py ====
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
if current_path not in sys.path:
    sys.path.append(current_path)
if root_path not in sys.path:
    sys.path.append(root_path)

from preprocess import get_data  # 使用你现有的数据处理流程


def build_continuous_segment(X, fs, seconds=60):
    """
    将多 trial 的数据按时间顺序拼接成一个连续片段，取前 seconds 秒。
    X: 形状 [n_trials, n_channels, n_samples_per_trial]
    返回: [n_channels, seconds * fs]
    """
    if X is None or len(X) == 0:
        raise ValueError("输入的 X 为空，检查 get_data 的返回值。")

    if X.ndim != 3:
        raise ValueError(f"期望 X 的形状为 [trial, channel, time]，实际为 {X.shape}")

    n_trials, n_ch, n_s = X.shape

    if fs <= 0:
        raise ValueError(f"采样率 fs 必须为正数，当前 fs = {fs}")

    total_samples = n_trials * n_s
    total_seconds = total_samples / fs
    needed_samples = int(seconds * fs)

    if total_samples < needed_samples:
        # 不够 60 秒时，直接用整个片段，并给出提示
        print(
            f"[警告] 当前数据总长度只有 {total_seconds:.2f} 秒，"
            f"不足 {seconds} 秒，将使用全部 {total_seconds:.2f} 秒的数据。"
        )
        needed_samples = total_samples

    # [n_trials, n_channels, n_samples] -> [n_channels, n_trials * n_samples]
    continuous = X.transpose(1, 0, 2).reshape(n_ch, -1)

    return continuous[:, :needed_samples]


def find_extrema(t, s):
    """Find local maxima and minima."""
    diff = np.diff(s)
    maxima = (diff[:-1] > 0) & (diff[1:] < 0)
    minima = (diff[:-1] < 0) & (diff[1:] > 0)
    maxima_idx = np.where(maxima)[0] + 1
    minima_idx = np.where(minima)[0] + 1
    return maxima_idx, minima_idx


def emd(s, max_imf=10, tolerance=0.05, max_iter=2000):
    """Simple implementation of Empirical Mode Decomposition (EMD)."""
    imfs = []
    r = s.copy()
    for i in range(max_imf):
        h = r.copy()
        for j in range(max_iter):
            max_idx, min_idx = find_extrema(np.arange(len(h)), h)
            if len(max_idx) < 2 or len(min_idx) < 2:
                break  # Not enough extrema
            upper = CubicSpline(max_idx, h[max_idx])(np.arange(len(h)))
            lower = CubicSpline(min_idx, h[min_idx])(np.arange(len(h)))
            mean = (upper + lower) / 2
            prev_h = h.copy()
            h -= mean
            # Check stopping criterion
            if np.sum((prev_h - h)**2) / np.sum(prev_h**2) < tolerance:
                break
        imfs.append(h)
        r -= h
        # Stop if residue is monotonic or constant
        if len(find_extrema(np.arange(len(r)), r)[0]) <= 1 and len(find_extrema(np.arange(len(r)), r)[1]) <= 1:
            break
    imfs.append(r)  # Add residue
    return np.array(imfs)


def plot_emd(imfs, t, fs, channel_name="Ch1"):
    """Plot EMD results (IMFs)."""
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs, 1, figsize=(10, 2 * n_imfs), sharex=True)
    for i in range(n_imfs):
        if i == n_imfs - 1:
            axes[i].plot(t, imfs[i])
            axes[i].set_title(f"Residue")
        else:
            axes[i].plot(t, imfs[i])
            axes[i].set_title(f"IMF {i+1}")
        axes[i].grid(True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EMD Decomposition - {channel_name} ({len(t)/fs:.1f} s)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def hht(imfs, fs):
    """Hilbert-Huang Transform: Compute instantaneous frequency and amplitude."""
    analytic_signals = [hilbert(imf) for imf in imfs[:-1]]  # Exclude residue
    amplitudes = [np.abs(z) for z in analytic_signals]
    phases = [np.unwrap(np.angle(z)) for z in analytic_signals]
    inst_freqs = [np.diff(p) / (2 * np.pi) * fs for p in phases]  # Diff reduces length by 1
    return amplitudes, inst_freqs


def plot_hht(amplitudes, inst_freqs, t, channel_name="Ch1"):
    """Plot Hilbert Spectrum (time-frequency-amplitude)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(amplitudes)):
        # Inst freq has one less point, so trim t and amp
        t_trim = t[1:]
        amp = amplitudes[i][1:]
        freq = inst_freqs[i]
        # Only plot positive frequencies
        mask = freq > 0
        ax.scatter(t_trim[mask], freq[mask], c=amp[mask], cmap='viridis', s=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Hilbert Spectrum - {channel_name}")
    ax.grid(True)
    fig.colorbar(ax.collections[0], ax=ax, label="Amplitude")
    fig.tight_layout()
    return fig


def plot_fft(signal, fs, channel_name="Ch1"):
    """Plot Fourier Transform (power spectrum)."""
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals)**2 / n
    # Only positive frequencies
    mask = freq > 0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq[mask], power[mask])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Fourier Power Spectrum - {channel_name}")
    ax.grid(True)
    fig.tight_layout()
    return fig


def main():
    # ======= 1. 基本参数（你只需要改这几行） =======
    # 确保和训练时传给 get_data 的 path 保持一致
    dataset_root = r"G:/Prove_EEG/EISATC-Fusion-main/dataLoad/BCICIV_2a/"  # 使用前向斜杠并添加尾部斜杠以修正路径拼接问题
    subject = 1                             # <-- 要看的被试编号
    data_type = "2a"                        # <-- "2a" 或 "2b"

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"数据根目录不存在，请检查 dataset_root: {dataset_root}")

    # 是否标准化：
    #   False: 保留物理量级（2a 中为 μV），更适合看“物理特性”
    #   True : 使用你现在流程中的 StandardScaler 标准化
    is_standard = False

    # ======= 2. 利用你现有的 get_data 读取数据 =======
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

    if X_train is None or len(X_train) == 0:
        raise RuntimeError("X_train 为空，请检查路径 / subject / data_type 参数是否和训练时一致。")
    if X_test is None or len(X_test) == 0:
        raise RuntimeError("X_test 为空，请确认该被试确实有测试数据。")

    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)

    # X 的形状应该是 [N_trial, N_channel, N_time]
    if X_train.ndim != 3:
        raise ValueError(f"X_train 期望为 3 维 [trial, channel, time]，实际为 {X_train.shape}")

    _, n_ch, T = X_train.shape

    # ======= 3. 决定采样率 fs =======
    # 按你当前流程：
    #   2a: load_data_2a 取 7s 窗，但在 get_data 中裁成 2~6s（4 秒）
    #   2b: get_epochs_* 中 tmin=0, tmax=4，得到 4 秒数据
    if data_type in ["2a", "2b"]:
        mi_window_sec = 4  # 每个 trial 的长度（秒），修正为4秒基于注释
        fs = int(round(T / mi_window_sec))
    else:
        # 兜底：假设 fs=250（和 preprocess.py 里一致），同时给出提示
        fs = 250
        print(f"[警告] 未知 data_type={data_type}，默认 fs=250 Hz，请确认。")

    print(f"推算采样率 fs = {fs} Hz, 单 trial 长度约 {T / fs:.2f} 秒")

    # ======= 4. 拼接出前 60 秒的连续数据 =======
    seconds = 60
    train_seg = build_continuous_segment(X_train, fs, seconds=seconds)

    # ======= 5. 选择一个通道作为示例（这里用第一个通道） =======
    channel_idx = 0
    signal = train_seg[channel_idx]
    t = np.arange(len(signal)) / fs
    channel_name = f"Ch {channel_idx + 1}"  # 可以替换为 ch_names[0] 如果有

    # ======= 添加保存功能：创建新的文件夹 =======
    output_dir = os.path.join(current_path, "transform_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ======= 6. EMD 分解 =======
    imfs = emd(signal)
    fig_emd = plot_emd(imfs, t, fs, channel_name)
    emd_save_path = os.path.join(output_dir, f"emd_subject_{subject}.png")
    fig_emd.savefig(emd_save_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图像
    plt.close(fig_emd)

    # ======= 7. HHT 变换 =======
    amplitudes, inst_freqs = hht(imfs, fs)
    fig_hht = plot_hht(amplitudes, inst_freqs, t, channel_name)
    hht_save_path = os.path.join(output_dir, f"hht_subject_{subject}.png")
    fig_hht.savefig(hht_save_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图像
    plt.close(fig_hht)

    # ======= 8. 傅里叶变换 =======
    fig_fft = plot_fft(signal, fs, channel_name)
    fft_save_path = os.path.join(output_dir, f"fft_subject_{subject}.png")
    fig_fft.savefig(fft_save_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图像
    plt.close(fig_fft)

    print(f"图像已保存至: {output_dir}")


if __name__ == "__main__":
    main()