import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

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


def plot_eeg_segment(data_seg, fs, title_prefix="Train", channel_names=None):
    """
    data_seg: [n_channels, n_samples]
    fs: 采样率
    """
    if data_seg.ndim != 2:
        raise ValueError(f"plot_eeg_segment 期望输入 [channels, time]，实际为 {data_seg.shape}")

    n_ch, n_samples = data_seg.shape
    time = np.arange(n_samples) / fs

    # 自动生成比较好看的子图布局
    if n_ch <= 4:
        n_cols = n_ch
    else:
        n_cols = 4  # 最多每行 4 个导联
    n_rows = math.ceil(n_ch / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows),
        sharex=True,
    )

    # 统一处理 axes 的形状（包括 n_ch==1 时）
    axes = np.array(axes).reshape(-1)

    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(time, data_seg[ch])

        if channel_names is not None and ch < len(channel_names):
            ch_name = channel_names[ch]
        else:
            ch_name = f"Ch {ch + 1}"

        # 把通道名放在 y 轴，横着写，更直观
        ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=20)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # 最后一行才画 x 轴标签
        if ch // n_cols == n_rows - 1:
            ax.set_xlabel("Time (s)")

    # 多出来的子图关掉坐标轴
    for ax in axes[n_ch:]:
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix} set - first {n_samples / fs:.1f} s ({n_ch} channels)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # ======= 1. 基本参数（你只需要改这几行） =======
    # 确保和训练时传给 get_data 的 path 保持一致
    dataset_root = r"/path/to/BCICIV_2a/"  # <-- 修改为你数据集根目录（包含 s1, s2, ...）
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
    #   2b: get_epochs_* 中 tmin=0, tmax=4，得到 4 秒数据:contentReference[oaicite:4]{index=4}
    if data_type in ["2a", "2b"]:
        mi_window_sec = 4.0  # 每个 trial 的长度（秒）
        fs = int(round(T / mi_window_sec))
    else:
        # 兜底：假设 fs=250（和 preprocess.py 里一致），同时给出提示
        fs = 250
        print(f"[警告] 未知 data_type={data_type}，默认 fs=250 Hz，请确认。")

    print(f"推算采样率 fs = {fs} Hz, 单 trial 长度约 {T / fs:.2f} 秒")

    # ======= 4. 拼接出前 60 秒的连续数据 =======
    seconds = 60
    train_seg = build_continuous_segment(X_train, fs, seconds=seconds)
    test_seg = build_continuous_segment(X_test, fs, seconds=seconds)

    # ======= 5. 导联名字（可选，更好看一点） =======
    ch_names = None
    if data_type == "2a" and n_ch == 22:
        # BCI IV 2a 的 22 个 EEG 通道
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz",
        ]
    elif data_type == "2b" and n_ch == 3:
        # BCI IV 2b 的 3 个 EEG 通道
        ch_names = ["C3", "Cz", "C4"]
    else:
        # 通道数对不上时，避免因为名字长度不匹配产生困惑
        print(f"[提示] 通道数为 {n_ch}，与预设的 2a/2b 通道数不一致，将使用默认 Ch1, Ch2, ... 命名。")
        ch_names = None

    # ======= 6. 画图：Train 前 60 秒 =======
    plot_eeg_segment(train_seg, fs, title_prefix="Train", channel_names=ch_names)

    # ======= 7. 画图：Test 前 60 秒 =======
    plot_eeg_segment(test_seg, fs, title_prefix="Test", channel_names=ch_names)


if __name__ == "__main__":
    main()
