import math
import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data  # 确保和你的工程在同一目录或已在 PYTHONPATH 中


def build_continuous_segment(X, fs, seconds=60):
    """
    将多 trial 的数据按时间顺序拼接成一个连续片段，取前 seconds 秒。
    X: 形状 [n_trials, n_channels, n_samples_per_trial]
    返回: [n_channels, seconds * fs]
    """
    n_trials, n_ch, n_s = X.shape
    total_samples = n_trials * n_s
    needed_samples = seconds * fs

    if total_samples < needed_samples:
        raise ValueError(
            f"当前数据总长度不足 {seconds} 秒："
            f"{total_samples/fs:.2f} 秒 < {seconds} 秒，请减少 seconds 或使用更多数据。"
        )

    # 先把 trial 维度和时间维度展平，得到每个导联的一整段连续信号
    # [n_trials, n_channels, n_samples] -> [n_channels, n_trials * n_samples]
    continuous = X.transpose(1, 0, 2).reshape(n_ch, -1)

    # 取前 needed_samples 个采样点
    return continuous[:, :needed_samples]


def plot_eeg_60s(data_60s, fs, title_prefix="Train", channel_names=None):
    """
    data_60s: [n_channels, n_samples]
    fs: 采样率
    """
    n_ch, n_samples = data_60s.shape
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
    axes = np.array(axes).reshape(-1)

    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(time, data_60s[ch])
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

    fig.suptitle(f"{title_prefix} set - first 60 s ({n_ch} channels)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # ======= 1. 基本参数（你只需要改这几行） =======
    dataset_root = "/path/to/BCICIV_2a/"  # <-- 修改为你数据集根目录（包含 s1, s2, ...）
    subject = 1                           # <-- 要看的被试编号
    data_type = "2a"                      # <-- "2a" 或 "2b"

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

    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)

    # X 的形状应该是 [N_trial, N_channel, N_time]
    _, n_ch, T = X_train.shape

    # ======= 3. 决定采样率 fs =======
    # 在你的预处理代码中，MI 时间窗固定为 2~6 秒（共 4s）
    # T = 4 * fs -> fs = T / 4
    fs = int(T / 4)
    print("推算采样率 fs =", fs, "Hz")

    # ======= 4. 拼接出前 60 秒的连续数据 =======
    seconds = 60
    train_60s = build_continuous_segment(X_train, fs, seconds=seconds)
    test_60s = build_continuous_segment(X_test, fs, seconds=seconds)

    # ======= 5. 导联名字（可选，更好看一点） =======
    if data_type == "2a":
        # BCI IV 2a 的 22 个 EEG 通道
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz",
        ]
    elif data_type == "2b":
        # BCI IV 2b 的 3 个 EEG 通道
        ch_names = ["C3", "Cz", "C4"]
    else:
        ch_names = None

    # ======= 6. 画图：Train 前 60 秒 =======
    plot_eeg_60s(train_60s, fs, title_prefix="Train", channel_names=ch_names)

    # ======= 7. 画图：Test 前 60 秒 =======
    plot_eeg_60s(test_60s, fs, title_prefix="Test", channel_names=ch_names)


if __name__ == "__main__":
    main()
