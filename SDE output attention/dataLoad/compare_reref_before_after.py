# -*- coding: utf-8 -*-
"""
compare_reref_before_after.py

功能：
1) 加载 BCICIV-2a 的 EEG 数据（默认取 2~6s 片段，长度 4s）
2) 对比 “原始信号” vs “重参考信号(相减)：X' = X - X_ref”
3) 展示并保存：前 0.5 / 1 / 2 / 3 / 4 秒 的处理前后对比图（PNG）

使用方式（推荐）：
- 把本脚本放在你的工程根目录（与 dataLoad/ 同级），例如：
    project/
      compare_reref_before_after.py   <-- 本脚本
      dataLoad/
        LoadData.py
        preprocess_reref.py
        BCICIV_2a/
          s1/A01T.mat ...
- 然后运行：
    python compare_reref_before_after.py --data_root ./dataLoad/BCICIV_2a --subject 1 --split train --trial 0

重要注意（通道顺序）：
BCICIV-2a 22 导的“官方常见顺序”通常是包含 C5/C6 且不含 P3/P4：
    Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
但你之前脚本里也出现过包含 P3/P4 的版本。
为了避免“ref_channel 位置错位”，本脚本提供 --ch_order 参数：
    --ch_order c56  (默认：包含 C5/C6)
    --ch_order p34  (包含 P3/P4 的旧顺序)
你必须选择与你实际数据矩阵前 22 列一致的顺序，否则会减错参考通道。

如果你不确定，就先用默认 c56 试一下，并把 PLOT_CHANNELS 里包含 "Cz"：
- 正确情况下：重参考后 Cz 通道应该接近全 0（数值误差范围内）。
"""

import os
import sys
import argparse
import numpy as np

# ---- 适配无显示环境（服务器/SSH）----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 默认参数
# =========================
FS = 250  # BCICIV-2a 采样率
WINDOW_SECONDS = [0.5, 1, 2, 3, 4]  # 你要求展示的前 N 秒
DEFAULT_OUT_DIRNAME = "reref_compare_figures"

# 你可以在不传 --channels 时修改这里：默认画这几个导联
DEFAULT_PLOT_CHANNELS = ["C3", "Cz", "CP2", "FC1", "C4", "P1", "FC2", "C1"]


# =========================
# 路径辅助：让脚本放在「工程根目录」或「dataLoad/」都能跑
# =========================
def _guess_project_root_and_data_root():
    """返回 (project_root, default_data_root)。

    - 若脚本位于 .../dataLoad/compare_reref_before_after.py：
        project_root = 上一级目录
        default_data_root = .../dataLoad/BCICIV_2a
    - 若脚本位于工程根目录：
        project_root = 当前目录
        default_data_root = .../dataLoad/BCICIV_2a
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() == "dataload":
        project_root = os.path.dirname(script_dir)
        default_data_root = os.path.join(script_dir, "BCICIV_2a")
    else:
        project_root = script_dir
        default_data_root = os.path.join(script_dir, "dataLoad", "BCICIV_2a")
    return project_root, default_data_root


# =========================
# 两种常见的 22 导通道顺序（你必须选对）
# =========================
CH_NAMES_22_C56 = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]

CH_NAMES_22_P34 = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P3", "P1", "Pz", "P2", "P4",
    "POz",
]


# =========================
# 工具：解析通道列表
# =========================
def parse_channels_arg(ch_str: str):
    if ch_str is None or str(ch_str).strip() == "":
        return None
    return [c.strip() for c in ch_str.split(",") if c.strip()]


def resolve_ref_index(ref_channel, ch_names):
    if isinstance(ref_channel, int):
        if ref_channel < 0 or ref_channel >= len(ch_names):
            raise ValueError(f"ref_channel 索引越界: {ref_channel}, 通道数={len(ch_names)}")
        return ref_channel
    if isinstance(ref_channel, str):
        lower_map = {c.lower(): i for i, c in enumerate(ch_names)}
        key = ref_channel.lower()
        if key not in lower_map:
            raise ValueError(f"ref_channel='{ref_channel}' 不在通道名列表里: {ch_names}")
        return int(lower_map[key])
    raise TypeError("ref_channel 必须是 int 或 str")


# =========================
# 核心：重参考（相减）
# =========================
def rereference_to_channel_np(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    X: (Trials, Channels, Time)
    返回: X' = X - X_ref
    """
    if X.ndim != 3:
        raise ValueError(f"X 必须是 3D (Trials, Channels, Time)，但收到: {X.shape}")
    ref = X[:, ref_idx:ref_idx + 1, :]
    return X - ref


# =========================
# 数据加载（优先用你的 preprocess_reref.py；若导入失败则 fallback 直接读 .mat）
# =========================
def try_import_preprocess_reref():
    """
    尝试导入 dataLoad/preprocess_reref.py
    返回 (get_data_func, ok, err_msg)
    """
    err_msg = None
    # 让脚本放在工程根目录或 dataLoad/ 都能 import
    project_root, _ = _guess_project_root_and_data_root()
    dataLoad_dir = os.path.join(project_root, "dataLoad")

    # 1) 工程根目录：用于 import dataLoad.xxx
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # 2) dataLoad 目录：用于 import preprocess_reref
    if dataLoad_dir not in sys.path:
        sys.path.insert(0, dataLoad_dir)

    try:
        # 方式1：工程里常见写法
        from dataLoad.preprocess_reref import get_data as get_data_func
        return get_data_func, True, None
    except Exception as e1:
        err_msg = f"[import fail] dataLoad.preprocess_reref: {repr(e1)}"

    try:
        # 方式2：脚本就在 dataLoad 目录或已加到 sys.path
        from preprocess_reref import get_data as get_data_func
        return get_data_func, True, None
    except Exception as e2:
        err_msg = err_msg + f"\n[import fail] preprocess_reref: {repr(e2)}"
        return None, False, err_msg


def load_with_preprocess_reref(
    data_root: str,
    subject: int,
    rereference: bool,
    ref_channel,
    ch_names,
    split: str = "train",
):
    """
    使用 preprocess_reref.get_data 读取：
      - isStandard=False（只看“相减”效果，不要标准化影响）
      - data_type='2a'
      - rereference 参数控制是否做相减

    返回:
      X_split: (Trials, Channels, Time)  其中 Time 默认是 2~6s => 4s => 1000点
      y_split: (Trials,)
    """
    get_data_func, ok, err = try_import_preprocess_reref()
    if not ok:
        raise RuntimeError(
            "无法导入 preprocess_reref.py。\n"
            "你可以：\n"
            "1) 确认本脚本与 dataLoad/ 同级；\n"
            "2) 确认 dataLoad/preprocess_reref.py 存在；\n"
            "3) 或者改用 fallback（本脚本会自动 fallback，但这里导入失败会直接抛错）。\n"
            f"详细错误：\n{err}"
        )

    X_train, y_train, X_test, y_test, _, _ = get_data_func(
        path=os.path.abspath(data_root) + os.sep,
        subject=int(subject),
        LOSO=False,
        Transfer=False,
        onLine_2a=False,
        data_model="one_session",
        isStandard=False,            # 关键：不要标准化
        data_type="2a",
        standardize_mode="channel_global",
        rereference=bool(rereference),
        ref_channel=ref_channel,
        drop_ref=False,              # 不删除参考通道，便于验证 Cz 是否变 0
        ch_names=ch_names,
        return_ch_names=False,
    )

    if split.lower() == "train":
        return X_train, y_train
    if split.lower() == "test":
        return X_test, y_test
    raise ValueError("--split 只能是 train 或 test")


# ---- fallback：直接读取 .mat（不依赖 mne）----
def load_data_2a_mat(data_path: str, subject: int, training: bool, all_trials: bool = True):
    """
    复制自 LoadData.py 的 load_data_2a（仅保留 .mat 读取逻辑，不依赖 mne）。
    返回:
      data_return: (Trials, 22, 7*250)
      class_return: (Trials,)
    """
    import scipy.io as scio

    n_channels = 22
    n_tests = 6 * 48
    window_length = 7 * 250

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))

    NO_valid_trial = 0
    if training:
        a = scio.loadmat(os.path.join(data_path, "A0" + str(subject) + "T.mat"))
    else:
        a = scio.loadmat(os.path.join(data_path, "A0" + str(subject) + "E.mat"))

    a_data = a["data"]
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0) and (not all_trials):
                continue
            seg = a_X[int(a_trial[trial]): (int(a_trial[trial]) + window_length), :22]  # (T,22)
            data_return[NO_valid_trial, :, :] = np.transpose(seg)  # (22,T)
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


def load_with_fallback_mat(
    data_root: str,
    subject: int,
    split: str,
):
    """
    fallback 读取 .mat，并裁剪到 2~6s（4s）。
    """
    # data_root 结构：.../BCICIV_2a/
    # 里面有 s1/ s2/ ...
    sub_dir = os.path.join(os.path.abspath(data_root), f"s{subject}")

    if split.lower() == "train":
        X, y = load_data_2a_mat(sub_dir, subject, training=True, all_trials=True)
    elif split.lower() == "test":
        X, y = load_data_2a_mat(sub_dir, subject, training=False, all_trials=True)
    else:
        raise ValueError("--split 只能是 train 或 test")

    # 裁剪 2~6s
    t1 = int(2 * FS)
    t2 = int(6 * FS)
    X = X[:, :, t1:t2]  # (Trials, 22, 1000)

    # 标签转 0~3
    y = y.astype(np.int64) - 1
    return X, y


# =========================
# 绘图与保存
# =========================
def plot_and_save_one_channel(
    raw_full: np.ndarray,
    reref_full: np.ndarray,
    ref_full: np.ndarray,
    fs: int,
    win_seconds_list,
    subject: int,
    split: str,
    trial_idx: int,
    ch_name: str,
    ref_name: str,
    label: int,
    out_path: str,
    show_ref_signal: bool = True,
):
    """
    raw_full, reref_full, ref_full: shape (T,)
    保存一张图：5行子图，对应前0.5/1/2/3/4秒
    """
    T = raw_full.shape[0]
    max_sec = T / fs
    wins = [w for w in win_seconds_list if w <= max_sec + 1e-9]
    if len(wins) == 0:
        raise ValueError(f"数据长度只有 {max_sec:.3f}s，无法画任何窗口 {win_seconds_list}")

    # 统一 y 轴范围（用全 4 秒范围决定，更便于对比）
    y_min = float(min(raw_full.min(), reref_full.min(), ref_full.min() if show_ref_signal else raw_full.min()))
    y_max = float(max(raw_full.max(), reref_full.max(), ref_full.max() if show_ref_signal else raw_full.max()))
    pad = 0.05 * (y_max - y_min + 1e-12)
    y_min -= pad
    y_max += pad

    fig, axes = plt.subplots(nrows=len(wins), ncols=1, figsize=(12, 2.6 * len(wins)), dpi=140)
    if len(wins) == 1:
        axes = [axes]

    for ax, w in zip(axes, wins):
        n = int(round(w * fs))
        n = min(n, T)

        t = np.arange(n) / fs
        ax.plot(t, raw_full[:n], label="Raw")
        ax.plot(t, reref_full[:n], label="ReRef (Raw - Ref)")
        if show_ref_signal:
            ax.plot(t, ref_full[:n], label=f"Ref({ref_name})", linestyle="--", alpha=0.8)

        ax.set_xlim(0, w)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        ax.set_title(f"First {w:g}s | Ch={ch_name} | Ref={ref_name} | sub={subject} | {split} | trial={trial_idx} | label={label}")
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    _, default_data_root = _guess_project_root_and_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=default_data_root,
                        help="BCICIV_2a 根目录（里面包含 s1/s2/...）")
    parser.add_argument("--subject", type=int, default=1, help="被试编号 1~9")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="画训练集还是测试集")
    parser.add_argument("--trial", type=int, default=0, help="要画第几个 trial（从 0 开始）")

    parser.add_argument("--ref_channel", type=str, default="Cz", help="参考通道名（默认 Cz）")
    parser.add_argument("--ch_order", type=str, default="c56", choices=["c56", "p34"],
                        help="22导通道顺序：c56(含C5/C6) 或 p34(含P3/P4)")
    parser.add_argument("--channels", type=str, default=",".join(DEFAULT_PLOT_CHANNELS),
                        help="要画的通道名（逗号分隔），例如: C3,Cz,C4 或 CP3,C3,CP4,FC1,C4,P1,FC2,C1")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIRNAME,
                        help="图片保存目录（会自动创建）")
    parser.add_argument("--show_ref", action="store_true", help="额外把参考通道波形也画出来（建议打开便于验证）")

    args = parser.parse_args()

    # 选通道顺序
    if args.ch_order.lower() == "c56":
        ch_names = list(CH_NAMES_22_C56)
    else:
        ch_names = list(CH_NAMES_22_P34)

    plot_channels = parse_channels_arg(args.channels) or list(DEFAULT_PLOT_CHANNELS)

    # 解析 ref 索引
    ref_idx = resolve_ref_index(args.ref_channel, ch_names)

    # 输出目录
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------
    # 1) 读取原始信号 raw（不做 rereference）
    # 2) 读取处理后信号 reref（做 rereference）
    # ---------------------------
    # 优先用 preprocess_reref.py；如失败则 fallback 直接读 .mat
    use_fallback = False
    try:
        X_raw, y_raw = load_with_preprocess_reref(
            data_root=args.data_root,
            subject=args.subject,
            rereference=False,
            ref_channel=args.ref_channel,
            ch_names=ch_names,
            split=args.split,
        )
        X_reref, y_reref = load_with_preprocess_reref(
            data_root=args.data_root,
            subject=args.subject,
            rereference=True,
            ref_channel=args.ref_channel,
            ch_names=ch_names,
            split=args.split,
        )
    except Exception as e:
        print("[WARN] preprocess_reref 导入/读取失败，将使用 fallback 直接读 .mat。")
        print("原因：", repr(e))
        use_fallback = True
        X_raw, y_raw = load_with_fallback_mat(args.data_root, args.subject, args.split)
        # fallback 下：手动做 rereference
        X_reref = rereference_to_channel_np(X_raw, ref_idx)
        y_reref = y_raw

    # 基本检查
    if X_raw.shape != X_reref.shape:
        raise RuntimeError(f"raw 和 reref shape 不一致：raw={X_raw.shape}, reref={X_reref.shape}")

    n_trials, n_ch, n_t = X_raw.shape
    if args.trial < 0 or args.trial >= n_trials:
        raise ValueError(f"--trial={args.trial} 越界：trial总数={n_trials}")

    # 取指定 trial
    trial_idx = int(args.trial)
    label = int(y_raw[trial_idx])

    # 参考通道原始波形（用于展示被减去的内容）
    ref_full = X_raw[trial_idx, ref_idx, :]

    # ---------------------------
    # 逐通道画图并保存
    # ---------------------------
    print("=" * 70)
    print("数据读取方式：", "fallback(.mat)" if use_fallback else "preprocess_reref.get_data()")
    print(f"Subject={args.subject} | Split={args.split} | Trial={trial_idx} | Label={label}")
    print(f"Data shape: {X_raw.shape}  (Trials, Channels, Time)  Time={n_t/FS:.2f}s")
    print(f"Ref channel: {args.ref_channel} (idx={ref_idx})")
    print(f"Plot channels: {plot_channels}")
    print(f"Output dir: {out_dir}")
    print("=" * 70)

    # 友情提示：如果你把 Cz 也画出来，重参考后 Cz 通道应该接近 0
    # 这能帮你验证 ref_channel 是否选对 & 通道顺序是否匹配。
    for ch in plot_channels:
        if ch not in ch_names:
            raise ValueError(
                f"你要画的通道 '{ch}' 不在当前 ch_order='{args.ch_order}' 的通道名列表里。\n"
                f"可选通道名：{ch_names}"
            )
        ch_idx = ch_names.index(ch)

        raw_full = X_raw[trial_idx, ch_idx, :]
        reref_full = X_reref[trial_idx, ch_idx, :]

        out_path = os.path.join(
            out_dir,
            f"sub{args.subject:02d}_{args.split}_trial{trial_idx:03d}_ch-{ch}_ref-{args.ref_channel}_wins.png"
        )

        plot_and_save_one_channel(
            raw_full=raw_full,
            reref_full=reref_full,
            ref_full=ref_full,
            fs=FS,
            win_seconds_list=WINDOW_SECONDS,
            subject=args.subject,
            split=args.split,
            trial_idx=trial_idx,
            ch_name=ch,
            ref_name=args.ref_channel,
            label=label,
            out_path=out_path,
            show_ref_signal=bool(args.show_ref),
        )

        print(f"[OK] saved: {out_path}")

    print("\n完成 ✅ 你现在可以在输出目录里查看每个通道的对比图。")


if __name__ == "__main__":
    main()
