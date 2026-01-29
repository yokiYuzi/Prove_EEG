"""preprocess_reref.py

在你原始的 BCICIV-2a 预处理流程（get_data + 标准化）基础上，加入“单点参考重参考”功能：

    x' = x - x_ref

对应你给出的公式：
    x_i' = x_i − x_top = (V_i − V_REF0) − (V_top − V_REF0) = V_i − V_top

也就是把原始记录的所有通道都换成“相对某个参考电极（例如 Cz）的电位差”。

特点/注意：
- 重参考后，参考通道会变为全 0（如果保留该通道，它将不含有效脑信息）。
- 若你希望删除这个全 0 通道，可设置 drop_ref=True。

接口兼容性：
- 尽量保持 preprocess.py 的 get_data(...) 返回值不变。
- 新增参数均提供默认值，不会影响旧代码的调用。

作者：由你提供的 preprocess.py / LoadData.py 逻辑改写扩展
"""

import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(current_path)[0])[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from LoadData import load_data_2a, Load_BCIC_2b
from LoadData import load_data_LOSO
from LoadData import load_data_onLine2a


# =============================================================================
# Channel name presets (用于通过名称定位 ref_channel)
# =============================================================================
# BCI Competition IV-2a: 22 EEG channels (顺序应与官方数据矩阵中前 22 列一致)
BCIC2A_CH_NAMES_22 = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]

# BCI Competition IV-2b: 通常为 3 EEG channels
BCIC2B_CH_NAMES_3 = ["C3", "Cz", "C4"]


def _default_ch_names(data_type: str, n_ch: int):
    """为常见数据集提供默认通道名列表。

    如果通道数与预期不匹配，会返回 None。
    """
    dt = (data_type or "").lower()
    if dt == "2a" and n_ch == 22:
        return list(BCIC2A_CH_NAMES_22)
    if dt == "2b" and n_ch == 3:
        return list(BCIC2B_CH_NAMES_3)
    return None


# =============================================================================
# Standardization / Normalization utilities
# =============================================================================
def _ensure_3d_numpy(X, name: str) -> np.ndarray:
    """Ensure X is a 3D numpy array: (Trials, Channels, Time)."""
    if isinstance(X, list):
        X = np.asarray(X)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"{name} 必须是 3D 数组 (Trials, Channels, Time)，但收到: {X.shape}")
    return X


def standardize_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    channels: int,
    mode: str = "channel_global",
    eps: float = 1e-6,
):
    """标准化 EEG 原始时域信号。

    期望输入形状: (Trials, Channels, Time)

    mode:
      - channel_global（推荐）：每通道 1 套 mean/std（在 train 的 trial*time 上统计）
      - trial：每个 trial、每通道在自身 time 上做 z-score
      - timepoint_across_trials（保留旧行为，不推荐）：把每个 timepoint 当特征，在 trial 维 fit
    """
    X_train = _ensure_3d_numpy(X_train, "X_train")
    X_test = _ensure_3d_numpy(X_test, "X_test")

    channels = int(X_train.shape[1])

    if mode == "channel_global":
        mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1,C,1)
        std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), eps)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        return X_train, X_test

    if mode == "trial":
        mean_tr = X_train.mean(axis=2, keepdims=True)
        std_tr = np.maximum(X_train.std(axis=2, keepdims=True), eps)
        X_train = (X_train - mean_tr) / std_tr

        mean_te = X_test.mean(axis=2, keepdims=True)
        std_te = np.maximum(X_test.std(axis=2, keepdims=True), eps)
        X_test = (X_test - mean_te) / std_te
        return X_train, X_test

    if mode == "timepoint_across_trials":
        for j in range(channels):
            scaler = StandardScaler()
            scaler.fit(X_train[:, j, :])
            X_train[:, j, :] = scaler.transform(X_train[:, j, :])
            X_test[:, j, :] = scaler.transform(X_test[:, j, :])
        return X_train, X_test

    raise ValueError(
        f"未知 standardize mode: {mode}. 可选: channel_global / trial / timepoint_across_trials"
    )


def standardize_data_trans(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_train_trans: np.ndarray,
    channels: int,
    mode: str = "channel_global",
    eps: float = 1e-6,
):
    """Transfer 场景标准化：默认用 TRAIN 的统计对 test/trans 应用。"""
    X_train = _ensure_3d_numpy(X_train, "X_train")
    X_test = _ensure_3d_numpy(X_test, "X_test")
    X_train_trans = _ensure_3d_numpy(X_train_trans, "X_train_trans")

    channels = int(X_train.shape[1])

    if mode == "channel_global":
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), eps)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        X_train_trans = (X_train_trans - mean) / std
        return X_train, X_test, X_train_trans

    if mode == "trial":
        mean_tr = X_train.mean(axis=2, keepdims=True)
        std_tr = np.maximum(X_train.std(axis=2, keepdims=True), eps)
        X_train = (X_train - mean_tr) / std_tr

        mean_te = X_test.mean(axis=2, keepdims=True)
        std_te = np.maximum(X_test.std(axis=2, keepdims=True), eps)
        X_test = (X_test - mean_te) / std_te

        mean_trans = X_train_trans.mean(axis=2, keepdims=True)
        std_trans = np.maximum(X_train_trans.std(axis=2, keepdims=True), eps)
        X_train_trans = (X_train_trans - mean_trans) / std_trans
        return X_train, X_test, X_train_trans

    if mode == "timepoint_across_trials":
        for j in range(channels):
            scaler = StandardScaler()
            scaler.fit(X_train[:, j, :])
            X_train[:, j, :] = scaler.transform(X_train[:, j, :])
            X_test[:, j, :] = scaler.transform(X_test[:, j, :])
            X_train_trans[:, j, :] = scaler.transform(X_train_trans[:, j, :])
        return X_train, X_test, X_train_trans

    raise ValueError(
        f"未知 standardize mode: {mode}. 可选: channel_global / trial / timepoint_across_trials"
    )


def standardize_data_onLine2a(
    X_train: np.ndarray,
    channels: int,
    mode: str = "channel_global",
    eps: float = 1e-6,
):
    """online_2a 场景：只有 X_train。"""
    X_train = _ensure_3d_numpy(X_train, "X_train")
    channels = int(X_train.shape[1])

    if mode == "channel_global":
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), eps)
        X_train = (X_train - mean) / std
        return X_train

    if mode == "trial":
        mean_tr = X_train.mean(axis=2, keepdims=True)
        std_tr = np.maximum(X_train.std(axis=2, keepdims=True), eps)
        X_train = (X_train - mean_tr) / std_tr
        return X_train

    if mode == "timepoint_across_trials":
        for j in range(channels):
            scaler = StandardScaler()
            scaler.fit(X_train[:, j, :])
            X_train[:, j, :] = scaler.transform(X_train[:, j, :])
        return X_train

    raise ValueError(
        f"未知 standardize mode: {mode}. 可选: channel_global / trial / timepoint_across_trials"
    )


# =============================================================================
# Re-referencing
# =============================================================================
def _resolve_ref_index(ref_channel, ch_names, n_ch: int) -> int:
    """把 ref_channel（int 或 str）解析为通道索引。"""
    if isinstance(ref_channel, (int, np.integer)):
        idx = int(ref_channel)
        if idx < 0 or idx >= n_ch:
            raise ValueError(f"ref_channel 索引越界: {idx}，通道数={n_ch}")
        return idx

    if isinstance(ref_channel, str):
        if ch_names is None:
            raise ValueError(
                "ref_channel 使用字符串时，必须提供 ch_names（或让程序自动推断）。"
            )
        # 允许大小写不敏感匹配
        lower_map = {c.lower(): i for i, c in enumerate(ch_names)}
        key = ref_channel.lower()
        if key not in lower_map:
            raise ValueError(
                f"ref_channel='{ref_channel}' 不在 ch_names 中。可选: {ch_names}"
            )
        return int(lower_map[key])

    raise TypeError(
        f"ref_channel 必须是 int 或 str，但收到: {type(ref_channel)}"
    )


def rereference_to_channel(
    X: np.ndarray,
    ref_channel="Cz",
    ch_names=None,
    drop_ref: bool = False,
    copy: bool = True,
):
    """对 3D EEG 数据做单点参考重参考：X' = X - X_ref。

    参数:
      - X: shape (Trials, Channels, Time)
      - ref_channel: int 或 str（例如 'Cz'）
      - ch_names: 通道名列表（用于 ref_channel 为 str 时定位索引）
      - drop_ref: 是否删除参考通道（否则该通道将变为全 0）
      - copy: True 返回新数组；False 尝试原地修改（会改变输入 X）

    返回:
      X_new, ch_names_new, ref_index
    """
    X = _ensure_3d_numpy(X, "X")
    n_ch = int(X.shape[1])
    ref_index = _resolve_ref_index(ref_channel, ch_names, n_ch)

    if copy:
        ref = X[:, ref_index : ref_index + 1, :]
        X_new = X - ref
    else:
        # 原地减法时，必须先拷贝 ref，否则减完后 ref 也变了
        ref = X[:, ref_index : ref_index + 1, :].copy()
        X -= ref
        X_new = X

    ch_names_new = None
    if ch_names is not None:
        ch_names_new = list(ch_names)

    if drop_ref:
        X_new = np.delete(X_new, ref_index, axis=1)
        if ch_names_new is not None:
            ch_names_new.pop(ref_index)

    return X_new, ch_names_new, ref_index


# =============================================================================
# Data loading
# =============================================================================
def get_data(
    path,
    subject=None,
    LOSO=False,
    Transfer=False,
    trans_num=1,
    onLine_2a=False,
    data_model='one_session',
    isStandard=True,
    data_type='2a',
    standardize_mode: str = "channel_global",
    # ---- 新增：重参考参数 ----
    rereference: bool = False,
    ref_channel="Cz",
    drop_ref: bool = False,
    ch_names=None,
    return_ch_names: bool = False,
):
    """加载 BCIC 数据并进行（可选）重参考 + 标准化。

    重参考实现：
      X' = X - X_ref

    关键点：
      - 建议在标准化之前做重参考（本实现就是如此）。
      - 若 drop_ref=False，参考通道会变为全 0。

    返回：
      默认与原 preprocess.get_data 一致：
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans
      若 return_ch_names=True，会在末尾额外返回 ch_names_new。
    """

    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(2 * fs)  # start time_point
    t2 = int(6 * fs)  # end time_point

    # Load and split the dataset into training and testing
    if LOSO:
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = load_data_LOSO(
            path, subject, data_model, Transfer, trans_num
        )
    elif onLine_2a:
        X_train, y_train = load_data_onLine2a(path, data_model)
        X_test = []
        y_test = []
        X_train_trans = []
        y_train_trans = []
    else:
        path = path + 's{:}/'.format(subject)
        if data_type == '2a':
            X_train, y_train = load_data_2a(path, subject, True)
            X_test, y_test = load_data_2a(path, subject, False)
            X_train_trans = []
            y_train_trans = []
        elif data_type == '2b':
            load_raw_data = Load_BCIC_2b(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=0.0, tmax=4.0)
            X_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=0.0, tmax=4.0)
            X_test, y_test = eeg_data['x_data'], eeg_data['y_labels']
            X_train_trans = []
            y_train_trans = []
        else:
            raise ValueError(f"未知 data_type: {data_type}（支持 '2a' / '2b'）")

    # Prepare training data
    if data_type == '2a':
        X_train = _ensure_3d_numpy(X_train, "X_train")
        X_train = X_train[:, :, t1:t2]
        y_train = y_train - 1

    # Prepare testing data
    if (onLine_2a is False) and (data_type == '2a'):
        X_test = _ensure_3d_numpy(X_test, "X_test")
        X_test = X_test[:, :, t1:t2]
        y_test = y_test - 1

    # Transfer set
    if Transfer:
        X_train_trans = _ensure_3d_numpy(X_train_trans, "X_train_trans")
        if data_type == '2a':
            X_train_trans = X_train_trans[:, :, t1:t2]
            y_train_trans = y_train_trans - 1
    else:
        X_train_trans = []
        y_train_trans = []

    # -----------------------------
    # Re-reference (before standardization)
    # -----------------------------
    ch_names_new = None
    if rereference:
        # 如果用户没传 ch_names，则尝试用预置通道名推断
        if ch_names is None:
            ch_names = _default_ch_names(data_type, int(X_train.shape[1]))

        # train
        X_train, ch_names_new, ref_idx = rereference_to_channel(
            X_train,
            ref_channel=ref_channel,
            ch_names=ch_names,
            drop_ref=drop_ref,
            copy=True,
        )

        # test
        if (onLine_2a is False) and isinstance(X_test, np.ndarray) and X_test.size != 0:
            X_test, _, _ = rereference_to_channel(
                X_test,
                ref_channel=ref_channel,
                ch_names=ch_names,
                drop_ref=drop_ref,
                copy=True,
            )

        # transfer
        if Transfer and isinstance(X_train_trans, np.ndarray) and X_train_trans.size != 0:
            X_train_trans, _, _ = rereference_to_channel(
                X_train_trans,
                ref_channel=ref_channel,
                ch_names=ch_names,
                drop_ref=drop_ref,
                copy=True,
            )

    # -----------------------------
    # Standardize
    # -----------------------------
    if isStandard is True:
        if Transfer:
            X_train, X_test, X_train_trans = standardize_data_trans(
                X_train, X_test, X_train_trans, int(X_train.shape[1]), mode=standardize_mode
            )
        elif onLine_2a:
            X_train = standardize_data_onLine2a(
                X_train, int(X_train.shape[1]), mode=standardize_mode
            )
        else:
            X_train, X_test = standardize_data(
                X_train, X_test, int(X_train.shape[1]), mode=standardize_mode
            )

    if return_ch_names:
        return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans, ch_names_new

    return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans


# =============================================================================
# Utils
# =============================================================================
def cross_validate(x_data, y_label, kfold, data_seed=20230520):
    """与原版一致：按标签分层的 K 折划分生成器。"""

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index, split_validation_index in skf.split(x_data, y_label):
        split_train_x = x_data[split_train_index]
        split_train_y = y_label[split_train_index]
        split_validation_x = x_data[split_validation_index]
        split_validation_y = y_label[split_validation_index]

        split_train_x = torch.FloatTensor(split_train_x)
        split_train_y = torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x = torch.FloatTensor(split_validation_x)
        split_validation_y = torch.LongTensor(split_validation_y).reshape(-1)

        split_train_dataset = TensorDataset(split_train_x, split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x, split_validation_y)

        yield split_train_dataset, split_validation_dataset


def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    """与原版一致：PyTorch DataLoader 封装。"""
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # 简单自检：构造随机数据，确保重参考功能不报错
    X = np.random.randn(10, 22, 1000).astype(np.float32)
    X2, names2, idx = rereference_to_channel(X, ref_channel="Cz", ch_names=BCIC2A_CH_NAMES_22, drop_ref=False)
    assert X2.shape == X.shape
    assert idx == BCIC2A_CH_NAMES_22.index("Cz")
    print("Sanity check passed. ref_idx=", idx)
