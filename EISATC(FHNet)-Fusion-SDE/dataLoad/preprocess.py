import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(current_path)[0])[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from LoadData import load_data_2a, Load_BCIC_2b
from LoadData import load_data_LOSO
from LoadData import load_data_onLine2a


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
    """
    标准化 EEG 原始时域信号（在 get_data() 中使用）。

    期望输入形状: (Trials, Channels, Time)

    你之前的实现本质是把“每个时间采样点 t”当作一个特征维度，
    然后在 trial 维做 StandardScaler —— 这会形成“随时间变化的缩放函数”，
    对跨 session 泛化和后续频域特征（STFT/bandpower）都可能不友好。

    这里提供更典型、更稳健的标准化方式（默认推荐）：
      - mode="channel_global"（默认，推荐）：
          对每个通道 c，仅学习 1 个 mean 和 1 个 std（在 TRAIN 上对 trial*time 做统计），
          再应用到 train/test。=> 每通道 1 套统计，不随时间变化。
      - mode="trial"：
          对每个 trial、每个通道独立做 z-score（仅在 trial 内按 time 统计）。
          优点：跨 session 不依赖训练集统计，鲁棒；缺点：会抹掉部分幅度信息。
      - mode="timepoint_across_trials"（保留旧行为，不推荐）：
          把每个 timepoint 当作特征，按 trial 维拟合 StandardScaler。

    参数:
      - channels: 通道数（为兼容旧接口；若与 X_train.shape[1] 不一致，会以 X_train 为准）
      - eps: 防止除零

    返回:
      X_train_std, X_test_std（与输入同 shape）
    """
    X_train = _ensure_3d_numpy(X_train, "X_train")
    X_test = _ensure_3d_numpy(X_test, "X_test")

    # 兼容：channels 以实际数据为准
    channels = int(X_train.shape[1])

    if mode == "channel_global":
        # 每通道 1 个 mean/std：在 (trial,time) 上做统计
        mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1,C,1)
        std = X_train.std(axis=(0, 2), keepdims=True)     # (1,C,1)
        std = np.maximum(std, eps)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        return X_train, X_test

    if mode == "trial":
        # 每个 trial、每个通道独立 z-score（train/test 各自用自己的 trial 统计）
        mean_tr = X_train.mean(axis=2, keepdims=True)
        std_tr = np.maximum(X_train.std(axis=2, keepdims=True), eps)
        X_train = (X_train - mean_tr) / std_tr

        mean_te = X_test.mean(axis=2, keepdims=True)
        std_te = np.maximum(X_test.std(axis=2, keepdims=True), eps)
        X_test = (X_test - mean_te) / std_te
        return X_train, X_test

    if mode == "timepoint_across_trials":
        # === 旧实现：保留用于对照实验，但不建议用于跨 session 泛化 ===
        for j in range(channels):
            scaler = StandardScaler()
            scaler.fit(X_train[:, j, :])          # (Trials, Time) -> 每个 timepoint 一套统计
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
    """
    Transfer 场景的标准化：
      - 默认仍然用 TRAIN 的统计（更符合“只用训练集拟合”的规范），并对 test/trans 应用。
      - trial 模式下：train/test/trans 各自 trial 内独立标准化。
    """
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
):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(2*fs)    # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)

    # Load and split the dataset into training and testing
    if LOSO:
        # Loading and Dividing of the data set based on the
        # 'Leave One Subject Out' (LOSO) evaluation approach.
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = load_data_LOSO(
            path, subject, data_model, Transfer, trans_num
        )
    elif onLine_2a:
        X_train, y_train = load_data_onLine2a(path, data_model)
        X_test = []
        y_test = []
    else:
        # Subject-dependent: Session T for training, Session E for testing
        path = path + 's{:}/'.format(subject)
        if data_type == '2a':
            X_train, y_train = load_data_2a(path, subject, True)
            X_test, y_test = load_data_2a(path, subject, False)
        elif data_type == '2b':
            load_raw_data = Load_BCIC_2b(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
            X_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
            X_test, y_test = eeg_data['x_data'], eeg_data['y_labels']

    # Prepare training data
    N_tr, N_ch, samples = X_train.shape
    if data_type == '2a':
        X_train = X_train[:, :, t1:t2]
        y_train = y_train - 1

    # Prepare testing data
    if onLine_2a is False:
        if data_type == '2a':
            X_test = X_test[:, :, t1:t2]
            y_test = y_test - 1

    if Transfer:
        X_train_trans = X_train_trans[:, :, t1:t2]
        y_train_trans = y_train_trans - 1
    else:
        X_train_trans = []
        y_train_trans = []

    # Standardize the data
    if isStandard is True:
        if Transfer:
            X_train, X_test, X_train_trans = standardize_data_trans(
                X_train, X_test, X_train_trans, N_ch, mode=standardize_mode
            )
        elif onLine_2a:
            X_train = standardize_data_onLine2a(
                X_train, N_ch, mode=standardize_mode
            )
        else:
            X_train, X_test = standardize_data(
                X_train, X_test, N_ch, mode=standardize_mode
            )

    return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans


# =============================================================================
# Utils
# =============================================================================
def cross_validate(x_data, y_label, kfold, data_seed=20230520):
    """
    This version doesn't use early stopping.

    Args:
        x_data: EEG data array
        y_label: labels
        kfold: number of folds
        data_seed: random seed for shuffle
    """

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index, split_validation_index in skf.split(x_data, y_label):
        split_train_x = x_data[split_train_index]
        split_train_y = y_label[split_train_index]
        split_validation_x = x_data[split_validation_index]
        split_validation_y = y_label[split_validation_index]

        split_train_x, split_train_y = torch.FloatTensor(split_train_x), torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x, split_validation_y = torch.FloatTensor(split_validation_x), torch.LongTensor(split_validation_y).reshape(-1)

        split_train_dataset = TensorDataset(split_train_x, split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x, split_validation_y)

        yield split_train_dataset, split_validation_dataset


def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    """
    Generate the batch data.

    Args:
        x_train: data to be trained
        y_train: label to be trained
        batch_size: the size of the one batch
        num_workers: how many subprocesses to use for data loading
        shuffle: shuffle the data
    """
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
