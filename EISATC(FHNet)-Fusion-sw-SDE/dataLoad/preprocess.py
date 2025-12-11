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



#%%
def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])
          X_test[:, j, :] = scaler.transform(X_test[:, j, :])

    return X_train, X_test

#%%
def standardize_data_trans(X_train, X_test, X_train_trans, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])
          X_test[:, j, :] = scaler.transform(X_test[:, j, :])
          X_train_trans[:, j, :] = scaler.transform(X_train_trans[:, j, :])

    return X_train, X_test, X_train_trans

#%%
def standardize_data_onLine2a(X_train, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])

    return X_train


#%% 新增：滑动窗口数据增强函数
def sliding_window_augment(X, y, fs=250, win_sec=2.0, step_sec=0.1):
    """
    对 4 秒 trial 做滑动窗口增强。
    参数
    ----
    X: np.ndarray, 形状 (n_trial, n_chan, T) ，T≈1000
    y: np.ndarray, 形状 (n_trial,)
    fs: 采样率, 默认 250 Hz
    win_sec: 窗口长度（秒），默认 2s -> 500 samples
    step_sec: 步长（秒），默认 0.1s -> 25 samples

    返回
    ----
    X_out: (n_trial * n_win, n_chan, win_len)
    y_out: (n_trial * n_win,)
    n_win: 每个原 trial 产生的窗口数
    """
    n_trial, n_chan, T = X.shape
    win_len = int(round(win_sec * fs))
    step = int(round(step_sec * fs))

    assert win_len <= T, f"window length {win_len} > trial length {T}"

    n_win = (T - win_len) // step + 1
    X_out = np.zeros((n_trial * n_win, n_chan, win_len), dtype=X.dtype)
    y_out = np.zeros((n_trial * n_win,), dtype=y.dtype)

    idx = 0
    for i in range(n_trial):
        for start in range(0, T - win_len + 1, step):
            X_out[idx] = X[i, :, start:start + win_len]
            y_out[idx] = y[i]
            idx += 1

    return X_out, y_out, n_win


#%%
def get_data(path, subject=None, LOSO=False, Transfer=False, trans_num=1,
             onLine_2a=False, data_model='one_session',
             isStandard=True, data_type='2a',
             use_sliding_window=False, win_sec=2.0, step_sec=0.1):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(2*fs)    # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)
 
    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = load_data_LOSO(path, subject, data_model, Transfer, trans_num)
    elif onLine_2a:
        X_train, y_train = load_data_onLine2a(path, data_model)
        X_test = []
        y_test = []
    else:
        # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
        # In this approach, we used the same training and testing data as the original competition, 
        # i.e., trials in session 1 for training, and trials in session 2 for testing.  
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

    # === 只对训练集做滑动窗口增强 ===
    if use_sliding_window and (not onLine_2a) and data_type == '2a':
        X_train, y_train, n_win = sliding_window_augment(
            X_train, y_train,
            fs=fs, win_sec=win_sec, step_sec=step_sec
        )
        print(f"[DataAug] sliding window: win={int(win_sec*fs)} samples, "
              f"step={int(step_sec*fs)} samples -> {n_win} windows/trial, "
              f"X_train shape = {X_train.shape}")

        # 更新形状变量，保证后面的 standardize 能正确运行
        N_tr, N_ch, samples = X_train.shape

    # Prepare testing data 
    if onLine_2a == False:
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
    if (isStandard == True):
        if Transfer:
            X_train, X_test, X_train_trans = standardize_data_trans(X_train, X_test, X_train_trans, N_ch)
        elif onLine_2a:
            X_train = standardize_data_onLine2a(X_train, N_ch)
        else:
            X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans


#%%
def cross_validate(x_data, y_label, kfold, data_seed=20230520):
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:Guangjin Liang
    '''

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index,split_validation_index in skf.split(x_data,y_label):
        split_train_x       = x_data[split_train_index]
        split_train_y       = y_label[split_train_index]
        split_validation_x  = x_data[split_validation_index]
        split_validation_y  = y_label[split_validation_index]

        split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
        split_train_dataset = TensorDataset(split_train_x,split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x,split_validation_y)
    
        yield split_train_dataset,split_validation_dataset


#%%
def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    '''
    Cenerate the batch data.

    Args:
        x_train: data to be trained
        y_train: label to be trained
        batch_size: the size of the one batch
        num_workers: how many subprocesses to use for data loading
        shuffle: shuffle the data
    '''
    # 将数据转换为TensorDataset类型
    dataset  = TensorDataset(x_train, y_train)
    # 分割数据，生成batch
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 函数返回值
    return dataloader

# %%