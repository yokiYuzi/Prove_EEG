# nifea_data_utils.py (修正后的稳健版本)
# 说明:
#   - [新增] 已根据您的要求修改 get_kfold_dataloaders 函数。
#   - 在分布式训练 (DDP) 模式下，为训练集 DataLoader 启用 DistributedSampler。
#   - 为所有 DataLoader 添加了 num_workers 和 pin_memory 参数以优化数据加载性能。
############################################

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import os

# 1. Dataset 类定义 (保持不变)
class NIFEA_Windowed_Dataset(Dataset):
    """用于存储提取出的固定长度窗口的数据集"""
    def __init__(self, signals, labels):
        # signals: (N_windows, T, C) - 已经是 Tensor, labels: (N_windows,) - 已经是 Tensor
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 数据在创建 DataLoader 时已转换为 Tensor，此处无需再次转换
        return self.signals[idx], self.labels[idx]

# 2. 数据加载和窗口化 (保持不变)
def load_and_window_nifea_data(file_path, num_channels=6, window_size=1000, step_size=500):
    """加载 NPZ 并提取窗口。返回按受试者组织的原始数据结构。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")

    print(f"[数据加载] 正在从 {file_path} 加载数据 (allow_pickle=True)...")
    # 必须使用 allow_pickle=True，因为 signals 是 object 类型
    data = np.load(file_path, allow_pickle=True)
    
    all_signals = data['signals'] 
    all_labels = data['labels']
    record_names = data.get('record_names', np.arange(len(all_signals)))

    print(f"[数据加载] 找到 {len(all_signals)} 个受试者记录。")

    processed_data = []
    
    for i, subject_signal in enumerate(all_signals):
        # 确保使用指定数量的通道
        if subject_signal.shape[1] < num_channels:
            print(f"[警告] 受试者 {record_names[i]} 通道数少于 {num_channels}。使用其所有通道 ({subject_signal.shape[1]})。")
            signal = subject_signal
        else:
            signal = subject_signal[:, :num_channels]
        
        # 填充 NaN 值 (如果有的话) 并确保类型为 float32
        signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

        # 提取滑动窗口
        windows = []
        T = signal.shape[0]
        for start in range(0, T - window_size + 1, step_size):
            windows.append(signal[start:start + window_size, :])
        
        if windows:
            processed_data.append({
                'subject_id': i,
                'record_name': record_names[i],
                'windows_raw': np.array(windows), # 【关键】存储原始窗口数据
                'label': all_labels[i]
            })
        else:
            print(f"[警告] 受试者 {record_names[i]} 信号太短 (长度 {T})，无法提取窗口 (大小 {window_size})。跳过。")
        
    return processed_data

# 3. K-Fold 数据加载器设置 (关键修改处)
def get_kfold_dataloaders(processed_data, n_splits=5, batch_size=16, random_state=42,
                          distributed=False, rank=0, world_size=1, num_workers=4, pin_memory=True):
    """
    关键：按受试者划分 K-Fold，并在划分后独立进行标准化，防止数据泄露。
    [修改]：新增分布式训练支持和性能参数。
    """
    if not processed_data:
        raise ValueError("处理后的数据为空。")

    subject_labels = np.array([d['label'] for d in processed_data])
    subject_indices = np.arange(len(processed_data))
    
    # 检查是否可以进行分层抽样 (稳健性处理)
    if len(np.unique(subject_labels)) < 2 or np.min(np.bincount(subject_labels)) < n_splits:
        print("[警告] 数据集太小或标签分布不均，无法进行分层 K-Fold。回退到标准 KFold。")
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = skf.split(subject_indices)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = skf.split(subject_indices, subject_labels)

    folds = []
    # 按受试者索引进行划分
    for train_subject_indices, test_subject_indices in split_generator:
        
        X_train_raw, y_train, X_test_raw, y_test = [], [], [], []

        # 提取训练/测试集原始数据
        for idx in train_subject_indices:
            X_train_raw.append(processed_data[idx]['windows_raw'])
            y_train.extend([processed_data[idx]['label']] * len(processed_data[idx]['windows_raw']))
            
        for idx in test_subject_indices:
            X_test_raw.append(processed_data[idx]['windows_raw'])
            y_test.extend([processed_data[idx]['label']] * len(processed_data[idx]['windows_raw']))

        # 合并
        X_train_raw = np.concatenate(X_train_raw, axis=0); y_train = np.array(y_train, dtype=np.int64)
        X_test_raw = np.concatenate(X_test_raw, axis=0); y_test = np.array(y_test, dtype=np.int64)

        # --- [关键] 标准化步骤 ---
        N_train, T, C = X_train_raw.shape
        N_test = X_test_raw.shape[0]
        
        X_train_reshaped = X_train_raw.reshape(-1, C)
        X_test_reshaped = X_test_raw.reshape(-1, C)

        scaler = StandardScaler()
        X_train_norm_reshaped = scaler.fit_transform(X_train_reshaped)
        X_test_norm_reshaped = scaler.transform(X_test_reshaped)
        
        # Reshape 回 (N_windows, T, C) 并转换为 Tensor
        X_train_norm = torch.from_numpy(X_train_norm_reshaped.reshape(N_train, T, C)).float()
        X_test_norm = torch.from_numpy(X_test_norm_reshaped.reshape(N_test, T, C)).float()

        # ====== [修改] 创建 DataLoaders（分布式下仅训练集用 DistributedSampler） ======
        train_ds = NIFEA_Windowed_Dataset(X_train_norm, torch.from_numpy(y_train).long())
        test_ds  = NIFEA_Windowed_Dataset(X_test_norm,  torch.from_numpy(y_test).long())
        
        if distributed:
            # 仅在需要时导入，避免不必要的依赖
            from torch.utils.data.distributed import DistributedSampler
            # 为训练集创建分布式采样器
            train_sampler = DistributedSampler(
                train_ds, 
                num_replicas=world_size, # 进程总数
                rank=rank,               # 当前进程ID
                shuffle=True,            # 每个 epoch 都打乱
                drop_last=False          # 不丢弃最后不完整的批次
            )
            # 创建使用分布式采样器的训练 DataLoader
            # 注意: 提供 sampler 时, shuffle 必须为 False
            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                sampler=train_sampler, 
                shuffle=False,
                num_workers=num_workers,       # 使用多进程加载数据
                pin_memory=pin_memory,         # 锁页内存，加快数据到GPU的传输
                persistent_workers=(num_workers > 0) # 避免重复启动 worker 进程
            )
        else:
            # 创建常规的训练 DataLoader
            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0)
            )
        
        # 测试集 DataLoader 不需要分布式采样器，每个进程都将对完整的测试集进行评估
        test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        folds.append((train_loader, test_loader))
        
    return folds