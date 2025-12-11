# run_2a_within_subject.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from preprocess import get_data
from model import My_Model
from functions import train_with_cross_validate, validate_model


def run_subject_2a(subject, data_path, save_root, device):
    print(f"\n========== Subject {subject} ==========")

    # 1) 读入 BCI-2a 数据（Session1 做 train，Session2 做 test）
    # get_data 默认就是 subject-dependent + 0-4s + Z-score
    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=data_path,
        subject=subject,
        data_type='2a'  # 明确指定一下
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2) 构建 EISATC-Fusion 模型
    n_chans  = X_train.shape[1]
    n_samples = X_train.shape[2]
    n_classes = 4

    model = My_Model(
        eeg_chans=n_chans,
        samples=n_samples,
        n_classes=n_classes,
        device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # 3) 论文中的两阶段 + 5-fold 训练（仅在 Session1 上做）
    subject_save_dir = os.path.join(save_root, f"s{subject}")
    os.makedirs(subject_save_dir, exist_ok=True)

    train_with_cross_validate(
        model_name="EISATC_Fusion_2a",
        subject=subject,
        frist_epochs=3000,      # 第一阶段 3000 epoch
        eary_stop_epoch=300,    # ES 容忍 300 epoch
        second_epochs=800,      # 第二阶段 800 epoch
        kfolds=5,               # 论文设定：5-fold cross-validation
        batch_size=64,          # 论文设定：batch size 64
        device=device,
        X_train=X_train,
        Y_train=y_train,
        model=model,
        losser=criterion,
        model_savePath=subject_save_dir,
        n_calsses=n_classes
    )

    # 4) 用最终模型在 Session2 上做测试
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    test_loss, test_acc, cm = validate_model(
        model=model,
        dataset=test_dataset,
        device=device,
        losser=criterion,
        batch_size=64,
        n_calsses=n_classes
    )

    acc_value = (test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    print(f"[Subject {subject}] Test Acc = {acc_value*100:.2f}%  | Test Loss = {test_loss:.4f}")
    print("Confusion matrix:\n", cm)

    return acc_value


if __name__ == "__main__":
    # 修改成你机器上的实际路径
    data_path = "/mnt/sdb/home/changw11/Prove_EEG/EISATC-Fusion-main/dataLoad/BCICIV_2a_mat/"
    save_root = "./Saved_files/BCIC_2a/within_subject/EISATC_Fusion/"
    os.makedirs(save_root, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_acc = []

    for sub in range(1, 9 + 1):
        acc = run_subject_2a(sub, data_path, save_root, device)
        all_acc.append(acc)

    all_acc = np.array(all_acc)
    print("\n========== Summary on BCI-2a (within-subject) ==========")
    print("Per-subject acc (%):", np.round(all_acc * 100, 2))
    print(f"Mean ± Std: {all_acc.mean()*100:.2f} ± {all_acc.std()*100:.2f}")
