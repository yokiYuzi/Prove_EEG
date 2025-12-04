# main_eeg_bcic2a_ddp.py
import os
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import classification_report

from dataLoad.preprocess import get_data, cross_validate
from DSTAGNN_my import make_model


# ================== 一些基础超参数（可以按需再改） ==================
NUM_CHANNELS = 22        # EEG 导联数（BCICIV-2a）
WINDOW_SIZE = 1000       # 每个 trial 的采样点数（2~6s, 4s * 250Hz）
NUM_CLASSES = 4          # 四分类 MI

BATCH_SIZE = 4
EPOCHS_PER_FOLD = 50     # 先给一个适中的值，你可以再往上调
K_FOLDS = 5
LR = 1e-3

# DSTAGNN 的结构参数（目前和 ECG 版保持一致）
K_CHEB = 2
NB_BLOCK = 2
NB_CHEV_FILTER = 64
D_MODEL_ATTN = 64
N_HEADS_ATTN = 4
DSTAGNN_D_K_ATTN = 16
DSTAGNN_D_V_ATTN = 16


# ================== 训练 & 评估函数 ==================
def train_one_epoch_eeg(model, loader, optimizer, criterion, device):
    """单个 epoch 的训练；支持 DDP 下的全局 loss / acc 统计"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    for inputs, labels in loader:
        # inputs: (B, 22, 1000)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # (B, 22, 1000) -> (B, N, 1, T)
        x_for_dstagnn = inputs.unsqueeze(2)

        optimizer.zero_grad()
        outputs = model(x_for_dstagnn)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    # 如果是 DDP，做 all_reduce 得到全局的 loss / acc
    if is_distributed:
        tensor = torch.tensor(
            [total_loss, total_correct, total_samples],
            dtype=torch.float64,
            device=device
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = tensor.tolist()

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def evaluate_eeg(model, loader, criterion, device):
    """评估函数：不使用分布式 sampler，在 rank0 上完整跑一遍即可"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            x_for_dstagnn = inputs.unsqueeze(2)  # (B, 22, 1, 1000)

            outputs = model(x_for_dstagnn)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    metrics = {
        "loss": avg_loss,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "full_report": report,
    }
    return metrics


# ================== 主函数 ==================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1,
                        help="BCICIV-2a 的被试编号（1~9）")
    args = parser.parse_args()
    subject = args.subject

    # ------- DDP 基础设置 -------
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0

    if is_distributed:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=12)
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    if rank == 0:
        print(f"使用设备: {device}")
        if is_distributed:
            print(f"[DDP] world size = {world_size}, local rank = {local_rank}")

    # ------- 1. 读取 EEG 数据（BCICIV-2a） -------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataLoad", "BCICIV_2a") + os.sep

    if rank == 0:
        print(f"从 {data_dir} 加载被试 {subject} 的 BCICIV-2a 数据...")

    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=data_dir,
        subject=subject,
        LOSO=False,
        data_type='2a'
    )
    # X_train: (N_tr, 22, 1000), y_train: (N_tr,)

    if rank == 0:
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # ------- 2. K 折划分（在训练集里做交叉验证） -------
    folds = list(cross_validate(X_train, y_train, kfold=K_FOLDS))
    if rank == 0:
        print(f"K 折 = {K_FOLDS}，每折都会重新初始化模型并训练。")

    # 简单的静态邻接矩阵：先用全 1，后续你可以换成基于导联空间位置的图
    adj_mx = np.ones((NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)

    all_fold_metrics = []

    # ------- 3. K 折循环 -------
    for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
        if rank == 0:
            print(f"\n===== Subject {subject} - Fold {fold_idx + 1}/{K_FOLDS} =====")
            print(f"Train samples: {len(train_dataset)}, "
                  f"Val samples: {len(val_dataset)}")

        # 训练集 DataLoader（DDP 使用 DistributedSampler）
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=train_sampler,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

        # 验证集 DataLoader（不需要分布式采样器）
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # ------- 4. 构建 DSTAGNN 模型 -------
        model = make_model(
            DEVICE=device,
            num_of_d_initial_feat=1,
            nb_block=NB_BLOCK,
            initial_in_channels_cheb=1,
            K_cheb=K_CHEB,
            nb_chev_filter=NB_CHEV_FILTER,
            nb_time_filter_block_unused=32,
            initial_time_strides=1,
            adj_mx=adj_mx,
            adj_pa_static=adj_mx,
            adj_TMD_static_unused=np.zeros_like(adj_mx),
            num_for_predict_per_node=1,
            len_input_total=WINDOW_SIZE,
            num_of_vertices=NUM_CHANNELS,
            d_model_for_spatial_attn=D_MODEL_ATTN,
            d_k_for_attn=DSTAGNN_D_K_ATTN,
            d_v_for_attn=DSTAGNN_D_V_ATTN,
            n_heads_for_attn=N_HEADS_ATTN,
            output_memory=False,
            return_internal_states=False,
            task_type="classification",
            num_classes=NUM_CLASSES
        )

        # DDP / DP 包装
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False
            )
        elif torch.cuda.device_count() > 1:
            if rank == 0:
                print(f"[多卡] 启用 DataParallel，GPU 数量: {torch.cuda.device_count()}")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        best_fold_metrics = None

        # ------- 5. 每折内部训练若干 epoch -------
        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            if is_distributed:
                # 保证每个 epoch sampler 的 shuffle 一致但不同 epoch 不同
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch_eeg(
                model, train_loader, optimizer, criterion, device
            )

            # 只在 rank0 上做验证 & 打印
            if rank == 0:
                model_for_eval = (model.module
                                  if isinstance(model, (DDP, nn.DataParallel))
                                  else model)

                val_metrics = evaluate_eeg(
                    model_for_eval, val_loader, criterion, device
                )
                print(
                    f"[Fold {fold_idx + 1}] Epoch {epoch}/{EPOCHS_PER_FOLD} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val F1(macro): {val_metrics['f1_macro']:.4f}"
                )

                # 简单的 early-best 记录
                if val_metrics["f1_macro"] > best_val_f1:
                    best_val_f1 = val_metrics["f1_macro"]
                    best_fold_metrics = val_metrics

        # ------- 6. 每折结束后，在 rank0 记录最优结果 -------
        if rank == 0 and best_fold_metrics is not None:
            all_fold_metrics.append(best_fold_metrics)
            print(f"[Fold {fold_idx + 1}] 最佳验证集 F1(macro): {best_val_f1:.4f}")

    # ------- 7. 所有折结束后，汇总 K 折性能 -------
    if rank == 0 and len(all_fold_metrics) > 0:
        avg_f1 = np.mean([m["f1_macro"] for m in all_fold_metrics])
        avg_acc = np.mean([m["accuracy"] for m in all_fold_metrics])
        print("\n===== K 折平均结果（验证集）=====")
        print(f"Avg F1(macro): {avg_f1:.4f}")
        print(f"Avg Accuracy : {avg_acc:.4f}")

    # ------- 8. 清理 DDP -------
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
