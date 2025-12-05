# run_2a_fhnet.py  —  BCICIV-2a + DSTAGNN + DDP + 保存最佳权重 + 小模型 + AdamW + label_smoothing + Cosine LR + 真实 EEG 10-20 拓扑

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


# ================== 基础超参数 ==================
NUM_CHANNELS = 22        # EEG 导联数（BCICIV-2a）
WINDOW_SIZE = 250       # 每个 trial 的采样点数（2~6s, 4s * 250Hz）
NUM_CLASSES = 4          # 四分类 MI

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
EPOCHS_PER_FOLD = int(os.environ.get("EPOCHS", "200"))
K_FOLDS = 5
LR = 1e-3

# DSTAGNN 的结构参数（EEG 专用小模型版，容量约为原版的 1/4~1/5）
K_CHEB = 2
NB_BLOCK = 1           # 块数降到 1
NB_CHEV_FILTER = 32    # Cheb滤波器数量 32
D_MODEL_ATTN = 32      # 空间注意力 d_model 32
N_HEADS_ATTN = 2       # 多头数 2
DSTAGNN_D_K_ATTN = 8   # d_k = 8
DSTAGNN_D_V_ATTN = 8   # d_v = 8


# ========= EEG 通道拓扑（BCICIV-2a, 22 导联） =========
CHANNELS_2A = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3",  "C1",  "Cz",  "C2",  "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P3",  "P1",  "Pz",  "P2",  "P4",
    "POz",
]

# 粗略 2D 坐标（行 = 前后，列 = 左右），只用来计算相对距离
CHAN_POS_2A = {
    "Fz":  (0, 2),

    "FC3": (1, 0), "FC1": (1, 1), "FCz": (1, 2),
    "FC2": (1, 3), "FC4": (1, 4),

    "C3":  (2, 0), "C1":  (2, 1), "Cz":  (2, 2),
    "C2":  (2, 3), "C4":  (2, 4),

    "CP3": (3, 0), "CP1": (3, 1), "CPz": (3, 2),
    "CP2": (3, 3), "CP4": (3, 4),

    "P3":  (4, 0), "P1":  (4, 1), "Pz":  (4, 2),
    "P2":  (4, 3), "P4":  (4, 4),

    "POz": (5, 2),
}


def build_eeg_2a_adj(num_channels: int,
                     connect_thresh: float = 1.5,
                     self_loop: bool = True) -> np.ndarray:
    """
    根据 BCICIV-2a 的 10-20 布局构造 EEG 邻接矩阵。

    参数
    ----
    num_channels : 实际使用的通道数（这里应为 22）
    connect_thresh : 空间距离阈值，小于等于该值的两点视为相邻
    self_loop : 是否保留自环（对图卷积和拉普拉斯较稳定，建议 True）

    返回
    ----
    adj : (num_channels, num_channels) 的对称邻接矩阵
    """
    assert num_channels == len(CHANNELS_2A), \
        f"num_channels={num_channels} 与 CHANNELS_2A 数量 {len(CHANNELS_2A)} 不一致"

    name_to_idx = {ch: i for i, ch in enumerate(CHANNELS_2A)}
    adj = np.zeros((num_channels, num_channels), dtype=np.float32)

    for ch_i, (ri, ci) in CHAN_POS_2A.items():
        i = name_to_idx[ch_i]
        for ch_j, (rj, cj) in CHAN_POS_2A.items():
            j = name_to_idx[ch_j]
            if i == j:
                continue
            dist = np.sqrt((ri - rj) ** 2 + (ci - cj) ** 2)
            if dist <= connect_thresh:
                w = 1.0
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = max(adj[j, i], w)

    if self_loop:
        np.fill_diagonal(adj, 1.0)

    return adj


# ================== 训练 & 评估函数 ==================
def train_one_epoch_eeg(model, loader, optimizer, criterion, device):
    """单个 epoch 的训练；支持 DDP 下的全局 loss / acc 统计"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    is_distributed = dist.is_initialized()

    for inputs, labels in loader:
        # inputs: (B, 22, 1000)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 下采样到 250Hz
        inputs = inputs[:, :, ::4]  # (B, 22, 1000) -> (B, 22, 250)

        # (B, 22, 250) -> (B, N, 1, T)
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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            x_for_dstagnn = inputs.unsqueeze(2)
            outputs = model(x_for_dstagnn)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(all_labels))
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
    # 先写死一个 subject，你可以改成从环境变量读：
    # subject = int(os.environ.get("SUBJECT", "1"))
    subject = 1

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

    # 脚本路径，用于组织数据和模型保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存 EEG + DSTAGNN 最佳模型权重的根目录
    save_root = os.path.join(script_dir, "eeg_bcic2a_dstagnn_ckpts")
    if rank == 0:
        os.makedirs(save_root, exist_ok=True)

    # ------- 1. 读取 EEG 数据（BCICIV-2a） -------
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

    # ------- 2. 准备图结构：基于 10-20 布局的 EEG 拓扑 -------
    adj_mx = build_eeg_2a_adj(NUM_CHANNELS)

    # 同一张图既给 Chebyshev 卷积用，也给 DSTAGNN 的静态空间图用
    # （如果以后你想做多图融合，可以再单独定义 adj_pa_static / adj_TMD_static）

    all_fold_metrics = []
    global_best_f1 = 0.0
    global_best_info = None

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

        model.to(device)

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

        # ================== 优化器、损失、学习率调度器（关键修改） ==================
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=1e-2          # 强 L2 正则，防过拟合
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # label smoothing，对小样本非常有效

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS_PER_FOLD,     # 每个 fold 独立一个周期
            eta_min=1e-5               # 最小学习率，可选
        )
        # ===========================================================================

        best_val_f1 = 0.0
        best_fold_metrics = None

        # 每折单独一个保存目录
        if rank == 0:
            fold_save_dir = os.path.join(
                save_root, f"sub{subject}_fold{fold_idx + 1}"
            )
            os.makedirs(fold_save_dir, exist_ok=True)
            best_model_path = os.path.join(
                fold_save_dir, "best_val_model.pth"
            )

        # ------- 5. 每折内部训练若干 epoch -------
        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            if is_distributed:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch_eeg(
                model, train_loader, optimizer, criterion, device
            )

            # 学习率调度器 step（每个 epoch 结束后）
            scheduler.step()

            # 只在 rank0 上做验证 & 打印 & 保存最佳模型
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
                    f"Val F1(macro): {val_metrics['f1_macro']:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

                # 记录并保存“最佳验证集 F1”的模型权重
                if val_metrics["f1_macro"] > best_val_f1:
                    best_val_f1 = val_metrics["f1_macro"]
                    best_fold_metrics = val_metrics

                    state = model_for_eval.state_dict()
                    torch.save(state, best_model_path)
                    print(
                        f"  -> 新的最佳模型已保存到 {best_model_path} "
                        f"(Val F1={best_val_f1:.4f})"
                    )

        # ------- 6. 每折结束后，在 rank0 记录最优结果 -------
        if rank == 0 and best_fold_metrics is not None:
            all_fold_metrics.append(best_fold_metrics)
            print(f"[Fold {fold_idx + 1}] 最佳验证集 F1(macro): {best_val_f1:.4f}")

            if best_val_f1 > global_best_f1:
                global_best_f1 = best_val_f1
                global_best_info = {
                    "subject": subject,
                    "fold": fold_idx + 1,
                    "metrics": best_fold_metrics,
                    "path": best_model_path,
                }

    # ------- 7. 所有折结束后，汇总 K 折性能 -------
    if rank == 0 and len(all_fold_metrics) > 0:
        avg_f1 = np.mean([m["f1_macro"] for m in all_fold_metrics])
        avg_acc = np.mean([m["accuracy"] for m in all_fold_metrics])
        print("\n===== K 折平均结果（验证集）=====")
        print(f"Avg F1(macro): {avg_f1:.4f}")
        print(f"Avg Accuracy : {avg_acc:.4f}")

        if global_best_info is not None:
            print("\n===== 全局最佳模型信息 =====")
            print(f"Subject {global_best_info['subject']}, "
                  f"Fold {global_best_info['fold']}, "
                  f"Best Val F1: {global_best_info['metrics']['f1_macro']:.4f}")
            print(f"权重文件路径: {global_best_info['path']}")

    # ------- 8. 清理 DDP -------
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()