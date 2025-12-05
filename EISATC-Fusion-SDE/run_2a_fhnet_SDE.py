# run_2a_fhnet.py  —  BCICIV-2a 官方协议版：Train on Session T, Test on Session E
# DSTAGNN + 真实 EEG 10-20 拓扑 + 小模型 + AdamW + label_smoothing + Cosine LR + 250点

import os
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from dataLoad.preprocess import get_data
from DSTAGNN_my1 import make_model


# ================== 基础超参数 ==================
NUM_CHANNELS = 22
WINDOW_SIZE = 1000                  # 模型输入时间长度
NUM_CLASSES = 4
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
N_EPOCHS = int(os.environ.get("EPOCHS", "200"))   # 原来的 EPOCHS_PER_FOLD 改名更清晰
LR = 1e-3
VAL_RATIO = 0.1                    # 从 Session T 中划分 10% 做验证集

# DSTAGNN 小模型参数
K_CHEB = 2
NB_BLOCK = 1
NB_CHEV_FILTER = 32
NB_TIME_FILTER_BLOCK_UNUSED = 32
D_MODEL_ATTN = 32
N_HEADS_ATTN = 2
DSTAGNN_D_K_ATTN = 8
DSTAGNN_D_V_ATTN = 8


# ========= EEG 通道真实 10-20 拓扑 =========
CHANNELS_2A = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3",  "C1",  "Cz",  "C2",  "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P3",  "P1",  "Pz",  "P2",  "P4", "POz",
]

CHAN_POS_2A = {
    "Fz":  (0, 2),
    "FC3": (1, 0), "FC1": (1, 1), "FCz": (1, 2), "FC2": (1, 3), "FC4": (1, 4),
    "C3":  (2, 0), "C1":  (2, 1), "Cz":  (2, 2), "C2":  (2, 3), "C4":  (2, 4),
    "CP3": (3, 0), "CP1": (3, 1), "CPz": (3, 2), "CP2": (3, 3), "CP4": (3, 4),
    "P3":  (4, 0), "P1":  (4, 1), "Pz":  (4, 2), "P2":  (4, 3), "P4":  (4, 4),
    "POz": (5, 2),
}


def build_eeg_2a_adj(num_channels: int = 22, connect_thresh: float = 1.5, self_loop: bool = True) -> np.ndarray:
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
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    if self_loop:
        np.fill_diagonal(adj, 1.0)
    return adj


# ================== 输入统一处理：1000 → 250 ==================
def prepare_eeg_for_dstagnn(inputs: torch.Tensor) -> torch.Tensor:
    """
    (B, 22, 1000) or any T → (B, 22, WINDOW_SIZE=250)
    """
    B, C, T = inputs.shape
    assert C == NUM_CHANNELS

    if T == WINDOW_SIZE:
        return inputs
    if T == 1000 and WINDOW_SIZE == 250:
        return inputs[:, :, ::4]
    if T > WINDOW_SIZE and T % WINDOW_SIZE == 0:
        stride = T // WINDOW_SIZE
        return inputs[:, :, ::stride]
    if T > WINDOW_SIZE:
        start = (T - WINDOW_SIZE) // 2
        return inputs[:, :, start:start + WINDOW_SIZE]
    # T < WINDOW_SIZE → pad
    pad_len = WINDOW_SIZE - T
    return torch.nn.functional.pad(inputs, (0, pad_len))


# ================== 训练 & 验证函数 ==================
def train_one_epoch_eeg(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_distributed = dist.is_initialized()

    for inputs, labels in loader:
        inputs = inputs.to(device)          # (B, 22, 1000)
        labels = labels.to(device)

        #inputs = prepare_eeg_for_dstagnn(inputs)   # → (B, 22, 250)
        x = inputs.unsqueeze(2)                     # (B, N, 1, T)

        optimizer.zero_grad()
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    if is_distributed:
        tensor = torch.tensor([total_loss, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = tensor.tolist()

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def evaluate_eeg(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = prepare_eeg_for_dstagnn(inputs)
            x = inputs.unsqueeze(2)

            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(all_labels))
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return {
        "loss": avg_loss,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "full_report": report,
    }


# ================== 主函数（官方协议版） ==================
def main():
    subject = 1   # 可改为循环 1~9

    # ----------------- DDP 初始化 -----------------
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0

    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
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
            print(f"[DDP] world_size={world_size}, local_rank={local_rank}")

    # ----------------- 路径 -----------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataLoad", "BCICIV_2a") + os.sep
    save_root = os.path.join(script_dir, "eeg_bcic2a_dstagnn_ckpts")
    if rank == 0:
        os.makedirs(save_root, exist_ok=True)

    # ----------------- 1. 读取数据（Session T / Session E） -----------------
    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=data_dir, subject=subject, LOSO=False, data_type='2a'
    )   # X_train = Session T, X_test = Session E

    if rank == 0:
        print(f"Subject {subject} | Session T: {X_train.shape} | Session E: {X_test.shape}")

    # ----------------- 2. 从 Session T 划分 train / val -----------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    train_dataset = TensorDataset(torch.FloatTensor(X_train[train_idx]), torch.LongTensor(y_train[train_idx]))
    val_dataset   = TensorDataset(torch.FloatTensor(X_train[val_idx]),   torch.LongTensor(y_train[val_idx]))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test),            torch.LongTensor(y_test))

    # ----------------- 3. DataLoader -----------------
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ----------------- 4. 拓扑图 & 模型 -----------------
    adj_mx = build_eeg_2a_adj(NUM_CHANNELS)

    model = make_model(
        DEVICE=device,
        num_of_d_initial_feat=1,
        nb_block=NB_BLOCK,
        initial_in_channels_cheb=1,
        K_cheb=K_CHEB,
        nb_chev_filter=NB_CHEV_FILTER,
        nb_time_filter_block_unused=NB_TIME_FILTER_BLOCK_UNUSED,
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
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    # ----------------- 5. 训练 -----------------
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_root, f"sub{subject}", "best_model_SessionE.pth")
    if rank == 0:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch_eeg(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if rank == 0:
            model_eval = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            val_metrics = evaluate_eeg(model_eval, val_loader, criterion, device)

            print(f"[S{subject}] Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1_macro']:.4f}")

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                torch.save(model_eval.state_dict(), best_model_path)
                print(f"  → 新最佳模型已保存 (Val F1 = {best_val_f1:.4f})")

    # ----------------- 6. 最终在 Session E 上测试 -----------------
    if rank == 0:
        final_model = make_model(
            DEVICE=device,
            num_of_d_initial_feat=1, nb_block=NB_BLOCK, initial_in_channels_cheb=1,
            K_cheb=K_CHEB, nb_chev_filter=NB_CHEV_FILTER,
            nb_time_filter_block_unused=NB_TIME_FILTER_BLOCK_UNUSED,
            initial_time_strides=1, adj_mx=adj_mx, adj_pa_static=adj_mx,
            adj_TMD_static_unused=np.zeros_like(adj_mx),
            num_for_predict_per_node=1, len_input_total=WINDOW_SIZE,
            num_of_vertices=NUM_CHANNELS,
            d_model_for_spatial_attn=D_MODEL_ATTN,
            d_k_for_attn=DSTAGNN_D_K_ATTN, d_v_for_attn=DSTAGNN_D_V_ATTN,
            n_heads_for_attn=N_HEADS_ATTN,
            task_type="classification", num_classes=NUM_CLASSES
        ).to(device)

        final_model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_model.eval()

        test_metrics = evaluate_eeg(final_model, test_loader, criterion, device)

        print("\n" + "="*60)
        print(f"Subject {subject} 最终结果（Session E 测试集，官方协议）")
        print(f"Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"F1(macro): {test_metrics['f1_macro']:.4f}")
        print("="*60)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()