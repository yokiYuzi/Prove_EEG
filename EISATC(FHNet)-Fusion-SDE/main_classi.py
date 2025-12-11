# main_classi.py
# 说明:
#   - 已完全移除 SDE 及解释性导出逻辑。
#   - 只做 K 折分类训练，并打印每个折叠的最佳 F1 (macro) 以及均值±方差。

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

from DSTAGNN_my import make_model
from nifea_data_utils import load_and_window_nifea_data, get_kfold_dataloaders

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from datetime import timedelta

cudnn.benchmark = True

# ================== 超参数和全局设置 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
EPOCHS_PER_FOLD = int(os.environ.get("EPOCHS", "15"))
LR = 1e-4
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
NUM_CLASSES = 2

ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1"))

FS_MIN = 500
WINDOW_SIZE = FS_MIN * 2
STEP_SIZE = FS_MIN * 2
NUM_CHANNELS = 6

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_NPZ = os.path.join(SCRIPT_DIR, "NIFEA_DB_6lead_500Hz_processed.npz")

K_CHEB = 2
NB_BLOCK = 2
NB_CHEV_FILTER = 64
D_MODEL_ATTN = 64
N_HEADS_ATTN = 4
DSTAGNN_D_K_ATTN = 16
DSTAGNN_D_V_ATTN = 16


# ================== 训练 & 评估函数 ==================
def train_one_epoch_cls(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    pbar = tqdm(loader, desc="训练中", leave=False,
                disable=(is_distributed and rank != 0))

    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, labels) in enumerate(pbar, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        # (B,T,C) -> (B,N,1,T)
        x_for_dstagnn = inputs.permute(0, 2, 1).unsqueeze(2)

        if hasattr(train_one_epoch_cls, "_scaler"):
            scaler = train_one_epoch_cls._scaler
            with autocast('cuda'):
                outputs = model(x_for_dstagnn)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels) / ACCUM_STEPS
            scaler.scale(loss).backward()

            if step % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(x_for_dstagnn)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels) / ACCUM_STEPS
            loss.backward()
            if step % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUM_STEPS
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

    if is_distributed:
        tensor = torch.tensor([total_loss, num_batches],
                              dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = tensor.tolist()

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def evaluate_cls(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="评估中", leave=False,
                    disable=(is_distributed and rank != 0))
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            x_for_dstagnn = inputs.permute(0, 2, 1).unsqueeze(2)

            outputs = model(x_for_dstagnn)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0)

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
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0

    if is_distributed:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(hours=12))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if dist.get_rank() == 0:
            print(f"[DDP] 已启用分布式训练。World Size: {dist.get_world_size()}, "
                  f"Local Rank: {local_rank}")
    else:
        device = torch.device(DEVICE)

    if not is_distributed or dist.get_rank() == 0:
        print(f"使用设备: {device}")
        if ACCUM_STEPS > 1:
            print(f"[梯度累积] 已启用，累积步数: {ACCUM_STEPS}。")

    # 1. 加载数据
    try:
        processed_data = load_and_window_nifea_data(
            file_path=DATA_NPZ,
            num_channels=NUM_CHANNELS,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE
        )
    except FileNotFoundError:
        if not is_distributed or dist.get_rank() == 0:
            print(f"[错误] 未找到 NPZ 文件: {DATA_NPZ}")
            print("[提示] 请先运行 data_process.py 生成该文件。")
        return
    except Exception as e:
        if not is_distributed or dist.get_rank() == 0:
            print(f"[错误] 数据加载或窗口化失败: {e}")
        return

    # 2. K 折 DataLoader
    try:
        folds = get_kfold_dataloaders(
            processed_data, n_splits=N_SPLITS,
            batch_size=BATCH_SIZE,
            distributed=is_distributed,
            rank=(dist.get_rank() if is_distributed else 0),
            world_size=(dist.get_world_size() if is_distributed else 1),
            num_workers=4, pin_memory=True
        )
    except ValueError as e:
        if not is_distributed or dist.get_rank() == 0:
            print(f"[错误] 设置 K-Fold 失败: {e}")
        return

    adj_mx = np.ones((NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)
    all_fold_metrics = []

    # 3. K 折循环
    for fold_idx, (train_loader, test_loader) in enumerate(folds):
        if not is_distributed or dist.get_rank() == 0:
            print(f"\n===== 折叠 {fold_idx + 1}/{N_SPLITS} =====")

        model = make_model(
            DEVICE=device, num_of_d_initial_feat=1, nb_block=NB_BLOCK,
            initial_in_channels_cheb=1, K_cheb=K_CHEB,
            nb_chev_filter=NB_CHEV_FILTER,
            nb_time_filter_block_unused=32, initial_time_strides=1,
            adj_mx=adj_mx, adj_pa_static=adj_mx,
            adj_TMD_static_unused=np.zeros_like(adj_mx),
            num_for_predict_per_node=1, len_input_total=WINDOW_SIZE,
            num_of_vertices=NUM_CHANNELS,
            d_model_for_spatial_attn=D_MODEL_ATTN,
            d_k_for_attn=DSTAGNN_D_K_ATTN,
            d_v_for_attn=DSTAGNN_D_V_ATTN,
            n_heads_for_attn=N_HEADS_ATTN,
            output_memory=False, return_internal_states=False,
            task_type="classification", num_classes=NUM_CLASSES
        )

        # DDP / DP 包装
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,  # 现在模型里已经没有未使用参数了，可以关掉
            )
        elif torch.cuda.device_count() > 1:
            if not is_distributed or dist.get_rank() == 0:
                print(f"[多卡] DataParallel 已启用，GPU 数量: {torch.cuda.device_count()}。"
                      f"全局 batch={BATCH_SIZE} 将按GPU数均分。")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        train_one_epoch_cls._scaler = GradScaler('cuda', enabled=True)

        best_test_f1 = 0.0
        output_dir = f"fold_{fold_idx + 1}_results"
        if not is_distributed or dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)

        # 训练若干 epoch
        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            if is_distributed:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

            train_loss = train_one_epoch_cls(model, train_loader,
                                             optimizer, criterion, device)

            # 只在 rank0 上评估 & 打印
            if not is_distributed or dist.get_rank() == 0:
                test_metrics = evaluate_cls(model, test_loader,
                                            criterion, device)
                print(f"Epoch {epoch}/{EPOCHS_PER_FOLD} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_metrics['loss']:.4f}, "
                      f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")

                if test_metrics["f1_macro"] > best_test_f1:
                    best_test_f1 = test_metrics["f1_macro"]
                    best_model_path = os.path.join(output_dir, "best_model.pth")
                    state = (model.module.state_dict()
                             if isinstance(model, (DDP, nn.DataParallel))
                             else model.state_dict())
                    torch.save(state, best_model_path)
                    print(f"  -> 新的最佳模型已保存，测试集 F1-Score: {best_test_f1:.4f}")

        if is_distributed:
            dist.barrier()

        if not is_distributed or dist.get_rank() == 0:
            all_fold_metrics.append({"f1": best_test_f1})
            print(f"折叠 {fold_idx + 1} 完成。最佳 F1-Score: {best_test_f1:.4f}")

    # 4. 汇总 K 折 F1
    if not is_distributed or dist.get_rank() == 0:
        print("\n\n===== 交叉验证结果汇总 =====")
        f1_scores = [m["f1"] for m in all_fold_metrics]
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"所有折叠的最佳 F1-Scores: {[f'{s:.4f}' for s in f1_scores]}")
        print(f"平均 F1-Score (Macro): {avg_f1:.4f} ± {std_f1:.4f}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()