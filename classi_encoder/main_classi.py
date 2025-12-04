# main_classi.py
# 说明:
#   - 本脚本已根据您的意见修改，用于驱动真实的NIFEA DB分类任务。
#   - 移除了占位数据函数，改为调用 nifea_data_utils.py 进行数据加载、窗口化和K-折划分。
#   - 更新了超参数以匹配真实数据处理流程。
#   - 增加了对 NPZ 文件的依赖和相应的错误处理。
#   - 已集成 DDP (分布式数据并行) 和 AMP (自动混合精度) 以支持多GPU训练。
#   - [新增] 添加了 DP (DataParallel) 作为多卡训练的回退选项。
#   - [新增] 添加了梯度累积功能，以实现更大的有效批处理大小。
#   - [新增] 集成了详细的可解释性特征导出和绘图流程，在每折训练后自动对测试集执行。
############################################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 导入修改后的模型和新的数据工具
from DSTAGNN_my import make_model
from nifea_data_utils import load_and_window_nifea_data, get_kfold_dataloaders
# DDP/AMP 导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from datetime import timedelta
cudnn.benchmark = True
# === 新增：导出与绘图工具 ===
from viz_time_features import (
    ensure_dir, save_sequences_npz, plot_time_features,
)

# === 新增：工具函数 ===
def unwrap_model(m):
    # 兼容单卡 / DP / DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m

# === 可选：限制每折最多导出的样本数（默认导出全部）
MAX_EXPORT_PER_FOLD = int(os.environ.get("MAX_EXPORT_PER_FOLD", "-1"))  # -1 表示全部


# ========== 超参数和全局设置 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4")) # 根据 GPU 显存调整
EPOCHS_PER_FOLD = int(os.environ.get("EPOCHS", "15")) # 完整训练建议 50+
LR = 1e-4
N_SPLITS = int(os.environ.get("N_SPLITS", "5")) # 5-Fold CV
NUM_CLASSES = 2 # 假设是二分类任务

# +++ [新增] 梯度累积步数，可通过环境变量覆盖 +++
# 例如: ACCUM_STEPS=4 python main_classi.py
ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1")) 

# 数据集参数 (根据 NIFEA DB 调整)
FS_MIN = 500
WINDOW_SIZE = FS_MIN * 2 
STEP_SIZE = FS_MIN * 2   #这里是间隔的步长，如果想要完全密闭，应该选择和选择步长相同的长度
NUM_CHANNELS = 6      # NIFEA DB 通常是 4腹部 + 1胸部

# 定义 NPZ 文件路径
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_NPZ = os.path.join(SCRIPT_DIR, "NIFEA_DB_6lead_500Hz_processed.npz")

# DSTAGNN 参数
K_CHEB = 2
NB_BLOCK = 2
NB_CHEV_FILTER = 64
D_MODEL_ATTN = 64
N_HEADS_ATTN = 4
DSTAGNN_D_K_ATTN = 16
DSTAGNN_D_V_ATTN = 16

# ========== 训练和评估函数 (已修改以支持梯度累积) ==========
def train_one_epoch_cls(model, loader, optimizer, criterion, device):
    # ===== 支持 DDP + AMP + 梯度累积 =====
    model.train() # 设置模型为训练模式
    total_loss = 0.0
    num_batches = 0
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    pbar = tqdm(loader, desc="训练中", leave=False, disable=(is_distributed and rank != 0))
    
    # 梯度累积：在循环外清零一次梯度
    optimizer.zero_grad(set_to_none=True)
    
    for step, (inputs, labels) in enumerate(pbar, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        x_for_dstagnn = inputs.permute(0, 2, 1).unsqueeze(2)

        # AMP: 使用 autocast 上下文管理器
        if hasattr(train_one_epoch_cls, "_scaler"):
            scaler = train_one_epoch_cls._scaler
            with autocast('cuda'):
                outputs = model(x_for_dstagnn)
                if isinstance(outputs, tuple): outputs = outputs[0]
                # 梯度累积：对损失进行缩放
                loss = criterion(outputs, labels) / ACCUM_STEPS
            
            # AMP: 缩放损失、反向传播
            scaler.scale(loss).backward()

            # 梯度累积：每 ACCUM_STEPS 步更新一次权重
            if step % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # 更新后清零梯度
        else: # 如果没有 scaler (例如 AMP 未启用)
            outputs = model(x_for_dstagnn)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels) / ACCUM_STEPS
            loss.backward()

            if step % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # 乘以 ACCUM_STEPS 以记录真实的 loss 大小
        total_loss += loss.item() * ACCUM_STEPS 
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

    # DDP: 聚合所有进程的损失
    if is_distributed:
        tensor = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, num_batches = tensor.tolist()
        
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, None, None

def evaluate_cls(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="评估中", leave=False, disable=(is_distributed and rank != 0))
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            x_for_dstagnn = inputs.permute(0, 2, 1).unsqueeze(2)

            outputs = model(x_for_dstagnn)
            # 对于 DP 模式，输出可能被包裹在 list 中
            if isinstance(outputs, tuple): outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'full_report': report
    }
    return metrics

# ========== [新增] 导出与绘图相关函数 ==========
@torch.no_grad()
def _extract_sde_sequences(plain_model, x):
    """
    从最后一个 block 抓取 SDE 动态空间注意力序列 sat_scores_seq，并派生节点级时间统计。
    输入:
      plain_model: 已 unwrap 的 DSTAGNN_submodule（非 DDP/DP 包裹）
      x: (B, N, 1, T) 到设备的张量
    输出:
      dict {
        'sat_scores_seq': (B, T, K, N, N)  logits（未 softmax）
        'node_entropy_seq': (B, T, N)      每个节点逐时刻熵（对 P(n->*) 取熵并对 head 平均）
        'node_inflow_seq': (B, T, N)       每个节点逐时刻“入流中心度” Σ_src P(src->node)（对 head 平均）
        'sde_embedding': (B, D)            SDEParallelFeatureHead 输出（如 64 维）
      }
    """
    # 手动过一遍 block，抓取最后一个 block 的 internal_states
    res_att_prev = 0
    cur = x
    states = None
    for blk in plain_model.BlockList:
        cur, res_att_prev, states = blk(cur, res_att_prev)

    sat_scores_seq = states.get("sat_scores_seq", None)  # (B, T, K, N, N)
    if sat_scores_seq is None:
        # 若未启用 SDE 分支，回零
        B, _, _, T = x.shape
        N = plain_model.num_of_vertices
        K = plain_model.K_cheb
        device = x.device
        sat_scores_seq = torch.zeros(B, T, K, N, N, device=device)

    # 概率化（沿最后一维 dst 做 softmax）
    P = torch.softmax(sat_scores_seq, dim=-1)            # (B, T, K, N, N)

    # 1) 源节点分布的熵：对 dst 取熵 -> (B,T,K,N) -> 平均 head -> (B,T,N)
    eps = 1e-8
    p_clamp = P.clamp_min(eps)
    ent = -(p_clamp * p_clamp.log()).sum(dim=-1)         # (B,T,K,N)
    node_entropy_seq = ent.mean(dim=2)                   # (B,T,N)

    # 2) 节点入流中心度：Σ_src P(src->dst) -> (B,T,K,N) -> 平均 head -> (B,T,N)
    node_inflow_seq = P.sum(dim=3).mean(dim=2)           # (B,T,N)

    # 3) SDE 并行特征向量
    sde_emb = plain_model.sde_head(sat_scores_seq)       # (B, D)

    return {
        "sat_scores_seq": sat_scores_seq.detach().cpu(),
        "node_entropy_seq": node_entropy_seq.detach().cpu(),
        "node_inflow_seq": node_inflow_seq.detach().cpu(),
        "sde_embedding": sde_emb.detach().cpu(),
    }

@torch.no_grad()
def export_and_plot_for_fold(model, loader, device, save_root, fs_hz, fold_idx):
    """
    对测试集逐窗导出：
      - 三类时间特征: tat_only / gtu_only / mixed
      - SDE 动态空间注意力: 原始 logits 序列 + 节点熵/入流序列 + 并行向量
      - 绘图: 叠加原始(标准化后)信号与上述序列，并标注窗口起止秒
    所有结果按文件夹分类保存到 save_root 下。
    """
    # 只在 rank 0 执行
    is_distributed = dist.is_initialized()
    if is_distributed and dist.get_rank() != 0:
        return

    plain = unwrap_model(model)
    plain.eval()

    # 目录结构
    seq_root = os.path.join(save_root, "sequences")
    sde_root = os.path.join(save_root, "sde")
    plot_root = os.path.join(save_root, "plots")
    for d in [
        os.path.join(seq_root, "tat_only"),
        os.path.join(seq_root, "gtu_only"),
        os.path.join(seq_root, "mixed"),
        os.path.join(sde_root, "raw_sat_scores_seq"),
        os.path.join(sde_root, "node_entropy_seq"),
        os.path.join(sde_root, "node_inflow_seq"),
        os.path.join(sde_root, "sde_embedding"),
        os.path.join(plot_root, "tat_only"),
        os.path.join(plot_root, "gtu_only"),
        os.path.join(plot_root, "mixed"),
        os.path.join(plot_root, "sde_entropy"),
        os.path.join(plot_root, "sde_inflow"),
    ]:
        ensure_dir(d)

    exported = 0
    # 逐 batch 导出
    for bidx, (inputs, labels) in enumerate(tqdm(loader, desc=f"[Fold {fold_idx}] 导出", leave=False)):
        # inputs: (B, T, C) —— 注意：这是标准化后的窗口
        B, T, C = inputs.shape
        x_for_dstagnn = inputs.to(device).permute(0, 2, 1).unsqueeze(2)  # (B, N, 1, T)

        # 1) 三类时间特征（模型内部API）
        seqs = plain.export_time_feature_sequences(x_for_dstagnn)
        tat_only = seqs["tat_only"].numpy()    # (B,N,T)
        gtu_only = seqs["gtu_only"].numpy()    # (B,N,T)
        mixed    = seqs["mixed"].numpy()       # (B,N,T)
        meta     = seqs["meta"]                # {'T':T, 'N':N}

        # 2) SDE 动态空间注意力及派生统计
        sde = _extract_sde_sequences(plain, x_for_dstagnn)
        sat_seq = sde["sat_scores_seq"].numpy()          # (B,T,K,N,N)
        node_entropy_seq = sde["node_entropy_seq"].numpy() # (B,T,N)
        node_inflow_seq  = sde["node_inflow_seq"].numpy()  # (B,T,N)
        sde_emb = sde["sde_embedding"].numpy()             # (B,D)

        # 3) 逐样本落盘 + 绘图
        inputs_np = inputs.numpy()   # (B,T,C) —— 归一化后信号
        labels_np = labels.numpy()

        for i in range(B):
            sample_tag = f"fold{fold_idx}_b{bidx}_i{i}_y{int(labels_np[i])}"
            # 3.1 保存 npz
            save_sequences_npz(
                base_dir=save_root,
                sample_tag=sample_tag,
                tat_only=tat_only[i],       # (N,T)
                gtu_only=gtu_only[i],       # (N,T)
                mixed=mixed[i],             # (N,T)
                sde_raw=sat_seq[i],         # (T,K,N,N)
                sde_node_entropy=node_entropy_seq[i].transpose(1,0), # -> (N,T) 便于统一
                sde_node_inflow=node_inflow_seq[i].transpose(1,0),   # -> (N,T)
                sde_embedding=sde_emb[i],   # (D,)
                fs=fs_hz,
                label=int(labels_np[i]),
                # 原始(标准化后)窗口，用于绘图叠加
                window_signal=inputs_np[i], # (T,C)
            )

            # 3.2 绘图（带起止秒标注）
            # x 轴统一 0~T/fs（相对该窗口）
            duration_sec = T / fs_hz
            start_sec, end_sec = 0.0, duration_sec

            # a) TAt-only
            plot_time_features(
                save_dir=os.path.join(plot_root, "tat_only"),
                sample_tag=sample_tag,
                window_signal=inputs_np[i],                 # (T,C)
                seq_per_node=tat_only[i],                  # (N,T)
                fs=fs_hz, start_sec=start_sec, end_sec=end_sec,
                title_prefix="TAt-only"
            )
            # b) GTU-only
            plot_time_features(
                save_dir=os.path.join(plot_root, "gtu_only"),
                sample_tag=sample_tag,
                window_signal=inputs_np[i],
                seq_per_node=gtu_only[i],
                fs=fs_hz, start_sec=start_sec, end_sec=end_sec,
                title_prefix="GTU-only"
            )
            # c) Mixed
            plot_time_features(
                save_dir=os.path.join(plot_root, "mixed"),
                sample_tag=sample_tag,
                window_signal=inputs_np[i],
                seq_per_node=mixed[i],
                fs=fs_hz, start_sec=start_sec, end_sec=end_sec,
                title_prefix="Mixed"
            )
            # d) SDE: 节点熵
            plot_time_features(
                save_dir=os.path.join(plot_root, "sde_entropy"),
                sample_tag=sample_tag,
                window_signal=inputs_np[i],
                seq_per_node=node_entropy_seq[i].transpose(1,0),  # (N,T)
                fs=fs_hz, start_sec=start_sec, end_sec=end_sec,
                title_prefix="SDE-NodeEntropy"
            )
            # e) SDE: 入流中心度
            plot_time_features(
                save_dir=os.path.join(plot_root, "sde_inflow"),
                sample_tag=sample_tag,
                window_signal=inputs_np[i],
                seq_per_node=node_inflow_seq[i].transpose(1,0),   # (N,T)
                fs=fs_hz, start_sec=start_sec, end_sec=end_sec,
                title_prefix="SDE-NodeInflow"
            )

            exported += 1
            if MAX_EXPORT_PER_FOLD > 0 and exported >= MAX_EXPORT_PER_FOLD:
                return

# ========== 主函数 ==========
def main():
    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0

    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if dist.get_rank() == 0:
            print(f"[DDP] 已启用分布式训练。World Size: {dist.get_world_size()}, Local Rank: {local_rank}")
    else:
        device = torch.device(DEVICE)
    
    if not is_distributed or dist.get_rank() == 0:
        print(f"使用设备: {device}")
        if ACCUM_STEPS > 1:
            print(f"[梯度累积] 已启用，累积步数: {ACCUM_STEPS}。")

    # 1. 加载和窗口化数据
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
    
    # 2. 设置 K-Fold CV 数据加载器
    try:
        folds = get_kfold_dataloaders(
            processed_data, n_splits=N_SPLITS, batch_size=BATCH_SIZE,
            distributed=is_distributed,
            rank=(dist.get_rank() if is_distributed else 0),
            world_size=(dist.get_world_size() if is_distributed else 1),
            num_workers=4, pin_memory=True
        )
    except ValueError as e:
        if not is_distributed or dist.get_rank() == 0:
            print(f"[错误] 设置 K-Fold 失败: {e}")
        return
    
    adj_mx = np.ones((NUM_CHANNELS, NUM_CHANNELS))
    all_fold_metrics = []

    # 3. K-Fold 循环
    for fold_idx, (train_loader, test_loader) in enumerate(folds):
        if not is_distributed or dist.get_rank() == 0:
            print(f"\n===== 折叠 {fold_idx+1}/{N_SPLITS} =====")

        model = make_model(
            DEVICE=device, num_of_d_initial_feat=1, nb_block=NB_BLOCK, 
            initial_in_channels_cheb=1, K_cheb=K_CHEB, nb_chev_filter=NB_CHEV_FILTER, 
            nb_time_filter_block_unused=32, initial_time_strides=1,
            adj_mx=adj_mx, adj_pa_static=adj_mx, adj_TMD_static_unused=np.zeros_like(adj_mx),
            num_for_predict_per_node=1, len_input_total=WINDOW_SIZE, num_of_vertices=NUM_CHANNELS,
            d_model_for_spatial_attn=D_MODEL_ATTN, d_k_for_attn=DSTAGNN_D_K_ATTN, 
            d_v_for_attn=DSTAGNN_D_V_ATTN, n_heads_for_attn=N_HEADS_ATTN,
            output_memory=False, return_internal_states=False, # 导出时手动前向，无需在训练中返回
            task_type='classification', num_classes=NUM_CLASSES
        )
        
        # ==== [修改] 多卡并行选择：优先 DDP；否则回退到 DP ====
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=False, find_unused_parameters=False)
        elif torch.cuda.device_count() > 1:
            if not is_distributed or dist.get_rank() == 0:
                 print(f"[多卡] DataParallel 已启用，GPU 数量: {torch.cuda.device_count()}。全局 batch={BATCH_SIZE} 将按GPU数均分。")
            model = nn.DataParallel(model)

        # 4. 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        train_one_epoch_cls._scaler = GradScaler('cuda', enabled=True)
        
        # 5. 训练循环
        best_test_f1 = 0.0
        output_dir = f"fold_{fold_idx+1}_results"
        if not is_distributed or dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)

        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
                
            train_loss, _, _ = train_one_epoch_cls(model, train_loader, optimizer, criterion, device)
            
            # 仅在 rank 0 进程执行评估、打印和模型保存
            if not is_distributed or dist.get_rank() == 0:
                test_metrics = evaluate_cls(model, test_loader, criterion, device)
                
                print(f"Epoch {epoch}/{EPOCHS_PER_FOLD} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_metrics['loss']:.4f}, Test F1: {test_metrics['f1_macro']:.4f}")
                
                if test_metrics['f1_macro'] > best_test_f1:
                    best_test_f1 = test_metrics['f1_macro']
                    best_model_path = os.path.join(output_dir, "best_model.pth")
                    # ==== [修改] 兼容 DDP/DP/单卡 的保存方式 ====
                    state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
                    torch.save(state, best_model_path)
                    print(f"  -> 新的最佳模型已保存，测试集 F1-Score: {best_test_f1:.4f}")
            # ===== [修复] 在导出前先全员同步一次（保证都已完成本折训练/评估）=====
        if is_distributed:
            dist.barrier()
            # ===== 在最佳模型训练完后：仅 rank 0 装载并做解释性导出与绘图 =====
        if not is_distributed or dist.get_rank() == 0:
            best_model_path = os.path.join(output_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                # 回载最佳权重
                plain_model = unwrap_model(model)
                state_dict = torch.load(best_model_path, map_location=device)
                plain_model.load_state_dict(state_dict, strict=True)

                # 创建本折“分析”目录
                analysis_root = os.path.join(output_dir, "analysis")
                ensure_dir(analysis_root)

                # 对测试集逐窗导出三类时间序列 + SDE 动态空间注意力 + 绘图
                print(f"开始对折叠 {fold_idx+1} 的最佳模型进行解释性特征导出...")
                export_and_plot_for_fold(
                    model=model,
                    loader=test_loader,
                    device=device,
                    save_root=analysis_root,
                    fs_hz=FS_MIN,
                    fold_idx=fold_idx + 1
                )
                print(f"折叠 {fold_idx+1} 的特征导出与绘图完成。")
            else:
                print("[警告] 未找到最佳模型权重，跳过本折的解释性导出。")
        # ===== [修复] 导出结束后再次全员同步（防止其它 rank 先进入下一折构造 DDP）=====
        if is_distributed:
            dist.barrier()

        if not is_distributed or dist.get_rank() == 0:
            all_fold_metrics.append({'f1': best_test_f1})
            print(f"折叠 {fold_idx+1} 完成。最佳 F1-Score: {best_test_f1:.4f}")

    # 6. 汇总交叉验证结果
    if not is_distributed or dist.get_rank() == 0:
        print("\n\n===== 交叉验证结果汇总 =====")
        f1_scores = [m['f1'] for m in all_fold_metrics]
        avg_f1 = np.mean(f1_scores); std_f1 = np.std(f1_scores)
        
        print(f"所有折叠的最佳 F1-Scores: {[f'{s:.4f}' for s in f1_scores]}")
        print(f"平均 F1-Score (Macro): {avg_f1:.4f} ± {std_f1:.4f}")
    
    # DDP: 清理进程组
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()