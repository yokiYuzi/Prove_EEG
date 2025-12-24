# run_2a_fhnet_ppSDE_8lead.py
# ============================================================
# BCICIV-2a 官方协议版：Train on Session T, Test on Session E
# DSTAGNN (可选 SDE) + 真实 EEG 10-20 拓扑 + AdamW + Cosine LR
#
# 本脚本已做“对照实验模块化”，你只需要在【实验开关】里改 4 个参数即可：
#   1) USE_SDE         : 是否启用 SDE（动态空间注意力）
#   2) USE_8_LEADS     : 是否仅使用 8 导联（缩减版）
#   3) USE_FILTERBANK  : 是否使用“滤波器组”分离频带（否则用 STFT 频带功率）
#   4) INPUT_SECONDS   : 输入信号长度（2s / 4s）
#
# 说明：
#   - 两种频带特征都输出 (B, C, N_BANDS, T_frames)，因此模型输入维度保持一致，
#     便于做对照实验。
#   - USE_FILTERBANK=True 时：先做 FIR bandpass，再做滑窗能量平均得到 bandpower；
#     USE_FILTERBANK=False 时：直接做 STFT bandpower（你当前脚本的默认做法）。
# ============================================================

import os
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from dataLoad.preprocess import get_data
from DSTAGNN_my1 import make_model


# ================== 实验开关（你只需要改这里 4 个） ==================
USE_SDE: bool = False                 # 1) 是否使用 SDE（动态空间注意力）
USE_8_LEADS: bool = False            # 2) 是否使用 8 导联（缩减版）
USE_FILTERBANK: bool = True         # 3) 是否使用滤波器组分离频带（False=STFT频带功率）
INPUT_SECONDS: float = 4.0           # 4) 输入长度（2.0 或 4.0）

# 输入裁剪方式（不是 4 个核心参数之一，但通常不需要改）
CROP_MODE: str = "start"             # "start" 或 "center"


# ================== 基础超参数（训练相关） ==================
NUM_CLASSES = 4
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
N_EPOCHS = int(os.environ.get("EPOCHS", "200"))
LR = 1e-3
VAL_RATIO = 0.1

# DSTAGNN 小模型参数
K_CHEB = 2
NB_BLOCK = 1
NB_CHEV_FILTER = 32
NB_TIME_FILTER_BLOCK_UNUSED = 32
D_MODEL_ATTN = 32
N_HEADS_ATTN = 2
DSTAGNN_D_K_ATTN = 8
DSTAGNN_D_V_ATTN = 8

# SDE 相关（不是 4 个核心参数之一，但你可能会做更细粒度消融）
SDE_INJECT_TO_GCN: bool = True           # SDE 的动态空间注意力是否混入 GCN 邻接权重
SDE_DYNAMIC_ALPHA: float = 0.5           # 混入比例：0=纯静态, 1=纯动态(时间平均)


# ================== 频带特征超参数 ==================
FS = 250  # BCICIV-2a 采样率

# 频带划分（你可以改）
BANDS = [(8, 12), (12, 16), (16, 20), (20, 28)]
N_BANDS = len(BANDS)

# STFT 参数（USE_FILTERBANK=False 时使用）
N_FFT = 256
WIN = 128
HOP = 32
CENTER = False

# FilterBank 参数（USE_FILTERBANK=True 时使用）
FIR_TAPS = 129          # FIR 滤波器长度（奇数更好）
FB_FRAME_LEN = N_FFT    # 滑窗能量的窗口长度；设为 N_FFT 可与 STFT 帧数保持一致

# 通用参数
EPS = 1e-6
BASE_FRAMES = 3         # baseline 帧数（前几帧均值作为 baseline）


# ================== EEG 通道真实 10-20 拓扑（BCICIV-2a 原始 22 导） ==================
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

# 你指定的 8 导联（顺序严格按你给定）
LEADS_8 = ["CP3", "C3", "CP4", "FC1", "C4", "P1", "FC2", "C1"]


def get_channels_used() -> list:
    """根据 USE_8_LEADS 返回当前使用的通道列表（其顺序即图节点顺序）。"""
    return LEADS_8 if USE_8_LEADS else CHANNELS_2A


def build_eeg_2a_adj(
    channels,
    connect_thresh: float = 1.5,
    self_loop: bool = True,
    pos_map: dict = CHAN_POS_2A,
) -> np.ndarray:
    """根据给定 channels 列表(及其在10-20拓扑中的坐标)构建邻接矩阵。"""
    channels = list(channels)
    name_to_idx = {ch: i for i, ch in enumerate(channels)}
    adj = np.zeros((len(channels), len(channels)), dtype=np.float32)

    missing = [ch for ch in channels if ch not in pos_map]
    if len(missing) > 0:
        raise ValueError(f"build_eeg_2a_adj: 这些通道在 CHAN_POS_2A 中找不到坐标: {missing}")

    for ch_i in channels:
        ri, ci = pos_map[ch_i]
        i = name_to_idx[ch_i]
        for ch_j in channels:
            if ch_i == ch_j:
                continue
            rj, cj = pos_map[ch_j]
            j = name_to_idx[ch_j]
            dist_ij = np.sqrt((ri - rj) ** 2 + (ci - cj) ** 2)
            if dist_ij <= connect_thresh:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    if self_loop:
        np.fill_diagonal(adj, 1.0)
    return adj


def ensure_trials_C_T(X: np.ndarray, n_total_channels: int) -> np.ndarray:
    """
    将 get_data 输出统一到 (Trials, C, T) 形式。
    支持两种常见形状：
      - (Trials, C, T)
      - (Trials, T, C)
    """
    if X.ndim != 3:
        raise ValueError(f"期望 3D 张量，但收到: {X.shape}")

    if X.shape[1] == n_total_channels:
        return X
    if X.shape[2] == n_total_channels:
        return np.transpose(X, (0, 2, 1))
    raise ValueError(
        f"无法识别通道维位置: X.shape={X.shape}，期望第二维或第三维为 {n_total_channels}。"
    )


def select_and_crop_channels(
    X: np.ndarray,
    channels_used: list,
    input_samples: int,
    crop_mode: str = "start",
) -> np.ndarray:
    """
    从原始 22 导数据中选择通道，并裁剪到指定输入长度。
    输出: (Trials, len(channels_used), input_samples)
    """
    X = ensure_trials_C_T(X, n_total_channels=len(CHANNELS_2A))

    missing = [ch for ch in channels_used if ch not in CHANNELS_2A]
    if len(missing) > 0:
        raise ValueError(f"这些导联不在 BCICIV-2a 的 22 导列表 CHANNELS_2A 中: {missing}")

    used_idx = [CHANNELS_2A.index(ch) for ch in channels_used]
    X = X[:, used_idx, :]  # (Trials, C_used, T)

    T = X.shape[-1]
    if T < input_samples:
        raise ValueError(
            f"原始时间长度 T={T} 小于 input_samples={input_samples}，无法裁剪。"
            f"请减小 INPUT_SECONDS 或检查数据。"
        )

    if crop_mode == "start":
        X = X[:, :, :input_samples]
    elif crop_mode == "center":
        start = (T - input_samples) // 2
        X = X[:, :, start:start + input_samples]
    else:
        raise ValueError(f"未知 crop_mode={crop_mode}，请用 'start' 或 'center'。")

    return X


def compute_t_frames(input_samples: int) -> int:
    """计算 STFT / FilterBank bandpower 的时间帧数（与 center 设置一致）。"""
    if input_samples < N_FFT:
        raise ValueError(f"input_samples={input_samples} 不能小于 N_FFT={N_FFT}（否则帧数为负）。")
    # frames = floor((L + pad - n_fft)/hop) + 1
    return (input_samples + (N_FFT if CENTER else 0) - N_FFT) // HOP + 1


# ================== 特征提取（STFT / FilterBank 二选一） ==================
class EEGFeatureExtractor:
    """
    输入:  (B, C, T)
    输出:  (B, C, N_BANDS, T_frames)
    """
    def __init__(self, fs: int, bands: list, use_filterbank: bool):
        self.fs = int(fs)
        self.bands = list(bands)
        self.use_filterbank = bool(use_filterbank)
        self._cache = {}

    @staticmethod
    def _hamming_window(numtaps: int, device, dtype):
        n = torch.arange(numtaps, device=device, dtype=dtype)
        return 0.54 - 0.46 * torch.cos(2.0 * torch.pi * n / (numtaps - 1))

    def _design_fir_bandpass(self, f0: float, f1: float, numtaps: int, device, dtype) -> torch.Tensor:
        """
        简单 windowed-sinc FIR bandpass（线性相位）。
        返回: (numtaps,)
        """
        if not (0.0 < f0 < f1 < self.fs / 2):
            raise ValueError(f"非法频带: ({f0},{f1})，需满足 0<f0<f1<fs/2={self.fs/2}")

        M = numtaps
        n = torch.arange(M, device=device, dtype=dtype) - (M - 1) / 2.0

        # bandpass = lowpass(f1) - lowpass(f0)
        # lowpass(fc): 2*fc/fs * sinc(2*fc*n/fs)
        h1 = 2.0 * (f1 / self.fs) * torch.sinc(2.0 * (f1 / self.fs) * n)
        h0 = 2.0 * (f0 / self.fs) * torch.sinc(2.0 * (f0 / self.fs) * n)
        h = h1 - h0

        w = self._hamming_window(M, device=device, dtype=dtype)
        h = h * w

        # 这里不做严格幅度归一化；后续会做 z-score，尺度影响不大。
        return h

    def _get_stft_window(self, device, dtype):
        key = ("hann", device.type, device.index, str(dtype), WIN)
        if key not in self._cache:
            self._cache[key] = torch.hann_window(WIN, device=device, dtype=dtype)
        return self._cache[key]

    def _get_filterbank_weight(self, C: int, device, dtype):
        key = ("fb_weight", C, device.type, device.index, str(dtype), tuple(self.bands), FIR_TAPS)
        if key in self._cache:
            return self._cache[key]

        kernels = []
        for (f0, f1) in self.bands:
            k = self._design_fir_bandpass(f0, f1, FIR_TAPS, device=device, dtype=dtype)  # (taps,)
            kernels.append(k)
        kernels = torch.stack(kernels, dim=0)              # (N_BANDS, taps)
        weight = kernels.repeat(C, 1).unsqueeze(1)         # (C*N_BANDS, 1, taps)

        self._cache[key] = weight
        return weight

    def stft_bandpower(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        STFT bandpower（你当前脚本的默认做法）
        inputs: (B, C, T)
        return: (B, C, N_BANDS, T_frames)
        """
        device = inputs.device
        dtype = inputs.dtype
        B, C, T = inputs.shape

        window = self._get_stft_window(device, dtype)
        x_2d = inputs.reshape(B * C, T)

        X = torch.stft(
            x_2d,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window=window,
            center=CENTER,
            return_complex=True
        )  # (B*C, Freq, T_frames)

        X = X.reshape(B, C, X.size(1), X.size(2))          # (B,C,Freq,T_frames)
        P = (X.abs() ** 2)

        freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / self.fs).to(device)

        feats = []
        for (f0, f1) in self.bands:
            idx = (freqs >= f0) & (freqs < f1)
            bp = P[:, :, idx, :].mean(dim=2)               # (B,C,T_frames)
            feats.append(bp)

        feats = torch.stack(feats, dim=2)                  # (B,C,N_BANDS,T_frames)
        feats = torch.log(feats + EPS)

        if feats.size(-1) >= BASE_FRAMES:
            base = feats[:, :, :, :BASE_FRAMES].mean(dim=-1, keepdim=True)
            feats = feats - base

        m = feats.mean(dim=-1, keepdim=True)
        s = feats.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        feats = (feats - m) / s
        return feats

    def filterbank_bandpower(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        FilterBank bandpower：
          1) FIR bandpass 得到不同频带信号
          2) 对每个频带做能量 (x^2)
          3) 用长度 FB_FRAME_LEN、步长 HOP 做滑窗平均 -> 得到 T_frames
          4) log + baseline + z-score

        inputs: (B, C, T)
        return: (B, C, N_BANDS, T_frames)
        """
        device = inputs.device
        dtype = inputs.dtype
        B, C, T = inputs.shape

        weight = self._get_filterbank_weight(C, device, dtype)  # (C*N_BANDS,1,taps)
        pad = FIR_TAPS // 2

        # group conv：每个通道独立卷积，并为每个通道输出 N_BANDS 个滤波结果
        y = F.conv1d(inputs, weight, bias=None, padding=pad, groups=C)  # (B, C*N_BANDS, T)
        y = y.view(B, C, N_BANDS, T)                                    # (B,C,N_BANDS,T)

        # 能量
        p = y * y                                                       # (B,C,N_BANDS,T)

        # 帧化：用 FB_FRAME_LEN 窗口、HOP 步长做平均
        if T < FB_FRAME_LEN:
            raise ValueError(f"T={T} < FB_FRAME_LEN={FB_FRAME_LEN}，请减小 N_FFT 或增大输入长度。")

        frames = p.unfold(dimension=-1, size=FB_FRAME_LEN, step=HOP)     # (B,C,N_BANDS,T_frames,FB_FRAME_LEN)
        bp = frames.mean(dim=-1)                                         # (B,C,N_BANDS,T_frames)

        feats = torch.log(bp + EPS)

        if feats.size(-1) >= BASE_FRAMES:
            base = feats[:, :, :, :BASE_FRAMES].mean(dim=-1, keepdim=True)
            feats = feats - base

        m = feats.mean(dim=-1, keepdim=True)
        s = feats.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        feats = (feats - m) / s
        return feats

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_filterbank:
            return self.filterbank_bandpower(inputs)
        return self.stft_bandpower(inputs)


# ================== 训练 & 验证 ==================
def train_one_epoch_eeg(model, loader, optimizer, criterion, device, feat_extractor: EEGFeatureExtractor):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_distributed = dist.is_initialized()

    for inputs, labels in loader:
        inputs = inputs.to(device)  # (B, C, T)
        labels = labels.to(device)

        x = feat_extractor(inputs)  # (B, C, N_BANDS, T_frames)

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


def evaluate_eeg(model, loader, criterion, device, feat_extractor: EEGFeatureExtractor):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            x = feat_extractor(inputs)

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

    # ----------------- 打印配置 -----------------
    channels_used = get_channels_used()
    num_channels = len(channels_used)
    input_samples = int(round(INPUT_SECONDS * FS))
    t_frames = compute_t_frames(input_samples)

    exp_tag = f"SDE{int(USE_SDE)}_8L{int(USE_8_LEADS)}_FB{int(USE_FILTERBANK)}_{int(INPUT_SECONDS)}s"
    feat_name = "FilterBank" if USE_FILTERBANK else "STFT"

    if rank == 0:
        print("=" * 70)
        print(f"使用设备: {device} | DDP={is_distributed} | world_size={world_size} | rank={rank}")
        print(f"[EXP] {exp_tag}")
        print(f"  - USE_SDE={USE_SDE} (inject_to_gcn={SDE_INJECT_TO_GCN}, alpha={SDE_DYNAMIC_ALPHA})")
        print(f"  - USE_8_LEADS={USE_8_LEADS} | NUM_CHANNELS={num_channels} | channels={channels_used}")
        print(f"  - USE_FILTERBANK={USE_FILTERBANK} | feature={feat_name} | bands={BANDS}")
        print(f"  - INPUT_SECONDS={INPUT_SECONDS} | input_samples={input_samples} | T_frames={t_frames}")
        print("=" * 70)

    # ----------------- 路径 -----------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataLoad", "BCICIV_2a") + os.sep
    save_root = os.path.join(script_dir, "eeg_bcic2a_dstagnn_ckpts")
    if rank == 0:
        os.makedirs(save_root, exist_ok=True)

    # ----------------- 1) 读取数据（Session T / Session E） -----------------
    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=data_dir, subject=subject, LOSO=False, data_type='2a'
    )   # X_train = Session T, X_test = Session E

    # ----------------- 2) 通道选择 & 时间裁剪（Train/Val/Test 全部一致） -----------------
    X_train = select_and_crop_channels(X_train, channels_used=channels_used,
                                      input_samples=input_samples, crop_mode=CROP_MODE)
    X_test = select_and_crop_channels(X_test, channels_used=channels_used,
                                     input_samples=input_samples, crop_mode=CROP_MODE)

    if rank == 0:
        print(f"Subject {subject} | Session T(after): {X_train.shape} | Session E(after): {X_test.shape}")

    # ----------------- 3) 从 Session T 划分 train / val -----------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    train_dataset = TensorDataset(torch.FloatTensor(X_train[train_idx]), torch.LongTensor(y_train[train_idx]))
    val_dataset   = TensorDataset(torch.FloatTensor(X_train[val_idx]),   torch.LongTensor(y_train[val_idx]))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test),            torch.LongTensor(y_test))

    # ----------------- 4) DataLoader -----------------
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ----------------- 5) 拓扑图 & 模型 -----------------
    adj_mx = build_eeg_2a_adj(channels=channels_used)

    model = make_model(
        DEVICE=device,
        num_of_d_initial_feat=N_BANDS,
        nb_block=NB_BLOCK,
        initial_in_channels_cheb=N_BANDS,
        K_cheb=K_CHEB,
        nb_chev_filter=NB_CHEV_FILTER,
        nb_time_filter_block_unused=NB_TIME_FILTER_BLOCK_UNUSED,
        initial_time_strides=1,
        adj_mx=adj_mx,
        adj_pa_static=adj_mx,
        adj_TMD_static_unused=np.zeros_like(adj_mx),
        num_for_predict_per_node=1,
        len_input_total=t_frames,
        num_of_vertices=num_channels,
        d_model_for_spatial_attn=D_MODEL_ATTN,
        d_k_for_attn=DSTAGNN_D_K_ATTN,
        d_v_for_attn=DSTAGNN_D_V_ATTN,
        n_heads_for_attn=N_HEADS_ATTN,
        output_memory=False,
        return_internal_states=False,
        task_type="classification",
        num_classes=NUM_CLASSES,
        # ===== 消融开关传入模型 =====
        use_sde=USE_SDE,
        use_dynamic_spatial_for_gcn=(SDE_INJECT_TO_GCN if USE_SDE else False),
        dynamic_spatial_alpha=SDE_DYNAMIC_ALPHA,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    # 特征提取器
    feat_extractor = EEGFeatureExtractor(fs=FS, bands=BANDS, use_filterbank=USE_FILTERBANK)

    # ----------------- 6) 训练 -----------------
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_root, f"sub{subject}", f"best_model_{exp_tag}.pth")
    if rank == 0:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch_eeg(model, train_loader, optimizer, criterion, device, feat_extractor)
        scheduler.step()

        if rank == 0:
            model_eval = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            val_metrics = evaluate_eeg(model_eval, val_loader, criterion, device, feat_extractor)

            print(f"[S{subject}][{exp_tag}] Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1_macro']:.4f}")

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                torch.save(model_eval.state_dict(), best_model_path)
                print(f"  → 新最佳模型已保存: {best_model_path} (Val F1 = {best_val_f1:.4f})")

    # ----------------- 7) 最终在 Session E 上测试 -----------------
    if rank == 0:
        final_model = make_model(
            DEVICE=device,
            num_of_d_initial_feat=N_BANDS,
            nb_block=NB_BLOCK,
            initial_in_channels_cheb=N_BANDS,
            K_cheb=K_CHEB,
            nb_chev_filter=NB_CHEV_FILTER,
            nb_time_filter_block_unused=NB_TIME_FILTER_BLOCK_UNUSED,
            initial_time_strides=1,
            adj_mx=adj_mx,
            adj_pa_static=adj_mx,
            adj_TMD_static_unused=np.zeros_like(adj_mx),
            num_for_predict_per_node=1,
            len_input_total=t_frames,
            num_of_vertices=num_channels,
            d_model_for_spatial_attn=D_MODEL_ATTN,
            d_k_for_attn=DSTAGNN_D_K_ATTN,
            d_v_for_attn=DSTAGNN_D_V_ATTN,
            n_heads_for_attn=N_HEADS_ATTN,
            task_type="classification",
            num_classes=NUM_CLASSES,
            use_sde=USE_SDE,
            use_dynamic_spatial_for_gcn=(SDE_INJECT_TO_GCN if USE_SDE else False),
            dynamic_spatial_alpha=SDE_DYNAMIC_ALPHA,
        ).to(device)

        final_model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_model.eval()

        test_metrics = evaluate_eeg(final_model, test_loader, criterion, device, feat_extractor)

        print("\n" + "=" * 60)
        print(f"Subject {subject} 最终结果（Session E 测试集，官方协议）")
        print(f"EXP     : {exp_tag}")
        print(f"Acc     : {test_metrics['accuracy']:.4f}")
        print(f"F1(macro): {test_metrics['f1_macro']:.4f}")
        print("=" * 60)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
