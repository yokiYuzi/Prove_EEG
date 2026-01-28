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
import random
import numpy as np
import math
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

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from dataLoad.preprocess_reref import get_data  # [MOD] 使用带重参考的预处理
from DSTAGNN_my1 import make_model


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seed."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ================== [MOD] 重参考（rereference）开关 ==================
# 说明：x' = x - x_ref（例如以 Cz 为参考），建议在标准化之前做（preprocess_reref 已实现）。
USE_REREF: bool = True
REREF_CHANNEL = "Cz"
DROP_REF_CHANNEL: bool = False  # True 会让通道数减少（如 22->21），需要同步修改通道列表/邻接构图（见文末说明）


# ================== 实验开关（你只需要改这里 4 个） ==================
USE_SDE: bool = True                 # 1) 是否使用 SDE（动态空间注意力）
USE_8_LEADS: bool = False            # 2) 是否使用 8 导联（缩减版）
USE_FILTERBANK: bool = True         # 3) 是否使用滤波器组分离频带（False=STFT频带功率）
INPUT_SECONDS: float = 4.0           # 4) 输入长度（2.0 或 4.0）

# ================== Lead-Gated Attention（方案1） ==================
# 目的：在空间注意力 logits 的 softmax 前加入“导联重要性先验 g”，抑制“注意力平均化”；
# 同时需要避免 g / 空间注意力过度塌缩（只押单一导联），否则会伤害跨 session 泛化。
USE_LEAD_GATING: bool = True           # 是否启用导联门控

# (1) 引导强度（建议配合 warmup；不要一上来就很强）
LEAD_GATE_BETA: float = 1.0           # key-bias 强度：score(i->j) += beta * log(g_j)
LEAD_GATE_GAMMA: float = 0.0          # 可选边门控：score(i->j) += gamma * (g_i * g_j)

# (2) g 的温度与 log 截断：让引导更“温和”，避免把某些导联直接判死刑
LEAD_GATE_TEMPERATURE: float = 2.0    # g = softmax(score / tau)，tau>1 更平滑
LEAD_GATE_G_MIN: float = 5e-3         # 计算 log(g) 时的下限 clamp，避免 log(0)
LEAD_GATE_HIDDEN: int = 32            # 门控 MLP 隐层宽度

# (3) warmup：前若干 epoch 先学基本表征，再逐步打开引导（避免早期押注捷径）
LEAD_GATE_WARMUP_EPOCHS: int = 40     # 0~40: beta 从 0 线性升到 LEAD_GATE_BETA

# (4) 熵带宽正则：既防平均（太大），也防塌缩（太小）
# 解释：entropy 的 exp(H) 可以理解为“有效导联数”。
USE_LEAD_G_ENTROPY_BAND_REG: bool = True
LEAD_G_ENTROPY_EFF_MIN: int = 4       # 希望 g 至少参考 ~4 个导联
LEAD_G_ENTROPY_EFF_MAX: int = 10      # 希望 g 不要扩散到太多导联
LEAD_G_ENTROPY_LAMBDA: float = 0.02   # 正则系数（建议 0.005~0.05 搜索）

USE_SPATIAL_A_ENTROPY_BAND_REG: bool = True
SPATIAL_A_ENTROPY_EFF_MIN: int = 4    # 空间注意力每一行至少覆盖 ~4 个 key
SPATIAL_A_ENTROPY_EFF_MAX: int = 10   # 但不要太接近均匀
SPATIAL_A_ENTROPY_LAMBDA: float = 0.02

# (5) 训练时的 channel dropout（空间增强）：逼模型学到“多导联冗余稳健表示”
CHANNEL_DROPOUT_P: float = 0.2        # 每个样本随机丢弃 20% 导联（仅训练）
CHANNEL_DROPOUT_RESCALE: bool = True  # 是否按 (1-p) 反向缩放以保持期望幅值

# ================== 注意力平均化(Attention Averaging) 监控打印 ==================
# 说明：
#   - 若注意力趋向平均化：entropy 接近 log(N)、l2/kl 接近 0、max_weight 接近 1/N
PRINT_ATTN_AVG_STATS: bool = True
ATTN_AVG_CHECK_EVERY: int = 10       # 每隔多少 epoch 打印一次（建议 5~20）
ATTN_AVG_MAX_BATCHES: int = 10       # 统计时最多跑多少个 batch（越大越准，越慢）

# 输入裁剪方式（不是 4 个核心参数之一，但通常不需要改）
CROP_MODE: str = "start"             # "start" 或 "center"


# ================== 基础超参数（训练相关） ==================
NUM_CLASSES = 4
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
N_EPOCHS = int(os.environ.get("EPOCHS", "200"))
LR = 1e-3
VAL_RATIO = 0.1

# Reproducibility
SEED: int = int(os.environ.get("SEED", "42"))


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
    # [MOD] 与 BCICIV-2a (22 EEG) 常见官方通道顺序保持一致
    # Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
    # CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

CHAN_POS_2A = {
    # [MOD] 10-20 拓扑坐标（用于构图）。这里只需要相对位置合理即可。
    # 采用简单网格：行从前到后(0->5)，列从左到右（允许出现 -1 和 5 表示更外侧的 C5/C6）。
    "Fz":  (0, 2),
    "FC3": (1, 0), "FC1": (1, 1), "FCz": (1, 2), "FC2": (1, 3), "FC4": (1, 4),
    "C5":  (2, -1), "C3":  (2, 0), "C1":  (2, 1), "Cz":  (2, 2), "C2":  (2, 3), "C4":  (2, 4), "C6":  (2, 5),
    "CP3": (3, 0), "CP1": (3, 1), "CPz": (3, 2), "CP2": (3, 3), "CP4": (3, 4),
    "P1":  (4, 1), "Pz":  (4, 2), "P2":  (4, 3),
    "POz": (5, 2),
}

# 你指定的 8 导联（顺序严格按你给定）
LEADS_8 = ["CP3", "C3", "CP4", "FC1", "C4", "P1", "FC2", "C1"]


# ================== [MOD] 数据中真实的通道顺序（用于索引/裁剪） ==================
# get_data(...) 返回的 X 默认是 22 通道 CHANNELS_2A 的顺序；
# 若在 preprocess_reref.get_data(..., drop_ref=True) 删除了参考通道，则这里也会同步少 1 个通道。
DATA_CHANNELS_2A = list(CHANNELS_2A)
if USE_REREF and DROP_REF_CHANNEL:
    if REREF_CHANNEL not in DATA_CHANNELS_2A:
        raise ValueError(
            f"[Config Error] REREF_CHANNEL='{REREF_CHANNEL}' 不在 CHANNELS_2A 中，无法 drop。"
        )
    DATA_CHANNELS_2A.remove(REREF_CHANNEL)


def get_channels_used() -> list:
    """
    根据 USE_8_LEADS 返回当前使用的通道列表（其顺序即图节点顺序）。

    [MOD] 当使用单点重参考 X' = X - X_ref 时：
      - 参考通道本身会变为全 0（见 preprocess_reref 说明）
      - 作为模型输入/图节点没有意义，因此这里默认把该参考通道从 channels_used 中移除
    """
    chs = list(LEADS_8) if USE_8_LEADS else list(DATA_CHANNELS_2A)

    # 单点重参考下，ref 通道=0（若你不 drop_ref），所以不作为模型输入
    if USE_REREF and (REREF_CHANNEL in chs):
        chs = [c for c in chs if c != REREF_CHANNEL]

    # sanity check
    if len(chs) != len(set(chs)):
        raise ValueError(f"channels_used 内存在重复通道: {chs}")

    return chs


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
    X = ensure_trials_C_T(X, n_total_channels=len(DATA_CHANNELS_2A))

    missing = [ch for ch in channels_used if ch not in DATA_CHANNELS_2A]
    if len(missing) > 0:
        raise ValueError(f"这些导联不在当前数据通道列表 DATA_CHANNELS_2A 中: {missing}")

    used_idx = [DATA_CHANNELS_2A.index(ch) for ch in channels_used]
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
    """
    Train for one epoch on Session T.

    Key add-ons for cross-session generalization:
      1) Channel dropout (space augmentation) on raw EEG
      2) Entropy band regularization to avoid:
           - attention too uniform (average)
           - attention too collapsed (single-lead shortcut)
    """
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    total_correct = 0
    total_samples = 0

    is_distributed = dist.is_available() and dist.is_initialized()

    # precompute entropy bounds (natural log)
    need_reg = bool((USE_LEAD_G_ENTROPY_BAND_REG and USE_LEAD_GATING) or USE_SPATIAL_A_ENTROPY_BAND_REG)
    H_g_low = math.log(max(2, LEAD_G_ENTROPY_EFF_MIN))
    H_g_high = math.log(max(2, LEAD_G_ENTROPY_EFF_MAX))
    H_a_low = math.log(max(2, SPATIAL_A_ENTROPY_EFF_MIN))
    H_a_high = math.log(max(2, SPATIAL_A_ENTROPY_EFF_MAX))

    # unwrap to access buffers (adj, masks) for spatial-attn reg
    base_model = model
    if isinstance(model, (DDP, nn.DataParallel)):
        base_model = model.module

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)  # (B,C,T)
        labels = labels.to(device, non_blocking=True)

        # --------- (A) Channel dropout on raw EEG (train only) ---------
        if CHANNEL_DROPOUT_P > 0:
            B, C, T = inputs.shape
            keep = (torch.rand((B, C), device=inputs.device) > CHANNEL_DROPOUT_P).float()
            # avoid pathological all-drop for any sample
            all_drop = (keep.sum(dim=1) < 1.0)
            if all_drop.any():
                keep[all_drop, torch.randint(0, C, (int(all_drop.sum().item()),), device=inputs.device)] = 1.0
            keep = keep.unsqueeze(-1)  # (B,C,1)
            inputs = inputs * keep
            if CHANNEL_DROPOUT_RESCALE and CHANNEL_DROPOUT_P < 1.0:
                inputs = inputs / (1.0 - CHANNEL_DROPOUT_P)

        # --------- feature extraction (B,C,T) -> (B,C,N_BANDS,T_frames) ---------
        x = feat_extractor(inputs)

        optimizer.zero_grad(set_to_none=True)

        # --------- forward (need internal states only when using entropy reg) ---------
        outputs = model(x, return_internal_states=True) if need_reg else model(x)
        if isinstance(outputs, tuple):
            logits, internal_states = outputs
        else:
            logits, internal_states = outputs, None

        loss_ce = criterion(logits, labels)

        # --------- (B) Entropy band regularization ---------
        loss_reg = torch.zeros((), device=device)

        if need_reg and internal_states is not None:
            # (B1) lead-g entropy band (encourage "small subset" rather than one-hot)
            if USE_LEAD_G_ENTROPY_BAND_REG and USE_LEAD_GATING:
                # use last block's g
                g = internal_states[-1].get("lead_gate_g", None)
                if g is not None:
                    g = torch.clamp(g, min=1e-12)
                    g = g / g.sum(dim=-1, keepdim=True)
                    H = -(g * torch.log(g)).sum(dim=-1)  # (B,)
                    band = F.relu(torch.tensor(H_g_low, device=device) - H) ** 2 + F.relu(H - torch.tensor(H_g_high, device=device)) ** 2
                    loss_reg = loss_reg + LEAD_G_ENTROPY_LAMBDA * band.mean()

            # (B2) spatial attention entropy band on the effective logits used by GCN
            if USE_SPATIAL_A_ENTROPY_BAND_REG:
                # each block may have multiple heads/orders K
                for bi, st in enumerate(internal_states):
                    sat_logits = st.get("sat_scores_for_gcn", None)  # (B,K,N,N) logits
                    if sat_logits is None:
                        continue
                    Bk, K, N, _ = sat_logits.shape
                    block = base_model.BlockList[bi]
                    adj = block.adj_pa_static.to(device)
                    for k in range(K):
                        mask_k = block.cheb_conv_SAt.mask_per_k[k].to(device)
                        combined = sat_logits[:, k] + (adj * mask_k).unsqueeze(0)   # (B,N,N)
                        A = torch.softmax(combined, dim=-1)                         # (B,N,N)
                        A = torch.clamp(A, min=1e-12)
                        A = A / A.sum(dim=-1, keepdim=True)
                        H = -(A * torch.log(A)).sum(dim=-1)                         # (B,N)
                        band = F.relu(torch.tensor(H_a_low, device=device) - H) ** 2 + F.relu(H - torch.tensor(H_a_high, device=device)) ** 2
                        loss_reg = loss_reg + SPATIAL_A_ENTROPY_LAMBDA * band.mean()

        loss = loss_ce + loss_reg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)
            total_correct += correct
            total_samples += bs
            total_loss += loss.item() * bs
            total_reg += loss_reg.item() * bs

    # DDP reduce
    if is_distributed:
        loss_tensor = torch.tensor([total_loss, total_reg, total_correct, total_samples], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_reg, total_correct, total_samples = loss_tensor.tolist()

    avg_loss = total_loss / max(total_samples, 1)
    avg_reg = total_reg / max(total_samples, 1)
    avg_acc  = total_correct / max(total_samples, 1)

    return avg_loss, avg_acc, avg_reg


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


# -----------------------------------------------------------------------------
# Lead-gate warmup schedule (to avoid early collapse)
# -----------------------------------------------------------------------------
def set_lead_gate_schedule(model, epoch: int):
    """Update lead-gate hyperparameters inside the model blocks."""
    if not USE_LEAD_GATING:
        beta = 0.0
        gamma = 0.0
        tau = float(LEAD_GATE_TEMPERATURE)
    else:
        warm = max(1, int(LEAD_GATE_WARMUP_EPOCHS))
        factor = min(1.0, float(epoch) / float(warm))
        beta = float(LEAD_GATE_BETA) * factor
        gamma = float(LEAD_GATE_GAMMA) * factor
        tau = float(LEAD_GATE_TEMPERATURE)

    base_model = model
    if isinstance(model, (DDP, nn.DataParallel)):
        base_model = model.module

    if hasattr(base_model, "BlockList"):
        for blk in base_model.BlockList:
            if hasattr(blk, "lead_gate_beta"):
                blk.lead_gate_beta = beta
            if hasattr(blk, "lead_gate_gamma"):
                blk.lead_gate_gamma = gamma
            if hasattr(blk, "lead_gate_temperature"):
                blk.lead_gate_temperature = tau

    return beta, gamma, tau


# ================== 主函数（官方协议版） ==================

# -----------------------------------------------------------------------------
# Attention averaging analysis (空间注意力平均化定量监控)
# -----------------------------------------------------------------------------

@dataclass
class SpatialAttnAveragingStats:
    mean_entropy: float
    mean_l2_to_uniform: float
    mean_kl_to_uniform: float
    mean_max_weight: float


def _entropy_torch(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0)
    return -(p * torch.log(p)).sum(dim=-1)


def _l2_to_uniform_torch(p: torch.Tensor) -> torch.Tensor:
    # p: (..., N)
    n = p.size(-1)
    u = 1.0 / float(n)
    return torch.sqrt(((p - u) ** 2).mean(dim=-1))


def _kl_to_uniform_torch(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # KL(p || uniform)
    p = p.clamp(min=eps, max=1.0)
    n = p.size(-1)
    log_u = -math.log(float(n))
    return (p * (torch.log(p) - log_u)).sum(dim=-1)


@torch.no_grad()
def collect_spatial_attention_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feat_extractor: "EEGFeatureExtractor",
    max_batches: int = 10,
) -> Dict[str, Any]:
    """收集 DSTAGNN 空间注意力（实际用于 GCN 的归一化邻接权重）的“平均化”统计量。

    统计对象：
      A = softmax( sat_scores_for_gcn + adj_pa_static * mask_k )

    若出现注意力平均化（更“均匀”）：
      - entropy 接近 log(N)
      - l2/kl_to_uniform 接近 0
      - max_weight 接近 1/N
    """
    model.eval()

    # unwrap DataParallel / DDP
    base_model = model.module if isinstance(model, (DDP, nn.DataParallel)) else model

    nb_block = getattr(base_model, "nb_block", None) or len(base_model.BlockList)
    num_vertices = int(getattr(base_model, "num_of_vertices", base_model.BlockList[0].adj_pa_static.size(0)))
    logN = math.log(float(num_vertices))
    invN = 1.0 / float(num_vertices)

    per_block = []

    # accumulators for global stats
    g_sum_entropy = 0.0
    g_sum_l2 = 0.0
    g_sum_kl = 0.0
    g_sum_max = 0.0
    g_count = 0

    for bi in range(nb_block):
        per_block.append(
            {
                "block": bi,
                "per_head": [],
                "mean": None,
                "lead_g": None,
            }
        )

    # Iterate batches (subset)
    for batch_i, batch in enumerate(loader):
        if batch_i >= max_batches:
            break
        inputs, _ = batch
        inputs = inputs.to(device)  # (B, C, T)
        x = feat_extractor(inputs)  # (B, C, N_BANDS, T_frames)

        out = base_model(x, return_internal_states=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("collect_spatial_attention_stats 需要模型 forward 返回 (logits, internal_states)")
        _, internal_states_list = out

        for bi in range(nb_block):
            st = internal_states_list[bi]
            sat_logits = st["sat_scores_for_gcn"].to(device)  # (B,K,N,N)
            B, K, N, N2 = sat_logits.shape
            assert N == num_vertices and N2 == num_vertices

            block = base_model.BlockList[bi]
            adj = block.adj_pa_static.to(device)  # (N,N)

            # lead gate stats (g)
            lead_g = st.get("lead_gate_g", None)
            if lead_g is not None and lead_g.numel() != 0:
                g = lead_g.to(device).float()  # (B,N)
                ent_g = _entropy_torch(g).mean().item()
                l2_g = _l2_to_uniform_torch(g).mean().item()
                kl_g = _kl_to_uniform_torch(g).mean().item()
                max_g = g.max(dim=-1).values.mean().item()
                per_block[bi]["lead_g"] = {
                    "entropy": ent_g,
                    "l2_to_uniform": l2_g,
                    "kl_to_uniform": kl_g,
                    "max_weight": max_g,
                    "note": f"uniform: entropy~log(N)={logN:.3f}, max~1/N={invN:.3f}",
                }

            # per head stats for effective adjacency attention
            head_stats = []
            for k in range(K):
                mask_k = block.cheb_conv_SAt.mask_per_k[k].to(device)  # (N,N)
                combined = sat_logits[:, k, :, :] + (adj * mask_k).unsqueeze(0)  # (B,N,N)

                A = torch.softmax(combined, dim=-1)  # (B,N,N) row-wise over keys
                # Flatten rows: (B*N, N)
                A_flat = A.reshape(-1, N)

                ent = _entropy_torch(A_flat).mean().item()
                l2u = _l2_to_uniform_torch(A_flat).mean().item()
                klu = _kl_to_uniform_torch(A_flat).mean().item()
                maxw = A_flat.max(dim=-1).values.mean().item()

                head_stats.append(
                    {
                        "head": k,
                        "entropy": ent,
                        "l2_to_uniform": l2u,
                        "kl_to_uniform": klu,
                        "max_weight": maxw,
                        "note": f"uniform: entropy~log(N)={logN:.3f}, max~1/N={invN:.3f}",
                    }
                )

                # global accumulate
                g_sum_entropy += ent
                g_sum_l2 += l2u
                g_sum_kl += klu
                g_sum_max += maxw
                g_count += 1

            per_block[bi]["per_head"] = head_stats

            # per block mean
            ent_m = float(np.mean([h["entropy"] for h in head_stats]))
            l2_m = float(np.mean([h["l2_to_uniform"] for h in head_stats]))
            kl_m = float(np.mean([h["kl_to_uniform"] for h in head_stats]))
            mx_m = float(np.mean([h["max_weight"] for h in head_stats]))
            per_block[bi]["mean"] = asdict(
                SpatialAttnAveragingStats(
                    mean_entropy=ent_m,
                    mean_l2_to_uniform=l2_m,
                    mean_kl_to_uniform=kl_m,
                    mean_max_weight=mx_m,
                )
            )

    if g_count == 0:
        raise RuntimeError("collect_spatial_attention_stats 未收集到任何 attention；请检查 loader")

    global_stats = SpatialAttnAveragingStats(
        mean_entropy=g_sum_entropy / float(g_count),
        mean_l2_to_uniform=g_sum_l2 / float(g_count),
        mean_kl_to_uniform=g_sum_kl / float(g_count),
        mean_max_weight=g_sum_max / float(g_count),
    )

    return {
        "global": asdict(global_stats),
        "per_block": per_block,
        "N": num_vertices,
        "logN": logN,
        "invN": invN,
        "note": "若注意力平均化：entropy 接近 log(N)、l2/kl 接近 0、max_weight 接近 1/N。",
    }


def pretty_print_attn_stats(stats: Dict[str, Any], prefix: str = "") -> None:
    g = stats["global"]
    N = stats["N"]
    logN = stats["logN"]
    invN = stats["invN"]
    print(f"{prefix}[AttnAvg] Global | N={N} logN={logN:.3f} 1/N={invN:.3f} | "
          f"entropy={g['mean_entropy']:.4f} l2={g['mean_l2_to_uniform']:.4f} "
          f"kl={g['mean_kl_to_uniform']:.4f} max={g['mean_max_weight']:.4f}")

    for b in stats["per_block"]:
        bm = b["mean"]
        print(f"{prefix}  Block{b['block']} mean | "
              f"entropy={bm['mean_entropy']:.4f} l2={bm['mean_l2_to_uniform']:.4f} "
              f"kl={bm['mean_kl_to_uniform']:.4f} max={bm['mean_max_weight']:.4f}")
        lg = b.get("lead_g", None)
        if lg is not None:
            print(f"{prefix}    lead-g | entropy={lg['entropy']:.4f} l2={lg['l2_to_uniform']:.4f} "
                  f"kl={lg['kl_to_uniform']:.4f} max={lg['max_weight']:.4f}")

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

    # ----------------- Seed -----------------
    seed_everything(SEED)

    # ----------------- 打印配置 -----------------
    channels_used = get_channels_used()
    num_channels = len(channels_used)
    input_samples = int(round(INPUT_SECONDS * FS))
    t_frames = compute_t_frames(input_samples)

    exp_tag = f"SDE{int(USE_SDE)}_8L{int(USE_8_LEADS)}_FB{int(USE_FILTERBANK)}_{int(INPUT_SECONDS)}s"
    feat_name = "FilterBank" if USE_FILTERBANK else "STFT"

    if rank == 0:
        print("=" * 70)
        print(f"使用设备: {device} | DDP={is_distributed} | world_size={world_size} | rank={rank} | seed={SEED}")
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
        path=data_dir,
        subject=subject,
        LOSO=False,
        data_type='2a',
        # [MOD] 重参考：x' = x - x_ref
        rereference=USE_REREF,
        ref_channel=REREF_CHANNEL,
        drop_ref=DROP_REF_CHANNEL,
    )   # X_train = Session T, X_test = Session E

    # ----------------- 2) 通道选择 & 时间裁剪（Train/Val/Test 全部一致） -----------------
    X_train = select_and_crop_channels(X_train, channels_used=channels_used,
                                      input_samples=input_samples, crop_mode=CROP_MODE)
    X_test = select_and_crop_channels(X_test, channels_used=channels_used,
                                     input_samples=input_samples, crop_mode=CROP_MODE)

    if rank == 0:
        print(f"Subject {subject} | Session T(after): {X_train.shape} | Session E(after): {X_test.shape}")

    # ----------------- 3) 从 Session T 划分 train / val -----------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    train_dataset = TensorDataset(torch.FloatTensor(X_train[train_idx]), torch.LongTensor(y_train[train_idx]))
    val_dataset   = TensorDataset(torch.FloatTensor(X_train[val_idx]),   torch.LongTensor(y_train[val_idx]))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test),            torch.LongTensor(y_test))

    # ----------------- 4) DataLoader -----------------
    # 为了尽量可复现：
    # - 非DDP: 使用 generator 固定 shuffle 的随机序列
    # - DDP: DistributedSampler 使用 seed 并在每个 epoch set_epoch
    dl_gen = torch.Generator()
    dl_gen.manual_seed(SEED)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False, seed=SEED)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(NUM_WORKERS > 0),
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=dl_gen,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(NUM_WORKERS > 0),
            worker_init_fn=seed_worker,
        )

    val_loader  = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )

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
        use_lead_gating=USE_LEAD_GATING,
        lead_gate_beta=LEAD_GATE_BETA,
        lead_gate_gamma=LEAD_GATE_GAMMA,
        lead_gate_temperature=LEAD_GATE_TEMPERATURE,
        lead_gate_g_min=LEAD_GATE_G_MIN,
        lead_gate_hidden=LEAD_GATE_HIDDEN,
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
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_metrics = None
    best_model_path = os.path.join(save_root, f"sub{subject}", f"best_model_{exp_tag}.pth")
    if rank == 0:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # warmup the lead-gate strength (avoid early collapse to single lead)
        beta_cur, gamma_cur, tau_cur = set_lead_gate_schedule(model, epoch)
        if rank == 0 and (epoch == 1 or epoch % 10 == 0):
            print(f"[S{subject}][{exp_tag}] [LeadGateSched] epoch={epoch} beta={beta_cur:.4f} gamma={gamma_cur:.4f} tau={tau_cur:.2f} g_min={LEAD_GATE_G_MIN}")

        train_loss, train_acc, train_reg = train_one_epoch_eeg(model, train_loader, optimizer, criterion, device, feat_extractor)
        scheduler.step()

        if rank == 0:
            model_eval = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            val_metrics = evaluate_eeg(model_eval, val_loader, criterion, device, feat_extractor)

            print(f"[S{subject}][{exp_tag}] Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} (reg {train_reg:.4f}) Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1_macro']:.4f}")

            if PRINT_ATTN_AVG_STATS and (epoch == 1 or epoch == N_EPOCHS or (epoch % ATTN_AVG_CHECK_EVERY == 0)):
                try:
                    attn_stats = collect_spatial_attention_stats(model_eval, val_loader, device, feat_extractor, max_batches=ATTN_AVG_MAX_BATCHES)
                    pretty_print_attn_stats(attn_stats, prefix=f"[S{subject}][{exp_tag}] ")
                except Exception as e:
                    print(f"[S{subject}][{exp_tag}][AttnAvg] warning: {e}")


            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)

                ckpt = {
                    "epoch": best_epoch,
                    "val_metrics": best_val_metrics,
                    "model_state": model_eval.state_dict(),
                    "exp_tag": exp_tag,
                    "seed": SEED,
                    "lead_gate": {
                        "beta": float(beta_cur),
                        "gamma": float(gamma_cur),
                        "tau": float(tau_cur),
                        "g_min": float(LEAD_GATE_G_MIN),
                    },
                }
                torch.save(ckpt, best_model_path)
                print(f"  → 新最佳模型已保存: {best_model_path} (BestEpoch={best_epoch} | Val F1 = {best_val_f1:.4f})")

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
            use_lead_gating=USE_LEAD_GATING,
            lead_gate_beta=LEAD_GATE_BETA,
            lead_gate_gamma=LEAD_GATE_GAMMA,
            lead_gate_temperature=LEAD_GATE_TEMPERATURE,
            lead_gate_g_min=LEAD_GATE_G_MIN,
            lead_gate_hidden=LEAD_GATE_HIDDEN,
        ).to(device)

        # Load best checkpoint and print the exact best epoch/val metrics
        ckpt = torch.load(best_model_path, map_location=device)
        if isinstance(ckpt, dict) and ("model_state" in ckpt):
            final_model.load_state_dict(ckpt["model_state"])
            loaded_best_epoch = int(ckpt.get("epoch", -1))
            loaded_best_val = ckpt.get("val_metrics", {})
            # restore lead-gate hyperparameters (they are buffers/attrs, not in state_dict)
            lg = ckpt.get("lead_gate", None)
            if isinstance(lg, dict):
                beta0 = float(lg.get("beta", LEAD_GATE_BETA))
                gamma0 = float(lg.get("gamma", LEAD_GATE_GAMMA))
                tau0 = float(lg.get("tau", LEAD_GATE_TEMPERATURE))
                gmin0 = float(lg.get("g_min", LEAD_GATE_G_MIN))
                for blk in final_model.BlockList:
                    if hasattr(blk, "lead_gate_beta"):
                        blk.lead_gate_beta = beta0
                    if hasattr(blk, "lead_gate_gamma"):
                        blk.lead_gate_gamma = gamma0
                    if hasattr(blk, "lead_gate_temperature"):
                        blk.lead_gate_temperature = tau0
                    if hasattr(blk, "lead_gate_g_min"):
                        blk.lead_gate_g_min = gmin0
        else:
            # Backward compatibility: old checkpoints may be plain state_dict
            final_model.load_state_dict(ckpt)
            loaded_best_epoch = int(best_epoch)
            loaded_best_val = best_val_metrics if best_val_metrics is not None else {}
            # approximate lead-gate attrs via the same schedule (for old checkpoints)
            set_lead_gate_schedule(final_model, loaded_best_epoch)

        if rank == 0:
            lb_f1 = loaded_best_val.get("f1_macro", None)
            lb_acc = loaded_best_val.get("acc", None)
            print(f"[S{subject}][{exp_tag}] Loaded best checkpoint: epoch={loaded_best_epoch} | val_acc={lb_acc} | val_f1={lb_f1}")
        final_model.eval()

        if PRINT_ATTN_AVG_STATS:
            try:
                attn_stats_test = collect_spatial_attention_stats(final_model, test_loader, device, feat_extractor, max_batches=ATTN_AVG_MAX_BATCHES)
                pretty_print_attn_stats(attn_stats_test, prefix=f"[S{subject}][{exp_tag}][TEST] ")
            except Exception as e:
                print(f"[S{subject}][{exp_tag}][TEST][AttnAvg] warning: {e}")

        test_metrics = evaluate_eeg(final_model, test_loader, criterion, device, feat_extractor)

        print("\n" + "=" * 60)
        print(f"Subject {subject} 最终结果（跨 session：Session T → Session E）")
        print(f"EXP     : {exp_tag}")
        if (lb_acc is not None) and (lb_f1 is not None):
            print(f"BestVal : epoch={loaded_best_epoch} | acc={float(lb_acc):.4f} | f1={float(lb_f1):.4f}  (Session T 验证集)")
        else:
            print(f"BestVal : epoch={loaded_best_epoch} (Session T 验证集)")
        print(f"Test    : acc={test_metrics['accuracy']:.4f} | f1={test_metrics['f1_macro']:.4f}  (Session E 测试集)")
        print("=" * 60)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
