from __future__ import annotations

"""
main_transformer_encoder_attn_avg.py

Transformer 基础 Encoder 结构，用于在 BCICIV-2a 上验证“注意力平均化(Attention Averaging)”现象。

在你现有版本基础上新增：
1) 支持在 Session T 内部做 8:2 划分（train/test），用于“训练集内部检验”；
2) 可选择运行：
   - official : 官方协议 Train(Session T) -> Test(Session E)
   - within   : Session T 内部 8:2 划分
   - both     : 两者都跑（默认）
3) 对每个实验，输出：
   - Test acc、Test macro-F1、classification_report
   - 保存 predictions_*.csv
   - 注意力平均化证据（CLS attention mean + stats JSON）
   - （可选）也对 train/val 进行评估与保存（便于对比）

依赖（同目录或 PYTHONPATH 可见）：
  - preprocess.py 或 preprocess_reref.py（优先使用 preprocess_reref，如果存在）
  - LoadData.py

运行示例：
  # 同时跑 official + within (默认)
  python main_transformer_encoder_attn_avg.py --subject 1 --epochs 200 --batch_size 32

  # 只跑 Session T 的 8:2
  python main_transformer_encoder_attn_avg.py --subject 1 --exp_mode within --within_test_ratio 0.2

  # 只跑官方协议
  python main_transformer_encoder_attn_avg.py --subject 1 --exp_mode official

数据目录：
  你的工程常见结构：
    dataLoad/
      ├─ main_transformer_encoder_attn_avg.py
      ├─ preprocess.py / preprocess_reref.py
      ├─ LoadData.py
      └─ BCICIV_2a/
          ├─ s1/A01T.mat ...
          └─ ...

  默认会自动寻找 BCICIV_2a 根目录；也可用 --data_dir 显式指定。
"""

import os
import json
import math
import random
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Import your existing data loader / preprocessing utilities
#   - 优先 preprocess_reref.get_data（如果存在）
#   - 否则 fallback 到 preprocess.get_data
# -----------------------------------------------------------------------------
_GET_DATA = None
_PREPROCESS_NAME = None
_HAS_REREF_ARGS = False

try:
    from preprocess_reref import get_data as _get_data  # type: ignore
    _GET_DATA = _get_data
    _PREPROCESS_NAME = "preprocess_reref"
except Exception:
    try:
        from preprocess import get_data as _get_data  # type: ignore
        _GET_DATA = _get_data
        _PREPROCESS_NAME = "preprocess"
    except Exception as e:
        raise ImportError(
            "无法 import get_data。请确保 preprocess.py 或 preprocess_reref.py 与本脚本同目录，"
            "或其所在目录已加入 PYTHONPATH。"
        ) from e

# 检测 get_data 是否支持 rereference 参数（preprocess_reref 支持；preprocess 不支持）
try:
    import inspect
    _sig = inspect.signature(_GET_DATA)
    _HAS_REREF_ARGS = "rereference" in _sig.parameters
except Exception:
    _HAS_REREF_ARGS = False


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


# -----------------------------------------------------------------------------
# Utils: crop / pad
# -----------------------------------------------------------------------------
def crop_time(X: np.ndarray, target_len: int, mode: str = "start") -> np.ndarray:
    """Crop a (Trials, C, T) array to target_len on time axis."""
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (Trials,C,T), got {X.shape}")
    T = X.shape[-1]
    if T == target_len:
        return X
    if T < target_len:
        pad = target_len - T
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    if mode == "start":
        return X[:, :, :target_len]
    if mode == "center":
        start = (T - target_len) // 2
        return X[:, :, start : start + target_len]
    raise ValueError(f"Unknown crop mode: {mode}")


# -----------------------------------------------------------------------------
# Standardization (避免 Session T 内部 8:2 时的 test 泄漏)
# -----------------------------------------------------------------------------
def _ensure_3d_numpy(X: np.ndarray, name: str) -> np.ndarray:
    if isinstance(X, list):
        X = np.asarray(X)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"{name} must be 3D (Trials,Channels,Time), got {X.shape}")
    return X


def standardize_multiple(
    X_fit: np.ndarray,
    X_list: List[np.ndarray],
    mode: str = "channel_global",
    eps: float = 1e-6,
) -> List[np.ndarray]:
    """
    用 X_fit 学到的统计量，对 X_list 进行标准化并返回（顺序与输入一致）。

    mode:
      - channel_global：每通道 1 套 mean/std（在 fit 的 trial*time 上统计）
      - trial：每 trial、每通道在自身 time 上做 z-score（不依赖 X_fit）
      - timepoint_across_trials：旧行为（每个 timepoint 1 套统计），在 fit 的 trial 维拟合
    """
    X_fit = _ensure_3d_numpy(X_fit, "X_fit")
    X_list = [_ensure_3d_numpy(x, "X") for x in X_list]

    if mode == "channel_global":
        mean = X_fit.mean(axis=(0, 2), keepdims=True)  # (1,C,1)
        std = np.maximum(X_fit.std(axis=(0, 2), keepdims=True), eps)
        return [(x - mean) / std for x in X_list]

    if mode == "trial":
        out = []
        for x in X_list:
            mean = x.mean(axis=2, keepdims=True)
            std = np.maximum(x.std(axis=2, keepdims=True), eps)
            out.append((x - mean) / std)
        return out

    if mode == "timepoint_across_trials":
        C = int(X_fit.shape[1])
        scalers: List[StandardScaler] = []
        for j in range(C):
            sc = StandardScaler()
            sc.fit(X_fit[:, j, :])  # (Trials, Time)
            scalers.append(sc)

        out = []
        for x in X_list:
            x2 = x.copy()
            for j in range(C):
                x2[:, j, :] = scalers[j].transform(x2[:, j, :])
            out.append(x2)
        return out

    raise ValueError(f"Unknown standardize mode: {mode}")


# -----------------------------------------------------------------------------
# Transformer Encoder with attention weight outputs
# -----------------------------------------------------------------------------
def _mha_supports_avg_flag() -> bool:
    import inspect
    sig = inspect.signature(nn.MultiheadAttention.forward)
    return "average_attn_weights" in sig.parameters


class EncoderLayerWithAttn(nn.Module):
    """Vanilla Transformer Encoder layer that can return attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be gelu or relu")

        self._supports_avg_flag = _mha_supports_avg_flag()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, L, D)

        Return:
          x_out: (B, L, D)
          attn_w:
            - (B,H,L,L) if return_attn and supports_avg_flag
            - (B,L,L)   if return_attn and older pytorch (head-avg)
            - None      if return_attn=False
        """
        if return_attn:
            if self._supports_avg_flag:
                attn_out, attn_w = self.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
            else:
                attn_out, attn_w = self.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                )
        else:
            attn_out, attn_w = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff = self.linear2(self.dropout2(self.act(self.linear1(x))))
        x = self.norm2(x + ff)

        return x, attn_w


class EEGTransformerEncoderClassifier(nn.Module):
    """EEG -> patch tokens -> vanilla Transformer Encoder -> classification."""

    def __init__(
        self,
        n_channels: int,
        input_samples: int,
        patch_size: int,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")

        self.n_channels = int(n_channels)
        self.input_samples = int(input_samples)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.use_cls_token = bool(use_cls_token)

        self.n_patches = int(math.ceil(self.input_samples / self.patch_size))
        self.patch_embed = nn.Linear(self.n_channels * self.patch_size, d_model)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            seq_len = 1 + self.n_patches
        else:
            self.cls_token = None
            seq_len = self.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayerWithAttn(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,T) -> (B,n_patches,C*patch)"""
        B, C, T = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {C}")

        total_len = self.n_patches * self.patch_size
        if T < total_len:
            x = F.pad(x, (0, total_len - T), mode="constant", value=0.0)
        elif T > total_len:
            x = x[:, :, :total_len]

        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(B, self.n_patches, C * self.patch_size)
        return patches

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        patches = self._to_patches(x)
        tok = self.patch_embed(patches)

        if self.use_cls_token:
            cls = self.cls_token.expand(tok.size(0), -1, -1)
            tok = torch.cat([cls, tok], dim=1)

        tok = tok + self.pos_embed
        tok = self.pos_drop(tok)

        attn_all = [] if return_attn else None
        for layer in self.layers:
            tok, attn_w = layer(tok, return_attn=return_attn)
            if return_attn:
                attn_all.append(attn_w)

        tok = self.norm(tok)
        feat = tok[:, 0] if self.use_cls_token else tok.mean(dim=1)
        logits = self.head(feat)
        return logits, attn_all


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
def to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        x, y = to_device(batch, device)
        logits, _ = model(x, return_attn=False)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return {
        "loss": total_loss / max(1, n),
        "accuracy": report_dict["accuracy"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "report_dict": report_dict,
        "y_true": np.array(all_labels, dtype=np.int64),
        "y_pred": np.array(all_preds, dtype=np.int64),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x, y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, return_attn=False)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / max(1, n)


# -----------------------------------------------------------------------------
# Attention averaging analysis
# -----------------------------------------------------------------------------
@dataclass
class AttnAveragingStats:
    mean_entropy: float
    mean_l2_to_uniform: float
    mean_kl_to_uniform: float
    mean_max_weight: float


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


@torch.no_grad()
def collect_mean_attention(
    model: EEGTransformerEncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> np.ndarray:
    """Return mean attention per layer/head: (Layers, Heads(or1), L, L)."""
    model.eval()
    attn_sum = None
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, _ = to_device(batch, device)
        _, attn_list = model(x, return_attn=True)

        layer_attns = []
        for attn in attn_list:
            if attn is None:
                raise RuntimeError("return_attn=True but got None attention")
            if attn.dim() == 3:
                attn = attn.unsqueeze(1)  # (B,1,L,L)
            layer_attns.append(attn.detach().cpu().float().numpy())

        stacked = np.stack(layer_attns, axis=0)  # (Layers,B,H,L,L)

        if attn_sum is None:
            attn_sum = stacked.sum(axis=1)  # (Layers,H,L,L)
        else:
            attn_sum += stacked.sum(axis=1)

        count += stacked.shape[1]

    if attn_sum is None or count == 0:
        raise RuntimeError("No attention collected; check loader")

    return attn_sum / float(count)


def compute_attn_averaging_stats(mean_attn: np.ndarray, cls_index: int = 0) -> Dict:
    """Compute attention-averaging stats from mean attention (focus CLS query)."""
    if mean_attn.ndim != 4:
        raise ValueError(f"mean_attn must be 4D (layer,head,L,L), got {mean_attn.shape}")

    num_layers, num_heads, L, L2 = mean_attn.shape
    if L != L2:
        raise ValueError(f"mean_attn last two dims must be equal, got {L} vs {L2}")

    uniform = np.ones((L,), dtype=np.float64) / float(L)

    per_layer = []
    for li in range(num_layers):
        ent_list, l2_list, kl_list, max_list = [], [], [], []
        for hi in range(num_heads):
            p = mean_attn[li, hi, cls_index, :].astype(np.float64)
            p = p / max(1e-12, p.sum())
            ent_list.append(_entropy(p))
            l2_list.append(float(np.sqrt(((p - uniform) ** 2).mean())))
            kl_list.append(_kl(p, uniform))
            max_list.append(float(p.max()))

        per_layer.append(
            {
                "layer": li,
                "entropy": float(np.mean(ent_list)),
                "l2_to_uniform": float(np.mean(l2_list)),
                "kl_to_uniform": float(np.mean(kl_list)),
                "max_weight": float(np.mean(max_list)),
            }
        )

    global_stats = AttnAveragingStats(
        mean_entropy=float(np.mean([x["entropy"] for x in per_layer])),
        mean_l2_to_uniform=float(np.mean([x["l2_to_uniform"] for x in per_layer])),
        mean_kl_to_uniform=float(np.mean([x["kl_to_uniform"] for x in per_layer])),
        mean_max_weight=float(np.mean([x["max_weight"] for x in per_layer])),
    )

    return {
        "per_layer": per_layer,
        "global": asdict(global_stats),
        "note": "若注意力趋向平均化：entropy 接近 log(L)、l2/kl 接近 0、max_weight 接近 1/L。",
        "L": int(L),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
    }


# -----------------------------------------------------------------------------
# Path resolving
# -----------------------------------------------------------------------------
def resolve_bcic2a_root(user_data_dir: Optional[str]) -> str:
    """Return absolute path to directory that contains s1/, s2/, ... and description.pdf."""
    if user_data_dir is not None and str(user_data_dir).strip() != "":
        p = os.path.abspath(os.path.expanduser(user_data_dir))

        if os.path.isdir(os.path.join(p, "BCICIV_2a")) and os.path.isdir(os.path.join(p, "BCICIV_2a", "s1")):
            p = os.path.join(p, "BCICIV_2a")

        if not os.path.isdir(p):
            raise FileNotFoundError(f"--data_dir 指向的路径不存在或不是目录: {p}")

        if not os.path.isdir(os.path.join(p, "s1")):
            raise FileNotFoundError(f"--data_dir={p} 不是 BCICIV_2a 根目录（里面未找到 s1/）")

        return p

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    candidates = [
        os.path.join(script_dir, "BCICIV_2a"),
        os.path.join(script_dir, "dataLoad", "BCICIV_2a"),
        os.path.join(cwd, "BCICIV_2a"),
        os.path.join(cwd, "dataLoad", "BCICIV_2a"),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "s1")):
            return os.path.abspath(cand)

    tried = "\n".join([f"  - {os.path.abspath(x)}" for x in candidates])
    raise FileNotFoundError(
        "未能自动定位 BCICIV_2a 数据目录。已尝试：\n"
        f"{tried}\n\n"
        "请显式指定：\n"
        "  python main_transformer_encoder_attn_avg.py --data_dir ./BCICIV_2a --subject 1\n"
    )


# -----------------------------------------------------------------------------
# Splitting helpers
# -----------------------------------------------------------------------------
def stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx)."""
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {test_size}")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(y))
    train_idx, test_idx = next(sss.split(idx, y))
    return train_idx, test_idx


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------
def _save_predictions_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
            f.write(f"{int(yt)},{int(yp)}\n")


def _run_one_experiment(
    exp_name: str,
    X_train_full_raw: np.ndarray,
    y_train_full: np.ndarray,
    eval_sets_raw: Dict[str, Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: str,
    n_channels: int,
    target_len: int,
) -> Dict[str, Dict]:
    """
    exp_name: 'official' / 'within'
    X_train_full_raw: 用于训练的“全量训练集合”（之后会再从中切 train/val）
    eval_sets_raw: 需要评估的集合（比如 {'SessionE':(X_E,y_E)} 或 {'T_holdout':(...), 'SessionE':(...)}）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 标准化：fit 在 X_train_full_raw 上（不包含任何 eval_sets）
    keys = list(eval_sets_raw.keys())
    X_to_std = [X_train_full_raw] + [eval_sets_raw[k][0] for k in keys]
    X_std_list = standardize_multiple(X_train_full_raw, X_to_std, mode=args.standardize_mode)

    X_train_full = X_std_list[0]
    eval_sets = {k: (X_std_list[i + 1], eval_sets_raw[k][1]) for i, k in enumerate(keys)}

    # 2) 从 X_train_full 内部分 train/val（用于选 best）
    if args.val_ratio > 0.0:
        train_idx, val_idx = stratified_split_indices(y_train_full, test_size=args.val_ratio, seed=args.seed)
        has_val = True
    else:
        train_idx = np.arange(len(y_train_full))
        val_idx = np.array([], dtype=np.int64)
        has_val = False

    train_ds = TensorDataset(
        torch.tensor(X_train_full[train_idx], dtype=torch.float32),
        torch.tensor(y_train_full[train_idx], dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if has_val:
        val_ds = TensorDataset(
            torch.tensor(X_train_full[val_idx], dtype=torch.float32),
            torch.tensor(y_train_full[val_idx], dtype=torch.long),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        val_ds = None
        val_loader = None

    # 3) model
    model = EEGTransformerEncoderClassifier(
        n_channels=n_channels,
        input_samples=target_len,
        patch_size=args.patch_size,
        num_classes=4,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="gelu",
        use_cls_token=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 4) train
    best_val_f1 = -1.0
    ckpt_path = os.path.join(out_dir, "best_model.pth")

    print("\n" + "=" * 80)
    print(f"[EXPERIMENT] {exp_name}")
    print(f"Output dir: {out_dir}")
    print(f"Train_full size: {X_train_full.shape[0]} | Train/Val split: {len(train_idx)}/{len(val_idx)}")
    print(f"Eval sets: { {k: v[0].shape[0] for k, v in eval_sets.items()} }")
    print(f"Input: C={n_channels}, T={target_len}, patch={args.patch_size}, n_patches={model.n_patches}")
    print(f"Transformer: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
    print(f"Standardize mode: {args.standardize_mode} | preprocess used: {_PREPROCESS_NAME}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if has_val:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train loss {train_loss:.4f} | "
                f"Val loss {val_metrics['loss']:.4f} | "
                f"Val acc {val_metrics['accuracy']:.4f} | "
                f"Val F1(macro) {val_metrics['f1_macro']:.4f}"
            )
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                torch.save(model.state_dict(), ckpt_path)
        else:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train loss {train_loss:.4f}")

    if not has_val:
        torch.save(model.state_dict(), ckpt_path)

    # 5) load best/last and evaluate
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    results: Dict[str, Dict] = {}

    # 可选：也评估 train / val（方便观测过拟合）
    if args.eval_train:
        train_eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_full[train_idx], dtype=torch.float32),
                torch.tensor(y_train_full[train_idx], dtype=torch.long),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        train_metrics = evaluate(model, train_eval_loader, device)
        results["TrainSplit"] = train_metrics

        print("\n" + "-" * 80)
        print(f"[EVAL] TrainSplit")
        print(f"Acc      : {train_metrics['accuracy']:.4f}")
        print(f"F1(macro): {train_metrics['f1_macro']:.4f}")

        _save_predictions_csv(
            os.path.join(out_dir, "predictions_TrainSplit.csv"),
            train_metrics["y_true"],
            train_metrics["y_pred"],
        )

        if args.attn_on_train:
            mean_attn = collect_mean_attention(model, train_eval_loader, device=device, max_batches=args.attn_max_batches)
            stats = compute_attn_averaging_stats(mean_attn, cls_index=0)
            np.save(os.path.join(out_dir, "mean_attention_TrainSplit.npy"), mean_attn)
            with open(os.path.join(out_dir, "attention_stats_TrainSplit.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    if has_val and args.eval_val:
        val_eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_full[val_idx], dtype=torch.float32),
                torch.tensor(y_train_full[val_idx], dtype=torch.long),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        val_metrics_final = evaluate(model, val_eval_loader, device)
        results["ValSplit"] = val_metrics_final

        print("\n" + "-" * 80)
        print(f"[EVAL] ValSplit")
        print(f"Acc      : {val_metrics_final['accuracy']:.4f}")
        print(f"F1(macro): {val_metrics_final['f1_macro']:.4f}")

        _save_predictions_csv(
            os.path.join(out_dir, "predictions_ValSplit.csv"),
            val_metrics_final["y_true"],
            val_metrics_final["y_pred"],
        )

        if args.attn_on_val:
            mean_attn = collect_mean_attention(model, val_eval_loader, device=device, max_batches=args.attn_max_batches)
            stats = compute_attn_averaging_stats(mean_attn, cls_index=0)
            np.save(os.path.join(out_dir, "mean_attention_ValSplit.npy"), mean_attn)
            with open(os.path.join(out_dir, "attention_stats_ValSplit.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    # 对每个 eval set 输出 classification_report + attention stats
    for set_name, (X_ev, y_ev) in eval_sets.items():
        ds = TensorDataset(torch.tensor(X_ev, dtype=torch.float32), torch.tensor(y_ev, dtype=torch.long))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        metrics = evaluate(model, loader, device)
        results[set_name] = metrics

        report_str = classification_report(metrics["y_true"], metrics["y_pred"], digits=4, zero_division=0)

        print("\n" + "=" * 80)
        print(f"[TEST RESULTS] ({set_name})")
        print(f"Acc      : {metrics['accuracy']:.4f}")
        print(f"F1(macro): {metrics['f1_macro']:.4f}")
        print("\nclassification_report:")
        print(report_str)
        print("=" * 80)

        _save_predictions_csv(os.path.join(out_dir, f"predictions_{set_name}.csv"), metrics["y_true"], metrics["y_pred"])

        # attention averaging evidence
        mean_attn = collect_mean_attention(model, loader, device=device, max_batches=args.attn_max_batches)
        stats = compute_attn_averaging_stats(mean_attn, cls_index=0)

        np.save(os.path.join(out_dir, f"mean_attention_{set_name}.npy"), mean_attn)
        with open(os.path.join(out_dir, f"attention_stats_{set_name}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("\n[ATTENTION AVERAGING EVIDENCE] (CLS attention)")
        print(json.dumps(stats["global"], ensure_ascii=False, indent=2))

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--subject", type=int, default=1, help="BCICIV-2a subject id (1..9)")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="BCICIV_2a 根目录（里面应包含 s1/, s2/, ...）。不传则自动寻找。",
    )
    p.add_argument("--seed", type=int, default=42)

    # experiment selection
    p.add_argument(
        "--exp_mode",
        type=str,
        default="both",
        choices=["official", "within", "both"],
        help="official: Train(T)->Test(E); within: Session T 内部 8:2; both: 两者都跑",
    )
    p.add_argument(
        "--within_test_ratio",
        type=float,
        default=0.2,
        help="Session T 内部划分的 test 比例（8:2 即 0.2）",
    )

    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_ratio", type=float, default=0.1, help="在训练集合内部再划分 val 的比例（用于选 best）")

    # input
    p.add_argument(
        "--input_seconds",
        type=float,
        default=4.0,
        help="进一步裁剪长度(秒)。preprocess.get_data 默认裁剪 2s~6s(4s)，一般保持 4.0。",
    )
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--crop_mode", type=str, default="start", choices=["start", "center"])

    # transformer
    p.add_argument("--patch_size", type=int, default=25, help="Temporal patch size in samples")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # standardization
    p.add_argument(
        "--standardize_mode",
        type=str,
        default="channel_global",
        choices=["channel_global", "trial", "timepoint_across_trials"],
    )

    # preprocess_reref options (only if supported)
    p.add_argument("--reref", action="store_true", help="启用单点重参考（仅 preprocess_reref 支持）")
    p.add_argument("--ref_channel", type=str, default="Cz", help="重参考通道名（例如 Cz）")
    p.add_argument("--drop_ref", action="store_true", help="重参考后删除 ref 通道（通道数会减少 1）")

    # attention analysis
    p.add_argument("--attn_max_batches", type=int, default=20, help="计算 mean attention 使用多少个 batch（每个评估集合）")
    p.add_argument("--eval_train", action="store_true", help="额外评估 TrainSplit（不只是 test）")
    p.add_argument("--eval_val", action="store_true", help="额外评估 ValSplit（不只是 test）")
    p.add_argument("--attn_on_train", action="store_true", help="对 TrainSplit 也计算注意力证据（更慢）")
    p.add_argument("--attn_on_val", action="store_true", help="对 ValSplit 也计算注意力证据（更慢）")

    # output
    p.add_argument("--out_dir", type=str, default="./outputs_transformer_encoder")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) resolve data root
    data_root = resolve_bcic2a_root(args.data_dir)
    if not data_root.endswith(os.sep):
        data_root = data_root + os.sep

    # Fail-fast check
    sub_dir = os.path.join(data_root, f"s{args.subject}")
    expected_train_mat = os.path.join(sub_dir, f"A0{args.subject}T.mat")
    expected_test_mat = os.path.join(sub_dir, f"A0{args.subject}E.mat")
    if not (os.path.isdir(sub_dir) and os.path.exists(expected_train_mat) and os.path.exists(expected_test_mat)):
        raise FileNotFoundError(
            f"数据路径检查失败。\n"
            f"sub_dir={sub_dir}\n"
            f"expected_train_mat={expected_train_mat}\n"
            f"expected_test_mat ={expected_test_mat}\n"
        )

    # 2) load raw data (不在 get_data 内部做标准化，避免 within split 的 test 泄漏)
    get_data_kwargs = dict(
        path=data_root,
        subject=args.subject,
        LOSO=False,
        data_type="2a",
        isStandard=False,  # <-- 关键：我们自己标准化
    )

    # 如果当前 get_data 支持 rereference 参数，则允许用户开启
    if _HAS_REREF_ARGS:
        get_data_kwargs.update(
            dict(
                rereference=bool(args.reref),
                ref_channel=args.ref_channel,
                drop_ref=bool(args.drop_ref),
            )
        )
    else:
        if args.reref or args.drop_ref:
            print(
                "[WARN] 当前导入的 get_data 不支持 rereference 参数（可能使用的是 preprocess.py）。"
                "已忽略 --reref/--drop_ref。"
            )

    X_T, y_T, X_E, y_E, _, _ = _GET_DATA(**get_data_kwargs)

    # 3) further crop (统一输入长度)
    target_len = int(round(args.input_seconds * args.fs))
    if target_len > X_T.shape[-1]:
        raise ValueError(
            f"--input_seconds 对应长度 {target_len} > get_data 输出长度 {X_T.shape[-1]}。"
            f"请把 input_seconds <= {X_T.shape[-1] / args.fs:.2f}."
        )

    X_T = crop_time(X_T, target_len=target_len, mode=args.crop_mode)
    X_E = crop_time(X_E, target_len=target_len, mode=args.crop_mode)

    n_channels = int(X_T.shape[1])

    # 4) build output tag
    reref_tag = ""
    if _HAS_REREF_ARGS and args.reref:
        reref_tag = f"_reref-{args.ref_channel}_drop{int(args.drop_ref)}"
    base_out = os.path.join(args.out_dir, f"sub{args.subject}{reref_tag}")
    os.makedirs(base_out, exist_ok=True)

    print("=" * 80)
    print(f"[Device] {device}")
    print(f"[Data ] BCICIV_2a root: {data_root}")
    print(f"[Data ] preprocess used: {_PREPROCESS_NAME} | reref_supported={_HAS_REREF_ARGS}")
    print(f"[Data ] Session T: {X_T.shape} | Session E: {X_E.shape}")
    print(f"[Exp  ] exp_mode={args.exp_mode} | within_test_ratio={args.within_test_ratio} | val_ratio={args.val_ratio}")
    print("=" * 80)

    # 5) run experiments
    all_results: Dict[str, Dict] = {}

    if args.exp_mode in ("official", "both"):
        out_dir_official = os.path.join(base_out, "official_T_to_E")
        res_official = _run_one_experiment(
            exp_name="official (Train=T, Test=E)",
            X_train_full_raw=X_T,
            y_train_full=y_T,
            eval_sets_raw={"SessionE": (X_E, y_E)},
            args=args,
            device=device,
            out_dir=out_dir_official,
            n_channels=n_channels,
            target_len=target_len,
        )
        all_results["official"] = res_official

    if args.exp_mode in ("within", "both"):
        # Session T 内部 8:2 划分（train_full / holdout_test）
        train_full_idx, holdout_idx = stratified_split_indices(y_T, test_size=args.within_test_ratio, seed=args.seed)

        X_T_trainfull = X_T[train_full_idx]
        y_T_trainfull = y_T[train_full_idx]
        X_T_holdout = X_T[holdout_idx]
        y_T_holdout = y_T[holdout_idx]

        out_dir_within = os.path.join(
            base_out, f"within_T_split_{int(round((1-args.within_test_ratio)*100))}_{int(round(args.within_test_ratio*100))}"
        )
        res_within = _run_one_experiment(
            exp_name=f"within (Train=T*{1-args.within_test_ratio:.2f}, Test=T*{args.within_test_ratio:.2f})",
            X_train_full_raw=X_T_trainfull,
            y_train_full=y_T_trainfull,
            eval_sets_raw={
                "T_holdout": (X_T_holdout, y_T_holdout),
                # 额外：也看一下该模型在 Session E 上表现（非官方协议，但便于对照）
                "SessionE": (X_E, y_E),
            },
            args=args,
            device=device,
            out_dir=out_dir_within,
            n_channels=n_channels,
            target_len=target_len,
        )
        all_results["within"] = res_within

    # 6) save summary json
    summary_path = os.path.join(base_out, "summary_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[DONE] Summary saved: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
