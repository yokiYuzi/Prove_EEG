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

from HG_ATTN import BCIC2A_CH_NAMES_22

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
    _HAS_RETURN_CH_NAMES = "return_ch_names" in _sig.parameters
except Exception:
    _HAS_REREF_ARGS = False
    _HAS_RETURN_CH_NAMES = False


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
    """
    EEG -> (optional HG-ATTN spatial fusion per temporal patch) -> Transformer Encoder -> classification.

    Two modes:
      - baseline: original patch embedding (C*patch -> d_model), then time-Transformer over patches.
      - hgattn  : for each temporal patch, create channel tokens (patch_size -> d_model),
                 apply HG-ATTN over channels (+ a patch-level CLS node), and use that CLS output
                 as the patch token. Then the same time-Transformer over patches.

    This design lets the model "look at channel groups" (hyperedges) when forming patch tokens,
    instead of collapsing channels too early.
    """

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
        # ---- new: model selection ----
        model_type: str = "hgattn",  # "baseline" or "hgattn"
        # ---- HG-ATTN config (used only if model_type == "hgattn") ----
        ch_names: Optional[List[str]] = None,
        hg_depth: int = 1,
        hg_heads: int = 0,            # 0 => use nhead
        hg_pool: str = "attn",        # "attn" or "mean"
        hg_d_pool: int = 64,
        hg_ffn_dim: int = 256,
        hg_dropout: float = 0.1,
        hg_priors: str = "region,hemisphere,midline,symmetry,neighborhood,global",
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

        self.model_type = str(model_type).lower()
        if self.model_type not in ("baseline", "hgattn"):
            raise ValueError("model_type must be 'baseline' or 'hgattn'")

        self.n_patches = int(math.ceil(self.input_samples / self.patch_size))

        # ------------------------------------------------------------------
        # Patch tokenization path
        # ------------------------------------------------------------------
        if self.model_type == "baseline":
            # Original behavior: flatten (C*patch) -> d_model
            self.patch_embed = nn.Linear(self.n_channels * self.patch_size, d_model)
            self.ch_patch_embed = None
            self.hg_blocks = None
            self.patch_cls_token = None
            self.channel_pos_embed = None
            self.hg_membership_mask = None
            self.hg_incident_mask = None
            self.hg_edge_names = None
            self.hg_edge_types = None
        else:
            # HG-ATTN behavior:
            # - For each temporal patch, embed each channel patch (patch_size -> d_model)
            # - Add a patch-level CLS node, run HG-ATTN over (CLS + channels)
            # - Use the patch-level CLS output as the patch token
            from HG_ATTN import build_bcic2a_priors, HGAttnBlock

            self.patch_embed = None
            self.ch_patch_embed = nn.Linear(self.patch_size, d_model)

            # Patch-level CLS used to summarize channels for each patch
            self.patch_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            # Channel identity embedding (shared across patches)
            self.channel_pos_embed = nn.Parameter(torch.zeros(1, self.n_channels, d_model))

            # Build hypergraph priors from channel names (or fallback)
            priors_enabled = {s.strip().lower() for s in str(hg_priors).split(",") if s.strip()}

            include_region = "region" in priors_enabled
            include_hemisphere = "hemisphere" in priors_enabled
            include_midline = "midline" in priors_enabled
            include_symmetry = "symmetry" in priors_enabled
            include_neighborhood = "neighborhood" in priors_enabled
            include_global = "global" in priors_enabled

            priors, edge_specs = build_bcic2a_priors(
                ch_names=ch_names,
                include_region=include_region,
                include_hemisphere=include_hemisphere,
                include_midline=include_midline,
                include_symmetry=include_symmetry,
                include_neighborhood=include_neighborhood,
                include_global=include_global,
                add_virtual_cls=True,
                virtual_cls_attend_all=True,
            )

            # Register masks as buffers so they move with .to(device)
            self.register_buffer("hg_membership_mask", priors.membership_mask, persistent=False)
            self.register_buffer("hg_incident_mask", priors.incident_mask, persistent=False)
            self.hg_edge_names = priors.edge_names
            self.hg_edge_types = priors.edge_types

            # HG blocks
            hg_depth = int(hg_depth)
            if hg_depth <= 0:
                raise ValueError("hg_depth must be >= 1")
            hg_heads = int(hg_heads) if int(hg_heads) > 0 else int(nhead)

            self.hg_blocks = nn.ModuleList(
                [
                    HGAttnBlock(
                        d_model=d_model,
                        num_heads=hg_heads,
                        dim_feedforward=int(hg_ffn_dim),
                        dropout=float(hg_dropout),
                        activation=activation,
                        pool=str(hg_pool),
                        d_pool=int(hg_d_pool),
                        gate=True,
                    )
                    for _ in range(hg_depth)
                ]
            )

        # ------------------------------------------------------------------
        # Time Transformer Encoder (over patch tokens)
        # ------------------------------------------------------------------
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

        if self.model_type == "hgattn":
            nn.init.trunc_normal_(self.patch_cls_token, std=0.02)
            nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)

    def _to_patches(self, x: torch.Tensor, as_4d: bool = False) -> torch.Tensor:
        """
        x: (B,C,T)
        If as_4d=False:
            -> (B, n_patches, C*patch)
        If as_4d=True:
            -> (B, n_patches, C, patch)
        """
        B, C, T = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {C}")

        total_len = self.n_patches * self.patch_size
        if T < total_len:
            x = F.pad(x, (0, total_len - T), mode="constant", value=0.0)
        elif T > total_len:
            x = x[:, :, :total_len]

        # unfold: (B,C,n_patches,patch)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        if as_4d:
            return patches.permute(0, 2, 1, 3).contiguous()  # (B, n_patches, C, patch)

        # flatten channels
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, n_patches, C, patch)
        patches = patches.view(B, self.n_patches, C * self.patch_size)
        return patches

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Returns:
          logits: (B,num_classes)
          attn_info:
            - if return_attn=False: None
            - if return_attn=True:
                dict with keys:
                  'time': list of attention weights from time Transformer layers (same as before)
                  'hg'  : (optional) list of HG-ATTN attention weights (CLS->hyperedges) per HG block
                  'hg_meta': (optional) edge names/types
                  'hg_n_patches': int
        """
        if self.model_type == "baseline":
            patches = self._to_patches(x, as_4d=False)          # (B,P,C*patch)
            tok = self.patch_embed(patches)                     # (B,P,D)
            hg_attn_all = None
        else:
            # HG-ATTN spatial fusion per patch
            patches4d = self._to_patches(x, as_4d=True)         # (B,P,C,patch)
            B, P, C, Lp = patches4d.shape

            # Per-channel patch embedding: (B,P,C,D)
            ch_tok = self.ch_patch_embed(patches4d)

            # Add channel identity embedding (broadcast across patches)
            ch_tok = ch_tok + self.channel_pos_embed.unsqueeze(1)

            # Flatten patches into batch: (B*P, C, D)
            ch_tok = ch_tok.view(B * P, C, self.d_model)

            # Add patch-level CLS: (B*P, 1, D)
            cls = self.patch_cls_token.expand(B * P, -1, -1)
            nodes = torch.cat([cls, ch_tok], dim=1)             # (B*P, 1+C, D)

            hg_attn_all = [] if return_attn else None
            for blk in self.hg_blocks:
                nodes, A, _ = blk(
                    nodes,
                    membership_mask=self.hg_membership_mask,
                    incident_mask=self.hg_incident_mask,
                    return_attn=return_attn,
                    return_pool=False,
                )
                if return_attn:
                    # Keep only patch-CLS node attention over hyperedges to save memory:
                    # A: (B*P, H, N_nodes, E) -> (B*P, H, E)
                    hg_attn_all.append(A[:, :, 0, :])

            # Use patch-CLS output as patch token
            tok = nodes[:, 0, :].view(B, P, self.d_model)       # (B,P,D)

        # Time Transformer over patch tokens (same as baseline)
        if self.use_cls_token:
            cls = self.cls_token.expand(tok.size(0), -1, -1)
            tok = torch.cat([cls, tok], dim=1)                  # (B,1+P,D)

        tok = tok + self.pos_embed
        tok = self.pos_drop(tok)

        time_attn_all = [] if return_attn else None
        for layer in self.layers:
            tok, attn_w = layer(tok, return_attn=return_attn)
            if return_attn:
                time_attn_all.append(attn_w)

        tok = self.norm(tok)
        feat = tok[:, 0] if self.use_cls_token else tok.mean(dim=1)
        logits = self.head(feat)

        if not return_attn:
            return logits, None

        attn_info = {
            "time": time_attn_all,
            "hg": hg_attn_all,
            "hg_meta": None,
            "hg_n_patches": int(self.n_patches),
        }
        if self.model_type == "hgattn":
            attn_info["hg_meta"] = {"edge_names": self.hg_edge_names, "edge_types": self.hg_edge_types}
        return logits, attn_info


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
def collect_mean_attentions(
    model: EEGTransformerEncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Collect mean attentions for analysis.

    Returns a dict with:
      - 'time': mean time-Transformer attention, shape (Layers, Heads(or1), L, L), or None
      - 'hg'  : mean HG-ATTN attention (patch-CLS -> hyperedges), shape (HG_Layers, HG_Heads, E), or None
      - 'hg_meta': {'edge_names': [...], 'edge_types': [...]} if available else None
      - 'hg_n_patches': int if available else None
    """
    model.eval()

    # time attention accumulators
    time_sum = None
    time_count = 0

    # HG attention accumulators
    hg_sum = None
    hg_count = 0
    hg_meta = None
    hg_n_patches = None

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        x, _ = to_device(batch, device)
        _, attn_info = model(x, return_attn=True)

        if attn_info is None:
            continue

        # ---- parse time vs hg attentions ----
        if isinstance(attn_info, dict):
            time_list = attn_info.get("time", None)
            hg_list = attn_info.get("hg", None)
            if hg_meta is None:
                hg_meta = attn_info.get("hg_meta", None)
            if hg_n_patches is None:
                hg_n_patches = attn_info.get("hg_n_patches", None)
        else:
            time_list = attn_info
            hg_list = None

        # ---- time Transformer attention (same as original) ----
        if time_list is not None:
            layer_attns = []
            for attn in time_list:
                if attn is None:
                    raise RuntimeError("return_attn=True but got None attention")
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)  # (B,1,L,L)
                layer_attns.append(attn.detach().cpu().float().numpy())

            stacked = np.stack(layer_attns, axis=0)  # (Layers,B,H,L,L)

            if time_sum is None:
                time_sum = stacked.sum(axis=1)  # (Layers,H,L,L)
            else:
                time_sum += stacked.sum(axis=1)

            time_count += stacked.shape[1]

        # ---- HG attention: list of (B*n_patches, H_hg, E) (CLS-only) ----
        if hg_list is not None and len(hg_list) > 0:
            # Each element: (B*P, H, E)
            hg_layer_sums = []
            total_bp = 0
            for attn_hg in hg_list:
                if attn_hg is None:
                    continue
                a = attn_hg.detach().cpu().float().numpy()  # (BP, H, E)
                hg_layer_sums.append(a.sum(axis=0))         # (H, E)
                total_bp = a.shape[0]  # BP for this batch (same for all layers)
            if total_bp > 0 and len(hg_layer_sums) > 0:
                stacked_hg = np.stack(hg_layer_sums, axis=0)  # (HG_Layers, H, E)
                if hg_sum is None:
                    hg_sum = stacked_hg
                else:
                    hg_sum += stacked_hg
                hg_count += total_bp

    out: Dict[str, Optional[np.ndarray]] = {
        "time": None,
        "hg": None,
        "hg_meta": hg_meta,
        "hg_n_patches": hg_n_patches,
    }

    if time_sum is not None and time_count > 0:
        out["time"] = time_sum / float(time_count)

    if hg_sum is not None and hg_count > 0:
        out["hg"] = hg_sum / float(hg_count)

    return out


@torch.no_grad()
def collect_mean_attention(
    model: EEGTransformerEncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> np.ndarray:
    """Backward-compatible wrapper: returns only mean time attention."""
    attn = collect_mean_attentions(model, loader, device=device, max_batches=max_batches)
    if attn["time"] is None:
        raise RuntimeError("No time attention collected; check loader/model")
    return attn["time"]


def compute_hg_attn_stats(
    mean_hg_attn: np.ndarray,
    edge_names: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
) -> Dict:
    """
    Compute HG-ATTN stats from mean hyperedge attention.

    mean_hg_attn: (HG_Layers, HG_Heads, E)

    We compute statistics on distributions over hyperedges (E):
      - entropy
      - L2 distance to uniform
      - KL(p || uniform)
      - max weight

    Also (optional): type-wise attention mass if edge_types is provided.
    """
    if mean_hg_attn.ndim != 3:
        raise ValueError(f"mean_hg_attn must be 3D (hg_layers,hg_heads,E), got {mean_hg_attn.shape}")

    Lhg, Hhg, E = mean_hg_attn.shape
    uniform = np.ones((E,), dtype=np.float64) / float(E)

    per_layer = []
    for li in range(Lhg):
        ent_list, l2_list, kl_list, max_list = [], [], [], []
        for hi in range(Hhg):
            p = mean_hg_attn[li, hi, :].astype(np.float64)
            p = p / max(1e-12, p.sum())
            ent_list.append(_entropy(p))
            l2_list.append(float(np.sqrt(((p - uniform) ** 2).mean())))
            kl_list.append(_kl(p, uniform))
            max_list.append(float(p.max()))
        per_layer.append(
            {
                "layer": int(li),
                "entropy": float(np.mean(ent_list)),
                "l2_to_uniform": float(np.mean(l2_list)),
                "kl_to_uniform": float(np.mean(kl_list)),
                "max_weight": float(np.mean(max_list)),
            }
        )

    global_stats = {
        "mean_entropy": float(np.mean([x["entropy"] for x in per_layer])),
        "mean_l2_to_uniform": float(np.mean([x["l2_to_uniform"] for x in per_layer])),
        "mean_kl_to_uniform": float(np.mean([x["kl_to_uniform"] for x in per_layer])),
        "mean_max_weight": float(np.mean([x["max_weight"] for x in per_layer])),
    }

    type_mass = None
    if edge_types is not None and len(edge_types) == E:
        # Aggregate across heads and layers first: p_bar (E,)
        p_bar = mean_hg_attn.mean(axis=(0, 1)).astype(np.float64)
        p_bar = p_bar / max(1e-12, p_bar.sum())
        type_mass = {}
        for et in sorted(set(edge_types)):
            idxs = [i for i, t in enumerate(edge_types) if t == et]
            type_mass[et] = float(p_bar[idxs].sum())

    return {
        "per_layer": per_layer,
        "global": global_stats,
        "E": int(E),
        "hg_layers": int(Lhg),
        "hg_heads": int(Hhg),
        "edge_names": edge_names,
        "edge_types": edge_types,
        "type_mass": type_mass,
        "note": "HG-ATTN 分布越接近均匀：entropy 趋近 log(E)、l2/kl 趋近 0、max_weight 趋近 1/E。过度尖锐可能表示只用极少数超边。",
    }


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




def save_attention_evidence(
    model: EEGTransformerEncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    set_name: str,
    max_batches: int = 20,
    print_global: bool = True,
) -> None:
    """
    Save attention evidence to disk.

    Time Transformer evidence (keeps original file names):
      - mean_attention_{set_name}.npy
      - attention_stats_{set_name}.json

    HG-ATTN evidence (new files):
      - mean_hg_attention_{set_name}.npy
      - hg_attention_stats_{set_name}.json
    """
    os.makedirs(out_dir, exist_ok=True)

    attn_means = collect_mean_attentions(model, loader, device=device, max_batches=max_batches)

    # 1) Time Transformer attention (same behavior as original)
    if attn_means.get("time") is not None:
        mean_time = attn_means["time"]
        stats_time = compute_attn_averaging_stats(mean_time, cls_index=0)

        np.save(os.path.join(out_dir, f"mean_attention_{set_name}.npy"), mean_time)
        with open(os.path.join(out_dir, f"attention_stats_{set_name}.json"), "w", encoding="utf-8") as f:
            json.dump(stats_time, f, ensure_ascii=False, indent=2)

        if print_global:
            print("\n[ATTENTION AVERAGING EVIDENCE] (Time Transformer, CLS attention)")
            print(json.dumps(stats_time["global"], ensure_ascii=False, indent=2))

    # 2) HG-ATTN (patch-CLS -> hyperedges)
    if attn_means.get("hg") is not None:
        mean_hg = attn_means["hg"]
        meta = attn_means.get("hg_meta") or {}
        edge_names = meta.get("edge_names", None)
        edge_types = meta.get("edge_types", None)

        stats_hg = compute_hg_attn_stats(mean_hg, edge_names=edge_names, edge_types=edge_types)

        np.save(os.path.join(out_dir, f"mean_hg_attention_{set_name}.npy"), mean_hg)
        with open(os.path.join(out_dir, f"hg_attention_stats_{set_name}.json"), "w", encoding="utf-8") as f:
            json.dump(stats_hg, f, ensure_ascii=False, indent=2)

        if print_global:
            print("\n[HG-ATTN EVIDENCE] (Patch-CLS -> Hyperedges)")
            print(json.dumps(stats_hg["global"], ensure_ascii=False, indent=2))
            if stats_hg.get("type_mass") is not None:
                print("[HG-ATTN TYPE MASS] (avg over layers/heads)")
                print(json.dumps(stats_hg["type_mass"], ensure_ascii=False, indent=2))


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
    ch_names: Optional[List[str]],
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
        model_type=args.model,
        ch_names=ch_names,
        hg_depth=args.hg_depth,
        hg_heads=args.hg_heads,
        hg_pool=args.hg_pool,
        hg_d_pool=args.hg_d_pool,
        hg_ffn_dim=args.hg_ffn_dim,
        hg_dropout=args.hg_dropout,
        hg_priors=args.hg_priors,
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
    print(f"Model: {args.model} | HG(depth={args.hg_depth}, heads={(args.hg_heads if args.hg_heads>0 else args.nhead)}, pool={args.hg_pool}, priors={args.hg_priors})")
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
            save_attention_evidence(
                model=model,
                loader=train_eval_loader,
                device=device,
                out_dir=out_dir,
                set_name="TrainSplit",
                max_batches=args.attn_max_batches,
                print_global=False,
            )

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
            save_attention_evidence(
                model=model,
                loader=val_eval_loader,
                device=device,
                out_dir=out_dir,
                set_name="ValSplit",
                max_batches=args.attn_max_batches,
                print_global=False,
            )

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

        # attention evidence (Time Transformer + optional HG-ATTN)
        save_attention_evidence(
            model=model,
            loader=loader,
            device=device,
            out_dir=out_dir,
            set_name=set_name,
            max_batches=args.attn_max_batches,
            print_global=True,
        )

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

    # model selection
    p.add_argument(
        "--model",
        type=str,
        default="hgattn",
        choices=["baseline", "hgattn"],
        help="baseline: 原始 Only-Encoder; hgattn: 在每个 temporal patch 内做 HG-ATTN(超图引导) 的空间融合",
    )

    # HG-ATTN hypergraph priors + block config (only used when --model=hgattn)
    p.add_argument("--hg_depth", type=int, default=1, help="每个 temporal patch 内堆叠多少层 HG-ATTN block")
    p.add_argument("--hg_heads", type=int, default=0, help="HG-ATTN 的 head 数；0 表示与 --nhead 相同")
    p.add_argument("--hg_pool", type=str, default="attn", choices=["attn", "mean"], help="超边 token 的聚合方式")
    p.add_argument("--hg_d_pool", type=int, default=64, help="超边 attention-pooling 的隐藏维度（pool=attn 时生效）")
    p.add_argument("--hg_ffn_dim", type=int, default=256, help="HG-ATTN block 内部 FFN 维度")
    p.add_argument("--hg_dropout", type=float, default=0.1, help="HG-ATTN block 的 dropout")
    p.add_argument(
        "--hg_priors",
        type=str,
        default="region,hemisphere,midline,symmetry,neighborhood,global",
        help="启用哪些先验超边集合，用逗号分隔。可选: region,hemisphere,midline,symmetry,neighborhood,global",
    )

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

    # 如果当前 get_data 支持 return_ch_names，则请求返回通道名列表（用于构建 HG-ATTN 先验超边）
    if _HAS_RETURN_CH_NAMES:
        get_data_kwargs.update(dict(return_ch_names=True))

    data_out = _GET_DATA(**get_data_kwargs)

    # 兼容返回值长度：旧版 6 项；新版（return_ch_names=True）7 项
    ch_names = None
    if isinstance(data_out, (tuple, list)) and len(data_out) == 7:
        X_T, y_T, X_E, y_E, _, _, ch_names = data_out
    else:
        X_T, y_T, X_E, y_E, _, _ = data_out

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

    # 3.5) channel names (for HG-ATTN priors)
    # 如果 get_data 没返回通道名，则对 BCICIV-2a 做一个医学常识的默认假设：通道顺序为 22 导标准顺序
    if ch_names is not None and not isinstance(ch_names, list):
        try:
            ch_names = list(ch_names)
        except Exception:
            ch_names = None

    if ch_names is None or (isinstance(ch_names, list) and len(ch_names) != n_channels):
        if n_channels == 22:
            ch_names = list(BCIC2A_CH_NAMES_22)
        elif n_channels == 21 and _HAS_REREF_ARGS and bool(args.reref) and bool(args.drop_ref):
            base = list(BCIC2A_CH_NAMES_22)
            # 若 ref_channel 在默认列表中，则删除对应项，使长度匹配
            if isinstance(args.ref_channel, str) and args.ref_channel in base:
                base.pop(base.index(args.ref_channel))
            if len(base) == 21:
                ch_names = base
            else:
                ch_names = [f"Ch{i}" for i in range(n_channels)]
        else:
            ch_names = [f"Ch{i}" for i in range(n_channels)]

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
    print(f"[Model] model={args.model} | HG(depth={args.hg_depth}, heads={(args.hg_heads if args.hg_heads>0 else args.nhead)}, pool={args.hg_pool}, priors={args.hg_priors})")
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
            ch_names=ch_names,
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
            ch_names=ch_names,
            n_channels=n_channels,
            target_len=target_len,
        )
        all_results["within"] = res_within

    # 6) save summary json
    summary_path = os.path.join(base_out, "summary_results.json")
    # NOTE: all_results 中包含 numpy 数组（y_true/y_pred），直接 json.dump 会报错；
    # 这里保存一个“精简但可读”的 summary，不包含大数组。
    def _jsonable_metrics(m: Dict) -> Dict:
        if not isinstance(m, dict):
            return {}
        drop_keys = {"y_true", "y_pred"}
        out = {}
        for k, v in m.items():
            if k in drop_keys:
                continue
            out[k] = v
        return out

    summary_slim: Dict[str, Dict] = {}
    for exp_key, exp_val in all_results.items():
        if not isinstance(exp_val, dict):
            continue
        summary_slim[exp_key] = {}
        for set_key, metrics in exp_val.items():
            summary_slim[exp_key][set_key] = _jsonable_metrics(metrics)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_slim, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[DONE] Summary saved: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
