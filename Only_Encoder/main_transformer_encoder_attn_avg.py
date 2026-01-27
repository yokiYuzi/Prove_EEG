"""main_transformer_encoder_attn_avg.py

Transformer 基础 Encoder 结构，用于在 BCICIV-2a (Session T -> Session E) 上验证
“注意力平均化(Attention Averaging)”现象。

你只需要保证以下文件同目录（或在 PYTHONPATH 中）：
  - preprocess.py  (提供 get_data)
  - LoadData.py    (preprocess 会 import 它)

脚本输出：
  1) Test accuracy
  2) Test macro-F1
  3) sklearn classification_report（test）
  4) 注意力平均化的定量证据（attention entropy / 与 uniform 的距离等）
  5) 保存 test 预测结果 CSV、以及每层每头的平均注意力矩阵 .npy

运行示例：
  python main_transformer_encoder_attn_avg.py --subject 1 --epochs 200 --batch_size 32

注意：本脚本默认使用 preprocess.get_data 内部的 2s~6s (4s) 裁剪。
你也可以用 --input_seconds 进一步裁剪到 2s 或 4s。
"""

from __future__ import annotations

import os
import json
import math
import random
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

# -----------------------------------------------------------------------------
# Import your existing data loader / preprocessing utilities
# -----------------------------------------------------------------------------
try:
    from preprocess import get_data
except ImportError as e:
    raise ImportError(
        "无法 import preprocess.get_data。请把 main_transformer_encoder_attn_avg.py 与 preprocess.py 放在同一目录，"
        "或把包含 preprocess.py 的目录加入 PYTHONPATH。"
    ) from e


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
        # pad zeros at the end
        pad = target_len - T
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    if mode == "start":
        return X[:, :, :target_len]
    if mode == "center":
        start = (T - target_len) // 2
        return X[:, :, start : start + target_len]
    raise ValueError(f"Unknown crop mode: {mode}")


# -----------------------------------------------------------------------------
# Transformer Encoder with attention weight outputs
# -----------------------------------------------------------------------------


def _mha_supports_avg_flag() -> bool:
    """PyTorch 1.12+ supports average_attn_weights flag in MultiheadAttention forward."""
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
        Args:
            x: (B, L, D)
            return_attn: whether to return attention weights

        Returns:
            x_out: (B, L, D)
            attn_w: None or attention weights
              - if supports_avg_flag and return_attn=True: (B, H, L, L)
              - else: (B, L, L) (averaged across heads)
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
                # Older PyTorch: no per-head weights
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

        # number of temporal patches
        self.n_patches = int(math.ceil(self.input_samples / self.patch_size))

        # patch embedding: (C * patch_size) -> d_model
        self.patch_embed = nn.Linear(self.n_channels * self.patch_size, d_model)

        # [CLS]
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            seq_len = 1 + self.n_patches
        else:
            self.cls_token = None
            seq_len = self.n_patches

        # positional embedding
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
        # patch_embed/head use default init or you can add more

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,T) -> patches: (B, n_patches, C*patch_size)"""
        B, C, T = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {C}")

        # pad T to multiple of patch_size
        total_len = self.n_patches * self.patch_size
        if T < total_len:
            pad = total_len - T
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        elif T > total_len:
            x = x[:, :, :total_len]

        # unfold: (B,C,n_patches,patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B,n_patches,C,patch)
        patches = patches.view(B, self.n_patches, C * self.patch_size)
        return patches

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """Forward.

        Args:
            x: (B, C, T)
            return_attn: if True, also return attention weights (list per layer)

        Returns:
            logits: (B, num_classes)
            attn_list: list of attention weights per layer (or None)
        """
        patches = self._to_patches(x)  # (B, n_patches, C*patch)
        tok = self.patch_embed(patches)  # (B, n_patches, D)

        if self.use_cls_token:
            cls = self.cls_token.expand(tok.size(0), -1, -1)
            tok = torch.cat([cls, tok], dim=1)  # (B, 1+n_patches, D)

        tok = tok + self.pos_embed
        tok = self.pos_drop(tok)

        attn_all = [] if return_attn else None
        for layer in self.layers:
            tok, attn_w = layer(tok, return_attn=return_attn)
            if return_attn:
                attn_all.append(attn_w)

        tok = self.norm(tok)

        if self.use_cls_token:
            feat = tok[:, 0]  # CLS
        else:
            feat = tok.mean(dim=1)

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


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
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
    # CLS attention distribution metrics
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
    """Collect mean attention matrices per layer/head over a subset of data.

    Returns:
        mean_attn: (num_layers, num_heads(or 1), L, L)
    """
    model.eval()
    num_layers = len(model.layers)

    attn_sum = None
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, _ = to_device(batch, device)
        _, attn_list = model(x, return_attn=True)

        # attn_list: list length num_layers
        # each attn: (B,H,L,L) or (B,L,L)
        layer_attns = []
        for attn in attn_list:
            if attn is None:
                raise RuntimeError("return_attn=True but got None attention")
            if attn.dim() == 3:
                # (B,L,L) -> (B,1,L,L)
                attn = attn.unsqueeze(1)
            layer_attns.append(attn.detach().cpu().float().numpy())

        # stack layers -> (num_layers,B,H,L,L)
        stacked = np.stack(layer_attns, axis=0)

        if attn_sum is None:
            attn_sum = stacked.sum(axis=1)  # sum over B -> (num_layers,H,L,L)
        else:
            attn_sum += stacked.sum(axis=1)

        count += stacked.shape[1]

    if attn_sum is None or count == 0:
        raise RuntimeError("No attention collected; check loader")

    mean_attn = attn_sum / float(count)
    return mean_attn


def compute_attn_averaging_stats(mean_attn: np.ndarray, cls_index: int = 0) -> Dict:
    """Compute attention-averaging statistics from mean attention.

    We focus on distribution of CLS query over all keys.

    Args:
        mean_attn: (LAYER, HEAD, L, L)
        cls_index: index of CLS token (0 if use_cls_token=True)

    Returns:
        dict with per-layer stats + global average.
    """
    if mean_attn.ndim != 4:
        raise ValueError(f"mean_attn must be 4D (layer,head,L,L), got {mean_attn.shape}")

    num_layers, num_heads, L, L2 = mean_attn.shape
    assert L == L2

    uniform = np.ones((L,), dtype=np.float64) / float(L)

    per_layer = []
    for li in range(num_layers):
        ent_list, l2_list, kl_list, max_list = [], [], [], []
        for hi in range(num_heads):
            p = mean_attn[li, hi, cls_index, :].astype(np.float64)
            # ensure normalized (just in case numerical issue)
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
        "note": (
            "若注意力趋向平均化，则：entropy 接近 log(L)、l2_to_uniform/kl_to_uniform 接近 0、max_weight 接近 1/L。"
        ),
        "L": int(L),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1, help="BCICIV-2a subject id (1..9)")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="BCICIV_2a root dir (contains s1/, s2/, ...). Default: ./dataLoad/BCICIV_2a",
    )
    p.add_argument("--seed", type=int, default=42)

    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # input
    p.add_argument(
        "--input_seconds",
        type=float,
        default=4.0,
        help="Further crop length in seconds (<=4.0). preprocess.get_data already crops 2s~6s (4s).",
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

    # attention analysis
    p.add_argument(
        "--attn_max_batches",
        type=int,
        default=20,
        help="Use how many batches from TEST to compute mean attention (for analysis)",
    )

    # output
    p.add_argument("--out_dir", type=str, default="./outputs_transformer_encoder")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(script_dir, "dataLoad", "BCICIV_2a")

    # Make sure ends with os.sep because preprocess.get_data will append 's{}/'
    if not data_dir.endswith(os.sep):
        data_dir = data_dir + os.sep

    # 1) load data (official protocol)
    X_train, y_train, X_test, y_test, _, _ = get_data(
        path=data_dir,
        subject=args.subject,
        LOSO=False,
        data_type="2a",
        isStandard=True,
        standardize_mode=args.standardize_mode,
    )

    # 2) further crop
    target_len = int(round(args.input_seconds * args.fs))
    if target_len > X_train.shape[-1]:
        raise ValueError(
            f"--input_seconds 对应长度 {target_len} > preprocess.get_data 输出长度 {X_train.shape[-1]}。"
            f"请把 input_seconds <= {X_train.shape[-1] / args.fs:.2f}."
        )

    X_train = crop_time(X_train, target_len=target_len, mode=args.crop_mode)
    X_test = crop_time(X_test, target_len=target_len, mode=args.crop_mode)

    n_trials, n_channels, _ = X_train.shape

    # 3) split val from Session T
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    train_ds = TensorDataset(torch.tensor(X_train[train_idx], dtype=torch.float32), torch.tensor(y_train[train_idx], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_train[val_idx], dtype=torch.float32), torch.tensor(y_train[val_idx], dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4) model
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

    # 5) train
    best_val_f1 = -1.0
    out_dir = os.path.join(args.out_dir, f"sub{args.subject}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.pth")

    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Subject: {args.subject}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Input: C={n_channels}, T={target_len} ({args.input_seconds}s), patch={args.patch_size}, n_patches={model.n_patches}")
    print(f"Transformer: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}, ff={args.dim_feedforward}, dropout={args.dropout}")
    print(f"Standardize mode: {args.standardize_mode}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

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

    # 6) test with best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    # classification_report string
    report_str = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("[TEST RESULTS] (Session E)")
    print(f"Test acc      : {test_metrics['accuracy']:.4f}")
    print(f"Test F1(macro): {test_metrics['f1_macro']:.4f}")
    print("\nclassification_report(test):")
    print(report_str)
    print("=" * 80)

    # 7) save test predictions
    pred_path = os.path.join(out_dir, "test_predictions.csv")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for yt, yp in zip(test_metrics["y_true"].tolist(), test_metrics["y_pred"].tolist()):
            f.write(f"{yt},{yp}\n")

    # 8) attention averaging evidence on TEST
    mean_attn = collect_mean_attention(model, test_loader, device=device, max_batches=args.attn_max_batches)
    stats = compute_attn_averaging_stats(mean_attn, cls_index=0)

    attn_npy_path = os.path.join(out_dir, "mean_attention_layer_head_L_L.npy")
    np.save(attn_npy_path, mean_attn)

    attn_stats_path = os.path.join(out_dir, "attention_averaging_stats.json")
    with open(attn_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Console summary
    print("\n[ATTENTION AVERAGING EVIDENCE] (CLS attention on test)")
    print(f"Saved mean attention: {attn_npy_path}")
    print(f"Saved stats JSON    : {attn_stats_path}")
    print("Global stats:")
    print(json.dumps(stats["global"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
