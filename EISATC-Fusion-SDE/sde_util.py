import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElectrodeEmbedding(nn.Module):
    """
    给每个电极一个可学习的 embedding，并做 LayerNorm。
    输入/输出: (B, T, N, D)
    """
    def __init__(self, num_electrodes: int, d_model: int):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.d_model = d_model
        self.pos_embed = nn.Embedding(num_electrodes, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        B, T, N, D = x.shape
        assert N == self.num_electrodes, f"ElectrodeEmbedding expects N={self.num_electrodes}, got {N}"
        idx = torch.arange(N, device=x.device, dtype=torch.long)  # (N,)
        e = self.pos_embed(idx)[None, None, :, :]  # (1,1,N,D)
        return self.norm(x + e)


class SpatialDynamicExtractor(nn.Module):
    """
    逐时间片的跨导联多头注意力(仅输出 logits)，用于提取 (B,T,H,N,N) 的动态空间注意力序列。

    输入:  node_tokens_time: (B, T, N, D)
    输出:  sat_logits_seq:   (B, T, H, N, N)   (未 softmax)
    """
    def __init__(
        self,
        num_vertices: int,
        d_model: int = 64,
        d_k: int = 8,
        n_heads: int = 8,
        use_lead_embed: bool = True,
        use_temporal_smoothing: bool = False,
        smoothing_kernel_size: int = 3,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.use_temporal_smoothing = bool(use_temporal_smoothing and smoothing_kernel_size > 1)

        # 可选：电极位置嵌入（导联身份编码）
        self.lead_embed = ElectrodeEmbedding(num_vertices, d_model) if use_lead_embed else None

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

        if self.use_temporal_smoothing:
            # 对 (H*N*N, T) 做 depthwise 1D smoothing
            channels = n_heads * num_vertices * num_vertices
            pad = smoothing_kernel_size // 2
            self.temporal_smoother = nn.Conv1d(
                channels,
                channels,
                kernel_size=smoothing_kernel_size,
                padding=pad,
                groups=channels,
                bias=True,
            )
        else:
            self.temporal_smoother = None

    def forward(self, node_tokens_time: torch.Tensor) -> torch.Tensor:
        # node_tokens_time: (B,T,N,D)
        B, T, N, D = node_tokens_time.shape
        assert N == self.num_vertices, f"SpatialDynamicExtractor expects N={self.num_vertices}, got {N}"
        assert D == self.d_model, f"SpatialDynamicExtractor expects D={self.d_model}, got {D}"

        x = node_tokens_time
        if self.lead_embed is not None:
            x = self.lead_embed(x)  # (B,T,N,D)

        # 展平时间维：对每个时间片单独做空间注意力
        x_bt = x.reshape(B * T, N, D)  # (B*T, N, D)
        q = self.W_Q(x_bt).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # (BT,H,N,d_k)
        k = self.W_K(x_bt).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # (BT,H,N,d_k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (BT,H,N,N)
        sat_seq = scores.view(B, T, self.n_heads, N, N)  # (B,T,H,N,N)

        # 可选：沿时间维做平滑（防止抖动，提高可解释性）
        if self.temporal_smoother is not None and T > 1:
            logits = sat_seq.permute(0, 2, 3, 4, 1).reshape(B, self.n_heads * N * N, T)  # (B,H*N*N,T)
            logits = self.temporal_smoother(logits)
            sat_seq = logits.reshape(B, self.n_heads, N, N, T).permute(0, 4, 1, 2, 3)  # (B,T,H,N,N)

        return sat_seq


class SDEParallelFeatureHead(nn.Module):
    """
    将 sat_logits_seq (B, T, H, N, N) 压缩为固定向量 (B, out_dim)。
    统计项（逐节点）:
      1) 注意力熵 ent: mean/std/range/slope (4)
      2) 分布变化率 diff: mean/std/max (3) —— 这是“注意力转移强度”的关键指标
      3) 自环概率 diag: mean/std (2)
    合计每节点 9 维，拼接 N 个节点 -> (B, N*9) -> MLP -> (B, out_dim)
    """
    def __init__(self, num_vertices: int, n_heads: int, out_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.num_vertices = num_vertices
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.eps = 1e-8

        self.proj = nn.Sequential(
            nn.LayerNorm(num_vertices * 9),
            nn.Linear(num_vertices * 9, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, sat_logits_seq: torch.Tensor) -> torch.Tensor:
        # sat_logits_seq: (B,T,H,N,N) logits
        B, T, H, N, _ = sat_logits_seq.shape
        assert N == self.num_vertices, f"SDEHead expects N={self.num_vertices}, got {N}"
        assert H == self.n_heads, f"SDEHead expects H={self.n_heads}, got {H}"

        P = torch.softmax(sat_logits_seq, dim=-1)  # (B,T,H,N,N)

        # (1) entropy per node: -(p log p) over neighbors
        p_clamped = P.clamp_min(self.eps)
        ent = -(p_clamped * p_clamped.log()).sum(dim=-1)    # (B,T,H,N)
        ent = ent.mean(dim=2)                               # mean over heads -> (B,T,N)
        ent_mean  = ent.mean(dim=1)                         # (B,N)
        ent_std   = ent.std(dim=1, unbiased=False)          # (B,N)
        ent_range = ent.max(dim=1).values - ent.min(dim=1).values  # (B,N)
        ent_slope = (ent[:, -1, :] - ent[:, 0, :]) / max(1, T - 1) # (B,N)

        # (2) transfer strength: sum_j |P_t - P_{t-1}| (then mean over heads)
        if T > 1:
            diff = (P[:, 1:] - P[:, :-1]).abs().sum(dim=-1)  # (B,T-1,H,N)
            diff = diff.mean(dim=2)                           # (B,T-1,N)
            diff_mean = diff.mean(dim=1)                      # (B,N)
            diff_std  = diff.std(dim=1, unbiased=False)       # (B,N)
            diff_max  = diff.max(dim=1).values                # (B,N)
        else:
            zero = torch.zeros(B, N, device=P.device, dtype=P.dtype)
            diff_mean = diff_std = diff_max = zero

        # (3) self-loop prob: diag of attention matrix
        diag = P.diagonal(dim1=3, dim2=4).mean(dim=2)  # (B,T,N)
        diag_mean = diag.mean(dim=1)                    # (B,N)
        diag_std  = diag.std(dim=1, unbiased=False)     # (B,N)

        feat_nodes = torch.stack(
            [ent_mean, ent_std, ent_range, ent_slope,
             diff_mean, diff_std, diff_max,
             diag_mean, diag_std],
            dim=-1
        )  # (B,N,9)

        feat = feat_nodes.reshape(B, N * 9)  # (B, N*9)
        emb = self.proj(feat)                # (B,out_dim)
        return emb
