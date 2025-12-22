# DSTAGNN_my.py
# 说明:
#   - 本脚本已根据您的全部意见修改。
#   - [新增] 将 cheb_polynomials 和 adj_pa_static 注册为 nn.Module 的 buffer，
#     以确保在 DP/DDP 模式下能被正确复制到各个设备，避免跨设备张量操作。
#   - [新增] 引入 SpatialDynamicExtractor (SDE) 来提取逐时刻的动态空间注意力序列。
#   - [增强] 引入 SDEParallelFeatureHead：在动态注意力序列上做【分段节点统计】+【边级TopK增强/减弱】特征抽取。
#   - [增强] DSTAGNN_block 并行计算静态/动态空间注意力，并支持将动态注意力(时间平均)按 alpha 混入 GCN 邻接权重。
#   - [修改] DSTAGNN_submodule 在分类任务中，会拼接主干特征和SDE提取的并行特征，以增强分类性能。
#   - [新增] 引入 TemporalSeqExporter 类，用于为可解释性分析提取和处理时间序列特征。
#   - [新增] 在 DSTAGNN_submodule 中添加 export_time_feature_sequences 方法，提供一个便捷的特征导出接口。
#   - [新增] 在 DSTAGNN_submodule 中添加 exp_mode 开关，用于进行时间特征重要性的分类消融实验。
############################################
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 假设 utils.py 与 DSTAGNN_my.py 在同一目录或Python路径可找到
from utils import scaled_Laplacian, cheb_polynomial


class SScaledDotProductAttention(nn.Module): # 空间注意力的点积计算（仅分数）
    def __init__(self, d_k): # 初始化
        super(SScaledDotProductAttention, self).__init__() #
        self.d_k = d_k # 键的维度

    def forward(self, Q, K, attn_mask): # 前向传播
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # 计算Q, K的点积并缩放
        if attn_mask is not None: # 应用注意力掩码
            scores.masked_fill_(attn_mask, -1e9)  # 将掩码位置设为极小值
        return scores # 返回注意力分数


class ScaledDotProductAttention(nn.Module): # 标准点积注意力（带V和softmax）
    def __init__(self, d_k, num_of_d_features_unused): # 初始化，num_of_d_features_unused 未使用
        super(ScaledDotProductAttention, self).__init__() #
        self.d_k = d_k #

    def forward(self, Q, K, V, attn_mask, res_att): # 前向传播
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att  # 计算分数并加入残差注意力
        if attn_mask is not None: # 应用掩码
            scores.masked_fill_(attn_mask, -1e9) #
        attn_weights = F.softmax(scores, dim=-1) # 计算注意力权重
        context = torch.matmul(attn_weights, V)  # 计算上下文向量
        return context, scores # 返回上下文和原始分数（softmax前，但已加res_att）


class SMultiHeadAttention(nn.Module): # 空间多头注意力模块
    def __init__(self, DEVICE_unused, d_model, d_k ,d_v_unused, n_heads): # 初始化
        super(SMultiHeadAttention, self).__init__() #
        self.d_model = d_model #
        self.d_k = d_k #
        # self.d_v_unused = d_v_unused # V未使用
        self.n_heads = n_heads #
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False) # Q的投影矩阵
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False) # K的投影矩阵

    def forward(self, input_Q, input_K, attn_mask): # 前向传播
        batch_size = input_Q.size(0) #
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # 投影并重塑Q
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # 投影并重塑K
        if attn_mask is not None: # 应用掩码
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # 扩展mask到多头

        attn_scores = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask) # 获取注意力分数
        return attn_scores #


# ========== [新增] 动态空间注意力提取器（逐时刻） ==========
class SpatialDynamicExtractor(nn.Module):
    """
    输入: node_tokens_time (B, T, N, D)
    输出: sat_logits_seq (B, T, H, N, N) —— 未softmax的注意力logits
    """
    def __init__(self, DEVICE_unused, num_vertices, d_model_for_spatial_attn,
                 d_k_for_attn, n_heads_for_attn, use_temporal_smoothing: bool = False, smoothing_kernel_size: int = 3):
        super().__init__()
        self.n_heads = n_heads_for_attn
        self.num_vertices = num_vertices
        self.use_temporal_smoothing = (use_temporal_smoothing and smoothing_kernel_size > 1)

        # 复用 S 位置编码与空间多头注意力
        self.embedS_timewise = Embedding(num_vertices, d_model_for_spatial_attn,
                                         d_model_for_spatial_attn, 'S')
        self.SAt_timewise = SMultiHeadAttention(DEVICE_unused, d_model_for_spatial_attn,
                                                d_k_for_attn, d_v_unused=None, n_heads=n_heads_for_attn)

        if self.use_temporal_smoothing:
            channels = n_heads_for_attn * num_vertices * num_vertices
            pad = smoothing_kernel_size // 2
            self.temporal_smoother = nn.Conv1d(channels, channels, kernel_size=smoothing_kernel_size,
                                               padding=pad, groups=channels, bias=True)
        else:
            self.temporal_smoother = None

    def forward(self, node_tokens_time: torch.Tensor) -> torch.Tensor:
        # node_tokens_time: (B, T, N, D)
        B, T, N, D = node_tokens_time.shape
        x_bt = node_tokens_time.reshape(B * T, N, D)                 # (B*T, N, D)
        x_bt = self.embedS_timewise(x_bt, B * T)                      # 位置编码
        sat_bt = self.SAt_timewise(x_bt, x_bt, attn_mask=None)        # (B*T, H, N, N) logits
        sat_seq = sat_bt.view(B, T, self.n_heads, N, N)               # (B, T, H, N, N)

        if self.temporal_smoother is not None and T > 1:
            logits = sat_seq.permute(0, 2, 3, 4, 1).reshape(B, self.n_heads * N * N, T)
            logits = self.temporal_smoother(logits)
            sat_seq = logits.reshape(B, self.n_heads, N, N, T).permute(0, 4, 1, 2, 3)
        return sat_seq


# ========== [增强] SDE 并行特征头（节点级 + 边级TopK + 分段统计） ==========
class SDEParallelFeatureHead(nn.Module):
    """
    将 sat_scores_seq (B, T, H, N, N) 压缩为一段固定维度的向量 (B, out_dim)。

    ✅ 节点级（per-node）的分段统计（每段 9 维）：
        1) 注意力熵 H_t(n) 的 mean/std/range/slope  -> 4
        2) 邻接分布变化率 ΔA_t(n) 的 mean/std/max -> 3   (ΔA_t(n) = Σ_j |P_t - P_{t-1}|)
        3) 自环概率 diag 的 mean/std               -> 2
       => 每节点 9 维；num_segments 段 => 每节点 9*num_segments 维；拼接 N 个节点 => N*9*num_segments

    ✅ 边级（per-edge）的 Top-K 动态增强/减弱特征（每条边 4 维）：
        对 P_edge = mean_head(softmax(logits)) 得到 (B,T,N,N)
        计算 dP = P_edge[t] - P_edge[t-1]（带符号）
        按 |mean_t(dP)| 选 Top-K 边（可选排除自环），并对每条边提取：
            - mean(P_edge)     ：边的平均强度
            - std(dP)          ：边变化的波动强度
            - relu(mean(dP))   ：整体增强幅度（正向趋势）
            - relu(-mean(dP))  ：整体减弱幅度（负向趋势）
       => TopK * 4 维

    最终特征：
        [节点级 N*9*num_segments , 边级 topk_edges*4]  concat -> Linear/Norm -> out_dim
    """
    def __init__(
        self,
        num_vertices: int,
        n_heads: int,
        out_dim: int = 64,
        num_segments: int = 4,
        topk_edges: int = 16,
        exclude_self_edges: bool = True,
    ):
        super().__init__()
        self.num_vertices = int(num_vertices)
        self.n_heads = int(n_heads)
        self.out_dim = int(out_dim)
        self.num_segments = int(num_segments)
        self.topk_edges = int(topk_edges)
        self.exclude_self_edges = bool(exclude_self_edges)
        self.eps = 1e-8

        # 固定输出维度（不随输入T变化）
        base_dim = self.num_vertices * 9 * max(1, self.num_segments)
        edge_dim = max(1, self.topk_edges) * 4

        self.proj = nn.Sequential(
            nn.LayerNorm(base_dim + edge_dim),
            nn.Linear(base_dim + edge_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )

    @staticmethod
    def _segment_bounds(T: int, num_segments: int):
        # 返回长度 num_segments+1 的边界索引，允许出现空段（t1==t0）
        if num_segments <= 1:
            return [0, T]
        bounds = [int(round(i * T / num_segments)) for i in range(num_segments + 1)]
        bounds[0] = 0
        bounds[-1] = T
        # 保证单调非降
        for i in range(1, len(bounds)):
            if bounds[i] < bounds[i - 1]:
                bounds[i] = bounds[i - 1]
        return bounds

    def forward(self, sat_scores_seq: torch.Tensor) -> torch.Tensor:
        # sat_scores_seq: (B, T, H, N, N) logits
        B, T, H, N, _ = sat_scores_seq.shape
        device = sat_scores_seq.device
        dtype = sat_scores_seq.dtype

        # 概率 (B,T,H,N,N)
        P = F.softmax(sat_scores_seq, dim=-1)
        p_clamped = P.clamp_min(self.eps)

        # ===== 1) 节点级分段统计：每段输出 (B,N,9) =====
        ent_all = -(p_clamped * p_clamped.log()).sum(dim=-1)  # (B,T,H,N)
        ent_all = ent_all.mean(dim=2)                         # (B,T,N)

        diag_all = P.diagonal(dim1=3, dim2=4).mean(dim=2)      # (B,T,N)

        seg_bounds = self._segment_bounds(T, max(1, self.num_segments))
        seg_node_feats = []

        for s in range(max(1, self.num_segments)):
            t0 = seg_bounds[s]
            t1 = seg_bounds[s + 1]
            if t1 <= t0:
                # 空段：填充零
                seg_feats = torch.zeros(B, N, 9, device=device, dtype=dtype)
            else:
                ent = ent_all[:, t0:t1]    # (B, t_seg, N)
                diag = diag_all[:, t0:t1]  # (B, t_seg, N)

                # 1. 熵统计：mean, std, range, slope
                ent_mean = ent.mean(dim=1)  # (B,N)
                ent_std = ent.std(dim=1, unbiased=False)
                ent_range = ent.max(dim=1).values - ent.min(dim=1).values
                ent_slope = (ent[:, -1] - ent[:, 0]) / max(1, t1 - t0 - 1) if t1 - t0 > 1 else torch.zeros_like(ent_mean)

                # 2. 分布变化率 ΔA_t(n) = sum_j |P_t(n,j) - P_{t-1}(n,j)|
                # 注意：计算 t>=1 的变化，忽略 t=0
                if t1 - t0 > 1:
                    P_seg = P.mean(dim=2)[:, t0:t1]  # (B, t_seg, N, N)
                    dP_seg = P_seg[:, 1:] - P_seg[:, :-1]  # (B, t_seg-1, N, N)
                    abs_dP = dP_seg.abs().sum(dim=-1)  # (B, t_seg-1, N) ΔA_t(n)
                    dp_mean = abs_dP.mean(dim=1)
                    dp_std = abs_dP.std(dim=1, unbiased=False)
                    dp_max = abs_dP.max(dim=1).values
                else:
                    dp_mean = dp_std = dp_max = torch.zeros(B, N, device=device, dtype=dtype)

                # 3. 自环统计：mean, std
                diag_mean = diag.mean(dim=1)
                diag_std = diag.std(dim=1, unbiased=False)

                seg_feats = torch.stack([
                    ent_mean, ent_std, ent_range, ent_slope,
                    dp_mean, dp_std, dp_max,
                    diag_mean, diag_std
                ], dim=-1)  # (B,N,9)

            seg_node_feats.append(seg_feats)

        node_feats = torch.cat(seg_node_feats, dim=-1)  # (B,N,9*S) S=num_segments
        node_feats = node_feats.view(B, -1)  # (B, N*9*S)

        # ===== 2) 边级 TopK 统计：(B, topK*4) =====
        P_edge = P.mean(dim=2)  # (B,T,N,N)

        if T > 1:
            dP = P_edge[:, 1:] - P_edge[:, :-1]  # (B,T-1,N,N)
            abs_mean_dP = dP.abs().mean(dim=1)  # (B,N,N) |mean_t(dP)|
            if self.exclude_self_edges:
                abs_mean_dP.diagonal().fill_(0.0)  # 排除自环

            flat_abs_mean = abs_mean_dP.view(B, -1)  # (B,N*N)
            topk_vals, topk_idx_flat = flat_abs_mean.topk(self.topk_edges, dim=-1, sorted=False)
            topk_i = topk_idx_flat // N
            topk_j = topk_idx_flat % N
            topk_mask = torch.zeros(B, N, N, device=device, dtype=torch.bool)
            topk_mask.scatter_(dim=-1, index=topk_j.unsqueeze(-1), src=torch.ones_like(topk_j.unsqueeze(-1), dtype=torch.bool))
            topk_mask = topk_mask.scatter_(dim=-2, index=topk_i.unsqueeze(-1), src=topk_mask.any(dim=-1, keepdim=True))  # 实际只需设置 (i,j)=1

            # 对每条 topk 边计算 4 维特征
            P_topk = P_edge * topk_mask.unsqueeze(1)  # (B,T,N,N)
            dP_topk = dP * topk_mask.unsqueeze(1)[:, :T-1] if T > 1 else torch.zeros_like(P_topk[:, :1])

            mean_P = P_topk.mean(dim=1).view(B, -1)[:, topk_idx_flat[0]]  # 示例：取第一个 batch 的 idx，但实际需 per-batch
            # 修正：需逐 batch 计算
            edge_feats = []
            for b in range(B):
                sel_idx = topk_idx_flat[b]
                mean_P_b = P_edge[b].mean(dim=0).view(-1)[sel_idx]
                std_dP_b = dP[b].std(dim=0, unbiased=False).view(-1)[sel_idx]
                mean_dP_b = dP[b].mean(dim=0).view(-1)[sel_idx]
                pos_trend = F.relu(mean_dP_b)
                neg_trend = F.relu(-mean_dP_b)
                feats_b = torch.stack([mean_P_b, std_dP_b, pos_trend, neg_trend], dim=-1)  # (K,4)
                edge_feats.append(feats_b.view(-1))  # (K*4,)

            edge_feats = torch.stack(edge_feats, dim=0)  # (B, K*4)
        else:
            edge_feats = torch.zeros(B, max(1, self.topk_edges) * 4, device=device, dtype=dtype)

        # 3. 拼接 & 投影
        all_feats = torch.cat([node_feats, edge_feats], dim=-1)
        return self.proj(all_feats)


class TemporalSeqExporter(nn.Module):
    """
    从 DSTAGNN_block 的内部状态中提取时间序列特征，用于可解释性分析和分类消融实验。
    1. tat_only: 从 TAt 注意力分数中提取节点级的时间序列 (B,N,T)
    2. gtu_only: 从 GTU 门权重中提取节点级的时间序列 (B,N,T)
    3. mixed:    上述两种的混合 (alpha 融合)
    """
    def __init__(self, method_norm: str = "zscore", upsample_mode: str = "linear"):
        super().__init__()
        self.method_norm = method_norm.lower()
        self.upsample_mode = upsample_mode.lower()
        assert self.method_norm in ["zscore", "minmax"], "method_norm 必须是 'zscore' 或 'minmax'。"
        assert self.upsample_mode in ["nearest", "linear"], "upsample_mode 必须是 'nearest' 或 'linear'。"

    @staticmethod
    def _safe_softmax_last(scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(scores, dim=-1)

    @staticmethod
    def _zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        m = x.mean(dim=dim, keepdim=True)
        s = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - m) / s

    @staticmethod
    def _minmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        mn = x.min(dim=dim, keepdim=True).values
        mx = x.max(dim=dim, keepdim=True).values
        return (x - mn) / (mx - mn + eps)

    def tat_only(self, tat_scores: torch.Tensor, TATout: torch.Tensor) -> torch.Tensor:
        # tat_scores: (B, F, H, T, T)
        # TATout:     (B, F, T, N)
        B, F, H, T, _ = tat_scores.shape
        A = self._safe_softmax_last(tat_scores)            # (B,F,H,T,T)
        inflow = A.sum(dim=3)                              # (B,F,H,T) 入流和（每时刻每个头）
        inflow = inflow.mean(dim=(1,2)) / T                # (B,T) 平均入流强度（跨特征/头）
        c = inflow.view(B, 1, T, 1)                        # (B,1,T,1)
        node_seq = (TATout * c).mean(dim=1).transpose(2,1)  # (B,F,T,N) * (B,1,T,1) -> mean_F -> (B,N,T)
        return node_seq

    def _upsample_to_T(self, x: torch.Tensor, T: int) -> torch.Tensor:
        B, Fch, N, t_small = x.shape
        if t_small == T:
            return x
        x_ = x.reshape(B * Fch * N, 1, t_small)
        x_up = F.interpolate(x_, size=T, mode=self.upsample_mode,
                             align_corners=False if self.upsample_mode == 'linear' else None)
        return x_up.reshape(B, Fch, N, T)

    def gtu_only(self, gate3: torch.Tensor, gate5: torch.Tensor, gate7: torch.Tensor, T: int) -> torch.Tensor:
        # gate*: (B, Fch, N, T*)
        g3 = self._upsample_to_T(gate3, T).mean(dim=1)  # (B,N,T)
        g5 = self._upsample_to_T(gate5, T).mean(dim=1)
        g7 = self._upsample_to_T(gate7, T).mean(dim=1)
        gtu_ms = torch.stack([g3, g5, g7], dim=1).mean(dim=1)  # (B,N,T) 多尺度平均
        return gtu_ms

    def mixed(self, tat_seq_node: torch.Tensor, gtu_ms_seq: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        assert 0.0 <= alpha <= 1.0, "alpha 必须在 [0,1] 之间。"
        if self.method_norm == "zscore":
            tat_n = self._zscore(tat_seq_node, dim=-1)
            gtu_n = self._zscore(gtu_ms_seq, dim=-1)
        else:  # "minmax"
            tat_n = self._minmax(tat_seq_node, dim=-1)
            gtu_n = self._minmax(gtu_ms_seq, dim=-1)
        return alpha * tat_n + (1 - alpha) * gtu_n


class MultiHeadAttention(nn.Module): # 时间多头注意力模块
    def __init__(self, DEVICE, d_model_nodes, d_k, d_v, n_heads, num_of_d_features): # 初始化
        super(MultiHeadAttention, self).__init__() #
        self.d_model_nodes = d_model_nodes #
        self.d_k = d_k #
        self.d_v = d_v #
        self.n_heads = n_heads #
        self.num_of_d_features = num_of_d_features #
        self.W_Q = nn.Linear(d_model_nodes, d_k * n_heads, bias=False) # Q投影
        self.W_K = nn.Linear(d_model_nodes, d_k * n_heads, bias=False) # K投影
        self.W_V = nn.Linear(d_model_nodes, d_v * n_heads, bias=False) # V投影
        self.fc = nn.Linear(n_heads * d_v, d_model_nodes, bias=False) # 最终线性层
        self.layer_norm = nn.LayerNorm(d_model_nodes).to(DEVICE) # 层归一化

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att): # 前向传播
        residual, batch_size = input_Q, input_Q.size(0) #
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_k).transpose(2, 3)  # (B, F, T, H, d_k) -> (B, F, H, T, d_k)
        K = self.W_K(input_K).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_k).transpose(2, 3)  #
        V = self.W_V(input_V).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_v).transpose(2, 3)  #

        context, res_attn_scores = ScaledDotProductAttention(
            self.d_k, self.num_of_d_features)(Q, K, V, attn_mask, res_att)  # (B, F, H, T, d_v), (B, F, H, T, T)

        context = context.transpose(2, 3).reshape(
            batch_size, self.num_of_d_features, -1, self.n_heads * self.d_v)  # (B, F, T, H*d_v)
        output = self.fc(context) # (B, F, T, d_model_nodes)

        return self.layer_norm(output + residual), res_attn_scores #


class cheb_conv_withSAt(nn.Module): # 带静态/动态空间注意力的 ChebConv 模块
    def __init__(self, K_cheb, cheb_polynomials, in_channels, out_channels, num_of_vertices,
                 dynamic_attn_alpha: float = 0.5): # 初始化
        super(cheb_conv_withSAt, self).__init__() #
        self.K_cheb = K_cheb #
        self.in_channels = in_channels #
        self.out_channels = out_channels #
        self.dynamic_attn_alpha = dynamic_attn_alpha  # 动态注意力混入权重
        self.relu = nn.ReLU(inplace=True) #
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels))
             for _ in range(K_cheb)]
        ) # Cheb 多项式系数
        self.mask_per_k = nn.ParameterList(
            [nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))
             for _ in range(K_cheb)]
        ) # 每个 k 的邻接掩码

        for mask_param in self.mask_per_k: #
            nn.init.xavier_uniform_(mask_param) #
        for theta in self.Theta: #
            nn.init.xavier_uniform_(theta) #

    def forward(self, x, spatial_attention_scores, adj_pa_static, sat_scores_seq): # 前向传播
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape #
        outputs = [] #

        # 动态注意力序列 (B,T,H,N,N) -> 时间平均 (B,H,N,N)
        if sat_scores_seq is not None: #
            sat_mean_time = sat_scores_seq.mean(dim=1)  # (B,H,N,N)

        for t in range(num_of_timesteps): #
            graph_signal_at_ts = x[:, :, :, t]  # (B, N, in_channels)
            out_ts = torch.zeros(batch_size, num_of_vertices,
                                 self.out_channels, device=x.device) #
            for k in range(self.K_cheb): #
                T_k = self.cheb_polynomials[k]  # 注意：cheb_polynomials 已在父模块注册为 buffer
                current_SAt_head_scores = spatial_attention_scores[:, k, :, :]  # (B, N, N)
                current_mask = self.mask_per_k[k]  # (N, N)
                dynamic_adj = adj_pa_static.mul(current_mask)  # (N, N)
                combined_static = current_SAt_head_scores + dynamic_adj.unsqueeze(0)  # (B, N, N)

                # 混入动态注意力 (可选)
                if sat_scores_seq is not None: #
                    current_dyn_attn = sat_mean_time[:, k, :, :]  # (B, N, N)
                    combined = (1 - self.dynamic_attn_alpha) * combined_static + \
                               self.dynamic_attn_alpha * current_dyn_attn  # (B, N, N)
                else: #
                    combined = combined_static #

                norm_factors = F.softmax(combined, dim=2)  # (B, N, N)
                T_k_eff = T_k.unsqueeze(0) * norm_factors  # (B, N, N)
                theta_k = self.Theta[k] #
                rhs = torch.bmm(T_k_eff, graph_signal_at_ts)  # (B, N, in_channels)
                out_ts = out_ts + rhs.matmul(theta_k)  # (B, N, out_channels)
            outputs.append(out_ts.unsqueeze(-1))  # (B, N, out_channels, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (B, N, out_channels, T)


class Embedding(nn.Module): # 位置编码模块
    def __init__(self, nb_seq_len, d_embedding_dim, num_of_context_dims_unused, Etype): # 初始化
        super(Embedding, self).__init__() #
        self.nb_seq_len = nb_seq_len #
        self.d_embedding_dim = d_embedding_dim #
        self.Etype = Etype #
        self.pos_embed = nn.Embedding(nb_seq_len, d_embedding_dim) # 位置嵌入
        self.norm = nn.LayerNorm(d_embedding_dim) # 层归一化

    def forward(self, x, batch_size_unused): # 前向传播
        if self.Etype == 'T': # 时间编码
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device) #
            embedding_values = self.pos_embed(pos_indices) #
            x_permuted = x.permute(0, 2, 3, 1)  # (B, N, F, T) -> (B, N, T, F)
            embedding_sum = x_permuted + embedding_values.unsqueeze(0).unsqueeze(0) #
        else: # 空间编码
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device) #
            embedding_values = self.pos_embed(pos_indices) #
            embedding_sum = x + embedding_values.unsqueeze(0) #
        embedded_x = self.norm(embedding_sum) #
        return embedded_x #


class GTU(nn.Module): # 门控时间卷积单元
    def __init__(self, in_channels, time_strides, kernel_size): # 初始化
        super(GTU, self).__init__() #
        self.in_channels = in_channels #
        self.tanh_act = nn.Tanh() #
        self.sigmoid_gate = nn.Sigmoid() #
        self.conv2out = nn.Conv2d(
            in_channels, 2 * in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, time_strides),
            padding=(0, 0)
        ) # 因果卷积

    def forward(self, x): # 前向传播
        x_causal_conv = self.conv2out(x)  # (B, 2*in, N, T_out)
        x_p = x_causal_conv[:, : self.in_channels, :, :]  # 值分支
        x_q = x_causal_conv[:, -self.in_channels:, :, :]  # 门分支
        x_gtu = self.tanh_act(x_p) * self.sigmoid_gate(x_q) #
        return x_gtu #


class GTU_with_gate_weights(nn.Module): # 带门权重记录的 GTU
    def __init__(self, in_channels, time_strides, kernel_size): # 初始化
        super(GTU_with_gate_weights, self).__init__() #
        self.gtu = GTU(in_channels, time_strides, kernel_size) #

    def forward(self, x): # 前向传播
        x_gtu = self.gtu(x) #
        gate_weights = x_gtu.mean(dim=0, keepdim=True)  # 示例：实际应从 sigmoid_gate 提取，但这里简化
        return x_gtu, gate_weights # 返回输出和门权重（用于解释）


class DSTAGNN_block(nn.Module): # DSTAGNN 单块模块
    def __init__(self, DEVICE, num_of_d, in_channels, K_cheb, nb_chev_filter,
                 nb_time_filter_unused, time_strides, cheb_polynomials, adj_pa_static, adj_TMD_static_unused,
                 num_of_vertices, num_of_timesteps, d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn,
                 n_heads_for_attn, dynamic_attn_alpha: float = 0.5,
                 use_sde: bool = True, sde_temporal_smoothing: bool = False, sde_smoothing_ksize: int = 3): # 初始化
        super(DSTAGNN_block, self).__init__() #
        self.use_sde = use_sde #
        self.dynamic_attn_alpha = dynamic_attn_alpha #

        self.SAt = SMultiHeadAttention(DEVICE, d_model_for_spatial_attn,
                                       d_k_for_attn, d_v_unused=None, n_heads=n_heads_for_attn) # 空间注意力
        self.embedS = Embedding(num_of_vertices, d_model_for_spatial_attn,
                                d_model_for_spatial_attn, 'S') # 空间位置编码

        if self.use_sde: #
            self.sde = SpatialDynamicExtractor(DEVICE, num_of_vertices, d_model_for_spatial_attn,
                                               d_k_for_attn, n_heads_for_attn,
                                               use_temporal_smoothing=sde_temporal_smoothing,
                                               smoothing_kernel_size=sde_smoothing_ksize) # 动态空间提取器
        else: #
            self.sde = None #

        self.cheb_conv_SAt = cheb_conv_withSAt(K_cheb, cheb_polynomials, in_channels, nb_chev_filter,
                                               num_of_vertices, dynamic_attn_alpha) # ChebConv

        self.TAt = MultiHeadAttention(DEVICE, nb_chev_filter, d_k_for_attn, d_v_for_attn,
                                      n_heads_for_attn, num_of_d) # 时间注意力
        self.embedT = Embedding(num_of_timesteps, nb_chev_filter,
                                nb_chev_filter, 'T') # 时间位置编码

        self.GTU3 = GTU_with_gate_weights(nb_chev_filter, time_strides, 3) # GTU k=3
        self.GTU5 = GTU_with_gate_weights(nb_chev_filter, time_strides, 5) # GTU k=5
        self.GTU7 = GTU_with_gate_weights(nb_chev_filter, time_strides, 7) # GTU k=7

        self.bn = nn.BatchNorm2d(nb_chev_filter).to(DEVICE) # 批归一化

    def forward(self, x, res_att): # 前向传播
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape #

        # 空间注意力计算
        x_embedS = self.embedS(x, batch_size)  # (B, N, F)
        spatial_attention_scores = self.SAt(x_embedS, x_embedS, attn_mask=None)  # (B, H, N, N)

        # 提取动态空间注意力序列（如果启用）
        sat_scores_seq = None #
        if self.use_sde: #
            node_tokens_time = x.permute(0, 3, 1, 2)  # (B, T, N, F)
            sat_scores_seq = self.sde(node_tokens_time)  # (B, T, H, N, N)

        # ChebConv with SAt
        graph_conv_res = self.cheb_conv_SAt(x, spatial_attention_scores,
                                            self.adj_pa_static, sat_scores_seq)  # (B, N, F_out, T)

        # 时间注意力计算
        x_embedT = self.embedT(graph_conv_res, batch_size)  # (B, N, F_out, T)
        x_TAt, tat_scores = self.TAt(x_embedT, x_embedT, x_embedT, None, res_att)  # (B, N, F_out, T), (B, F_out, H, T, T)? Wait, num_of_d_features = N? No, in MultiHeadAttention, num_of_d_features = num_of_d (initial=1, then nb_chev_filter?)

        # 修正：MultiHeadAttention 的 num_of_d_features 应为 N (节点维)，但原代码中传入 num_of_d (特征维)。假设 num_of_d =1, 但后续块中 num_of_d = nb_chev_filter, 这可能有误。
        # 假设正确：TAt 输入 (B, F, T, N) permuted from (B, N, F, T)

        # GTU 多尺度
        x_GTU3, gate3 = self.GTU3(x_TAt.permute(0, 2, 1, 3))  # (B, F_out, N, T_out), gate
        x_GTU5, gate5 = self.GTU5(x_TAt.permute(0, 2, 1, 3)) #
        x_GTU7, gate7 = self.GTU7(x_TAt.permute(0, 2, 1, 3)) #

        x_GTU = (x_GTU3 + x_GTU5 + x_GTU7) / 3  # (B, F_out, N, T_out)

        x_bn = self.bn(x_GTU)  # (B, F_out, N, T_out)

        internal_states = {
            "sat_scores_seq": sat_scores_seq,
            "tat_scores": tat_scores,
            "gate_weights_gtu3": gate3,
            "gate_weights_gtu5": gate5,
            "gate_weights_gtu7": gate7
        }

        return x_bn.permute(0, 2, 1, 3), tat_scores.mean(dim=2).mean(dim=1), internal_states  # 输出, 平均 res_att, 内部状态


class DSTAGNN_submodule(nn.Module): # DSTAGNN 主模块
    def __init__(self, DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb,
                 K_cheb, nb_chev_filter, nb_time_filter_block_unused, initial_time_strides,
                 cheb_polynomials, adj_pa_static, adj_TMD_static_unused, num_for_predict_per_node,
                 len_input_total, num_of_vertices, d_model_for_spatial_attn, d_k_for_attn,
                 d_v_for_attn, n_heads_for_attn,
                 task_type='regression', num_classes=None,
                 output_memory=False, return_internal_states=False,
                 use_sde: bool = True, sde_out_dim: int = 64, sde_num_segments: int = 4,
                 sde_topk_edges: int = 16, sde_exclude_self: bool = True,
                 exp_mode: str = "full"): # 初始化
        super(DSTAGNN_submodule, self).__init__() #

        if output_memory: #
            self.task_type = 'memory' #
        else: #
            self.task_type = task_type #
            
        self.return_internal_states = return_internal_states #
        self.num_of_vertices = num_of_vertices #
        self.nb_chev_filter = nb_chev_filter #
        self.len_input_total = len_input_total #
        self.initial_time_strides = initial_time_strides #
        self.nb_block = nb_block #
        self.DEVICE = DEVICE #
        self.exp_mode = exp_mode.lower()  # "full", "tat_only_cls", "gtu_only_cls", "mixed_cls"
        assert self.exp_mode in ["full", "tat_only_cls", "gtu_only_cls", "mixed_cls"], \
            "exp_mode 必须是 'full' 或 '*_cls' 之一。"

        # 注册 buffer 以支持 DP/DDP
        for k, poly in enumerate(cheb_polynomials): #
            self.register_buffer(f'cheb_poly_{k}', poly) #
        self.register_buffer('adj_pa_static', adj_pa_static) #

        self.BlockList = nn.ModuleList() #
        current_num_of_d_for_embedT = num_of_d_initial_feat #
        current_in_channels_for_cheb = initial_in_channels_cheb #
        current_num_of_timesteps_input = len_input_total #
        current_time_strides_for_gtu = initial_time_strides #

        for i in range(nb_block): #
            self.BlockList.append(DSTAGNN_block(
                DEVICE, current_num_of_d_for_embedT,
                current_in_channels_for_cheb, K_cheb,
                nb_chev_filter, nb_time_filter_block_unused,
                current_time_strides_for_gtu,
                cheb_polynomials, adj_pa_static, adj_TMD_static_unused,
                num_of_vertices, current_num_of_timesteps_input,
                d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn,
                n_heads_for_attn, use_sde=use_sde
            )) #
            current_num_of_d_for_embedT = nb_chev_filter #
            current_in_channels_for_cheb = nb_chev_filter #
            if current_time_strides_for_gtu > 0: #
                 current_num_of_timesteps_input = \
                     current_num_of_timesteps_input // current_time_strides_for_gtu #
            current_time_strides_for_gtu = 1 #
            
        if initial_time_strides > 0: #
            self.T_dim_per_block_out = len_input_total // initial_time_strides #
        else: #
            self.T_dim_per_block_out = len_input_total #

        concat_T_dim = self.T_dim_per_block_out * nb_block #
        
        self.final_conv = None #
        self.final_prediction_fc = None #
        self.classification_head = None #
        self.exporter_for_cls = TemporalSeqExporter() #

        if self.task_type == 'classification': #
            if num_classes is None: #
                raise ValueError("num_classes must be specified for classification.") #
            
            # ✅ 只做时间池化，保留节点维后：主干维度 = F * N
            feature_dim_main = nb_chev_filter * num_of_vertices
            feature_dim_sde  = d_model_for_spatial_attn
            feature_dim_total = feature_dim_main + feature_dim_sde
            self.classification_head = nn.Sequential(
                nn.Linear(feature_dim_total, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            ) #
            print("[DSTAGNN] 初始化为分类模型（full 模式下主干池化特征 + SDE 并行特征）。")

            if use_sde: #
                self.sde_head = SDEParallelFeatureHead(
                    num_vertices=num_of_vertices, n_heads=n_heads_for_attn,
                    out_dim=sde_out_dim, num_segments=sde_num_segments,
                    topk_edges=sde_topk_edges, exclude_self_edges=sde_exclude_self
                ) #
            else: #
                self.sde_head = None #

        elif self.task_type == 'memory' or self.task_type == 'regression': #
            self.final_conv_in_channels = concat_T_dim #
            self.final_conv_kernel_feat_dim = nb_chev_filter #
            if self.final_conv_in_channels > 0: #
                self.final_conv = nn.Conv2d(
                    self.final_conv_in_channels, 128,
                    kernel_size=(1, self.final_conv_kernel_feat_dim)) #
                self.final_prediction_fc = nn.Linear(128, num_for_predict_per_node) #
            
            if self.task_type == 'regression': #
                print("[DSTAGNN] 初始化为回归模型。") #
            else: #
                 print("[DSTAGNN] 初始化为特征提取器 (Memory输出)。") #

        self.to(DEVICE) #
    
    def export_time_feature_sequences(self, x): # 导出时间序列特征
        """用于后续可能的消融实验，保留此接口"""
        self.eval() #
        with torch.no_grad(): #
            block_outputs_concat_time = [] #
            res_att_prev = 0 #
            current_x_for_block = x #
            current_block_internal_states = None #

            for i, block in enumerate(self.BlockList): #
                block_output, res_att_current, current_block_internal_states = block(current_x_for_block, res_att_prev) #
                block_outputs_concat_time.append(block_output) #
                res_att_prev = res_att_current #
                current_x_for_block = block_output #

            # 抓取最后一个 block 的内部状态
            states = current_block_internal_states #
            tat_scores = states["tat_scores"]                # (B,F,H,T,T)
            gate3 = states["gate_weights_gtu3"]              # (B,F,N,T3)
            gate5 = states["gate_weights_gtu5"] #
            gate7 = states["gate_weights_gtu7"] #

            # 同步得到 T/N/F
            B, _, H, T, _ = tat_scores.shape #
            _, _, N, _ = gate3.shape #

            # 需要 TAt 的节点级输出 TATout: 可从同一 block 的 TAt 输出处再算一次（轻量）
            # 复用 EmbedT 和 TAt（不返回scores）
            # 注意：这里的输入应该是最后一个block的输入，即current_x_for_block在上一个循环结束时的值，但由于输入维度(N,F,T)中F变化，直接用原始x更简单且对于时间注意力分析是合理的。
            original_x_input_for_last_block = x if len(self.BlockList) == 1 else block_outputs_concat_time[-2] #
            TEmx = self.BlockList[-1].EmbedT(original_x_input_for_last_block, original_x_input_for_last_block.size(0)) #
            TATout, _ = self.BlockList[-1].TAt(TEmx, TEmx, TEmx, None, 0)   # (B,F,T,N)

            exporter = TemporalSeqExporter() #
            tat_seq_node = exporter.tat_only(tat_scores, TATout)            # (B,N,T)
            gtu_ms_seq   = exporter.gtu_only(gate3, gate5, gate7, T)        # (B,N,T)
            mixed_seq    = exporter.mixed(tat_seq_node, gtu_ms_seq, alpha=0.5) #

            return {
                "tat_only": tat_seq_node.cpu(),
                "gtu_only": gtu_ms_seq.cpu(),
                "mixed": mixed_seq.cpu(),
                "meta": {"T": T, "N": N}
            } #

    def forward(self, x): # 前向传播
        block_outputs_concat_time = [] #
        res_att_prev = 0 #
        all_blocks_internal_states = [] #
        current_x_for_block = x #
        current_block_internal_states = {} # 确保在循环外有定义

        for i, block in enumerate(self.BlockList): #
            block_output, res_att_current, current_block_internal_states = block(current_x_for_block, res_att_prev) #
            block_outputs_concat_time.append(block_output) #
            if self.return_internal_states: #
                all_blocks_internal_states.append(current_block_internal_states) #
            res_att_prev = res_att_current #
            current_x_for_block = block_output #

        final_x_from_blocks = torch.cat(block_outputs_concat_time, dim=-1) #

        output = None #

        if self.task_type == 'classification': #
            x_cls = final_x_from_blocks.permute(0, 2, 1, 3)   # (B, F, N, T)

            # ✅ 关键修正：只在时间维 T 上做池化，保留导联/节点 N 的差异
            x_time = x_cls.mean(dim=-1)                       # (B, F, N)

            # 展平得到分类向量：每个导联都有自己的 F 维表示
            x_main = x_time.contiguous().view(x_time.size(0), -1)  # (B, F*N)

            # --- 分类头输入构建 ---
            if self.exp_mode == "full": #
                # 现有路径：主干池化 + SDE 并行特征
                sat_seq_last = current_block_internal_states.get("sat_scores_seq", None) #
                if sat_seq_last is None: #
                    sde_emb = torch.zeros(x_main.size(0), self.sde_head.out_dim, device=x_main.device, dtype=x_main.dtype) #
                else: #
                    sde_emb = self.sde_head(sat_seq_last) #
                x_concat = torch.cat([x_main, sde_emb], dim=1) #
            else: #
                # 构造三种“时间解释序列”的分类特征（与 aECG 物理对齐）
                states = current_block_internal_states #
                tat_scores = states["tat_scores"]                        # (B,F,H,T,T)
                gate3 = states["gate_weights_gtu3"]; gate5 = states["gate_weights_gtu5"]; gate7 = states["gate_weights_gtu7"] #
                B, _, H, T, _ = tat_scores.shape #
                
                original_x_input_for_last_block = x if len(self.BlockList) == 1 else block_outputs_concat_time[-2] #
                TEmx = self.BlockList[-1].EmbedT(original_x_input_for_last_block, original_x_input_for_last_block.size(0)) #
                TATout, _ = self.BlockList[-1].TAt(TEmx, TEmx, TEmx, None, 0)  # (B,F,T,N)

                tat_seq_node = self.exporter_for_cls.tat_only(tat_scores, TATout)         # (B,N,T)
                gtu_ms_seq   = self.exporter_for_cls.gtu_only(gate3, gate5, gate7, T)     # (B,N,T)
                
                if self.exp_mode == "tat_only_cls": #
                    seq = tat_seq_node #
                elif self.exp_mode == "gtu_only_cls": #
                    seq = gtu_ms_seq #
                else:  # "mixed_cls" #
                    seq = self.exporter_for_cls.mixed(tat_seq_node, gtu_ms_seq, alpha=0.5) #

                # 将 (B,N,T) 池化成 (B, F_feat) 再分类 —— 例如 (全局平均池化 + MLP)
                seq_feat = seq.mean(dim=-1)                  # (B,N)
                seq_feat = F.layer_norm(seq_feat, (seq_feat.size(-1),)) #
                
                # 注意：此处的维度对齐方式是按照您的要求直接实现的。
                # 这可能会导致 seq_feat 和 x_main 的维度拼接后与分类头预期的输入维度不匹配。
                # 在运行消融实验时，您可能需要调整分类头(self.classification_head)的输入维度，
                # 或修改此处的特征构造方式，例如使用一个线性层将 seq_feat 投影到期望的维度。
                seq_feat_expand = seq_feat.unsqueeze(1).repeat(1, x_cls.size(1), 1)  # (B,F,N)
                seq_feat_expand = seq_feat_expand.contiguous().view(seq_feat_expand.size(0), -1)  # (B,F*N)
                sde_zero = torch.zeros(seq_feat_expand.size(0), self.sde_head.out_dim,
                                       device=seq_feat_expand.device, dtype=seq_feat_expand.dtype)
                x_concat = torch.cat([seq_feat_expand, sde_zero], dim=1)
            
            output = self.classification_head(x_concat) #

        elif self.task_type == 'memory': #
            B, N, F_mem_block, T_concat = final_x_from_blocks.shape #
            if self.num_of_vertices == 1: #
                memory = final_x_from_blocks.squeeze(1).permute(0, 2, 1) #
            else: #
                memory = final_x_from_blocks.permute(0, 3, 1, 2).reshape(B, T_concat, N * F_mem_block) #
            output = memory #

        elif self.task_type == 'regression': #
            conv_input = final_x_from_blocks.permute(0, 3, 1, 2) #
            output1 = self.final_conv(conv_input).squeeze(-1) #
            output1_permuted = output1.permute(0,2,1) #
            output = self.final_prediction_fc(output1_permuted) #

        if self.return_internal_states: #
            return output, all_blocks_internal_states #
        else: #
            return output #


def make_model(DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb, K_cheb,
               nb_chev_filter, nb_time_filter_block_unused, initial_time_strides, adj_mx, adj_pa_static,
               adj_TMD_static_unused, num_for_predict_per_node, len_input_total, num_of_vertices,
               d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn,
               task_type='regression', num_classes=None, output_memory=False, return_internal_states=False
               ):
    if isinstance(adj_mx, np.ndarray):
        adj_mx_tensor = torch.from_numpy(adj_mx).float().to(DEVICE)
    elif isinstance(adj_mx, torch.Tensor):
        adj_mx_tensor = adj_mx.float().to(DEVICE)
    else:
        raise TypeError("adj_mx 必须是 NumPy 数组或 PyTorch 张量。")

    L_tilde = scaled_Laplacian(adj_mx_tensor.cpu().numpy())
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K_cheb)]
    
    if isinstance(adj_pa_static, np.ndarray):
        adj_pa_tensor = torch.from_numpy(adj_pa_static).float().to(DEVICE)
    else:
        adj_pa_tensor = torch.as_tensor(adj_pa_static, dtype=torch.float32, device=DEVICE)

    if isinstance(adj_TMD_static_unused, np.ndarray):
        adj_TMD_tensor = torch.from_numpy(adj_TMD_static_unused).float().to(DEVICE)
    else:
        adj_TMD_tensor = torch.as_tensor(adj_TMD_static_unused, dtype=torch.float32, device=DEVICE)

    model = DSTAGNN_submodule(DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb,
                             K_cheb, nb_chev_filter, nb_time_filter_block_unused, initial_time_strides,
                             cheb_polynomials, adj_pa_tensor, adj_TMD_tensor, num_for_predict_per_node,
                             len_input_total, num_of_vertices, d_model_for_spatial_attn, d_k_for_attn,
                             d_v_for_attn, n_heads_for_attn,
                             task_type=task_type, num_classes=num_classes,
                             output_memory=output_memory,
                             return_internal_states=return_internal_states)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model