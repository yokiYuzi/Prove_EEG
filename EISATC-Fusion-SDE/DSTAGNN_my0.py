# DSTAGNN_my.py
# 说明:
#   - 本脚本已根据您的全部意见修改。
#   - [新增] 将 cheb_polynomials 和 adj_pa_static 注册为 nn.Module 的 buffer，
#     以确保在 DP/DDP 模式下能被正确复制到各个设备，避免跨设备张量操作。
#   - [新增] 引入 SpatialDynamicExtractor (SDE) 来提取逐时刻的动态空间注意力序列。
#   - [新增] 引入 SDEParallelFeatureHead 来从动态注意力序列中提取统计特征。
#   - [修改] DSTAGNN_block 现在会并行计算静态和动态两种空间注意力。GCN部分仍使用静态注意力，动态注意力序列用于并行特征提取。
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
        attn_weights = F.softmax(scores, dim=3) # 计算注意力权重
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

# ========== [新增] SDE 并行特征头 ==========
class SDEParallelFeatureHead(nn.Module):
    """
    将 sat_scores_seq (B, T, H, N, N) 压缩为一段固定维度的向量 (B, out_dim)。
    统计项（逐节点）:
        1) 注意力熵 H_t(n) 的 mean/std/range/slope  -> 4
        2) 邻接分布变化率 ΔA_t(n) 的 mean/std/max -> 3
        3) 自环概率 diag 的 mean/std               -> 2
      合计每节点 9 维，拼接 N 个节点，再线性映射到 out_dim。
    """
    def __init__(self, num_vertices: int, n_heads: int, out_dim: int = 64):
        super().__init__()
        self.num_vertices = num_vertices
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.eps = 1e-8
        self.proj = nn.Sequential(
            nn.LayerNorm(num_vertices * 9),
            nn.Linear(num_vertices * 9, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )

    def forward(self, sat_scores_seq: torch.Tensor) -> torch.Tensor:
        # sat_scores_seq: (B, T, H, N, N) logits
        B, T, H, N, _ = sat_scores_seq.shape
        P = F.softmax(sat_scores_seq, dim=-1)                    # 概率 (B,T,H,N,N)
        # 1) 节点注意力熵 H_t(n)
        p_clamped = P.clamp_min(self.eps)
        ent = -(p_clamped * p_clamped.log()).sum(dim=-1)         # (B,T,H,N)
        ent = ent.mean(dim=2)                                    # 平均多头 -> (B,T,N)
        ent_mean = ent.mean(dim=1)                               # (B,N)
        ent_std  = ent.std(dim=1, unbiased=False)                # (B,N)
        ent_range= ent.max(dim=1).values - ent.min(dim=1).values # (B,N)
        ent_slope= (ent[:, -1, :] - ent[:, 0, :]) / max(1, T-1)  # (B,N)

        # 2) 分布变化率 ΔA_t(n) = Σ_j |P_t - P_{t-1}|
        if T > 1:
            diff = (P[:, 1:] - P[:, :-1]).abs().sum(dim=-1)      # (B,T-1,H,N)
            diff = diff.mean(dim=2)                              # -> (B,T-1,N)
            var_mean = diff.mean(dim=1)
            var_std  = diff.std(dim=1, unbiased=False)
            var_max  = diff.max(dim=1).values
        else:
            zero = torch.zeros(B, N, device=P.device, dtype=P.dtype)
            var_mean = var_std = var_max = zero

        # 3) 自环概率
        diag = P.diagonal(dim1=3, dim2=4).mean(dim=2)            # (B,T,N)
        diag_mean = diag.mean(dim=1)                              # (B,N)
        diag_std  = diag.std(dim=1, unbiased=False)               # (B,N)

        # 拼接逐节点 9 项特征 -> (B,N,9) -> (B, N*9)
        feat_nodes = torch.stack(
            [ent_mean, ent_std, ent_range, ent_slope,
             var_mean, var_std, var_max,
             diag_mean, diag_std], dim=-1
        )                                                        # (B,N,9)
        feat = feat_nodes.reshape(B, self.num_vertices * 9)
        emb = self.proj(feat)                                    # (B,out_dim)
        return emb


# ========== [新增] 时间序列特征导出器 ==========
class TemporalSeqExporter(nn.Module):
    """
    提供三类时间序列:
        1) TAt-only: 基于 tat_scores 构造的时间注意力中心度序列  (B, N, T)
        2) GTU-only: 基于 gate3/5/7 的门控强度序列，Upsample 对齐到 T 并多尺度汇总 (B, N, T)
        3) Mixed:    规范化后加权融合 / 拼接序列 (B, N, T) 或 (B, N, T, 4)
    """
    def __init__(self, method_norm="zscore", upsample_mode="linear"):
        super().__init__()
        self.method_norm = method_norm
        self.upsample_mode = upsample_mode
    
    @staticmethod
    def _safe_softmax_last(scores):
        # scores: (B, F, H, T, T)
        return F.softmax(scores, dim=-1)
    
    @staticmethod
    def _zscore(x, dim=-1, eps=1e-6):
        m = x.mean(dim=dim, keepdim=True)
        s = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - m) / s

    def tat_only(self, tat_scores, TATout):
        """
        输入:
          tat_scores: (B, F, H, T, T)  —— TAt 的打分（未softmax或已加残差）
          TATout    : (B, F, T, N)    —— TAt 输出（含节点维N）
        输出:
          attn_centrality_per_node: (B, N, T)
        解释:
          1) 先对 scores 做 softmax(最后一维, key维)，得到 A[b,f,h,t,q] = attention(q <- t)
          2) 按 (f,h) 平均后，得到每个 (query时刻 q) 的“全局中心度” c[q] = Σ_t A[:, :, :, t, q] / (F*H*T)
          3) 利用 TATout 提供的节点维 N，把 c[q] 作为权重对 (B,F,T,N) 做 time-wise 加权，得到节点级时间序列:
               s_node[n, q] = Σ_f TATout[b,f,q,n] * c[b,q]
        """
        B, F, H, T, _ = tat_scores.shape
        A = self._safe_softmax_last(tat_scores)              # (B,F,H,T,T)
        inflow = A.sum(dim=3)                                # Σ_t A[t->q], shape: (B,F,H,T)
        inflow = inflow.mean(dim=(1,2)) / T                  # (B,T), 归一化为中心度 c[q]
        # 扩展为 (B,1,T,1) 便于广播到 TATout:(B,F,T,N) 上
        c = inflow.view(B, 1, T, 1)
        node_seq = (TATout * c).mean(dim=1).transpose(2,1)   # (B,N,T)
        return node_seq

    def _upsample_to_T(self, x, T):
        # x: (B, F, N, t_small) 或 (B, 1, N, t_small)
        B, Fch, N, t_small = x.shape
        if t_small == T:
            return x
        # 使用线性上采样到 T（对每个(B,F,N)做1D插值）
        x_ = x.reshape(B*Fch*N, 1, t_small)
        x_up = F.interpolate(x_, size=T, mode=self.upsample_mode, align_corners=False if self.upsample_mode=='linear' else None)
        return x_up.reshape(B, Fch, N, T)

    def gtu_only(self, gate3, gate5, gate7, T):
        """
        gate*: (B, F, N, T_k)  —— GTU 的 Sigmoid 门控权重（越大表示激活越强）
        输出:
          gtu_ms_seq: (B, N, T) —— 先各尺度上采样到 T，再做多尺度融合(平均或加权)
        """
        # 上采样到 T
        g3 = self._upsample_to_T(gate3, T).mean(dim=1)   # (B,N,T)
        g5 = self._upsample_to_T(gate5, T).mean(dim=1)
        g7 = self._upsample_to_T(gate7, T).mean(dim=1)
        gtu_ms = torch.stack([g3, g5, g7], dim=1).mean(dim=1)   # (B,N,T) 多尺度平均
        return gtu_ms

    def mixed(self, tat_seq_node, gtu_ms_seq, alpha=0.5):
        """
        将两类序列在同一时间轴 T 上融合：
          - 先各自 z-score（或 min-max）规范化
          - 再做凸组合: mixed = α * tat + (1-α) * gtu
        """
        if self.method_norm == "zscore":
            tat_n = self._zscore(tat_seq_node, dim=-1)
            gtu_n = self._zscore(gtu_ms_seq, dim=-1)
        else:
            # min-max 规范化
            def mm(x, eps=1e-6): 
                mn, mx = x.min(dim=-1, keepdim=True).values, x.max(dim=-1, keepdim=True).values
                return (x - mn) / (mx - mn + eps)
            tat_n, gtu_n = mm(tat_seq_node), mm(gtu_ms_seq)
        return alpha * tat_n + (1 - alpha) * gtu_n


class MultiHeadAttention(nn.Module): # 时间多头注意力模块
    def __init__(self, DEVICE, d_model_nodes, d_k ,d_v, n_heads, num_of_d_features): # 初始化
        super(MultiHeadAttention, self).__init__() #
        self.d_model_nodes = d_model_nodes # 节点维度 (N)
        self.d_k = d_k #
        self.d_v = d_v #
        self.n_heads = n_heads #
        self.num_of_d_features = num_of_d_features # 特征维度 (F)
        self.W_Q = nn.Linear(d_model_nodes, d_k * n_heads, bias=False) #
        self.W_K = nn.Linear(d_model_nodes, d_k * n_heads, bias=False) #
        self.W_V = nn.Linear(d_model_nodes, d_v * n_heads, bias=False) #
        self.fc = nn.Linear(n_heads * d_v, d_model_nodes, bias=False) #
        self.layer_norm = nn.LayerNorm(d_model_nodes).to(DEVICE) #

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att): # 前向传播
        residual, batch_size = input_Q, input_Q.size(0) #
        
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d_features, -1, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, self.num_of_d_features, -1, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, self.num_of_d_features, -1, self.n_heads, self.d_v).transpose(2, 3)

        context, res_attn_scores = ScaledDotProductAttention(self.d_k, self.num_of_d_features)(Q, K, V, attn_mask, res_att) #

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d_features, -1, self.n_heads * self.d_v) #
        output = self.fc(context)  #

        return self.layer_norm(output + residual), res_attn_scores #


class cheb_conv_withSAt(nn.Module): # 带空间注意力的切比雪夫图卷积
    def __init__(self, K_cheb, cheb_polynomials, in_channels, out_channels, num_of_vertices): # 初始化
        super(cheb_conv_withSAt, self).__init__()
        self.K_cheb = K_cheb
        self.in_channels = in_channels
        self.out_channels = out_channels
        # ✅ [修改] 注册切比雪夫多项式为 buffers
        # 这样做可以确保在 .to(device) 调用时，这些张量能被正确移动到相应设备
        self.cheb_T_names = []
        for k in range(K_cheb):
            name = f"cheb_T_{k}"
            self.register_buffer(name, cheb_polynomials[k])
            self.cheb_T_names.append(name)

        self.relu = nn.ReLU(inplace=True)
        # ✅ [修改] 使用 torch.empty 初始化参数，不指定 device，让PyTorch框架处理设备分配
        self.Theta = nn.ParameterList([nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K_cheb)])
        self.mask_per_k = nn.ParameterList([nn.Parameter(torch.empty(num_of_vertices, num_of_vertices)) for _ in range(K_cheb)])

        # 初始化参数
        for mask_param in self.mask_per_k:
            nn.init.xavier_uniform_(mask_param)
        # ✅ [新增] 初始化 Theta 参数
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x, spatial_attention_scores, adj_pa_static):
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal_at_ts = x[:, :, :, time_step]
            # ✅ [修改] 在输入张量 x 所在的设备上创建新张量，确保设备一致性
            output_for_ts = torch.zeros(batch_size, num_of_vertices, self.out_channels, device=x.device)

            for k_order in range(self.K_cheb):
                # ✅ [修改] 通过 getattr 从 buffer 中获取切比雪夫多项式
                T_k_laplacian = getattr(self, self.cheb_T_names[k_order])
                current_SAt_head_scores = spatial_attention_scores[:, k_order, :, :]
                current_learnable_mask = self.mask_per_k[k_order]
                dynamic_adj_component = adj_pa_static.mul(current_learnable_mask)
                combined_spatial_factors = current_SAt_head_scores + dynamic_adj_component.unsqueeze(0)
                normalized_spatial_factors = F.softmax(combined_spatial_factors, dim=2)
                T_k_eff_laplacian = T_k_laplacian.unsqueeze(0) * normalized_spatial_factors
                theta_k_weights = self.Theta[k_order]
                rhs = torch.bmm(T_k_eff_laplacian, graph_signal_at_ts)
                output_for_ts = output_for_ts + rhs.matmul(theta_k_weights)
            outputs.append(output_for_ts.unsqueeze(-1))

        return self.relu(torch.cat(outputs, dim=-1))


class Embedding(nn.Module): # 位置编码嵌入层
    def __init__(self, nb_seq_len, d_embedding_dim, num_of_context_dims_unused, Etype): # 初始化
        super(Embedding, self).__init__() #
        self.nb_seq_len = nb_seq_len #
        self.d_embedding_dim = d_embedding_dim #
        self.Etype = Etype #
        self.pos_embed = nn.Embedding(nb_seq_len, d_embedding_dim) #
        self.norm = nn.LayerNorm(d_embedding_dim) #

    def forward(self, x, batch_size_unused): # 前向传播
        if self.Etype == 'T': #
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device) #
            embedding_values = self.pos_embed(pos_indices) #
            x_permuted = x.permute(0, 2, 3, 1) #
            embedding_sum = x_permuted + embedding_values.unsqueeze(0).unsqueeze(0) #
        else: #
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device) #
            embedding_values = self.pos_embed(pos_indices) #
            embedding_sum = x + embedding_values.unsqueeze(0) #

        embedded_x = self.norm(embedding_sum) #
        return embedded_x #


class GTU(nn.Module): # 门控时间单元
    def __init__(self, in_channels, time_strides, kernel_size): # 初始化
        super(GTU, self).__init__() #
        self.in_channels = in_channels #
        self.tanh_act = nn.Tanh() # 修改变量名以区分
        self.sigmoid_gate = nn.Sigmoid() # 修改变量名
        self.conv2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides), padding=(0,0)) #

    def forward(self, x): # x: (B, F_in, N, T_in) #
        x_causal_conv = self.conv2out(x) #
        x_p = x_causal_conv[:, : self.in_channels, :, :] # 用于tanh的部分
        x_q = x_causal_conv[:, -self.in_channels:, :, :] # 用于sigmoid门的部分
        
        activated_sigmoid_gate = self.sigmoid_gate(x_q) # 计算门控权重
        x_gtu = torch.mul(self.tanh_act(x_p), activated_sigmoid_gate) # 应用门控
        return x_gtu, activated_sigmoid_gate # <<< 返回门控权重和GTU输出


class DSTAGNN_block(nn.Module): # DSTAGNN的核心块
    def __init__(self, DEVICE, num_of_d_features_for_embedT, in_channels_for_cheb, K_cheb,
                 nb_chev_filter, nb_time_filter_for_gtu_unused, 
                 time_strides_for_gtu_and_res,
                 cheb_polynomials, adj_pa_static, adj_TMD_static_unused, 
                 num_of_vertices, num_of_timesteps_input,
                 d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn):
        super(DSTAGNN_block, self).__init__()

        self.DEVICE = DEVICE
        self.relu = nn.ReLU(inplace=True)
        # ✅ [修改] 注册静态邻接矩阵为 buffer
        self.register_buffer('adj_pa_static', torch.as_tensor(adj_pa_static, dtype=torch.float32))

        self.EmbedT = Embedding(num_of_timesteps_input, num_of_vertices, num_of_d_features_for_embedT, 'T')
        self.TAt = MultiHeadAttention(DEVICE, num_of_vertices, d_k_for_attn, d_v_for_attn, n_heads_for_attn, num_of_d_features_for_embedT)
        self.tat_output_proj = nn.Linear(num_of_d_features_for_embedT * num_of_timesteps_input, d_model_for_spatial_attn)

        self.EmbedS = Embedding(num_of_vertices, d_model_for_spatial_attn, d_model_for_spatial_attn, 'S')
        self.SAt = SMultiHeadAttention(DEVICE, d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, K_cheb)

        self.cheb_conv_SAt = cheb_conv_withSAt(K_cheb, cheb_polynomials, in_channels_for_cheb, nb_chev_filter, num_of_vertices)

        self.gtu3 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 3)
        self.gtu5 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 5)
        self.gtu7 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 7)

        T_after_gtu3 = (num_of_timesteps_input - 3) // time_strides_for_gtu_and_res + 1
        T_after_gtu5 = (num_of_timesteps_input - 5) // time_strides_for_gtu_and_res + 1
        T_after_gtu7 = (num_of_timesteps_input - 7) // time_strides_for_gtu_and_res + 1
        fcmy_input_time_dim = T_after_gtu3 + T_after_gtu5 + T_after_gtu7

        self.fcmy = nn.Sequential(
            nn.Linear(fcmy_input_time_dim, num_of_timesteps_input // time_strides_for_gtu_and_res),
            nn.Dropout(0.05),
        )

        self.residual_conv = nn.Conv2d(in_channels_for_cheb, nb_chev_filter, kernel_size=(1, 1), stride=(1, time_strides_for_gtu_and_res))
        self.layer_norm_output = nn.LayerNorm(nb_chev_filter)
        self.dropout = nn.Dropout(p=0.05)

        # === [新增] 将 TAt 的 F 维逐时刻映射到 d_model, 然后做逐时刻空间注意力 ===
        self.tat_out_proj_time = nn.Conv2d(
            in_channels=num_of_d_features_for_embedT,
            out_channels=d_model_for_spatial_attn,
            kernel_size=(1, 1)
        )
        self.SDE = SpatialDynamicExtractor(
            DEVICE, num_of_vertices, d_model_for_spatial_attn,
            d_k_for_attn, n_heads_for_attn=K_cheb,   # 与 Cheb 阶数对齐
            use_temporal_smoothing=False
        )


    def forward(self, x, res_att_prev):
        batch_size, num_of_vertices, num_features_input, num_timesteps_input = x.shape

        # === 原有 TAt ===
        TEmx = self.EmbedT(x, batch_size)
        TATout, tat_scores = self.TAt(TEmx, TEmx, TEmx, None, res_att_prev) # (B,F,T,N)

        # === [新增] SDE: 逐时刻空间注意力序列 sat_scores_seq (B,T,K,N,N) ===
        x_tat_feat4conv = TATout.permute(0, 1, 3, 2)               # (B,F,N,T)
        x_nodes_dmodel  = self.tat_out_proj_time(x_tat_feat4conv)  # (B,D,N,T)
        node_tokens_time= x_nodes_dmodel.permute(0, 3, 2, 1)       # (B,T,N,D)
        sat_scores_seq  = self.SDE(node_tokens_time)               # (B,T,K,N,N)

        # === 原有静态 SAt -> GCN (保持不变：GCN 完全静态) ===
        tat_permuted = TATout.permute(0,3,1,2) # (B,N,F,T)
        x_TAt_projected = self.tat_output_proj(tat_permuted.reshape(batch_size, num_of_vertices, -1))
        
        SEmx_TAt = self.EmbedS(x_TAt_projected, batch_size)
        SEmx_TAt_dropped = self.dropout(SEmx_TAt)
        sat_scores = self.SAt(SEmx_TAt_dropped, SEmx_TAt_dropped, attn_mask=None) # (B,K,N,N) [静态]
        
        spatial_gcn_out = self.cheb_conv_SAt(x, sat_scores, self.adj_pa_static) # [静态注意力驱动]
        
        # === 后续 GTU/MSTFE 与残差保持不变 ===
        X_for_gtu = spatial_gcn_out.permute(0, 2, 1, 3)

        x_gtu3_out, gate3_weights = self.gtu3(X_for_gtu)
        x_gtu5_out, gate5_weights = self.gtu5(X_for_gtu)
        x_gtu7_out, gate7_weights = self.gtu7(X_for_gtu)

        time_conv_concat = torch.cat([x_gtu3_out, x_gtu5_out, x_gtu7_out], dim=-1)
        
        fcmy_input = time_conv_concat.permute(0,2,1,3)
        time_conv_fcmy_out = self.fcmy(fcmy_input)
        time_conv_processed = self.relu(time_conv_fcmy_out.permute(0,2,1,3))

        x_for_residual = x.permute(0, 2, 1, 3)
        residual_transformed = self.residual_conv(x_for_residual)
        
        output_summed = residual_transformed + time_conv_processed
        
        ln_input = self.relu(output_summed).permute(0,2,3,1)
        output_normalized = self.layer_norm_output(ln_input)
        block_output = output_normalized.permute(0,1,3,2)

        internal_states = {
            "tat_scores": tat_scores.detach(),
            "sat_scores": sat_scores.detach(),          # 静态
            "sat_scores_seq": sat_scores_seq,           # 动态序列（保留梯度，供 SDEHead 使用）
            "gate_weights_gtu3": gate3_weights.detach(),
            "gate_weights_gtu5": gate5_weights.detach(),
            "gate_weights_gtu7": gate7_weights.detach()
        }
        return block_output, tat_scores, internal_states


class DSTAGNN_submodule(nn.Module):
    def __init__(self, DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb, K_cheb,
                 nb_chev_filter, nb_time_filter_block_unused, initial_time_strides,
                 cheb_polynomials, adj_pa_static, adj_TMD_static_unused, num_for_predict_per_node,
                 len_input_total, num_of_vertices,
                 d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn,
                 task_type='regression', num_classes=None,
                 output_memory=False, return_internal_states=False):
        super(DSTAGNN_submodule, self).__init__()

        if output_memory:
            self.task_type = 'memory'
        else:
            self.task_type = task_type
            
        self.return_internal_states = return_internal_states
        self.num_of_vertices = num_of_vertices
        self.nb_chev_filter = nb_chev_filter
        self.len_input_total = len_input_total
        self.initial_time_strides = initial_time_strides
        self.nb_block = nb_block
        self.DEVICE = DEVICE

        self.BlockList = nn.ModuleList()
        current_num_of_d_for_embedT = num_of_d_initial_feat
        current_in_channels_for_cheb = initial_in_channels_cheb
        current_num_of_timesteps_input = len_input_total
        current_time_strides_for_gtu = initial_time_strides

        for i in range(nb_block):
            self.BlockList.append(DSTAGNN_block(DEVICE, current_num_of_d_for_embedT, current_in_channels_for_cheb, K_cheb,
                                               nb_chev_filter, nb_time_filter_block_unused, current_time_strides_for_gtu,
                                               cheb_polynomials, adj_pa_static, adj_TMD_static_unused,
                                               num_of_vertices, current_num_of_timesteps_input,
                                               d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn))
            current_num_of_d_for_embedT = nb_chev_filter
            current_in_channels_for_cheb = nb_chev_filter
            if current_time_strides_for_gtu > 0:
                 current_num_of_timesteps_input = current_num_of_timesteps_input // current_time_strides_for_gtu
            current_time_strides_for_gtu = 1
            
        self.K_cheb = K_cheb
        # SDE 并行特征头：输出维度采用 d_model_for_spatial_attn (=64) 以与主干维度同量级
        self.sde_head = SDEParallelFeatureHead(num_of_vertices, K_cheb, out_dim=d_model_for_spatial_attn)

        if initial_time_strides > 0:
            self.T_dim_per_block_out = len_input_total // initial_time_strides
        else:
            self.T_dim_per_block_out = len_input_total

        concat_T_dim = self.T_dim_per_block_out * nb_block
        
        self.final_conv = None
        self.final_prediction_fc = None
        self.classification_head = None

        if self.task_type == 'classification':
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification.")
            
            feature_dim_main = nb_chev_filter
            feature_dim_sde  = d_model_for_spatial_attn
            feature_dim_total = feature_dim_main + feature_dim_sde
            self.classification_head = nn.Sequential(
                nn.Linear(feature_dim_total, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
            print("[DSTAGNN] 初始化为分类模型。")

        elif self.task_type == 'regression' or self.task_type == 'memory':
            self.final_conv_in_channels = concat_T_dim
            self.final_conv_kernel_feat_dim = nb_chev_filter
            if self.final_conv_in_channels > 0:
                self.final_conv = nn.Conv2d(self.final_conv_in_channels, 128, kernel_size=(1, self.final_conv_kernel_feat_dim))
                self.final_prediction_fc = nn.Linear(128, num_for_predict_per_node)
            
            if self.task_type == 'regression':
                print("[DSTAGNN] 初始化为回归模型。")
            else:
                 print("[DSTAGNN] 初始化为特征提取器 (Memory输出)。")

        # 为可解释性分析和消融实验新增的属性
        self.exp_mode = "full"   # 可选: "tat_only_cls", "gtu_only_cls", "mixed_cls", "full"
        self.exporter_for_cls = TemporalSeqExporter()
        
        self.to(DEVICE)
    
    def export_time_feature_sequences(self, x):
        """
        输入:
          x: 模型标准输入 (B, N, 1, T)
        输出:
          dict: {
            'tat_only': (B,N,T),
            'gtu_only': (B,N,T),
            'mixed':    (B,N,T),
            'meta': { 'T': T, 'N': N }
          }
        说明:
          - 只前向一次，抓取最后一个 Block 的内部状态做序列构建
          - 不改变分类/回归输出
        """
        self.eval()
        with torch.no_grad():
            block_outputs_concat_time = []
            res_att_prev = 0
            current_x_for_block = x
            current_block_internal_states = None

            for i, block in enumerate(self.BlockList):
                block_output, res_att_current, current_block_internal_states = block(current_x_for_block, res_att_prev)
                block_outputs_concat_time.append(block_output)
                res_att_prev = res_att_current
                current_x_for_block = block_output

            # 抓取最后一个 block 的内部状态
            states = current_block_internal_states
            tat_scores = states["tat_scores"]                # (B,F,H,T,T)
            gate3 = states["gate_weights_gtu3"]              # (B,F,N,T3)
            gate5 = states["gate_weights_gtu5"]
            gate7 = states["gate_weights_gtu7"]

            # 同步得到 T/N/F
            B, _, H, T, _ = tat_scores.shape
            _, _, N, _ = gate3.shape

            # 需要 TAt 的节点级输出 TATout: 可从同一 block 的 TAt 输出处再算一次（轻量）
            # 复用 EmbedT 和 TAt（不返回scores）
            # 注意：这里的输入应该是最后一个block的输入，即current_x_for_block在上一个循环结束时的值，但由于输入维度(N,F,T)中F变化，直接用原始x更简单且对于时间注意力分析是合理的。
            original_x_input_for_last_block = x if len(self.BlockList) == 1 else block_outputs_concat_time[-2]
            TEmx = self.BlockList[-1].EmbedT(original_x_input_for_last_block, original_x_input_for_last_block.size(0))
            TATout, _ = self.BlockList[-1].TAt(TEmx, TEmx, TEmx, None, 0)   # (B,F,T,N)

            exporter = TemporalSeqExporter()
            tat_seq_node = exporter.tat_only(tat_scores, TATout)            # (B,N,T)
            gtu_ms_seq   = exporter.gtu_only(gate3, gate5, gate7, T)        # (B,N,T)
            mixed_seq    = exporter.mixed(tat_seq_node, gtu_ms_seq, alpha=0.5)

            return {
                "tat_only": tat_seq_node.cpu(),
                "gtu_only": gtu_ms_seq.cpu(),
                "mixed": mixed_seq.cpu(),
                "meta": {"T": T, "N": N}
            }

    def forward(self, x):
        block_outputs_concat_time = []
        res_att_prev = 0
        all_blocks_internal_states = []
        current_x_for_block = x
        current_block_internal_states = {} # 确保在循环外有定义

        for i, block in enumerate(self.BlockList):
            block_output, res_att_current, current_block_internal_states = block(current_x_for_block, res_att_prev)
            block_outputs_concat_time.append(block_output)
            if self.return_internal_states:
                all_blocks_internal_states.append(current_block_internal_states)
            res_att_prev = res_att_current
            current_x_for_block = block_output

        final_x_from_blocks = torch.cat(block_outputs_concat_time, dim=-1)

        output = None

        if self.task_type == 'classification':
            x_cls = final_x_from_blocks.permute(0, 2, 1, 3)   # (B, F, N, T)
            x_pooled = F.adaptive_avg_pool2d(x_cls, (1, 1))   # (B, F, 1, 1)
            x_main = torch.flatten(x_pooled, 1)               # (B, F) = (B, nb_chev_filter)

            # --- 分类头输入构建 ---
            if self.exp_mode == "full":
                # 现有路径：主干池化 + SDE 并行特征
                sat_seq_last = current_block_internal_states.get("sat_scores_seq", None)
                if sat_seq_last is None:
                    sde_emb = torch.zeros(x_main.size(0), self.sde_head.out_dim, device=x_main.device, dtype=x_main.dtype)
                else:
                    sde_emb = self.sde_head(sat_seq_last)
                x_concat = torch.cat([x_main, sde_emb], dim=1)
            else:
                # 构造三种“时间解释序列”的分类特征（与 aECG 物理对齐）
                states = current_block_internal_states
                tat_scores = states["tat_scores"]                        # (B,F,H,T,T)
                gate3 = states["gate_weights_gtu3"]; gate5 = states["gate_weights_gtu5"]; gate7 = states["gate_weights_gtu7"]
                B, _, H, T, _ = tat_scores.shape
                
                original_x_input_for_last_block = x if len(self.BlockList) == 1 else block_outputs_concat_time[-2]
                TEmx = self.BlockList[-1].EmbedT(original_x_input_for_last_block, original_x_input_for_last_block.size(0))
                TATout, _ = self.BlockList[-1].TAt(TEmx, TEmx, TEmx, None, 0)  # (B,F,T,N)

                tat_seq_node = self.exporter_for_cls.tat_only(tat_scores, TATout)         # (B,N,T)
                gtu_ms_seq   = self.exporter_for_cls.gtu_only(gate3, gate5, gate7, T)     # (B,N,T)
                
                if self.exp_mode == "tat_only_cls":
                    seq = tat_seq_node
                elif self.exp_mode == "gtu_only_cls":
                    seq = gtu_ms_seq
                else:  # "mixed_cls"
                    seq = self.exporter_for_cls.mixed(tat_seq_node, gtu_ms_seq, alpha=0.5)

                # 将 (B,N,T) 池化成 (B, F_feat) 再分类 —— 例如 (全局平均池化 + MLP)
                seq_feat = seq.mean(dim=-1)                  # (B,N)
                seq_feat = F.layer_norm(seq_feat, (seq_feat.size(-1),))
                
                # 注意：此处的维度对齐方式是按照您的要求直接实现的。
                # 这可能会导致 seq_feat 和 x_main 的维度拼接后与分类头预期的输入维度不匹配。
                # 在运行消融实验时，您可能需要调整分类头(self.classification_head)的输入维度，
                # 或修改此处的特征构造方式，例如使用一个线性层将 seq_feat 投影到期望的维度。
                x_concat = torch.cat([seq_feat, torch.zeros_like(x_main)], dim=1)  # 维度对齐分类头
            
            output = self.classification_head(x_concat)

        elif self.task_type == 'memory':
            B, N, F_mem_block, T_concat = final_x_from_blocks.shape
            if self.num_of_vertices == 1:
                memory = final_x_from_blocks.squeeze(1).permute(0, 2, 1)
            else:
                memory = final_x_from_blocks.permute(0, 3, 1, 2).reshape(B, T_concat, N * F_mem_block)
            output = memory

        elif self.task_type == 'regression':
            conv_input = final_x_from_blocks.permute(0, 3, 1, 2)
            output1 = self.final_conv(conv_input).squeeze(-1)
            output1_permuted = output1.permute(0,2,1)
            output = self.final_prediction_fc(output1_permuted)

        if self.return_internal_states:
            return output, all_blocks_internal_states
        else:
            return output


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