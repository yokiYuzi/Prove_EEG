# DSTAGNN_my.py (纯静态 + GTU，完全移除 SDE)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import scaled_Laplacian, cheb_polynomial


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        return scores


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d_features_unused):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask, res_att):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn_weights = F.softmax(scores, dim=3)
        context = torch.matmul(attn_weights, V)
        return context, scores


class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE_unused, d_model, d_k, d_v_unused, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn_scores = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn_scores


class TemporalSeqExporter(nn.Module):
    def __init__(self, method_norm="zscore", upsample_mode="linear"):
        super().__init__()
        self.method_norm = method_norm
        self.upsample_mode = upsample_mode
    
    @staticmethod
    def _safe_softmax_last(scores):
        return F.softmax(scores, dim=-1)
    
    @staticmethod
    def _zscore(x, dim=-1, eps=1e-6):
        m = x.mean(dim=dim, keepdim=True)
        s = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - m) / s

    def tat_only(self, tat_scores, TATout):
        B, F, H, T, _ = tat_scores.shape
        A = self._safe_softmax_last(tat_scores)
        inflow = A.sum(dim=3)
        inflow = inflow.mean(dim=(1,2)) / T
        c = inflow.view(B, 1, T, 1)
        node_seq = (TATout * c).mean(dim=1).transpose(2,1)
        return node_seq

    def _upsample_to_T(self, x, T):
        B, Fch, N, t_small = x.shape
        if t_small == T:
            return x
        x_ = x.reshape(B*Fch*N, 1, t_small)
        x_up = F.interpolate(x_, size=T, mode=self.upsample_mode,
                             align_corners=False if self.upsample_mode=='linear' else None)
        return x_up.reshape(B, Fch, N, T)

    def gtu_only(self, gate3, gate5, gate7, T):
        g3 = self._upsample_to_T(gate3, T).mean(dim=1)
        g5 = self._upsample_to_T(gate5, T).mean(dim=1)
        g7 = self._upsample_to_T(gate7, T).mean(dim=1)
        gtu_ms = torch.stack([g3, g5, g7], dim=1).mean(dim=1)
        return gtu_ms

    def mixed(self, tat_seq_node, gtu_ms_seq, alpha=0.5):
        if self.method_norm == "zscore":
            tat_n = self._zscore(tat_seq_node, dim=-1)
            gtu_n = self._zscore(gtu_ms_seq, dim=-1)
        else:
            def mm(x, eps=1e-6): 
                mn, mx = x.min(dim=-1, keepdim=True).values, x.max(dim=-1, keepdim=True).values
                return (x - mn) / (mx - mn + eps)
            tat_n, gtu_n = mm(tat_seq_node), mm(gtu_ms_seq)
        return alpha * tat_n + (1 - alpha) * gtu_n


class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model_nodes, d_k, d_v, n_heads, num_of_d_features):
        super(MultiHeadAttention, self).__init__()
        self.d_model_nodes = d_model_nodes
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d_features = num_of_d_features
        self.W_Q = nn.Linear(d_model_nodes, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model_nodes, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model_nodes, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model_nodes, bias=False)
        self.layer_norm = nn.LayerNorm(d_model_nodes).to(DEVICE)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, self.num_of_d_features, -1,
                                   self.n_heads, self.d_v).transpose(2, 3)

        context, res_attn_scores = ScaledDotProductAttention(
            self.d_k, self.num_of_d_features)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(
            batch_size, self.num_of_d_features, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        return self.layer_norm(output + residual), res_attn_scores


class cheb_conv_withSAt(nn.Module):
    def __init__(self, K_cheb, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        super(cheb_conv_withSAt, self).__init__()
        self.K_cheb = K_cheb
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cheb_T_names = []
        for k in range(K_cheb):
            name = f"cheb_T_{k}"
            self.register_buffer(name, cheb_polynomials[k])
            self.cheb_T_names.append(name)

        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels))
             for _ in range(K_cheb)]
        )
        self.mask_per_k = nn.ParameterList(
            [nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))
             for _ in range(K_cheb)]
        )

        for mask_param in self.mask_per_k:
            nn.init.xavier_uniform_(mask_param)
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)

    def forward(self, x, spatial_attention_scores, adj_pa_static):
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []

        for t in range(num_of_timesteps):
            graph_signal_at_ts = x[:, :, :, t]
            out_ts = torch.zeros(batch_size, num_of_vertices,
                                 self.out_channels, device=x.device)
            for k in range(self.K_cheb):
                T_k = getattr(self, self.cheb_T_names[k])
                current_SAt_head_scores = spatial_attention_scores[:, k, :, :]
                current_mask = self.mask_per_k[k]
                dynamic_adj = adj_pa_static.mul(current_mask)
                combined = current_SAt_head_scores + dynamic_adj.unsqueeze(0)
                norm_factors = F.softmax(combined, dim=2)
                T_k_eff = T_k.unsqueeze(0) * norm_factors
                theta_k = self.Theta[k]
                rhs = torch.bmm(T_k_eff, graph_signal_at_ts)
                out_ts = out_ts + rhs.matmul(theta_k)
            outputs.append(out_ts.unsqueeze(-1))

        return self.relu(torch.cat(outputs, dim=-1))


class Embedding(nn.Module):
    def __init__(self, nb_seq_len, d_embedding_dim, num_of_context_dims_unused, Etype):
        super(Embedding, self).__init__()
        self.nb_seq_len = nb_seq_len
        self.d_embedding_dim = d_embedding_dim
        self.Etype = Etype
        self.pos_embed = nn.Embedding(nb_seq_len, d_embedding_dim)
        self.norm = nn.LayerNorm(d_embedding_dim)

    def forward(self, x, batch_size_unused):
        if self.Etype == 'T':
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device)
            embedding_values = self.pos_embed(pos_indices)
            x_permuted = x.permute(0, 2, 3, 1)
            embedding_sum = x_permuted + embedding_values.unsqueeze(0).unsqueeze(0)
        else:
            pos_indices = torch.arange(self.nb_seq_len, dtype=torch.long, device=x.device)
            embedding_values = self.pos_embed(pos_indices)
            embedding_sum = x + embedding_values.unsqueeze(0)
        embedded_x = self.norm(embedding_sum)
        return embedded_x


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh_act = nn.Tanh()
        self.sigmoid_gate = nn.Sigmoid()
        self.conv2out = nn.Conv2d(
            in_channels, 2 * in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, time_strides),
            padding=(0, 0)
        )

    def forward(self, x):
        x_causal_conv = self.conv2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        gate = self.sigmoid_gate(x_q)
        x_gtu = torch.mul(self.tanh_act(x_p), gate)
        return x_gtu, gate


class DSTAGNN_block(nn.Module):
    def __init__(self, DEVICE, num_of_d_features_for_embedT, in_channels_for_cheb, K_cheb,
                 nb_chev_filter, nb_time_filter_for_gtu_unused, 
                 time_strides_for_gtu_and_res,
                 cheb_polynomials, adj_pa_static, adj_TMD_static_unused, 
                 num_of_vertices, num_of_timesteps_input,
                 d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn):
        super(DSTAGNN_block, self).__init__()

        self.DEVICE = DEVICE
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('adj_pa_static',
                             torch.as_tensor(adj_pa_static, dtype=torch.float32))

        self.EmbedT = Embedding(num_of_timesteps_input, num_of_vertices,
                                num_of_d_features_for_embedT, 'T')
        self.TAt = MultiHeadAttention(
            DEVICE, num_of_vertices, d_k_for_attn, d_v_for_attn,
            n_heads_for_attn, num_of_d_features_for_embedT
        )
        self.tat_output_proj = nn.Linear(
            num_of_d_features_for_embedT * num_of_timesteps_input,
            d_model_for_spatial_attn
        )

        self.EmbedS = Embedding(num_of_vertices, d_model_for_spatial_attn,
                                d_model_for_spatial_attn, 'S')
        self.SAt = SMultiHeadAttention(
            DEVICE, d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, K_cheb
        )

        self.cheb_conv_SAt = cheb_conv_withSAt(
            K_cheb, cheb_polynomials, in_channels_for_cheb,
            nb_chev_filter, num_of_vertices
        )

        self.gtu3 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 3)
        self.gtu5 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 5)
        self.gtu7 = GTU(nb_chev_filter, time_strides_for_gtu_and_res, 7)

        T_after_gtu3 = (num_of_timesteps_input - 3) // time_strides_for_gtu_and_res + 1
        T_after_gtu5 = (num_of_timesteps_input - 5) // time_strides_for_gtu_and_res + 1
        T_after_gtu7 = (num_of_timesteps_input - 7) // time_strides_for_gtu_and_res + 1
        fcmy_input_time_dim = T_after_gtu3 + T_after_gtu5 + T_after_gtu7

        self.fcmy = nn.Sequential(
            nn.Linear(fcmy_input_time_dim,
                      num_of_timesteps_input // time_strides_for_gtu_and_res),
            nn.Dropout(0.05),
        )

        self.residual_conv = nn.Conv2d(
            in_channels_for_cheb, nb_chev_filter,
            kernel_size=(1, 1),
            stride=(1, time_strides_for_gtu_and_res)
        )
        self.layer_norm_output = nn.LayerNorm(nb_chev_filter)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x, res_att_prev):
        batch_size, num_of_vertices, num_features_input, num_timesteps_input = x.shape

        TEmx = self.EmbedT(x, batch_size)
        TATout, tat_scores = self.TAt(TEmx, TEmx, TEmx, None, res_att_prev)  # (B,F,T,N)

        tat_permuted = TATout.permute(0, 3, 1, 2)  # (B,N,F,T)
        x_TAt_projected = self.tat_output_proj(
            tat_permuted.reshape(batch_size, num_of_vertices, -1))

        SEmx_TAt = self.EmbedS(x_TAt_projected, batch_size)
        SEmx_TAt_dropped = self.dropout(SEmx_TAt)
        sat_scores = self.SAt(SEmx_TAt_dropped, SEmx_TAt_dropped, attn_mask=None)  # (B,K,N,N)

        spatial_gcn_out = self.cheb_conv_SAt(x, sat_scores, self.adj_pa_static)

        X_for_gtu = spatial_gcn_out.permute(0, 2, 1, 3)
        x_gtu3_out, gate3 = self.gtu3(X_for_gtu)
        x_gtu5_out, gate5 = self.gtu5(X_for_gtu)
        x_gtu7_out, gate7 = self.gtu7(X_for_gtu)

        time_conv_concat = torch.cat(
            [x_gtu3_out, x_gtu5_out, x_gtu7_out], dim=-1)

        fcmy_input = time_conv_concat.permute(0, 2, 1, 3)
        time_conv_fcmy_out = self.fcmy(fcmy_input)
        time_conv_processed = self.relu(
            time_conv_fcmy_out.permute(0, 2, 1, 3))

        x_for_residual = x.permute(0, 2, 1, 3)
        residual_transformed = self.residual_conv(x_for_residual)

        output_summed = residual_transformed + time_conv_processed

        ln_input = self.relu(output_summed).permute(0, 2, 3, 1)
        output_normalized = self.layer_norm_output(ln_input)
        block_output = output_normalized.permute(0, 1, 3, 2)

        internal_states = {
            "tat_scores": tat_scores.detach(),
            "sat_scores": sat_scores.detach(),
            "gate_weights_gtu3": gate3.detach(),
            "gate_weights_gtu5": gate5.detach(),
            "gate_weights_gtu7": gate7.detach()
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
            self.BlockList.append(DSTAGNN_block(
                DEVICE, current_num_of_d_for_embedT,
                current_in_channels_for_cheb, K_cheb,
                nb_chev_filter, nb_time_filter_block_unused,
                current_time_strides_for_gtu,
                cheb_polynomials, adj_pa_static, adj_TMD_static_unused,
                num_of_vertices, current_num_of_timesteps_input,
                d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn,
                n_heads_for_attn
            ))
            current_num_of_d_for_embedT = nb_chev_filter
            current_in_channels_for_cheb = nb_chev_filter
            if current_time_strides_for_gtu > 0:
                 current_num_of_timesteps_input = \
                     current_num_of_timesteps_input // current_time_strides_for_gtu
            current_time_strides_for_gtu = 1
            
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
            # 只保留主干池化特征，彻底删除所有消融分支
            self.classification_head = nn.Sequential(
                nn.Linear(feature_dim_main, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
            print("[DSTAGNN] 初始化为分类模型（已移除 SDE，已锁定为 full 主干池化特征模式）。")

        elif self.task_type == 'regression' or self.task_type == 'memory':
            self.final_conv_in_channels = concat_T_dim
            self.final_conv_kernel_feat_dim = nb_chev_filter
            if self.final_conv_in_channels > 0:
                self.final_conv = nn.Conv2d(
                    self.final_conv_in_channels, 128,
                    kernel_size=(1, self.final_conv_kernel_feat_dim))
                self.final_prediction_fc = nn.Linear(128, num_for_predict_per_node)
            
            if self.task_type == 'regression':
                print("[DSTAGNN] 初始化为回归模型。")
            else:
                 print("[DSTAGNN] 初始化为特征提取器 (Memory输出)。")

        # 保留 exporter 只为了如果以后需要做消融分析时还能用
        self.exporter_for_cls = TemporalSeqExporter()
        
        self.to(DEVICE)
    
    def export_time_feature_sequences(self, x):
        """用于后续可能的消融实验，保留此接口"""
        self.eval()
        with torch.no_grad():
            block_outputs_concat_time = []
            res_att_prev = 0
            current_x_for_block = x

            for block in self.BlockList:
                block_output, res_att_current, internal_states = \
                    block(current_x_for_block, res_att_prev)
                block_outputs_concat_time.append(block_output)
                res_att_prev = res_att_current
                current_x_for_block = block_output

            states = internal_states
            tat_scores = states["tat_scores"]
            gate3 = states["gate_weights_gtu3"]
            gate5 = states["gate_weights_gtu5"]
            gate7 = states["gate_weights_gtu7"]

            B, _, H, T, _ = tat_scores.shape

            original_x_input_for_last_block = \
                x if len(self.BlockList) == 1 else block_outputs_concat_time[-2]
            TEmx = self.BlockList[-1].EmbedT(
                original_x_input_for_last_block,
                original_x_input_for_last_block.size(0))
            TATout, _ = self.BlockList[-1].TAt(TEmx, TEmx, TEmx, None, 0)

            tat_seq_node = self.exporter_for_cls.tat_only(tat_scores, TATout)
            gtu_ms_seq   = self.exporter_for_cls.gtu_only(gate3, gate5, gate7, T)
            mixed_seq    = self.exporter_for_cls.mixed(tat_seq_node, gtu_ms_seq, alpha=0.5)

            return {
                "tat_only": tat_seq_node.cpu(),
                "gtu_only": gtu_ms_seq.cpu(),
                "mixed": mixed_seq.cpu(),
                "meta": {"T": T, "N": self.num_of_vertices}
            }

    def forward(self, x):
        block_outputs_concat_time = []
        res_att_prev = 0
        all_blocks_internal_states = []
        current_x_for_block = x

        for block in self.BlockList:
            block_output, res_att_current, internal_states = \
                block(current_x_for_block, res_att_prev)
            block_outputs_concat_time.append(block_output)
            if self.return_internal_states:
                all_blocks_internal_states.append(internal_states)
            res_att_prev = res_att_current
            current_x_for_block = block_output

        final_x_from_blocks = torch.cat(block_outputs_concat_time, dim=-1)

        if self.task_type == 'classification':
            # 只使用主干多块特征在时间维度上的拼接 + 全局平均池化 → 分类
            x_cls = final_x_from_blocks.permute(0, 2, 1, 3)   # (B, F, N, T_concat)
            x_pooled = F.adaptive_avg_pool2d(x_cls, (1, 1))   # (B, F, 1, 1)
            x_main = torch.flatten(x_pooled, 1)              # (B, nb_chev_filter)
            output = self.classification_head(x_main)

        elif self.task_type == 'memory':
            B, N, F_mem_block, T_concat = final_x_from_blocks.shape
            if self.num_of_vertices == 1:
                memory = final_x_from_blocks.squeeze(1).permute(0, 2, 1)
            else:
                memory = final_x_from_blocks.permute(
                    0, 3, 1, 2).reshape(B, T_concat, N * F_mem_block)
            output = memory

        elif self.task_type == 'regression':
            conv_input = final_x_from_blocks.permute(0, 3, 1, 2)
            output1 = self.final_conv(conv_input).squeeze(-1)
            output1_permuted = output1.permute(0, 2, 1)
            output = self.final_prediction_fc(output1_permuted)

        if self.return_internal_states:
            return output, all_blocks_internal_states
        else:
            return output


def make_model(DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb, K_cheb,
               nb_chev_filter, nb_time_filter_block_unused, initial_time_strides, adj_mx, adj_pa_static,
               adj_TMD_static_unused, num_for_predict_per_node, len_input_total, num_of_vertices,
               d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn,
               task_type='regression', num_classes=None,
               output_memory=False, return_internal_states=False):
    if isinstance(adj_mx, np.ndarray):
        adj_mx_tensor = torch.from_numpy(adj_mx).float().to(DEVICE)
    elif isinstance(adj_mx, torch.Tensor):
        adj_mx_tensor = adj_mx.float().to(DEVICE)
    else:
        raise TypeError("adj_mx 必须是 NumPy 数组或 PyTorch 张量。")

    L_tilde = scaled_Laplacian(adj_mx_tensor.cpu().numpy())
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE)
        for i in cheb_polynomial(L_tilde, K_cheb)
    ]
    
    if isinstance(adj_pa_static, np.ndarray):
        adj_pa_tensor = torch.from_numpy(adj_pa_static).float().to(DEVICE)
    else:
        adj_pa_tensor = torch.as_tensor(adj_pa_static, dtype=torch.float32, device=DEVICE)

    if isinstance(adj_TMD_static_unused, np.ndarray):
        adj_TMD_tensor = torch.from_numpy(adj_TMD_static_unused).float().to(DEVICE)
    else:
        adj_TMD_tensor = torch.as_tensor(adj_TMD_static_unused, dtype=torch.float32, device=DEVICE)

    model = DSTAGNN_submodule(
        DEVICE, num_of_d_initial_feat, nb_block, initial_in_channels_cheb,
        K_cheb, nb_chev_filter, nb_time_filter_block_unused,
        initial_time_strides, cheb_polynomials, adj_pa_tensor, adj_TMD_tensor,
        num_for_predict_per_node, len_input_total, num_of_vertices,
        d_model_for_spatial_attn, d_k_for_attn, d_v_for_attn, n_heads_for_attn,
        task_type=task_type, num_classes=num_classes,
        output_memory=output_memory,
        return_internal_states=return_internal_states
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model