""" 
Copyright (C) 2023 Qufu Normal University, Guangjin Liang
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Guangjin Liang

[修改说明 - 方案C: 并行跨导联注意力转移分支]
1) 在 temp_conv+BN 后、conv_depth 前分叉：保留 C=22 导联维，提取动态跨导联注意力序列 (B,Te,H,C,C)
2) 使用 SDEParallelFeatureHead 将注意力转移统计压缩为 (B,out_dim)，并作为第三路决策 sde_out
3) 在原论文的 β 融合 (EEG vs TCN) 基础上新增 α 融合 (原融合 vs SDE)，且 α 初始化偏向原模型，保证训练策略不被破坏
4) 模型 forward 返回 logits（不再 softmax），与 CrossEntropyLoss 正确匹配
"""

import os 
import sys

current_path = os.path.abspath(os.path.dirname(__file__))

# ---- 兼容性路径注入：避免 "utils.py" 与 "utils/" 包冲突 ----
# 你工程里存在 utils.py，会导致 "from utils.xxx import yyy" 报错：
#   'utils' is not a package
# 因此统一采用 “按文件名直接 import”，并把可能存在的工具目录加入 sys.path。
candidate_dirs = [
    current_path,
    os.path.join(current_path, "utils"),
    os.path.join(current_path, "dataLoad", "utils"),
]
for d in candidate_dirs:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)
# -----------------------------------------------------------

import torch
import torch.nn as nn
from torchstat import stat

# ====== 原模型依赖（按文件名导入）======
from utils.TemInc_util import TemporalInception
from utils.CNNMHAS_util import CNNAttention
from utils.TCN_util import TemporalConvNet
from utils.util import Conv2dWithConstraint, LinearWithConstraint
# =====================================

# ====== 方案C新增：跨导联动态注意力分支 ======
from sde_util import SpatialDynamicExtractor, SDEParallelFeatureHead
# ===========================================


class My_Model(nn.Module):
    def __init__(
        self,
        eeg_chans=22,
        samples=1000,
        kerSize=32,
        kerSize_Tem=4,
        F1=16,
        D=2,
        poolSize1=8,
        poolSize2=8,
        heads_num=8,
        head_dim=8,
        tcn_filters=32,
        tcn_kernelSize=4,
        dropout_dep=0.1,
        dropout_temp=0.3,
        dropout_atten=0.3,
        dropout_tcn=0.3,
        n_classes=4,
        device='cpu',
        # ====== 方案C参数（默认打开，但对原模型影响最小）======
        use_sde: bool = True,
        sde_d_model: int = 64,
        sde_heads: int = 8,
        sde_d_k: int = 8,
        sde_out_dim: int = 64,
        dropout_sde: float = 0.10,
        alpha_init: float = 3.0,  # sigmoid(3)≈0.95 => 初期几乎等于原模型
        # ==================================================
    ):
        super(My_Model, self).__init__()

        # 记录关键超参
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.poolSize1 = poolSize1
        self.poolSize2 = poolSize2
        self.samples = samples
        self.eeg_chans = eeg_chans

        self.use_sde = use_sde

        # ============================= EEGINC model =============================
        self.temp_conv = Conv2dWithConstraint(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kerSize),
            stride=1,
            padding='same',
            bias=False,
            max_norm=.5
        )
        self.bn = nn.BatchNorm2d(num_features=F1)

        self.conv_depth = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(eeg_chans, 1),
            groups=F1,
            bias=False,
            max_norm=.5
        )
        self.bn_depth = nn.BatchNorm2d(num_features=self.F2)
        self.act_depth = nn.ELU()
        self.avgpool_depth = nn.AvgPool2d(kernel_size=(1, poolSize1), stride=(1, poolSize1))
        self.drop_depth = nn.Dropout(p=dropout_dep)

        self.incept_temp = TemporalInception(
            in_chan     = self.F2,
            kerSize_1   = (1, kerSize_Tem * 4),
            kerSize_2   = (1, kerSize_Tem * 2),
            kerSize_3   = (1, kerSize_Tem),
            kerStr      = 1,
            out_chan    = self.F2 // 4,
            pool_ker    = (1, 3),
            pool_str    = 1,
            bias        = False,
            max_norm    = .5
        )
        self.bn_temp = nn.BatchNorm2d(num_features=self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(kernel_size=(1, poolSize2), stride=(1, poolSize2))
        self.drop_temp = nn.Dropout(p=dropout_temp)

        # ============================= EEG 分支分类 (Pe) =============================
        self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(
            in_features  = self.F2 * (samples // poolSize1 // poolSize2),
            out_features = n_classes,
            max_norm     = .5,
            bias         = True
        )

        # ============================= MSA model (时间注意力) =============================
        self.layerNorm = nn.LayerNorm(normalized_shape=(samples // poolSize1 // poolSize2), eps=1e-6)
        self.cnnMSA = CNNAttention(
            dim          = self.F2,
            heads        = heads_num,
            dim_head     = head_dim,
            keral_size   = 3,
            patch_height = 1,
            patch_width  = (samples // poolSize1 // poolSize2),
            dropout      = dropout_atten,
            max_norm1    = .5,
            max_norm2    = .5,
            device       = device,
            groups       = True
        )

        # ============================= TCN model (Pt 分支) =============================
        self.tcn_block = TemporalConvNet(
            num_inputs   = self.F2 * 2,
            num_channels = [tcn_filters * 2, tcn_filters * 2],
            kernel_size  = tcn_kernelSize,
            dropout      = dropout_tcn,
            bias         = False,
            WeightNorm   = True,
            group        = True,
            max_norm     = .5
        )

        self.flatten_tcn = nn.Flatten()
        self.liner_tcn = LinearWithConstraint(
            in_features  = tcn_filters * 2,
            out_features = n_classes,
            max_norm     = .5,
            bias         = True
        )

        # ============================= 决策融合参数 β（原模型已有） =============================
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta_sigmoid = nn.Sigmoid()

        # ============================= 方案C: SDE 分支（跨导联注意力转移） =============================
        if self.use_sde:
            # (B,F1,C,T) -> (B,F1,C,Te)
            self.sde_pool1 = nn.AvgPool2d(kernel_size=(1, poolSize1), stride=(1, poolSize1))
            self.sde_pool2 = nn.AvgPool2d(kernel_size=(1, poolSize2), stride=(1, poolSize2))

            # token 投影：F1 -> sde_d_model
            self.sde_proj = nn.Linear(F1, sde_d_model, bias=False)
            self.sde_proj_norm = nn.LayerNorm(sde_d_model)

            # 动态空间注意力：输出 (B,Te,H,C,C) logits
            self.sde_attn = SpatialDynamicExtractor(
                num_vertices=eeg_chans,
                d_model=sde_d_model,
                d_k=sde_d_k,
                n_heads=sde_heads,
                use_lead_embed=True,              # 可消融：False
                use_temporal_smoothing=False,     # 可消融：True
                smoothing_kernel_size=3
            )

            # 注意力转移统计 -> embedding
            self.sde_head = SDEParallelFeatureHead(
                num_vertices=eeg_chans,
                n_heads=sde_heads,
                out_dim=sde_out_dim,
                dropout=dropout_sde
            )

            # SDE 分支分类输出 Ps (logits)
            self.liner_sde = LinearWithConstraint(
                in_features=sde_out_dim,
                out_features=n_classes,
                max_norm=.5,
                bias=True
            )

            # α 融合： P = σ(α)*P_et + (1-σ(α))*P_s
            self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32))
            self.alpha_sigmoid = nn.Sigmoid()
        # =================================================================================

    def forward(self, x, return_sde: bool = False):
        """
        默认返回 logits (B, n_classes)，用于 nn.CrossEntropyLoss。
        如果 return_sde=True，则额外返回 dict（sat_logits_seq 等，用于你验证“注意力转移”现象）。
        """
        if x.dim() != 4:
            x = x.unsqueeze(1)  # (B,1,C,T)

        # ====== 1) temp conv + BN（此处仍保留导联维 C，最适合做跨导联注意力）======
        x_temp = self.bn(self.temp_conv(x))  # (B,F1,C,T)

        # ====== 2) 方案C：并行 SDE 分支（不影响主干）======
        if self.use_sde:
            x_sde = self.sde_pool2(self.sde_pool1(x_temp))  # (B,F1,C,Te)
            x_sde = x_sde.permute(0, 3, 2, 1).contiguous()  # (B,Te,C,F1)

            x_sde = self.sde_proj(x_sde)                    # (B,Te,C,Ds)
            x_sde = self.sde_proj_norm(x_sde)

            sat_logits_seq = self.sde_attn(x_sde)           # (B,Te,H,C,C)
            sde_emb = self.sde_head(sat_logits_seq)         # (B,out_dim)
            sde_out = self.liner_sde(sde_emb)               # (B,n_classes) logits
        else:
            sat_logits_seq, sde_emb, sde_out = None, None, None

        # ====== 3) 主干：EEGINC (conv_depth 后导联维变 1) ======
        x_main = self.conv_depth(x_temp)  # (B,F2,1,T)
        x_main = self.drop_depth(self.avgpool_depth(self.act_depth(self.bn_depth(x_main))))
        x_main = self.incept_temp(x_main)
        x_main = self.drop_temp(self.avgpool_temp(self.act_temp(self.bn_temp(x_main))))  # (B,F2,1,Te)

        eegFatures = x_main
        eeg_out = self.liner_eeg(self.flatten_eeg(x_main))          # (B,n_classes) logits

        x_msa = self.layerNorm(x_main)
        x_msa = self.cnnMSA(x_msa)                                  # (B,F2,1,Te)
        msaFatures = x_msa

        fusionFeature = torch.cat((eegFatures, msaFatures), dim=1)  # (B,2F2,1,Te)
        x_tcn = torch.squeeze(fusionFeature, dim=2)                 # (B,2F2,Te)
        x_tcn = self.tcn_block(x_tcn)                               # (B,2*tcn_filters,Te)
        x_tcn = x_tcn[:, :, -1]                                     # (B,2*tcn_filters)
        tcn_out = self.liner_tcn(self.flatten_tcn(x_tcn))            # (B,n_classes) logits

        # ====== 4) 决策融合：先 β（原模型），再 α（方案C）======
        beta_w = self.beta_sigmoid(self.beta)
        fusion_et = beta_w * eeg_out + (1.0 - beta_w) * tcn_out

        if self.use_sde:
            alpha_w = self.alpha_sigmoid(self.alpha)
            fusion_all = alpha_w * fusion_et + (1.0 - alpha_w) * sde_out
        else:
            alpha_w = None
            fusion_all = fusion_et

        if return_sde:
            extra = {
                "beta_w": beta_w.detach().cpu(),
                "alpha_w": (alpha_w.detach().cpu() if alpha_w is not None else None),
                "sat_logits_seq": (sat_logits_seq.detach().cpu() if sat_logits_seq is not None else None),
                "sde_emb": (sde_emb.detach().cpu() if sde_emb is not None else None),
                "sde_out": (sde_out.detach().cpu() if sde_out is not None else None),
            }
            return fusion_all, extra

        return fusion_all


if __name__ == "__main__":
    inp = torch.randn(4, 22, 1000)  # (B,C,T)
    model = My_Model(eeg_chans=22, samples=1000, n_classes=4, device='cpu', use_sde=True)
    logits, extra = model(inp, return_sde=True)
    print("logits:", logits.shape)
    print("alpha:", extra["alpha_w"])
    print("beta:", extra["beta_w"])
    if extra["sat_logits_seq"] is not None:
        print("sat_logits_seq:", extra["sat_logits_seq"].shape)  # (B,Te,H,C,C)
    stat(model, (1, 22, 1000))
