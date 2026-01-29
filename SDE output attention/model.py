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
"""

import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat

# ==================== 关键修复：正确导入缺失的类 ====================
from utils.TemInc_util import TemporalInception
from utils.CNNMHAS_util import CNNAttention
from utils.TCN_util import TemporalConvNet                    # 新增：TCN 模块
from utils.util import Conv2dWithConstraint, LinearWithConstraint  # 新增：带约束的线性层
# =====================================================================

class My_Model(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000,
                 kerSize=32, kerSize_Tem=4, F1=16, D=2, poolSize1=8, poolSize2=8,
                 heads_num=8, head_dim=8,
                 tcn_filters=32, tcn_kernelSize=4,
                 dropout_dep=0.1, dropout_temp=0.3, dropout_atten=0.3, dropout_tcn=0.3,
                 n_classes=4, device='cpu'):
        super(My_Model, self).__init__()
        self.F2 = F1 * D

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
            out_channels=F1*D,
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
            kerSize_1   = (1, kerSize_Tem*4),
            kerSize_2   = (1, kerSize_Tem*2),
            kerSize_3   = (1, kerSize_Tem),
            kerStr      = 1,
            out_chan    = self.F2//4,
            pool_ker    = (1, 3),
            pool_str    = 1,
            bias        = False,
            max_norm    = .5
        )
        self.bn_temp = nn.BatchNorm2d(num_features=self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(kernel_size=(1, poolSize2), stride=(1, poolSize2))
        self.drop_temp = nn.Dropout(p=dropout_temp)

        # ============================= Decision Fusion model ============================= 
        self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(
            in_features  = self.F2 * (samples // poolSize1 // poolSize2),
            out_features = n_classes,
            max_norm     = .5,
            bias         = True
        )

        # ============================= MSA model ============================= 
        self.layerNorm = nn.LayerNorm(normalized_shape=(samples // poolSize1 // poolSize2), eps=1e-6)
        self.cnnMSA = CNNAttention(
            dim         = self.F2,
            heads       = heads_num,
            dim_head    = head_dim,
            keral_size  = 3,
            patch_height= 1,
            patch_width = (samples // poolSize1 // poolSize2),
            dropout     = dropout_atten,
            max_norm1   = .5,
            max_norm2   = .5,
            device      = device,
            groups      = True
        )

        # ============================= TCN model ============================= 
        self.tcn_block = TemporalConvNet(
            num_inputs   = self.F2 * 2,
            num_channels = [tcn_filters*2, tcn_filters*2],
            kernel_size  = tcn_kernelSize,
            dropout      = dropout_tcn,
            bias         = False,
            WeightNorm   = True,
            group        = True,
            max_norm     = .5
        )

        # ============================= Decision Fusion model (TCN branch) ============================= 
        self.flatten_tcn = nn.Flatten()
        self.liner_tcn = LinearWithConstraint(
            in_features  = tcn_filters*2,
            out_features = n_classes,
            max_norm     = .5,
            bias         = True
        )

        # ============================= Decision Fusion 参数 ============================= 
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta_sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # ==================== 推荐写法（更安全、更规范） ====================
        if x.dim() != 4:           # 原来是 len(x.shape) is not 4，已改为推荐写法
            x = x.unsqueeze(1)
        # ===================================================================

        # ============================= EEGINC model ============================= 
        x = self.temp_conv(x)
        x = self.bn(x)
        x = self.conv_depth(x)
        x = self.drop_depth(self.avgpool_depth(self.act_depth(self.bn_depth(x))))
        x = self.incept_temp(x)
        x = self.drop_temp(self.avgpool_temp(self.act_temp(self.bn_temp(x))))  # (B, F2, 1, T)

        eegFatures = x   # (B, F2, 1, T)

        # ============================= EEG 分支分类 ============================= 
        eeg_out = self.liner_eeg(self.flatten_eeg(x))   # (B, n_classes)

        # ============================= MSA model ============================= 
        x = self.layerNorm(x)
        x = self.cnnMSA(x)               # (B, F2, 1, T)
        msaFatures = x

        # ============================= Feature Fusion ============================= 
        fusionFeature = torch.cat((eegFatures, msaFatures), dim=1)   # (B, F2*2, 1, T)

        # ============================= TCN model ============================= 
        x = torch.squeeze(fusionFeature, dim=2)   # (B, F2*2, T)
        x = self.tcn_block(x)                     # (B, tcn_filters*2, T)
        x = x[:, :, -1]                           # 取最后一个时间步 (B, tcn_filters*2)
        tcnFeature = x

        tcn_out = self.liner_tcn(self.flatten_tcn(tcnFeature))   # (B, n_classes)

        # ============================= 决策融合 ============================= 
        fusionDecision = self.beta_sigmoid(self.beta) * eeg_out + \
                        (1 - self.beta_sigmoid(self.beta)) * tcn_out

        out = self.softmax(fusionDecision)

        return out


# ============================== 测试代码 ==============================
if __name__ == "__main__":
    input = torch.randn(32, 22, 1000)          # (batch, channels, samples)
    model = My_Model(eeg_chans=22, samples=1000, n_classes=4, device='cpu')
    out = model(input)
    print('===============================================================')
    print('output shape:', out.shape)         # 应该输出 torch.Size([32, 4])
    print('model forward success! No NameError anymore!')
    stat(model, (1, 22, 1000))