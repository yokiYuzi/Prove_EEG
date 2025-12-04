#!/usr/bin/env bash
# run_ddp_fast.sh — DDP极速跑法（固定用 GPU 1–7）

set -euo pipefail

# 选 GPU1–7；开启 NCCL 异常处理；减少日志阻塞；启用可扩展分配器；开启 cuDNN benchmark 由代码控制
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# 极速参数：非重叠窗口、较大的 batch、较少折数与 epoch、最小化导出
export BATCH_SIZE=2        # 12*7 卡 ≈ 84 全局 batch（看显存可再加/减）
export EPOCHS=8               # 先冲速度；要更高精度再加
export N_SPLITS=3             # 5→3 可节省 40% 时间
export STEP_SIZE=1500         # = WINDOW_SIZE（无重叠，IO/训练都更快）


# 运行（选择一个空闲端口）
torchrun --nproc_per_node=7 --master_port=29513 main_classi.py
