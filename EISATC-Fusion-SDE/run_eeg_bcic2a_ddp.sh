#!/usr/bin/env bash
set -euo pipefail

# 使用所有 8 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 你想跑的被试编号（1~9），默认 1
SUBJECT=${1:-1}

# 建议指定一个固定端口，避免和其他人冲突
MASTER_PORT=29501

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=${MASTER_PORT} \
  run_2a_fhnet.py \
  --subject "${SUBJECT}"
