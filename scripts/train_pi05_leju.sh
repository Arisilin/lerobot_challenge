#!/bin/bash
# 单/多 GPU 训练 TextVLA（CALVIN dream 数据）
# 参考：train_textvla_libero.sh + train_vlaada_calvin.sh

set -euo pipefail

# HF 镜像与代理
export HF_ENDPOINT=https://hf-mirror.com
unset all_proxy ALL_PROXY HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

# ==================== 配置区域 ====================
NUM_GPUS=7                # GPU 数量，改成你的卡数
BATCH_SIZE=16             # 单卡 batch size
CHUNK_SIZE=24              # 与配置的 chunk_size / n_action_steps 对齐
PRETRAINED_PATH=./ckpts/pi05_base/
RESUME=false              # 若要从中断继续，置为 true 并填写 RESUME_CONFIG_PATH
RESUME_CONFIG_PATH=null
JOB_NAME=pi05_leju
DATA_PATH=./data/leju_task1/
# ==================== 配置区域 ====================


# 多 GPU 训练
accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  $(which lerobot-train) \
  --dataset.root=${DATA_PATH} \
  --dataset.repo_id=${DATA_PATH} \
  --num_workers=16 \
  --batch_size=${BATCH_SIZE} \
  --wandb.enable=true \
  --wandb.mode=offline \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.pretrained_path=${PRETRAINED_PATH} \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${CHUNK_SIZE} \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --job_name=${JOB_NAME} \
  --eval_freq=10000 \
  --save_freq=5000 \
  --log_freq=100 \
  --steps=80000 \
  --policy.gradient_checkpointing=true

# 如果需要恢复训练，取消下面的注释并设置 RESUME=true 和 RESUME_CONFIG_PATH
#   --resume=${RESUME} \
#   --resume_config_path=${RESUME_CONFIG_PATH} \
