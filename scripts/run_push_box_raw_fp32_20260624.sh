#!/usr/bin/env bash
set -eo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

RUN_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/libero_push_box_rollout_target_4way_clean_visible_straight_raw_2cam224_fp32/20260624_b2_acc8_gpu01"
TRAIN_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260624_push_box_raw_fp32_b2_acc8_gpu01.log"

mkdir -p "$(dirname "$TRAIN_LOG")" "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src \
PYTHONUNBUFFERED=1 \
HYDRA_FULL_ERROR=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIFFSYNTH_MODEL_BASE_PATH=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
.venv/bin/accelerate launch \
  --num_processes 2 \
  --multi_gpu \
  --mixed_precision no \
  --num_cpu_threads_per_process 24 \
  scripts/train.py \
  task=libero_push_box_rollout_target_4way_clean_visible_straight_raw_2cam224_fp32 \
  output_dir="$RUN_DIR" \
  batch_size=2 \
  gradient_accumulation_steps=8 \
  mixed_precision=no \
  max_steps=10000 \
  num_workers=4 \
  learning_rate=2.0e-5 \
  lr_scheduler_type=cosine \
  save_every=1000 \
  keep_last_checkpoints=2 \
  log_every=1 \
  wandb.enabled=false \
  2>&1 | tee "$TRAIN_LOG"

status=${PIPESTATUS[0]}
echo
echo "TRAIN_EXIT_STATUS=${status}"
exit "$status"
