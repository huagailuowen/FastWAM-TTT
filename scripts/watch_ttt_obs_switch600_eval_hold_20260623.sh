#!/usr/bin/env bash
set -eo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

HIGH_SESSION="ttt_obs_fp32_lr5e4_acc1_save50_gpu01_20260623"
LOW_SESSION="ttt_obs_fp32_lr2e4_const_nowarmup_acc1_resume600_gpu01_20260623"

HIGH_RUN_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt/20260623_fp32_from_stage1_step010000_gpu01_obs_g10_execbw_g2_b4_acc1_lr5e4_save50_log1"
LOW_RUN_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt/20260623_fp32_resume_step000600_gpu01_obs_g10_execbw_g2_b4_acc1_lr2e4_const_nowarmup_save50_log1"

HIGH_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260623_ttt_obs_fp32_from_stage1_step010000_gpu01_b4_acc1_lr5e4_save50_log1.log"
LOW_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260623_ttt_obs_fp32_resume_step000600_gpu01_b4_acc1_lr2e4_const_nowarmup_save50_log1.log"
WATCH_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260623_ttt_obs_fp32_switch600_nowarmup_eval_hold.log"
WGET_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/wget-log"

META="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/ttt_dynamic_carrier_slow100_observe10_shard00_lerobot/dynamic_carrier_generation_metadata.json"
EVAL_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/evaluate_results/dynamic_carrier/20260623_ttt_obs_fp32_switch600_nowarmup_final_in_domain_observe40_60_n20"
HOLDER_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/gpu_hold/after_ttt_obs_fp32_switch600_nowarmup_eval_gpu1.log"

mkdir -p "$(dirname "$WATCH_LOG")" "$(dirname "$LOW_LOG")" "$(dirname "$HOLDER_LOG")" "$LOW_RUN_DIR" "$EVAL_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -1
}

latest_step() {
  local run_dir="$1"
  local ckpt
  ckpt="$(latest_checkpoint "$run_dir" || true)"
  if [[ -n "$ckpt" ]]; then
    basename "$ckpt" .pt | sed 's/^step_//'
  else
    echo "none"
  fi
}

train_running_for() {
  local run_dir="$1"
  pgrep -f "scripts/train.py.*${run_dir}" >/dev/null 2>&1
}

wait_for_gpus_free() {
  local waited=0
  while true; do
    local used0 used1
    used0="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}')"
    used1="$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}')"
    if (( used0 < 4096 && used1 < 4096 )); then
      return 0
    fi
    log "waiting GPUs to free used0=${used0}MiB used1=${used1}MiB waited=${waited}s"
    sleep 10
    waited=$((waited + 10))
  done
}

stop_high_lr_training() {
  log "stopping high-LR training session=${HIGH_SESSION}"
  if tmux has-session -t "$HIGH_SESSION" 2>/dev/null; then
    tmux kill-session -t "$HIGH_SESSION" || true
  fi
  sleep 20
  if train_running_for "$HIGH_RUN_DIR"; then
    log "high-LR train still running after tmux kill; terminating matching train processes"
    pkill -TERM -f "scripts/train.py.*${HIGH_RUN_DIR}" || true
    sleep 20
  fi
  if train_running_for "$HIGH_RUN_DIR"; then
    log "high-LR train still running after TERM; killing matching train processes"
    pkill -KILL -f "scripts/train.py.*${HIGH_RUN_DIR}" || true
    sleep 10
  fi
}

start_low_lr_resume() {
  local ckpt="$HIGH_RUN_DIR/checkpoints/weights/step_000600.pt"
  if [[ ! -f "$ckpt" ]]; then
    log "missing required checkpoint ${ckpt}; cannot start low-LR resume"
    return 1
  fi
  if tmux has-session -t "$LOW_SESSION" 2>/dev/null; then
    log "low-LR session already exists"
    return 0
  fi
  wait_for_gpus_free
  log "starting low-LR resume from ${ckpt}"
  tmux new-session -d -s "$LOW_SESSION" "cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 DIFFSYNTH_MODEL_BASE_PATH=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true .venv/bin/accelerate launch --num_processes 2 --multi_gpu --mixed_precision no --num_cpu_threads_per_process 24 scripts/train.py task=ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt output_dir=$LOW_RUN_DIR mixed_precision=no resume=$ckpt batch_size=4 gradient_accumulation_steps=1 learning_rate=2.0e-4 lr_scheduler_type=constant lr_warmup_fraction=0 lr_warmup_steps=0 max_steps=29400 num_workers=4 save_every=50 keep_last_checkpoints=2 log_every=1 wandb.enabled=false 2>&1 | tee $LOW_LOG; status=\$?; echo; echo TRAIN_EXIT_STATUS=\$status; exec zsh"
}

start_gpu1_holder() {
  if pgrep -f "scripts/gpu_hold.py.*--physical-gpu 1" >/dev/null 2>&1; then
    log "GPU1 holder already running"
    return 0
  fi
  log "starting GPU1 holder"
  setsid env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/gpu_hold.py \
    --physical-gpu 1 \
    --fraction 0.60 \
    --chunk-gib 2 \
    --matmul-size 12288 \
    --matmul-batches 16 \
    --sync-every 8 \
    --sleep 0.0 \
    --poll-seconds 10 \
    --idle-seconds 10 \
    --busy-util-threshold 20 \
    --allow-memory-only-hold \
    --release-extra-memory-mib 2048 \
    --probe-release-seconds 300 \
    --probe-window-seconds 20 \
    >> "$HOLDER_LOG" 2>&1 < /dev/null &
}

run_eval() {
  local run_dir="$1"
  local ckpt step run_name eval_log stats_path
  ckpt="$(latest_checkpoint "$run_dir" || true)"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "no checkpoint found in ${run_dir}; skip eval"
    return 1
  fi
  stats_path="${run_dir}/dataset_stats.json"
  if [[ ! -f "$stats_path" ]]; then
    stats_path="${HIGH_RUN_DIR}/dataset_stats.json"
  fi

  step="$(basename "$ckpt" .pt)"
  run_name="ttt_obs_fp32_switch600_nowarmup_${step}_observe40_60_execution_start"
  eval_log="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/eval_20260623/${run_name}.log"
  mkdir -p "$(dirname "$eval_log")"

  log "starting observe-then-act eval ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HYDRA_FULL_ERROR=1 DIFFSYNTH_MODEL_BASE_PATH=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true \
    .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
      task=ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt \
      "ckpt=${ckpt}" \
      "EVALUATION.dataset_stats_path=${stats_path}" \
      "+EVALUATION.dynamic_metadata_path=${META}" \
      "EVALUATION.output_dir=${EVAL_ROOT}" \
      "EVALUATION.num_trials=20" \
      "+EVALUATION.case_start=0" \
      "EVALUATION.num_steps_wait=0" \
      "EVALUATION.replan_steps=10" \
      "+EVALUATION.max_rollout_videos=20" \
      "EVALUATION.visualize_future_video=true" \
      "+EVALUATION.max_prediction_videos=20" \
      "+EVALUATION.in_domain_execution_start=true" \
      "+EVALUATION.execution_start_frame=100" \
      "+EVALUATION.action_start_frame=100" \
      "+EVALUATION.run_name=${run_name}" \
      "+EVALUATION.observe_then_act_chunks=30" \
      "+EVALUATION.observe_then_act_interval=10" \
      "+EVALUATION.observe_then_act_update_interval=2" \
      "+EVALUATION.observe_then_act_frames_min=40" \
      "+EVALUATION.observe_then_act_frames_max=60" \
      "+EVALUATION.observe_then_act_random_frames=true" \
      "+EVALUATION.observe_then_act_observe_policy=dummy" \
      "+EVALUATION.observe_then_act_count_observe_steps=true" \
      "+EVALUATION.observe_then_act_update_during_observe=true" \
      "+EVALUATION.observe_then_act_update_during_policy=true" \
      "+EVALUATION.observe_then_act_action_infer_updates_ttt=false" \
      "+EVALUATION.action_infer_updates_ttt=false" \
      "EVALUATION.device=cuda" \
      "gpu_id=0" \
      "seed=20260623" \
    2>&1 | tee "$eval_log"

  {
    echo "=== $(date -Is) ttt_obs_fp32_switch600_nowarmup observe_then_act eval ==="
    echo "checkpoint=${ckpt}"
    echo "eval_root=${EVAL_ROOT}"
    echo "run_name=${run_name}"
    find "${EVAL_ROOT}" -type f -name '*.mp4' | sort
  } >> "$WGET_LOG"

  log "eval finished; paths appended to ${WGET_LOG}"
}

log "switch watcher started high_run=${HIGH_RUN_DIR} low_run=${LOW_RUN_DIR}"

while true; do
  high_step="$(latest_step "$HIGH_RUN_DIR")"
  log "high-LR phase running_or_waiting latest_checkpoint_step=${high_step}"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  if [[ -f "$HIGH_RUN_DIR/checkpoints/weights/step_000600.pt" && -d "$HIGH_RUN_DIR/checkpoints/state/step_000600" ]]; then
    log "found step_000600 checkpoint; switching to low LR"
    stop_high_lr_training
    start_low_lr_resume
    break
  fi
  if ! train_running_for "$HIGH_RUN_DIR"; then
    log "high-LR training stopped before step_000600; not switching"
    start_gpu1_holder
    exit 1
  fi
  sleep 60
done

while train_running_for "$LOW_RUN_DIR" || tmux has-session -t "$LOW_SESSION" 2>/dev/null; do
  low_step="$(latest_step "$LOW_RUN_DIR")"
  log "low-LR phase running latest_checkpoint_step=${low_step}"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  sleep 600
done

log "low-LR training process no longer running"
if grep -Eq '\[done\] (max_steps reached|training finished)' "$LOW_LOG"; then
  log "low-LR train completed according to trainer log"
  run_eval "$LOW_RUN_DIR" || log "eval failed"
  start_gpu1_holder
else
  log "low-LR train did not report trainer [done]; not running eval"
  tail -n 120 "$LOW_LOG" >> "$WATCH_LOG" 2>&1 || true
  start_gpu1_holder
fi

log "switch watcher finished"
