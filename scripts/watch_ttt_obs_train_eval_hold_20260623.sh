#!/usr/bin/env bash
set -eo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

RUN_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt/20260623_fp32_from_stage1_step010000_gpu01_obs_g10_execbw_g2_b4_acc1_lr5e4_save50_log1"
TRAIN_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260623_ttt_obs_fp32_from_stage1_step010000_gpu01_b4_acc1_lr5e4_save50_log1.log"
WATCH_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260623_ttt_obs_fp32_acc1_watch_eval_hold.log"
WGET_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/wget-log"

META="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/ttt_dynamic_carrier_slow100_observe10_shard00_lerobot/dynamic_carrier_generation_metadata.json"
EVAL_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/evaluate_results/dynamic_carrier/20260623_ttt_obs_fp32_acc1_final_in_domain_observe40_60_n20"
HOLDER_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/gpu_hold/after_ttt_obs_fp32_acc1_eval_gpu1.log"

mkdir -p "$(dirname "$WATCH_LOG")" "$(dirname "$HOLDER_LOG")" "$EVAL_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"
}

latest_checkpoint() {
  find "$RUN_DIR/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -1
}

latest_step() {
  local ckpt
  ckpt="$(latest_checkpoint || true)"
  if [[ -n "$ckpt" ]]; then
    basename "$ckpt" .pt | sed 's/^step_//'
  else
    echo "none"
  fi
}

train_running() {
  pgrep -f "scripts/train.py.*${RUN_DIR}" >/dev/null 2>&1
}

start_gpu1_holder() {
  if pgrep -f "scripts/gpu_hold.py.*--physical-gpu 1" >/dev/null 2>&1; then
    log "GPU1 holder already running"
    return 0
  fi
  mkdir -p "$(dirname "$HOLDER_LOG")"
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
  local ckpt step run_name eval_log
  ckpt="$(latest_checkpoint || true)"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "no checkpoint found; skip eval"
    return 1
  fi

  step="$(basename "$ckpt" .pt)"
  run_name="ttt_obs_fp32_acc1_${step}_observe40_60_execution_start"
  eval_log="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/eval_20260623/${run_name}.log"
  mkdir -p "$(dirname "$eval_log")"

  log "starting observe-then-act eval ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HYDRA_FULL_ERROR=1 DIFFSYNTH_MODEL_BASE_PATH=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true \
    .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
      task=ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt \
      "ckpt=${ckpt}" \
      "EVALUATION.dataset_stats_path=${RUN_DIR}/dataset_stats.json" \
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
    echo "=== $(date -Is) ttt_obs_fp32_acc1 observe_then_act eval ==="
    echo "checkpoint=${ckpt}"
    echo "eval_root=${EVAL_ROOT}"
    echo "run_name=${run_name}"
    find "${EVAL_ROOT}" -type f -name '*.mp4' | sort
  } >> "$WGET_LOG"

  log "eval finished; paths appended to ${WGET_LOG}"
}

log "watcher started run_dir=${RUN_DIR}"
while train_running; do
  step="$(latest_step)"
  log "training running latest_checkpoint_step=${step}"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  sleep 600
done

log "training process no longer running"
sleep 30

if grep -Eq '\[done\] (max_steps reached|training finished)' "$TRAIN_LOG"; then
  log "train completed according to trainer log"
  run_eval || log "eval failed"
  start_gpu1_holder
else
  log "train did not report trainer [done]; not running eval"
  tail -n 120 "$TRAIN_LOG" >> "$WATCH_LOG" 2>&1 || true
  start_gpu1_holder
fi

log "watcher finished"
