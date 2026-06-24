#!/usr/bin/env bash
set -eo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

RUN_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/libero_push_box_rollout_target_4way_clean_visible_straight_raw_2cam224_fp32/20260624_b2_acc8_gpu01"
TRAIN_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260624_push_box_raw_fp32_b2_acc8_gpu01.log"
WATCH_LOG="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/train/20260624_push_box_raw_fp32_b2_acc8_monitor.log"
HOLDER_LOG_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/gpu_hold"
POLL_SECONDS=5
LOG_SECONDS=2400

mkdir -p "$(dirname "$WATCH_LOG")" "$HOLDER_LOG_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"
}

train_running() {
  pgrep -f "scripts/run_push_box_raw_fp32_20260624.sh|scripts/train.py.*${RUN_DIR}|accelerate launch.*scripts/train.py.*${RUN_DIR}" >/dev/null 2>&1
}

latest_checkpoint_step() {
  local ckpt
  ckpt="$(find "$RUN_DIR/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$ckpt" ]]; then
    basename "$ckpt" .pt | sed 's/^step_//'
  else
    echo "none"
  fi
}

start_gpu_holder() {
  local gpu="$1"
  local log_path="${HOLDER_LOG_DIR}/after_push_box_raw_fp32_gpu${gpu}.log"
  if pgrep -f "scripts/gpu_hold.py.*--physical-gpu ${gpu}" >/dev/null 2>&1; then
    log "GPU${gpu} holder already running"
    return 0
  fi
  log "starting GPU${gpu} holder"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 .venv/bin/python scripts/gpu_hold.py \
    --physical-gpu "${gpu}" \
    --fraction 0.90 \
    --chunk-gib 4 \
    --matmul-size 16384 \
    --matmul-batches 32 \
    --sync-every 4 \
    --sleep 0.0 \
    --poll-seconds 10 \
    --idle-seconds 10 \
    --busy-util-threshold 20 \
    --allow-memory-only-hold \
    --release-extra-memory-mib 2048 \
    --probe-release-seconds 300 \
    --probe-window-seconds 20 \
    >> "$log_path" 2>&1 < /dev/null &
}

log "monitor started for existing push-box raw run"
next_log_ts=0
while train_running; do
  now_ts="$(date +%s)"
  if (( now_ts >= next_log_ts )); then
    step="$(latest_checkpoint_step)"
    log "training running latest_checkpoint_step=${step}"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
    tail -n 20 "$TRAIN_LOG" >> "$WATCH_LOG" 2>&1 || true
    next_log_ts=$((now_ts + LOG_SECONDS))
  fi
  sleep "$POLL_SECONDS"
done

log "training process no longer running"
if grep -Eq '\[done\] (max_steps reached|training finished)' "$TRAIN_LOG"; then
  log "train completed according to trainer log"
else
  log "train did not complete cleanly"
  tail -n 160 "$TRAIN_LOG" >> "$WATCH_LOG" 2>&1 || true
fi

log "starting GPU holders after training exit"
start_gpu_holder 0
start_gpu_holder 1
log "monitor finished"
