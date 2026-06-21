#!/usr/bin/env bash
set -euo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

TRAIN_PID="${TRAIN_PID:-2501319}"
OUT="${OUT:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/runs/ttt_dynamic_carrier_observe20_raw_2cam224_ft/20260617_0634_observe20_raw_stride4_gpu1}"
META="${META:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/ttt_dynamic_carrier_observe20_200_lerobot/dynamic_carrier_generation_metadata.json}"
EVAL_OUT="${EVAL_OUT:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/evaluate_results/dynamic_carrier/20260617_observe20_raw_stride4_execution_start_n20}"
RUN_NAME="${RUN_NAME:-observe20_raw_stride4_step010000_execution_start}"
CKPT="${CKPT:-$OUT/checkpoints/weights/step_010000.pt}"
STATS="${STATS:-$OUT/dataset_stats.json}"
WGET_LOG="${WGET_LOG:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/wget-log}"
HOLD_LOG="${HOLD_LOG:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/logs/gpu_hold/after_observe20_raw_stride4_gpu1.log}"
HOLD_PID_FILE="${HOLD_PID_FILE:-$OUT/gpu_hold_after_train.pid}"
RUN_EVAL_AFTER_TRAIN="${RUN_EVAL_AFTER_TRAIN:-1}"

AFTER_TRAIN=0

holder_pid() {
  local pid=""
  if [[ -f "$HOLD_PID_FILE" ]]; then
    pid="$(cat "$HOLD_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi

  pid="$(pgrep -f "scripts/gpu_hold.py --physical-gpu 1" | head -n 1 || true)"
  if [[ -n "$pid" ]]; then
    echo "$pid"
  fi
}

start_holder() {
  local reason="${1:-manual}"
  local existing=""
  existing="$(holder_pid || true)"
  if [[ -n "$existing" ]]; then
    echo "[$(date -Is)] gpu holder already running pid=$existing reason=$reason"
    echo "$existing" > "$HOLD_PID_FILE"
    return 0
  fi

  mkdir -p "$(dirname "$HOLD_LOG")"
  echo "[$(date -Is)] starting gpu holder immediately reason=$reason"
  setsid env CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/gpu_hold.py \
    --physical-gpu 1 --fraction 0.60 --chunk-gib 2 \
    --matmul-size 12288 --matmul-batches 16 --sync-every 8 \
    --sleep 0.0 --poll-seconds 10 --idle-seconds 10 \
    --busy-util-threshold 20 --allow-memory-only-hold \
    --release-extra-memory-mib 2048 --probe-release-seconds 300 \
    --probe-window-seconds 20 \
    > "$HOLD_LOG" 2>&1 < /dev/null &
  local hp=$!
  echo "$hp" > "$HOLD_PID_FILE"
  echo "[$(date -Is)] gpu holder pid=$hp log=$HOLD_LOG"
}

stop_holder() {
  local reason="${1:-manual}"
  local pid=""
  pid="$(holder_pid || true)"
  if [[ -z "$pid" ]]; then
    echo "[$(date -Is)] no gpu holder to stop reason=$reason"
    return 0
  fi

  echo "[$(date -Is)] stopping gpu holder pid=$pid reason=$reason"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(date -Is)] gpu holder still alive; sending SIGKILL pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  if [[ "$AFTER_TRAIN" == "1" ]]; then
    start_holder "watcher-exit-after-train"
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$EVAL_OUT"
echo "[$(date -Is)] hold-first watcher started for train pid=$TRAIN_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  echo "[$(date -Is)] training still running"
  tail -n 12 "$OUT/train.log" || true
  sleep 600
done

AFTER_TRAIN=1
echo "[$(date -Is)] training pid exited; starting holder before any post-train work"
start_holder "training-complete"

if [[ ! -f "$CKPT" ]]; then
  echo "[$(date -Is)] missing checkpoint: $CKPT; holder left running"
  tail -n 80 "$OUT/train.log" || true
  exit 1
fi

if [[ ! -f "$STATS" ]]; then
  echo "[$(date -Is)] missing dataset stats: $STATS; holder left running"
  exit 1
fi

if [[ "$RUN_EVAL_AFTER_TRAIN" != "1" ]]; then
  echo "[$(date -Is)] RUN_EVAL_AFTER_TRAIN=$RUN_EVAL_AFTER_TRAIN; holder left running"
  exit 0
fi

# This preserves the no-idle guarantee at training completion, then frees GPU1
# only for the real eval workload and restores the holder afterwards.
sleep 5
stop_holder "starting-eval"
echo "[$(date -Is)] starting eval"
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
  task=ttt_dynamic_carrier_slow100_raw_2cam224_ft \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path="$STATS" \
  +EVALUATION.dynamic_metadata_path="$META" \
  EVALUATION.output_dir="$EVAL_OUT" \
  EVALUATION.num_trials=20 \
  +EVALUATION.case_start=0 \
  EVALUATION.num_steps_wait=0 \
  EVALUATION.replan_steps=10 \
  EVALUATION.visualize_future_video=false \
  +EVALUATION.max_rollout_videos=20 \
  +EVALUATION.in_domain_execution_start=true \
  EVALUATION.device=cuda \
  gpu_id=0 \
  seed=20260617 \
  +EVALUATION.run_name="$RUN_NAME" \
  +EVALUATION.action_infer_updates_ttt=false

VIDEO_DIR="$EVAL_OUT/$RUN_NAME/rollout_videos"
echo "$VIDEO_DIR" >> "$WGET_LOG"
echo "[$(date -Is)] eval finished; video_dir=$VIDEO_DIR"
start_holder "eval-complete"
