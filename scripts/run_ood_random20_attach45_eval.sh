#!/usr/bin/env bash
set -eo pipefail

cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM

mkdir -p logs/eval_20260610

OUT="./evaluate_results/dynamic_carrier/20260610_ood_random20_attach45"
META="${OUT}/dynamic_carrier_ood_random20_seed20260610_metadata.json"

COMMON_ARGS=(
  "+EVALUATION.dynamic_metadata_path=${META}"
  "EVALUATION.output_dir=${OUT}"
  "EVALUATION.num_trials=20"
  "+EVALUATION.case_start=0"
  "EVALUATION.num_steps_wait=0"
  "EVALUATION.replan_steps=10"
  "EVALUATION.visualize_future_video=false"
  "+EVALUATION.max_rollout_videos=20"
  "+EVALUATION.in_domain_execution_start=true"
  "+EVALUATION.execution_start_frame=100"
  "+EVALUATION.action_start_frame=100"
  "+EVALUATION.grasp_release_distance_override=0.045"
  "EVALUATION.device=cuda"
  "gpu_id=0"
  "seed=20260610"
)

echo "[$(date -Is)] starting raw OOD random20 attach45 eval"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
  task=ttt_dynamic_carrier_slow100_raw_2cam224_ft \
  ckpt=runs/ttt_dynamic_carrier_slow100_raw_2cam224_ft/20260607_1100_gpu0_b16/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=runs/ttt_dynamic_carrier_slow100_raw_2cam224_ft/20260607_1100_gpu0_b16/dataset_stats.json \
  +EVALUATION.run_name=raw_step010000_ood_random20_attach45 \
  +EVALUATION.action_infer_updates_ttt=false \
  "${COMMON_ARGS[@]}" 2>&1 | tee logs/eval_20260610/raw_step010000_ood_random20_attach45.log

echo "[$(date -Is)] starting stage1 W0 OOD random20 attach45 eval"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
  task=ttt_dynamic_carrier_stage1_w0_sft_2cam224_video_ttt \
  ckpt=runs/ttt_dynamic_carrier_stage1_w0_sft_2cam224_video_ttt/20260607_1100_gpu1_b16/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=runs/ttt_dynamic_carrier_stage1_w0_sft_2cam224_video_ttt/20260607_1100_gpu1_b16/dataset_stats.json \
  +EVALUATION.run_name=stage1_step010000_w0_ood_random20_no_inner_update_attach45 \
  +EVALUATION.action_infer_updates_ttt=false \
  +EVALUATION.observe_then_act_frames_min=40 \
  +EVALUATION.observe_then_act_frames_max=60 \
  +EVALUATION.observe_then_act_random_frames=true \
  "${COMMON_ARGS[@]}" 2>&1 | tee logs/eval_20260610/stage1_step010000_w0_ood_random20_attach45.log

echo "[$(date -Is)] starting TTT step002000 OOD random20 attach45 observe eval"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 .venv/bin/python experiments/dynamic_carrier/eval_dynamic_carrier_single.py \
  task=ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt \
  ckpt=runs/checkpoint_backups/ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt_20260608_1016_step002000_20260610_024324/checkpoints/weights/step_002000.pt \
  EVALUATION.dataset_stats_path=runs/checkpoint_backups/ttt_dynamic_carrier_slow100_observe_ttt_2cam224_video_ttt_20260608_1016_step002000_20260610_024324/dataset_stats.json \
  +EVALUATION.run_name=ttt_step002000_backup_ood_random20_observe40_60_attach45 \
  +EVALUATION.observe_then_act_chunks=30 \
  +EVALUATION.observe_then_act_interval=10 \
  +EVALUATION.observe_then_act_update_interval=2 \
  +EVALUATION.observe_then_act_frames_min=40 \
  +EVALUATION.observe_then_act_frames_max=60 \
  +EVALUATION.observe_then_act_random_frames=true \
  +EVALUATION.observe_then_act_observe_policy=dummy \
  +EVALUATION.observe_then_act_count_observe_steps=true \
  +EVALUATION.observe_then_act_update_during_observe=true \
  +EVALUATION.observe_then_act_update_during_policy=true \
  +EVALUATION.observe_then_act_action_infer_updates_ttt=false \
  "${COMMON_ARGS[@]}" 2>&1 | tee logs/eval_20260610/ttt_step002000_backup_ood_random20_attach45.log

echo "[$(date -Is)] all OOD random20 attach45 evals finished"
