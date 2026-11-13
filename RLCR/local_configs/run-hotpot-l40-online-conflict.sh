#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/weiliu/student/zss/calibration
cd "$ROOT/RLCR"
mkdir -p logs

# Separate output directory: rl_runner auto-resumes from an existing checkpoint.
exec env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  RLCR_ONLINE_VLLM_GPU_MEMORY_UTILIZATION=0.50 \
  "$ROOT/.venv-rlcr/bin/accelerate" launch \
  --num_processes 2 \
  --config_file "$ROOT/RLCR/local_configs/accelerate-zero2-cpu-gbs2048.yaml" \
  online_conflict_runner.py \
  --config configs/Qwen-7B/hotpot/RLCR.yaml \
  --gradient_accumulation_steps 128 \
  --push_to_hub false \
  --report_to none \
  --save_steps 5 \
  --output_dir "$ROOT/RLCR/data/RLCR-hotpot-l40-gbs2048-online-conflict" \
  >> "$ROOT/RLCR/logs/rlcr-hotpot-l40-online-conflict.log" 2>&1
