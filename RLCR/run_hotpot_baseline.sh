#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-RLCR-hotpot-weight-sweep}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT/deepspeed.yaml}"
RUN_ROOT="${RUN_ROOT:-$ROOT/data/hotpot-weight-sweep}"
TEMPLATE="$ROOT/configs/Qwen-7B/hotpot/RLCR.yaml"

run_id="lambda-0.50"
config_path="$RUN_ROOT/configs/$run_id.yaml"
output_dir="$RUN_ROOT/$run_id"
mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/logs"

"$PYTHON_BIN" - "$TEMPLATE" "$config_path" "$output_dir" <<'PY'
import sys
import yaml

template_path, config_path, output_dir = sys.argv[1:]
with open(template_path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
assert config["reward_funcs"] == [
    "format", "accuracy", "brier", "mean_confidence", "confidence_one_or_zero"
]
assert len(config["reward_weights"]) == 5
config["reward_weights"][1] = 0.5
config["reward_weights"][2] = 0.5
config["run_name"] = "RLCR-hotpot-lambda-0.50"
config["output_dir"] = output_dir
config["report_to"] = ["wandb"]
config["push_to_hub"] = False
with open(config_path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

"$ACCELERATE_BIN" launch --num_processes "$NUM_PROCESSES" \
  --config_file "$ACCELERATE_CONFIG" online_conflict_runner.py \
  --config "$config_path" 2>&1 | tee "$RUN_ROOT/logs/$run_id.log"
