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
mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/logs"

for spec in "0.00:0.00:1.00:rl_runner.py" \
            "0.25:0.25:0.75:online_conflict_runner.py" \
            "0.75:0.75:0.25:online_conflict_runner.py" \
            "1.00:1.00:0.00:rl_runner.py"; do
  IFS=: read -r label accuracy_weight brier_weight runner <<< "$spec"
  run_id="lambda-$label"
  config_path="$RUN_ROOT/configs/$run_id.yaml"
  output_dir="$RUN_ROOT/$run_id"

  "$PYTHON_BIN" - "$TEMPLATE" "$config_path" "$output_dir" \
    "$accuracy_weight" "$brier_weight" "$label" <<'PY'
import sys
import yaml

template_path, config_path, output_dir, accuracy, brier, label = sys.argv[1:]
with open(template_path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
assert config["reward_funcs"] == [
    "format", "accuracy", "brier", "mean_confidence", "confidence_one_or_zero"
]
assert len(config["reward_weights"]) == 5
config["reward_weights"][1] = float(accuracy)
config["reward_weights"][2] = float(brier)
config["run_name"] = f"RLCR-hotpot-lambda-{label}"
config["output_dir"] = output_dir
config["report_to"] = ["wandb"]
config["push_to_hub"] = False
with open(config_path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

  "$ACCELERATE_BIN" launch --num_processes "$NUM_PROCESSES" \
    --config_file "$ACCELERATE_CONFIG" "$runner" \
    --config "$config_path" 2>&1 | tee "$RUN_ROOT/logs/$run_id.log"
done
