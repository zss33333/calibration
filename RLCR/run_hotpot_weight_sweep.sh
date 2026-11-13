#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
(cd "$ROOT/official_upstream" && sha256sum -c SHA256SUMS)

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT/official_upstream/deepspeed.yaml}"
RUN_ROOT="${RUN_ROOT:-$ROOT/data/hotpot-weight-sweep}"
TEMPLATE="$ROOT/official_upstream/configs/Qwen-7B/hotpot/RLCR.yaml"
read -r -a labels <<< "${RUN_LABELS:-0.00 0.25 0.50 0.75 1.00}"

if ((${#labels[@]} == 0)); then
  echo "RUN_LABELS must contain at least one weight label" >&2
  exit 2
fi

unset RLCR_SKIP_FINAL_MODEL_SAVE
mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/logs"

for label in "${labels[@]}"; do
  case "$label" in
    0.00) accuracy_weight=0.00; brier_weight=1.00 ;;
    0.25) accuracy_weight=0.25; brier_weight=0.75 ;;
    0.50) accuracy_weight=0.50; brier_weight=0.50 ;;
    0.75) accuracy_weight=0.75; brier_weight=0.25 ;;
    1.00) accuracy_weight=1.00; brier_weight=0.00 ;;
    *) echo "Unsupported RUN_LABELS value: $label" >&2; exit 2 ;;
  esac

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
], "The official reward order changed; check the weights before running."
assert config["reward_weights"] == [0.5, 0.5, 0.5, 0.000001, 0.000001], (
    "The official Hotpot RLCR weights changed; check the template before running."
)
assert not config.get("conflict_diagnostic", False), "The template enables gradient diagnostics."

config["reward_weights"][1:3] = [float(accuracy), float(brier)]
config["run_name"] = f"RLCR-hotpot-lambda-{label}"
config["output_dir"] = output_dir
config["push_to_hub"] = False

with open(config_path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Prepared $config_path (accuracy=$accuracy_weight, brier=$brier_weight)"
    continue
  fi

  "$ACCELERATE_BIN" launch --num_processes "$NUM_PROCESSES" \
    --config_file "$ACCELERATE_CONFIG" official_upstream/rl_runner.py \
    --config "$config_path" 2>&1 | tee "$RUN_ROOT/logs/$run_id.log"
done
