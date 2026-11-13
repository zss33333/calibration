#!/usr/bin/env bash
# Run the paper-format DCPO math evaluation on the separate L40 server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${DCPO_PYTHON:-$ROOT/.venv/bin/python}"
ACTOR_DIR="${DCPO_ACTOR_DIR:?Set DCPO_ACTOR_DIR to the copied global_step_120/actor directory}"
STEP="${DCPO_EVAL_STEP:-120}"
PROMPT_FORMAT="${DCPO_PROMPT_FORMAT:-paper}"
PORT="${DCPO_EVAL_PORT:-8000}"
[[ -x "$PYTHON" && -d "$ACTOR_DIR" ]] || { echo 'Missing Python or actor directory' >&2; exit 1; }
[[ "$STEP" =~ ^[1-9][0-9]*$ && "$PORT" =~ ^[0-9]+$ ]] || { echo 'Invalid step or port' >&2; exit 1; }
[[ "$PROMPT_FORMAT" == paper || "$PROMPT_FORMAT" == trained ]] || { echo 'Invalid prompt format' >&2; exit 1; }
[[ "$(basename "$ACTOR_DIR")" == actor && "$(basename "$(dirname "$ACTOR_DIR")")" == "global_step_$STEP" ]] || {
  echo "Expected .../global_step_$STEP/actor" >&2; exit 1;
}
BASE_MODEL="${DCPO_BASE_MODEL:-$ROOT/models/Qwen3-8B}"
DATA_ROOT="${DCPO_EVAL_DATA_ROOT:-$ROOT/data}"
RUN_DIR="$(dirname "$(dirname "$ACTOR_DIR")")"
EVAL_DIR="$RUN_DIR/evaluation/step_$STEP"
MERGED_DIR="$EVAL_DIR/hf_model"
RESULT_DIR="$EVAL_DIR/math_table_$PROMPT_FORMAT"
MODEL_NAME="dcpo-step$STEP"
[[ -f "$BASE_MODEL/config.json" ]] || { echo "Missing base model: $BASE_MODEL" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$EVAL_DIR"

"$PYTHON" -c 'import pandas, requests, sklearn, vllm; from opencompass.evaluator import MATHVerifyEvaluator'
CHECK_OUTPUT="$("$PYTHON" scripts/eval_math_table.py --data-root "$DATA_ROOT" --check-data-only)"
printf '%s\n' "$CHECK_OUTPUT"
grep -Fxq 'EVAL_DATA_CHECK_PASSED' <<< "$CHECK_OUTPUT" || {
  echo 'Evaluation script did not run its data check; sync the current scripts/eval_math_table.py' >&2
  exit 2
}
shopt -s nullglob
SHARDS=("$ACTOR_DIR"/model_world_size_*_rank_*.pt)
((${#SHARDS[@]} > 0)) || { echo "No actor model shards in $ACTOR_DIR" >&2; exit 1; }

if [[ ! -f "$MERGED_DIR/.merge_complete" ]]; then
  mkdir -p "$MERGED_DIR"
  "$PYTHON" scripts/model_merger.py --backend fsdp \
    --hf_model_path "$BASE_MODEL" --local_dir "$ACTOR_DIR" --target_dir "$MERGED_DIR"
fi
"$PYTHON" - "$MERGED_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
assert (root / 'config.json').is_file(), 'Merged config.json is missing'
index = root / 'model.safetensors.index.json'
names = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
assert names and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names), 'Merged weights are incomplete'
print(f'Merged model: {len(names)} weight file(s)', flush=True)
PY
touch "$MERGED_DIR/.merge_complete"

"$PYTHON" - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0:
        raise SystemExit('Evaluation port is occupied; set DCPO_EVAL_PORT to a free port')
PY

"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MERGED_DIR" --tokenizer "$BASE_MODEL" --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.80 --max-num-seqs 16 --port "$PORT" \
  > "$EVAL_DIR/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

"$PYTHON" - "$PORT" "$MODEL_NAME" "$SERVER_PID" <<'PY'
import json
import os
import sys
import time
import urllib.request
port, expected, pid = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for _ in range(120):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
            models = json.load(response)['data']
        if expected in {model['id'] for model in models}:
            print(f'vLLM ready: {expected}', flush=True)
            break
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit('vLLM exited; inspect evaluation/step_*/vllm.log')
    time.sleep(5)
else:
    raise SystemExit('vLLM did not become ready in 10 minutes; inspect vllm.log')
PY

"$PYTHON" scripts/eval_math_table.py --data-root "$DATA_ROOT" \
  --output-dir "$RESULT_DIR" --model "$MODEL_NAME" --prompt-format "$PROMPT_FORMAT" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --repeats 4 --bins 20 --invalid-policy author --workers 16
test -s "$RESULT_DIR/per_run.csv"
test -s "$RESULT_DIR/summary.csv"
test -s "$RESULT_DIR/summary.json"
for dataset in MATH-500 AIME24 AIME25 AMC23 AMC24; do
  for repeat in 1 2 3 4; do
    test -s "$RESULT_DIR/${dataset}_run${repeat}.jsonl" || {
      echo "Missing result file: $RESULT_DIR/${dataset}_run${repeat}.jsonl" >&2
      exit 2
    }
  done
done
echo "Results: $RESULT_DIR"
