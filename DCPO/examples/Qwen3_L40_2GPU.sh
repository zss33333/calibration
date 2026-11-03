#!/usr/bin/env bash
# Conservative two-GPU launcher. Defaults to Qwen3-4B, not the 8B paper setup.
set -euo pipefail

dcpo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dcpo_root"
source "$dcpo_root/setup/validation_env.sh"

mode="${1:-smoke}"
if (( $# )); then shift; fi
case "$mode" in
  smoke) batch=2; mini_batch=2; response_length=256; epochs=1; steps=1; save_freq=-1; test_freq=-1 ;;
  train) batch=32; mini_batch=8; response_length=1024; epochs=5; steps=null; save_freq=20; test_freq=20 ;;
  *) echo "Usage: bash examples/Qwen3_L40_2GPU.sh [smoke|train] [--dry-run] [Hydra overrides...]" >&2; exit 2 ;;
esac

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
dcpo_python="${DCPO_PYTHON:-$dcpo_root/.venv/bin/python}"
model_path="${DCPO_MODEL_PATH:-models/Qwen3-4B}"
run_name="$(basename "$model_path")_L40_2GPU_${mode}_$(date +%Y%m%d_%H%M%S)"
run_dir="checkpoints/MATH/$run_name"

export PATH="$(dirname "$dcpo_python"):$PATH"
export PYTHONPATH="$dcpo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1=0
export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1

args=(
  algorithm.adv_estimator=confidence
  reward_model.reward_manager=hybrid
  algorithm.use_kl_in_reward=False
  "data.train_files=['data/deepscaler_uniform_train_with_confidence.parquet']"
  "data.val_files=['data/test_data/aime24_repeated_8_times.parquet']"
  "data.train_batch_size=$batch"
  data.max_prompt_length=1024
  "data.max_response_length=$response_length"
  data.filter_overlong_prompts=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  data.truncation=error
  data.shuffle=False
  "data.save_dir=training_logs_valid/$run_name"
  "actor_rollout_ref.model.path=$model_path"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.algorithm_name=grpo
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.optim.lr=1e-6
  "actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  trainer.critic_warmup=0
  'trainer.logger=["console"]'
  trainer.project_name=verl_dcpo
  "trainer.experiment_name=$run_name"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.resume_mode=disable
  "trainer.default_local_dir=$run_dir"
  "trainer.save_freq=$save_freq"
  "trainer.test_freq=$test_freq"
  "trainer.total_epochs=$epochs"
  "trainer.total_training_steps=$steps"
)

if "$dry_run"; then
  printf '%q ' "$dcpo_python" -m verl.trainer.main_ppo "${args[@]}" "$@"
  printf '\n'
  exit 0
fi

if [[ ! -x "$dcpo_python" ]]; then
  echo "Python not found: $dcpo_python. Create .venv or set DCPO_PYTHON." >&2
  exit 1
fi

"$dcpo_python" - "$model_path" <<'PY'
import json
import sys
from pathlib import Path
import torch
import flash_attn
import vllm
from transformers import AutoTokenizer
import pyarrow.parquet as pq

assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'
assert torch.cuda.device_count() == 2, 'Expose exactly two GPUs with CUDA_VISIBLE_DEVICES'
root = Path(sys.argv[1])
assert (root / 'config.json').is_file(), f'Missing model config: {root}'
index = root / 'model.safetensors.index.json'
shards = (set(json.loads(index.read_text())['weight_map'].values())
          if index.exists() else {'model.safetensors'})
for name in shards:
    assert (root / name).is_file(), f'Missing model weights: {root / name}'
AutoTokenizer.from_pretrained(root, local_files_only=True)
for path in ['data/deepscaler_uniform_train_with_confidence.parquet',
             'data/test_data/aime24_repeated_8_times.parquet']:
    print(f'{path}: {pq.read_metadata(path).num_rows} rows')
print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, '
      f'vllm={vllm.__version__}, flash_attn={flash_attn.__version__}')
for i in range(2):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB')
PY

mkdir -p "$run_dir"
echo "Mode: $mode; model: $model_path; log: $run_dir/training_process.log"
echo "This is a smaller-model hardware configuration, not the original 8B reproduction."
"$dcpo_python" -m verl.trainer.main_ppo "${args[@]}" "$@" 2>&1 | tee "$run_dir/training_process.log"
