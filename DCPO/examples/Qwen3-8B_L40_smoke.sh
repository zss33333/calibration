#!/usr/bin/env bash
# Full 8B model, two GPUs, reduced precision for a short pipeline check only.
# Do not use this launcher for the H200 reproduction: AdamW state is BF16 here.
set -euo pipefail

dcpo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dcpo_root"
source "$dcpo_root/setup/validation_env.sh"
dcpo_python="${DCPO_PYTHON:-$dcpo_root/.venv/bin/python}"
run_name="Qwen3-8B_L40_smoke_$(date +%Y%m%d_%H%M%S)"
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
  data.train_batch_size=2
  data.max_prompt_length=1024
  data.max_response_length=256
  data.filter_overlong_prompts=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  data.truncation=error
  data.shuffle=False
  "data.save_dir=training_logs_valid/$run_name"
  actor_rollout_ref.model.path=models/Qwen3-8B
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.algorithm_name=grpo
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.use_torch_compile=False
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
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
  actor_rollout_ref.rollout.max_num_batched_tokens=2048
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  trainer.critic_warmup=0
  'trainer.logger=["console"]'
  trainer.project_name=verl_dcpo_smoke
  "trainer.experiment_name=$run_name"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.resume_mode=disable
  "trainer.default_local_dir=$run_dir"
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.total_epochs=1
  trainer.total_training_steps=2
)

if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "$dcpo_python" -m verl.trainer.main_ppo "${args[@]}" "$@"
  printf '\n'
  exit 0
fi

if [[ ! -x "$dcpo_python" ]]; then
  echo "Python not found: $dcpo_python. Create .venv or set DCPO_PYTHON." >&2
  exit 1
fi

mkdir -p "$run_dir"
export NCCL_DEBUG_FILE="$dcpo_root/$run_dir/nccl.%h.%p.log"

"$dcpo_python" - <<'PY'
import json
from pathlib import Path
import torch
import flash_attn
import vllm
from transformers import AutoTokenizer
import pyarrow.parquet as pq

assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'
assert torch.cuda.device_count() == 2, 'Expose exactly two GPUs with CUDA_VISIBLE_DEVICES'
root = Path('models/Qwen3-8B')
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
echo "Full Qwen3-8B pipeline check: two updates, BF16 parameters and AdamW state."
echo "This checks execution, not learning quality or H200 numerical equivalence."
echo "Validation and checkpoint saving are disabled. Log: $run_dir/training_process.log"
"$dcpo_python" -m verl.trainer.main_ppo "${args[@]}" "$@" 2>&1 | tee "$run_dir/training_process.log"
