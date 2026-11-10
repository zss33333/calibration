#!/usr/bin/env bash
# Run on a server with exactly two assigned L40 GPUs and the DCPO data/model.
# Tests UUID device selection, Ray worker imports, training, validation,
# checkpoint saving, and checkpoint resume. It does not reproduce H200 memory use.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python_bin="${DCPO_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "$root/.venv/bin/python" "$root/.venv-hopper/bin/python"; do
    if [[ -x "$candidate" ]]; then
      python_bin="$candidate"
      break
    fi
  done
fi
[[ -n "$python_bin" && -x "$python_bin" ]] || {
  echo 'Set DCPO_PYTHON to the Python in the L40 server virtual environment.' >&2
  exit 1
}

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 2 && ${#gpu_uuids[@]} -eq 2 ]] || {
  echo 'This check requires a server with exactly two visible GPUs.' >&2
  exit 1
}
for name in "${gpu_names[@]}"; do
  [[ "$name" == *L40* ]] || { echo "Expected L40, found: $name" >&2; exit 1; }
done

# On this separate L40 server only, use the same UUID form as the Hopper PBS job.
# Preserve an existing two-GPU allocation and its device order.
visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible" || "$visible" == '0,1' ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_uuids[0]},${gpu_uuids[1]}"
elif [[ "$visible" == '1,0' ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_uuids[1]},${gpu_uuids[0]}"
elif [[ "$visible" == "${gpu_uuids[0]},${gpu_uuids[1]}" ||
        "$visible" == "${gpu_uuids[1]},${gpu_uuids[0]}" ]]; then
  :
else
  echo "Cannot safely convert CUDA_VISIBLE_DEVICES=$visible to two UUIDs." >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)_$$"
run_name="Qwen3-8B_L40_UUID_CHECK_$run_id"
run_dir="$root/checkpoints/MATH/$run_name"
cache_root="$root/.cache/l40_checks/$run_id"
ray_dir="$(mktemp -d /tmp/dcpo-ray.XXXXXX)"

export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
unset RAY_ADDRESS
export VLLM_USE_V1=0
export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1
# On this two-L40 server, NCCL's default P2P path timed out on the first
# all-reduce. Keep the confirmed workaround local to this L40 check.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export RAY_DEDUP_LOGS=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export DCPO_DISTRIBUTED_TIMEOUT_SECONDS=600
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/hf_datasets"
export XDG_CACHE_HOME="$cache_root/xdg"
export TORCH_HOME="$cache_root/torch"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TMPDIR="$ray_dir/tmp"
export RAY_TMPDIR="$ray_dir"
export DCPO_RAY_TEMP_DIR="$ray_dir"
mkdir -p "$run_dir" "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

echo "Python: $python_bin"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "Run directory: $run_dir"

"$python_bin" scripts/preflight_hopper.py compute --expected-gpus 2

"$python_bin" - <<'PY'
import hashlib
import json
import os
import socket
from pathlib import Path

import flash_attn
import pyarrow.parquet as pq
import ray
import torch
import vllm
from transformers import AutoTokenizer
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.platforms.cuda import device_id_to_physical_device_id

assert torch.cuda.device_count() == 2, torch.cuda.device_count()
model = Path('models/Qwen3-8B')
assert (model / 'config.json').is_file()
index = model / 'model.safetensors.index.json'
shards = (set(json.loads(index.read_text())['weight_map'].values())
          if index.exists() else {'model.safetensors'})
assert shards and all((model / shard).is_file() for shard in shards)
AutoTokenizer.from_pretrained(model, local_files_only=True)
for file in ('data/deepscaler_uniform_train_with_confidence.parquet',
             'data/test_data/aime24_repeated_8_times.parquet'):
    print(file, pq.read_metadata(file).num_rows, 'rows')
cuda_py = Path(vllm.__file__).parent / 'platforms' / 'cuda.py'
print('versions:', 'torch', torch.__version__, 'cuda', torch.version.cuda,
      'vllm', vllm.__version__, 'ray', ray.__version__,
      'flash_attn', flash_attn.__version__)
print('vllm cuda.py SHA256:', hashlib.sha256(cuda_py.read_bytes()).hexdigest())
print('driver mapping:', [device_id_to_physical_device_id(i) for i in range(2)])

ray.init(_temp_dir=os.environ['DCPO_RAY_TEMP_DIR'],
         num_gpus=2, include_dashboard=False)

@ray.remote(num_gpus=1)
def check_worker():
    import os
    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.platforms.cuda import device_id_to_physical_device_id
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible, visible
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    physical = device_id_to_physical_device_id(0)
    return visible, physical, torch.cuda.get_device_name(0)

try:
    workers = [check_worker.remote() for _ in range(2)]
    result = ray.get(workers)
    assert result[0][0] != result[1][0], result
    print('Ray UUID workers:', result)

    @ray.remote(num_gpus=1)
    def check_nccl(rank, port):
        from datetime import timedelta
        import os
        import socket
        import torch
        import torch.distributed as dist

        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        assert visible.startswith('GPU-') and ',' not in visible, visible
        print(f'NCCL preflight rank={rank} host={socket.gethostname()} '
              f'CUDA_VISIBLE_DEVICES={visible} device={torch.cuda.get_device_name(0)} '
              f'P2P_DISABLE={os.environ.get("NCCL_P2P_DISABLE", "unset")} '
              f'SHM_DISABLE={os.environ.get("NCCL_SHM_DISABLE", "unset")}',
              flush=True)
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=120),
        )
        try:
            value = torch.tensor([rank + 1], device='cuda:0')
            print(f'NCCL preflight rank={rank} entering all_reduce', flush=True)
            dist.all_reduce(value)
            print(f'NCCL preflight rank={rank} all_reduce={value.item()}', flush=True)
            assert value.item() == 3, value.item()
            dist.barrier(device_ids=[0])
            return rank, visible, value.item()
        finally:
            dist.destroy_process_group()

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    nccl_result = ray.get([check_nccl.remote(i, port) for i in range(2)],
                          timeout=180)
    assert nccl_result[0][1] != nccl_result[1][1], nccl_result
    print('Ray NCCL all-reduce and barrier:', nccl_result)
finally:
    ray.shutdown()
PY

if [[ "${DCPO_NCCL_ONLY:-0}" == 1 ]]; then
  echo 'NCCL-only preflight passed; DCPO training was not started.'
  exit 0
fi

val_file="$run_dir/aime24_first_8.parquet"
"$python_bin" - "$val_file" <<'PY'
import sys
import pyarrow.parquet as pq
source = 'data/test_data/aime24_repeated_8_times.parquet'
table = pq.read_table(source).slice(0, 8)
assert table.num_rows == 8
pq.write_table(table, sys.argv[1])
print('Validation subset:', sys.argv[1], table.num_rows, 'rows')
PY

args=(
  algorithm.adv_estimator=confidence
  reward_model.reward_manager=hybrid
  algorithm.use_kl_in_reward=False
  "data.train_files=['data/deepscaler_uniform_train_with_confidence.parquet']"
  "data.val_files=['$val_file']"
  data.train_batch_size=2
  data.max_prompt_length=1024
  data.max_response_length=256
  data.filter_overlong_prompts=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  data.truncation=error
  data.shuffle=False
  "data.save_dir=$run_dir/validation_logs"
  actor_rollout_ref.model.path=models/Qwen3-8B
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
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
  "actor_rollout_ref.actor.output_dir_of_record=$run_dir/reward_data"
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
  trainer.project_name=verl_dcpo_l40_check
  "trainer.experiment_name=$run_name"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.val_before_train=False
  "trainer.default_local_dir=$run_dir"
  trainer.save_freq=1
  trainer.test_freq=1
  trainer.total_epochs=1
)

echo 'Running two 8B training steps with validation and checkpoint saving.'
"$python_bin" -m verl.trainer.main_ppo "${args[@]}" \
  trainer.resume_mode=disable trainer.total_training_steps=2 \
  2>&1 | tee "$run_dir/train.log"

[[ -d "$run_dir/global_step_2" ]] || {
  echo "Expected checkpoint missing: $run_dir/global_step_2" >&2
  exit 1
}

if [[ "${DCPO_CHECK_RESUME:-1}" == 1 ]]; then
  echo 'Resuming from the saved checkpoint for one additional step.'
  "$python_bin" -m verl.trainer.main_ppo "${args[@]}" \
    trainer.resume_mode=auto trainer.total_training_steps=3 \
    2>&1 | tee "$run_dir/resume.log"
  [[ -d "$run_dir/global_step_3" ]] || {
    echo "Expected resumed checkpoint missing: $run_dir/global_step_3" >&2
    exit 1
  }
fi

echo "L40 software-path check completed. Logs: $run_dir"
