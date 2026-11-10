"""Cheap Hopper checks. Run with the Hopper venv inside its Singularity image.

The login check uses no GPU. The compute check runs inside the allocated PBS job
before DCPO starts and compiles one small Triton kernel in that job's cache.
"""

import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]


def common():
    assert sys.prefix != sys.base_prefix, f"Not in a virtual environment: {sys.executable}"
    assert str(ROOT) in sys.executable, f"Wrong Python: {sys.executable}"
    for name in ("torch", "triton", "vllm", "ray", "flash-attn", "transformers"):
        print(f"{name}={metadata.version(name)}", flush=True)
    model = ROOT / "models/Qwen3-8B"
    assert (model / "config.json").is_file(), model
    assert any(model.glob("*.safetensors")), "Model weights missing"
    for relative in (
        "data/deepscaler_uniform_train_with_confidence.parquet",
        "data/test_data/aime24_repeated_8_times.parquet",
    ):
        assert (ROOT / relative).is_file(), relative
    cuda_file = Path(sysconfig.get_paths()["purelib"]) / "vllm/platforms/cuda.py"
    source = cuda_file.read_text(encoding="utf-8")
    if "return int(physical_device_id)" in source and "nvmlDeviceGetHandleByUUID" not in source:
        raise RuntimeError(
            f"vLLM UUID mapping appears unpatched: {cuda_file}; "
            "run scripts/fix_vllm_uuid.py using this same Python environment"
        )
    ray_dir = os.environ.get("DCPO_RAY_TEMP_DIR")
    if ray_dir:
        example_socket = Path(ray_dir) / "session_2026-11-05_00-00-00_123456_12345678" / "sockets/plasma_store"
        assert len(os.fsencode(example_socket)) <= 107, (
            f"Ray AF_UNIX socket path may be too long: {example_socket}"
        )
    print("Common preflight passed", flush=True)


def login():
    """Inspect inputs and checkpoint state without importing CUDA or starting Ray."""
    import json
    import pyarrow.parquet as pq

    model = ROOT / "models/Qwen3-8B"
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    print("Model type:", config.get("model_type"), flush=True)
    index = model / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = [model / name for name in sorted(set(weight_map.values()))]
    else:
        shards = sorted(model.glob("*.safetensors"))
    missing = [str(path) for path in shards if not path.is_file() or path.stat().st_size == 0]
    assert shards and not missing, f"Missing or empty model shards: {missing}"
    print("Model shards:", len(shards), flush=True)

    required_columns = {"prompt", "data_source", "reward_model"}
    for relative in (
        "data/deepscaler_uniform_train_with_confidence.parquet",
        "data/test_data/aime24_repeated_8_times.parquet",
    ):
        path = ROOT / relative
        metadata = pq.read_metadata(path)
        columns = set(metadata.schema.to_arrow_schema().names)
        assert metadata.num_rows > 0, f"Empty dataset: {path}"
        assert required_columns <= columns, (
            f"{path} is missing {sorted(required_columns - columns)}"
        )
        print(relative, "rows:", metadata.num_rows, "columns:", sorted(columns), flush=True)

    checkpoint = ROOT / "checkpoints/MATH/Qwen3-8B_DCPO"
    latest = checkpoint / "latest_checkpointed_iteration.txt"
    if latest.exists():
        step = latest.read_text(encoding="utf-8").strip()
        actor = checkpoint / f"global_step_{step}" / "actor"
        print("Existing auto-resume checkpoint:", step, actor, flush=True)
        print("Actor shards:", [p.name for p in sorted(actor.glob("model_world_size_*"))], flush=True)
        print("Confirm this checkpoint matches the intended GPU world size before training.", flush=True)
    else:
        print("No auto-resume checkpoint tracker in:", checkpoint, flush=True)
    print("Login preflight passed (no GPU used)", flush=True)


def compute(expected_gpus):
    import ctypes
    import torch
    import triton
    global tl
    import triton.language as tl
    import vllm
    import flash_attn
    from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: F401
    from vllm.platforms.cuda import device_id_to_physical_device_id

    assert torch.cuda.device_count() == expected_gpus, torch.cuda.device_count()
    assert os.environ.get("VLLM_USE_V1") == "0", "Set VLLM_USE_V1=0 before launching Python"
    for env_name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "DCPO_RAY_TEMP_DIR"):
        value = os.environ.get(env_name)
        assert value and Path(value).is_dir(), f"Missing per-job {env_name}: {value}"
    print("Python:", sys.executable, "torch CUDA:", torch.version.cuda,
          "vLLM:", vllm.__version__, "flash-attn:", flash_attn.__version__, flush=True)
    print("Visible GPUs:", torch.cuda.device_count(),
          "vLLM physical IDs:", [device_id_to_physical_device_id(i) for i in range(expected_gpus)],
          flush=True)

    @triton.jit
    def add_kernel(a, b, out, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        tl.store(out + offsets, tl.load(a + offsets, mask, 0) +
                 tl.load(b + offsets, mask, 0), mask)

    a = torch.arange(128, device="cuda:0", dtype=torch.float32)
    b = torch.ones_like(a)
    out = torch.empty_like(a)
    add_kernel[(1,)](a, b, out, 128, 128)
    torch.cuda.synchronize()
    assert torch.equal(out, a + b), "Triton kernel produced wrong result"
    cuda_utils = list(Path(os.environ["TRITON_CACHE_DIR"]).rglob("cuda_utils.so"))
    for library in cuda_utils:
        ctypes.CDLL(str(library))
    print("cuda_utils.so in this job's Triton cache:",
          [str(path) for path in cuda_utils], flush=True)
    print("Compute preflight passed: fresh Triton compile and CUDA execution", flush=True)


def ray_check():
    import ray

    ray_dir = os.environ.get("DCPO_RAY_TEMP_DIR")
    assert ray_dir and Path(ray_dir).is_dir(), f"Missing DCPO_RAY_TEMP_DIR: {ray_dir}"
    ray.init(_temp_dir=ray_dir, num_gpus=1, include_dashboard=False)

    @ray.remote(num_gpus=1)
    def worker():
        import os
        import sys
        import torch
        import ray
        from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: F401
        from vllm.platforms.cuda import device_id_to_physical_device_id
        from verl.workers.fsdp_workers import ActorRolloutRefWorker  # noqa: F401

        assert torch.cuda.device_count() == 1, torch.cuda.device_count()
        value = torch.ones(1, device="cuda") + 1
        assert value.item() == 2
        return {
            "python": sys.executable,
            "ray_gpu_ids": ray.get_gpu_ids(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vllm_physical_id": device_id_to_physical_device_id(0),
            "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        }

    try:
        result = ray.get(worker.remote(), timeout=120)
        print("Ray GPU worker:", result, flush=True)
        if not str(result["CUDA_VISIBLE_DEVICES"]).startswith("GPU-"):
            print("Note: this allocation did not exercise the GPU UUID path", flush=True)
    finally:
        ray.shutdown()
    print("Ray preflight passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("login", "compute", "ray"))
    parser.add_argument("--expected-gpus", type=int, default=8)
    args = parser.parse_args()
    common()
    if args.stage == "login":
        login()
    elif args.stage == "compute":
        compute(args.expected_gpus)
    elif args.stage == "ray":
        ray_check()
