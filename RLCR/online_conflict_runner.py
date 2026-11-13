"""Opt-in RLCR training entry point with per-update gradient diagnosis."""

import os

from trl import TrlParser

import rl_runner
from arguments import GRPOScriptArguments, GRPOConfig, ModelConfig
from online_conflict_trainer import OnlineConflictTrainer
from reused_gradient_trainer import ReusedGradientTrainer


def main():
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.conflict_diagnostic = True
    utilization = os.environ.get("RLCR_ONLINE_VLLM_GPU_MEMORY_UTILIZATION")
    if utilization is not None:
        value = float(utilization)
        if not 0.0 < value < 1.0:
            raise ValueError("RLCR_ONLINE_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1).")
        training_args.vllm_gpu_memory_utilization = value
    print(
        "Online diagnostic vLLM gpu_memory_utilization="
        f"{training_args.vllm_gpu_memory_utilization}",
        flush=True,
    )
    engine = os.environ.get("RLCR_DIAG_ENGINE", "legacy")
    if engine not in {"legacy", "reuse"}:
        raise ValueError("RLCR_DIAG_ENGINE must be legacy or reuse.")
    if engine == "reuse" and not (
        training_args.bf16 and not training_args.fp16 and model_args.torch_dtype == "bfloat16"
    ):
        raise ValueError("Reused-gradient probe requires BF16 model and BF16 training.")
    rl_runner.CustomTrainer = ReusedGradientTrainer if engine == "reuse" else OnlineConflictTrainer
    rl_runner.main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
