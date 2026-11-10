"""Opt-in RLCR training entry point with per-update gradient diagnosis."""

import os

from trl import TrlParser

import rl_runner
from arguments import GRPOScriptArguments, GRPOConfig, ModelConfig
from online_conflict_trainer import OnlineConflictTrainer


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
    rl_runner.CustomTrainer = OnlineConflictTrainer
    rl_runner.main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
