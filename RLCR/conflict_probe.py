import copy
import json
import math
import os
from functools import partial

import torch
from datasets import load_dataset
from trl import TrlParser

from arguments import GRPOScriptArguments, GRPOConfig, ModelConfig
from dataset_processing import process_dataset
from GRPO_Trainer import CustomTrainer
from reward_fns import (
    format_reward,
    accuracy_reward,
    brier_reward,
    mean_confidence_reward,
    confidence_one_or_zero,
)
from rl_runner import model_init
from trainer_utils import shuffle_tensor_dict, split_tensor_dict


def accumulated_projection(trainer, microbatches):
    """Project gradients after accumulating every BNPO microbatch."""
    if trainer.accelerator.num_processes != 1:
        raise ValueError("This probe requires one training process.")

    model = trainer.model
    parameters = [p for p in model.parameters() if p.requires_grad]
    acc_sum = [None] * len(parameters)
    rlcr_sum = [None] * len(parameters)
    active_tokens = 0

    for inputs in microbatches:
        loss_acc, loss_rlcr, token_count = trainer.compute_conflict_losses(
            model, inputs
        )
        if token_count.item() == 0:
            raise ValueError("A microbatch has no active completion tokens.")
        active_tokens += int(token_count.item())

        grad_acc = torch.autograd.grad(
            loss_acc, parameters, retain_graph=True, allow_unused=True
        )
        for i, grad in enumerate(grad_acc):
            if grad is not None:
                value = grad.detach().to(device="cpu", dtype=torch.float32)
                if acc_sum[i] is None:
                    acc_sum[i] = value
                else:
                    acc_sum[i].add_(value)
        del grad_acc

        grad_rlcr = torch.autograd.grad(
            loss_rlcr, parameters, allow_unused=True
        )
        for i, grad in enumerate(grad_rlcr):
            if grad is not None:
                value = grad.detach().to(device="cpu", dtype=torch.float32)
                if rlcr_sum[i] is None:
                    rlcr_sum[i] = value
                else:
                    rlcr_sum[i].add_(value)
        del grad_rlcr, loss_acc, loss_rlcr

    dot = 0.0
    acc_norm_sq = 0.0
    rlcr_norm_sq = 0.0
    for acc_grad, rlcr_grad in zip(acc_sum, rlcr_sum):
        if acc_grad is not None:
            acc_norm_sq += torch.sum(acc_grad.double().square()).item()
        if rlcr_grad is not None:
            rlcr_norm_sq += torch.sum(rlcr_grad.double().square()).item()
        if acc_grad is not None and rlcr_grad is not None:
            dot += torch.sum(
                acc_grad.double() * rlcr_grad.double()
            ).item()

    result = {
        "active_tokens": active_tokens,
        "microbatches": len(microbatches),
        "dot": dot,
        "acc_norm_sq": acc_norm_sq,
        "rlcr_norm_sq": rlcr_norm_sq,
    }
    if not all(math.isfinite(x) for x in (dot, acc_norm_sq, rlcr_norm_sq)):
        result["status"] = "nonfinite_gradient"
        return result
    if acc_norm_sq <= 1e-24:
        result["status"] = "zero_accuracy_gradient"
        return result

    result.update({
        "status": "ok",
        "D_bnpo": dot / acc_norm_sq,
        "cosine": dot / (acc_norm_sq * rlcr_norm_sq) ** 0.5
        if rlcr_norm_sq > 0 else None,
    })
    return result


def main():
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # This process diagnoses a checkpoint; it never runs training.
    training_args.conflict_diagnostic = True
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"
    training_args.push_to_hub = False
    training_args.report_to = []
    training_args.output_dir = "data/conflict-probe"

    # Match one original single-process gradient accumulation window.
    training_args.steps_per_generation = training_args.gradient_accumulation_steps
    training_args.generation_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.steps_per_generation
    )
    if training_args.generation_batch_size % training_args.num_generations:
        raise ValueError("Generation batch must contain complete G groups.")
    num_groups = (
        training_args.generation_batch_size // training_args.num_generations
    )
    training_args.model_init_kwargs = model_init(
        model_args, training_args
    )

    registry = {
        "format": partial(
            format_reward, format_pattern=script_args.format_pattern
        ),
        "accuracy": partial(
            accuracy_reward, format_pattern=script_args.format_pattern
        ),
        "brier": partial(
            brier_reward, format_pattern=script_args.format_pattern
        ),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    reward_funcs = [
        registry[name] for name in script_args.reward_funcs
    ]

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
    )
    dataset = process_dataset(dataset, script_args)
    probe_split = os.environ.get("RLCR_PROBE_SPLIT", script_args.dataset_train_split)
    probe_dataset = dataset[probe_split]
    start_index = int(os.environ.get("RLCR_PROBE_START", "0"))
    if start_index < 0 or start_index + num_groups > len(probe_dataset):
        raise ValueError("Not enough consecutive prompts for this probe window.")
    prompt_indices = list(range(start_index, start_index + num_groups))

    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
    )
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Run this probe with one training process.")
    # CustomTrainer's original no-argument checkpoint setup ignores the YAML
    # use_reentrant=False setting. autograd.grad requires non-reentrant mode.
    if training_args.gradient_checkpointing:
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    trainer.model.train()

    generation_batch = [
        copy.deepcopy(probe_dataset[index])
        for index in prompt_indices
        for _ in range(training_args.num_generations)
    ]
    prepared = trainer._generate_and_score_completions(generation_batch)
    # Preserve alignment of tokens, masks, old logps, and both advantages.
    prepared = shuffle_tensor_dict(prepared)
    microbatches = split_tensor_dict(
        prepared, training_args.steps_per_generation
    )
    result = accumulated_projection(trainer, microbatches)
    result.update({
        "dataset_split": probe_split,
        "prompt_indices": prompt_indices,
        "group_size": training_args.num_generations,
        "num_groups": num_groups,
        "microbatch_size": training_args.per_device_train_batch_size,
        "projection_scope": "full_accumulation_window",
        "reward_scope": "configured_weighted_rewards",
    })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
