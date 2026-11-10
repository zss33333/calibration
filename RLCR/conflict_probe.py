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

    # Reject the whole window before any forward or gradient calculation.
    # A zero-token microbatch makes the original loss's log-probability
    # trimming path invalid; dropping it would change the update window.
    token_counts = [
        int(inputs["completion_mask"].sum().item())
        for inputs in microbatches
    ]
    for index, count in enumerate(token_counts):
        if count == 0:
            return {
                "status": "invalid_zero_token_microbatch",
                "zero_token_microbatch_index": index,
                "microbatches": len(microbatches),
            }

    model = trainer.model
    parameters = [p for p in model.parameters() if p.requires_grad]
    no_brier_sum = [None] * len(parameters)
    full_sum = [None] * len(parameters)
    active_tokens = 0

    for index, inputs in enumerate(microbatches):
        loss_no_brier, loss_full, token_count = trainer.compute_conflict_losses(
            model, inputs
        )
        if int(token_count.item()) != token_counts[index]:
            raise ValueError("Completion mask changed during diagnostic loss.")
        active_tokens += token_counts[index]

        grad_no_brier = torch.autograd.grad(
            loss_no_brier, parameters, retain_graph=True, allow_unused=True
        )
        for i, grad in enumerate(grad_no_brier):
            if grad is not None:
                value = grad.detach().to(device="cpu", dtype=torch.float32)
                if no_brier_sum[i] is None:
                    no_brier_sum[i] = value
                else:
                    no_brier_sum[i].add_(value)
        del grad_no_brier

        grad_full = torch.autograd.grad(
            loss_full, parameters, allow_unused=True
        )
        for i, grad in enumerate(grad_full):
            if grad is not None:
                value = grad.detach().to(device="cpu", dtype=torch.float32)
                if full_sum[i] is None:
                    full_sum[i] = value
                else:
                    full_sum[i].add_(value)
        del grad_full, loss_no_brier, loss_full

    dot = 0.0
    no_brier_norm_sq = 0.0
    full_norm_sq = 0.0
    for no_brier_grad, full_grad in zip(no_brier_sum, full_sum):
        if no_brier_grad is not None:
            no_brier_norm_sq += torch.sum(no_brier_grad.double().square()).item()
        if full_grad is not None:
            full_norm_sq += torch.sum(full_grad.double().square()).item()
        if no_brier_grad is not None and full_grad is not None:
            dot += torch.sum(
                no_brier_grad.double() * full_grad.double()
            ).item()

    result = {
        "active_tokens": active_tokens,
        "microbatches": len(microbatches),
        "dot": dot,
        "no_brier_norm_sq": no_brier_norm_sq,
        "full_norm_sq": full_norm_sq,
    }
    if not all(math.isfinite(x) for x in (dot, no_brier_norm_sq, full_norm_sq)):
        result["status"] = "nonfinite_gradient"
        return result
    if no_brier_norm_sq <= 1e-24:
        result["status"] = "zero_no_brier_gradient"
        return result

    result.update({
        "status": "ok",
        "D_full_on_no_brier": dot / no_brier_norm_sq,
        "D_brier_contribution_on_no_brier": dot / no_brier_norm_sq - 1.0,
        "cosine": dot / (no_brier_norm_sq * full_norm_sq) ** 0.5
        if full_norm_sq > 0 else None,
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
        "reward_scope": "weighted_without_brier_vs_weighted_full",
    })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
