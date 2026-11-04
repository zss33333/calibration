import copy
import json
from functools import partial

from datasets import load_dataset
from trl import TrlParser

from arguments import GRPOScriptArguments, GRPOConfig, ModelConfig
from conflict_gradients import microbatch_projection
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

    # One Hotpot prompt × 32 completions = one complete advantage group.
    # Four prepared microbatches of eight match the original local batch size.
    training_args.generation_batch_size = 32
    training_args.steps_per_generation = 4
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
    probe_dataset = dataset[script_args.dataset_test_split]

    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
    )
    trainer.model.train()

    max_attempts = min(128, len(probe_dataset))
    for index in range(max_attempts):
        example = probe_dataset[index]
        generation_batch = [
            copy.deepcopy(example)
            for _ in range(training_args.num_generations)
        ]

        prepared = trainer._generate_and_score_completions(
            generation_batch
        )
        if not prepared["diagnostic_valid"].all().item():
            continue

        # Use the repository's joint shuffle/split, preserving alignment
        # among token IDs, masks, old logps, and both advantages.
        prepared = shuffle_tensor_dict(prepared)
        microbatches = split_tensor_dict(
            prepared, training_args.steps_per_generation
        )

        result = microbatch_projection(
            trainer, trainer.model, microbatches[0]
        )
        result["prompt_index"] = index
        result["attempted_groups"] = index + 1
        print(json.dumps(result, ensure_ascii=False))
        return

    print(json.dumps({
        "status": "no_complete_valid_group",
        "attempted_groups": max_attempts,
    }))


if __name__ == "__main__":
    main()