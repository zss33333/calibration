"""Scalar scaling used to compare ZeRO-2 gradients with raw sidecar gradients."""


def training_to_raw_sum_scale(gradient_accumulation_steps, world_size):
    """Undo Trainer, DeepSpeed, and data-parallel gradient averaging.

    Trainer and DeepSpeed each divide this trainer's scalar loss by GAS.
    ZeRO-2 also averages reduced gradients over the data-parallel group.
    """
    return gradient_accumulation_steps**2 * world_size
