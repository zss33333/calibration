import torch


def microbatch_projection(trainer, model, inputs):
    """Measure D for one prepared microbatch in an isolated diagnostic run."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1:
            raise ValueError(
                "This first diagnostic step supports one process only."
            )

    # Check before compute_conflict_losses, which cannot process an all-zero
    # completion mask because its log-probability width would be zero.
    if inputs["completion_mask"].sum().item() == 0:
        return {"status": "no_valid_tokens", "active_tokens": 0}

    parameters = [p for p in model.parameters() if p.requires_grad]
    loss_acc, loss_rlcr, active_tokens = trainer.compute_conflict_losses(
        model, inputs
    )

    if active_tokens.item() == 0:
        return {"status": "no_valid_tokens"}

    # Keep the shared forward graph for the second gradient calculation.
    grad_acc = torch.autograd.grad(
        loss_acc,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    acc_cpu = [
        None if grad is None
        else grad.detach().to(device="cpu", dtype=torch.float32)
        for grad in grad_acc
    ]
    del grad_acc

    grad_rlcr = torch.autograd.grad(
        loss_rlcr,
        parameters,
        allow_unused=True,
    )

    dot = 0.0
    acc_norm_sq = 0.0
    rlcr_norm_sq = 0.0

    for acc_grad, rlcr_grad in zip(acc_cpu, grad_rlcr):
        if acc_grad is not None:
            acc_norm_sq += torch.sum(acc_grad.double().square()).item()

        if rlcr_grad is not None:
            rlcr_cpu = rlcr_grad.detach().to(
                device="cpu", dtype=torch.float32
            )
            rlcr_norm_sq += torch.sum(rlcr_cpu.double().square()).item()

            if acc_grad is not None:
                dot += torch.sum(
                    acc_grad.double() * rlcr_cpu.double()
                ).item()

    if acc_norm_sq <= 1e-24:
        return {
            "status": "zero_accuracy_gradient",
            "active_tokens": active_tokens.item(),
            "acc_norm_sq": acc_norm_sq,
        }

    return {
        "status": "ok",
        "active_tokens": active_tokens.item(),
        "dot": dot,
        "acc_norm_sq": acc_norm_sq,
        "rlcr_norm_sq": rlcr_norm_sq,
        "D_bnpo": dot / acc_norm_sq,
        "cosine": dot / (acc_norm_sq * rlcr_norm_sq) ** 0.5
        if rlcr_norm_sq > 0 else float("nan"),
    }
