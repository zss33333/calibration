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
    loss_no_brier, loss_full, active_tokens = trainer.compute_conflict_losses(
        model, inputs
    )

    if active_tokens.item() == 0:
        return {"status": "no_valid_tokens"}

    # Keep the shared forward graph for the second gradient calculation.
    grad_no_brier = torch.autograd.grad(
        loss_no_brier,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    no_brier_cpu = [
        None if grad is None
        else grad.detach().to(device="cpu", dtype=torch.float32)
        for grad in grad_no_brier
    ]
    del grad_no_brier

    grad_full = torch.autograd.grad(
        loss_full,
        parameters,
        allow_unused=True,
    )

    dot = 0.0
    no_brier_norm_sq = 0.0
    full_norm_sq = 0.0

    for no_brier_grad, full_grad in zip(no_brier_cpu, grad_full):
        if no_brier_grad is not None:
            no_brier_norm_sq += torch.sum(no_brier_grad.double().square()).item()

        if full_grad is not None:
            full_cpu = full_grad.detach().to(
                device="cpu", dtype=torch.float32
            )
            full_norm_sq += torch.sum(full_cpu.double().square()).item()

            if no_brier_grad is not None:
                dot += torch.sum(
                    no_brier_grad.double() * full_cpu.double()
                ).item()

    if no_brier_norm_sq <= 1e-24:
        return {
            "status": "zero_no_brier_gradient",
            "active_tokens": active_tokens.item(),
            "no_brier_norm_sq": no_brier_norm_sq,
        }

    return {
        "status": "ok",
        "active_tokens": active_tokens.item(),
        "dot": dot,
        "no_brier_norm_sq": no_brier_norm_sq,
        "full_norm_sq": full_norm_sq,
        "D_full_on_no_brier": dot / no_brier_norm_sq,
        "D_brier_contribution_on_no_brier": dot / no_brier_norm_sq - 1.0,
        "cosine": dot / (no_brier_norm_sq * full_norm_sq) ** 0.5
        if full_norm_sq > 0 else None,
    }
