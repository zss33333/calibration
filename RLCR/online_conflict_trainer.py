"""Opt-in, per-update BNPO gradient projection during RLCR training.

This module is used only by online_conflict_runner.py. It does not replace the
training loss or write to parameter .grad buffers.
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback
from transformers.integrations import is_wandb_available

from GRPO_Trainer import CustomTrainer

if is_wandb_available():
    import wandb


class _WriteAfterUpdate(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._write_completed_update(state.global_step)
        return control


class OnlineConflictTrainer(CustomTrainer):
    """Measure both raw BNPO gradients for every real accumulation window."""

    _reduction_chunk_elements = 2_000_000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.conflict_diagnostic:
            raise ValueError("OnlineConflictTrainer requires conflict_diagnostic=True.")
        if self.args.steps_per_generation != self.args.gradient_accumulation_steps:
            raise ValueError("Online diagnosis requires one generation per update.")
        if self.args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        self._diag_accum_device = os.environ.get("RLCR_DIAG_ACCUM_DEVICE", "cpu").lower()
        if self._diag_accum_device not in {"cpu", "cuda", "hybrid"}:
            raise ValueError("RLCR_DIAG_ACCUM_DEVICE must be cpu, cuda, or hybrid.")
        if os.environ.get("RLCR_DIAG_ENGINE", "legacy") != "reuse":
            print(f"RLCR diagnostic accumulation device: {self._diag_accum_device}", flush=True)
        self._diag_count = 0
        self._diag_tokens = 0
        self._diag_no_brier = None
        self._diag_full = None
        self._diag_pending = None
        self._wandb_conflict_metrics_defined = False
        self.add_callback(_WriteAfterUpdate(self))

    def _base_model(self, model):
        return self.accelerator.unwrap_model(model)

    def _accumulate_pair(self, model, inputs):
        if self._diag_pending is not None:
            raise RuntimeError("Previous diagnostic update was not logged.")
        if inputs["completion_mask"].sum().item() == 0:
            raise RuntimeError("All-zero completion mask in a training microbatch.")

        base_model = self._base_model(model)
        parameters = [p for p in base_model.parameters() if p.requires_grad]
        if self._diag_no_brier is None:
            self._diag_no_brier = [None] * len(parameters)
            self._diag_full = [None] * len(parameters)

        # The repository enables reentrant checkpointing despite its YAML
        # setting. Only sidecar autograd.grad uses non-reentrant checkpointing.
        if self.args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        try:
            loss_no_brier, loss_full, token_count = self.compute_conflict_losses(
                model, inputs
            )
            grad_no_brier = torch.autograd.grad(
                loss_no_brier, parameters, retain_graph=True, allow_unused=True
            )
            for i, grad in enumerate(grad_no_brier):
                if grad is not None:
                    value = grad.detach().to(
                        device="cpu" if self._diag_accum_device == "hybrid" else self._diag_accum_device,
                        dtype=torch.float32,
                    )
                    if self._diag_no_brier[i] is None:
                        self._diag_no_brier[i] = value
                    else:
                        self._diag_no_brier[i].add_(value)
            del grad_no_brier

            grad_full = torch.autograd.grad(
                loss_full, parameters, allow_unused=True
            )
            for i, grad in enumerate(grad_full):
                if grad is not None:
                    value = grad.detach().to(
                        device=self.accelerator.device if self._diag_accum_device == "hybrid" else self._diag_accum_device,
                        dtype=torch.float32,
                    )
                    if self._diag_full[i] is None:
                        self._diag_full[i] = value
                    else:
                        self._diag_full[i].add_(value)
            del grad_full, loss_no_brier, loss_full
        finally:
            if self.args.gradient_checkpointing:
                base_model.gradient_checkpointing_enable()

        self._diag_tokens += int(token_count.item())
        self._diag_count += 1
        if self._diag_count == self.args.gradient_accumulation_steps:
            self._diag_pending = self._finish_window(parameters)
            self._diag_no_brier = None
            self._diag_full = None
            self._diag_tokens = 0
            self._diag_count = 0

    def _finish_window(self, parameters):
        """All-reduce parameter chunks, then project the global vectors."""
        rank = self.accelerator.process_index
        device = self.accelerator.device
        dot = no_brier_norm_sq = full_norm_sq = 0.0
        for i, parameter in enumerate(parameters):
            no_brier_accum = self._diag_no_brier[i]
            full_accum = self._diag_full[i]
            size = parameter.numel()
            for start in range(0, size, self._reduction_chunk_elements):
                stop = min(start + self._reduction_chunk_elements, size)
                length = stop - start
                no_brier = (
                    no_brier_accum.reshape(-1)[start:stop].to(device)
                    if no_brier_accum is not None
                    else torch.zeros(length, device=device, dtype=torch.float32)
                )
                full = (
                    full_accum.reshape(-1)[start:stop].to(device)
                    if full_accum is not None
                    else torch.zeros(length, device=device, dtype=torch.float32)
                )
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(no_brier, op=dist.ReduceOp.SUM)
                    dist.all_reduce(full, op=dist.ReduceOp.SUM)
                if rank == 0:
                    no_brier64 = no_brier.double()
                    full64 = full.double()
                    dot += torch.sum(no_brier64 * full64).item()
                    no_brier_norm_sq += torch.sum(no_brier64.square()).item()
                    full_norm_sq += torch.sum(full64.square()).item()

        tokens = torch.tensor(self._diag_tokens, device=device, dtype=torch.int64)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        if rank != 0:
            return None

        brier_weight = float(self.reward_weights[self.reward_func_names.index("brier_reward")])
        result = {
            "active_tokens": int(tokens.item()),
            "microbatches_per_rank": self.args.gradient_accumulation_steps,
            "microbatch_size": self.args.per_device_train_batch_size,
            "num_processes": self.accelerator.num_processes,
            "group_size": self.num_generations,
            "seed": self.args.seed,
            "model_name_or_path": self.model_name_or_path,
            "diagnostic_accum_device": self._diag_accum_device,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "reward_scope": "weighted_without_brier_vs_weighted_full",
            "projection_scope": "global_update_preclip",
            "brier_weight": brier_weight,
            "dot": dot,
            "no_brier_norm_sq": no_brier_norm_sq,
            "full_norm_sq": full_norm_sq,
            "brier_contribution_dot": dot - no_brier_norm_sq if brier_weight else 0.0,
        }
        if not all(math.isfinite(x) for x in (dot, no_brier_norm_sq, full_norm_sq)):
            result.update({
                "status": "nonfinite_gradient",
                "dot": None,
                "no_brier_norm_sq": None,
                "full_norm_sq": None,
                "brier_contribution_dot": None,
            })
        elif no_brier_norm_sq <= 1e-24:
            result["status"] = "zero_no_brier_gradient"
        else:
            result.update({
                "status": "ok",
                "D_full_on_no_brier": dot / no_brier_norm_sq,
                "D_brier_contribution_on_no_brier": (
                    dot / no_brier_norm_sq - 1.0 if brier_weight else 0.0
                ),
                "cosine": dot / math.sqrt(no_brier_norm_sq * full_norm_sq)
                if full_norm_sq > 0 else None,
            })
        return result

    def _write_completed_update(self, global_step):
        if self._diag_pending is None:
            return
        if self.accelerator.is_main_process:
            record = {"global_step": global_step, **self._diag_pending}
            path = Path(self.args.output_dir) / "conflict_gradients.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and is_wandb_available()
                and wandb.run is not None
            ):
                if not self._wandb_conflict_metrics_defined:
                    wandb.run.define_metric(
                        "conflict/*", step_metric="conflict/global_step"
                    )
                    self._wandb_conflict_metrics_defined = True

                metrics = {
                    "conflict/global_step": global_step,
                    "conflict/active_tokens": record["active_tokens"],
                    "conflict/status_ok": int(record["status"] == "ok"),
                }
                for key in (
                    "dot", "no_brier_norm_sq", "full_norm_sq",
                    "brier_contribution_dot", "D_full_on_no_brier",
                    "D_brier_contribution_on_no_brier", "cosine"
                ):
                    value = record.get(key)
                    if value is not None and math.isfinite(value):
                        metrics[f"conflict/{key}"] = value
                wandb.run.log(metrics)
        self._diag_pending = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            self._accumulate_pair(model, inputs)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
