"""Experimental ZeRO-2 conflict diagnostic using the training gradient."""

import math

import torch
import torch.distributed as dist

from online_conflict_trainer import OnlineConflictTrainer
from pre_step_hook import install_pre_step_hook
from reused_gradient_math import training_to_raw_sum_scale


class ReusedGradientTrainer(OnlineConflictTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("RLCR reused-gradient probe: BF16 model and sharded BF16 no-Brier gradient", flush=True)
        if not self.is_deepspeed_enabled or self.accelerator.num_processes != 4:
            raise ValueError("Reused gradient diagnostic requires four-rank DeepSpeed ZeRO-2.")
        self._no_brier_shards = None
        self._hooked_engine = None

    def _reduce_scatter_gradient(self, grad, parameter):
        """Sum one parameter's auxiliary gradient across ranks into equal shards."""
        world = dist.get_world_size()
        size = parameter.numel()
        shard_size = (size + world - 1) // world
        padded = torch.zeros(shard_size * world, device=parameter.device, dtype=torch.float32)
        if grad is not None:
            padded[:size].copy_(grad.detach().reshape(-1))
        shard = torch.empty(shard_size, device=parameter.device, dtype=torch.float32)
        dist.reduce_scatter_tensor(shard, padded, op=dist.ReduceOp.SUM)
        return shard

    def _accumulate_pair(self, model, inputs):
        if self._diag_pending is not None:
            raise RuntimeError("Previous diagnostic update was not logged.")
        base_model = self._base_model(model)
        parameters = [p for p in base_model.parameters() if p.requires_grad]
        if self._no_brier_shards is None:
            self._no_brier_shards = [None] * len(parameters)

        if self.args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        try:
            loss_no_brier, _, token_count = self.compute_conflict_losses(
                model, inputs, include_full=False
            )
            grad_no_brier = torch.autograd.grad(
                loss_no_brier,
                parameters,
                allow_unused=True,
            )
            for i, (grad, parameter) in enumerate(zip(grad_no_brier, parameters)):
                shard = self._reduce_scatter_gradient(grad, parameter)
                shard_bf16 = shard.to(torch.bfloat16)
                if self._no_brier_shards[i] is None:
                    self._no_brier_shards[i] = shard_bf16
                else:
                    self._no_brier_shards[i].add_(shard_bf16)
            del grad_no_brier
        finally:
            if self.args.gradient_checkpointing:
                base_model.gradient_checkpointing_enable()

        self._diag_tokens += int(token_count.item())
        self._diag_count += 1

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Accelerate's DeepSpeed wrapper calls engine.step() from inside
        # accelerator.backward() on the last microbatch. Intercept step itself
        # so gradients are read after backward and before clipping/zeroing.
        wrapper = self.accelerator.deepspeed_engine_wrapped
        if wrapper is None:
            raise RuntimeError("DeepSpeed engine is unavailable before training_step")
        if self._hooked_engine is not wrapper.engine:
            install_pre_step_hook(wrapper.engine, self._before_engine_step)
            self._hooked_engine = wrapper.engine
        return super().training_step(model, inputs, num_items_in_batch)

    def _before_engine_step(self):
        if self._diag_count == self.args.gradient_accumulation_steps:
            self._diag_pending = self._finish_reused_window()
            self._no_brier_shards = None
            self._diag_tokens = 0
            self._diag_count = 0

    def _finish_reused_window(self):
        from deepspeed.utils import safe_get_full_grad

        world = dist.get_world_size()
        rank = dist.get_rank()
        device = self.accelerator.device
        if self.model_accepts_loss_kwargs or self.compute_loss_func is not None:
            raise RuntimeError("Reused-gradient scaling assumes Trainer divides the scalar loss by GAS")
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        if not engine.enable_backward_allreduce or engine.scale_wrt_gas is False:
            raise RuntimeError("Reused-gradient scaling requires DeepSpeed GAS and DP averaging")
        scale = training_to_raw_sum_scale(self.args.gradient_accumulation_steps, world)
        stats = torch.zeros(3, device=device, dtype=torch.float64)
        parameters = [p for p in self._base_model(self.model).parameters() if p.requires_grad]
        for i, parameter in enumerate(parameters):
            # All ranks must call safe_get_full_grad in the same parameter order.
            full_gradient = safe_get_full_grad(parameter)
            if full_gradient is None:
                raise RuntimeError("ZeRO-2 training gradient is unavailable before engine.step")
            size = parameter.numel()
            shard_size = (size + world - 1) // world
            valid = max(0, min(shard_size, size - rank * shard_size))
            if valid == 0:
                del full_gradient
                continue
            full_flat = full_gradient.reshape(-1)
            for start in range(0, valid, self._reduction_chunk_elements):
                stop = min(start + self._reduction_chunk_elements, valid)
                full = full_flat[rank * shard_size + start:rank * shard_size + stop].float() * scale
                no_brier = self._no_brier_shards[i][start:stop].float()
                full64, no_brier64 = full.double(), no_brier.double()
                stats[0] += torch.dot(full64, no_brier64)
                stats[1] += torch.dot(no_brier64, no_brier64)
                stats[2] += torch.dot(full64, full64)
            del full_gradient
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        tokens = torch.tensor(self._diag_tokens, device=device, dtype=torch.int64)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        if rank != 0:
            return None

        dot, no_brier_sq, full_sq = stats.tolist()
        brier_weight = float(self.reward_weights[self.reward_func_names.index("brier_reward")])
        result = {
            "gradient_engine": "reused_zero2_sharded",
            "diagnostic_accum_dtype": "bfloat16",
            "full_grad_scale_factor": scale,
            "active_tokens": int(tokens.item()),
            "microbatches_per_rank": self.args.gradient_accumulation_steps,
            "microbatch_size": self.args.per_device_train_batch_size,
            "num_processes": world,
            "group_size": self.num_generations,
            "seed": self.args.seed,
            "model_name_or_path": self.model_name_or_path,
            "reward_scope": "weighted_without_brier_vs_weighted_full",
            "projection_scope": "global_update_preclip",
            "brier_weight": brier_weight,
            "dot": dot,
            "no_brier_norm_sq": no_brier_sq,
            "full_norm_sq": full_sq,
            "brier_contribution_dot": dot - no_brier_sq if brier_weight else 0.0,
        }
        if not all(math.isfinite(x) for x in (dot, no_brier_sq, full_sq)):
            result["status"] = "nonfinite_gradient"
        elif no_brier_sq <= 1e-24:
            result["status"] = "zero_no_brier_gradient"
        else:
            result.update({
                "status": "ok",
                "D_full_on_no_brier": dot / no_brier_sq,
                "D_brier_contribution_on_no_brier": dot / no_brier_sq - 1.0 if brier_weight else 0.0,
                "cosine": dot / math.sqrt(no_brier_sq * full_sq) if full_sq > 0 else None,
            })
        return result
