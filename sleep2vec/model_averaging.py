from __future__ import annotations

import typing as t

import torch.nn as nn

from sleep2vec.config import ModelAveragingConfig
from sleep2vec.pretrain.ema import clone_ema_model, cosine_ema_momentum, ema_update
from sleep2vec.registry import (
    available_model_averagers,
    get_model_averager_builder,
    register_model_averager,
)


class BaseModelAverager:
    """Hooks that manage a tracked copy of the student model."""

    def __init__(
        self,
        *,
        name: str,
        student: nn.Module,
        use_for_eval: bool = True,
        state_prefix: str | None = None,
    ):
        self.name = name
        self.student = student
        self.use_for_eval = use_for_eval
        self.state_prefix = state_prefix or f"{name}_model"
        self.averaged_model: nn.Module | None = None

    @property
    def enabled(self) -> bool:
        return self.averaged_model is not None

    def attach_to_module(self, lightning_module: nn.Module) -> None:
        """Registers the averaged model on the LightningModule for checkpointing."""
        if self.averaged_model is not None:
            setattr(lightning_module, self.state_prefix, self.averaged_model)

    def eval_model(self) -> nn.Module:
        if self.use_for_eval and self.averaged_model is not None:
            return self.averaged_model
        return self.student

    # Lifecycle hooks -----------------------------------------------------
    def on_fit_start(self, trainer) -> None:
        return None

    def on_load_checkpoint(self, checkpoint: dict[str, t.Any]) -> None:
        return None

    def on_train_batch_end(self, *, trainer, global_step: int) -> None:
        return None


@register_model_averager("ema")
class EmaModelAverager(BaseModelAverager):
    """Exponential moving average with cosine momentum schedule."""

    def __init__(self, cfg: ModelAveragingConfig, student: nn.Module):
        params = dict(cfg.params or {})
        enabled = bool(params.get("enabled", False))
        use_for_eval = bool(params.get("use_for_eval", True))
        state_prefix = params.get("state_prefix") or "ema_model"
        super().__init__(name="ema", student=student, use_for_eval=use_for_eval, state_prefix=state_prefix)
        self.base_momentum = float(params.get("base_momentum", 0.996))
        self.final_momentum = float(params.get("final_momentum", 1.0))
        self._total_steps: int | None = None
        self._enabled_flag = enabled

        if enabled:
            self.averaged_model = clone_ema_model(student)

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and self.averaged_model is not None

    def on_fit_start(self, trainer) -> None:
        if not self.enabled:
            return
        if hasattr(trainer, "estimated_stepping_batches"):
            self._total_steps = int(trainer.estimated_stepping_batches)
        else:
            self._total_steps = None

    def on_load_checkpoint(self, checkpoint: dict[str, t.Any]) -> None:
        if not self.enabled:
            return

        state_dict = checkpoint.get("state_dict", {})
        prefix = f"{self.state_prefix}."
        has_ema = any(k.startswith(prefix) for k in state_dict)
        if has_ema:
            return

        if self.averaged_model is None:
            self.averaged_model = clone_ema_model(self.student)

        student_prefix = "model."
        ema_state = {f"{prefix}{k[len(student_prefix):]}": v for k, v in state_dict.items() if k.startswith(student_prefix)}
        if ema_state:
            state_dict.update(ema_state)
            checkpoint["state_dict"] = state_dict

    def on_train_batch_end(self, *, trainer, global_step: int) -> None:
        if not self.enabled:
            return
        momentum = self._momentum_for_step(global_step, trainer)
        ema_update(self.student, self.averaged_model, momentum=momentum)

    def _momentum_for_step(self, global_step: int, trainer) -> float:
        total_steps = self._total_steps
        if total_steps is None:
            total_steps = int(getattr(trainer, "estimated_stepping_batches", 0))
        return cosine_ema_momentum(
            step=global_step,
            total_steps=total_steps,
            base_momentum=self.base_momentum,
            final_momentum=self.final_momentum,
        )


def build_model_averager(cfg: ModelAveragingConfig | None, student: nn.Module) -> BaseModelAverager | None:
    if cfg is None or not cfg.name:
        return None
    builder = get_model_averager_builder(cfg.name)
    averager = builder(cfg, student)
    if averager is None:
        return None
    if hasattr(averager, "enabled") and not averager.enabled:
        return averager
    return averager


__all__ = [
    "BaseModelAverager",
    "EmaModelAverager",
    "available_model_averagers",
    "build_model_averager",
    "register_model_averager",
]
