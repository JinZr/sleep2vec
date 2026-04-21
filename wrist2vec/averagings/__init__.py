from __future__ import annotations

from wrist2vec.registry import available_model_averagers, register_model_averager

from .base import BaseModelAverager, build_model_averager
from .ema import EmaModelAverager, clone_ema_model, cosine_ema_momentum, ema_update
from .running_mean import RunningAverageModelAverager

__all__ = [
    "BaseModelAverager",
    "EmaModelAverager",
    "RunningAverageModelAverager",
    "build_model_averager",
    "clone_ema_model",
    "cosine_ema_momentum",
    "ema_update",
    "available_model_averagers",
    "register_model_averager",
]
