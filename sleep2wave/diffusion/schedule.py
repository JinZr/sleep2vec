from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor


def _validate_num_steps(num_steps: int) -> int:
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    return num_steps


def _validate_betas(betas: torch.Tensor) -> torch.Tensor:
    if betas.dim() != 1:
        raise ValueError("betas must be a 1D tensor.")
    if not torch.isfinite(betas).all():
        raise ValueError("betas must be finite.")
    if not ((betas > 0) & (betas < 1)).all():
        raise ValueError("betas must satisfy 0 < beta < 1.")
    return betas


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    num_steps = _validate_num_steps(num_steps)
    if not isinstance(s, (int, float)) or isinstance(s, bool) or s < 0:
        raise ValueError("s must be a non-negative number.")

    steps = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    alpha_bars = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, min=1e-8, max=0.999).to(dtype=torch.float32)
    return _validate_betas(betas)


def build_diffusion_schedule(num_steps: int, beta_schedule: str = "cosine") -> DiffusionSchedule:
    if beta_schedule != "cosine":
        raise ValueError("beta_schedule must be 'cosine'.")

    betas = cosine_beta_schedule(num_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    if not torch.isfinite(alpha_bars).all():
        raise ValueError("alpha_bars must be finite.")
    if not torch.all(alpha_bars[1:] <= alpha_bars[:-1]):
        raise ValueError("alpha_bars must be monotonically non-increasing.")

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


__all__ = [
    "DiffusionSchedule",
    "build_diffusion_schedule",
    "cosine_beta_schedule",
]
