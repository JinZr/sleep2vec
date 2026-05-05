from __future__ import annotations

import math
import typing as t

import torch


def _generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _randint(high: int, *, seed: int | None) -> int:
    if high <= 0:
        return 0
    value = torch.randint(high, (1,), generator=_generator(seed))
    return int(value.item())


def _full_mask(signal: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(signal, dtype=torch.bool)


def _empty_mask(signal: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(signal, dtype=torch.bool)


def _window_mask(signal: torch.Tensor, window_frames: int, *, seed: int | None) -> torch.Tensor:
    if signal.dim() < 1:
        raise ValueError("Signal tensor must have at least one dimension.")
    frames = signal.shape[-1]
    if window_frames <= 0:
        raise ValueError("window_frames must be positive.")
    if window_frames > frames:
        raise ValueError(f"window_frames={window_frames} exceeds signal length {frames}.")

    left = _randint(frames - window_frames + 1, seed=seed)
    mask = _empty_mask(signal)
    mask[..., left : left + window_frames] = True
    return mask


def contiguous_window_mask(
    signal: torch.Tensor,
    *,
    window_frames: int,
    fill_value: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _window_mask(signal, window_frames, seed=seed)
    corrupted = signal.clone()
    corrupted[mask] = fill_value
    return corrupted, mask


def flatline_dropout(
    signal: torch.Tensor,
    *,
    window_frames: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _window_mask(signal, window_frames, seed=seed)
    corrupted = signal.clone()
    frames = signal.shape[-1]
    left = int(mask.reshape(-1, frames).any(dim=0).nonzero()[0].item())
    fill = signal[..., left : left + 1]
    corrupted[mask] = fill.expand_as(signal)[mask]
    return corrupted, mask


def gaussian_noise(
    signal: torch.Tensor,
    *,
    std: float = 0.1,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if std < 0:
        raise ValueError("std must be non-negative.")
    noise = torch.randn(signal.shape, generator=_generator(seed), dtype=signal.dtype) * std
    return signal + noise, _full_mask(signal)


def baseline_drift(
    signal: torch.Tensor,
    *,
    amplitude: float = 0.1,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    drift = torch.linspace(0.0, float(amplitude), signal.shape[-1], dtype=signal.dtype)
    view_shape = [1] * signal.dim()
    view_shape[-1] = signal.shape[-1]
    return signal + drift.view(view_shape), _full_mask(signal)


def line_noise(
    signal: torch.Tensor,
    *,
    amplitude: float = 0.1,
    cycles: float = 10.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_axis = torch.linspace(0.0, 2.0 * math.pi * float(cycles), signal.shape[-1], dtype=signal.dtype)
    noise = torch.sin(t_axis) * float(amplitude)
    view_shape = [1] * signal.dim()
    view_shape[-1] = signal.shape[-1]
    return signal + noise.view(view_shape), _full_mask(signal)


def saturation_clipping(
    signal: torch.Tensor,
    *,
    min_value: float,
    max_value: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value.")
    corrupted = signal.clamp(min_value, max_value)
    return corrupted, corrupted.ne(signal)


def spike_artifact(
    signal: torch.Tensor,
    *,
    num_spikes: int = 1,
    magnitude: float = 5.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_spikes <= 0:
        raise ValueError("num_spikes must be positive.")
    corrupted = signal.clone()
    mask = _empty_mask(signal)
    frames = signal.shape[-1]
    generator = _generator(seed)
    indices = torch.randperm(frames, generator=generator)[: min(num_spikes, frames)]
    corrupted[..., indices] += float(magnitude)
    mask[..., indices] = True
    return corrupted, mask


def amplitude_attenuation(
    signal: torch.Tensor,
    *,
    factor: float = 0.5,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return signal * float(factor), _full_mask(signal)


def phase_inversion(signal: torch.Tensor, *, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    return -signal, _full_mask(signal)


def spo2_plateau_dropout(
    signal: torch.Tensor,
    *,
    window_frames: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flatline_dropout(signal, window_frames=window_frames, seed=seed)


def rpeak_drop_or_jitter_for_ibi(
    signal: torch.Tensor,
    *,
    window_frames: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return contiguous_window_mask(signal, window_frames=window_frames, fill_value=0.0, seed=seed)


def airflow_cannula_displacement(
    signal: torch.Tensor,
    *,
    attenuation: float = 0.2,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return amplitude_attenuation(signal, factor=attenuation)


def belt_failure(
    signal: torch.Tensor,
    *,
    window_frames: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flatline_dropout(signal, window_frames=window_frames, seed=seed)


def high_frequency_contamination(
    signal: torch.Tensor,
    *,
    amplitude: float = 0.1,
    cycles: float = 40.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return line_noise(signal, amplitude=amplitude, cycles=cycles)


CORRUPTION_REGISTRY: dict[str, t.Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "flatline_dropout": flatline_dropout,
    "contiguous_window_mask": contiguous_window_mask,
    "gaussian_noise": gaussian_noise,
    "baseline_drift": baseline_drift,
    "line_noise": line_noise,
    "saturation_clipping": saturation_clipping,
    "spike_artifact": spike_artifact,
    "amplitude_attenuation": amplitude_attenuation,
    "phase_inversion": phase_inversion,
    "spo2_plateau_dropout": spo2_plateau_dropout,
    "rpeak_drop_or_jitter_for_ibi": rpeak_drop_or_jitter_for_ibi,
    "airflow_cannula_displacement": airflow_cannula_displacement,
    "belt_failure": belt_failure,
    "high_frequency_contamination": high_frequency_contamination,
}


def apply_corruption(
    name: str,
    signal: torch.Tensor,
    *,
    seed: int | None = None,
    **kwargs: t.Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name not in CORRUPTION_REGISTRY:
        raise ValueError(f"Unknown sleep2wave corruption: {name}")
    return CORRUPTION_REGISTRY[name](signal, seed=seed, **kwargs)


__all__ = [
    "CORRUPTION_REGISTRY",
    "airflow_cannula_displacement",
    "amplitude_attenuation",
    "apply_corruption",
    "baseline_drift",
    "belt_failure",
    "contiguous_window_mask",
    "flatline_dropout",
    "gaussian_noise",
    "high_frequency_contamination",
    "line_noise",
    "phase_inversion",
    "rpeak_drop_or_jitter_for_ibi",
    "saturation_clipping",
    "spike_artifact",
    "spo2_plateau_dropout",
]
