from __future__ import annotations

import numpy as np
import torch


def default_epoch_mask(num_epochs: int, *, value: bool | float, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((num_epochs,), value, dtype=dtype)


def load_epoch_mask(npz, key: str, start: int, end: int, *, dtype: torch.dtype) -> torch.Tensor:
    if key not in npz:
        raise KeyError(f"Mask key '{key}' not found in NPZ.")

    raw = np.asarray(npz[key])
    if raw.ndim == 0:
        return torch.full((end - start,), raw.item(), dtype=dtype)
    if raw.ndim != 1:
        raise ValueError(f"Epoch mask '{key}' must be scalar or 1D, got shape {raw.shape}.")
    if raw.shape[0] < end:
        raise ValueError(f"Epoch mask '{key}' is too short for epochs {start}:{end}.")
    return torch.as_tensor(raw[start:end], dtype=dtype)


def resolve_quality_mask(
    npz,
    key: str | None,
    start: int,
    end: int,
    *,
    available: bool,
) -> torch.Tensor:
    if not available:
        return default_epoch_mask(end - start, value=0.0, dtype=torch.float32)
    if key is None:
        return default_epoch_mask(end - start, value=1.0, dtype=torch.float32)
    return load_epoch_mask(npz, key, start, end, dtype=torch.float32)


def resolve_availability_mask(
    npz,
    key: str | None,
    start: int,
    end: int,
    *,
    available: bool,
) -> torch.Tensor:
    if not available:
        return default_epoch_mask(end - start, value=False, dtype=torch.bool)
    if key is None:
        return default_epoch_mask(end - start, value=True, dtype=torch.bool)
    return load_epoch_mask(npz, key, start, end, dtype=torch.bool)


__all__ = ["default_epoch_mask", "load_epoch_mask", "resolve_availability_mask", "resolve_quality_mask"]
