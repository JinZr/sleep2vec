from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch


@dataclass(frozen=True)
class FusedWindowTensor:
    values: torch.Tensor
    epoch_index: torch.Tensor


def validate_single_night(metadata_rows: t.Sequence[dict[str, t.Any]]) -> None:
    if not metadata_rows:
        raise ValueError("At least one metadata row is required.")
    night_keys = {
        (
            row.get("subject_id"),
            row.get("night_id"),
            row.get("path"),
        )
        for row in metadata_rows
    }
    if len(night_keys) != 1:
        raise ValueError("Sleep2Wave generation currently supports one subject/night per run.")


def _prepare_starts(start_epochs: t.Sequence[int], window_count: int, device: torch.device) -> torch.Tensor:
    starts = torch.as_tensor(list(start_epochs), dtype=torch.long, device=device)
    if starts.shape != (window_count,):
        raise ValueError(f"start_epochs must have shape ({window_count},), got {tuple(starts.shape)}.")
    if (starts < 0).any():
        raise ValueError("start_epochs must be non-negative.")
    return starts


def _layout_from_starts(
    start_epochs: t.Sequence[int],
    *,
    window_count: int,
    context_epochs: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    starts = _prepare_starts(start_epochs, window_count, device)
    order = torch.argsort(starts)
    sorted_starts = starts[order]
    min_epoch = int(sorted_starts[0].item())
    max_epoch = int((sorted_starts + context_epochs).max().item())
    total_epochs = max_epoch - min_epoch
    coverage = torch.zeros(total_epochs, dtype=torch.long, device=device)
    for start in sorted_starts.tolist():
        left = start - min_epoch
        coverage[left : left + context_epochs] += 1
    if (coverage == 0).any():
        raise ValueError("Sliding-window predictions do not cover a contiguous epoch range.")
    return order, min_epoch, torch.arange(min_epoch, max_epoch, dtype=torch.long, device=device)


def fuse_overlapping_windows(
    windows: torch.Tensor,
    start_epochs: t.Sequence[int],
    *,
    mode: str = "mean",
    eps: float = 1e-6,
) -> FusedWindowTensor:
    if windows.dim() < 3:
        raise ValueError("windows must have shape [num_samples, windows, context_epochs, ...].")
    if mode not in {"mean", "median", "uncertainty_weighted"}:
        raise ValueError("mode must be 'mean', 'median', or 'uncertainty_weighted'.")

    num_samples, window_count, context_epochs = windows.shape[:3]
    if num_samples <= 0 or window_count <= 0 or context_epochs <= 0:
        raise ValueError("windows must have positive sample, window, and context dimensions.")
    order, min_epoch, epoch_index = _layout_from_starts(
        start_epochs,
        window_count=window_count,
        context_epochs=context_epochs,
        device=windows.device,
    )
    windows = windows[:, order]
    starts = torch.as_tensor(list(start_epochs), dtype=torch.long, device=windows.device)[order]
    total_epochs = int(epoch_index.numel())
    trailing_shape = windows.shape[3:]

    if mode == "median":
        fused = torch.empty((num_samples, total_epochs, *trailing_shape), dtype=windows.dtype, device=windows.device)
        for epoch in range(total_epochs):
            absolute_epoch = min_epoch + epoch
            pieces = []
            for window_idx, start in enumerate(starts.tolist()):
                local_epoch = absolute_epoch - start
                if 0 <= local_epoch < context_epochs:
                    pieces.append(windows[:, window_idx, local_epoch])
            fused[:, epoch] = torch.stack(pieces, dim=0).median(dim=0).values
        return FusedWindowTensor(values=fused, epoch_index=epoch_index)

    fused = torch.zeros((num_samples, total_epochs, *trailing_shape), dtype=windows.dtype, device=windows.device)
    weights = torch.zeros((total_epochs,), dtype=windows.dtype, device=windows.device)
    if mode == "uncertainty_weighted":
        uncertainty = windows.std(dim=0, unbiased=False)
        reduce_dims = tuple(range(2, uncertainty.dim()))
        epoch_uncertainty = uncertainty.mean(dim=reduce_dims) if reduce_dims else uncertainty
        window_weights = 1.0 / (epoch_uncertainty + eps)
    else:
        window_weights = torch.ones((window_count, context_epochs), dtype=windows.dtype, device=windows.device)

    view_shape = (1, *([1] * len(trailing_shape)))
    for window_idx, start in enumerate(starts.tolist()):
        left = start - min_epoch
        for local_epoch in range(context_epochs):
            weight = window_weights[window_idx, local_epoch]
            fused[:, left + local_epoch] += windows[:, window_idx, local_epoch] * weight.reshape(view_shape)
            weights[left + local_epoch] += weight
    fused = fused / weights.reshape(1, total_epochs, *([1] * len(trailing_shape)))
    return FusedWindowTensor(values=fused, epoch_index=epoch_index)


def fuse_mask_windows(
    windows: torch.Tensor,
    start_epochs: t.Sequence[int],
    *,
    mode: str,
) -> FusedWindowTensor:
    if windows.dim() < 2:
        raise ValueError("windows must have shape [windows, context_epochs, ...].")
    if mode not in {"any", "mean"}:
        raise ValueError("mode must be 'any' or 'mean'.")

    window_count, context_epochs = windows.shape[:2]
    order, min_epoch, epoch_index = _layout_from_starts(
        start_epochs,
        window_count=window_count,
        context_epochs=context_epochs,
        device=windows.device,
    )
    windows = windows[order]
    starts = torch.as_tensor(list(start_epochs), dtype=torch.long, device=windows.device)[order]
    total_epochs = int(epoch_index.numel())
    trailing_shape = windows.shape[2:]

    if mode == "any":
        fused = torch.zeros((total_epochs, *trailing_shape), dtype=torch.bool, device=windows.device)
        for window_idx, start in enumerate(starts.tolist()):
            left = start - min_epoch
            for local_epoch in range(context_epochs):
                fused[left + local_epoch] |= windows[window_idx, local_epoch].to(dtype=torch.bool)
        return FusedWindowTensor(values=fused, epoch_index=epoch_index)

    fused = torch.zeros((total_epochs, *trailing_shape), dtype=torch.float32, device=windows.device)
    counts = torch.zeros((total_epochs,), dtype=torch.float32, device=windows.device)
    for window_idx, start in enumerate(starts.tolist()):
        left = start - min_epoch
        for local_epoch in range(context_epochs):
            fused[left + local_epoch] += windows[window_idx, local_epoch].to(dtype=torch.float32)
            counts[left + local_epoch] += 1.0
    fused = fused / counts.reshape(total_epochs, *([1] * len(trailing_shape)))
    return FusedWindowTensor(values=fused, epoch_index=epoch_index)


__all__ = [
    "FusedWindowTensor",
    "fuse_mask_windows",
    "fuse_overlapping_windows",
    "validate_single_night",
]
