from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModalityUncertainty:
    mean: torch.Tensor
    std: torch.Tensor
    sample_count: torch.Tensor
    high_uncertainty_mask: torch.Tensor


def compute_modality_uncertainty(
    samples: torch.Tensor,
    *,
    high_uncertainty_threshold: float | None = None,
) -> ModalityUncertainty:
    if samples.dim() < 2:
        raise ValueError("samples must have shape [num_samples, epochs, ...].")
    sample_count = samples.shape[0]
    if sample_count <= 0:
        raise ValueError("samples must contain at least one sample.")
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False) if sample_count > 1 else torch.zeros_like(mean)
    reduce_dims = tuple(range(1, std.dim()))
    epoch_uncertainty = std.mean(dim=reduce_dims) if reduce_dims else std
    if high_uncertainty_threshold is None:
        threshold = epoch_uncertainty.mean() + 2.0 * epoch_uncertainty.std(unbiased=False)
    else:
        threshold = torch.as_tensor(high_uncertainty_threshold, dtype=epoch_uncertainty.dtype, device=std.device)
    return ModalityUncertainty(
        mean=mean,
        std=std,
        sample_count=torch.tensor([sample_count], dtype=torch.long, device=samples.device),
        high_uncertainty_mask=epoch_uncertainty > threshold,
    )


def compute_uncertainty(
    generated: dict[str, torch.Tensor],
    *,
    high_uncertainty_threshold: float | None = None,
) -> dict[str, ModalityUncertainty]:
    if not generated:
        raise ValueError("generated must be non-empty.")
    return {
        modality: compute_modality_uncertainty(
            values,
            high_uncertainty_threshold=high_uncertainty_threshold,
        )
        for modality, values in generated.items()
    }


__all__ = ["ModalityUncertainty", "compute_modality_uncertainty", "compute_uncertainty"]
