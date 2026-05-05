from __future__ import annotations

import torch

from sleep2wave.diffusion.tasks import GenerationTask, validate_generation_task


def compute_diffusion_loss(
    predicted_noise: dict[str, torch.Tensor],
    target_noise: dict[str, torch.Tensor],
    task: GenerationTask,
    *,
    target_mask: dict[str, torch.Tensor] | None = None,
    quality_mask: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    task = validate_generation_task(task)
    losses: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for modality in task.target_modalities:
        if modality not in predicted_noise:
            raise ValueError(f"Missing predicted noise for target modality '{modality}'.")
        if modality not in target_noise:
            raise ValueError(f"Missing target noise for target modality '{modality}'.")
        if predicted_noise[modality].shape != target_noise[modality].shape:
            raise ValueError(
                f"Noise shape mismatch for '{modality}': "
                f"{tuple(predicted_noise[modality].shape)} != {tuple(target_noise[modality].shape)}."
            )
        mask = target_mask[modality] if target_mask is not None and modality in target_mask else None
        if quality_mask is not None and modality in quality_mask:
            quality = torch.as_tensor(quality_mask[modality], device=predicted_noise[modality].device) > 0
            mask = quality if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=quality.device) & quality
        value = _mse_loss(predicted_noise[modality], target_noise[modality], mask, modality)
        losses[f"{modality}_mse"] = value
        total = value if total is None else total + value
    if total is None:
        raise ValueError("Diffusion loss requires at least one target modality.")
    losses["loss"] = total / len(task.target_modalities)
    return losses


def _mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    modality: str,
) -> torch.Tensor:
    values = (predicted - target).pow(2)
    if mask is None:
        return values.mean()
    mask = torch.as_tensor(mask, dtype=values.dtype, device=values.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != values.shape[:2]:
        raise ValueError(f"target_mask['{modality}'] must have shape {tuple(values.shape[:2])}.")
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum()
    if denominator <= 0:
        raise ValueError(f"target_mask['{modality}'] does not contain any available target entries.")
    return (values * mask).sum() / denominator


__all__ = ["compute_diffusion_loss"]
