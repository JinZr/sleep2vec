from __future__ import annotations

import torch
import torch.nn as nn

from sleep2wave.generative.config import AutoencoderLossConfig


def _ensure_signal4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(2)
    if tensor.dim() == 4:
        return tensor
    raise ValueError(f"Signal tensor must be [B, E, S] or [B, E, C, S], got shape {tuple(tensor.shape)}.")


def _epoch_weights(
    modality: str,
    target: torch.Tensor,
    availability_mask: dict[str, torch.Tensor] | None,
    quality_mask: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    batch_size, epoch_count = target.shape[:2]
    weights = torch.ones((batch_size, epoch_count), dtype=target.dtype, device=target.device)
    if availability_mask is not None and modality in availability_mask:
        weights = weights * availability_mask[modality].to(device=target.device, dtype=target.dtype)
    if quality_mask is not None and modality in quality_mask:
        weights = weights * quality_mask[modality].to(device=target.device, dtype=target.dtype)
    return weights


def _masked_waveform_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    broadcast_weights = weights
    while broadcast_weights.dim() < values.dim():
        broadcast_weights = broadcast_weights.unsqueeze(-1)
    weighted = values * broadcast_weights
    denominator = broadcast_weights.sum() * values.shape[-1]
    if values.dim() == 4:
        denominator = denominator * values.shape[-2]
    return weighted.sum(), denominator


def _masked_epoch_mean(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(), weights.sum() * values.shape[-1]


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(denominator.dtype).eps
    return numerator / denominator.clamp_min(eps)


def _spectral_epoch_error(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction = _ensure_signal4d(reconstruction)
    target = _ensure_signal4d(target)
    recon_spec = torch.fft.rfft(reconstruction, dim=-1).abs()
    target_spec = torch.fft.rfft(target, dim=-1).abs()
    return torch.abs(torch.log1p(recon_spec) - torch.log1p(target_spec)).mean(dim=-1)


def _validate_loss_weights(config: AutoencoderLossConfig) -> None:
    total = config.waveform_l1_weight + config.waveform_l2_weight + config.spectral_weight
    if total <= 0:
        raise ValueError("At least one Sleep2Wave autoencoder loss weight must be positive.")


def compute_autoencoder_loss(
    reconstructions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    availability_mask: dict[str, torch.Tensor] | None = None,
    quality_mask: dict[str, torch.Tensor] | None = None,
    config: AutoencoderLossConfig,
) -> dict[str, torch.Tensor]:
    _validate_loss_weights(config)

    device = next(iter(targets.values())).device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    l1_numerator = zero.clone()
    l1_denominator = zero.clone()
    l2_numerator = zero.clone()
    l2_denominator = zero.clone()
    spectral_numerator = zero.clone()
    spectral_denominator = zero.clone()

    for modality, target in targets.items():
        if modality not in reconstructions:
            raise ValueError(f"Missing reconstruction for modality '{modality}'.")
        reconstruction = reconstructions[modality]
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"Reconstruction shape for '{modality}' must match target: "
                f"{tuple(reconstruction.shape)} != {tuple(target.shape)}."
            )
        target4d = _ensure_signal4d(target)
        reconstruction4d = _ensure_signal4d(reconstruction)
        weights = _epoch_weights(modality, target4d, availability_mask, quality_mask)

        l1_sum, l1_count = _masked_waveform_mean(torch.abs(reconstruction4d - target4d), weights)
        l2_sum, l2_count = _masked_waveform_mean((reconstruction4d - target4d).pow(2), weights)
        l1_numerator = l1_numerator + l1_sum
        l1_denominator = l1_denominator + l1_count
        l2_numerator = l2_numerator + l2_sum
        l2_denominator = l2_denominator + l2_count

        spectral_error = _spectral_epoch_error(reconstruction4d, target4d)
        spectral_sum, spectral_count = _masked_epoch_mean(spectral_error, weights)
        spectral_numerator = spectral_numerator + spectral_sum
        spectral_denominator = spectral_denominator + spectral_count

    waveform_l1 = _safe_divide(l1_numerator, l1_denominator)
    waveform_l2 = _safe_divide(l2_numerator, l2_denominator)
    spectral = _safe_divide(spectral_numerator, spectral_denominator)
    total = (
        config.waveform_l1_weight * waveform_l1
        + config.waveform_l2_weight * waveform_l2
        + config.spectral_weight * spectral
    )
    return {
        "loss": total,
        "waveform_l1_loss": waveform_l1,
        "waveform_l2_loss": waveform_l2,
        "spectral_loss": spectral,
    }


class Sleep2WaveAutoencoderLoss(nn.Module):
    def __init__(self, config: AutoencoderLossConfig) -> None:
        super().__init__()
        _validate_loss_weights(config)
        self.config = config

    def forward(
        self,
        reconstructions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        availability_mask: dict[str, torch.Tensor] | None = None,
        quality_mask: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return compute_autoencoder_loss(
            reconstructions,
            targets,
            availability_mask=availability_mask,
            quality_mask=quality_mask,
            config=self.config,
        )


__all__ = ["Sleep2WaveAutoencoderLoss", "compute_autoencoder_loss"]
