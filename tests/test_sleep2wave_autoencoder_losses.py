from __future__ import annotations

import pytest
import torch

from sleep2wave.autoencoders.losses import Sleep2WaveAutoencoderLoss, compute_autoencoder_loss
from sleep2wave.generative.config import AutoencoderLossConfig


def test_autoencoder_waveform_losses_respect_availability_and_quality_masks():
    target = {"spo2": torch.zeros(1, 2, 1, 4)}
    reconstruction = {"spo2": torch.tensor([[[[1.0, 1.0, 1.0, 1.0]], [[10.0, 10.0, 10.0, 10.0]]]])}
    availability = {"spo2": torch.tensor([[True, True]])}
    quality = {"spo2": torch.tensor([[1.0, 0.0]])}
    config = AutoencoderLossConfig(waveform_l1_weight=1.0, waveform_l2_weight=1.0, spectral_weight=0.0)

    losses = compute_autoencoder_loss(
        reconstruction,
        target,
        availability_mask=availability,
        quality_mask=quality,
        config=config,
    )

    assert torch.isclose(losses["waveform_l1_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["waveform_l2_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["loss"], torch.tensor(2.0))


def test_autoencoder_spectral_loss_is_finite():
    target = {"eeg": torch.sin(torch.linspace(0, 1, 64)).view(1, 1, 1, 64)}
    reconstruction = {"eeg": target["eeg"] + 0.1}
    config = AutoencoderLossConfig(waveform_l1_weight=0.0, waveform_l2_weight=0.0, spectral_weight=1.0)

    losses = compute_autoencoder_loss(reconstruction, target, config=config)

    assert torch.isfinite(losses["spectral_loss"])
    assert torch.isfinite(losses["loss"])


def test_autoencoder_loss_rejects_all_zero_weights():
    config = AutoencoderLossConfig(waveform_l1_weight=0.0, waveform_l2_weight=0.0, spectral_weight=0.0)

    with pytest.raises(ValueError, match="At least one Sleep2Wave autoencoder loss weight"):
        Sleep2WaveAutoencoderLoss(config)
