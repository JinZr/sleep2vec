from __future__ import annotations

import pytest
import torch

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS


def test_autoencoder_returns_one_latent_per_epoch_for_all_modalities():
    model = Sleep2WaveAutoencoder(latent_dim=8)
    batch = {
        modality: torch.randn(2, 3, 1, MODALITY_SPECS[modality].frames_per_epoch) for modality in CANONICAL_MODALITIES
    }

    output = model(batch)

    for modality in CANONICAL_MODALITIES:
        assert output.latents[modality].shape == (2, 3, 8)
        assert output.reconstructions[modality].shape == batch[modality].shape


def test_autoencoder_accepts_three_dimensional_signals():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["spo2"])
    batch = {"spo2": torch.randn(2, 3, MODALITY_SPECS["spo2"].frames_per_epoch)}

    output = model(batch)

    assert output.latents["spo2"].shape == (2, 3, 8)
    assert output.reconstructions["spo2"].shape == batch["spo2"].shape


def test_autoencoder_rejects_wrong_epoch_length():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])
    batch = {"eeg": torch.randn(2, 3, 1, MODALITY_SPECS["eeg"].frames_per_epoch - 1)}

    with pytest.raises(ValueError, match="Expected 3840 frames per epoch"):
        model(batch)


def test_autoencoder_rejects_multi_channel_input():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])
    batch = {"eeg": torch.randn(2, 3, 2, MODALITY_SPECS["eeg"].frames_per_epoch)}

    with pytest.raises(ValueError, match="supports only one channel"):
        model(batch)


def test_autoencoder_decode_latents_returns_waveform_with_channel_dim():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])

    decoded = model.decode_latents({"eeg": torch.randn(2, 3, 8)})

    assert decoded["eeg"].shape == (2, 3, 1, MODALITY_SPECS["eeg"].frames_per_epoch)
