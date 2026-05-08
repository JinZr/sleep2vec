from __future__ import annotations

import pytest
import torch

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS


def test_autoencoder_returns_temporal_latents_for_all_modalities():
    model = Sleep2WaveAutoencoder(latent_dim=8)
    batch = {
        modality: torch.randn(2, 3, 1, MODALITY_SPECS[modality].frames_per_epoch) for modality in CANONICAL_MODALITIES
    }

    output = model(batch)

    for modality in ("eeg", "eog", "emg", "ecg"):
        assert output.latents[modality].shape == (2, 3, 1, 60, 8)
        assert output.reconstructions[modality].shape == batch[modality].shape
    for modality in ("airflow", "belt", "spo2", "ibi", "resp"):
        assert output.latents[modality].shape == (2, 3, 1, 30, 8)
        assert output.reconstructions[modality].shape == batch[modality].shape


def test_autoencoder_accepts_three_dimensional_signals():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["spo2"])
    batch = {"spo2": torch.randn(2, 3, MODALITY_SPECS["spo2"].frames_per_epoch)}

    output = model(batch)

    assert output.latents["spo2"].shape == (2, 3, 1, 30, 8)
    assert output.reconstructions["spo2"].shape == batch["spo2"].shape


def test_autoencoder_rejects_wrong_epoch_length():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])
    batch = {"eeg": torch.randn(2, 3, 1, MODALITY_SPECS["eeg"].frames_per_epoch - 1)}

    with pytest.raises(ValueError, match="Expected 3840 frames per epoch"):
        model(batch)


def test_autoencoder_preserves_multi_channel_latents_and_reconstructions():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])
    batch = {"eeg": torch.randn(2, 3, 2, MODALITY_SPECS["eeg"].frames_per_epoch)}

    output = model(batch)

    assert output.latents["eeg"].shape == (2, 3, 2, 60, 8)
    assert output.reconstructions["eeg"].shape == batch["eeg"].shape


def test_autoencoder_uses_convtranspose_decoder():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["spo2"])

    decoder = model.modality_autoencoders["spo2"].decoder

    assert any(isinstance(module, torch.nn.ConvTranspose1d) for module in decoder.modules())


def test_autoencoder_decode_latents_returns_waveform_with_channel_dim():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])

    decoded = model.decode_latents({"eeg": torch.randn(2, 3, 1, 60, 8)})

    assert decoded["eeg"].shape == (2, 3, 1, MODALITY_SPECS["eeg"].frames_per_epoch)


def test_autoencoder_decode_latents_keeps_channel_specific_outputs():
    model = Sleep2WaveAutoencoder(latent_dim=8, modalities=["spo2"])
    latents = torch.zeros(1, 1, 2, 30, 8)
    latents[:, :, 1] = 1.0

    decoded = model.decode_latents({"spo2": latents})["spo2"]

    assert decoded.shape == (1, 1, 2, MODALITY_SPECS["spo2"].frames_per_epoch)
    assert not torch.allclose(decoded[:, :, 0], decoded[:, :, 1])


def test_autoencoder_rejects_non_power_of_two_downsample_factor():
    with pytest.raises(ValueError, match="downsample_factor must be a positive power of two"):
        Sleep2WaveAutoencoder(
            latent_dim=8,
            modalities=["spo2"],
            latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 24},
        )
