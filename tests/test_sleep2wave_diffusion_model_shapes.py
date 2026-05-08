from __future__ import annotations

import pytest
import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.tasks import build_generation_task


def _epoch_masks(batch_size: int, context_epochs: int):
    availability = {
        modality: torch.ones(batch_size, context_epochs, dtype=torch.bool) for modality in CANONICAL_MODALITIES
    }
    quality = {
        modality: torch.ones(batch_size, context_epochs, dtype=torch.float32) for modality in CANONICAL_MODALITIES
    }
    return availability, quality


def _channel_masks(batch_size: int, context_epochs: int, channels: int = 2):
    return {
        modality: torch.ones(batch_size, context_epochs, channels, dtype=torch.bool)
        for modality in CANONICAL_MODALITIES
    }


def _model() -> Sleep2WaveDiffusionTransformer:
    return Sleep2WaveDiffusionTransformer(
        latent_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        diffusion_steps=16,
        context_epochs=2,
        latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
        patches_per_epoch=6,
    )


def test_diffusion_forward_predicts_high_frequency_translation_noise_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    output = model(
        noisy_target_latents={"eeg": torch.randn(2, 2, 1, 60, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 1, 60, 8)},
        availability_mask=availability,
        quality_mask=quality,
        night_position=torch.tensor([[0.0, 0.5], [0.25, 0.75]]),
    )

    assert set(output.predicted_noise) == {"eeg"}
    assert output.predicted_noise["eeg"].shape == (2, 2, 1, 60, 8)


def test_diffusion_forward_predicts_low_frequency_translation_noise_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["spo2"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    output = model(
        noisy_target_latents={"spo2": torch.randn(2, 2, 1, 30, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 1, 60, 8)},
        availability_mask=availability,
        quality_mask=quality,
        night_position=torch.tensor([[0.0, 0.5], [0.25, 0.75]]),
    )

    assert set(output.predicted_noise) == {"spo2"}
    assert output.predicted_noise["spo2"].shape == (2, 2, 1, 30, 8)


def test_diffusion_forward_routes_restoration_prediction_through_aux():
    model = _model()
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    output = model(
        noisy_target_latents={"eeg": torch.randn(2, 2, 1, 60, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"eeg": torch.randn(2, 2, 1, 60, 8)},
        availability_mask=availability,
        quality_mask=quality,
        night_position=torch.zeros(2, 2),
    )

    assert set(output.predicted_noise) == {"eeg"}
    assert output.predicted_noise["eeg"].shape == (2, 2, 1, 60, 8)


def test_diffusion_forward_rejects_wrong_latent_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    with pytest.raises(ValueError, match="expected 8"):
        model(
            noisy_target_latents={"eeg": torch.randn(2, 2, 1, 60, 7)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 1, 60, 8)},
            availability_mask=availability,
            quality_mask=quality,
            night_position=torch.zeros(2, 2),
        )


def test_diffusion_forward_rejects_mismatched_latent_batch_size():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    with pytest.raises(ValueError, match="batch size 1; expected 2"):
        model(
            noisy_target_latents={"eeg": torch.randn(1, 2, 1, 60, 8)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 1, 60, 8)},
            availability_mask=availability,
            quality_mask=quality,
            night_position=torch.zeros(2, 2),
        )


def test_diffusion_forward_requires_night_position_when_enabled():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    with pytest.raises(ValueError, match="night_position is required"):
        model(
            noisy_target_latents={"eeg": torch.randn(2, 2, 1, 60, 8)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 1, 60, 8)},
            availability_mask=availability,
            quality_mask=quality,
        )


def test_diffusion_forward_predicts_multi_channel_high_frequency_noise_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)
    channel_mask = _channel_masks(batch_size=2, context_epochs=2, channels=2)

    output = model(
        noisy_target_latents={"eeg": torch.randn(2, 2, 2, 60, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 2, 60, 8)},
        availability_mask=availability,
        channel_mask=channel_mask,
        quality_mask=quality,
        night_position=torch.zeros(2, 2),
    )

    assert output.predicted_noise["eeg"].shape == (2, 2, 2, 60, 8)


def test_diffusion_forward_predicts_multi_channel_low_frequency_noise_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["spo2"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)
    channel_mask = _channel_masks(batch_size=2, context_epochs=2, channels=2)

    output = model(
        noisy_target_latents={"spo2": torch.randn(2, 2, 2, 30, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 2, 60, 8)},
        availability_mask=availability,
        channel_mask=channel_mask,
        quality_mask=quality,
        night_position=torch.zeros(2, 2),
    )

    assert output.predicted_noise["spo2"].shape == (2, 2, 2, 30, 8)


def test_diffusion_forward_uses_channel_mask_for_padded_targets():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)
    channel_mask = _channel_masks(batch_size=2, context_epochs=2, channels=2)
    channel_mask["eeg"][1, :, 1] = False
    channel_mask["ecg"][1, :, 1] = False

    output = model(
        noisy_target_latents={"eeg": torch.randn(2, 2, 2, 60, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 2, 60, 8)},
        availability_mask=availability,
        channel_mask=channel_mask,
        quality_mask=quality,
        night_position=torch.zeros(2, 2),
    )
    padded_token = output.task_mask.active_tokens[1, model._layout_for_channel_count(2).token_index("eeg", 0, 1, 0)]

    assert output.predicted_noise["eeg"].shape == (2, 2, 2, 60, 8)
    assert not bool(padded_token)


def test_diffusion_forward_requires_channel_mask_for_multi_channel_latents():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    with pytest.raises(ValueError, match="channel_mask"):
        model(
            noisy_target_latents={"eeg": torch.randn(2, 2, 2, 60, 8)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 2, 60, 8)},
            availability_mask=availability,
            quality_mask=quality,
            night_position=torch.zeros(2, 2),
        )
