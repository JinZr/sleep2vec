from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

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


def _model() -> Sleep2WaveDiffusionTransformer:
    return Sleep2WaveDiffusionTransformer(
        latent_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        diffusion_steps=16,
        context_epochs=2,
    )


def test_diffusion_forward_predicts_translation_noise_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    output = model(
        noisy_target_latents={"eeg": torch.randn(2, 2, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"ecg": torch.randn(2, 2, 8)},
        availability_mask=availability,
        quality_mask=quality,
        night_position=torch.tensor([[0.0, 0.5], [0.25, 0.75]]),
    )

    assert set(output.predicted_noise) == {"eeg"}
    assert output.predicted_noise["eeg"].shape == (2, 2, 8)


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
        noisy_target_latents={"eeg": torch.randn(2, 2, 8)},
        timesteps=torch.tensor([1, 3]),
        task=task,
        condition_latents={"eeg": torch.randn(2, 2, 8)},
        availability_mask=availability,
        quality_mask=quality,
        night_position=torch.zeros(2, 2),
    )

    assert set(output.predicted_noise) == {"eeg"}
    assert output.predicted_noise["eeg"].shape == (2, 2, 8)


def test_diffusion_forward_rejects_wrong_latent_shape():
    model = _model()
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    availability, quality = _epoch_masks(batch_size=2, context_epochs=2)

    with pytest.raises(ValueError, match="expected 8"):
        model(
            noisy_target_latents={"eeg": torch.randn(2, 2, 7)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 8)},
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
            noisy_target_latents={"eeg": torch.randn(1, 2, 8)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 8)},
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
            noisy_target_latents={"eeg": torch.randn(2, 2, 8)},
            timesteps=torch.tensor([1, 3]),
            task=task,
            condition_latents={"ecg": torch.randn(2, 2, 8)},
            availability_mask=availability,
            quality_mask=quality,
        )
