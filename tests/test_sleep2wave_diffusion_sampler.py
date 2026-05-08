from __future__ import annotations

import pytest
import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.diffusion.model import Sleep2WaveDiffusionOutput, Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.samplers import DDIMSampler, DDPMSampler
from sleep2wave.diffusion.schedule import build_diffusion_schedule
from sleep2wave.diffusion.tasks import build_generation_task


def _model() -> Sleep2WaveDiffusionTransformer:
    return Sleep2WaveDiffusionTransformer(
        latent_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        diffusion_steps=8,
        context_epochs=2,
        latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
        patches_per_epoch=6,
    )


def _inputs():
    availability = {modality: torch.ones(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    quality = {modality: torch.ones(2, 2, dtype=torch.float32) for modality in CANONICAL_MODALITIES}
    return {
        "condition_latents": {"ecg": torch.randn(2, 2, 1, 60, 8)},
        "task": build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"]),
        "availability_mask": availability,
        "quality_mask": quality,
        "night_position": torch.zeros(2, 2),
    }


def _multi_channel_inputs():
    inputs = _inputs()
    inputs["condition_latents"] = {"ecg": torch.randn(2, 2, 2, 60, 8)}
    inputs["channel_mask"] = {modality: torch.ones(2, 2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    return inputs


def test_ddim_sampler_returns_requested_num_samples():
    sampler = DDIMSampler(build_diffusion_schedule(8), steps=2, num_samples=3)
    output = sampler.sample(_model(), **_inputs())

    assert output.generated_latents["eeg"].shape == (3, 2, 2, 1, 60, 8)


def test_ddpm_sampler_returns_requested_num_samples():
    sampler = DDPMSampler(build_diffusion_schedule(8), steps=8, num_samples=2)
    output = sampler.sample(_model(), **_inputs())

    assert output.generated_latents["eeg"].shape == (2, 2, 2, 1, 60, 8)


def test_sampler_initializes_low_frequency_target_shape():
    inputs = _inputs()
    inputs["task"] = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["spo2"])
    sampler = DDIMSampler(build_diffusion_schedule(8), steps=2, num_samples=1)
    output = sampler.sample(_model(), **inputs)

    assert output.generated_latents["spo2"].shape == (1, 2, 2, 1, 30, 8)


def test_ddim_sampler_returns_multi_channel_targets():
    sampler = DDIMSampler(build_diffusion_schedule(8), steps=2, num_samples=3)
    output = sampler.sample(_model(), **_multi_channel_inputs())

    assert output.generated_latents["eeg"].shape == (3, 2, 2, 2, 60, 8)


def test_sampler_target_channel_count_follows_channel_mask():
    inputs = _multi_channel_inputs()
    inputs["channel_mask"]["eeg"] = torch.ones(2, 2, 3, dtype=torch.bool)
    sampler = DDIMSampler(build_diffusion_schedule(8), steps=2, num_samples=1)
    output = sampler.sample(_model(), **inputs)

    assert output.generated_latents["eeg"].shape == (1, 2, 2, 3, 60, 8)


def test_ddim_sampler_forwards_condition_availability_mask(monkeypatch):
    model = _model()
    inputs = _inputs()
    condition_availability = {"ecg": torch.ones(2, 2, 6, dtype=torch.bool)}
    seen_masks = []

    def fake_forward(**kwargs):
        seen_masks.append(kwargs.get("condition_availability_mask"))
        target = kwargs["noisy_target_latents"]["eeg"]
        return Sleep2WaveDiffusionOutput(
            predicted_noise={"eeg": torch.zeros_like(target)},
            task_mask=None,
        )

    monkeypatch.setattr(model, "forward", fake_forward)
    sampler = DDIMSampler(build_diffusion_schedule(8), steps=2, num_samples=1)

    sampler.sample(
        model,
        **inputs,
        condition_availability_mask=condition_availability,
    )

    assert seen_masks
    assert all(mask is condition_availability for mask in seen_masks)


def test_ddpm_sampler_rejects_sparse_steps():
    with pytest.raises(ValueError, match="DDPM sampling requires steps to equal"):
        DDPMSampler(build_diffusion_schedule(8), steps=2)


def test_sampler_rejects_steps_above_schedule_length():
    with pytest.raises(ValueError, match="steps must be <= diffusion schedule length"):
        DDIMSampler(build_diffusion_schedule(8), steps=9)
