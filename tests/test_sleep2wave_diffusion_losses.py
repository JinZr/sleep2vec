from __future__ import annotations

import pytest
import torch

from sleep2wave.diffusion.losses import compute_diffusion_loss
from sleep2wave.diffusion.tasks import build_generation_task


def test_diffusion_loss_respects_target_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.tensor([[[[[0.0], [0.0], [0.0]]]], [[[[10.0], [10.0], [10.0]]]]])}
    target = {"eeg": torch.zeros(2, 1, 1, 3, 1)}
    mask = {"eeg": torch.tensor([[True], [False]])}

    losses = compute_diffusion_loss(predicted, target, task, target_mask=mask)

    assert losses["eeg_mse"].item() == pytest.approx(0.0)
    assert losses["loss"].item() == pytest.approx(0.0)


def test_diffusion_loss_respects_quality_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.tensor([[[[[0.0], [0.0], [0.0]]]], [[[[10.0], [10.0], [10.0]]]]])}
    target = {"eeg": torch.zeros(2, 1, 1, 3, 1)}
    availability = {"eeg": torch.tensor([[True], [True]])}
    quality = {"eeg": torch.tensor([[1.0], [0.0]])}

    losses = compute_diffusion_loss(predicted, target, task, target_mask=availability, quality_mask=quality)

    assert losses["eeg_mse"].item() == pytest.approx(0.0)
    assert losses["loss"].item() == pytest.approx(0.0)


def test_diffusion_loss_rejects_empty_target_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.zeros(2, 1, 1, 3, 1)}
    target = {"eeg": torch.zeros(2, 1, 1, 3, 1)}
    mask = {"eeg": torch.zeros(2, 1, dtype=torch.bool)}

    with pytest.raises(ValueError, match="does not contain any available target entries"):
        compute_diffusion_loss(predicted, target, task, target_mask=mask)


def test_diffusion_loss_accepts_channel_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.tensor([[[[[0.0], [0.0]], [[10.0], [10.0]]]]])}
    target = {"eeg": torch.zeros(1, 1, 2, 2, 1)}
    mask = {"eeg": torch.tensor([[[True, False]]])}

    losses = compute_diffusion_loss(predicted, target, task, target_mask=mask)

    assert losses["loss"].item() == pytest.approx(0.0)


def test_diffusion_loss_ignores_padded_channels_from_channel_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.tensor([[[[[0.0], [0.0]], [[10.0], [10.0]]]]])}
    target = {"eeg": torch.zeros(1, 1, 2, 2, 1)}
    channel_mask = {"eeg": torch.tensor([[[True, False]]])}

    losses = compute_diffusion_loss(predicted, target, task, channel_mask=channel_mask)

    assert losses["loss"].item() == pytest.approx(0.0)


def test_diffusion_loss_accepts_latent_frame_mask():
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    predicted = {"eeg": torch.tensor([[[[[0.0], [10.0], [0.0]]]]])}
    target = {"eeg": torch.zeros(1, 1, 1, 3, 1)}
    mask = {"eeg": torch.tensor([[[[True, False, True]]]])}

    losses = compute_diffusion_loss(predicted, target, task, target_mask=mask)

    assert losses["loss"].item() == pytest.approx(0.0)
