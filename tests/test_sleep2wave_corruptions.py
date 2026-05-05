from __future__ import annotations

import pytest
import torch

from sleep2wave.data import corruptions


def test_corruptions_are_seed_reproducible():
    signal = torch.arange(20, dtype=torch.float32).view(1, 1, 20)

    first, first_mask = corruptions.contiguous_window_mask(signal, window_frames=5, seed=11)
    second, second_mask = corruptions.contiguous_window_mask(signal, window_frames=5, seed=11)

    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert first_mask.sum().item() == 5


def test_gaussian_noise_is_seed_reproducible_and_shape_preserving():
    signal = torch.ones((2, 1, 16), dtype=torch.float32)

    first, first_mask = corruptions.gaussian_noise(signal, std=0.2, seed=3)
    second, second_mask = corruptions.gaussian_noise(signal, std=0.2, seed=3)

    assert first.shape == signal.shape
    assert first_mask.shape == signal.shape
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert first_mask.all()


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("flatline_dropout", {"window_frames": 4}),
        ("contiguous_window_mask", {"window_frames": 4}),
        ("gaussian_noise", {"std": 0.1}),
        ("baseline_drift", {"amplitude": 0.1}),
        ("line_noise", {"amplitude": 0.1, "cycles": 3}),
        ("saturation_clipping", {"min_value": 2.0, "max_value": 8.0}),
        ("spike_artifact", {"num_spikes": 2, "magnitude": 5.0}),
        ("amplitude_attenuation", {"factor": 0.5}),
        ("phase_inversion", {}),
        ("spo2_plateau_dropout", {"window_frames": 4}),
        ("rpeak_drop_or_jitter_for_ibi", {"window_frames": 4}),
        ("airflow_cannula_displacement", {"attenuation": 0.2}),
        ("belt_failure", {"window_frames": 4}),
        ("high_frequency_contamination", {"amplitude": 0.1, "cycles": 5}),
    ],
)
def test_corruptions_return_signal_and_mask(name: str, kwargs: dict):
    signal = torch.arange(20, dtype=torch.float32).view(1, 1, 20)

    corrupted, mask = corruptions.apply_corruption(name, signal, seed=5, **kwargs)

    assert corrupted.shape == signal.shape
    assert mask.shape == signal.shape
    assert mask.dtype == torch.bool


def test_unknown_corruption_raises():
    signal = torch.zeros((1, 1, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown Sleep2Wave corruption"):
        corruptions.apply_corruption("unknown", signal)
