from __future__ import annotations

import math

import numpy as np
import pytest

from sleep2wave.evaluation.waveform_metrics import (
    compute_waveform_metrics,
    correlation,
    min_mae,
    min_rmse,
    snr_improvement,
    spectral_distance,
)


def test_waveform_metrics_basic_values():
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    generated = np.array([0.0, 2.0, 2.0, 4.0])

    metrics = compute_waveform_metrics(reference, generated)

    assert metrics["rmse"] == pytest.approx(math.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["correlation"] > 0.9
    assert metrics["spectral_distance"] >= 0.0


def test_min_shift_metrics_allow_small_timing_offsets():
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    generated = np.array([9.0, 0.0, 1.0, 2.0])

    assert min_rmse(reference, generated, max_shift_frames=1) == pytest.approx(0.0)
    assert min_mae(reference, generated, max_shift_frames=1) == pytest.approx(0.0)


def test_snr_improvement_uses_baseline_error():
    reference = np.array([0.0, 0.0, 0.0, 0.0])
    baseline = np.array([2.0, 2.0, 2.0, 2.0])
    generated = np.array([1.0, 1.0, 1.0, 1.0])

    assert snr_improvement(reference, generated, baseline) == pytest.approx(10.0 * np.log10(4.0))


def test_spectral_distance_is_zero_for_identical_signals():
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 64))

    assert spectral_distance(signal, signal) == pytest.approx(0.0)


def test_correlation_returns_nan_for_constant_signal():
    assert np.isnan(correlation(np.ones(4), np.ones(4)))
