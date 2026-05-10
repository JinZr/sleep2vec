from __future__ import annotations

import numpy as np
import pytest

from sleep2wave.evaluation.feature_metrics import (
    airflow_belt_coherence_metrics,
    compute_feature_metrics,
    ecg_peak_metrics,
    eeg_bandpower_error,
    emg_tone_error,
    ibi_mae,
    respiratory_amplitude_error,
    respiratory_cycle_metrics,
    spo2_nadir_metrics,
)


def test_eeg_bandpower_error_is_small_for_identical_sine():
    t = np.arange(128) / 128.0
    signal = np.sin(2.0 * np.pi * 10.0 * t)

    metrics = eeg_bandpower_error(signal, signal, sample_rate_hz=128)

    assert metrics["alpha_bandpower_error"] == pytest.approx(0.0)
    assert set(metrics) >= {"delta_bandpower_error", "theta_bandpower_error", "alpha_bandpower_error"}


def test_emg_tone_error_uses_rms_amplitude():
    assert emg_tone_error(np.ones(8), np.ones(8) * 2.0) == pytest.approx(1.0)


def test_ibi_mae_reports_interval_error():
    assert ibi_mae(np.array([0.8, 0.9, 1.0]), np.array([0.7, 1.0, 1.1])) == pytest.approx(0.1)


def test_spo2_nadir_metrics_track_value_and_timing():
    reference = np.array([[98.0, 96.0, 93.0, 95.0]])
    generated = np.array([[98.0, 92.0, 95.0, 96.0]])

    metrics = spo2_nadir_metrics(reference, generated)

    assert metrics["nadir_error"] == pytest.approx(1.0)
    assert metrics["nadir_timing_error"] == pytest.approx(1.0)
    assert metrics["desaturation_count_error"] == pytest.approx(0.0)


def test_spo2_desaturation_metrics_track_depth_duration_and_slope():
    reference = np.full((1, 1, 40), 97.0)
    generated = reference.copy()
    reference[..., 8:24] = np.linspace(96.0, 90.0, 16)
    generated[..., 8:24] = np.linspace(96.0, 91.0, 16)

    metrics = spo2_nadir_metrics(reference, generated, sample_rate_hz=4)

    assert metrics["desaturation_count_error"] == pytest.approx(0.0)
    assert metrics["desaturation_depth_error"] == pytest.approx(1.0)
    assert metrics["desaturation_duration_error"] > 0.0
    assert metrics["desaturation_slope_error"] > 0.0


def test_respiratory_amplitude_error_uses_percentile_range():
    reference = np.linspace(-1.0, 1.0, 100)
    generated = np.linspace(-0.5, 0.5, 100)

    assert respiratory_amplitude_error(reference, generated) == pytest.approx(0.9)


def test_respiratory_cycle_metrics_track_known_cycles():
    t = np.arange(120) / 4.0
    reference = np.sin(2.0 * np.pi * 0.25 * t)
    generated = np.roll(reference, 1)

    metrics = respiratory_cycle_metrics(reference, generated, sample_rate_hz=4)

    assert metrics["cycle_count_error"] == pytest.approx(0.0)
    assert metrics["peak_timing_error"] == pytest.approx(1.0)
    assert metrics["trough_timing_error"] == pytest.approx(1.0)
    assert metrics["cycle_amplitude_error"] >= 0.0
    assert metrics["inspiration_expiration_ratio_error"] >= 0.0


def test_ecg_peak_metrics_detect_count_and_timing():
    reference = np.zeros(128)
    generated = np.zeros(128)
    reference[[20, 70, 110]] = 1.0
    generated[[21, 71, 111]] = 1.0

    metrics = ecg_peak_metrics(reference, generated, sample_rate_hz=128)

    assert metrics["peak_count_error"] == pytest.approx(0.0)
    assert metrics["peak_timing_error"] == pytest.approx(1.0)
    assert metrics["rr_interval_mae"] == pytest.approx(0.0)
    assert metrics["peak_amplitude_error"] == pytest.approx(0.0)
    assert metrics["qrs_slope_error"] == pytest.approx(0.0)


def test_ecg_peak_metrics_use_all_channels():
    reference = np.zeros((1, 2, 128))
    generated = np.zeros((1, 2, 128))
    reference[:, 0, [20, 70]] = 1.0
    generated[:, 0, [20, 70]] = 1.0
    reference[:, 1, [30, 90]] = 1.0
    generated[:, 1, [34, 94]] = 1.0

    metrics = ecg_peak_metrics(reference, generated, sample_rate_hz=128)

    assert metrics["peak_timing_error"] == pytest.approx(2.0)


def test_compute_feature_metrics_dispatches_by_modality():
    metrics = compute_feature_metrics("spo2", np.array([98.0, 94.0]), np.array([97.0, 95.0]), sample_rate_hz=4)

    assert metrics["nadir_error"] == pytest.approx(1.0)


def test_airflow_belt_coherence_metrics_compare_cross_channel_relationship():
    t = np.arange(120) / 4.0
    airflow = np.sin(2.0 * np.pi * 0.25 * t)[None, None, :]
    reference_belt = airflow.copy()
    generated_belt = -airflow

    metrics = airflow_belt_coherence_metrics(airflow, airflow, reference_belt, generated_belt)

    assert metrics["reference_coherence"] == pytest.approx(1.0)
    assert metrics["generated_coherence"] == pytest.approx(-1.0)
    assert metrics["coherence_error"] == pytest.approx(2.0)
