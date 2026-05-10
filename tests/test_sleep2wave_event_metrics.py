from __future__ import annotations

import numpy as np
import pytest

from sleep2wave.evaluation.event_metrics import (
    compute_event_metric_groups,
    compute_event_metrics,
    compute_generated_signal_event_groups,
    detect_emg_burst_events,
    detect_low_amplitude_epoch_events,
    detect_sigma_burst_events,
    detect_spo2_desaturation_events,
    interval_iou,
)


def test_interval_iou_uses_continuous_intervals():
    assert interval_iou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(5.0 / 15.0)


def test_event_metrics_report_tp_fp_fn_and_timing_errors():
    reference = [[0, 10], [20, 30]]
    generated = [[1, 11], [40, 50]]

    metrics = compute_event_metrics(reference, generated, iou_threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["onset_mae"] == pytest.approx(1.0)
    assert metrics["offset_mae"] == pytest.approx(1.0)


def test_event_metrics_return_nan_timing_when_no_matches():
    metrics = compute_event_metrics([[0, 10]], [[20, 30]], iou_threshold=0.5)

    assert metrics["tp"] == 0.0
    assert np.isnan(metrics["onset_mae"])


def test_event_metric_groups_accept_truth_prediction_aliases():
    metrics = compute_event_metric_groups(
        {"desaturation": {"truth": [[0, 10]], "prediction": [[0, 10]]}},
        iou_threshold=0.5,
    )

    assert metrics["desaturation"]["f1"] == pytest.approx(1.0)


def test_event_metric_groups_reject_missing_intervals():
    with pytest.raises(ValueError, match="must define reference/generated intervals"):
        compute_event_metric_groups({"apnea": {"reference": []}}, iou_threshold=0.5)


def test_generated_signal_event_adapter_detects_spo2_desaturation():
    reference = np.full((1, 1, 40), 96.0, dtype=np.float32)
    generated = reference.copy()
    reference[..., 8:16] = 90.0
    generated[..., 8:16] = 90.0

    metrics = compute_generated_signal_event_groups(
        {"spo2": reference},
        {"spo2": generated},
        sample_rates={"spo2": 4},
        iou_threshold=0.5,
    )

    assert metrics["spo2_desaturation"]["f1"] == pytest.approx(1.0)


def test_spo2_desaturation_detector_filters_short_drops():
    signal = np.full((1, 1, 40), 96.0, dtype=np.float32)
    signal[..., 4:7] = 90.0
    signal[..., 12:20] = 90.0

    events = detect_spo2_desaturation_events(signal, sample_rate_hz=4)

    assert events == [[3.0, 5.0]]


def test_respiratory_low_amplitude_detector_uses_all_channels():
    t = np.arange(120) / 4.0
    signal = np.sin(2.0 * np.pi * 0.25 * t)[None, None, :]
    signal = np.repeat(signal, 2, axis=1)
    signal[:, 1, 40:68] = 0.0

    events = detect_low_amplitude_epoch_events(signal, sample_rate_hz=4)

    assert events
    assert any(start <= 10.5 and end >= 16.5 for start, end in events)


def test_sigma_and_emg_burst_detectors_find_synthetic_bursts():
    t = np.arange(3840) / 128.0
    eeg = np.zeros((1, 1, 3840), dtype=np.float64)
    eeg[..., 128:256] = np.sin(2.0 * np.pi * 13.0 * t[128:256])
    emg = np.zeros((1, 1, 3840), dtype=np.float64)
    emg[..., 512:576] = 1.0

    sigma_events = detect_sigma_burst_events(eeg, sample_rate_hz=128)
    emg_events = detect_emg_burst_events(emg, sample_rate_hz=128)

    assert sigma_events
    assert emg_events


def test_generated_signal_event_adapter_adds_waveform_derived_event_groups():
    t = np.arange(3840) / 128.0
    eeg = np.zeros((1, 1, 3840), dtype=np.float64)
    eeg[..., 128:256] = np.sin(2.0 * np.pi * 13.0 * t[128:256])
    emg = np.zeros((1, 1, 3840), dtype=np.float64)
    emg[..., 512:576] = 1.0

    metrics = compute_generated_signal_event_groups(
        {"eeg": eeg, "emg": emg},
        {"eeg": eeg.copy(), "emg": emg.copy()},
        sample_rates={"eeg": 128, "emg": 128},
        iou_threshold=0.5,
    )

    assert metrics["eeg_sigma_burst"]["f1"] == pytest.approx(1.0)
    assert metrics["emg_burst"]["f1"] == pytest.approx(1.0)
