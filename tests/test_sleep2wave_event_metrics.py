from __future__ import annotations

import numpy as np
import pytest

from sleep2wave.evaluation.event_metrics import compute_event_metric_groups, compute_event_metrics, interval_iou


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
