from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl

from sleep2vec.infer import run_inference
import sleep2vec.metrics as metrics_mod
from sleep2vec.metrics import (
    AHI_COARSE_THRESHOLD_GRID,
    AHI_FINE_THRESHOLD_GRID,
    _evaluate_single_ahi_record,
    binary_sequence_to_segments,
    compute_ahi_event_metrics,
    extract_ahi_summary_scatter_arrays,
    filter_segments_by_duration,
    filter_segments_by_stage,
    merge_intervals,
    select_best_ahi_threshold,
)
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning


def _ahi_record(
    *,
    num_stage_tokens: int,
    truth_segments: list[tuple[int, int]],
    score_segments: list[tuple[int, int, float]],
    base_score: float = 0.05,
    true_ahi: float,
    tst_hours: float,
    stage_value: int | None = 2,
) -> dict[str, np.ndarray]:
    length = num_stage_tokens * 30
    truth = np.zeros(length, dtype=np.int64)
    score = np.full(length, base_score, dtype=np.float32)
    for start, end in truth_segments:
        truth[start : end + 1] = 1
    for start, end, value in score_segments:
        score[start : end + 1] = value
    record = {
        "truth": truth,
        "score": score,
        "true_ahi": np.float32(true_ahi),
        "tst_hours": np.float32(tst_hours),
    }
    if stage_value is not None:
        record["stage5"] = np.full(num_stage_tokens, stage_value, dtype=np.int64)
    return record


def _prepared_record(*, summary_enabled: bool = True) -> metrics_mod.PreparedAHIRecord:
    return metrics_mod.PreparedAHIRecord(
        score=np.array([0.1], dtype=np.float32),
        gt_segments=[],
        sleep_mask=np.array([1], dtype=np.int64),
        true_ahi=1.0,
        tst_hours=4.0,
        summary_enabled=summary_enabled,
    )


def test_binary_sequence_to_segments_preserves_exact_boundaries():
    assert binary_sequence_to_segments([0, 1, 1, 0, 1, 1, 1, 0]) == [[1, 2], [4, 6]]


def test_merge_intervals_uses_three_second_tolerance():
    merged = merge_intervals([[0, 10], [12, 20], [30, 35]])
    assert merged == [[0, 20], [30, 35]]


def test_filter_segments_by_stage_requires_sleep_overlap():
    sleep_mask = np.zeros(50, dtype=np.int64)
    sleep_mask[22:35] = 1
    filtered = filter_segments_by_stage([[0, 10], [20, 40]], sleep_mask)
    assert filtered == [[20, 40]]


def test_filter_segments_by_stage_keeps_sleep_overlap_at_segment_end():
    sleep_mask = np.zeros(20, dtype=np.int64)
    sleep_mask[11] = 1
    filtered = filter_segments_by_stage([[0, 11]], sleep_mask)
    assert filtered == [[0, 11]]


def test_filter_segments_by_duration_uses_inclusive_duration_semantics():
    filtered = filter_segments_by_duration([[0, 8], [0, 9], [0, 10], [20, 35]])
    assert filtered == [[0, 9], [0, 10], [20, 35]]


def test_evaluate_single_ahi_record_summary_ahi_does_not_merge_close_predictions():
    record = _ahi_record(
        num_stage_tokens=240,
        truth_segments=[(10, 35)],
        score_segments=[(10, 20, 0.9), (23, 33, 0.9)],
        true_ahi=0.5,
        tst_hours=4.0,
    )

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (1.0, 0.0, 0.0)
    assert pred_ahi == pytest.approx(0.5)
    assert true_ahi == 0.5


def test_evaluate_single_ahi_record_summary_ahi_keeps_short_predictions():
    record = _ahi_record(
        num_stage_tokens=240,
        truth_segments=[(40, 55)],
        score_segments=[(40, 48, 0.9)],
        true_ahi=0.25,
        tst_hours=4.0,
    )

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (0.0, 0.0, 1.0)
    assert pred_ahi == pytest.approx(0.25)
    assert true_ahi == 0.25


def test_evaluate_single_ahi_record_skips_short_tst_for_ahi_summary():
    record = _ahi_record(
        num_stage_tokens=120,
        truth_segments=[(10, 25)],
        score_segments=[(10, 25, 0.9)],
        true_ahi=8.0,
        tst_hours=1.5,
    )

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (1.0, 0.0, 0.0)
    assert pred_ahi is None
    assert true_ahi is None


def test_evaluate_single_ahi_record_keeps_short_tst_false_positive_when_stage5_marks_sleep():
    record = _ahi_record(
        num_stage_tokens=120,
        truth_segments=[],
        score_segments=[(0, 11, 0.9)],
        true_ahi=0.0,
        tst_hours=1.0,
    )

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (0.0, 1.0, 0.0)
    assert pred_ahi is None
    assert true_ahi is None


def test_evaluate_single_ahi_record_masks_wake_only_false_positive_when_stage5_is_present():
    record = _ahi_record(
        num_stage_tokens=4,
        truth_segments=[],
        score_segments=[(0, 11, 0.9)],
        true_ahi=0.0,
        tst_hours=4.0,
    )
    record["stage5"] = np.array([0, 2, 2, 2], dtype=np.int64)

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (0.0, 0.0, 0.0)
    assert pred_ahi == 0.0
    assert true_ahi == 0.0


def test_evaluate_single_ahi_record_requires_stage5():
    record = _ahi_record(
        num_stage_tokens=4,
        truth_segments=[],
        score_segments=[],
        true_ahi=0.0,
        tst_hours=4.0,
        stage_value=None,
    )

    with pytest.raises(KeyError, match="stage5"):
        _evaluate_single_ahi_record(record, threshold=0.5)


def test_evaluate_single_ahi_record_uses_scalar_ground_truth_ahi_for_summary():
    record = _ahi_record(
        num_stage_tokens=240,
        truth_segments=[(0, 12)],
        score_segments=[],
        true_ahi=7.5,
        tst_hours=4.0,
    )

    detection, pred_ahi, true_ahi = _evaluate_single_ahi_record(record, threshold=0.5)

    assert detection == (0.0, 0.0, 1.0)
    assert pred_ahi == 0.0
    assert true_ahi == 7.5


def test_compute_ahi_event_metrics_reports_perfect_event_and_ahi_scores():
    records = [
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(10, 22)],
            score_segments=[(10, 22, 0.9)],
            true_ahi=0.25,
            tst_hours=4.0,
        ),
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(30, 42), (100, 112)],
            score_segments=[(30, 42, 0.9), (100, 112, 0.9)],
            true_ahi=0.5,
            tst_hours=4.0,
        ),
        _ahi_record(num_stage_tokens=240, truth_segments=[], score_segments=[], true_ahi=0.0, tst_hours=4.0),
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_event_f1"] == 1.0
    assert metrics["ahi_mae"] == 0.0
    assert metrics["ahi_pearson"] == 1.0
    assert metrics["ahi_icc"] == 1.0
    assert metrics["ahi_opt_threshold"] == 0.5


def test_compute_ahi_event_metrics_summary_ahi_aligns_with_scalar_ground_truth_without_changing_detection():
    records = [
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(10, 35)],
            score_segments=[(10, 20, 0.9), (23, 33, 0.9)],
            true_ahi=0.5,
            tst_hours=4.0,
        ),
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(40, 55)],
            score_segments=[(40, 48, 0.9)],
            true_ahi=0.25,
            tst_hours=4.0,
        ),
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 0.5
    assert metrics["ahi_event_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["ahi_mae"] == 0.0
    assert metrics["ahi_pearson"] == pytest.approx(1.0)
    assert metrics["ahi_icc"] == pytest.approx(1.0)
    assert metrics["ahi_opt_threshold"] == 0.5


def test_compute_ahi_event_metrics_uses_inclusive_iou_for_boundary_match():
    records = [
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(0, 9)],
            score_segments=[(0, 18, 0.9)],
            true_ahi=0.25,
            tst_hours=4.0,
        )
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_event_f1"] == 1.0


def test_compute_ahi_event_metrics_aggregates_windows_by_recording():
    records = [
        {
            "path": "rec_a.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 22)],
                score_segments=[(10, 22, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_a.npz",
            "token_start": 240,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(40, 52)],
                score_segments=[(40, 52, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_b.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 22), (100, 112)],
                score_segments=[(10, 22, 0.9), (100, 112, 0.9)],
                true_ahi=1.0,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_b.npz",
            "token_start": 240,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(40, 52), (130, 142)],
                score_segments=[(40, 52, 0.9), (130, 142, 0.9)],
                true_ahi=1.0,
                tst_hours=4.0,
            ),
        },
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_mae"] == 0.0
    assert metrics["ahi_pearson"] == pytest.approx(1.0)


def test_compute_ahi_event_metrics_supports_second_level_masked_stage5_windows():
    second_valid_mask = np.zeros(60, dtype=np.bool_)
    second_valid_mask[:15] = True
    second_valid_mask[30:45] = True
    truth = np.zeros(30, dtype=np.int64)
    truth[5:19] = 1
    score = np.full(30, 0.05, dtype=np.float32)
    score[5:19] = 0.9
    records = [
        {
            "path": "rec_a.npz",
            "token_start": 0,
            "truth": truth,
            "score": score,
            "true_ahi": np.float32(0.25),
            "tst_hours": np.float32(4.0),
            "stage5": np.array([2, 2], dtype=np.int64),
            "second_valid_mask": second_valid_mask,
        }
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_event_f1"] == 1.0
    assert metrics["ahi_mae"] == 0.0


def test_compute_ahi_event_metrics_deduplicates_identical_windows_by_recording():
    records = [
        {
            "path": "rec_a.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 22)],
                score_segments=[(10, 22, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_a.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 22)],
                score_segments=[(10, 22, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_a.npz",
            "token_start": 240,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(40, 52)],
                score_segments=[(40, 52, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_mae"] == 0.0


def test_compute_ahi_event_metrics_deduplicates_duplicate_windows_with_different_padding_layouts():
    truth = np.zeros(30, dtype=np.int64)
    truth[5:19] = 1
    score = np.full(30, 0.05, dtype=np.float32)
    score[5:19] = 0.9
    first_mask = np.zeros(60, dtype=np.bool_)
    first_mask[:30] = True
    second_mask = np.zeros(90, dtype=np.bool_)
    second_mask[:30] = True
    records = [
        {
            "path": "rec_a.npz",
            "token_start": 0,
            "truth": truth,
            "score": score,
            "true_ahi": np.float32(0.25),
            "tst_hours": np.float32(4.0),
            "stage5": np.array([2, 2], dtype=np.int64),
            "second_valid_mask": first_mask,
        },
        {
            "path": "rec_a.npz",
            "token_start": 0,
            "truth": truth,
            "score": score,
            "true_ahi": np.float32(0.25),
            "tst_hours": np.float32(4.0),
            "stage5": np.array([2, 2, -1], dtype=np.int64),
            "second_valid_mask": second_mask,
        },
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_mae"] == 0.0


def test_compute_ahi_event_metrics_uses_first_seen_duplicate_window():
    records = [
        {
            "path": "rec_a.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 22)],
                score_segments=[(10, 22, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_a.npz",
            "token_start": 0,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[],
                score_segments=[],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
        {
            "path": "rec_a.npz",
            "token_start": 240,
            **_ahi_record(
                num_stage_tokens=240,
                truth_segments=[(40, 52)],
                score_segments=[(40, 52, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        },
    ]

    metrics, threshold = compute_ahi_event_metrics(records, threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_precision"] == 1.0
    assert metrics["ahi_event_recall"] == 1.0
    assert metrics["ahi_mae"] == 0.0


def test_compute_ahi_event_metrics_uses_inclusive_clinical_cutoffs(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(prepared_records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([6.0, 16.0, 31.0, 0.0], dtype=np.float32),
            "true_ahi": np.array([5.0, 15.0, 30.0, 0.0], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)

    metrics, threshold = compute_ahi_event_metrics([{}], threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_threshold_5_precision"] == 1.0
    assert metrics["ahi_threshold_5_recall"] == 1.0
    assert metrics["ahi_threshold_5_f1"] == 1.0
    assert metrics["ahi_threshold_15_precision"] == 1.0
    assert metrics["ahi_threshold_15_recall"] == 1.0
    assert metrics["ahi_threshold_15_f1"] == 1.0
    assert metrics["ahi_threshold_30_precision"] == 1.0
    assert metrics["ahi_threshold_30_recall"] == 1.0
    assert metrics["ahi_threshold_30_f1"] == 1.0
    assert metrics["ahi_acc"] == 1.0


def test_select_best_ahi_threshold_uses_pearson_then_mae_tiebreak(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(prepared_records, *, threshold):
        if threshold == 0.1:
            pred = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        else:
            pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": pred,
            "true_ahi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))

    assert threshold == 0.2


def test_select_best_ahi_threshold_allows_single_sample_mae_tiebreak(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(prepared_records, *, threshold):
        if threshold == 0.1:
            pred = np.array([2.0], dtype=np.float32)
        else:
            pred = np.array([1.0], dtype=np.float32)
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": pred,
            "true_ahi": np.array([1.5], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))

    assert threshold == 0.2


def test_select_best_ahi_threshold_prefers_higher_threshold_on_exact_metric_tie(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(prepared_records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "true_ahi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2, 0.3))

    assert threshold == 0.3


def test_select_best_ahi_threshold_rejects_all_skipped_samples(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(prepared_records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([], dtype=np.float32),
            "true_ahi": np.array([], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record(summary_enabled=False)])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Need at least 1 non-skipped sample"):
        select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))


def test_select_best_ahi_threshold_prepares_records_once(monkeypatch: pytest.MonkeyPatch):
    calls = {"prepare": 0, "aggregate": 0}

    def fake_prepare(records):
        calls["prepare"] += 1
        return [_prepared_record()]

    def fake_aggregate(prepared_records, *, threshold):
        calls["aggregate"] += 1
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([threshold], dtype=np.float32),
            "true_ahi": np.array([0.2], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", fake_prepare)
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2, 0.3))

    assert calls["prepare"] == 1
    assert calls["aggregate"] == 3


def test_compute_ahi_event_metrics_fixed_threshold_prepares_records_once(monkeypatch: pytest.MonkeyPatch):
    calls = {"prepare": 0, "aggregate": 0}

    def fake_prepare(records):
        calls["prepare"] += 1
        return [_prepared_record()]

    def fake_aggregate(prepared_records, *, threshold):
        calls["aggregate"] += 1
        assert threshold == 0.5
        return {
            "event_tp": 1.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([0.5], dtype=np.float32),
            "true_ahi": np.array([0.5], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", fake_prepare)
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)

    metrics, threshold = compute_ahi_event_metrics([{}], threshold=0.5)

    assert threshold == 0.5
    assert metrics["ahi_event_f1"] == 1.0
    assert calls["prepare"] == 1
    assert calls["aggregate"] == 1


def test_select_best_ahi_threshold_logs_coarse_progress(monkeypatch: pytest.MonkeyPatch):
    messages: list[str] = []

    def fake_info(message, *args):
        messages.append(message % args if args else message)

    def fake_aggregate(prepared_records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([threshold], dtype=np.float32),
            "true_ahi": np.array([0.2], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", fake_info)

    select_best_ahi_threshold([{}], search_thresholds=AHI_COARSE_THRESHOLD_GRID)

    assert messages[0] == "AHI threshold search start: mode=coarse thresholds=9 recordings=1 eligible=1"
    assert "AHI threshold search progress: 1/9 threshold=0.10" in messages
    assert "AHI threshold search progress: 2/9 threshold=0.20" in messages
    assert "AHI threshold search progress: 9/9 threshold=0.90" in messages
    assert messages[-1].startswith("AHI threshold search done: best=0.20 elapsed=")


def test_select_best_ahi_threshold_logs_sparse_fine_progress(monkeypatch: pytest.MonkeyPatch):
    messages: list[str] = []

    def fake_info(message, *args):
        messages.append(message % args if args else message)

    def fake_aggregate(prepared_records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([0.2], dtype=np.float32),
            "true_ahi": np.array([0.2], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", fake_info)

    select_best_ahi_threshold([{}], search_thresholds=AHI_FINE_THRESHOLD_GRID)

    progress_messages = [message for message in messages if "AHI threshold search progress:" in message]
    assert progress_messages[0] == "AHI threshold search progress: 1/99 threshold=0.01"
    assert progress_messages[1] == "AHI threshold search progress: 10/99 threshold=0.10"
    assert progress_messages[-1] == "AHI threshold search progress: 99/99 threshold=0.99"
    assert len(progress_messages) == 11


@dataclass
class _DummyModelConfig:
    marker: int = 1


def test_ahi_threshold_is_persisted_in_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pl.LightningModule, "on_save_checkpoint", lambda self, checkpoint: None)
    monkeypatch.setattr(pl.LightningModule, "on_load_checkpoint", lambda self, checkpoint: None)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi")
    module.model_config = _DummyModelConfig()
    module.finetune_config = None
    module.model = object()
    module.model_averager = None
    module._ahi_eval_threshold = 0.42

    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["ahi_eval_threshold"] == 0.42

    module._ahi_eval_threshold = None
    module.on_load_checkpoint(checkpoint)
    assert module._ahi_eval_threshold == 0.42


def test_ahi_test_start_requires_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pl.LightningModule, "on_test_start", lambda self: None)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi")
    module._ahi_eval_threshold = None

    with pytest.raises(ValueError, match="ahi_eval_threshold"):
        module.on_test_start()


def test_ahi_test_start_allows_explicit_test_search_without_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pl.LightningModule, "on_test_start", lambda self: None)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = None

    module.on_test_start()


def test_ahi_test_epoch_reuses_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    used: dict[str, float] = {}

    def fake_compute(records, *, threshold=None, **_):
        used["threshold"] = threshold
        return {"ahi_pearson": 0.7, "ahi_opt_threshold": float(threshold)}, float(threshold)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics", fake_compute)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi")
    module._ahi_eval_threshold = 0.37
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records

    module._finalize_epoch("test")

    assert used["threshold"] == 0.37


def test_ahi_test_epoch_searches_requested_threshold_grid(monkeypatch: pytest.MonkeyPatch):
    used: dict[str, object] = {}

    def fake_compute(records, *, threshold=None, search_thresholds=None, **_):
        used["threshold"] = threshold
        used["search_thresholds"] = search_thresholds
        return {"ahi_pearson": 0.7, "ahi_opt_threshold": 0.03}, 0.03

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics", fake_compute)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02, 0.03))
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module.trainer = argparse.Namespace(is_global_zero=False)
    module.current_epoch = 0

    module._finalize_epoch("test")

    assert used["threshold"] is None
    assert used["search_thresholds"] == (0.01, 0.02, 0.03)


def test_ahi_test_epoch_search_falls_back_to_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[float | None, tuple[float, ...] | None]] = []
    messages: list[str] = []

    def fake_compute(records, *, threshold=None, search_thresholds=None, **_):
        calls.append((threshold, search_thresholds))
        if threshold is None:
            raise ValueError(
                "Unable to fit an AHI event threshold from validation records. "
                "Need at least 1 non-skipped sample with TST >= 2h."
            )
        return {"ahi_event_precision": 1.0, "ahi_pearson": np.nan}, float(threshold)

    def fake_info(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics", fake_compute)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.logging.info", fake_info)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = 0.37
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 1.0}],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module.trainer = argparse.Namespace(is_global_zero=False)
    module.current_epoch = 0

    module._finalize_epoch("test")

    assert calls == [(None, (0.01, 0.02)), (0.37, None)]
    assert messages == [
        "AHI threshold search fallback: reusing saved threshold=0.37 because no eligible summary samples were found"
    ]


def test_ahi_test_epoch_uses_broadcast_payload_on_nonzero_rank(monkeypatch: pytest.MonkeyPatch):
    def fake_broadcast(payload, src=0):
        payload[0] = {
            "metrics": {"ahi_pearson": 0.7},
            "eval_threshold": 0.33,
            "error_type": None,
            "error_message": None,
        }

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.broadcast_object_list", fake_broadcast)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("non-zero rank must not compute AHI search")),
    )

    logged: list[tuple[str, float]] = []
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
    }
    module.log = lambda name, value, **kwargs: logged.append((name, value))
    module._gather_ahi_event_records = lambda records: records
    module.trainer = argparse.Namespace(is_global_zero=False)
    module.current_epoch = 0

    module._finalize_epoch("test")

    assert ("test_ahi_pearson", 0.7) in logged


def test_extract_ahi_summary_scatter_arrays_returns_scalar_pairs():
    true_ahi, pred_ahi = extract_ahi_summary_scatter_arrays(
        [
            _ahi_record(
                num_stage_tokens=240,
                truth_segments=[(10, 35)],
                score_segments=[(10, 35, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
            _ahi_record(
                num_stage_tokens=240,
                truth_segments=[(20, 45)],
                score_segments=[(20, 45, 0.9)],
                true_ahi=0.5,
                tst_hours=4.0,
            ),
        ],
        threshold=0.5,
    )

    assert true_ahi.shape == (2,)
    assert pred_ahi.shape == (2,)
    assert np.allclose(true_ahi, np.array([0.5, 0.5], dtype=np.float32))
    assert np.allclose(pred_ahi, np.array([0.25, 0.25], dtype=np.float32))


def test_ahi_val_epoch_logs_summary_scatter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics",
        lambda records, *, threshold=None, **_: ({"ahi_pearson": 0.7}, 0.37 if threshold is None else float(threshold)),
    )
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning.extract_ahi_summary_scatter_arrays",
        lambda records, *, threshold: (np.array([1.0], dtype=np.float32), np.array([1.2], dtype=np.float32)),
    )

    captured: dict[str, object] = {}

    class _DummyVisualizer:
        def log_ahi_summary_scatter(self, **kwargs):
            captured.update(kwargs)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi")
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
        "test": [],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module._eval_visualizer = _DummyVisualizer()
    module.trainer = argparse.Namespace(is_global_zero=True)
    module.current_epoch = 3

    module._finalize_epoch("val")

    assert module._ahi_eval_threshold == 0.37
    assert captured["stage"] == "val"
    assert captured["label_name"] == "ahi"
    assert np.allclose(captured["targets"], np.array([1.0], dtype=np.float32))
    assert np.allclose(captured["preds"], np.array([1.2], dtype=np.float32))


def test_ahi_val_epoch_uses_default_coarse_search_grid(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_compute(records, *, threshold=None, search_thresholds=None, **_):
        captured["threshold"] = threshold
        captured["search_thresholds"] = search_thresholds
        return {"ahi_pearson": 0.7}, 0.3

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_ahi_event_metrics", fake_compute)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning.extract_ahi_summary_scatter_arrays",
        lambda records, *, threshold: (np.array([1.0], dtype=np.float32), np.array([1.2], dtype=np.float32)),
    )

    class _DummyVisualizer:
        def log_ahi_summary_scatter(self, **kwargs):
            return None

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi")
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
        "test": [],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module._eval_visualizer = _DummyVisualizer()
    module.trainer = argparse.Namespace(is_global_zero=True)
    module.current_epoch = 0

    module._finalize_epoch("val")

    assert captured["threshold"] is None
    assert captured["search_thresholds"] == AHI_COARSE_THRESHOLD_GRID


def test_run_inference_allows_ahi_checkpoint_averaging_with_fine_search(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            captured["args"] = args

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = state_dict
            return [], []

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            captured["ckpt_path"] = ckpt_path
            captured["dataloaders"] = dataloaders
            return [{"ahi_pearson": 0.5}]

    monkeypatch.setattr("sleep2vec.infer.apply_finetune_config", lambda args: (_DummyBundle(), _DummyModelConfig()))
    monkeypatch.setattr("sleep2vec.infer._build_inference_loader", lambda args: "loader")
    monkeypatch.setattr("sleep2vec.infer.Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr("sleep2vec.infer.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.infer._init_wandb", lambda args: None)
    monkeypatch.setattr("sleep2vec.infer.select_checkpoints", lambda *args, **kwargs: [Path("a.ckpt"), Path("b.ckpt")])
    monkeypatch.setattr("sleep2vec.infer.average_checkpoints", lambda *args, **kwargs: {"weight": np.array([1.0])})

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=2,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=False,
        results_csv_path=None,
    )

    run_inference(args)

    assert captured["args"].ahi_test_search_thresholds == AHI_FINE_THRESHOLD_GRID
    assert captured["ckpt_path"] is None
    assert captured["dataloaders"] == "loader"
