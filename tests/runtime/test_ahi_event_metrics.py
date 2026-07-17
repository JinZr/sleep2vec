from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

import sleep2vec.finetune as finetune
from sleep2vec.finetune import supervised
from sleep2vec.infer import parse_args, run_inference
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
    vectorized_event_stats,
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


def test_vectorized_event_stats_uses_one_to_one_matching():
    tp, fp, fn = vectorized_event_stats([[0, 9], [14, 23]], [[0, 19]], threshold=0.1)
    assert tp == 1.0
    assert fp == 0.0
    assert fn == 1.0


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


def test_summary_only_ahi_aggregate_matches_full_summary_outputs():
    records = [
        _ahi_record(
            num_stage_tokens=240,
            truth_segments=[(40, 55)],
            score_segments=[(40, 48, 0.9), (80, 93, 0.9)],
            true_ahi=0.25,
            tst_hours=4.0,
        ),
        _ahi_record(
            num_stage_tokens=120,
            truth_segments=[(10, 25)],
            score_segments=[(10, 25, 0.9)],
            true_ahi=8.0,
            tst_hours=1.5,
        ),
    ]
    records[0]["stage5"] = np.array([0] + [2] * 239, dtype=np.int64)
    prepared = metrics_mod._prepare_ahi_records(records)

    full = metrics_mod._aggregate_prepared_ahi_records(prepared, threshold=0.5)
    summary = metrics_mod._aggregate_prepared_ahi_summary_records(prepared, threshold=0.5)

    np.testing.assert_array_equal(summary["pred_ahi"], full["pred_ahi"])
    np.testing.assert_array_equal(summary["true_ahi"], full["true_ahi"])


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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Need at least 1 non-skipped sample"):
        select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))


def test_select_best_ahi_threshold_prepares_records_once(monkeypatch: pytest.MonkeyPatch):
    calls = {"prepare": 0, "summary": 0, "aggregate": 0}

    def fake_prepare(records):
        calls["prepare"] += 1
        return [_prepared_record()]

    def fake_summary_aggregate(prepared_records, *, threshold):
        calls["summary"] += 1
        return {
            "pred_ahi": np.array([threshold], dtype=np.float32),
            "true_ahi": np.array([0.2], dtype=np.float32),
        }

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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_summary_aggregate)
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", lambda *args, **kwargs: None)

    select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2, 0.3))

    assert calls["prepare"] == 1
    assert calls["summary"] == 3
    assert calls["aggregate"] == 1


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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
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
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_summary_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod, "_aggregate_prepared_ahi_records", fake_aggregate)
    monkeypatch.setattr(metrics_mod.logging, "info", fake_info)

    select_best_ahi_threshold([{}], search_thresholds=AHI_FINE_THRESHOLD_GRID)

    progress_messages = [message for message in messages if "AHI threshold search progress:" in message]
    assert progress_messages[0] == "AHI threshold search progress: 10/99 threshold=0.10"
    assert progress_messages[1] == "AHI threshold search progress: 20/99 threshold=0.20"
    assert progress_messages[-1] == "AHI threshold search progress: 90/99 threshold=0.90"
    assert len(progress_messages) == 9


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


def test_ahi_test_start_rejects_legacy_test_search_without_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pl.LightningModule, "on_test_start", lambda self: None)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = None

    with pytest.raises(ValueError, match="ahi_eval_threshold"):
        module.on_test_start()


def test_ahi_test_epoch_reuses_saved_threshold(monkeypatch: pytest.MonkeyPatch):
    used: dict[str, float] = {}

    def fake_compute(prepared_records, *, threshold=None, **_):
        used["threshold"] = threshold
        return {"ahi_pearson": 0.7, "ahi_opt_threshold": float(threshold)}, float(threshold)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._compute_ahi_event_metrics_from_prepared", fake_compute)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning._aggregate_prepared_ahi_records",
        lambda prepared_records, *, threshold: {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([0.7], dtype=np.float32),
            "true_ahi": np.array([0.7], dtype=np.float32),
        },
    )

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


def test_ahi_test_epoch_ignores_legacy_test_search_thresholds(monkeypatch: pytest.MonkeyPatch):
    used: dict[str, object] = {}

    def fake_compute(prepared_records, *, threshold=None, search_thresholds=None, **_):
        used["threshold"] = threshold
        used["search_thresholds"] = search_thresholds
        return {"ahi_pearson": 0.7, "ahi_opt_threshold": float(threshold)}, float(threshold)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._compute_ahi_event_metrics_from_prepared", fake_compute)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning._aggregate_prepared_ahi_records",
        lambda prepared_records, *, threshold: {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([0.7], dtype=np.float32),
            "true_ahi": np.array([0.7], dtype=np.float32),
        },
    )

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02, 0.03))
    module._ahi_eval_threshold = 0.37
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=False, current_epoch=0)

    module._finalize_epoch("test")

    assert used["threshold"] == 0.37
    assert used["search_thresholds"] is None


def test_supervised_does_not_inject_ahi_test_search_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            captured["has_ahi_test_search_thresholds"] = hasattr(args_ns, "ahi_test_search_thresholds")
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: None)

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert captured["has_ahi_test_search_thresholds"] is False


@pytest.mark.parametrize("module_name", ["sleep2vec.finetune", "sleep2vec2.finetune", "sleep2expert.finetune"])
def test_supervised_separates_periodic_and_best_checkpoint_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str
):
    finetune_mod = importlib.import_module(module_name)
    captured: dict[str, object] = {}
    checkpoints: list[object] = []
    copies: list[tuple[object, object]] = []

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyModel:
        moe_finetune_status = {}

        def moe_finetune_hparams(self):
            return {}

        def moe_finetune_param_group_rows(self):
            return []

    class _DummyLogger:
        experiment = argparse.Namespace(log=lambda *args, **kwargs: None)

        def log_hyperparams(self, *args, **kwargs):
            return None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.dirpath = kwargs["dirpath"]
            self.best_model_path = str(tmp_path / f"{len(checkpoints)}.ckpt")
            checkpoints.append(self)

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["callbacks"] = kwargs["callbacks"]
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            captured["ckpt_path"] = kwargs["ckpt_path"]
            return [{"metric": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_roc_auc",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=5,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="custom",
    )

    monkeypatch.setattr(finetune_mod, "persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr(finetune_mod, "prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr(finetune_mod, "Sleep2vecFinetuning", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(finetune_mod, "WandbLogger", lambda *args, **kwargs: _DummyLogger())
    monkeypatch.setattr(finetune_mod, "EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr(finetune_mod, "LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr(finetune_mod, "ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr(finetune_mod.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(finetune_mod.shutil, "copy2", lambda *args, **kwargs: copies.append((args[0], args[1])))
    monkeypatch.setattr(finetune_mod, "save_result_csv", lambda *args, **kwargs: None)
    if hasattr(finetune_mod, "is_rank_zero_process"):
        monkeypatch.setattr(finetune_mod, "is_rank_zero_process", lambda: False)

    finetune_mod.supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert len(checkpoints) == 2
    periodic_checkpoint, best_checkpoint = checkpoints
    assert periodic_checkpoint.kwargs["save_top_k"] == -1
    assert periodic_checkpoint.kwargs["save_last"] is True
    assert periodic_checkpoint.kwargs["every_n_epochs"] == 5
    assert periodic_checkpoint.kwargs["filename"] == "{epoch:02d}"
    assert "every_n_epochs" not in best_checkpoint.kwargs
    assert best_checkpoint.kwargs["monitor"] == "val_roc_auc"
    assert best_checkpoint.kwargs["mode"] == "max"
    assert best_checkpoint.kwargs["save_top_k"] == 1
    assert best_checkpoint.kwargs["save_last"] is False
    assert best_checkpoint.kwargs["filename"] == "best-{epoch:02d}"
    assert best_checkpoint.kwargs["save_on_train_epoch_end"] is False
    assert periodic_checkpoint in captured["callbacks"]
    assert best_checkpoint in captured["callbacks"]
    assert captured["ckpt_path"] == best_checkpoint.best_model_path
    assert copies == [(best_checkpoint.best_model_path, Path(best_checkpoint.dirpath) / "best.ckpt")]


def test_supervised_finishes_owned_wandb_run_after_writing_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    created_run = object()

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    def _build_logger(*args, **kwargs):
        finetune.wandb.run = created_run
        return object()

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", _build_logger)
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: events.append("csv"))
    monkeypatch.setattr("sleep2vec.finetune.wandb.run", None, raising=False)
    monkeypatch.setattr("sleep2vec.finetune.wandb.finish", lambda: events.append("finish"))

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert events == ["csv", "finish"]


def test_supervised_does_not_finish_preexisting_wandb_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    preexisting_run = object()

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: events.append("csv"))
    monkeypatch.setattr("sleep2vec.finetune.wandb.run", preexisting_run, raising=False)
    monkeypatch.setattr("sleep2vec.finetune.wandb.finish", lambda: events.append("finish"))

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert events == ["csv"]


def test_supervised_raises_wandb_finish_failure_after_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    created_run = object()

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    def _build_logger(*args, **kwargs):
        finetune.wandb.run = created_run
        return object()

    def _finish():
        events.append("finish")
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", _build_logger)
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: events.append("csv"))
    monkeypatch.setattr("sleep2vec.finetune.wandb.run", None, raising=False)
    monkeypatch.setattr("sleep2vec.finetune.wandb.finish", _finish)
    monkeypatch.setattr(
        "sleep2vec.finetune.logging.warning",
        lambda msg, *args: events.append(msg % args),
    )

    with pytest.raises(RuntimeError, match="cleanup failure"):
        supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert events == ["csv", "finish"]


def test_supervised_preserves_primary_error_when_wandb_finish_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    created_run = object()

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            raise RuntimeError("primary test failure")

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    def _build_logger(*args, **kwargs):
        finetune.wandb.run = created_run
        return object()

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", _build_logger)
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.wandb.run", None, raising=False)
    monkeypatch.setattr(
        "sleep2vec.finetune.wandb.finish",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failure")),
    )

    with pytest.raises(RuntimeError, match="primary test failure"):
        supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))


def test_supervised_epochs_zero_preserves_ckpt_path_without_test_search_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.best_model_path = ""
            self.dirpath = str(tmp_path / "checkpoints")

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            raise AssertionError("epochs=0 must not call fit")

        def test(self, *args, **kwargs):
            captured["has_ahi_test_search_thresholds"] = hasattr(args_ns, "ahi_test_search_thresholds")
            captured["ckpt_path"] = kwargs["ckpt_path"]
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=0,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="manual.ckpt",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: None)

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    assert captured["has_ahi_test_search_thresholds"] is False
    assert captured["ckpt_path"] == "manual.ckpt"


def test_supervised_uses_custom_progress_bar_for_distributed_ahi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from sleep2vec.callbacks.progress_bar import DistributedAHIRichProgressBar, DistributedAHITQDMProgressBar

    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["callbacks"] = kwargs["callbacks"]
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return [{"ahi_pearson": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_ahi_pearson",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0, 1, 2, 3],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="ahi",
    )

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: None)

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    callbacks = captured["callbacks"]
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
    assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    assert any(isinstance(cb, LearningRateMonitor) for cb in callbacks)
    assert any(isinstance(cb, (DistributedAHIRichProgressBar, DistributedAHITQDMProgressBar)) for cb in callbacks)


def test_supervised_leaves_nonahi_on_default_progress_bar_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from sleep2vec.callbacks.progress_bar import DistributedAHIRichProgressBar, DistributedAHITQDMProgressBar

    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        model: object
        averaging: object = None
        finetune: object = None

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["callbacks"] = kwargs["callbacks"]
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return [{"accuracy": 0.5}]

    args_ns = argparse.Namespace(
        version="unit-test",
        monitor="val_accuracy",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0, 1, 2, 3],
        epochs=1,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="stage5",
    )

    monkeypatch.setattr("sleep2vec.finetune.persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr("sleep2vec.finetune.Sleep2vecFinetuning", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr("sleep2vec.finetune.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.finetune.shutil.copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr("sleep2vec.finetune.save_result_csv", lambda *args, **kwargs: None)

    supervised(args_ns, _DummyBundle(model=_DummyModelConfig()))

    callbacks = captured["callbacks"]
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
    assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    assert not any(isinstance(cb, (DistributedAHIRichProgressBar, DistributedAHITQDMProgressBar)) for cb in callbacks)


def test_ahi_test_epoch_uses_saved_threshold_without_fallback_search(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[float | None, tuple[float, ...] | None]] = []

    def fake_compute(prepared_records, *, threshold=None, search_thresholds=None, **_):
        calls.append((threshold, search_thresholds))
        return {"ahi_event_precision": 1.0, "ahi_pearson": np.nan}, float(threshold)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._compute_ahi_event_metrics_from_prepared", fake_compute)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning._aggregate_prepared_ahi_records",
        lambda prepared_records, *, threshold: {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([0.7], dtype=np.float32),
            "true_ahi": np.array([0.7], dtype=np.float32),
        },
    )

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = 0.37
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 1.0}],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=False, current_epoch=0)

    module._finalize_epoch("test")

    assert calls == [(0.37, None)]


def test_ahi_test_epoch_computes_metrics_on_nonzero_rank_without_broadcast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning.dist.broadcast_object_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("AHI metrics must not use broadcast")),
    )

    logged: list[tuple[str, float]] = []
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", ahi_test_search_thresholds=(0.01, 0.02))
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [],
        "test": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
    }
    module.log = lambda name, value, **kwargs: logged.append((name, value))
    module._gather_ahi_event_records = lambda records: records
    module._compute_ahi_metrics_for_stage = lambda stage, records: ({"ahi_pearson": 0.7}, 0.33, None)
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=False, current_epoch=0)

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
    captured: dict[str, object] = {}

    class _DummyVisualizer:
        def log_ahi_summary_scatter(self, **kwargs):
            captured.update(kwargs)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi")
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
        "test": [],
    }
    module.log = lambda *args, **kwargs: None
    module._gather_ahi_event_records = lambda records: records

    def fake_compute(stage, records):
        module._ahi_eval_threshold = 0.37
        return (
            {"ahi_pearson": 0.7},
            0.37,
            (np.array([1.0], dtype=np.float32), np.array([1.2], dtype=np.float32)),
        )

    module._compute_ahi_metrics_for_stage = fake_compute
    module._eval_visualizer = _DummyVisualizer()
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=True, current_epoch=3)

    module._finalize_epoch("val")

    assert module._ahi_eval_threshold == 0.37
    assert captured["stage"] == "val"
    assert captured["label_name"] == "ahi"
    assert np.allclose(captured["targets"], np.array([1.0], dtype=np.float32))
    assert np.allclose(captured["preds"], np.array([1.2], dtype=np.float32))


def test_ahi_val_epoch_uses_default_fine_search_grid(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_compute(prepared_records, *, threshold=None, search_thresholds=None, **_):
        captured["threshold"] = threshold
        captured["search_thresholds"] = search_thresholds
        return {"ahi_pearson": 0.7}, 0.3

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._prepare_ahi_records", lambda records: [_prepared_record()])
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning._compute_ahi_event_metrics_from_prepared", fake_compute)
    monkeypatch.setattr(
        "sleep2vec.sleep2vec_finetuning._aggregate_prepared_ahi_records",
        lambda prepared_records, *, threshold: {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([1.2], dtype=np.float32),
            "true_ahi": np.array([1.0], dtype=np.float32),
        },
    )

    class _DummyVisualizer:
        def log_ahi_summary_scatter(self, **kwargs):
            return None

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
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
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=True, current_epoch=0)

    module._finalize_epoch("val")

    assert captured["threshold"] is None
    assert captured["search_thresholds"] == AHI_FINE_THRESHOLD_GRID


def test_ahi_val_shared_step_accumulates_eval_loss_without_step_logging():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max")
    module.model = lambda batch: torch.zeros((1, 1), dtype=torch.float32)
    module._compute_loss = lambda logits, batch: (torch.tensor(1.5), 3)
    module._extract_ahi_event_records = lambda batch, logits: [{"truth": np.array([0], dtype=np.int64)}]
    module._stage_outputs = {"train": [], "val": [], "test": []}
    module._eval_loss_sums = {"val": 0.0, "test": 0.0}
    module._eval_loss_counts = {"val": 0, "test": 0}
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)

    module._shared_step({}, stage="val")

    assert logged == []
    assert module._eval_loss_sums["val"] == pytest.approx(4.5)
    assert module._eval_loss_counts["val"] == 3
    assert len(module._stage_outputs["val"]) == 1


def test_ahi_train_shared_step_accumulates_confusion_counts_without_storing_epoch_outputs():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max")
    module.model = lambda batch: torch.tensor([[0.9, -0.9, 0.1]], dtype=torch.float32)
    module._compute_loss = lambda logits, batch: (torch.tensor(1.0), 2)
    module._stage_outputs = {"train": [], "val": [], "test": []}
    module._eval_loss_sums = {"val": 0.0, "test": 0.0}
    module._eval_loss_counts = {"val": 0, "test": 0}
    module._ahi_train_pointwise_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    module.log = lambda *args, **kwargs: None
    module._get_targets = lambda batch: torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)

    module._shared_step({}, stage="train")

    assert module._ahi_train_pointwise_counts == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert module._stage_outputs["train"] == []


def test_ahi_train_epoch_logs_reduced_pointwise_metrics(monkeypatch: pytest.MonkeyPatch):
    reduce_calls: list[tuple[tuple[float, float, float, float], str]] = []

    def fake_reduce(tensor, reduce_op="mean"):
        reduce_calls.append((tuple(float(v.item()) for v in tensor), str(reduce_op)))
        return torch.tensor([6.0, 2.0, 10.0, 2.0], dtype=tensor.dtype)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max", device="cpu")
    module._stage_outputs = {"train": [], "val": [], "test": []}
    module._eval_loss_sums = {"val": 0.0, "test": 0.0}
    module._eval_loss_counts = {"val": 0, "test": 0}
    module._ahi_train_pointwise_counts = {"tp": 1, "fp": 2, "tn": 3, "fn": 4}
    logged: list[tuple[str, float, bool]] = []
    module.log = lambda name, value, **kwargs: logged.append((name, float(value), bool(kwargs.get("sync_dist"))))
    module.__dict__["_trainer"] = argparse.Namespace(
        is_global_zero=True,
        current_epoch=0,
        strategy=argparse.Namespace(reduce=fake_reduce, barrier=lambda name: None),
    )

    module._finalize_epoch("train")

    assert reduce_calls == [((1.0, 2.0, 3.0, 4.0), "sum")]
    assert ("train_ahi_pointwise_accuracy", 0.8, True) in logged
    assert ("train_ahi_pointwise_precision", 0.75, True) in logged
    assert ("train_ahi_pointwise_recall", 0.75, True) in logged
    assert ("train_ahi_pointwise_f1", 0.75, True) in logged
    assert all(name != "train_ahi_pointwise_roc_auc" for name, _, _ in logged)
    assert module._ahi_train_pointwise_counts == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def test_ahi_train_epoch_nonzero_rank_still_logs_reduced_metrics(monkeypatch: pytest.MonkeyPatch):
    def fake_reduce(tensor, reduce_op="mean"):
        return torch.tensor([6.0, 2.0, 10.0, 2.0], dtype=tensor.dtype)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max", device="cpu")
    module._stage_outputs = {"train": [], "val": [], "test": []}
    module._eval_loss_sums = {"val": 0.0, "test": 0.0}
    module._eval_loss_counts = {"val": 0, "test": 0}
    module._ahi_train_pointwise_counts = {"tp": 1, "fp": 2, "tn": 3, "fn": 4}
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)
    module.__dict__["_trainer"] = argparse.Namespace(
        is_global_zero=False,
        current_epoch=0,
        strategy=argparse.Namespace(reduce=fake_reduce, barrier=lambda name: None),
    )

    module._finalize_epoch("train")

    assert "train_ahi_pointwise_accuracy" in logged
    assert "train_ahi_pointwise_f1" in logged
    assert module._ahi_train_pointwise_counts == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def test_ahi_train_epoch_omits_roc_auc_from_reduced_metric_set(monkeypatch: pytest.MonkeyPatch):
    def fake_reduce(tensor, reduce_op="mean"):
        return torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=tensor.dtype)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max", device="cpu")
    module._stage_outputs = {"train": [], "val": [], "test": []}
    module._eval_loss_sums = {"val": 0.0, "test": 0.0}
    module._eval_loss_counts = {"val": 0, "test": 0}
    module._ahi_train_pointwise_counts = {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)
    module.__dict__["_trainer"] = argparse.Namespace(
        is_global_zero=True,
        current_epoch=0,
        strategy=argparse.Namespace(reduce=fake_reduce),
    )

    module._finalize_epoch("train")

    assert all(name != "train_ahi_pointwise_roc_auc" for name in logged)


def test_ahi_val_epoch_logs_reduced_eval_loss_before_event_metrics(monkeypatch: pytest.MonkeyPatch):
    reduce_calls: list[tuple[tuple[float, float], str]] = []

    def fake_reduce(tensor, reduce_op="mean"):
        reduce_calls.append(((float(tensor[0].item()), float(tensor[1].item())), str(reduce_op)))
        return torch.tensor([12.0, 4.0], dtype=tensor.dtype)

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.dist.is_initialized", lambda: True)

    captured: dict[str, object] = {}

    class _DummyVisualizer:
        def log_ahi_summary_scatter(self, **kwargs):
            captured.update(kwargs)

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="ahi", monitor="val_ahi_pearson", monitor_mod="max", device="cpu")
    module._ahi_eval_threshold = None
    module._stage_outputs = {
        "train": [],
        "val": [{"truth": np.array([0]), "score": np.array([0.1]), "true_ahi": 0.0, "tst_hours": 4.0}],
        "test": [],
    }
    module._eval_loss_sums = {"val": 6.0, "test": 0.0}
    module._eval_loss_counts = {"val": 2, "test": 0}
    module._gather_ahi_event_records = lambda records: records
    module._compute_ahi_metrics_for_stage = lambda stage, records: (
        {"ahi_pearson": 0.7},
        0.37,
        (np.array([1.0], dtype=np.float32), np.array([1.2], dtype=np.float32)),
    )
    module._eval_visualizer = _DummyVisualizer()
    logged: list[tuple[str, float, bool]] = []
    module.log = lambda name, value, **kwargs: logged.append((name, float(value), bool(kwargs.get("sync_dist"))))
    module.__dict__["_trainer"] = argparse.Namespace(
        is_global_zero=True,
        current_epoch=0,
        strategy=argparse.Namespace(reduce=fake_reduce, barrier=lambda name: None),
    )

    module._finalize_epoch("val")

    assert reduce_calls == [((6.0, 2.0), "sum")]
    assert logged[0] == ("val_loss", 3.0, False)
    assert ("val_ahi_pearson", 0.7, False) in logged
    assert module._eval_loss_sums["val"] == 0.0
    assert module._eval_loss_counts["val"] == 0
    assert captured["stage"] == "val"


def test_run_inference_rejects_ahi_checkpoint_averaging(monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.setattr("sleep2vec.infer.apply_finetune_config", lambda args: (_DummyBundle(), _DummyModelConfig()))
    monkeypatch.setattr("sleep2vec.infer.Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr(
        "sleep2vec.infer.select_checkpoints",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AHI averaging must fail before checkpoint selection")
        ),
    )
    monkeypatch.setattr(
        "sleep2vec.infer.average_checkpoints",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("AHI averaging must fail before averaging")),
    )

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

    with pytest.raises(ValueError, match="does not support average checkpoints"):
        run_inference(args)

    assert "args" not in captured


def test_run_inference_uses_single_ahi_checkpoint_without_search_injection(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            captured["args"] = args

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

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=1,
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

    assert not hasattr(captured["args"], "ahi_test_search_thresholds")
    assert captured["ckpt_path"] == "/tmp/model.ckpt"
    assert captured["dataloaders"] == "loader"


def test_infer_parse_args_accepts_inference_preset_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sleep2vec.infer",
            "--config",
            "config.yaml",
            "--ckpt-path",
            "best.ckpt",
            "--label-name",
            "ahi",
            "--inference-preset-path",
            "preset.pkl",
        ],
    )

    args = parse_args()

    assert args.inference_preset_path == Path("preset.pkl")


def test_run_inference_applies_inference_preset_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}
    config_preset = tmp_path / "config.pkl"
    override_preset = tmp_path / "override.pkl"

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            captured["module_preset_path"] = args.finetune_preset_path

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            return [{"ahi_pearson": 0.5}]

    def _apply_config(args):
        args.finetune_preset_path = config_preset
        return _DummyBundle(), _DummyModelConfig()

    def _build_loader(args):
        captured["loader_preset_path"] = args.finetune_preset_path
        return "loader"

    monkeypatch.setattr("sleep2vec.infer.apply_finetune_config", _apply_config)
    monkeypatch.setattr("sleep2vec.infer._build_inference_loader", _build_loader)
    monkeypatch.setattr("sleep2vec.infer.Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr("sleep2vec.infer.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.infer._init_wandb", lambda args: None)

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=1,
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
        inference_preset_path=override_preset,
    )

    run_inference(args)

    assert captured["loader_preset_path"] == override_preset
    assert captured["module_preset_path"] == override_preset


def test_run_inference_rejects_kaldi_inference_preset_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    override_preset = tmp_path / "override.pkl"

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    def _apply_config(args):
        args.data_backend = "kaldi"
        args.finetune_preset_path = None
        args.kaldi_data_root = tmp_path / "kaldi"
        args.kaldi_manifest = tmp_path / "kaldi" / "manifest.json"
        return _DummyBundle(), _DummyModelConfig()

    monkeypatch.setattr("sleep2vec.infer.apply_finetune_config", _apply_config)

    args = argparse.Namespace(
        label_name="stage5",
        avg_ckpts=1,
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
        inference_preset_path=override_preset,
    )

    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        run_inference(args)


def test_run_inference_preserves_primary_error_when_wandb_finish_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    warnings: list[str] = []
    created_run = object()

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            pass

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            raise RuntimeError("primary inference failure")

    monkeypatch.setattr("sleep2vec.infer.apply_finetune_config", lambda args: (_DummyBundle(), _DummyModelConfig()))
    monkeypatch.setattr("sleep2vec.infer._build_inference_loader", lambda args: "loader")
    monkeypatch.setattr("sleep2vec.infer.Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr("sleep2vec.infer.pl.Trainer", _DummyTrainer)
    monkeypatch.setattr("sleep2vec.infer._init_wandb", lambda args: created_run)
    monkeypatch.setattr(
        "sleep2vec.infer.wandb.finish",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failure")),
    )
    monkeypatch.setattr(
        "sleep2vec.infer.logging.warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=1,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=True,
        results_csv_path=tmp_path / "results.csv",
    )

    with pytest.raises(RuntimeError, match="primary inference failure"):
        run_inference(args)

    assert warnings == ["wandb.finish() failed during inference cleanup: cleanup failure"]
