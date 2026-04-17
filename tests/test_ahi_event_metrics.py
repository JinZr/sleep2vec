from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pytest
import pytorch_lightning as pl

from sleep2vec.infer import run_inference
import sleep2vec.metrics as metrics_mod
from sleep2vec.metrics import (
    _evaluate_single_ahi_record,
    binary_sequence_to_segments,
    compute_ahi_event_metrics,
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
    def fake_aggregate(records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([6.0, 16.0, 31.0, 0.0], dtype=np.float32),
            "true_ahi": np.array([5.0, 15.0, 30.0, 0.0], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_aggregate_ahi_records", fake_aggregate)

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
    def fake_aggregate(records, *, threshold):
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

    monkeypatch.setattr(metrics_mod, "_aggregate_ahi_records", fake_aggregate)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))

    assert threshold == 0.2


def test_select_best_ahi_threshold_allows_single_sample_mae_tiebreak(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(records, *, threshold):
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

    monkeypatch.setattr(metrics_mod, "_aggregate_ahi_records", fake_aggregate)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))

    assert threshold == 0.2


def test_select_best_ahi_threshold_prefers_higher_threshold_on_exact_metric_tie(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "true_ahi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_aggregate_ahi_records", fake_aggregate)

    threshold, _ = select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2, 0.3))

    assert threshold == 0.3


def test_select_best_ahi_threshold_rejects_all_skipped_samples(monkeypatch: pytest.MonkeyPatch):
    def fake_aggregate(records, *, threshold):
        return {
            "event_tp": 0.0,
            "event_fp": 0.0,
            "event_fn": 0.0,
            "pred_ahi": np.array([], dtype=np.float32),
            "true_ahi": np.array([], dtype=np.float32),
        }

    monkeypatch.setattr(metrics_mod, "_aggregate_ahi_records", fake_aggregate)

    with pytest.raises(ValueError, match="Need at least 1 non-skipped sample"):
        select_best_ahi_threshold([{}], search_thresholds=(0.1, 0.2))


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


def test_run_inference_rejects_ahi_checkpoint_averaging():
    args = argparse.Namespace(label_name="ahi", avg_ckpts=2)

    with pytest.raises(ValueError, match="does not support --avg-ckpts > 1"):
        run_inference(args)
