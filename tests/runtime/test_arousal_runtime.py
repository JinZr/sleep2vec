from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sleep2vec.arousal_metrics import (
    AROUSAL_SUBTYPES,
    AROUSAL_THRESHOLD_GRID,
    compute_arousal_metrics,
    evaluate_arousal_record,
    merge_arousal_window_records,
    select_arousal_thresholds,
    validate_arousal_thresholds,
)
from sleep2vec.sleep2vec_inference import build_arousal_prediction_rows


def _record(
    *,
    truth: np.ndarray,
    score: np.ndarray | None = None,
    tst_hours: float = 3.0,
    path: str | None = None,
    token_start: int | None = None,
) -> dict[str, object]:
    score = truth.astype(np.float32) * 0.8 + 0.1 if score is None else np.asarray(score, dtype=np.float32)
    subtype_counts = [len(_segments_from_binary(truth[:, subtype_idx])) for subtype_idx in range(len(AROUSAL_SUBTYPES))]
    total_count = len(_segments_from_binary(truth.any(axis=1)))
    record: dict[str, object] = {
        "truth": np.asarray(truth),
        "score": score,
        "tst_hours": tst_hours,
        "arousal_res_index_per_hour": subtype_counts[0] / tst_hours,
        "arousal_spont_index_per_hour": subtype_counts[1] / tst_hours,
        "arousal_limb_index_per_hour": subtype_counts[2] / tst_hours,
        "arousal_plm_index_per_hour": subtype_counts[3] / tst_hours,
        "arousal_index_per_hour": total_count / tst_hours,
    }
    if path is not None:
        record["path"] = path
    if token_start is not None:
        record["token_start"] = token_start
        record["n_tokens"] = truth.shape[0] // 30
    return record


def _segments_from_binary(binary: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(binary)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [(int(group[0]), int(group[-1])) for group in groups]


def test_select_arousal_thresholds_fits_subtypes_independently_and_ties_high(caplog):
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:3, 0] = 1
    score = np.full((30, 4), 0.1, dtype=np.float32)
    score[0:3, 0] = 0.9
    score[10:13, 0] = 0.6

    with caplog.at_level(logging.WARNING):
        thresholds, fallback = select_arousal_thresholds(
            [_record(truth=truth, score=score)], search_thresholds=(0.5, 0.7, 0.8)
        )

    assert thresholds == {"RES": 0.8, "SPONT": 0.5, "Limb": 0.5, "PLM": 0.5}
    assert fallback == ["SPONT", "Limb", "PLM"]
    assert "no ground-truth SPONT events" in caplog.text


def test_arousal_metrics_keep_overlapping_subtypes_and_union_total_once():
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[4:8, :] = 1
    metrics, thresholds, fallback = compute_arousal_metrics(
        [_record(truth=truth)], thresholds={name: 0.5 for name in AROUSAL_SUBTYPES}
    )

    assert fallback == []
    assert thresholds == {name: 0.5 for name in AROUSAL_SUBTYPES}
    assert metrics["arousal_subtype_macro_event_f1"] == 1.0
    assert metrics["arousal_total_event_f1"] == 1.0
    assert metrics["arousal_total_event_support"] == 1.0
    assert all(metrics[f"arousal_{slug}_event_support"] == 1.0 for slug in ("res", "spont", "limb", "plm"))


def test_no_positive_subtypes_contribute_zero_event_f1_but_nan_auprc():
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:3, 0] = 1
    metrics, _, _ = compute_arousal_metrics([_record(truth=truth)], thresholds={name: 0.5 for name in AROUSAL_SUBTYPES})

    assert metrics["arousal_res_event_f1"] == 1.0
    assert metrics["arousal_spont_event_f1"] == 0.0
    assert np.isnan(metrics["arousal_spont_pointwise_auprc"])
    assert metrics["arousal_subtype_macro_event_f1"] == 0.25
    assert metrics["arousal_subtype_macro_pointwise_auprc"] == 1.0


def test_arousal_postprocessing_uses_strict_threshold_removes_short_runs_and_does_not_gap_merge():
    truth = np.zeros((30, 4), dtype=np.int64)
    score = np.zeros((30, 4), dtype=np.float32)
    score[0:2, 0] = 0.9
    score[5:8, 0] = 0.5
    score[10:13, 0] = 0.9
    score[14:17, 0] = 0.9
    evaluated = evaluate_arousal_record(_record(truth=truth, score=score), {name: 0.5 for name in AROUSAL_SUBTYPES})

    assert evaluated["prediction"][:, 0].sum() == 6
    assert evaluated["pred_segments"][0] == [[10, 12], [14, 16]]


def test_arousal_metrics_keep_authoritative_short_truth_while_removing_short_prediction():
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:2, 0] = 1

    metrics, _, _ = compute_arousal_metrics([_record(truth=truth)], thresholds={name: 0.5 for name in AROUSAL_SUBTYPES})

    assert metrics["arousal_res_event_support"] == 1.0
    assert metrics["arousal_res_event_recall"] == 0.0
    assert metrics["arousal_res_pointwise_recall"] == 0.0
    assert metrics["arousal_res_pointwise_auprc"] == 1.0
    assert metrics["arousal_total_event_support"] == 1.0


def test_arousal_metrics_reject_fractional_truth_before_integer_conversion():
    truth = np.zeros((30, 4), dtype=np.float32)
    truth[0, 0] = 0.5

    with pytest.raises(ValueError, match="finite, exact 0/1"):
        compute_arousal_metrics([_record(truth=truth)], thresholds={name: 0.5 for name in AROUSAL_SUBTYPES})


def test_arousal_event_matching_is_one_to_one_any_positive_overlap():
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:3, 0] = 1
    truth[6:9, 0] = 1
    score = np.zeros((30, 4), dtype=np.float32)
    score[0:9, 0] = 0.9
    metrics, _, _ = compute_arousal_metrics(
        [_record(truth=truth, score=score)], thresholds={name: 0.5 for name in AROUSAL_SUBTYPES}
    )

    assert metrics["arousal_res_event_precision"] == 1.0
    assert metrics["arousal_res_event_recall"] == 0.5
    assert metrics["arousal_res_event_f1"] == pytest.approx(2.0 / 3.0)


def test_arousal_ari_summary_excludes_short_tst_but_detection_keeps_it():
    short_truth = np.zeros((30, 4), dtype=np.int64)
    short_truth[0:3, 0] = 1
    short_score = np.zeros((30, 4), dtype=np.float32)
    eligible_truth = np.zeros((30, 4), dtype=np.int64)
    records = [
        _record(truth=short_truth, score=short_score, tst_hours=1.0),
        _record(truth=eligible_truth, score=eligible_truth, tst_hours=2.0),
    ]

    metrics, _, _ = compute_arousal_metrics(records, thresholds={name: 0.5 for name in AROUSAL_SUBTYPES})

    assert metrics["arousal_res_event_support"] == 1.0
    assert metrics["arousal_res_event_recall"] == 0.0
    assert metrics["arousal_res_index_per_hour_mae"] == 0.0


def test_merge_arousal_windows_preserves_token_offsets_and_deduplicates_ddp_padding():
    first_truth = np.zeros((30, 4), dtype=np.int64)
    second_truth = np.zeros((30, 4), dtype=np.int64)
    first_truth[0:3, 0] = 1
    second_truth[10:13, 1] = 1
    first = _record(truth=first_truth, path="night.npz", token_start=0)
    second = _record(truth=second_truth, path="night.npz", token_start=1)
    for key in (
        "arousal_res_index_per_hour",
        "arousal_spont_index_per_hour",
        "arousal_limb_index_per_hour",
        "arousal_plm_index_per_hour",
        "arousal_index_per_hour",
    ):
        second[key] = first[key]

    merged = merge_arousal_window_records([first, dict(first), second])

    assert len(merged) == 1
    assert merged[0]["truth"].shape == (60, 4)
    assert merged[0]["token_starts"] == [0, 1]
    assert merged[0]["n_windows"] == 2


def test_merge_arousal_windows_rejects_noncontiguous_token_offsets():
    truth = np.zeros((30, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="not contiguous"):
        merge_arousal_window_records(
            [
                _record(truth=truth, path="night.npz", token_start=0),
                _record(truth=truth, path="night.npz", token_start=2),
            ]
        )


def test_arousal_prediction_row_contains_protocol_and_summary_fields():
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:3, 0] = 1
    record = _record(truth=truth, path="night.npz", token_start=0)

    rows = build_arousal_prediction_rows([record], {name: 0.5 for name in AROUSAL_SUBTYPES})

    assert len(rows) == 1
    assert np.asarray(rows[0]["groundtruth"]).shape == (30, 4)
    assert np.asarray(rows[0]["prob"]).shape == (30, 4)
    assert rows[0]["arousal_subtypes"] == ["RES", "SPONT", "Limb", "PLM"]
    assert rows[0]["arousal_res_threshold"] == 0.5
    assert rows[0]["true_arousal_res_event_count"] == 1
    assert rows[0]["pred_arousal_event_count"] == 1
    assert rows[0]["token_starts"] == [0]


def test_extract_arousal_records_restores_second_major_subtype_shape():
    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    labels = torch.zeros((1, 1, 120), dtype=torch.float32)
    labels[0, 0, 0] = 1.0
    labels[0, 0, 1] = 1.0
    logits = torch.zeros_like(labels)
    batch = {
        "tokens": {"arousal": labels, "stage5": torch.tensor([[2.0]])},
        "metadata": {
            "path": ["night.npz"],
            "tst": torch.tensor([3.0]),
            "arousal_index_per_hour": torch.tensor([1.0]),
            "arousal_res_index_per_hour": torch.tensor([1.0]),
            "arousal_spont_index_per_hour": torch.tensor([1.0]),
            "arousal_limb_index_per_hour": torch.tensor([0.0]),
            "arousal_plm_index_per_hour": torch.tensor([0.0]),
        },
        "token_start": torch.tensor([7]),
    }

    records = module._extract_arousal_event_records(batch, logits)

    assert records[0]["truth"].shape == (30, 4)
    assert records[0]["truth"][0].tolist() == [1, 1, 0, 0]
    assert records[0]["token_start"] == 7
    assert records[0]["n_tokens"] == 1


def test_extract_arousal_records_does_not_hide_fractional_truth():
    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    labels = torch.zeros((1, 1, 120), dtype=torch.float32)
    labels[0, 0, 0] = 0.5
    batch = {
        "tokens": {"arousal": labels, "stage5": torch.tensor([[2.0]])},
        "metadata": {
            "path": ["night.npz"],
            "tst": torch.tensor([3.0]),
            "arousal_index_per_hour": torch.tensor([0.0]),
            "arousal_res_index_per_hour": torch.tensor([0.0]),
            "arousal_spont_index_per_hour": torch.tensor([0.0]),
            "arousal_limb_index_per_hour": torch.tensor([0.0]),
            "arousal_plm_index_per_hour": torch.tensor([0.0]),
        },
        "token_start": torch.tensor([0]),
    }

    records = module._extract_arousal_event_records(batch, torch.zeros_like(labels))

    assert records[0]["truth"][0, 0] == 0.5
    with pytest.raises(ValueError, match="finite, exact 0/1"):
        evaluate_arousal_record(records[0], {name: 0.5 for name in AROUSAL_SUBTYPES})


def test_arousal_masked_bce_counts_overlapping_subtype_targets():
    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=True,
        is_classification=True,
        label_name="arousal",
        label_source_name="arousal",
        is_multilabel=True,
        device="cpu",
    )
    module._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    targets = torch.zeros((1, 1, 120), dtype=torch.float32)
    targets[0, 0, 0:2] = 1.0
    logits = torch.zeros_like(targets)

    loss, valid_count = module._compute_loss(logits, {"tokens": {"arousal": targets}})

    assert valid_count == 120
    assert torch.isclose(loss, torch.tensor(math.log(2.0)))


def test_arousal_checkpoint_protocol_requires_exact_thresholds_and_fallback_metadata(caplog):
    import pytorch_lightning as pl

    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(module)
    module.args = argparse.Namespace(label_name="arousal")
    module.model_averager = None
    module._arousal_eval_thresholds = None
    module._arousal_eval_threshold_fallback_subtypes = None

    with pytest.raises(ValueError, match="must contain `arousal_eval_thresholds`"):
        module.on_test_start()
    with pytest.raises(ValueError, match="exactly"):
        module.on_load_checkpoint(
            {
                "arousal_eval_thresholds": {"RES": 0.5},
                "arousal_eval_threshold_fallback_subtypes": [],
            }
        )
    with pytest.raises(ValueError, match="stored threshold 0.5"):
        module.on_load_checkpoint(
            {
                "arousal_eval_thresholds": {
                    "RES": 0.5,
                    "SPONT": 0.5,
                    "Limb": 0.5,
                    "PLM": 0.4,
                },
                "arousal_eval_threshold_fallback_subtypes": ["PLM"],
            }
        )

    module.on_load_checkpoint(
        {
            "arousal_eval_thresholds": {name: 0.5 for name in AROUSAL_SUBTYPES},
            "arousal_eval_threshold_fallback_subtypes": ["PLM"],
        }
    )
    with caplog.at_level(logging.WARNING):
        module.on_test_start()
    assert "fallback threshold 0.5 for subtype PLM" in caplog.text


def test_validate_arousal_thresholds_rejects_extra_key():
    with pytest.raises(ValueError, match="exactly"):
        validate_arousal_thresholds({**{name: 0.5 for name in AROUSAL_SUBTYPES}, "TOTAL": 0.5})


@dataclass
class _ModelConfig:
    value: int = 1


def test_arousal_checkpoint_save_load_roundtrip_and_fixed_test_thresholds(monkeypatch):
    import pytorch_lightning as pl

    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    source = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(source)
    source.args = argparse.Namespace(label_name="arousal")
    source.model_config = _ModelConfig()
    source.finetune_config = None
    source.model = torch.nn.Identity()
    source.model_averager = None
    source._arousal_eval_thresholds = {name: 0.5 for name in AROUSAL_SUBTYPES}
    source._arousal_eval_thresholds["RES"] = 0.7
    source._arousal_eval_threshold_fallback_subtypes = ["PLM"]
    checkpoint: dict[str, object] = {}

    source.on_save_checkpoint(checkpoint)

    restored = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    pl.LightningModule.__init__(restored)
    restored.args = argparse.Namespace(label_name="arousal")
    restored.model_averager = None
    restored._arousal_eval_thresholds = None
    restored._arousal_eval_threshold_fallback_subtypes = None
    restored.on_load_checkpoint(checkpoint)

    captured: dict[str, object] = {}

    def fake_compute(records, *, thresholds=None, search_thresholds=None):
        captured["thresholds"] = thresholds
        captured["search_thresholds"] = search_thresholds
        return {"arousal_subtype_macro_event_f1": 1.0}, dict(thresholds), []

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_arousal_metrics", fake_compute)
    restored._compute_arousal_metrics_for_stage("test", [{}])

    assert restored._arousal_eval_thresholds == checkpoint["arousal_eval_thresholds"]
    assert restored._arousal_eval_threshold_fallback_subtypes == ["PLM"]
    assert captured["thresholds"] == checkpoint["arousal_eval_thresholds"]
    assert captured["search_thresholds"] is None


def test_arousal_validation_uses_fixed_declared_threshold_grid(monkeypatch):
    from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning

    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(label_name="arousal", arousal_val_search_thresholds=(0.2,))
    module._arousal_eval_thresholds = None
    module._arousal_eval_threshold_fallback_subtypes = None
    captured: dict[str, object] = {}

    def fake_compute(records, *, thresholds=None, search_thresholds=None):
        captured["thresholds"] = thresholds
        captured["search_thresholds"] = search_thresholds
        resolved = {name: 0.5 for name in AROUSAL_SUBTYPES}
        return {"arousal_subtype_macro_event_f1": 0.0}, resolved, []

    monkeypatch.setattr("sleep2vec.sleep2vec_finetuning.compute_arousal_metrics", fake_compute)
    module._compute_arousal_metrics_for_stage("val", [{}])

    assert captured["thresholds"] is None
    assert captured["search_thresholds"] == AROUSAL_THRESHOLD_GRID


def test_run_inference_rejects_arousal_checkpoint_averaging(monkeypatch):
    from sleep2vec import infer as infer_mod

    monkeypatch.setattr(infer_mod, "apply_finetune_config", lambda args: (object(), object()))
    args = argparse.Namespace(label_name="arousal", avg_ckpts=2, inference_preset_path=None)

    with pytest.raises(ValueError, match="Arousal inference does not support average checkpoints"):
        infer_mod.run_inference(args)


def test_arousal_result_task_family_is_dedicated():
    from sleep2vec.results import _resolve_task_family

    args = argparse.Namespace(label_name="arousal", is_multilabel=True, is_classification=True, is_seq=True)
    assert _resolve_task_family(args) == "arousal_sequence"


def test_build_finetune_loader_routes_arousal_channels_and_scalar_metadata(monkeypatch):
    from sleep2vec.utils import _build_finetune_loader

    captured: dict[str, object] = {}

    class _Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def dataloader(self, device="cpu"):
            return {"device": device}

    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _Dataset)
    args = argparse.Namespace(
        label_name="arousal",
        label_source_name="arousal",
        auxiliary_label_source_names=["stage5"],
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        channel_aliases={},
        finetune_preset_path=None,
        finetune_data_index=Path("index.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        is_multilabel=True,
        is_survival=False,
        output_dim=120,
        weighted_random_sampler=False,
    )

    loader = _build_finetune_loader(args, split=["train"], sources=["demo"], shuffle=False, is_train_set=False)

    assert loader == {"device": "cpu"}
    assert captured["channel_names"] == ["eeg", "arousal", "stage5"]
    assert captured["meta_data_names"] == [
        "arousal_index_per_hour",
        "arousal_res_index_per_hour",
        "arousal_spont_index_per_hour",
        "arousal_limb_index_per_hour",
        "arousal_plm_index_per_hour",
        "tst",
    ]
    assert captured["meta_data_regression_names"] == captured["meta_data_names"]
