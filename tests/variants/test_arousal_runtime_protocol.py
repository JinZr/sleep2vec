from __future__ import annotations

import argparse
import importlib

import numpy as np
import pytest

PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")


@pytest.mark.parametrize("package_name", PACKAGES)
@pytest.mark.parametrize(
    ("label_name", "devices", "expected"),
    [
        ("ahi", [0, 1], True),
        ("arousal", [0, 1], True),
        ("arousal", [0], False),
        ("stage5", [0, 1], False),
    ],
)
def test_distributed_event_progress_bar_selection_matches_across_variants(
    package_name: str, label_name: str, devices: list[int], expected: bool
):
    finetune = importlib.import_module(f"{package_name}.finetune")
    args = argparse.Namespace(label_name=label_name, devices=devices)

    assert finetune._is_distributed_ahi_finetune(args) is expected


@pytest.mark.parametrize("package_name", PACKAGES)
def test_arousal_builtin_runtime_contract_is_package_local(package_name: str):
    common = importlib.import_module(f"{package_name}.common")
    arousal_metrics = importlib.import_module(f"{package_name}.metrics.arousal")
    inference = importlib.import_module(f"{package_name}.sleep2vec_inference")
    finetuning = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    results = importlib.import_module(f"{package_name}.results")

    args = argparse.Namespace(label_name="arousal")
    common.apply_task_flags(args)

    assert args.output_dim == 120
    assert args.is_seq is True
    assert args.is_multilabel is True
    assert args.label_source_name == "arousal"
    assert args.auxiliary_label_source_names == ["stage5"]
    assert args.monitor == "val_arousal_subtype_macro_event_f1"
    assert arousal_metrics.compute_arousal_metrics.__module__ == f"{package_name}.metrics.arousal"
    assert arousal_metrics.vectorized_event_stats.__module__ == f"{package_name}.metrics.core"
    assert inference.build_arousal_prediction_rows.__module__ == f"{package_name}.sleep2vec_inference"
    assert finetuning.Sleep2vecFinetuning._is_arousal_task.__module__ == f"{package_name}.sleep2vec_finetuning"
    assert results._resolve_task_family(args) == "arousal_sequence"


@pytest.mark.parametrize("package_name", PACKAGES)
def test_arousal_metric_protocol_matches_across_variants(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics.arousal")
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[2:5, :] = 1
    record = {
        "truth": truth,
        "score": truth.astype(np.float32) * 0.8 + 0.1,
        "tst_hours": 3.0,
        "arousal_res_index_per_hour": 1.0 / 3.0,
        "arousal_spont_index_per_hour": 1.0 / 3.0,
        "arousal_limb_index_per_hour": 1.0 / 3.0,
        "arousal_plm_index_per_hour": 1.0 / 3.0,
        "arousal_index_per_hour": 1.0 / 3.0,
    }

    metrics, thresholds, fallback = metrics_mod.compute_arousal_metrics(
        [record], thresholds={name: 0.5 for name in metrics_mod.AROUSAL_SUBTYPES}
    )

    assert fallback == []
    assert thresholds == {name: 0.5 for name in metrics_mod.AROUSAL_SUBTYPES}
    assert metrics["arousal_subtype_macro_event_f1"] == 1.0
    assert metrics["arousal_total_event_support"] == 1.0


@pytest.mark.parametrize("package_name", PACKAGES)
def test_arousal_metric_rejects_fractional_truth_across_variants(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics.arousal")
    truth = np.zeros((30, 4), dtype=np.float32)
    truth[0, 0] = 0.5
    record = {
        "truth": truth,
        "score": np.zeros((30, 4), dtype=np.float32),
        "tst_hours": 3.0,
        "arousal_res_index_per_hour": 0.0,
        "arousal_spont_index_per_hour": 0.0,
        "arousal_limb_index_per_hour": 0.0,
        "arousal_plm_index_per_hour": 0.0,
        "arousal_index_per_hour": 0.0,
    }

    with pytest.raises(ValueError, match="finite, exact 0/1"):
        metrics_mod.evaluate_arousal_record(
            record,
            {name: 0.5 for name in metrics_mod.AROUSAL_SUBTYPES},
        )


@pytest.mark.parametrize("package_name", PACKAGES)
def test_arousal_duplicate_window_identity_is_strict_across_variants(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics.arousal")
    truth = np.zeros((30, 4), dtype=np.int64)
    record = {
        "sample_id": "sample-a",
        "path": "night.npz",
        "token_start": 0,
        "n_tokens": 1,
        "truth": truth,
        "score": np.zeros((30, 4), dtype=np.float32),
        "tst_hours": 3.0,
        "arousal_res_index_per_hour": 0.0,
        "arousal_spont_index_per_hour": 0.0,
        "arousal_limb_index_per_hour": 0.0,
        "arousal_plm_index_per_hour": 0.0,
        "arousal_index_per_hour": 0.0,
    }
    conflicting = {**record, "sample_id": "sample-b"}

    with pytest.raises(ValueError, match="different sample_id"):
        metrics_mod.merge_arousal_window_records([record, conflicting])


@pytest.mark.parametrize("package_name", PACKAGES)
def test_arousal_export_counts_come_from_truth_raster_across_variants(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics.arousal")
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    truth = np.zeros((30, 4), dtype=np.int64)
    truth[0:3, 0] = 1
    record = {
        "truth": truth,
        "score": truth.astype(np.float32),
        "tst_hours": 3.0,
        "arousal_res_index_per_hour": 0.0,
        "arousal_spont_index_per_hour": 0.0,
        "arousal_limb_index_per_hour": 0.0,
        "arousal_plm_index_per_hour": 0.0,
        "arousal_index_per_hour": 0.0,
    }

    rows = inference_mod.build_arousal_prediction_rows([record], {name: 0.5 for name in metrics_mod.AROUSAL_SUBTYPES})

    assert rows[0]["n_predictions"] == truth.size
    assert rows[0]["true_arousal_res_event_count"] == 1
    assert rows[0]["true_arousal_event_count"] == 1
    assert rows[0]["true_arousal_res_index_per_hour"] == 0.0
    assert rows[0]["true_arousal_index_per_hour"] == 0.0
