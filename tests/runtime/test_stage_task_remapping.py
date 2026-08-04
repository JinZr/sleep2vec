from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sleep2vec.metrics.ahi import compute_ahi_pointwise_metrics
from sleep2vec.metrics.core import compute_downstream_metrics
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec.utils import _build_finetune_loader

PREDICTION_EXPORT_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")
PROBABILITY_METRIC_KEYS = {"auprc", "brier", "ece"}


class _DummyDataset:
    last_init_kwargs = None
    last_device = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    def dataloader(self, device="cpu"):
        type(self).last_device = device
        return {"device": device}


class _DummyDatasetWithSamples:
    samples = []
    last_device = None

    def __init__(self, **kwargs):
        self.data = type(self).samples

    def dataloader(self, device="cpu"):
        type(self).last_device = device
        return {"device": device}


def _seq_args(
    label_name: str,
    *,
    label_source_name: str,
    output_dim: int,
    is_multilabel: bool = False,
    auxiliary_label_source_names: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        label_source_name=label_source_name,
        auxiliary_label_source_names=auxiliary_label_source_names or [],
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        finetune_preset_path=None,
        finetune_data_index=Path("index.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=output_dim,
        is_multilabel=is_multilabel,
    )


def _metadata_args(label_name: str, *, is_classification: bool) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        finetune_preset_path=Path("preset.pkl"),
        finetune_data_index=None,
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=is_classification,
        output_dim=2 if is_classification else 1,
    )


def test_build_finetune_loader_uses_stage5_tokens_for_stage3(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _seq_args("stage3", label_source_name="stage5", output_dim=3)

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_device == "cpu"
    assert _DummyDataset.last_init_kwargs["channel_names"] == ["eeg", "stage5"]
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == []
    assert _DummyDataset.last_init_kwargs["meta_data_regression_names"] == []


@pytest.mark.parametrize(
    ("label_name", "is_classification", "metadata"),
    [
        ("age", False, {}),
        ("age", False, {"age": float("nan")}),
        ("sex", True, {}),
        ("sex", True, {"sex": float("nan")}),
    ],
)
def test_build_finetune_loader_rejects_missing_builtin_metadata_labels(
    monkeypatch,
    label_name: str,
    is_classification: bool,
    metadata: dict,
):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata=metadata)]
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDatasetWithSamples)
    args = _metadata_args(label_name, is_classification=is_classification)

    with pytest.raises(ValueError, match=f"invalid or missing '{label_name}' labels"):
        _build_finetune_loader(
            args,
            split=["test"],
            sources=[],
            shuffle=False,
            is_train_set=False,
        )


def test_build_finetune_loader_passes_weighted_random_sampler_for_train_metadata(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _metadata_args("src_isDep", is_classification=True)
    args.weighted_random_sampler = True

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=True,
        is_train_set=True,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == ["src_isDep"]
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler"] is True
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler_target"] == "src_isDep"


def test_build_finetune_loader_keeps_weighted_random_sampler_train_only(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _metadata_args("src_isDep", is_classification=True)
    args.weighted_random_sampler = True

    _build_finetune_loader(
        args,
        split=["val"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert _DummyDataset.last_init_kwargs["weighted_random_sampler"] is False
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler_target"] is None


@pytest.mark.parametrize(
    "args",
    [
        _seq_args("stage5", label_source_name="stage5", output_dim=5),
        _seq_args(
            "ahi",
            label_source_name="ahi",
            output_dim=30,
            is_multilabel=True,
            auxiliary_label_source_names=["stage5"],
        ),
    ],
)
def test_build_finetune_loader_allows_sequence_tasks_without_age_or_sex(monkeypatch, args):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata={})]
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDatasetWithSamples)

    loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=[],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}


def test_build_finetune_loader_uses_ahi_tokens_for_ahi(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _seq_args(
        "ahi",
        label_source_name="ahi",
        output_dim=30,
        is_multilabel=True,
        auxiliary_label_source_names=["stage5"],
    )

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_device == "cpu"
    assert _DummyDataset.last_init_kwargs["channel_names"] == ["eeg", "ahi", "stage5"]
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == ["ahi", "tst"]
    assert _DummyDataset.last_init_kwargs["meta_data_regression_names"] == ["ahi", "tst"]


def test_get_targets_remaps_stage_labels_and_preserves_ignore_index():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(
        is_seq=True,
        label_name="stage4",
        label_source_name="stage5",
        is_multilabel=False,
        device="cpu",
    )
    batch = {
        "tokens": {
            "stage5": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]]),
        }
    }

    labels = module._get_targets(batch)

    assert torch.equal(labels, torch.tensor([[0.0, 1.0, 1.0, 2.0, 3.0, -1.0]]))


def test_get_targets_returns_raw_ahi_labels():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(
        is_seq=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [1.0, -1.0]]]),
        }
    }

    labels = module._get_targets(batch)

    assert torch.equal(labels, torch.tensor([[[0.0, 1.0], [1.0, -1.0]]]))


def test_extract_ahi_event_records_keeps_sample_boundaries_and_scalar_summary():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([16.5], dtype=torch.float32),
            "tst": torch.tensor([5.25], dtype=torch.float32),
            "path": ["sample-a.npz"],
        },
        "token_start": torch.tensor([0], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert len(records) == 1
    assert records[0]["truth"].tolist() == [0, 1, 1, 0]
    assert records[0]["score"].shape == (4,)
    assert records[0]["true_ahi"] == 16.5
    assert records[0]["tst_hours"] == 5.25
    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["path"] == "sample-a.npz"
    assert records[0]["token_start"] == 0


def test_extract_ahi_event_records_keeps_stage5_tokens_with_second_level_mask():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [-1.0, -1.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([8.0], dtype=torch.float32),
            "tst": torch.tensor([4.0], dtype=torch.float32),
            "path": ["sample-b.npz"],
        },
        "token_start": torch.tensor([2], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["second_valid_mask"].tolist() == [True, True, False, False]


def test_extract_ahi_event_records_keeps_stage5_aligned_for_partially_masked_token():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, -1.0], [1.0, 0.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([8.0], dtype=torch.float32),
            "tst": torch.tensor([4.0], dtype=torch.float32),
            "path": ["sample-c.npz"],
        },
        "token_start": torch.tensor([4], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert records[0]["truth"].tolist() == [0, 1, 0]
    assert records[0]["score"].shape == (3,)
    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["second_valid_mask"].tolist() == [True, False, True, True]


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_scalar_classification_prediction_row_averages_probabilities(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "sample_id": "sample-0",
            "path": "sample.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": 1,
            "probabilities": [0.2, 0.8],
            "logits": [0.0, 2.0],
            "prediction": 1,
            "is_sequence": False,
        },
        {
            "sample_id": "sample-5",
            "path": "sample.npz",
            "token_start": 5,
            "kind": "classification",
            "groundtruth": 1,
            "probabilities": [0.6, 0.4],
            "logits": [1.0, 0.0],
            "prediction": 0,
            "is_sequence": False,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    row = rows[0]
    assert row["path"] == "sample.npz"
    assert row["groundtruth"] == 1
    assert row["prediction"] == 1
    assert row["n_predictions"] == 2
    assert row["n_windows"] == 2
    assert row["token_starts"] == [0, 5]
    assert row["prob_0"] == pytest.approx(0.4)
    assert row["prob_1"] == pytest.approx(0.6)
    assert row["logit_0"] == pytest.approx(0.5)
    assert row["logit_1"] == pytest.approx(1.0)
    assert row["logit"] == pytest.approx(0.5)


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_finalize_epoch_preserves_single_device_prediction_records(package_name: str, monkeypatch: pytest.MonkeyPatch):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    module = finetuning_cls.__new__(finetuning_cls)
    object.__setattr__(
        module,
        "args",
        argparse.Namespace(
            inference_prediction_csv_path="predictions.csv",
            is_survival=False,
            is_classification=True,
            is_multilabel=False,
            multilabel=None,
            label_name="sex",
            output_dim=2,
        ),
    )
    object.__setattr__(
        module,
        "_stage_outputs",
        {
            "train": [],
            "val": [],
            "test": [
                (
                    np.array([[0.1, 0.9]], dtype=np.float32),
                    np.array([1], dtype=np.int64),
                )
            ],
        },
    )
    object.__setattr__(
        module,
        "_prediction_records",
        {
            "val": [],
            "test": [
                {
                    "sample_id": "sample-0",
                    "path": "sample.npz",
                    "token_start": 0,
                    "kind": "classification",
                    "groundtruth": 1,
                    "probabilities": [0.1, 0.9],
                    "logits": [0.0, 2.0],
                    "prediction": 1,
                    "is_sequence": False,
                }
            ],
        },
    )
    object.__setattr__(module, "_eval_loss_sums", {})
    object.__setattr__(module, "prediction_rows", [])
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=False)
    monkeypatch.setattr(finetuning_mod, "is_torch_distributed_ready", lambda: False)
    monkeypatch.setattr(finetuning_mod, "compute_downstream_metrics", lambda *_args, **_kwargs: {})

    finetuning_cls._finalize_epoch(module, "test")

    assert module._prediction_records["test"] == []
    assert len(module.prediction_rows) == 1
    assert module.prediction_rows[0]["path"] == "sample.npz"
    assert module.prediction_rows[0]["prediction"] == 1


def _scalar_binary_record(
    path: str,
    token_start: int,
    groundtruth: int,
    positive_probability: float,
    *,
    sample_id: str | None = None,
):
    probabilities = [1.0 - positive_probability, positive_probability]
    return {
        "sample_id": sample_id or f"{path}:{token_start}",
        "path": path,
        "token_start": token_start,
        "kind": "classification",
        "groundtruth": groundtruth,
        "probabilities": probabilities,
        "logits": probabilities,
        "prediction": int(positive_probability >= 0.5),
        "is_sequence": False,
    }


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_scalar_classification_prediction_row_rejects_inconsistent_window_labels(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        _scalar_binary_record("episode.npz", 0, 0, 0.2),
        _scalar_binary_record("episode.npz", 5, 1, 0.8),
    ]

    with pytest.raises(ValueError, match="Classification labels differ across windows"):
        inference_mod.build_prediction_rows(records)


def _binary_epoch_module(
    finetuning_cls,
    stage: str,
    outputs,
    *,
    episode_records=None,
    export_predictions: bool = False,
):
    module = finetuning_cls.__new__(finetuning_cls)
    object.__setattr__(
        module,
        "args",
        argparse.Namespace(
            inference_prediction_csv_path="predictions.csv" if export_predictions else None,
            is_survival=False,
            is_classification=True,
            is_multilabel=False,
            is_seq=False,
            multilabel=None,
            label_name="sex",
            output_dim=2,
            stage_names=None,
            class_labels=None,
        ),
    )
    stage_outputs = {"train": [], "val": [], "test": []}
    stage_outputs[stage] = list(outputs)
    object.__setattr__(module, "_stage_outputs", stage_outputs)
    prediction_records = {"val": [], "test": []}
    if stage in prediction_records:
        prediction_records[stage] = list(episode_records or [])
    object.__setattr__(module, "_prediction_records", prediction_records)
    object.__setattr__(module, "_eval_loss_sums", {})
    object.__setattr__(module, "prediction_rows", [])
    module.__dict__["_trainer"] = argparse.Namespace(is_global_zero=False)
    return module


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_scalar_binary_val_shared_step_collects_prediction_records(
    package_name: str,
):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    module = _binary_epoch_module(finetuning_cls, "val", [])
    module.args.device = "cpu"
    object.__setattr__(module, "_eval_loss_counts", {})
    object.__setattr__(
        module,
        "_compute_loss",
        lambda _logits, _batch: None,
    )
    logits = torch.tensor([[-1.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
    batch = {
        "id": ["sample-a", "sample-b"],
        "metadata": {
            "path": ["episode-a.npz", "episode-b.npz"],
            "sex": torch.tensor([1, 0], dtype=torch.int64),
        },
        "token_start": torch.tensor([0, 5], dtype=torch.int64),
    }

    finetuning_cls._shared_step(module, batch, stage="val", model=lambda _batch: logits)

    assert len(module._stage_outputs["val"]) == 1
    window_preds, window_gts = module._stage_outputs["val"][0]
    np.testing.assert_allclose(window_preds, torch.softmax(logits, dim=-1).numpy())
    np.testing.assert_array_equal(window_gts, np.array([1, 0], dtype=np.int64))

    records = module._prediction_records["val"]
    assert len(records) == 2
    assert [record["sample_id"] for record in records] == ["sample-a", "sample-b"]
    assert [record["path"] for record in records] == ["episode-a.npz", "episode-b.npz"]
    assert [record["token_start"] for record in records] == [0, 5]
    assert [record["groundtruth"] for record in records] == [1, 0]
    assert records[0]["probabilities"] == pytest.approx(torch.softmax(logits[0], dim=-1).tolist())
    assert records[1]["probabilities"] == pytest.approx(torch.softmax(logits[1], dim=-1).tolist())


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
@pytest.mark.parametrize("stage", ["val", "test"])
def test_binary_epoch_metrics_use_one_probability_row_per_episode(
    package_name: str,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    records = [
        _scalar_binary_record("episode-a.npz", 0, 1, 0.2),
        _scalar_binary_record("episode-a.npz", 5, 1, 0.8),
        _scalar_binary_record("episode-b.npz", 0, 0, 0.1),
    ]
    window_output = (
        np.asarray([record["probabilities"] for record in records], dtype=np.float32),
        np.asarray([record["groundtruth"] for record in records], dtype=np.int64),
    )
    module = _binary_epoch_module(
        finetuning_cls,
        stage,
        [window_output],
        episode_records=records,
        export_predictions=(stage == "test"),
    )
    captured = []
    logged = {}
    original_compute = finetuning_mod.compute_downstream_metrics

    def capture_compute(gts, preds, **kwargs):
        captured.append(
            {
                "gts": np.asarray(gts).copy(),
                "preds": np.asarray(preds).copy(),
                "kwargs": dict(kwargs),
            }
        )
        return original_compute(gts, preds, **kwargs)

    object.__setattr__(
        module,
        "log",
        lambda name, value, **_kwargs: logged.__setitem__(name, float(value)),
    )
    monkeypatch.setattr(finetuning_mod, "is_torch_distributed_ready", lambda: False)
    monkeypatch.setattr(finetuning_mod, "compute_downstream_metrics", capture_compute)

    finetuning_cls._finalize_epoch(module, stage)

    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0]["gts"], np.array([1, 1, 0], dtype=np.int64))
    np.testing.assert_allclose(
        captured[0]["preds"],
        np.array([[0.8, 0.2], [0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
    )
    assert captured[0]["kwargs"]["include_binary_probability_metrics"] is False
    np.testing.assert_array_equal(captured[1]["gts"], np.array([1, 0], dtype=np.int64))
    np.testing.assert_allclose(
        captured[1]["preds"],
        np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float32),
    )
    assert captured[1]["kwargs"]["include_binary_probability_metrics"] is True
    assert logged[f"{stage}_accuracy"] == pytest.approx(2 / 3)
    assert logged[f"{stage}_episode_accuracy"] == pytest.approx(0.5)
    assert logged[f"{stage}_episode_auprc"] == pytest.approx(1.0)
    assert logged[f"{stage}_episode_brier"] == pytest.approx(0.13)
    assert logged[f"{stage}_episode_ece"] == pytest.approx(0.3)
    assert {f"{stage}_{key}" for key in PROBABILITY_METRIC_KEYS}.isdisjoint(logged)
    assert module._stage_outputs[stage] == []
    assert module._prediction_records[stage] == []
    if stage == "test":
        assert [row["path"] for row in module.prediction_rows] == ["episode-a.npz", "episode-b.npz"]
        assert module.prediction_rows[0]["n_windows"] == 2
        assert module.prediction_rows[0]["prob_1"] == pytest.approx(0.5)


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_binary_epoch_metrics_deduplicate_distributed_sampler_records(
    package_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    local_records = [_scalar_binary_record("episode-a.npz", 0, 1, 0.2)]
    remote_records = [
        _scalar_binary_record("episode-a.npz", 0, 1, 0.2),
        _scalar_binary_record("episode-a.npz", 5, 1, 0.8),
        _scalar_binary_record("episode-b.npz", 0, 0, 0.1),
    ]
    local_window_output = (
        np.array([[0.8, 0.2]], dtype=np.float32),
        np.array([1], dtype=np.int64),
    )
    gathered_window_preds = np.array(
        [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]],
        dtype=np.float32,
    )
    gathered_window_gts = np.array([1, 1, 1, 0], dtype=np.int64)
    module = _binary_epoch_module(
        finetuning_cls,
        "val",
        [local_window_output],
        episode_records=local_records,
    )
    captured = []
    original_compute = finetuning_mod.compute_downstream_metrics

    def capture_compute(gts, preds, **kwargs):
        captured.append((np.asarray(gts).copy(), np.asarray(preds).copy(), dict(kwargs)))
        return original_compute(gts, preds, **kwargs)

    object.__setattr__(module, "log", lambda *_args, **_kwargs: None)
    object.__setattr__(
        module,
        "_gather_eval_outputs",
        lambda _preds, _gts: (gathered_window_preds, gathered_window_gts),
    )
    object.__setattr__(
        module,
        "_gather_prediction_records",
        lambda records: records + remote_records,
    )
    monkeypatch.setattr(finetuning_mod, "compute_downstream_metrics", capture_compute)

    finetuning_cls._finalize_epoch(module, "val")

    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0][0], gathered_window_gts)
    np.testing.assert_allclose(captured[0][1], gathered_window_preds)
    assert captured[0][2]["include_binary_probability_metrics"] is False
    np.testing.assert_array_equal(captured[1][0], np.array([1, 0], dtype=np.int64))
    np.testing.assert_allclose(
        captured[1][1],
        np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float32),
    )
    assert captured[1][2]["include_binary_probability_metrics"] is True


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_binary_train_epoch_omits_probability_metrics(package_name: str, monkeypatch: pytest.MonkeyPatch):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    outputs = [
        (
            np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        )
    ]
    module = _binary_epoch_module(finetuning_cls, "train", outputs)
    captured = {}
    logged = {}
    original_compute = finetuning_mod.compute_downstream_metrics

    def capture_compute(gts, preds, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return original_compute(gts, preds, **kwargs)

    object.__setattr__(
        module,
        "log",
        lambda name, value, **_kwargs: logged.__setitem__(name, float(value)),
    )
    monkeypatch.setattr(finetuning_mod, "compute_downstream_metrics", capture_compute)

    finetuning_cls._finalize_epoch(module, "train")

    assert captured["kwargs"]["include_binary_probability_metrics"] is False
    assert {f"train_{key}" for key in PROBABILITY_METRIC_KEYS}.isdisjoint(logged)


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_binary_sequence_val_epoch_omits_scalar_probability_metrics(
    package_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    finetuning_mod = importlib.import_module(f"{package_name}.sleep2vec_finetuning")
    finetuning_cls = finetuning_mod.Sleep2vecFinetuning
    outputs = [
        (
            np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        )
    ]
    module = _binary_epoch_module(finetuning_cls, "val", outputs)
    module.args.is_seq = True
    captured = {}
    logged = {}
    original_compute = finetuning_mod.compute_downstream_metrics

    def capture_compute(gts, preds, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return original_compute(gts, preds, **kwargs)

    object.__setattr__(
        module,
        "log",
        lambda name, value, **_kwargs: logged.__setitem__(name, float(value)),
    )
    monkeypatch.setattr(finetuning_mod, "is_torch_distributed_ready", lambda: False)
    monkeypatch.setattr(finetuning_mod, "compute_downstream_metrics", capture_compute)

    finetuning_cls._finalize_epoch(module, "val")

    assert captured["kwargs"]["include_binary_probability_metrics"] is False
    assert {f"val_{key}" for key in PROBABILITY_METRIC_KEYS}.isdisjoint(logged)


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_extract_scalar_classification_prediction_records_preserves_logits(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    args = argparse.Namespace(is_multilabel=False, is_classification=True)
    batch = {
        "id": ["sample-a", "sample-b"],
        "metadata": {"path": ["a.npz", "b.npz"]},
        "token_start": torch.tensor([0, 5]),
    }
    logits = torch.tensor([[-1.0, 2.0], [3.0, 0.0]])
    targets = torch.tensor([1, 0])

    records = inference_mod.extract_prediction_records(args, batch, logits, targets)

    assert [record["sample_id"] for record in records] == ["sample-a", "sample-b"]
    assert records[0]["logits"] == pytest.approx([-1.0, 2.0])
    assert records[0]["probabilities"] == pytest.approx(torch.softmax(logits[0], dim=-1).tolist())
    assert records[1]["logits"] == pytest.approx([3.0, 0.0])
    assert records[1]["probabilities"] == pytest.approx(torch.softmax(logits[1], dim=-1).tolist())


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_scalar_regression_prediction_row_averages_windows(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "sample_id": "sample-0",
            "path": "sample.npz",
            "token_start": 0,
            "kind": "regression",
            "groundtruth": 60.0,
            "prediction": 61.0,
            "is_sequence": False,
        },
        {
            "sample_id": "sample-5",
            "path": "sample.npz",
            "token_start": 5,
            "kind": "regression",
            "groundtruth": 62.0,
            "prediction": 63.0,
            "is_sequence": False,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    row = rows[0]
    assert row["path"] == "sample.npz"
    assert row["groundtruth"] == pytest.approx(61.0)
    assert row["prediction"] == pytest.approx(62.0)
    assert row["n_predictions"] == 2
    assert row["n_windows"] == 2
    assert row["token_starts"] == [0, 5]


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_sequence_classification_prediction_row_concatenates_by_token_start(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "sample_id": "night-2",
            "path": "night.npz",
            "token_start": 2,
            "kind": "classification",
            "groundtruth": [2],
            "probabilities": [[0.1, 0.2, 0.7]],
            "logits": [[0.0, 1.0, 2.0]],
            "prediction": [2],
            "is_sequence": True,
        },
        {
            "sample_id": "night-0",
            "path": "night.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": [0, 1],
            "probabilities": [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]],
            "logits": [[2.0, 0.0, -1.0], [0.0, 2.0, 0.0]],
            "prediction": [0, 1],
            "is_sequence": True,
        },
        {
            "sample_id": "night-0",
            "path": "night.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": [0, 1],
            "probabilities": [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]],
            "logits": [[2.0, 0.0, -1.0], [0.0, 2.0, 0.0]],
            "prediction": [0, 1],
            "is_sequence": True,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    assert rows[0]["path"] == "night.npz"
    assert rows[0]["groundtruth"] == [0, 1, 2]
    assert rows[0]["prediction"] == [0, 1, 2]
    assert rows[0]["n_predictions"] == 3
    assert rows[0]["n_windows"] == 2
    assert rows[0]["token_starts"] == [0, 2]
    assert rows[0]["prob_0"] == pytest.approx([0.9, 0.1, 0.1])
    assert rows[0]["prob_1"] == pytest.approx([0.1, 0.8, 0.2])
    assert rows[0]["prob_2"] == pytest.approx([0.0, 0.1, 0.7])
    assert rows[0]["logit_0"] == pytest.approx([2.0, 0.0, 0.0])
    assert rows[0]["logit_1"] == pytest.approx([0.0, 2.0, 1.0])
    assert rows[0]["logit_2"] == pytest.approx([-1.0, 0.0, 2.0])


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_ahi_prediction_row_includes_threshold_and_summary(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    record = {
        "path": "night.npz",
        "token_start": 0,
        "truth": np.array([0] * 5 + [1] * 12 + [0] * 13, dtype=np.int64),
        "score": np.array([0.1] * 5 + [0.9] * 12 + [0.1] * 13, dtype=np.float32),
        "true_ahi": 10.0,
        "tst_hours": 3.0,
        "stage5": np.array([2], dtype=np.int64),
        "second_valid_mask": np.array([True] * 30),
    }

    rows = inference_mod.build_ahi_prediction_rows([record], threshold=0.5)

    assert rows[0]["path"] == "night.npz"
    assert rows[0]["groundtruth"] == record["truth"].tolist()
    assert rows[0]["prediction"] == ([0] * 5 + [1] * 12 + [0] * 13)
    assert rows[0]["prob"] == pytest.approx(record["score"].tolist())
    assert rows[0]["ahi_threshold"] == 0.5
    assert rows[0]["true_ahi"] == 10.0
    assert rows[0]["pred_ahi"] == pytest.approx(1 / 3.0)
    assert rows[0]["tst_hours"] == 3.0


def _config_package_name(path: Path) -> str:
    rel_path = path.relative_to(Path(__file__).resolve().parents[2] / "configs")
    if rel_path.parts and rel_path.parts[0] in {"sleep2vec2", "sleep2expert"}:
        return rel_path.parts[0]
    return "sleep2vec"


def _prediction_export_supports_task(task) -> bool:
    if not task.is_seq:
        return task.type in {"classification", "regression"}
    if task.type != "classification":
        return False
    return int(task.output_dim) in {3, 4, 5, 30, 120}


def test_prediction_export_supports_all_finetune_recipe_task_families():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_paths: list[Path] = []
    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        finetune = data.get("finetune")
        if isinstance(finetune, dict) and isinstance(finetune.get("task"), dict):
            recipe_paths.append(path)

    assert recipe_paths

    for path in recipe_paths:
        package_name = _config_package_name(path)
        bundle = importlib.import_module(f"{package_name}.config").load_finetune_config(path)
        assert _prediction_export_supports_task(bundle.finetune.task), str(path.relative_to(repo_root))


def test_compute_loss_ignores_ahi_padding():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=True,
        is_classification=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    module._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]]),
        }
    }

    loss, valid_count = module._compute_loss(logits, batch)

    assert valid_count == 5
    assert torch.isclose(loss, torch.tensor(math.log(2.0), dtype=torch.float32))


def test_compute_loss_applies_class_weights_for_binary_classification():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=False,
        is_classification=True,
        is_multilabel=False,
        label_name="src_isDep",
        device="cpu",
    )
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    module._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, -1], dtype=torch.long)
    batch = {"metadata": {"src_isDep": targets}}

    loss, valid_count = module._compute_loss(logits, batch)
    expected = torch.nn.functional.cross_entropy(logits[:2], targets[:2], weight=class_weights)
    unweighted = torch.nn.functional.cross_entropy(logits[:2], targets[:2])

    assert valid_count == 2
    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, unweighted)


def test_compute_loss_applies_ahi_pos_weight_and_ignores_padding():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=True,
        is_classification=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    pos_weight = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    module._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    targets = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]], dtype=torch.float32)
    batch = {"tokens": {"ahi": targets}}

    loss, valid_count = module._compute_loss(logits, batch)
    valid_mask = targets != -1.0
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )[valid_mask].mean()
    unweighted = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")[
        valid_mask
    ].mean()

    assert valid_count == 5
    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, unweighted)


def test_compute_downstream_metrics_reports_stage_specific_scores_for_stage3_and_stage4():
    for output_dim, stage_names in (
        (3, ["W", "NREM", "REM"]),
        (4, ["W", "N1N2", "N3", "REM"]),
    ):
        gts = np.arange(output_dim)
        preds = np.eye(output_dim, dtype=np.float32)

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=True,
            output_dim=output_dim,
            stage_names=stage_names,
        )

        for stage_name in stage_names:
            assert metrics[f"f1_{stage_name}"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["sens"] == 1.0
        assert metrics["spec"] == 1.0


def test_compute_downstream_metrics_reports_multiclass_recall_and_specificity():
    gts = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )

    metrics = compute_downstream_metrics(
        gts,
        preds,
        is_classification=True,
        output_dim=3,
    )

    assert metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["specificity"] == pytest.approx((0.75 + 0.75 + 1.0) / 3.0)
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["cohen_kappa"] == pytest.approx(0.5)
    assert metrics["f1_weighted"] == pytest.approx(59.0 / 90.0)
    assert metrics["f1_macro"] == pytest.approx(59.0 / 90.0)


def test_compute_downstream_metrics_reports_binary_recall_and_specificity():
    metrics = compute_downstream_metrics(
        np.array([0, 0, 1, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        is_classification=True,
        output_dim=2,
    )

    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5


def test_compute_downstream_metrics_preserves_macro_aliases_for_two_class_stage_names():
    metrics = compute_downstream_metrics(
        np.array([0, 0, 0, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        is_classification=True,
        output_dim=2,
        stage_names=["Wake", "Sleep"],
    )

    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == pytest.approx(1.0 / 3.0)
    assert metrics["sens"] == pytest.approx(2.0 / 3.0)
    assert metrics["spec"] == pytest.approx(2.0 / 3.0)


def test_compute_downstream_metrics_reports_binary_scores_for_ahi():
    metrics = compute_downstream_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
        is_classification=True,
        is_multilabel=True,
        output_dim=30,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_ahi_pointwise_metrics_uses_namespaced_keys():
    metrics = compute_ahi_pointwise_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
    )

    assert metrics["ahi_pointwise_accuracy"] == 1.0
    assert metrics["ahi_pointwise_precision"] == 1.0
    assert metrics["ahi_pointwise_recall"] == 1.0
    assert metrics["ahi_pointwise_specificity"] == 1.0
    assert metrics["ahi_pointwise_f1"] == 1.0
    assert metrics["ahi_pointwise_roc_auc"] == 1.0
