from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sleep2vec.metrics import compute_ahi_pointwise_metrics, compute_downstream_metrics
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec.utils import _build_finetune_loader


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


def test_compute_loss_ignores_ahi_padding():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
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
        assert metrics["sens"] == 1.0
        assert metrics["spec"] == 1.0


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
    assert metrics["ahi_pointwise_f1"] == 1.0
    assert metrics["ahi_pointwise_roc_auc"] == 1.0
