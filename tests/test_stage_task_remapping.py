from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sleep2vec.metrics import compute_downstream_metrics
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


def _stage_args(label_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        label_source_name="stage5",
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        finetune_preset_path=None,
        finetune_data_index=Path("index.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=3 if label_name == "stage3" else 4,
    )


def test_build_finetune_loader_uses_stage5_tokens_for_stage3(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _stage_args("stage3")

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


def test_get_targets_remaps_stage_labels_and_preserves_ignore_index():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(
        is_seq=True,
        label_name="stage4",
        label_source_name="stage5",
        device="cpu",
    )
    batch = {
        "tokens": {
            "stage5": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]]),
        }
    }

    labels = module._get_targets(batch)

    assert torch.equal(labels, torch.tensor([[0.0, 1.0, 1.0, 2.0, 3.0, -1.0]]))


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
