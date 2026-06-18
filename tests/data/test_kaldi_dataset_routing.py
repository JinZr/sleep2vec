from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from data.default_dataset import SampleIndex
from sleep2vec.utils import _build_finetune_loader, _dataset_class_for_args, get_pretrain_dataloader


class _DummyDataset:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataloader_config = {}
        self.data = [
            SampleIndex(
                id="sample-0",
                path="/tmp/sample-0.npz",
                start=0,
                end=2,
                payload={"available_channels": ["eeg", "ppg"]},
                metadata={"source": "demo", "path": "/tmp/sample-0.npz", "split": "val"},
            ),
            SampleIndex(
                id="sample-1",
                path="/tmp/sample-1.npz",
                start=0,
                end=2,
                payload={"available_channels": ["eeg", "ppg"]},
                metadata={"source": "demo", "path": "/tmp/sample-1.npz", "split": "val"},
            ),
        ]
        type(self).instances.append(self)

    def dataloader(self, device="cpu"):
        return {
            "device": device,
            "kwargs": self.kwargs,
            "dataloader_config": self.dataloader_config,
        }


def _reset_dummy():
    _DummyDataset.instances = []


def _pretrain_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_backend="kaldi",
        channel_names=["eeg", "ppg"],
        channel_input_dims={"eeg": 4, "ppg": 8},
        kaldi_data_root=Path("/kaldi/root"),
        kaldi_manifest=Path("/kaldi/root/manifest.json"),
        max_tokens=2,
        mask_rate=0.15,
        allow_missing_channels=True,
        min_channels=2,
        bucket_by_available_channels=True,
        train_pair_probs={("eeg", "ppg"): 1.0},
        train_pair_track_unique_samples=True,
        batch_size=2,
        num_workers=0,
        val_num_workers=0,
        device="cpu",
    )


def _finetune_args(
    label_name: str,
    *,
    label_source_name: str,
    output_dim: int,
    auxiliary_label_source_names: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_backend="kaldi",
        kaldi_data_root=Path("/kaldi/root"),
        kaldi_manifest=Path("/kaldi/root/manifest.json"),
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
    )


def test_get_pretrain_dataloader_routes_kaldi_kwargs(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("sleep2vec.utils.KaldiPSGDataset", _DummyDataset)

    train_loader, val_loader = get_pretrain_dataloader(_pretrain_args())

    assert train_loader["device"] == "cpu"
    assert val_loader["device"] == "cpu"
    assert len(_DummyDataset.instances) == 2

    train_kwargs = _DummyDataset.instances[0].kwargs
    val_kwargs = _DummyDataset.instances[1].kwargs
    assert train_kwargs["kaldi_data_root"] == Path("/kaldi/root")
    assert train_kwargs["manifest"] == Path("/kaldi/root/manifest.json")
    assert train_kwargs["channel_names"] == ["eeg", "ppg"]
    assert train_kwargs["channel_input_dims"] == {"eeg": 4, "ppg": 8}
    assert train_kwargs["split"] == ["train"]
    assert train_kwargs["train_pair_probs"] == {("eeg", "ppg"): 1.0}
    assert train_kwargs["train_pair_track_unique_samples"] is True
    assert val_kwargs["split"] == ["val"]
    assert val_kwargs["shuffle"] is False
    assert val_kwargs["train_pair_probs"] is None
    assert "save_preset_path" not in train_kwargs
    assert "load_preset_path" not in train_kwargs
    assert "index" not in train_kwargs
    assert "stride_tokens" not in train_kwargs
    assert val_loader["dataloader_config"]["batch_sampler"].pairs == [("eeg", "ppg")]


def test_build_finetune_loader_routes_kaldi_stage_channels(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("sleep2vec.utils.KaldiPSGDataset", _DummyDataset)
    args = _finetune_args("stage3", label_source_name="stage5", output_dim=3)

    loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    init_kwargs = loader["kwargs"]
    assert init_kwargs["channel_names"] == ["eeg", "stage5"]
    assert init_kwargs["kaldi_data_root"] == Path("/kaldi/root")
    assert init_kwargs["manifest"] == Path("/kaldi/root/manifest.json")
    assert init_kwargs["randomly_select_channels"] is False
    assert init_kwargs["allow_missing_channels"] is False
    assert init_kwargs["min_channels"] == 2
    assert init_kwargs["meta_data_names"] == []
    assert "save_preset_path" not in init_kwargs
    assert "load_preset_path" not in init_kwargs
    assert "index" not in init_kwargs
    assert "stride_tokens" not in init_kwargs


def test_build_finetune_loader_routes_kaldi_ahi_channels(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("sleep2vec.utils.KaldiPSGDataset", _DummyDataset)
    args = _finetune_args(
        "ahi",
        label_source_name="ahi",
        output_dim=30,
        auxiliary_label_source_names=["stage5"],
    )

    loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    init_kwargs = loader["kwargs"]
    assert init_kwargs["channel_names"] == ["eeg", "ahi", "stage5"]
    assert init_kwargs["meta_data_names"] == ["ahi", "tst"]
    assert init_kwargs["meta_data_regression_names"] == ["ahi", "tst"]
    assert init_kwargs["min_channels"] == 3


def test_dataset_class_for_args_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown data backend"):
        _dataset_class_for_args(argparse.Namespace(data_backend="parquet"))
