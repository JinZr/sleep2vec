from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


VARIANT_PACKAGES = ["sleep2vec2", "sleep2expert"]


def _write_ahi_npz(path: Path) -> None:
    np.savez(
        path,
        ah_event=np.arange(60, dtype=np.float32),
        ahi=np.asarray(9.5, dtype=np.float32),
        tst=np.asarray(3.5, dtype=np.float32),
        stage5=np.asarray([1.0, 2.0], dtype=np.float32),
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


class _DummyDatasetWithSamples:
    samples = []
    last_device = None

    def __init__(self, **kwargs):
        self.data = type(self).samples

    def dataloader(self, device="cpu"):
        type(self).last_device = device
        return {"device": device}


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_psg_dataset_allows_stage5_index_without_age_or_sex(tmp_path: Path, package_name: str):
    dataset_module = importlib.import_module(f"{package_name}.data.psg_pretrain_dataset")
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, stage5=np.array([0.0, 1.0], dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "test",
                "duration": 60,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = dataset_module.PSGPretrainDataset(
        channel_names=["stage5"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["test"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    assert "age" not in dataset.data[0].metadata
    assert "sex" not in dataset.data[0].metadata


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_psg_dataset_allows_ahi_index_without_age_or_sex(tmp_path: Path, package_name: str):
    dataset_module = importlib.import_module(f"{package_name}.data.psg_pretrain_dataset")
    npz_path = tmp_path / "sample.npz"
    _write_ahi_npz(npz_path)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "test",
                "duration": 60,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = dataset_module.PSGPretrainDataset(
        channel_names=["ahi", "stage5"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["test"],
        max_tokens=2,
        mask_rate=0.0,
        meta_data_names=["ahi", "tst"],
        meta_data_regression_names=["ahi", "tst"],
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    assert "age" not in dataset.data[0].metadata
    assert "sex" not in dataset.data[0].metadata
    assert dataset.data[0].metadata["ahi"] == 9.5
    assert dataset.data[0].metadata["tst"] == 3.5


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
@pytest.mark.parametrize("metadata_name", ["age", "sex"])
def test_variant_psg_dataset_requires_explicit_requested_metadata_columns(
    tmp_path: Path,
    package_name: str,
    metadata_name: str,
):
    dataset_module = importlib.import_module(f"{package_name}.data.psg_pretrain_dataset")
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, stage5=np.array([0.0, 1.0], dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "test",
                "duration": 60,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match=f"Required metadata column '{metadata_name}' is missing"):
        dataset_module.PSGPretrainDataset(
            channel_names=["stage5"],
            channel_input_dims={},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index_path),
            split=["test"],
            max_tokens=2,
            mask_rate=0.0,
            meta_data_names=[metadata_name],
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
@pytest.mark.parametrize(
    ("label_name", "is_classification", "metadata"),
    [
        ("age", False, {}),
        ("age", False, {"age": float("nan")}),
        ("sex", True, {}),
        ("sex", True, {"sex": float("nan")}),
    ],
)
def test_variant_build_finetune_loader_rejects_missing_builtin_metadata_labels(
    monkeypatch,
    package_name: str,
    label_name: str,
    is_classification: bool,
    metadata: dict,
):
    utils_module = importlib.import_module(f"{package_name}.utils")
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata=metadata)]
    monkeypatch.setattr(utils_module, "PSGPretrainDataset", _DummyDatasetWithSamples)

    with pytest.raises(ValueError, match=f"invalid or missing '{label_name}' labels"):
        utils_module._build_finetune_loader(
            _metadata_args(label_name, is_classification=is_classification),
            split=["test"],
            sources=[],
            shuffle=False,
            is_train_set=False,
        )


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
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
def test_variant_build_finetune_loader_allows_sequence_tasks_without_age_or_sex(
    monkeypatch,
    package_name: str,
    args: argparse.Namespace,
):
    utils_module = importlib.import_module(f"{package_name}.utils")
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata={})]
    monkeypatch.setattr(utils_module, "PSGPretrainDataset", _DummyDatasetWithSamples)

    loader = utils_module._build_finetune_loader(
        args,
        split=["test"],
        sources=[],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}
