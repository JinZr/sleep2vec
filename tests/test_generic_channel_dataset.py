from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from data.default_dataset import SampleIndex
from data.psg_pretrain_dataset import PSGPretrainDataset


def _write_ahi_npz(path: Path, ah_event, *, ahi: float, tst: float, stage5=None) -> None:
    payload = {
        "ah_event": np.asarray(ah_event, dtype=np.float32),
        "ahi": np.asarray(ahi, dtype=np.float32),
        "tst": np.asarray(tst, dtype=np.float32),
    }
    if stage5 is not None:
        payload["stage5"] = np.asarray(stage5, dtype=np.float32)
    np.savez(path, **payload)


def test_psg_dataset_uses_explicit_input_dims_for_custom_channels(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, wearable=np.arange(8, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 2,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["wearable"],
        channel_input_dims={"wearable": 4},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        token_sec=1,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))
    assert batch["tokens"]["wearable"].shape == (1, 2, 4)


def test_psg_dataset_uses_token_sec_for_window_count(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, wearable=np.arange(120, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 60,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["wearable"],
        channel_input_dims={"wearable": 4},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=30,
        token_sec=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))
    assert batch["tokens"]["wearable"].shape == (1, 30, 4)


def test_psg_dataset_allows_stage5_index_without_age_or_sex(tmp_path: Path):
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

    dataset = PSGPretrainDataset(
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

    batch = next(iter(dataset.dataloader(device="cpu")))
    assert "age" not in dataset.data[0].metadata
    assert "sex" not in dataset.data[0].metadata
    assert torch.equal(batch["tokens"]["stage5"], torch.tensor([[[0.0], [1.0]]]))


def test_psg_dataset_allows_ahi_index_without_age_or_sex(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    _write_ahi_npz(npz_path, np.arange(60, dtype=np.float32), ahi=9.5, tst=3.5, stage5=[1.0, 2.0])

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

    dataset = PSGPretrainDataset(
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

    batch = next(iter(dataset.dataloader(device="cpu")))
    assert "age" not in dataset.data[0].metadata
    assert "sex" not in dataset.data[0].metadata
    assert batch["metadata"]["ahi"].tolist() == [9.5]
    assert batch["metadata"]["tst"].tolist() == [3.5]


@pytest.mark.parametrize("metadata_name", ["age", "sex"])
def test_psg_dataset_requires_explicit_requested_metadata_columns(tmp_path: Path, metadata_name: str):
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
        PSGPretrainDataset(
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


def test_psg_dataset_requires_explicit_dims_for_non_stage5_channels(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 2,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Missing channel_input_dims"):
        PSGPretrainDataset(
            channel_names=["eeg"],
            channel_input_dims={},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index_path),
            split=["train"],
            max_tokens=2,
            token_sec=1,
            mask_rate=0.0,
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


def test_psg_dataset_pads_builtin_ahi_tokens_with_ignore_value(tmp_path: Path):
    first_npz = tmp_path / "first.npz"
    second_npz = tmp_path / "second.npz"
    _write_ahi_npz(first_npz, np.arange(30, dtype=np.float32), ahi=1.0, tst=5.0)
    _write_ahi_npz(second_npz, np.arange(60, dtype=np.float32).reshape(60, 1), ahi=2.0, tst=6.0)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(first_npz),
                "split": "train",
                "duration": 30,
                "age": 40,
                "sex": 1,
            },
            {
                "path": str(second_npz),
                "split": "train",
                "duration": 60,
                "age": 40,
                "sex": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["ahi"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        token_sec=30,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert batch["tokens"]["ahi"].shape == (2, 2, 30)
    assert torch.equal(batch["tokens"]["ahi"][0, 0], torch.arange(30, dtype=torch.float32))
    assert torch.equal(batch["tokens"]["ahi"][0, 1], torch.full((30,), -1.0))
    assert torch.equal(batch["tokens"]["ahi"][1, 1], torch.arange(30, 60, dtype=torch.float32))


def test_psg_dataset_rejects_legacy_builtin_ahi_npz_key(tmp_path: Path):
    npz_path = tmp_path / "legacy.npz"
    np.savez(npz_path, ahi=np.arange(30, dtype=np.float32), tst=np.asarray(5.0, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 30,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Built-in AHI contract"):
        PSGPretrainDataset(
            channel_names=["ahi"],
            channel_input_dims={},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index_path),
            split=["train"],
            max_tokens=2,
            token_sec=30,
            mask_rate=0.0,
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


def test_psg_dataset_rejects_missing_builtin_ahi_scalar(tmp_path: Path):
    npz_path = tmp_path / "missing_scalar.npz"
    np.savez(npz_path, ah_event=np.arange(30, dtype=np.float32), tst=np.asarray(5.0, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 30,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Built-in AHI contract"):
        PSGPretrainDataset(
            channel_names=["ahi"],
            channel_input_dims={},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index_path),
            split=["train"],
            max_tokens=2,
            token_sec=30,
            mask_rate=0.0,
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


@pytest.mark.parametrize(
    ("npz_payload",),
    [
        ({"ah_event": np.arange(30, dtype=np.float32), "ahi": np.asarray([1.0, 2.0], dtype=np.float32), "tst": 5.0},),
        ({"ah_event": np.arange(30, dtype=np.float32), "ahi": 1.0, "tst": 0.0},),
    ],
)
def test_psg_dataset_rejects_malformed_builtin_ahi_scalars(tmp_path: Path, npz_payload: dict[str, object]):
    npz_path = tmp_path / "malformed.npz"
    np.savez(npz_path, **npz_payload)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 30,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Built-in AHI contract"):
        PSGPretrainDataset(
            channel_names=["ahi"],
            channel_input_dims={},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index_path),
            split=["train"],
            max_tokens=2,
            token_sec=30,
            mask_rate=0.0,
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


def test_psg_dataset_loads_builtin_ahi_scalars_from_npz_for_old_finetune_preset_metadata(tmp_path: Path):
    npz_path = tmp_path / "preset_sample.npz"
    _write_ahi_npz(npz_path, np.arange(60, dtype=np.float32), ahi=12.5, tst=4.5)

    preset_path = tmp_path / "preset.pkl"
    samples = [
        SampleIndex(
            id=0,
            path=str(npz_path),
            start=0,
            end=2,
            metadata={"age": 40, "sex": 1, "source": "preset", "path": str(npz_path), "split": "train"},
        )
    ]
    with open(preset_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = PSGPretrainDataset(
        channel_names=["ahi"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=str(preset_path),
        index=None,
        split=["train"],
        max_tokens=2,
        token_sec=30,
        mask_rate=0.0,
        meta_data_names=["ahi", "tst"],
        meta_data_regression_names=["ahi", "tst"],
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert batch["metadata"]["ahi"].tolist() == [12.5]
    assert batch["metadata"]["tst"].tolist() == [4.5]


def test_psg_dataset_loads_builtin_ahi_scalars_from_npz_for_csv_finetune_metadata(tmp_path: Path):
    npz_path = tmp_path / "csv_sample.npz"
    _write_ahi_npz(npz_path, np.arange(60, dtype=np.float32), ahi=9.5, tst=3.5)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 60,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["ahi"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        token_sec=30,
        mask_rate=0.0,
        meta_data_names=["ahi", "tst"],
        meta_data_regression_names=["ahi", "tst"],
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert batch["metadata"]["ahi"].tolist() == [9.5]
    assert batch["metadata"]["tst"].tolist() == [3.5]


def test_psg_dataset_missing_channel_fallback_recognizes_builtin_ahi_for_legacy_preset(tmp_path: Path):
    npz_path = tmp_path / "preset_missing_channels.npz"
    _write_ahi_npz(npz_path, np.arange(60, dtype=np.float32), ahi=7.5, tst=4.0, stage5=[1.0, 2.0])

    preset_path = tmp_path / "preset_missing_channels.pkl"
    samples = [
        SampleIndex(
            id=0,
            path=str(npz_path),
            start=0,
            end=2,
            metadata={"age": 40, "sex": 1, "source": "preset", "path": str(npz_path), "split": "val"},
        )
    ]
    with open(preset_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = PSGPretrainDataset(
        channel_names=["ahi", "stage5"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=str(preset_path),
        index=None,
        split=["val"],
        max_tokens=2,
        token_sec=30,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        randomly_select_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch["tokens"].keys()) == {"ahi", "stage5"}
    assert batch["metadata"]["path"] == [str(npz_path)]


def test_psg_dataset_pair_first_uses_uniform_probs_when_pair_probs_omitted(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        eeg_original=np.arange(256, dtype=np.float32),
        ecg_original=np.arange(256, dtype=np.float32),
    )

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 2,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["eeg_original", "ecg_original"],
        channel_input_dims={"eeg_original": 128, "ecg_original": 128},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        token_sec=1,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=True,
        min_channels=2,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    loader = dataset.dataloader(device="cpu")
    assert dataset.train_pair_probs is None
    assert loader.batch_sampler.get_target_distribution() == {("eeg_original", "ecg_original"): 1.0}


def test_psg_dataset_reset_pair_selector_is_noop_when_selector_is_none(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, wearable=np.arange(8, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 2,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["wearable"],
        channel_input_dims={"wearable": 4},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        token_sec=1,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    dataset.reset_pair_selector()
