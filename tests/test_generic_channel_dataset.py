from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.psg_pretrain_dataset import PSGPretrainDataset


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
    np.savez(first_npz, ahi=np.arange(30, dtype=np.float32))
    np.savez(second_npz, ahi=np.arange(60, dtype=np.float32))

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
