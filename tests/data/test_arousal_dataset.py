from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.psg_pretrain_dataset import PSGPretrainDataset, _build_channel_registry
from data.utils import AROUSAL_INDEX_KEYS, AROUSAL_METADATA_KEYS, load_builtin_arousal_metadata


def _write_arousal_npz(path: Path, events: np.ndarray, *, tst: float = 4.0) -> None:
    payload = {
        "arousal_event": np.asarray(events, dtype=np.float32),
        "stage5": np.ones(max(1, len(events) // 30), dtype=np.float32),
        "tst": np.asarray(tst, dtype=np.float32),
    }
    payload.update({key: np.asarray(index + 1.0, dtype=np.float32) for index, key in enumerate(AROUSAL_INDEX_KEYS)})
    np.savez(path, **payload)


def _dataset(index_path: Path, *, max_tokens: int = 2, token_sec: int = 30) -> PSGPretrainDataset:
    return PSGPretrainDataset(
        channel_names=["arousal"],
        channel_input_dims={},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=max_tokens,
        token_sec=token_sec,
        mask_rate=0.9,
        meta_data_names=list(AROUSAL_METADATA_KEYS),
        meta_data_regression_names=list(AROUSAL_METADATA_KEYS),
        randomly_select_channels=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )


def test_arousal_registry_flattens_second_major_and_accepts_all_zero() -> None:
    events = torch.zeros(60, 4, dtype=torch.float32)
    events[0, [0, 3]] = 1
    events[1, 1] = 1
    events[30, 2] = 1
    registry = _build_channel_registry(
        channel_names=["arousal"],
        channel_input_dims={},
        mask_rate=0.9,
    )

    tokens = registry["arousal"][1](events)
    zero_tokens = registry["arousal"][1](torch.zeros_like(events))

    assert tokens.shape == (2, 120)
    assert torch.equal(tokens.view(2, 30, 4), events.view(2, 30, 4))
    assert torch.count_nonzero(zero_tokens) == 0
    assert not registry["arousal"][2](tokens).any()


def test_arousal_registry_requires_input_dim_120() -> None:
    with pytest.raises(ValueError, match="input_dim=120"):
        _build_channel_registry(
            channel_names=["arousal"],
            channel_input_dims={"arousal": 4},
            mask_rate=0.0,
        )


def test_arousal_dataset_backfills_scalars_and_pads_with_minus_one(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_events = np.zeros((30, 4), dtype=np.float32)
    second_events = np.zeros((60, 4), dtype=np.float32)
    second_events[0, [0, 1]] = 1.0
    _write_arousal_npz(first, first_events, tst=3.0)
    _write_arousal_npz(second, second_events, tst=4.0)
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": str(first), "split": "train", "duration": 30},
            {"path": str(second), "split": "train", "duration": 60},
        ]
    ).to_csv(index_path, index=False)

    batch = next(iter(_dataset(index_path).dataloader()))

    assert batch["tokens"]["arousal"].shape == (2, 2, 120)
    assert torch.equal(batch["tokens"]["arousal"][0, 0].view(30, 4), torch.from_numpy(first_events))
    assert torch.equal(batch["tokens"]["arousal"][0, 1], torch.full((120,), -1.0))
    assert torch.equal(batch["tokens"]["arousal"][1].view(60, 4), torch.from_numpy(second_events))
    assert not batch["mlm_mask"]["arousal"].any()
    assert batch["metadata"]["tst"].tolist() == [3.0, 4.0]
    for index, key in enumerate(AROUSAL_INDEX_KEYS):
        assert batch["metadata"][key].tolist() == [index + 1.0, index + 1.0]


def test_arousal_npz_contract_rejects_wrong_shape_values_and_scalar_dtype(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    _write_arousal_npz(path, np.zeros((30, 3), dtype=np.float32))
    with np.load(path, allow_pickle=False) as npz, pytest.raises(ValueError, match=r"shape \[T, 4\]"):
        load_builtin_arousal_metadata(npz)

    events = np.zeros((30, 4), dtype=np.float32)
    events[0, 0] = 2.0
    _write_arousal_npz(path, events)
    with np.load(path, allow_pickle=False) as npz, pytest.raises(ValueError, match="only 0/1"):
        load_builtin_arousal_metadata(npz)

    payload = {"arousal_event": np.zeros((30, 4), dtype=np.float32), "stage5": np.ones(1, dtype=np.float32)}
    payload.update({key: np.asarray(1.0, dtype=np.float32) for key in AROUSAL_METADATA_KEYS})
    payload["tst"] = np.asarray(1.0, dtype=np.float64)
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as npz, pytest.raises(ValueError, match="dtype float32"):
        load_builtin_arousal_metadata(npz)


def test_arousal_npz_dataset_requires_thirty_second_tokens(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    pd.DataFrame(columns=["path", "split", "duration"]).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="token_sec=30"):
        _dataset(index_path, token_sec=10)
