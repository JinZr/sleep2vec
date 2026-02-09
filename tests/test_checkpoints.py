from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from sleep2vec.checkpoints import _parse_epoch, average_checkpoints, select_checkpoints


def _save_ckpt(path: Path, state: dict[str, torch.Tensor], *, wrapper: str = "state_dict") -> None:
    if wrapper == "state_dict":
        torch.save({"state_dict": state}, path)
    elif wrapper == "model":
        torch.save({"model": state}, path)
    else:
        torch.save(state, path)


def test_parse_epoch_from_checkpoint_name():
    assert _parse_epoch(Path("epoch=12-step=100.ckpt")) == 12
    assert _parse_epoch(Path("model-epoch-3.ckpt")) == 3
    assert _parse_epoch(Path("last.ckpt")) is None


def test_select_checkpoints_validates_input_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        select_checkpoints(tmp_path / "missing", end_ckpt=None, num_ckpts=2)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No .ckpt files found"):
        select_checkpoints(empty_dir, end_ckpt=None, num_ckpts=1)


def test_select_checkpoints_prefers_epoch_ordering(tmp_path: Path):
    ckpt_dir = tmp_path / "epoch_ckpts"
    ckpt_dir.mkdir()
    state = {"w": torch.tensor([1.0])}

    e1 = ckpt_dir / "epoch=1-step=10.ckpt"
    e2 = ckpt_dir / "epoch=2-step=20.ckpt"
    e3 = ckpt_dir / "epoch=3-step=30.ckpt"
    _save_ckpt(e1, state)
    _save_ckpt(e2, state)
    _save_ckpt(e3, state)

    selected = select_checkpoints(ckpt_dir, end_ckpt=e2, num_ckpts=2)
    assert [p.name for p in selected] == ["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"]


def test_select_checkpoints_falls_back_to_mtime_when_epochs_absent(tmp_path: Path):
    ckpt_dir = tmp_path / "mtime_ckpts"
    ckpt_dir.mkdir()
    state = {"w": torch.tensor([1.0])}

    files = [ckpt_dir / "a.ckpt", ckpt_dir / "b.ckpt", ckpt_dir / "c.ckpt"]
    for i, path in enumerate(files):
        _save_ckpt(path, state)
        mtime = 100 + i
        os.utime(path, (mtime, mtime))

    selected = select_checkpoints(ckpt_dir, end_ckpt=None, num_ckpts=2)
    assert [p.name for p in selected] == ["b.ckpt", "c.ckpt"]


def test_select_checkpoints_rejects_when_not_enough_candidates(tmp_path: Path):
    ckpt_dir = tmp_path / "few_ckpts"
    ckpt_dir.mkdir()
    _save_ckpt(ckpt_dir / "only.ckpt", {"w": torch.tensor([1.0])})

    with pytest.raises(ValueError, match="Not enough checkpoints to average"):
        select_checkpoints(ckpt_dir, end_ckpt=None, num_ckpts=2)


def test_average_checkpoints_validates_non_empty_input():
    with pytest.raises(ValueError, match="No checkpoints provided"):
        average_checkpoints([])


def test_average_checkpoints_averages_float_and_integer_tensors(tmp_path: Path):
    state1 = {
        "float_weight": torch.tensor([1.0, 3.0]),
        "int_weight": torch.tensor([2, 4], dtype=torch.int64),
    }
    state2 = {
        "float_weight": torch.tensor([3.0, 5.0]),
        "int_weight": torch.tensor([4, 6], dtype=torch.int64),
    }
    ckpt1 = tmp_path / "epoch=1.ckpt"
    ckpt2 = tmp_path / "epoch=2.ckpt"
    _save_ckpt(ckpt1, state1, wrapper="state_dict")
    _save_ckpt(ckpt2, state2, wrapper="raw")

    averaged = average_checkpoints([ckpt1, ckpt2], device="cpu")

    assert torch.allclose(averaged["float_weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(averaged["int_weight"], torch.tensor([3, 5], dtype=torch.int64))


def test_average_checkpoints_supports_model_wrapper(tmp_path: Path):
    state1 = {"w": torch.tensor([1.0, 2.0])}
    state2 = {"w": torch.tensor([3.0, 4.0])}
    ckpt1 = tmp_path / "m1.ckpt"
    ckpt2 = tmp_path / "m2.ckpt"
    _save_ckpt(ckpt1, state1, wrapper="model")
    _save_ckpt(ckpt2, state2, wrapper="model")

    averaged = average_checkpoints([ckpt1, ckpt2], device="cpu")
    assert torch.allclose(averaged["w"], torch.tensor([2.0, 3.0]))


def test_average_checkpoints_rejects_missing_keys_across_checkpoints(tmp_path: Path):
    ckpt1 = tmp_path / "epoch=1.ckpt"
    ckpt2 = tmp_path / "epoch=2.ckpt"
    _save_ckpt(ckpt1, {"w": torch.tensor([1.0]), "b": torch.tensor([2.0])}, wrapper="state_dict")
    _save_ckpt(ckpt2, {"w": torch.tensor([3.0])}, wrapper="state_dict")

    with pytest.raises(KeyError, match="Missing key 'b'"):
        average_checkpoints([ckpt1, ckpt2], device="cpu")
