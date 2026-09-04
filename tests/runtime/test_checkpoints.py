from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sleep2vec.checkpoints import (
    _parse_epoch,
    average_checkpoints,
    backbone_init_prefixes,
    extract_pretrain_init_state_dict,
    get_state_dict_from_checkpoint,
    load_pretrain_init_weights,
    select_checkpoints,
)


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
    _save_ckpt(ckpt2, state2, wrapper="state_dict")

    averaged = average_checkpoints([ckpt1, ckpt2], device="cpu")

    assert torch.allclose(averaged["float_weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(averaged["int_weight"], torch.tensor([3, 5], dtype=torch.int64))


@pytest.mark.parametrize("payload", [{"model": {"w": torch.tensor([1.0])}}, {"w": torch.tensor([1.0])}])
def test_get_state_dict_from_checkpoint_requires_lightning_state_dict(payload: dict):
    with pytest.raises(ValueError, match="top-level 'state_dict'"):
        get_state_dict_from_checkpoint(payload)


def test_average_checkpoints_rejects_missing_keys_across_checkpoints(tmp_path: Path):
    ckpt1 = tmp_path / "epoch=1.ckpt"
    ckpt2 = tmp_path / "epoch=2.ckpt"
    _save_ckpt(ckpt1, {"w": torch.tensor([1.0]), "b": torch.tensor([2.0])}, wrapper="state_dict")
    _save_ckpt(ckpt2, {"w": torch.tensor([3.0])}, wrapper="state_dict")

    with pytest.raises(KeyError, match="Missing key 'b'"):
        average_checkpoints([ckpt1, ckpt2], device="cpu")


def test_extract_pretrain_init_state_dict_prefers_ema_model():
    ckpt = {
        "state_dict": {
            "ema_model.encoder.weight": torch.tensor([1.0]),
            "model.encoder.weight": torch.tensor([2.0]),
        }
    }

    state_dict, used_prefix = extract_pretrain_init_state_dict(ckpt)

    assert used_prefix == "ema_model."
    assert torch.equal(state_dict["encoder.weight"], torch.tensor([1.0]))


class _TinyInitModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.tokenizer_mapping = nn.ModuleDict({"legacy": nn.Linear(2, 2)})


def test_load_pretrain_init_weights_reports_partial_channel_mismatch(tmp_path: Path):
    module = _TinyInitModule()
    ckpt_path = tmp_path / "init.ckpt"
    state_dict = module.state_dict()
    state_dict["tokenizer_mapping.new.weight"] = torch.ones(2, 2)
    state_dict["tokenizer_mapping.new.bias"] = torch.ones(2)
    torch.save({"state_dict": {f"model.{k}": v for k, v in state_dict.items()}}, ckpt_path)

    result = load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)

    assert result.used_prefix == "model."
    assert "tokenizer_mapping.new.weight" in result.unexpected_keys
    assert "tokenizer_mapping.new.bias" in result.unexpected_keys


def test_backbone_init_prefixes_try_the_finetune_subtree_first():
    """A finetune checkpoint holds the backbone under `backbone.`, a pretrain one does not.

    The general prefixes have to come last: `model.` matches a finetune checkpoint's
    `model.backbone.*` and strips it to `backbone.*`, which a bare pretrain model does not
    have -- and extraction only raises when *no* prefix matches, so it looked like a hit.
    """
    assert backbone_init_prefixes("ema") == (
        "ema_model.backbone.",
        "model.backbone.",
        "backbone.",
        "ema_model.",
        "model.",
    )
    assert backbone_init_prefixes(None) == ("model.backbone.", "backbone.", "model.")


def test_extract_pretrain_init_state_dict_unwraps_a_finetune_checkpoint():
    """`Sleep2vecFinetuning` registers the same backbone twice, so both subtrees are present."""
    ckpt = {
        "state_dict": {
            "backbone.encoder.weight": torch.tensor([1.0]),
            "model.backbone.encoder.weight": torch.tensor([1.0]),
            "model.head.weight": torch.tensor([9.0]),
        }
    }

    state_dict, used_prefix = extract_pretrain_init_state_dict(ckpt, prefixes=backbone_init_prefixes(None))

    assert used_prefix == "model.backbone."
    assert set(state_dict) == {"encoder.weight"}


def test_extract_pretrain_init_state_dict_leaves_a_pretrain_checkpoint_alone():
    ckpt = {"state_dict": {"model.encoder.weight": torch.tensor([2.0])}}

    state_dict, used_prefix = extract_pretrain_init_state_dict(ckpt, prefixes=backbone_init_prefixes(None))

    assert used_prefix == "model."
    assert set(state_dict) == {"encoder.weight"}


def test_every_variant_agrees_on_the_backbone_init_prefixes():
    """Enforced forks each own a copy; a checkpoint must unwrap the same way in all three."""
    import importlib

    expected = backbone_init_prefixes("ema")
    for variant in ("sleep2vec", "sleep2vec2", "sleep2expert"):
        module = importlib.import_module(f"{variant}.checkpoints")
        assert module.backbone_init_prefixes("ema") == expected, variant
        assert module.backbone_init_prefixes(None) == backbone_init_prefixes(None), variant


def test_load_pretrain_init_weights_rejects_a_checkpoint_that_loads_nothing(tmp_path: Path):
    """`strict=False` tolerates a partial mismatch, not a total one.

    A total mismatch leaves every parameter at its random initialization, and the caller
    logs a successful load and trains on -- the failure mode that made the README's
    "pass the old checkpoint as --pretrained-backbone-path" advice unsafe.
    """
    module = _TinyInitModule()
    ckpt_path = tmp_path / "other.ckpt"
    torch.save({"state_dict": {"model.some.other.weight": torch.ones(2, 2)}}, ckpt_path)

    with pytest.raises(ValueError, match="No weights loaded"):
        load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)


def test_load_pretrain_init_weights_still_allows_a_partial_load(tmp_path: Path):
    module = _TinyInitModule()
    ckpt_path = tmp_path / "partial.ckpt"
    state_dict = dict(module.state_dict())
    state_dict["tokenizer_mapping.new.weight"] = torch.ones(2, 2)
    torch.save({"state_dict": {f"model.{k}": v for k, v in state_dict.items()}}, ckpt_path)

    result = load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)

    assert result.unexpected_keys == ["tokenizer_mapping.new.weight"]
