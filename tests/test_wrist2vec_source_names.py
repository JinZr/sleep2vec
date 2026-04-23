from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from wrist2vec.common import apply_model_config_args
from wrist2vec.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    ModelConfig,
    ProjectionConfig,
    TemporalAggConfig,
    TokenizerConfig,
    load_pretrain_config,
)
from wrist2vec.data.default_dataset import SampleIndex
from wrist2vec.data.psg_pretrain_dataset import PSGPretrainDataset
from wrist2vec.data.utils import filter_valid_sample_indices
from wrist2vec.downstream_model import Wrist2vecDownstreamModel
import wrist2vec.preprocess.save_dataset_presets as save_dataset_presets_module
from wrist2vec.preprocess.save_dataset_presets import (
    _filter_index_df_for_required_channels,
    _load_model_channel_source_names,
)


class _FakeNpz(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _write_yaml(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _pretrain_payload_with_source_names() -> dict:
    return {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "vocab_size": 1,
            },
            "projection": {
                "name": "simclr",
                "enabled": True,
                "hidden_dim": 8,
                "out_dim": 4,
            },
            "cls": {
                "embedding_type": None,
                "downstream": "tokens",
            },
            "channels": [
                {
                    "name": "ppg_green",
                    "input_dim": 4,
                    "source_names": ["ppg_green_gd1", "ppg_green_gd2"],
                    "tokenizer": {"name": "linear", "out_dim": 8},
                },
                {
                    "name": "acc_vm",
                    "input_dim": 4,
                    "tokenizer": {"name": "linear", "out_dim": 8},
                },
            ],
        },
        "loss": {"name": "info_nce", "temperature": 0.2},
        "data": {"mask_rate": 0.1, "max_tokens": 4},
    }


def test_wrist2vec_source_names_are_parsed_and_expand_effective_channels(tmp_path: Path):
    bundle = load_pretrain_config(_write_yaml(tmp_path, _pretrain_payload_with_source_names()))

    args = argparse.Namespace()
    apply_model_config_args(args, bundle.model)

    assert bundle.model.channels[0].source_names == ["ppg_green_gd1", "ppg_green_gd2"]
    assert args.channel_names == ["ppg_green", "acc_vm"]
    assert args.channel_source_names["ppg_green"] == ["ppg_green_gd1", "ppg_green_gd2"]
    assert args.effective_channel_names == ["ppg_green::ppg_green_gd1", "ppg_green::ppg_green_gd2", "acc_vm"]
    assert args.effective_channel_to_logical["ppg_green::ppg_green_gd2"] == "ppg_green"
    assert args.effective_channel_to_source["ppg_green::ppg_green_gd1"] == "ppg_green_gd1"


def test_wrist2vec_source_names_reject_built_in_label_channels(tmp_path: Path):
    payload = _pretrain_payload_with_source_names()
    payload["model"]["channels"][0]["name"] = "stage5"

    with pytest.raises(ValueError, match="cannot set source_names"):
        load_pretrain_config(_write_yaml(tmp_path, payload))


def test_wrist2vec_preset_source_names_reject_built_in_label_channels():
    payload = _pretrain_payload_with_source_names()
    payload["model"]["channels"][0]["name"] = "stage5"

    with pytest.raises(ValueError, match="cannot set source_names"):
        _load_model_channel_source_names(payload, ["stage5", "acc_vm"])


def test_wrist2vec_preset_source_names_reject_non_list_values():
    payload = _pretrain_payload_with_source_names()
    payload["model"]["channels"][0]["source_names"] = "ppg_green_gd1"

    with pytest.raises(ValueError, match="source_names must be a list of non-empty strings"):
        _load_model_channel_source_names(payload, ["ppg_green", "acc_vm"])


def test_wrist2vec_preset_defaults_to_strict_mask_prefilter_for_finetune(tmp_path: Path, monkeypatch, capsys):
    config_path = _write_yaml(
        tmp_path,
        {
            "model": {
                "channels": [
                    {"name": "ppg_green", "input_dim": 4},
                    {"name": "acc_vm", "input_dim": 4},
                ]
            },
            "finetune": {"task": {"type": "classification", "output_dim": 2, "is_seq": False}},
        },
        name="finetune.yaml",
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text("path\n")

    args = argparse.Namespace(
        index=[str(index_path)],
        config=config_path,
        dataset_name="demo",
        output_template="data/{dataset}_{split}_preset_{tokens}.pickle",
        split=["train"],
        n_tokens=2,
        stride_tokens=None,
        meta_data_names=[],
        include_no_metadata=False,
        channels=None,
        batch_size=1,
        shuffle=False,
        mask_rate=0.0,
        allow_missing_channels=None,
        min_channels=2,
        overwrite=False,
        num_workers=1,
        dry_run=True,
    )
    monkeypatch.setattr(save_dataset_presets_module, "parse_args", lambda: args)

    save_dataset_presets_module.main()

    out = capsys.readouterr().out
    assert "expand_source_branches=True" in out
    assert "enabled strict AND" in out


def test_wrist2vec_preset_finetune_rejects_allow_missing_channels(tmp_path: Path, monkeypatch):
    config_path = _write_yaml(
        tmp_path,
        {
            "model": {
                "channels": [
                    {"name": "ppg_green", "input_dim": 4},
                    {"name": "acc_vm", "input_dim": 4},
                ]
            },
            "finetune": {"task": {"type": "classification", "output_dim": 2, "is_seq": False}},
        },
        name="finetune.yaml",
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text("path\n")

    args = argparse.Namespace(
        index=[str(index_path)],
        config=config_path,
        dataset_name="demo",
        output_template="data/{dataset}_{split}_preset_{tokens}.pickle",
        split=["train"],
        n_tokens=2,
        stride_tokens=None,
        meta_data_names=[],
        include_no_metadata=False,
        channels=None,
        batch_size=1,
        shuffle=False,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        overwrite=False,
        num_workers=1,
        dry_run=True,
    )
    monkeypatch.setattr(save_dataset_presets_module, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="requires allow_missing_channels=False"):
        save_dataset_presets_module.main()


def test_wrist2vec_filter_valid_sample_indices_records_channel_sources(monkeypatch):
    npz_by_path = {
        "sample.npz": _FakeNpz(
            {
                "ppg_green_gd1": torch.arange(8, dtype=torch.float32).numpy(),
                "ppg_green_gd2": torch.arange(8, dtype=torch.float32).numpy(),
                "acc_vm": torch.arange(8, dtype=torch.float32).numpy(),
            }
        )
    }
    monkeypatch.setattr("wrist2vec.data.utils.load_npz", lambda path: npz_by_path[path])
    monkeypatch.setattr("wrist2vec.data.utils.random.choice", lambda seq: seq[0])

    sample = SampleIndex(id=0, path="sample.npz", start=0, end=2, payload={})
    filtered = filter_valid_sample_indices(
        [sample],
        extractors={},
        tokenizers={},
        allow_missing_channels=True,
        channel_names=["ppg_green", "acc_vm"],
        channel_input_dims={"ppg_green": 4, "acc_vm": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"], "acc_vm": ["acc_vm"]},
        min_channels=2,
        max_workers=1,
    )

    assert [item.id for item in filtered] == [0]
    assert filtered[0].payload["available_channels"] == ["ppg_green", "acc_vm"]
    assert filtered[0].payload["channel_sources"] == {
        "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
        "acc_vm": ["acc_vm"],
    }


def test_wrist2vec_preset_mask_filter_uses_or_for_missing_channels_and_and_for_strict():
    df = pd.DataFrame(
        [
            {"ppg_green_gd1_mask": 1, "ppg_green_gd2_mask": 0, "acc_vm_mask": 1},
            {"ppg_green_gd1_mask": 1, "ppg_green_gd2_mask": 1, "acc_vm_mask": 1},
            {"ppg_green_gd1_mask": 0, "ppg_green_gd2_mask": 0, "acc_vm_mask": 1},
        ]
    )
    channel_source_names = {
        "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
        "acc_vm": ["acc_vm"],
    }

    filtered_missing = _filter_index_df_for_required_channels(
        df,
        ["ppg_green", "acc_vm"],
        channel_source_names=channel_source_names,
        allow_missing_channels=True,
        min_channels=2,
    )
    filtered_strict = _filter_index_df_for_required_channels(
        df,
        ["ppg_green", "acc_vm"],
        channel_source_names=channel_source_names,
        allow_missing_channels=False,
    )

    assert len(filtered_missing) == 2
    assert len(filtered_strict) == 1


def test_wrist2vec_preset_mask_filter_allow_missing_respects_min_channels():
    df = pd.DataFrame(
        [
            {"ppg_green_gd1_mask": 1, "ppg_green_gd2_mask": 0, "acc_vm_mask": 1},
            {"ppg_green_gd1_mask": 1, "ppg_green_gd2_mask": 1, "acc_vm_mask": 1},
            {"ppg_green_gd1_mask": 0, "ppg_green_gd2_mask": 0, "acc_vm_mask": 1},
        ]
    )
    channel_source_names = {
        "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
        "acc_vm": ["acc_vm"],
    }

    filtered_missing = _filter_index_df_for_required_channels(
        df,
        ["ppg_green", "acc_vm"],
        channel_source_names=channel_source_names,
        allow_missing_channels=True,
        min_channels=1,
    )

    assert len(filtered_missing) == 3


def test_wrist2vec_strict_preset_mask_filter_requires_all_source_masks():
    df = pd.DataFrame(
        [
            {"ppg_green_gd1_mask": 1, "acc_vm_mask": 1},
        ]
    )
    channel_source_names = {
        "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
        "acc_vm": ["acc_vm"],
    }

    with pytest.raises(ValueError, match="Missing source-specific .*strict channel 'ppg_green'"):
        _filter_index_df_for_required_channels(
            df,
            ["ppg_green", "acc_vm"],
            channel_source_names=channel_source_names,
            allow_missing_channels=False,
        )


def test_wrist2vec_strict_filter_does_not_fallback_to_logical_channel_name(monkeypatch):
    npz_by_path = {
        "sample.npz": _FakeNpz(
            {
                "ppg_green": torch.arange(8, dtype=torch.float32).numpy(),
            }
        )
    }
    monkeypatch.setattr("wrist2vec.data.utils.load_npz", lambda path: npz_by_path[path])

    sample = SampleIndex(id=0, path="sample.npz", start=0, end=2, payload={})
    filtered = filter_valid_sample_indices(
        [sample],
        extractors={},
        tokenizers={},
        allow_missing_channels=False,
        channel_names=["ppg_green"],
        channel_input_dims={"ppg_green": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"]},
        expand_source_branches=True,
        min_channels=1,
        max_workers=1,
    )

    assert filtered == []


def test_wrist2vec_strict_dataset_expands_effective_branch_tokens(tmp_path: Path, monkeypatch):
    preset_path = tmp_path / "preset.pkl"
    sample = SampleIndex(
        id=0,
        path="sample.npz",
        start=0,
        end=2,
        payload={"channel_sources": {"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"], "stage5": ["stage5"]}},
        metadata={"age": 40, "sex": 1, "source": "a", "path": "sample.npz", "split": "train"},
    )
    preset_path.write_bytes(pickle.dumps([sample]))

    npz = _FakeNpz(
        {
            "ppg_green_gd1": torch.arange(8, dtype=torch.float32).numpy(),
            "ppg_green_gd2": torch.arange(8, dtype=torch.float32).numpy() + 10,
            "stage5": torch.tensor([0, 1], dtype=torch.float32).numpy(),
        }
    )
    monkeypatch.setattr("wrist2vec.data.default_dataset.load_npz", lambda path: npz)

    dataset = PSGPretrainDataset(
        channel_names=["ppg_green", "stage5"],
        channel_input_dims={"ppg_green": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"]},
        save_preset_path=None,
        load_preset_path=str(preset_path),
        index="unused.csv",
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        allow_missing_channels=False,
        min_channels=2,
        expand_source_branches=True,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch["tokens"]) == {"ppg_green::ppg_green_gd1", "ppg_green::ppg_green_gd2", "stage5"}


def test_wrist2vec_allow_missing_collate_keeps_logical_keys_and_picks_one_source(tmp_path: Path, monkeypatch):
    preset_path = tmp_path / "preset.pkl"
    sample = SampleIndex(
        id=0,
        path="sample.npz",
        start=0,
        end=2,
        payload={
            "available_channels": ["ppg_green", "acc_vm"],
            "channel_sources": {
                "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
                "acc_vm": ["acc_vm"],
            },
        },
        metadata={"age": 40, "sex": 1, "source": "a", "path": "sample.npz", "split": "train"},
    )
    preset_path.write_bytes(pickle.dumps([sample]))

    npz = _FakeNpz(
        {
            "ppg_green_gd1": torch.arange(8, dtype=torch.float32).numpy(),
            "ppg_green_gd2": torch.arange(8, dtype=torch.float32).numpy() + 10,
            "acc_vm": torch.arange(8, dtype=torch.float32).numpy() + 100,
        }
    )
    monkeypatch.setattr("wrist2vec.data.default_dataset.load_npz", lambda path: npz)
    monkeypatch.setattr("wrist2vec.data.utils.random.choice", lambda seq: seq[-1])

    dataset = PSGPretrainDataset(
        channel_names=["ppg_green", "acc_vm"],
        channel_input_dims={"ppg_green": 4, "acc_vm": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"], "acc_vm": ["acc_vm"]},
        save_preset_path=None,
        load_preset_path=str(preset_path),
        index="unused.csv",
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        expand_source_branches=False,
        randomly_select_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch["tokens"]) == {"ppg_green", "acc_vm"}
    assert torch.equal(
        batch["tokens"]["ppg_green"][0],
        torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0, 17.0],
            ]
        ),
    )


class _StubBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.transformer_hidden_size = feature_dim
        self.cls_embedding = None
        self.seen_channel_names: list[str] | None = None
        self.seen_channel_to_logical: dict[str, str] | None = None

    def _tokenize_all(self, tokens, *, channel_names=None, channel_to_logical=None):
        self.seen_channel_names = list(channel_names or [])
        self.seen_channel_to_logical = dict(channel_to_logical or {})
        return {name: tokens[name] for name in self.seen_channel_names}

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        B, L, _ = token_embeddings.shape
        attn_mask = torch.ones(B, L, dtype=torch.bool)
        return token_embeddings, attn_mask, None


def test_wrist2vec_downstream_model_consumes_effective_branches():
    feature_dim = 4
    backbone = _StubBackbone(feature_dim)
    model_cfg = ModelConfig(
        channels=[
            ChannelConfig(
                name="ppg_green",
                input_dim=4,
                source_names=["ppg_green_gd1", "ppg_green_gd2"],
                tokenizer=TokenizerConfig(name="linear", out_dim=feature_dim),
            ),
            ChannelConfig(
                name="acc_vm",
                input_dim=4,
                tokenizer=TokenizerConfig(name="linear", out_dim=feature_dim),
            ),
        ],
        backbone=BackboneConfig(name="roformer", hidden_size=feature_dim, num_hidden_layers=2, num_attention_heads=2),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=feature_dim, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            name="classification",
        ),
    )

    model = Wrist2vecDownstreamModel(
        target="sex",
        backbone=backbone,
        channel_names=["ppg_green", "acc_vm"],
        effective_channel_names=["ppg_green::ppg_green_gd1", "ppg_green::ppg_green_gd2", "acc_vm"],
        effective_channel_to_logical={
            "ppg_green::ppg_green_gd1": "ppg_green",
            "ppg_green::ppg_green_gd2": "ppg_green",
            "acc_vm": "acc_vm",
        },
        output_dim=2,
        is_classification=True,
        is_seq=False,
        model_config=model_cfg,
        head_config=model_cfg.head,
        device="cpu",
    )
    batch = {
        "tokens": {
            "ppg_green::ppg_green_gd1": torch.randn(1, 2, feature_dim),
            "ppg_green::ppg_green_gd2": torch.randn(1, 2, feature_dim),
            "acc_vm": torch.randn(1, 2, feature_dim),
        },
        "length": torch.tensor([2]),
    }

    logits = model(batch)

    assert backbone.seen_channel_names == ["ppg_green::ppg_green_gd1", "ppg_green::ppg_green_gd2", "acc_vm"]
    assert backbone.seen_channel_to_logical == {
        "ppg_green::ppg_green_gd1": "ppg_green",
        "ppg_green::ppg_green_gd2": "ppg_green",
        "acc_vm": "acc_vm",
    }
    assert logits.shape == (1, 2)
