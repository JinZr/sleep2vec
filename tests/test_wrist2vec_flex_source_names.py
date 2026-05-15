from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from wrist2vec_flex.common import apply_model_config_args
from wrist2vec_flex.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    ModelConfig,
    ProjectionConfig,
    SourceEmbeddingConfig,
    SourceFusionConfig,
    TemporalAggConfig,
    TokenizerConfig,
    load_pretrain_config,
)
from wrist2vec_flex.data.default_dataset import SampleIndex
from wrist2vec_flex.data.psg_pretrain_dataset import PSGPretrainDataset
from wrist2vec_flex.data.utils import filter_valid_sample_indices
from wrist2vec_flex.downstream_model import Wrist2vecDownstreamModel
from wrist2vec_flex.downstreams.channel_aggregation.concat import ConcatChannelAggregator
from wrist2vec_flex.modules.channel_source_encoder import ChannelSourceEncoder
import wrist2vec_flex.preprocess.save_dataset_presets as save_dataset_presets_module
from wrist2vec_flex.preprocess.save_dataset_presets import (
    _filter_index_df_for_required_channels,
    _load_model_channel_source_names,
)
from wrist2vec_flex.utils import _build_finetune_loader


class _FakeNpz(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyDatasetWithSamples:
    samples = []

    def __init__(self, **kwargs):
        self.data = type(self).samples

    def dataloader(self, device="cpu"):
        return {"device": device}


class _ZeroTokenizer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        return torch.zeros(x.shape[0], self.feature_dim, dtype=x.dtype, device=x.device)


class _SourceMaskCountFusion(nn.Module):
    def forward(self, x, source_mask=None):
        if source_mask.dim() == 2:
            counts = source_mask.sum(dim=-1).to(dtype=x.dtype).view(x.shape[0], 1, 1)
        else:
            counts = source_mask.sum(dim=-1, keepdim=True).to(dtype=x.dtype)
        self.last_counts = counts.detach().clone()
        return counts.expand(x.shape[0], x.shape[1], x.shape[-1])


def _write_yaml(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _wrist_metadata_args(label_name: str, *, is_classification: bool) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        data_channel_names=["ppg_green"],
        channel_input_dims={"ppg_green": 4},
        channel_source_names={},
        finetune_preset_path=Path("preset.pkl"),
        finetune_data_index=None,
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=is_classification,
        output_dim=2 if is_classification else 1,
    )


def _wrist_seq_args(label_name: str, *, label_source_name: str, output_dim: int) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        label_source_name=label_source_name,
        auxiliary_label_source_names=["stage5"] if label_name == "ahi" else [],
        data_channel_names=["ppg_green"],
        channel_input_dims={"ppg_green": 4},
        channel_source_names={},
        finetune_preset_path=Path("preset.pkl"),
        finetune_data_index=None,
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=output_dim,
    )


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


def test_wrist2vec_psg_dataset_allows_stage5_index_without_age_or_sex(tmp_path: Path):
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


def test_wrist2vec_psg_dataset_allows_ahi_index_without_age_or_sex(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        ah_event=np.arange(60, dtype=np.float32),
        ahi=np.asarray(9.5, dtype=np.float32),
        tst=np.asarray(3.5, dtype=np.float32),
        stage5=np.array([1.0, 2.0], dtype=np.float32),
    )

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
def test_wrist2vec_psg_dataset_requires_explicit_requested_metadata_columns(
    tmp_path: Path,
    metadata_name: str,
):
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


@pytest.mark.parametrize(
    ("label_name", "is_classification", "metadata"),
    [
        ("age", False, {}),
        ("age", False, {"age": float("nan")}),
        ("sex", True, {}),
        ("sex", True, {"sex": float("nan")}),
    ],
)
def test_wrist2vec_build_finetune_loader_rejects_missing_builtin_metadata_labels(
    monkeypatch,
    label_name: str,
    is_classification: bool,
    metadata: dict,
):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata=metadata)]
    monkeypatch.setattr("wrist2vec_flex.utils.PSGPretrainDataset", _DummyDatasetWithSamples)
    args = _wrist_metadata_args(label_name, is_classification=is_classification)

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
        _wrist_seq_args("stage5", label_source_name="stage5", output_dim=5),
        _wrist_seq_args("ahi", label_source_name="ahi", output_dim=30),
    ],
)
def test_wrist2vec_build_finetune_loader_allows_sequence_tasks_without_age_or_sex(monkeypatch, args):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata={})]
    monkeypatch.setattr("wrist2vec_flex.utils.PSGPretrainDataset", _DummyDatasetWithSamples)

    loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=[],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}


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
        legacy_expand_source_branches=False,
    )
    monkeypatch.setattr(save_dataset_presets_module, "parse_args", lambda: args)

    save_dataset_presets_module.main()

    out = capsys.readouterr().out
    assert "expand_source_branches=False" in out
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
        legacy_expand_source_branches=True,
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
    monkeypatch.setattr("wrist2vec_flex.data.utils.load_npz", lambda path: npz_by_path[path])
    monkeypatch.setattr("wrist2vec_flex.data.utils.random.choice", lambda seq: seq[0])

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


def test_wrist2vec_saved_source_aware_preset_preserves_channel_sources(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        ppg_green_gd1=np.arange(8, dtype=np.float32),
        ppg_green_gd2=np.arange(8, dtype=np.float32) + 10,
        acc_vm=np.arange(8, dtype=np.float32) + 100,
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 60,
            }
        ]
    ).to_csv(index_path, index=False)
    preset_path = tmp_path / "preset.pkl"

    PSGPretrainDataset(
        channel_names=["ppg_green", "acc_vm"],
        channel_input_dims={"ppg_green": 4, "acc_vm": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"], "acc_vm": ["acc_vm"]},
        save_preset_path=str(preset_path),
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        filter_max_workers=1,
    )

    saved = pickle.loads(preset_path.read_bytes())

    assert len(saved) == 1
    assert saved[0].payload["available_channels"] == ["ppg_green", "acc_vm"]
    assert saved[0].payload["channel_sources"] == {
        "ppg_green": ["ppg_green_gd1", "ppg_green_gd2"],
        "acc_vm": ["acc_vm"],
    }


def test_wrist2vec_stale_source_aware_preset_requires_channel_sources(tmp_path: Path):
    preset_path = tmp_path / "stale.pkl"
    sample = SampleIndex(
        id=0,
        path="sample.npz",
        start=0,
        end=2,
        payload={},
        metadata={"source": "a", "path": "sample.npz", "split": "train"},
    )
    preset_path.write_bytes(pickle.dumps([sample]))

    with pytest.raises(ValueError, match=r"payload\['channel_sources'\] required by source_names"):
        PSGPretrainDataset(
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
            randomly_select_channels=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


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
    filtered_source_aware_strict = _filter_index_df_for_required_channels(
        df,
        ["ppg_green", "acc_vm"],
        channel_source_names=channel_source_names,
        allow_missing_channels=False,
    )
    filtered_legacy_strict = _filter_index_df_for_required_channels(
        df,
        ["ppg_green", "acc_vm"],
        channel_source_names=channel_source_names,
        allow_missing_channels=False,
        expand_source_branches=True,
    )

    assert len(filtered_missing) == 2
    assert len(filtered_source_aware_strict) == 2
    assert len(filtered_legacy_strict) == 1


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
            expand_source_branches=True,
        )


def test_wrist2vec_strict_filter_does_not_fallback_to_logical_channel_name(monkeypatch):
    npz_by_path = {
        "sample.npz": _FakeNpz(
            {
                "ppg_green": torch.arange(8, dtype=torch.float32).numpy(),
            }
        )
    }
    monkeypatch.setattr("wrist2vec_flex.data.utils.load_npz", lambda path: npz_by_path[path])

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
    monkeypatch.setattr("wrist2vec_flex.data.default_dataset.load_npz", lambda path: npz)

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


def test_wrist2vec_allow_missing_collate_keeps_logical_keys_and_stacks_sources(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr("wrist2vec_flex.data.default_dataset.load_npz", lambda path: npz)
    monkeypatch.setattr("wrist2vec_flex.data.utils.random.choice", lambda seq: seq[-1])

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
                [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]],
                [[4.0, 5.0, 6.0, 7.0], [14.0, 15.0, 16.0, 17.0]],
            ]
        ),
    )
    assert batch["source_mask"]["ppg_green"].tolist() == [[True, True]]
    assert batch["source_ids"]["ppg_green"].tolist() == [0, 1]


class _StubBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.transformer_hidden_size = feature_dim
        self.cls_embedding = None
        self.seen_channel_names: list[str] | None = None
        self.seen_channel_to_logical: dict[str, str] | None = None

    def _tokenize_all(self, tokens, *, channel_names=None, channel_to_logical=None, **_kwargs):
        self.seen_channel_names = list(channel_names or [])
        self.seen_channel_to_logical = dict(channel_to_logical or {})
        return {name: tokens[name] for name in self.seen_channel_names}

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        B, L, _ = token_embeddings.shape
        attn_mask = torch.ones(B, L, dtype=torch.bool)
        return token_embeddings, attn_mask, None


class _SourceAwareBackbone(nn.Module):
    def __init__(self, *, feature_dim: int, source_dropout_rate: float, min_sources_after_dropout: int):
        super().__init__()
        self.transformer_hidden_size = feature_dim
        self.cls_embedding = None
        self.tokenizer = _ZeroTokenizer(feature_dim)
        self.encoder = ChannelSourceEncoder(
            feature_dim=feature_dim,
            num_sources=3,
            source_fusion=SourceFusionConfig(name="masked_mean"),
            source_embedding=SourceEmbeddingConfig(enabled=False),
            source_dropout_rate=source_dropout_rate,
            min_sources_after_dropout=min_sources_after_dropout,
        )
        self.encoder.source_fusion = _SourceMaskCountFusion()

    def _tokenize_all(self, tokens, *, channel_names=None, source_mask=None, source_ids=None, **_kwargs):
        embeddings = {}
        for name in channel_names or []:
            embeddings[name] = self.encoder(
                tokens[name],
                tokenizer=self.tokenizer,
                source_mask=None if source_mask is None else source_mask.get(name),
                source_ids=None if source_ids is None else source_ids.get(name),
            )
        return embeddings

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        B, L, _ = token_embeddings.shape
        attn_mask = torch.ones(B, L, dtype=torch.bool, device=token_embeddings.device)
        return token_embeddings, attn_mask, None


def _single_channel_downstream_model(backbone: nn.Module) -> Wrist2vecDownstreamModel:
    feature_dim = backbone.transformer_hidden_size
    model_cfg = ModelConfig(
        channels=[
            ChannelConfig(
                name="ppg_green",
                input_dim=4,
                source_names=["ppg_green_gd1", "ppg_green_gd2", "ppg_green_gd3"],
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
            dropout=0.0,
        ),
    )
    return Wrist2vecDownstreamModel(
        target="sex",
        backbone=backbone,
        channel_names=["ppg_green"],
        effective_channel_names=None,
        effective_channel_to_logical=None,
        output_dim=2,
        is_classification=True,
        is_seq=False,
        model_config=model_cfg,
        head_config=model_cfg.head,
        device="cpu",
    )


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


def test_wrist2vec_channel_source_encoder_keeps_min_sources_after_dropout():
    encoder = ChannelSourceEncoder(
        feature_dim=2,
        num_sources=3,
        source_fusion=SourceFusionConfig(name="masked_mean"),
        source_embedding=SourceEmbeddingConfig(enabled=False),
        source_dropout_rate=1.0,
        min_sources_after_dropout=2,
    )
    encoder.source_fusion = _SourceMaskCountFusion()
    encoder.train()
    tokens = torch.zeros(3, 2, 3, 4)
    source_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, True],
            [False, True, False],
        ]
    )

    output = encoder(tokens, tokenizer=_ZeroTokenizer(feature_dim=2), source_mask=source_mask)

    assert output[:, 0, 0].tolist() == [2.0, 2.0, 1.0]


def test_wrist2vec_channel_source_encoder_source_dropout_disabled_in_eval():
    encoder = ChannelSourceEncoder(
        feature_dim=2,
        num_sources=3,
        source_fusion=SourceFusionConfig(name="masked_mean"),
        source_embedding=SourceEmbeddingConfig(enabled=False),
        source_dropout_rate=1.0,
        min_sources_after_dropout=2,
    )
    encoder.source_fusion = _SourceMaskCountFusion()
    encoder.eval()
    tokens = torch.zeros(2, 2, 3, 4)
    source_mask = torch.tensor([[True, True, True], [True, False, False]])

    output = encoder(tokens, tokenizer=_ZeroTokenizer(feature_dim=2), source_mask=source_mask)

    assert output[:, 0, 0].tolist() == [3.0, 1.0]


def test_wrist2vec_downstream_source_dropout_train_only():
    backbone = _SourceAwareBackbone(feature_dim=2, source_dropout_rate=1.0, min_sources_after_dropout=2)
    model = _single_channel_downstream_model(backbone)
    batch = {
        "tokens": {"ppg_green": torch.zeros(2, 2, 3, 4)},
        "source_mask": {"ppg_green": torch.tensor([[True, True, True], [True, False, False]])},
        "source_ids": {"ppg_green": torch.tensor([0, 1, 2])},
        "channel_mask": torch.ones(2, 1, dtype=torch.bool),
        "length": torch.tensor([2, 2]),
    }
    original_source_mask = batch["source_mask"]["ppg_green"].clone()

    model.train()
    train_logits = model(batch)
    train_counts = backbone.encoder.source_fusion.last_counts.squeeze(-1).squeeze(-1)

    model.eval()
    eval_logits = model(batch)
    eval_counts = backbone.encoder.source_fusion.last_counts.squeeze(-1).squeeze(-1)

    assert train_logits.shape == (2, 2)
    assert eval_logits.shape == (2, 2)
    assert train_counts.tolist() == [2.0, 1.0]
    assert eval_counts.tolist() == [3.0, 1.0]
    assert torch.equal(batch["source_mask"]["ppg_green"], original_source_mask)


def test_wrist2vec_finetuning_passes_source_dropout_to_backbone(monkeypatch):
    finetuning_module = pytest.importorskip("wrist2vec_flex.wrist2vec_finetuning")
    captured = {}

    class _FakePretrainModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.transformer_hidden_size = kwargs["transformer_hidden_size"]
            self.cls_embedding = None

        def set_tokenizers_trainable(self, trainable):
            self.tokenizers_trainable = trainable

    monkeypatch.setattr(finetuning_module, "Wrist2vecPretrainModel", _FakePretrainModel)
    monkeypatch.setattr(finetuning_module, "DownstreamEvalVisualizer", lambda *_args, **_kwargs: None)

    model_cfg = ModelConfig(
        channels=[
            ChannelConfig(
                name="ppg_green",
                input_dim=4,
                source_names=["ppg_green_gd1", "ppg_green_gd2"],
                tokenizer=TokenizerConfig(name="linear", out_dim=8),
            )
        ],
        backbone=BackboneConfig(name="roformer", hidden_size=8, num_hidden_layers=2, num_attention_heads=2),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            name="classification",
        ),
    )
    args = argparse.Namespace(
        device="cpu",
        label_name="sex",
        output_dim=2,
        is_classification=True,
        is_seq=False,
        pretrained_backbone_path=None,
        freeze_backbone_and_insert_lora=False,
        freeze_tokenizer=True,
        channel_dropout_rate=0.0,
        min_channels_after_dropout=1,
        source_dropout_rate=0.35,
        min_sources_after_dropout=2,
        data_channel_names=["ppg_green"],
        head_kwargs={},
        print_diagnostics=False,
    )

    finetuning_module.Wrist2vecFinetuning(args, model_cfg, finetune_config=None, averaging_config=None)

    assert captured["source_dropout_rate"] == pytest.approx(0.35)
    assert captured["min_sources_after_dropout"] == 2


def test_wrist2vec_finetuning_rejects_frozen_missing_source_modules(monkeypatch):
    finetuning_module = pytest.importorskip("wrist2vec_flex.wrist2vec_finetuning")

    class _FakePretrainModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.transformer_hidden_size = kwargs["transformer_hidden_size"]
            self.cls_embedding = None

        def set_tokenizers_trainable(self, trainable):
            self.tokenizers_trainable = trainable

    def _load_missing_source_modules(self, _path):
        return argparse.Namespace(missing_keys=["channel_source_encoder_mapping.ppg_green.source_fusion.query.weight"])

    monkeypatch.setattr(finetuning_module, "Wrist2vecPretrainModel", _FakePretrainModel)
    monkeypatch.setattr(
        finetuning_module.Wrist2vecDownstreamModel,
        "load_pretrained_backbone",
        _load_missing_source_modules,
    )

    model_cfg = ModelConfig(
        channels=[
            ChannelConfig(
                name="ppg_green",
                input_dim=4,
                source_names=["ppg_green_gd1", "ppg_green_gd2"],
                tokenizer=TokenizerConfig(name="linear", out_dim=8),
            )
        ],
        backbone=BackboneConfig(name="roformer", hidden_size=8, num_hidden_layers=2, num_attention_heads=2),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            name="classification",
        ),
    )
    args = argparse.Namespace(
        device="cpu",
        label_name="sex",
        output_dim=2,
        is_classification=True,
        is_seq=False,
        pretrained_backbone_path="baseline.ckpt",
        freeze_backbone_and_insert_lora=False,
        freeze_tokenizer=True,
        channel_dropout_rate=0.0,
        min_channels_after_dropout=1,
        source_dropout_rate=0.0,
        min_sources_after_dropout=1,
        data_channel_names=["ppg_green"],
        head_kwargs={},
        print_diagnostics=False,
    )

    with pytest.raises(ValueError, match="wrist2vec_flex checkpoint or set freeze_tokenizer=False"):
        finetuning_module.Wrist2vecFinetuning(args, model_cfg, finetune_config=None, averaging_config=None)


def test_wrist2vec_finetuning_rejects_ckpt_missing_source_modules_when_frozen(monkeypatch):
    finetuning_module = pytest.importorskip("wrist2vec_flex.wrist2vec_finetuning")

    def _load_state_dict(_self, _state_dict, strict=False):
        return argparse.Namespace(
            missing_keys=["model.backbone.channel_source_encoder_mapping.ppg_green.source_fusion.query.weight"],
            unexpected_keys=[],
        )

    monkeypatch.setattr(finetuning_module.pl.LightningModule, "load_state_dict", _load_state_dict)
    module = finetuning_module.Wrist2vecFinetuning.__new__(finetuning_module.Wrist2vecFinetuning)
    module.args = argparse.Namespace(freeze_tokenizer=True)

    with pytest.raises(ValueError, match="wrist2vec_flex checkpoint or set freeze_tokenizer=False"):
        module.load_state_dict({}, strict=False)

    module.args = argparse.Namespace(freeze_tokenizer=False)
    result = module.load_state_dict({}, strict=False)

    assert result.missing_keys


def test_wrist2vec_downstream_channel_dropout_train_only():
    model = Wrist2vecDownstreamModel.__new__(Wrist2vecDownstreamModel)
    model.channel_dropout_rate = 1.0
    model.min_channels_after_dropout = 1
    tokens = {"ppg_green": torch.zeros(3, 2, 4), "acc_vm": torch.zeros(3, 2, 4)}
    channel_mask = torch.ones(3, 2, dtype=torch.bool)

    model.training = True
    train_mask = model._apply_channel_dropout(channel_mask, tokens, ["ppg_green", "acc_vm"])
    model.training = False
    eval_mask = model._apply_channel_dropout(channel_mask, tokens, ["ppg_green", "acc_vm"])

    assert train_mask.sum(dim=1).tolist() == [1, 1, 1]
    assert torch.equal(eval_mask, channel_mask)


def test_wrist2vec_concat_aggregation_rejects_missing_channels_by_default():
    agg = ConcatChannelAggregator(feature_dim=2, n_mods=2)
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    channel_mask = torch.tensor([[True, False], [False, True]])

    with pytest.raises(ValueError, match="requires all channels to be present"):
        agg([left, right], channel_mask=channel_mask)
