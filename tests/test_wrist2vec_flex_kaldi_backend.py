from __future__ import annotations

import argparse
from pathlib import Path
import types

import pytest
import yaml

from wrist2vec_flex.common import apply_data_backend_args, apply_finetune_config
from wrist2vec_flex.config import load_finetune_config, load_pretrain_config
from wrist2vec_flex.data.default_dataset import SampleIndex
from wrist2vec_flex.data.kaldi_psg_dataset import _validate_single_source_kaldi_contract
from wrist2vec_flex.utils import _build_finetune_loader, _dataset_class_for_args, get_pretrain_dataloader


def _write_yaml(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _base_model_block() -> dict:
    return {
        "backbone": {
            "name": "roformer",
            "hidden_size": 8,
            "num_hidden_layers": 3,
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
                "name": "eeg",
                "input_dim": 4,
                "tokenizer": {"name": "linear", "out_dim": 8},
            },
            {
                "name": "ppg",
                "input_dim": 4,
                "tokenizer": {"name": "linear", "out_dim": 8},
            },
        ],
    }


def _pretrain_payload() -> dict:
    return {
        "model": _base_model_block(),
        "loss": {"name": "info_nce", "temperature": 0.2},
        "data": {"mask_rate": 0.1, "max_tokens": 4},
    }


def _finetune_payload() -> dict:
    payload = {
        "model": _base_model_block(),
        "data": {
            "max_tokens": 4,
            "data_channel_names": ["eeg", "ppg"],
            "finetune_data_index": "index.csv",
            "finetune_preset_path": None,
            "train_dataset_names": ["train_ds"],
            "test_dataset_names": ["test_ds"],
            "n_few_shot": 16,
        },
        "finetune": {
            "freeze_tokenizer": True,
            "lora": {
                "freeze_backbone_and_insert_lora": False,
                "insert_lora": False,
                "separate_adapters": False,
            },
            "task": {
                "type": "classification",
                "output_dim": 2,
                "is_seq": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
            },
        },
    }
    payload["model"]["head"] = {
        "name": "classification",
        "dropout": 0.1,
        "hidden_dim": None,
        "channel_agg": {"name": "mean", "kwargs": {}},
        "temporal_agg": {"name": "mean", "kwargs": {}},
    }
    return payload


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
        channel_source_names={"eeg": ["eeg"], "ppg": ["ppg"]},
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
        channel_source_names={"eeg": ["eeg"]},
        finetune_preset_path=None,
        finetune_data_index=Path("index.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=output_dim,
    )


def test_wrist2vec_config_parses_kaldi_backend_fields(tmp_path: Path):
    pretrain_payload = _pretrain_payload()
    pretrain_payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "/tmp/kaldi_root",
            "kaldi_manifest": "/tmp/kaldi_root/manifest.json",
        }
    )
    pretrain_bundle = load_pretrain_config(_write_yaml(tmp_path, pretrain_payload, "pretrain.yaml"))

    assert pretrain_bundle.data.backend == "kaldi"
    assert pretrain_bundle.data.kaldi_data_root == "/tmp/kaldi_root"
    assert pretrain_bundle.data.kaldi_manifest == "/tmp/kaldi_root/manifest.json"

    finetune_payload = _finetune_payload()
    finetune_payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "/tmp/kaldi_root",
            "kaldi_manifest": "/tmp/kaldi_root/manifest.json",
        }
    )
    finetune_bundle = load_finetune_config(_write_yaml(tmp_path, finetune_payload, "finetune.yaml"))

    assert finetune_bundle.data.backend == "kaldi"
    assert finetune_bundle.data.kaldi_data_root == "/tmp/kaldi_root"
    assert finetune_bundle.data.kaldi_manifest == "/tmp/kaldi_root/manifest.json"


@pytest.mark.parametrize(
    ("loader", "payload_factory"),
    [
        (load_pretrain_config, _pretrain_payload),
        (load_finetune_config, _finetune_payload),
    ],
)
def test_wrist2vec_config_rejects_invalid_data_backend(tmp_path: Path, loader, payload_factory):
    payload = payload_factory()
    payload["data"]["backend"] = "hdf5"
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.backend must be one of"):
        loader(config_path)


def test_wrist2vec_apply_finetune_config_rejects_reordered_data_channels(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"]["data_channel_names"] = ["ppg", "eeg"]
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="preserve model.channels order"):
        apply_finetune_config(args)


def test_wrist2vec_apply_finetune_config_rejects_concat_with_missing_feature_channels(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["head"]["channel_agg"]["name"] = "concat"
    payload["data"]["allow_missing_feature_channels"] = True
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="concat.*allow_missing_feature_channels"):
        apply_finetune_config(args)


def test_wrist2vec_apply_finetune_config_rejects_concat_with_channel_dropout(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["head"]["channel_agg"]["name"] = "concat"
    payload["data"].update(
        {
            "channel_dropout_rate": 0.5,
            "min_channels_after_dropout": 1,
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="concat.*channel_dropout_rate"):
        apply_finetune_config(args)


def test_wrist2vec_apply_finetune_config_populates_kaldi_backend(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "kaldi/root",
            "kaldi_manifest": "kaldi/root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    apply_finetune_config(args)

    assert args.data_backend == "kaldi"
    assert args.kaldi_data_root == Path("kaldi/root")
    assert args.kaldi_manifest == Path("kaldi/root/manifest.json")
    assert args.finetune_preset_path is None


def test_wrist2vec_apply_finetune_config_populates_source_dropout_args(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update({"source_dropout_rate": 0.25, "min_sources_after_dropout": 2})
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    apply_finetune_config(args)

    assert args.source_dropout_rate == pytest.approx(0.25)
    assert args.min_sources_after_dropout == 2


def test_wrist2vec_apply_finetune_config_rejects_kaldi_missing_manifest(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update({"backend": "kaldi", "kaldi_data_root": "kaldi/root"})
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="Kaldi backend requires explicit kaldi_data_root and kaldi_manifest"):
        apply_finetune_config(args)


def test_wrist2vec_apply_finetune_config_rejects_kaldi_preset_path(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "kaldi/root",
            "kaldi_manifest": "kaldi/root/manifest.json",
            "finetune_preset_path": "preset.pkl",
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        apply_finetune_config(args)


def test_wrist2vec_apply_data_backend_rejects_kaldi_pretrain_preset_path():
    args = argparse.Namespace(
        data_backend=None,
        kaldi_data_root=None,
        kaldi_manifest=None,
        pretrain_preset_path=Path("preset.pkl"),
        channel_names=["eeg"],
        channel_source_names={"eeg": ["eeg"]},
    )
    data_cfg = types.SimpleNamespace(
        backend="kaldi",
        kaldi_data_root="kaldi/root",
        kaldi_manifest="kaldi/root/manifest.json",
    )

    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        apply_data_backend_args(args, data_cfg, preset_attr="pretrain_preset_path")


def test_wrist2vec_kaldi_rejects_multi_source_config_before_loader(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["channels"][0]["source_names"] = ["eeg_a", "eeg_b"]
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "kaldi/root",
            "kaldi_manifest": "kaldi/root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="Kaldi backend does not support source-aware manifests yet"):
        apply_finetune_config(args)


def test_wrist2vec_kaldi_accepts_single_explicit_source_name():
    args = argparse.Namespace(
        data_backend=None,
        kaldi_data_root=None,
        kaldi_manifest=None,
        channel_names=["ecg"],
        channel_source_names={"ecg": ["lead1"]},
    )
    data_cfg = types.SimpleNamespace(
        backend="kaldi",
        kaldi_data_root="kaldi/root",
        kaldi_manifest="kaldi/root/manifest.json",
    )

    apply_data_backend_args(args, data_cfg)
    _validate_single_source_kaldi_contract(["ecg"], {"ecg": ["lead1"]})


def test_wrist2vec_get_pretrain_dataloader_routes_kaldi_kwargs(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("wrist2vec_flex.utils.KaldiPSGDataset", _DummyDataset)

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
    assert train_kwargs["channel_source_names"] == {"eeg": ["eeg"], "ppg": ["ppg"]}
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


def test_wrist2vec_build_finetune_loader_routes_kaldi_stage_channels(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("wrist2vec_flex.utils.KaldiPSGDataset", _DummyDataset)
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


def test_wrist2vec_build_finetune_loader_routes_kaldi_ahi_channels(monkeypatch):
    _reset_dummy()
    monkeypatch.setattr("wrist2vec_flex.utils.KaldiPSGDataset", _DummyDataset)
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


def test_wrist2vec_dataset_class_for_args_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown data backend"):
        _dataset_class_for_args(argparse.Namespace(data_backend="parquet"))
