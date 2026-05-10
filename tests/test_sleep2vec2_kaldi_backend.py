from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sleep2vec2.common import apply_finetune_config
from sleep2vec2.config import load_finetune_config, load_pretrain_config
from sleep2vec2.data.default_dataset import SampleIndex
from sleep2vec2.utils import _build_finetune_loader, _dataset_class_for_args, get_pretrain_dataloader


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


def test_sleep2vec2_config_parses_kaldi_backend_fields(tmp_path: Path):
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
def test_sleep2vec2_config_rejects_invalid_data_backend(tmp_path: Path, loader, payload_factory):
    payload = payload_factory()
    payload["data"]["backend"] = "hdf5"
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.backend must be one of"):
        loader(config_path)


def test_sleep2vec2_apply_finetune_config_populates_kaldi_backend(tmp_path: Path):
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


def test_sleep2vec2_apply_finetune_config_rejects_kaldi_missing_manifest(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update({"backend": "kaldi", "kaldi_data_root": "kaldi/root"})
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="Kaldi backend requires explicit kaldi_data_root and kaldi_manifest"):
        apply_finetune_config(args)


def test_sleep2vec2_apply_finetune_config_rejects_kaldi_preset_path(tmp_path: Path):
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


def test_sleep2vec2_get_pretrain_dataloader_routes_kaldi_kwargs(monkeypatch):
    utils_module = importlib.import_module("sleep2vec2.utils")
    _reset_dummy()
    monkeypatch.setattr(utils_module, "KaldiPSGDataset", _DummyDataset)

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


def test_sleep2vec2_build_finetune_loader_routes_kaldi_stage_channels(monkeypatch):
    utils_module = importlib.import_module("sleep2vec2.utils")
    _reset_dummy()
    monkeypatch.setattr(utils_module, "KaldiPSGDataset", _DummyDataset)
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


def test_sleep2vec2_build_finetune_loader_routes_kaldi_ahi_channels(monkeypatch):
    utils_module = importlib.import_module("sleep2vec2.utils")
    _reset_dummy()
    monkeypatch.setattr(utils_module, "KaldiPSGDataset", _DummyDataset)
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


def test_sleep2vec2_dataset_class_for_args_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown data backend"):
        _dataset_class_for_args(argparse.Namespace(data_backend="parquet"))


def test_sleep2vec2_run_inference_rejects_kaldi_inference_preset_override(monkeypatch, tmp_path: Path):
    infer_mod = importlib.import_module("sleep2vec2.infer")

    def _apply_config(args):
        args.data_backend = "kaldi"
        args.finetune_preset_path = None
        return argparse.Namespace(finetune=None, averaging=None), object()

    monkeypatch.setattr(infer_mod, "apply_finetune_config", _apply_config)

    args = argparse.Namespace(
        label_name="stage5",
        avg_ckpts=1,
        inference_preset_path=tmp_path / "preset.pkl",
    )

    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        infer_mod.run_inference(args)


def _require_kaldi_native_io():
    return pytest.importorskip("kaldi_native_io")


def _write_kaldi_root(
    root: Path,
    channel_input_dims: dict[str, int],
    matrices: dict[str, dict[str, np.ndarray]],
    *,
    split: str = "train",
) -> None:
    manifest_channels = _write_kaldi_split(root, split, channel_input_dims, matrices)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "backend": "kaldi_native_io",
                "splits": {
                    split: {
                        "manifest": f"manifests/{split}.csv",
                        "channels": manifest_channels,
                    }
                },
            }
        )
        + "\n"
    )


def _write_kaldi_split(
    root: Path,
    split: str,
    channel_input_dims: dict[str, int],
    matrices: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, int | str]]:
    kaldi_native_io = _require_kaldi_native_io()
    channels_dir = root / "channels" / split
    channels_dir.mkdir(parents=True, exist_ok=True)
    manifest_channels = {}
    for channel, input_dim in channel_input_dims.items():
        ark_path = channels_dir / f"{channel}.ark"
        scp_path = channels_dir / f"{channel}.scp"
        with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
            for key, matrix in matrices.get(channel, {}).items():
                writer.write(key, np.asarray(matrix, dtype=np.float32))
        lines = scp_path.read_text().splitlines()
        lines.sort(key=lambda line: line.split(maxsplit=1)[0])
        scp_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest_channels[channel] = {"input_dim": input_dim, "scp": f"channels/{split}/{channel}.scp"}
    return manifest_channels


def _write_manifest(root: Path, rows: list[dict], *, split: str = "train") -> Path:
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    path = manifests_dir / f"{split}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return root / "manifest.json"


def _row(sample_key: str, channels: list[str], *, start: int = 0, end: int = 2, **metadata):
    row = {
        "sample_key": sample_key,
        "record_key": sample_key.rsplit("_", 2)[0],
        "path": f"/original/{sample_key}.npz",
        "source": "center-a",
        "dataset": "mesa",
        "split": "train",
        "token_start": start,
        "token_end": end,
        "num_tokens": end - start,
        "age": 40,
        "sex": 1,
        "available_channels": json.dumps(channels),
    }
    row.update(metadata)
    return row


def test_sleep2vec2_kaldi_dataset_batch_contract_without_npz_reads(tmp_path: Path, monkeypatch) -> None:
    from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset

    keys = ["mesa_s1_000000_000002", "mesa_s2_000002_000005"]
    _write_kaldi_root(
        tmp_path,
        {"eeg": 3, "ppg": 2},
        {
            "eeg": {
                keys[0]: np.ones((2, 3), dtype=np.float32),
                keys[1]: np.full((3, 3), 2.0, dtype=np.float32),
            },
            "ppg": {
                keys[0]: np.full((2, 2), 3.0, dtype=np.float32),
                keys[1]: np.full((3, 2), 4.0, dtype=np.float32),
            },
        },
    )
    manifest = _write_manifest(
        tmp_path,
        [
            _row(keys[0], ["eeg", "ppg"], age=40, sex=1),
            _row(keys[1], ["eeg", "ppg"], start=2, end=5, age=41, sex=0),
        ],
    )

    def fail_load_npz(path):
        raise AssertionError(f"Unexpected NPZ read: {path}")

    monkeypatch.setattr("sleep2vec2.data.default_dataset.load_npz", fail_load_npz)

    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg"],
        channel_input_dims={"eeg": 3, "ppg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=3,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert batch["id"] == keys
    assert batch["length"].tolist() == [2, 3]
    assert batch["token_start"].tolist() == [0, 2]
    assert batch["pair"] == ("eeg", "ppg")
    assert batch["tokens"]["eeg"].shape == (2, 3, 3)
    assert batch["tokens"]["ppg"].shape == (2, 3, 2)
    assert batch["tokens"]["eeg"][0, 2].eq(0.0).all()
    assert batch["tokens"]["ppg"][0, 2].eq(0.0).all()
    assert batch["metadata"]["age"].tolist() == [40.0, 41.0]
    assert batch["metadata"]["sex"].tolist() == [1, 0]
    assert batch["metadata"]["source"] == ["center-a", "center-a"]
    assert batch["metadata"]["path"] == [f"/original/{key}.npz" for key in keys]


def test_sleep2vec2_kaldi_dataset_uses_requested_split_manifest_and_scps(tmp_path: Path) -> None:
    from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset

    train_key = "mesa_train_000000_000002"
    val_key = "mesa_val_000000_000002"
    dims = {"eeg": 2, "ppg": 2}
    train_channels = _write_kaldi_split(
        tmp_path,
        "train",
        dims,
        {
            "eeg": {train_key: np.full((2, 2), 1.0, dtype=np.float32)},
            "ppg": {train_key: np.full((2, 2), 2.0, dtype=np.float32)},
        },
    )
    val_channels = _write_kaldi_split(
        tmp_path,
        "val",
        dims,
        {
            "eeg": {val_key: np.full((2, 2), 7.0, dtype=np.float32)},
            "ppg": {val_key: np.full((2, 2), 8.0, dtype=np.float32)},
        },
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "backend": "kaldi_native_io",
                "splits": {
                    "train": {"manifest": "manifests/train.csv", "channels": train_channels},
                    "val": {"manifest": "manifests/val.csv", "channels": val_channels},
                },
            }
        )
        + "\n"
    )
    _write_manifest(tmp_path, [_row(train_key, ["eeg", "ppg"], split="train")], split="train")
    manifest = _write_manifest(tmp_path, [_row(val_key, ["eeg", "ppg"], split="val")], split="val")

    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg"],
        channel_input_dims=dims,
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["val"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert batch["id"] == [val_key]
    assert batch["tokens"]["eeg"].eq(7.0).all()
    assert batch["tokens"]["ppg"].eq(8.0).all()


def test_sleep2vec2_kaldi_dataset_missing_channels_uses_pair_first_sampler(tmp_path: Path) -> None:
    from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset
    from sleep2vec2.data.samplers import PairFirstBatchSampler

    keys = [
        "mesa_ab_000000_000002",
        "mesa_ac_000000_000002",
        "mesa_bc_000000_000002",
    ]
    available_by_key = {
        keys[0]: {"eeg", "ppg"},
        keys[1]: {"eeg", "ecg"},
        keys[2]: {"ppg", "ecg"},
    }
    matrices = {
        channel: {
            key: np.full((2, 2), float(i + 1), dtype=np.float32)
            for i, key in enumerate(keys)
            if channel in available_by_key[key]
        }
        for channel in ["eeg", "ppg", "ecg"]
    }
    _write_kaldi_root(tmp_path, {"eeg": 2, "ppg": 2, "ecg": 2}, matrices)
    manifest = _write_manifest(
        tmp_path,
        [
            _row(keys[0], ["eeg", "ppg"]),
            _row(keys[1], ["eeg", "ecg"]),
            _row(keys[2], ["ppg", "ecg"]),
        ],
    )

    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg", "ecg"],
        channel_input_dims={"eeg": 2, "ppg": 2, "ecg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        is_train_set=True,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    loader = dataset.dataloader(device="cpu")
    batches = list(loader)

    assert isinstance(loader.batch_sampler, PairFirstBatchSampler)
    assert set(loader.batch_sampler.pairs) == {("eeg", "ppg"), ("eeg", "ecg"), ("ppg", "ecg")}
    assert batches
    for batch in batches:
        pair = tuple(batch["pair"])
        assert set(batch["tokens"]) == set(pair)
        for sample_id in batch["id"]:
            assert set(pair).issubset(available_by_key[sample_id])


def test_sleep2vec2_kaldi_dataset_stage_and_ahi_labels(tmp_path: Path) -> None:
    from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset

    keys = ["mesa_s1_000000_000002", "mesa_s2_000000_000001"]
    _write_kaldi_root(
        tmp_path,
        {"ppg": 4, "stage5": 1, "ahi": 30},
        {
            "ppg": {
                keys[0]: np.ones((2, 4), dtype=np.float32),
                keys[1]: np.full((1, 4), 2.0, dtype=np.float32),
            },
            "stage5": {
                keys[0]: np.asarray([[0.0], [4.0]], dtype=np.float32),
                keys[1]: np.asarray([[2.0]], dtype=np.float32),
            },
            "ahi": {
                keys[0]: np.ones((2, 30), dtype=np.float32),
                keys[1]: np.full((1, 30), 3.0, dtype=np.float32),
            },
        },
    )
    manifest = _write_manifest(
        tmp_path,
        [
            _row(keys[0], ["ppg", "stage5", "ahi"], ahi=7.5, tst=321.0),
            _row(keys[1], ["ppg", "stage5", "ahi"], end=1, ahi=8.5, tst=111.0),
        ],
    )

    dataset = KaldiPSGDataset(
        channel_names=["ppg", "stage5", "ahi"],
        channel_input_dims={"ppg": 4, "stage5": 1, "ahi": 30},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        meta_data_names=["ahi", "tst"],
        meta_data_regression_names=["ahi", "tst"],
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch["tokens"]) == {"ppg", "stage5", "ahi"}
    assert batch["tokens"]["ppg"].shape == (2, 2, 4)
    assert batch["tokens"]["stage5"].shape == (2, 2, 1)
    assert batch["tokens"]["ahi"].shape == (2, 2, 30)
    assert batch["tokens"]["ppg"][1, 1].eq(0.0).all()
    assert batch["tokens"]["stage5"][1, 1].eq(-1.0).all()
    assert batch["tokens"]["ahi"][1, 1].eq(-1.0).all()
    assert batch["metadata"]["ahi"].tolist() == pytest.approx([7.5, 8.5])
    assert batch["metadata"]["tst"].tolist() == pytest.approx([321.0, 111.0])
    assert batch["mlm_mask"]["stage5"].sum().item() == 0
    assert batch["mlm_mask"]["ahi"].sum().item() == 0


def test_sleep2vec2_kaldi_dataset_reader_pool_works_with_multiple_workers(tmp_path: Path) -> None:
    from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset

    keys = [f"mesa_s{i}_000000_000002" for i in range(4)]
    _write_kaldi_root(
        tmp_path,
        {"eeg": 2, "ppg": 2},
        {
            "eeg": {key: np.full((2, 2), i, dtype=np.float32) for i, key in enumerate(keys)},
            "ppg": {key: np.full((2, 2), i + 10, dtype=np.float32) for i, key in enumerate(keys)},
        },
    )
    manifest = _write_manifest(tmp_path, [_row(key, ["eeg", "ppg"]) for key in keys])
    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg"],
        channel_input_dims={"eeg": 2, "ppg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    batches = list(dataset.dataloader(device="cpu"))

    assert [batch["id"][0] for batch in batches] == keys


def _converter_config(tmp_path: Path, channel_dims: dict[str, int], preset_build: dict | None = None) -> Path:
    path = tmp_path / "converter.yaml"
    payload = {
        "model": {
            "channels": [{"name": name, "input_dim": input_dim} for name, input_dim in channel_dims.items()],
        }
    }
    if preset_build is not None:
        payload["preset_build"] = preset_build
    path.write_text(yaml.safe_dump(payload))
    return path


def _scp_keys(path: Path) -> list[str]:
    return [line.split(maxsplit=1)[0] for line in path.read_text().splitlines() if line.strip()]


def _read_matrix(scp_path: Path, key: str) -> np.ndarray:
    kaldi_native_io = _require_kaldi_native_io()
    with kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{scp_path}") as reader:
        return np.asarray(reader[key], dtype=np.float32)


def test_sleep2vec2_converter_roundtrip_writes_manifest_and_matching_scp(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.data.psg_pretrain_dataset import _build_channel_registry
    from sleep2vec2.data.utils import load_npz
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(tmp_path, {"eeg": 4, "ppg": 8})
    actual_root = tmp_path / "actual"
    original_root = tmp_path / "original"
    actual_root.mkdir()
    npz_path = actual_root / "sample.npz"
    original_npz_path = original_root / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(16, dtype=np.float32),
        ppg=np.arange(32, dtype=np.float32),
        stage5=np.arange(4, dtype=np.float32),
        ah_event=np.arange(120, dtype=np.float32),
        ahi=np.asarray(7.0, dtype=np.float32),
        tst=np.asarray(33.0, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(original_npz_path),
                "source": "original_source",
                "dataset": "mesa",
                "split": "train",
                "duration": 120,
                "session_id": "night 1",
                "age": 50,
                "sex": 1,
                "eeg_mask": 1,
                "ppg_mask": 1,
                "stage_mask": 1,
                "ah_event_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
                "--stride-tokens",
                "2",
                "--token-sec",
                "30",
                "--channels-from-config",
                "--extra-channels",
                "stage5",
                "ahi",
                "--source-field",
                "dataset",
                "--path-prefix-map",
                f"{original_root}={actual_root}",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert manifest["sample_key"].tolist() == [
        "mesa_night_1_000000_000002",
        "mesa_night_1_000002_000004",
    ]
    assert manifest.loc[0, "path"] == str(original_npz_path)
    assert manifest.loc[0, "source"] == "original_source"
    assert manifest.loc[0, "sample_source"] == "mesa"
    assert manifest.loc[0, "ahi"] == 7.0
    assert manifest.loc[0, "tst"] == 33.0
    assert json.loads(manifest.loc[0, "available_channels"]) == ["eeg", "ppg", "stage5", "ahi"]

    for channel in ["eeg", "ppg", "stage5", "ahi"]:
        assert _scp_keys(output_dir / "channels" / "train" / f"{channel}.scp") == manifest["sample_key"].tolist()

    registry = _build_channel_registry(
        channel_names=["eeg", "ppg", "stage5", "ahi"],
        channel_input_dims={"eeg": 4, "ppg": 8, "stage5": 1, "ahi": 30},
        mask_rate=0.0,
    )
    with load_npz(str(npz_path)) as npz:
        expected_eeg = registry["eeg"][1](registry["eeg"][0](npz, 0, 2)).numpy()
        expected_ppg = registry["ppg"][1](registry["ppg"][0](npz, 0, 2)).numpy()
        expected_stage5 = registry["stage5"][1](registry["stage5"][0](npz, 0, 2)).numpy()
        expected_ahi = registry["ahi"][1](registry["ahi"][0](npz, 0, 2)).numpy()

    key = "mesa_night_1_000000_000002"
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "eeg.scp", key), expected_eeg)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "ppg.scp", key), expected_ppg)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "stage5.scp", key), expected_stage5)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "ahi.scp", key), expected_ahi)

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["format_version"] == 2
    assert manifest_json["backend"] == "kaldi_native_io"
    assert manifest_json["splits"]["train"]["manifest"] == "manifests/train.csv"
    assert manifest_json["splits"]["train"]["channels"]["eeg"] == {"input_dim": 4, "scp": "channels/train/eeg.scp"}


def test_sleep2vec2_converter_writes_split_specific_manifests_and_sorted_scps(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "z",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "val",
                "duration": 60,
                "session_id": "b",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "a",
                "eeg_mask": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
                "--num-workers",
                "2",
                "--channels-from-config",
            ]
        )
    )

    train_manifest_path = output_dir / "manifests" / "train.csv"
    val_manifest_path = output_dir / "manifests" / "val.csv"
    train_scp_path = output_dir / "channels" / "train" / "eeg.scp"
    val_scp_path = output_dir / "channels" / "val" / "eeg.scp"
    assert train_manifest_path.exists()
    assert val_manifest_path.exists()
    assert train_scp_path.exists()
    assert val_scp_path.exists()
    assert (output_dir / "manifest.json").exists()
    assert not (output_dir / "manifest.csv").exists()

    train_manifest = pd.read_csv(train_manifest_path, low_memory=False)
    val_manifest = pd.read_csv(val_manifest_path, low_memory=False)
    assert train_manifest["split"].tolist() == ["train", "train"]
    assert val_manifest["split"].tolist() == ["val"]
    assert _scp_keys(train_scp_path) == sorted(train_manifest["sample_key"].tolist())
    assert _scp_keys(val_scp_path) == sorted(val_manifest["sample_key"].tolist())

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["format_version"] == 2
    assert manifest_json["splits"]["train"]["channels"]["eeg"] == {
        "input_dim": 4,
        "scp": "channels/train/eeg.scp",
    }
    assert manifest_json["splits"]["val"]["channels"]["eeg"] == {
        "input_dim": 4,
        "scp": "channels/val/eeg.scp",
    }


def test_sleep2vec2_converter_honors_preset_build_required_channels(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(
        tmp_path,
        {"ppg": 8},
        preset_build={"required_channels": ["ppg", "stage5"], "min_channels": 2},
    )
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.arange(16, dtype=np.float32), stage5=np.arange(2, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "ppg_mask": 1,
                "stage_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert json.loads(manifest.loc[0, "available_channels"]) == ["ppg", "stage5"]
    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["splits"]["train"]["channels"]["stage5"] == {
        "input_dim": 1,
        "scp": "channels/train/stage5.scp",
    }
    assert _scp_keys(output_dir / "channels" / "train" / "stage5.scp") == ["mesa_s1_000000_000002"]


def test_sleep2vec2_converter_auto_adds_stage5_when_preset_build_requires_ahi(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(
        tmp_path,
        {"ppg": 8},
        preset_build={"required_channels": ["ppg", "ahi"], "min_channels": 2},
    )
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        ppg=np.arange(16, dtype=np.float32),
        stage5=np.arange(2, dtype=np.float32),
        ah_event=np.arange(60, dtype=np.float32),
        ahi=np.asarray(7.0, dtype=np.float32),
        tst=np.asarray(33.0, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "ppg_mask": 1,
                "stage_mask": 1,
                "ah_event_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert json.loads(manifest.loc[0, "available_channels"]) == ["ppg", "ahi", "stage5"]
    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["splits"]["train"]["channels"]["stage5"] == {
        "input_dim": 1,
        "scp": "channels/train/stage5.scp",
    }


def test_sleep2vec2_converter_trims_one_token_channel_length_difference(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(12, dtype=np.float32), ppg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 90,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "3",
                "--channels-from-config",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    key = "mesa_s1_000000_000003"
    assert manifest.loc[0, "sample_key"] == key
    assert manifest.loc[0, "token_end"] == 2
    assert manifest.loc[0, "num_tokens"] == 2
    assert _read_matrix(output_dir / "channels" / "train" / "eeg.scp", key).shape == (2, 4)
    assert _read_matrix(output_dir / "channels" / "train" / "ppg.scp", key).shape == (2, 4)


def test_sleep2vec2_converter_rejects_channel_length_difference_greater_than_one(tmp_path: Path):
    _require_kaldi_native_io()
    from sleep2vec2.preprocess.convert_npz_to_kaldi import convert, parse_args

    config_path = _converter_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(16, dtype=np.float32), ppg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 120,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="differing by more than one"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "4",
                    "--channels-from-config",
                ]
            )
        )


def test_sleep2vec2_reader_pool_opens_sorted_scp_reader(tmp_path: Path, monkeypatch) -> None:
    from sleep2vec2.data.kaldi_io import KaldiChannelSpec, KaldiReaderPool

    opened_specs = []

    class FakeReader:
        def __contains__(self, key: str) -> bool:
            return key == "sample-a"

        def __getitem__(self, key: str) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    class FakeKaldiNativeIO:
        def RandomAccessFloatMatrixReader(self, spec: str) -> FakeReader:
            opened_specs.append(spec)
            return FakeReader()

    monkeypatch.setattr(KaldiReaderPool, "_import_kaldi_native_io", lambda self: FakeKaldiNativeIO())

    pool = KaldiReaderPool(
        tmp_path,
        {"eeg": KaldiChannelSpec(name="eeg", input_dim=3, scp_path=Path("channels/train/eeg.scp"))},
    )

    pool.read_matrix("eeg", "sample-a")

    assert opened_specs == [f"s,scp:{tmp_path / 'channels' / 'train' / 'eeg.scp'}"]
