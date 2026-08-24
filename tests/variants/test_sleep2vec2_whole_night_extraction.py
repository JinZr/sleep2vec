from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sleep2vec2 import extract_embeddings
from sleep2vec2.backbones.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding

MODEL_CHANNELS = ["heartbeat", "breath", *[f"channel_{index}" for index in range(9)]]


def _parse_whole_night_args(tmp_path: Path) -> argparse.Namespace:
    return extract_embeddings.parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--ckpt-path",
            str(tmp_path / "model.ckpt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--output-format",
            "npz",
            "--embedding-kind",
            "both",
            "--layer-index",
            "-1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--channels",
            "heartbeat",
            "breath",
            "--sequence-mode",
            "whole-night",
            "--max-source-tokens",
            "2880",
            "--data-index",
            str(tmp_path / "index.csv"),
        ]
    )


def _mock_config_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = SimpleNamespace(
        data=SimpleNamespace(max_tokens=180, backend="npz"),
        model=SimpleNamespace(backbone=SimpleNamespace(name="roformer", config_overrides={})),
    )
    monkeypatch.setattr(extract_embeddings, "_load_config_data", lambda _path: {})
    monkeypatch.setattr(extract_embeddings, "load_pretrain_config", lambda _path: bundle)

    def apply_model_config_args(args, _model_cfg):
        args.channel_names = list(MODEL_CHANNELS)
        args.channel_input_dims = {channel: 120 for channel in MODEL_CHANNELS}
        args.channel_aliases = {}

    monkeypatch.setattr(extract_embeddings, "apply_model_config_args", apply_model_config_args)
    monkeypatch.setattr(
        extract_embeddings,
        "apply_data_backend_args",
        lambda args, _data_cfg, preset_attr: setattr(args, "data_backend", "npz"),
    )
    monkeypatch.setattr(extract_embeddings, "_load_preset_build_block", lambda _config: (None, None))


def test_whole_night_cli_keeps_model_channels_and_selects_two_signal_channels(tmp_path: Path, monkeypatch):
    args = _parse_whole_night_args(tmp_path)
    _mock_config_loading(monkeypatch)

    extract_embeddings._load_config_bundle(args)

    assert args.selected_channels == ["heartbeat", "breath"]
    assert args.model_channel_names == MODEL_CHANNELS
    assert args.dataset_channel_names == ["heartbeat", "breath"]
    assert args.training_max_tokens == 180
    assert args.max_tokens == 2880


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("output_format", "kaldi"),
        ("layer_index", 1),
        ("batch_size", 2),
        ("max_source_tokens", 4096),
        ("selected_channels", ["heartbeat", "heartbeat"]),
        ("selected_channels", ["unknown"]),
    ],
)
def test_whole_night_cli_rejects_noncanonical_contract(tmp_path: Path, monkeypatch, attribute, value):
    args = _parse_whole_night_args(tmp_path)
    setattr(args, attribute, value)
    _mock_config_loading(monkeypatch)

    with pytest.raises(ValueError):
        extract_embeddings._load_config_bundle(args)


def test_whole_night_loader_constructs_one_unclipped_sample(tmp_path: Path):
    num_tokens = 181
    frames_per_token = 120
    npz_path = tmp_path / "night.npz"
    np.savez(
        npz_path,
        heartbeat=np.arange(num_tokens * frames_per_token, dtype=np.float32),
        breath=np.arange(num_tokens * frames_per_token, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": num_tokens * 30,
                "source": "cohorts",
            }
        ]
    ).to_csv(index_path, index=False)
    args = argparse.Namespace(
        channel_names=["heartbeat", "breath"],
        model_channel_names=list(MODEL_CHANNELS),
        selected_channels=["heartbeat", "breath"],
        dataset_channel_names=["heartbeat", "breath"],
        dataset_channel_input_dims={"heartbeat": frames_per_token, "breath": frames_per_token},
        channel_input_dims={channel: frames_per_token for channel in MODEL_CHANNELS},
        channel_aliases={},
        training_max_tokens=180,
        max_tokens=2880,
        max_source_tokens=2880,
        sequence_mode="whole-night",
        eval_split="train",
        override_dataset_names=[],
        data_backend="npz",
        kaldi_data_root=None,
        kaldi_manifest=None,
        preset_path=None,
        data_index=[index_path],
        batch_size=1,
        num_workers=0,
        device="cpu",
    )

    extract_embeddings._preflight_whole_night_index(args)
    loader = extract_embeddings._build_extraction_loader(args, SimpleNamespace(), "pretrain")
    batches = list(loader)

    assert len(batches) == 1
    batch = batches[0]
    assert batch["token_start"].tolist() == [0]
    assert batch["length"].tolist() == [num_tokens]
    assert set(batch["tokens"]) == {"heartbeat", "breath"}
    assert batch["tokens"]["heartbeat"].shape == (1, num_tokens, frames_per_token)
    assert batch["tokens"]["breath"].shape == (1, num_tokens, frames_per_token)


@pytest.mark.parametrize("invalid_channel", ["missing", "dtype", "length", "nonfinite"])
def test_whole_night_preflight_rejects_invalid_source_channel(tmp_path: Path, invalid_channel: str):
    heartbeat = np.ones(240, dtype=np.float32)
    breath = np.ones(240, dtype=np.float32)
    if invalid_channel == "dtype":
        breath = breath.astype(np.float64)
    elif invalid_channel == "length":
        breath = breath[:-1]
    elif invalid_channel == "nonfinite":
        breath[0] = np.nan
    arrays = {"heartbeat": heartbeat, "breath": breath}
    if invalid_channel == "missing":
        arrays.pop("breath")
    npz_path = tmp_path / "night.npz"
    np.savez(npz_path, **arrays)
    index_path = tmp_path / "index.csv"
    pd.DataFrame([{"path": str(npz_path), "split": "train", "duration": 60}]).to_csv(index_path, index=False)
    args = argparse.Namespace(
        data_index=[index_path],
        eval_split="train",
        max_source_tokens=2880,
        channel_names=["heartbeat", "breath"],
        channel_aliases={},
        channel_input_dims={"heartbeat": 120, "breath": 120},
    )

    with pytest.raises(ValueError):
        extract_embeddings._preflight_whole_night_index(args)


class _FakeCls:
    @property
    def has_cls(self):
        return True

    def split_hidden(self, hidden, attention_mask):
        return hidden[:, 1:], hidden[:, 0], attention_mask[:, 1:]


class _BothBackbone:
    def __init__(self):
        self.cls_embedding = _FakeCls()
        self.tokenizer_mapping = {
            "heartbeat": torch.nn.Identity(),
            "breath": torch.nn.Identity(),
        }
        self.forward_channels: list[str] = []

    def eval(self):
        return self

    def _tokenize_all(self, tokens):
        return tokens

    def _token_embeddings_to_hidden(
        self,
        token_embeddings,
        batch,
        *,
        return_hidden_states=False,
        modality_name=None,
    ):
        assert return_hidden_states is False
        self.forward_channels.append(modality_name)
        channel_offset = 100.0 if modality_name == "heartbeat" else 200.0
        cls = torch.full((token_embeddings.size(0), 1, token_embeddings.size(2)), channel_offset)
        hidden = torch.cat([cls, token_embeddings.to(torch.float32) + channel_offset], dim=1)
        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        return hidden, attention_mask, None


def _both_batch():
    return {
        "id": [0],
        "length": torch.tensor([3]),
        "token_start": torch.tensor([0]),
        "metadata": {
            "source": ["cohorts"],
            "path": ["/wujidata/jinzengrui/data/shenzhen/cohorts/night.npz"],
        },
        "tokens": {
            "heartbeat": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
            "breath": torch.tensor([[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]),
        },
    }


def test_both_final_layer_splits_cls_and_tokens_from_one_forward():
    model = _BothBackbone()
    batch = _both_batch()

    matrices, resolved_layer = extract_embeddings._encode_channel(
        model,
        batch,
        "heartbeat",
        batch["tokens"]["heartbeat"],
        -1,
        num_hidden_layers=1,
        embedding_kind="both",
    )

    assert model.forward_channels == ["heartbeat"]
    assert resolved_layer == 1
    assert set(matrices) == {"cls_embedding", "token_embedding"}
    assert matrices["cls_embedding"][0].shape == (1, 2)
    assert matrices["token_embedding"][0].shape == (3, 2)
    np.testing.assert_array_equal(matrices["cls_embedding"][0], np.array([[100.0, 100.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        matrices["token_embedding"][0],
        np.array([[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]], dtype=np.float32),
    )


def test_4096_position_table_preserves_prefix_and_is_not_checkpoint_state():
    training_table = RoFormerSinusoidalPositionalEmbedding(1536, 8)
    extraction_table = RoFormerSinusoidalPositionalEmbedding(4096, 8)

    torch.testing.assert_close(extraction_table.weight[:1536], training_table.weight)
    assert "weight" not in training_table.state_dict()
    assert "weight" not in extraction_table.state_dict()


def test_both_npz_export_writes_exact_keys_and_manifest_invariants(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    index_path = tmp_path / "index.csv"
    index_path.write_text("path,split,duration\n/wujidata/jinzengrui/data/shenzhen/cohorts/night.npz,train,90\n")
    signal_path = "/wujidata/jinzengrui/data/shenzhen/cohorts/night.npz"
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        output_format="npz",
        eval_split="train",
        channel_names=["heartbeat", "breath"],
        model_channel_names=list(MODEL_CHANNELS),
        selected_channels=["heartbeat", "breath"],
        layer_index=-1,
        device="cpu",
        config=config_path,
        ckpt_path=checkpoint_path,
        data_index=[index_path],
        embedding_kind="both",
        sequence_mode="whole-night",
        training_max_tokens=180,
        max_source_tokens=2880,
        training_position_capacity=1536,
        effective_position_capacity=4096,
        expected_tokens_by_path={signal_path: 3},
        expected_sample_count=1,
        expected_total_tokens=3,
        observed_min_source_tokens=3,
        observed_max_source_tokens=3,
    )
    model_cfg = SimpleNamespace(backbone=SimpleNamespace(hidden_size=2, num_hidden_layers=1))
    model = _BothBackbone()

    manifest_path = extract_embeddings._extract_and_write_embeddings(
        args,
        model,
        [_both_batch()],
        model_cfg,
        extract_embeddings.CheckpointLoadPlan("pretrain", "ema_model."),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["namespace"] == "sleep2vec2"
    assert manifest["embedding_kind"] == "both"
    assert manifest["sequence_mode"] == "whole-night"
    assert manifest["whole_night"]["declared_max_source_tokens"] == 2880
    assert manifest["channels"] == ["heartbeat", "breath"]
    assert manifest["model_channels"] == MODEL_CHANNELS
    assert manifest["resolved_layer_index"] == 1
    assert manifest["splits"]["train"]["sample_count"] == 1

    with (tmp_path / "output" / "manifests" / "train.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["token_start"] == "0"
    assert rows[0]["token_end"] == "3"
    assert rows[0]["num_tokens"] == "3"
    assert rows[0]["matrix_rows"] == "3"
    assert rows[0]["cls_matrix_rows"] == "1"
    assert json.loads(rows[0]["available_channels"]) == ["heartbeat", "breath"]

    sample_key = rows[0]["sample_key"]
    for channel in ("heartbeat", "breath"):
        npz_path = tmp_path / "output" / "channels" / "train" / channel / f"{sample_key}.npz"
        with np.load(npz_path) as npz:
            assert set(npz.files) == {"cls_embedding", "token_embedding"}
            assert npz["cls_embedding"].shape == (1, 2)
            assert npz["token_embedding"].shape == (3, 2)

    assert model.forward_channels == ["heartbeat", "breath"]
