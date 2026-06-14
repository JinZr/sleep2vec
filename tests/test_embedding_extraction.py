from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sleep2vec import extract_embeddings


def test_infer_checkpoint_load_plan_prefers_pretrain_ema():
    state = {
        "model.encoder.weight": torch.tensor([1.0]),
        "ema_model.encoder.weight": torch.tensor([2.0]),
    }

    plan = extract_embeddings._infer_checkpoint_load_plan(state)

    assert plan.checkpoint_kind == "pretrain"
    assert plan.checkpoint_prefix == "ema_model."


def test_infer_checkpoint_load_plan_prefers_finetune_ema_backbone():
    state = {
        "model.backbone.encoder.weight": torch.tensor([1.0]),
        "ema_model.backbone.encoder.weight": torch.tensor([2.0]),
        "model.head.classifier.weight": torch.tensor([3.0]),
    }

    plan = extract_embeddings._infer_checkpoint_load_plan(state)

    assert plan.checkpoint_kind == "finetune"
    assert plan.checkpoint_prefix == "ema_model.backbone."


def test_infer_checkpoint_load_plan_rejects_downstream_without_backbone():
    state = {"model.head.classifier.weight": torch.tensor([1.0])}

    with pytest.raises(ValueError, match="no backbone subtree"):
        extract_embeddings._infer_checkpoint_load_plan(state)


def test_infer_checkpoint_load_plan_rejects_mixed_layouts():
    state = {
        "model.encoder.weight": torch.tensor([1.0]),
        "model.backbone.encoder.weight": torch.tensor([2.0]),
    }

    with pytest.raises(ValueError, match="mixes downstream and pretrain-only"):
        extract_embeddings._infer_checkpoint_load_plan(state)


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.embedding_projection = torch.nn.Linear(1, 1, bias=False)


class _TinyClsBackbone(_TinyBackbone):
    def __init__(self):
        super().__init__()
        self.cls_embedding = torch.nn.Module()
        self.cls_embedding.cls_token = torch.nn.Parameter(torch.ones(1))


def test_load_backbone_checkpoint_rejects_missing_keys(tmp_path: Path):
    ckpt_path = tmp_path / "partial.ckpt"
    torch.save({"state_dict": {"model.encoder.weight": torch.ones(1, 1)}}, ckpt_path)

    with pytest.raises(ValueError, match="missing keys required"):
        extract_embeddings._load_backbone_checkpoint(_TinyBackbone(), ckpt_path, "cpu")


def test_load_backbone_checkpoint_rejects_adapter_keys_when_disabled(tmp_path: Path):
    ckpt_path = tmp_path / "adapter.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.backbone.encoder.weight": torch.ones(1, 1),
                "model.backbone.encoder.query.lora_A.default.weight": torch.ones(1, 1),
            }
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="contains adapter weights"):
        extract_embeddings._load_backbone_checkpoint(_TinyBackbone(), ckpt_path, "cpu")


def test_load_backbone_checkpoint_reports_unexpected_cls_weights(tmp_path: Path):
    ckpt_path = tmp_path / "cls.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.weight": torch.ones(1, 1),
                "model.embedding_projection.weight": torch.ones(1, 1),
                "model.embedding_projection.bias": torch.ones(1),
                "model.cls_embedding.cls_token": torch.ones(1),
            }
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="does not enable CLS embeddings"):
        extract_embeddings._load_backbone_checkpoint(_TinyBackbone(), ckpt_path, "cpu")


def test_load_backbone_checkpoint_reports_missing_cls_weights(tmp_path: Path):
    ckpt_path = tmp_path / "no_cls.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.weight": torch.ones(1, 1),
                "model.embedding_projection.weight": torch.ones(1, 1),
                "model.embedding_projection.bias": torch.ones(1),
            }
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="missing CLS embedding weights"):
        extract_embeddings._load_backbone_checkpoint(_TinyClsBackbone(), ckpt_path, "cpu")


def test_select_layer_state_supports_projected_input_positive_and_final():
    states = (
        torch.full((1, 2, 1), 0.0),
        torch.full((1, 2, 1), 1.0),
        torch.full((1, 2, 1), 2.0),
    )

    selected, resolved = extract_embeddings._select_layer_state(states, 0, num_hidden_layers=2)
    assert resolved == 0
    assert torch.equal(selected, states[0])

    selected, resolved = extract_embeddings._select_layer_state(states, 1, num_hidden_layers=2)
    assert resolved == 1
    assert torch.equal(selected, states[1])

    selected, resolved = extract_embeddings._select_layer_state(states, -1, num_hidden_layers=2)
    assert resolved == 2
    assert torch.equal(selected, states[2])


def test_trim_hidden_strips_cls_and_trims_padding():
    class _FakeCls:
        def split_hidden(self, hidden, attention_mask):
            return hidden[:, 1:], hidden[:, 0], None

    model = SimpleNamespace(cls_embedding=_FakeCls())
    hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).view(2, 4, 1)
    lengths = torch.tensor([2, 1])

    rows = extract_embeddings._trim_hidden_to_numpy(model, hidden, None, lengths, embedding_kind="token")

    assert [row.shape for row in rows] == [(2, 1), (1, 1)]
    np.testing.assert_array_equal(rows[0], np.array([[1.0], [2.0]], dtype=np.float32))
    np.testing.assert_array_equal(rows[1], np.array([[5.0]], dtype=np.float32))


def test_trim_hidden_cls_requested_without_cls_embedding_errors():
    model = SimpleNamespace(cls_embedding=None)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    lengths = torch.tensor([1])

    with pytest.raises(ValueError, match="Requested --embedding-kind cls"):
        extract_embeddings._trim_hidden_to_numpy(model, hidden, None, lengths, embedding_kind="cls")


def test_trim_hidden_cls_returns_one_row_per_sample():
    class _FakeCls:
        @property
        def has_cls(self):
            return True

        def split_hidden(self, hidden, attention_mask):
            return hidden[:, 1:], hidden[:, 0], None

    model = SimpleNamespace(cls_embedding=_FakeCls())
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    lengths = torch.tensor([3, 2])

    rows = extract_embeddings._trim_hidden_to_numpy(model, hidden, None, lengths, embedding_kind="cls")

    assert [row.shape for row in rows] == [(1, 2), (1, 2)]
    np.testing.assert_array_equal(rows[0], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(rows[1], np.array([[7.0, 8.0]], dtype=np.float32))


class _DummyBackbone:
    cls_embedding = None

    def eval(self):
        return self

    def _tokenize_all(self, tokens):
        return tokens

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        assert return_hidden_states is True
        attention_mask = torch.ones(token_embeddings.shape[:2], dtype=torch.bool)
        hidden_states = (token_embeddings.to(torch.float32), token_embeddings.to(torch.float32) + 10.0)
        return hidden_states[-1], attention_mask, hidden_states


class _AdapterEncoder:
    def __init__(self):
        self.active_adapter = None

    def set_adapter(self, adapter_name: str):
        self.active_adapter = adapter_name


class _AdapterBackbone(_DummyBackbone):
    _extract_separate_adapters = True

    def __init__(self):
        self.encoder = _AdapterEncoder()

    def get_encoder(self):
        return self.encoder


def test_encode_channel_sets_separate_adapter():
    model = _AdapterBackbone()
    batch = _dummy_batch()

    matrices, resolved = extract_embeddings._encode_channel(
        model,
        batch,
        "ppg",
        batch["tokens"]["ppg"],
        -1,
        num_hidden_layers=1,
        embedding_kind="token",
    )

    assert resolved == 1
    assert model.encoder.active_adapter == "ch_ppg"
    assert matrices[0].shape == (2, 2)


class _FakeCls:
    @property
    def has_cls(self):
        return True

    def split_hidden(self, hidden, attention_mask):
        return hidden[:, 1:], hidden[:, 0], attention_mask[:, 1:]


class _DummyClsBackbone(_DummyBackbone):
    def __init__(self):
        self.cls_embedding = _FakeCls()


def _dummy_args(tmp_path: Path, output_format: str, embedding_kind: str = "token") -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / output_format,
        output_format=output_format,
        eval_split="test",
        channel_names=["ppg"],
        layer_index=-1,
        device="cpu",
        config=Path("config.yaml"),
        ckpt_path=Path("model.ckpt"),
        embedding_kind=embedding_kind,
    )


def _dummy_model_cfg() -> SimpleNamespace:
    return SimpleNamespace(backbone=SimpleNamespace(hidden_size=2, num_hidden_layers=1))


def _dummy_batch():
    return {
        "id": [0],
        "length": torch.tensor([2]),
        "token_start": torch.tensor([3]),
        "metadata": {"source": ["mesa"], "path": ["/tmp/s1.npz"]},
        "tokens": {"ppg": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])},
    }


def test_npz_export_writes_manifest_and_embedding_matrix(tmp_path: Path):
    manifest_path = extract_embeddings._extract_and_write_embeddings(
        _dummy_args(tmp_path, "npz"),
        _DummyBackbone(),
        [_dummy_batch()],
        _dummy_model_cfg(),
        extract_embeddings.CheckpointLoadPlan("pretrain", "ema_model."),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["namespace"] == "sleep2vec"
    assert manifest["output_format"] == "npz"
    assert manifest["embedding_kind"] == "token"
    assert manifest["resolved_layer_index"] == 1

    rows = pd.read_csv(tmp_path / "npz" / "manifests" / "test.csv")
    assert rows.loc[0, "sample_key"] == "mesa_tmp_s1_000003_000005"
    assert rows.loc[0, "num_tokens"] == 2
    assert json.loads(rows.loc[0, "available_channels"]) == ["ppg"]

    matrix_path = tmp_path / "npz" / "channels" / "test" / "ppg" / "mesa_tmp_s1_000003_000005.npz"
    with np.load(matrix_path) as npz:
        np.testing.assert_array_equal(
            npz["embedding"],
            np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32),
        )


def test_npz_export_writes_cls_embedding_matrix(tmp_path: Path):
    manifest_path = extract_embeddings._extract_and_write_embeddings(
        _dummy_args(tmp_path, "npz", embedding_kind="cls"),
        _DummyClsBackbone(),
        [_dummy_batch()],
        _dummy_model_cfg(),
        extract_embeddings.CheckpointLoadPlan("pretrain", "ema_model."),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["embedding_kind"] == "cls"

    rows = pd.read_csv(tmp_path / "npz" / "manifests" / "test.csv")
    assert rows.loc[0, "sample_key"] == "mesa_tmp_s1_000003_000005"
    assert rows.loc[0, "num_tokens"] == 2

    matrix_path = tmp_path / "npz" / "channels" / "test" / "ppg" / "mesa_tmp_s1_000003_000005.npz"
    with np.load(matrix_path) as npz:
        np.testing.assert_array_equal(npz["embedding"], np.array([[11.0, 12.0]], dtype=np.float32))


def test_kaldi_export_writes_scp_and_matrix(tmp_path: Path):
    kaldi_native_io = pytest.importorskip("kaldi_native_io")
    manifest_path = extract_embeddings._extract_and_write_embeddings(
        _dummy_args(tmp_path, "kaldi"),
        _DummyBackbone(),
        [_dummy_batch()],
        _dummy_model_cfg(),
        extract_embeddings.CheckpointLoadPlan("finetune", "model.backbone."),
    )

    manifest = json.loads(manifest_path.read_text())
    assert "format_version" not in manifest
    assert manifest["output_format"] == "kaldi"
    assert manifest["splits"]["test"]["channels"]["ppg"]["input_dim"] == 2
    assert "hidden_size" not in manifest["splits"]["test"]["channels"]["ppg"]

    scp_path = tmp_path / "kaldi" / "channels" / "test" / "ppg.scp"
    assert scp_path.exists()
    with kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{scp_path}") as reader:
        matrix = reader["mesa_tmp_s1_000003_000005"]

    np.testing.assert_allclose(
        matrix,
        np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32),
    )
