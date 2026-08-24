from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert")
STANDALONE_VARIANTS = ("sleep2vec2", "sleep2expert")
MODEL_CHANNELS = ["heartbeat", "breath", *[f"channel_{index}" for index in range(9)]]


def _extractor(variant: str):
    if variant == "sleep2vec":
        return pytest.importorskip("sleep2vec.extract_embeddings", exc_type=ImportError)
    return importlib.import_module(f"{variant}.extract_embeddings")


def _config_args(module, tmp_path: Path, *, embedding_kind: str = "token") -> argparse.Namespace:
    return module.parse_args(
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
            embedding_kind,
            "--sequence-mode",
            "config-windows",
        ]
    )


def _mock_config_loading(module, monkeypatch: pytest.MonkeyPatch, *, preset_build=(None, None)) -> None:
    bundle = SimpleNamespace(
        data=SimpleNamespace(max_tokens=180, backend="npz"),
        model=SimpleNamespace(backbone=SimpleNamespace(name="roformer", config_overrides={})),
    )
    monkeypatch.setattr(module, "_load_config_data", lambda _path: {})
    monkeypatch.setattr(module, "load_pretrain_config", lambda _path: bundle)

    def apply_model_config_args(args, _model_cfg):
        args.channel_names = list(MODEL_CHANNELS)
        args.channel_input_dims = {channel: 120 for channel in MODEL_CHANNELS}
        args.channel_aliases = {}

    monkeypatch.setattr(module, "apply_model_config_args", apply_model_config_args)
    monkeypatch.setattr(
        module,
        "apply_data_backend_args",
        lambda args, _data_cfg, preset_attr: setattr(args, "data_backend", "npz"),
    )
    monkeypatch.setattr(module, "_load_preset_build_block", lambda _config: preset_build)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_config_windows_preserves_config_owned_preset_channels(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _extractor(variant)
    args = _config_args(module, tmp_path)
    _mock_config_loading(module, monkeypatch, preset_build=(["heartbeat", "breath"], 2))

    module._load_config_bundle(args)

    assert args.selected_channels is None
    assert args.channel_names == MODEL_CHANNELS
    assert args.dataset_channel_names == MODEL_CHANNELS


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_config_windows_rejects_dual_export(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _extractor(variant)
    args = _config_args(module, tmp_path, embedding_kind="both")
    _mock_config_loading(module, monkeypatch)

    with pytest.raises(ValueError, match="both requires --sequence-mode whole-night"):
        module._load_config_bundle(args)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_whole_night_rejects_fractional_tokens(variant: str, tmp_path: Path):
    module = _extractor(variant)
    signal_path = tmp_path / "night.npz"
    signal_path.touch()
    index_path = tmp_path / "index.csv"
    pd.DataFrame([{"path": str(signal_path), "split": "train", "duration": 45}]).to_csv(index_path, index=False)
    args = argparse.Namespace(data_index=[index_path], eval_split="train", max_source_tokens=2880)

    with pytest.raises(ValueError, match="aligned to 30-second tokens"):
        module._preflight_whole_night_index(args)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_preflight_rejects_dangling_output_symlink(variant: str, tmp_path: Path):
    module = _extractor(variant)
    output_dir = tmp_path / "output"
    output_dir.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        module._preflight_output_dir(output_dir)


class _PositionEmbedding(torch.nn.Module):
    def __init__(self, capacity: int, embedding_dim: int):
        super().__init__()
        weight = torch.arange(capacity * embedding_dim, dtype=torch.float32).view(capacity, embedding_dim)
        self.register_buffer("weight", weight, persistent=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_position_extension_updates_capacity(variant: str):
    module = _extractor(variant)
    encoder = SimpleNamespace(
        encoder=SimpleNamespace(embed_positions=_PositionEmbedding(4, 2)),
        config=SimpleNamespace(max_position_embeddings=4),
    )
    model = SimpleNamespace(get_encoder=lambda: encoder)

    training_capacity = module._extend_roformer_position_capacity(model, 8)

    assert training_capacity == 4
    assert encoder.encoder.embed_positions.weight.shape == (8, 2)
    assert encoder.config.max_position_embeddings == 8


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_run_hashes_preset_before_loading_data(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _extractor(variant)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    preset_path = tmp_path / "preset.pkl"
    preset_path.write_bytes(b"preset")
    expected_preset_hash = hashlib.sha256(b"preset").hexdigest()
    args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--ckpt-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--output-format",
            "npz",
            "--preset-path",
            str(preset_path),
        ]
    )
    model_cfg = SimpleNamespace(backbone=SimpleNamespace())
    model = SimpleNamespace()
    captured_hashes = {}
    monkeypatch.setattr(module, "_load_config_bundle", lambda _args: (SimpleNamespace(), model_cfg, "pretrain"))

    def build_loader(*_args, **_kwargs):
        preset_path.write_bytes(b"changed")
        return []

    monkeypatch.setattr(module, "_build_extraction_loader", build_loader)
    monkeypatch.setattr(module, "_finetune_adapters_enabled", lambda *_args: False)
    monkeypatch.setattr(module, "_build_backbone", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        module,
        "_load_backbone_checkpoint",
        lambda *_args, **_kwargs: module.CheckpointLoadPlan("pretrain", "ema_model."),
    )
    monkeypatch.setattr(module, "_validate_embedding_kind_compatible", lambda *_args: None)

    def extract(args, *_args, **_kwargs):
        captured_hashes.update(args.input_hashes)
        return args.output_dir / "manifest.json"

    monkeypatch.setattr(module, "_extract_and_write_embeddings", extract)

    module.run_extraction(args)

    assert captured_hashes["preset_sha256"] == expected_preset_hash


@pytest.mark.parametrize("variant", STANDALONE_VARIANTS)
def test_standalone_variant_extends_positions_before_real_cls_forward(variant: str):
    module = _extractor(variant)
    roformer = importlib.import_module(f"{variant}.backbones.roformer")
    cls_module = importlib.import_module(f"{variant}.cls")
    encoder = roformer.RoFormerEncoderModel(
        roformer.RoFormerConfig(
            vocab_size=1,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=4,
            attention_backend="sdpa",
        )
    ).eval()
    model = SimpleNamespace(get_encoder=lambda: encoder)
    training_prefix = encoder.encoder.embed_positions.weight.detach().clone()

    training_capacity = module._extend_roformer_position_capacity(model, 8)

    assert training_capacity == 4
    torch.testing.assert_close(encoder.encoder.embed_positions.weight[:4], training_prefix)
    cls_embedding = cls_module.BertClsEmbedding(8)
    tokens = torch.randn(1, 6, 8)
    tokens_with_cls, attention_mask = cls_embedding.add_cls_and_mask(tokens, torch.tensor([6]))
    with torch.no_grad():
        hidden = encoder(inputs_embeds=tokens_with_cls, attention_mask=attention_mask).last_hidden_state
    token_hidden, cls_hidden, token_mask = cls_embedding.split_hidden(hidden, attention_mask)

    assert hidden.shape == (1, 7, 8)
    assert token_hidden.shape == (1, 6, 8)
    assert cls_hidden.shape == (1, 8)
    assert token_mask.shape == (1, 6)
