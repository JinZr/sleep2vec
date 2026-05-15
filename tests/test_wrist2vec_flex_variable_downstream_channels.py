from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from wrist2vec_flex.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    LayerMixConfig,
    ModelConfig,
    ProjectionConfig,
    SourceEmbeddingConfig,
    SourceFusionConfig,
    TemporalAggConfig,
    TokenizerConfig,
)
from wrist2vec_flex.downstream_model import Wrist2vecDownstreamModel
import wrist2vec_flex.downstreams.heads  # noqa: F401
from wrist2vec_flex.modules.channel_source_encoder import ChannelSourceEncoder
from wrist2vec_flex.utils import _build_finetune_loader
from wrist2vec_flex.wrist2vec_finetuning import Wrist2vecFinetuning

FEATURE_DIM = 4


class _RecordingBackbone(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.transformer_hidden_size = feature_dim
        self.cls_embedding = None
        self.seen_channel_names: list[str] = []
        self.seen_channel_to_logical: dict[str, str] = {}

    def _tokenize_all(self, tokens, *, channel_names=None, channel_to_logical=None, **_kwargs):
        self.seen_channel_names = list(channel_names or [])
        self.seen_channel_to_logical = dict(channel_to_logical or {})
        return {name: tokens[name] for name in self.seen_channel_names}

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        batch_size, num_tokens, _ = token_embeddings.shape
        attn_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=token_embeddings.device)
        if return_hidden_states:
            hidden_states = [token_embeddings, token_embeddings + 1.0, token_embeddings + 2.0]
            return token_embeddings, attn_mask, hidden_states
        return token_embeddings, attn_mask, None


class _ZeroTokenizer(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        return torch.zeros(x.shape[0], self.feature_dim, dtype=x.dtype, device=x.device)


class _SourceAwareBackbone(nn.Module):
    def __init__(self, model_config: ModelConfig, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.transformer_hidden_size = feature_dim
        self.cls_embedding = None
        self.tokenizers = nn.ModuleDict(
            {channel.name: _ZeroTokenizer(feature_dim) for channel in model_config.channels}
        )
        self.source_encoders = nn.ModuleDict(
            {
                channel.name: ChannelSourceEncoder.from_channel(
                    channel,
                    feature_dim=feature_dim,
                )
                for channel in model_config.channels
            }
        )
        self.seen_channel_names: list[str] = []
        self.seen_source_masks: dict[str, torch.Tensor] = {}

    def _tokenize_all(self, tokens, *, channel_names=None, channel_to_logical=None, source_mask=None, source_ids=None):
        self.seen_channel_names = list(channel_names or [])
        self.seen_source_masks = {}
        logical_lookup = dict(channel_to_logical or {})
        embeddings = {}
        for channel_name in self.seen_channel_names:
            logical_name = logical_lookup.get(channel_name, channel_name)
            mask = None if source_mask is None else source_mask.get(channel_name)
            ids = None if source_ids is None else source_ids.get(channel_name)
            if mask is not None:
                self.seen_source_masks[channel_name] = mask.detach().cpu().clone()
            embeddings[channel_name] = self.source_encoders[logical_name](
                tokens[channel_name],
                tokenizer=self.tokenizers[logical_name],
                source_mask=mask,
                source_ids=ids,
            )
        return embeddings

    def _token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False):
        batch_size, num_tokens, _ = token_embeddings.shape
        attn_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=token_embeddings.device)
        return token_embeddings, attn_mask, None


class _LossHarness:
    _compute_loss = Wrist2vecFinetuning._compute_loss
    _get_targets = Wrist2vecFinetuning._get_targets


def _channel(name: str, *, source_names: list[str] | None = None) -> ChannelConfig:
    return ChannelConfig(
        name=name,
        input_dim=4,
        source_names=list(source_names or []),
        source_fusion=SourceFusionConfig(name="masked_mean") if source_names else None,
        source_embedding=SourceEmbeddingConfig(enabled=False) if source_names else None,
        tokenizer=TokenizerConfig(name="linear", out_dim=FEATURE_DIM),
    )


def _model_config(*, channel_agg: str = "mean", head_name: str = "classification") -> ModelConfig:
    return ModelConfig(
        channels=[
            _channel("ppg_green", source_names=["ppg_green_gd1", "ppg_green_gd2"]),
            _channel("ppg_ir"),
            _channel("acc"),
            _channel("ecg"),
        ],
        backbone=BackboneConfig(name="roformer", hidden_size=FEATURE_DIM, num_hidden_layers=2, num_attention_heads=2),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=FEATURE_DIM, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            channel_agg=ChannelAggConfig(name=channel_agg),
            temporal_agg=TemporalAggConfig(name="mean"),
            name=head_name,
            dropout=0.0,
        ),
    )


def _downstream_model(
    model_config: ModelConfig,
    *,
    backbone: nn.Module | None = None,
    channel_names: list[str] | None = None,
    is_seq: bool = False,
    output_dim: int = 2,
    layer_mix_cfg: LayerMixConfig | None = None,
) -> Wrist2vecDownstreamModel:
    return Wrist2vecDownstreamModel(
        target="stage5" if is_seq else "sex",
        backbone=backbone or _RecordingBackbone(),
        channel_names=channel_names or ["ppg_green", "ppg_ir"],
        effective_channel_names=None,
        effective_channel_to_logical=None,
        output_dim=output_dim,
        is_classification=True,
        is_seq=is_seq,
        model_config=model_config,
        layer_mix_cfg=layer_mix_cfg,
        head_config=model_config.head,
        device="cpu",
    )


def _write_npz(path: Path, *, include_ppg_green: bool, include_ppg_ir: bool, include_acc: bool, include_stage5: bool):
    payload = {}
    if include_ppg_green:
        payload["ppg_green_gd1"] = np.arange(8, dtype=np.float32)
        payload["ppg_green_gd2"] = np.arange(8, 16, dtype=np.float32)
    if include_ppg_ir:
        payload["ppg_ir"] = np.arange(16, 24, dtype=np.float32)
    if include_acc:
        payload["acc"] = np.arange(24, 32, dtype=np.float32)
    if include_stage5:
        payload["stage5"] = np.array([0, 1], dtype=np.float32)
    np.savez(path, **payload)


def _variable_dataset_index(tmp_path: Path) -> Path:
    specs = [
        dict(include_ppg_green=True, include_ppg_ir=True, include_acc=True, include_stage5=True, sex="female"),
        dict(include_ppg_green=True, include_ppg_ir=True, include_acc=False, include_stage5=True, sex="male"),
        dict(include_ppg_green=False, include_ppg_ir=True, include_acc=True, include_stage5=True, sex="female"),
    ]
    rows = []
    for idx, spec in enumerate(specs):
        path = tmp_path / f"sample_{idx}.npz"
        _write_npz(
            path,
            include_ppg_green=spec["include_ppg_green"],
            include_ppg_ir=spec["include_ppg_ir"],
            include_acc=spec["include_acc"],
            include_stage5=spec["include_stage5"],
        )
        rows.append(
            {
                "path": str(path),
                "duration": 60,
                "split": "train",
                "source": "center_a",
                "age": 50 + idx,
                "sex": spec["sex"],
            }
        )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def _loader_args(index_path: Path, *, label_name: str, is_seq: bool, output_dim: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_backend="npz",
        label_name=label_name,
        label_source_name=label_name,
        auxiliary_label_source_names=[],
        data_channel_names=["ppg_green", "ppg_ir", "acc"],
        channel_input_dims={"ppg_green": 4, "ppg_ir": 4, "acc": 4, "ecg": 4},
        channel_source_names={"ppg_green": ["ppg_green_gd1", "ppg_green_gd2"]},
        allow_missing_feature_channels=True,
        min_feature_channels=1,
        finetune_preset_path=None,
        finetune_data_index=index_path,
        max_tokens=2,
        batch_size=3,
        num_workers=0,
        device="cpu",
        is_classification=True,
        is_multilabel=False,
        is_seq=is_seq,
        output_dim=output_dim,
    )


def _load_batch(args: argparse.Namespace):
    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=[],
        shuffle=False,
        is_train_set=False,
    )
    return next(iter(loader))


def _loss_info(logits: torch.Tensor, batch: dict, args: argparse.Namespace):
    harness = _LossHarness()
    harness.args = argparse.Namespace(
        device="cpu",
        label_name=args.label_name,
        label_source_name=getattr(args, "label_source_name", args.label_name),
        is_classification=args.is_classification,
        is_multilabel=False,
        is_seq=args.is_seq,
    )
    harness._classification_loss = nn.CrossEntropyLoss(ignore_index=-1)
    harness._multilabel_loss = nn.BCEWithLogitsLoss(reduction="none")
    harness._regression_loss = nn.MSELoss()
    return harness._compute_loss(logits, batch)


def test_downstream_model_uses_data_channel_subset_not_full_model_channels():
    model_config = _model_config()
    backbone = _RecordingBackbone()
    model = _downstream_model(model_config, backbone=backbone, channel_names=["ppg_green", "ecg"])
    batch = {
        "tokens": {
            "ppg_green": torch.randn(2, 3, FEATURE_DIM),
            "ecg": torch.randn(2, 3, FEATURE_DIM),
        },
        "channel_mask": torch.ones(2, 2, dtype=torch.bool),
        "length": torch.tensor([3, 3]),
    }

    logits = model(batch)

    assert model.logical_channel_names == ["ppg_green", "ppg_ir", "acc", "ecg"]
    assert model.channel_names == ["ppg_green", "ecg"]
    assert backbone.seen_channel_names == ["ppg_green", "ecg"]
    assert logits.shape == (2, 2)


def test_downstream_head_n_mods_and_layer_mix_rows_match_active_data_channels():
    model_config = _model_config()
    layer_mix_cfg = LayerMixConfig(enabled=True, shared_across_modalities=False, layer_indices=[1, 2])

    model = _downstream_model(
        model_config,
        channel_names=["ppg_green", "acc"],
        layer_mix_cfg=layer_mix_cfg,
    )

    assert model.n_channels == 2
    assert model.head.fusion.aggregator.n_mods == 2
    assert model.layer_mix.weight.shape[0] == 2
    assert list(model.layer_mix_snapshot()["rows"]) == ["ppg_green", "acc"]


def test_downstream_forward_does_not_require_model_channels_outside_data_subset():
    model_config = _model_config()
    backbone = _RecordingBackbone()
    model = _downstream_model(model_config, backbone=backbone, channel_names=["ppg_green", "acc"])
    full_width_mask = torch.tensor([[True, False, True, False], [True, False, True, False]])
    batch = {
        "tokens": {
            "ppg_green": torch.randn(2, 2, FEATURE_DIM),
            "acc": torch.randn(2, 2, FEATURE_DIM),
        },
        "channel_mask": full_width_mask,
        "length": torch.tensor([2, 2]),
    }

    logits = model(batch)

    assert backbone.seen_channel_names == ["ppg_green", "acc"]
    assert logits.shape == (2, 2)


def test_wrist2vec_finetuning_passes_data_channel_names_to_downstream(monkeypatch):
    finetuning_module = pytest.importorskip("wrist2vec_flex.wrist2vec_finetuning")
    captured = {}

    class _FakePretrainModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured["backbone_channel_names"] = list(kwargs["channel_names"])
            self.transformer_hidden_size = kwargs["transformer_hidden_size"]
            self.cls_embedding = None

        def set_tokenizers_trainable(self, trainable):
            self.tokenizers_trainable = trainable

    class _FakeDownstreamModel(nn.Module):
        def __init__(self, *args, channel_names, **kwargs):
            super().__init__()
            captured["downstream_channel_names"] = list(channel_names)

    monkeypatch.setattr(finetuning_module, "Wrist2vecPretrainModel", _FakePretrainModel)
    monkeypatch.setattr(finetuning_module, "Wrist2vecDownstreamModel", _FakeDownstreamModel)
    monkeypatch.setattr(finetuning_module, "DownstreamEvalVisualizer", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        device="cpu",
        data_channel_names=["ppg_green", "ecg"],
        label_name="sex",
        output_dim=2,
        is_classification=True,
        is_seq=False,
        pretrained_backbone_path=None,
        freeze_backbone_and_insert_lora=False,
        freeze_tokenizer=True,
        channel_dropout_rate=0.0,
        min_channels_after_dropout=1,
        source_dropout_rate=0.0,
        min_sources_after_dropout=1,
        head_kwargs={},
        print_diagnostics=False,
        diagnostics_steps=5,
    )

    finetuning_module.Wrist2vecFinetuning(args, _model_config(), finetune_config=None, averaging_config=None)

    assert captured["backbone_channel_names"] == ["ppg_green", "ppg_ir", "acc", "ecg"]
    assert captured["downstream_channel_names"] == ["ppg_green", "ecg"]


def test_stage5_downstream_forward_with_sample_varying_feature_channels(tmp_path: Path):
    args = _loader_args(_variable_dataset_index(tmp_path), label_name="stage5", is_seq=True, output_dim=5)
    batch = _load_batch(args)
    row_by_id = {sample_id: idx for idx, sample_id in enumerate(batch["id"])}

    assert set(batch["tokens"]) == {"ppg_green", "ppg_ir", "acc", "stage5"}
    assert batch["channel_mask"].shape == (3, 3)
    assert batch["channel_mask"][row_by_id[0]].tolist() == [True, True, True]
    assert batch["channel_mask"][row_by_id[1]].tolist() == [True, True, False]
    assert batch["channel_mask"][row_by_id[2]].tolist() == [False, True, True]

    model_config = _model_config()
    backbone = _SourceAwareBackbone(model_config)
    model = _downstream_model(
        model_config,
        backbone=backbone,
        channel_names=args.data_channel_names,
        is_seq=True,
        output_dim=5,
    )

    logits = model(batch)
    loss, valid_count = _loss_info(logits, batch, args)

    assert backbone.seen_channel_names == ["ppg_green", "ppg_ir", "acc"]
    assert logits.shape == (3, 2, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert valid_count == 6


def test_metadata_downstream_forward_with_sample_varying_feature_channels(tmp_path: Path):
    args = _loader_args(_variable_dataset_index(tmp_path), label_name="sex", is_seq=False, output_dim=2)
    batch = _load_batch(args)

    assert set(batch["tokens"]) == {"ppg_green", "ppg_ir", "acc"}
    assert "stage5" not in batch["tokens"]

    model_config = _model_config()
    model = _downstream_model(
        model_config,
        backbone=_SourceAwareBackbone(model_config),
        channel_names=args.data_channel_names,
        output_dim=2,
    )

    logits = model(batch)
    loss, valid_count = _loss_info(logits, batch, args)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert valid_count == 3


def test_downstream_forward_with_missing_multi_source_channel(tmp_path: Path):
    args = _loader_args(_variable_dataset_index(tmp_path), label_name="stage5", is_seq=True, output_dim=5)
    batch = _load_batch(args)
    missing_row = batch["id"].index(2)

    assert batch["channel_mask"][missing_row].tolist() == [False, True, True]
    assert batch["source_mask"]["ppg_green"][missing_row].tolist() == [True, False]
    assert torch.equal(
        batch["tokens"]["ppg_green"][missing_row],
        torch.zeros_like(batch["tokens"]["ppg_green"][missing_row]),
    )

    model_config = _model_config()
    backbone = _SourceAwareBackbone(model_config)
    model = _downstream_model(
        model_config,
        backbone=backbone,
        channel_names=args.data_channel_names,
        is_seq=True,
        output_dim=5,
    )

    logits = model(batch)

    assert backbone.seen_source_masks["ppg_green"][missing_row].tolist() == [True, False]
    assert torch.isfinite(logits).all()
