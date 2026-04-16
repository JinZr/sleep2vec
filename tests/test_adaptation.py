from __future__ import annotations

import pytest

from sleep2vec.config import BackboneConfig, ChannelConfig, ClsConfig, ModelConfig, ProjectionConfig, TokenizerConfig
from sleep2vec.pretrain_model import Sleep2vecPretrainModel
from sleep2vec.sleep2vec_adaptation import build_new_modality_pair_probs


def _adapt_model_config() -> ModelConfig:
    return ModelConfig(
        channels=[
            ChannelConfig(name="eeg", input_dim=4, tokenizer=TokenizerConfig(name="sundial", out_dim=8)),
            ChannelConfig(name="ecg", input_dim=4, tokenizer=TokenizerConfig(name="sundial", out_dim=8)),
            ChannelConfig(name="ppg", input_dim=4, tokenizer=TokenizerConfig(name="sundial", out_dim=8)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=1,
        ),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type="bert"),
        head=None,
    )


def _build_model() -> Sleep2vecPretrainModel:
    return Sleep2vecPretrainModel(
        transformer_hidden_size=8,
        transformer_num_hidden_layers=1,
        transformer_num_attention_heads=2,
        channel_names=["eeg", "ecg", "ppg"],
        projection=True,
        model_config=_adapt_model_config(),
        projection_config=_adapt_model_config().projection,
        device="cpu",
    )


def test_build_new_modality_pair_probs_splits_new_vs_legacy_pairs():
    probs = build_new_modality_pair_probs(
        [("eeg", "ecg"), ("eeg", "ppg"), ("ecg", "ppg")],
        new_channels=["ppg"],
        new_pair_ratio=0.7,
    )

    assert probs[("eeg", "ppg")] == pytest.approx(0.35)
    assert probs[("ecg", "ppg")] == pytest.approx(0.35)
    assert probs[("eeg", "ecg")] == pytest.approx(0.3)


def test_stage1_freeze_policy_trains_only_new_modalities_and_keeps_frozen_modules_in_eval():
    model = _build_model()
    model.apply_adaptation_freeze_policy(phase="stage1", new_channels=["ppg"], train_shared_projection=False)
    model.train()
    model.apply_forced_module_modes()

    param_groups = model.get_adaptation_param_groups(["ppg"])
    assert all(param.requires_grad for _, param in param_groups["new_modalities"])
    assert all(not param.requires_grad for _, param in param_groups["encoder_cls"])
    assert all(not param.requires_grad for _, param in param_groups["shared_projection"])
    assert all(not param.requires_grad for _, param in param_groups["legacy_modalities"])

    assert model.tokenizer_mapping["ppg"].training is True
    assert model.tokenizer_mapping["eeg"].training is False
    assert model.tokenizer_mapping["ecg"].training is False
    assert model.proj_head.training is False
    assert model.encoder.training is False


def test_stage2_freeze_policy_unfreezes_encoder_cls_and_all_tokenizers():
    model = _build_model()
    model.apply_adaptation_freeze_policy(phase="stage2", new_channels=["ppg"])
    model.train()
    model.apply_forced_module_modes()

    param_groups = model.get_adaptation_param_groups(["ppg"])
    for group_name in ["encoder_cls", "shared_projection", "legacy_modalities", "new_modalities"]:
        assert param_groups[group_name]
        assert all(param.requires_grad for _, param in param_groups[group_name])

    assert model.tokenizer_mapping["ppg"].training is True
    assert model.tokenizer_mapping["eeg"].training is True
    assert model.proj_head.training is True
    assert model.encoder.training is True
