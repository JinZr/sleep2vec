from dataclasses import asdict
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sleep2vec2 import config
from sleep2vec2.sleep2vec_finetuning import Sleep2vecFinetuning


def test_sleep2vec2_cli_advertises_weights_only_init():
    result = subprocess.run([sys.executable, "-m", "sleep2vec2.finetune", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--init-checkpoint-path" in result.stdout


def _module(*, checkpoint=None, preset="full", lr=0.001):
    model = config.ModelConfig(
        channels=[
            config.ChannelConfig(name="ppg", input_dim=8, tokenizer=config.TokenizerConfig(name="linear", out_dim=16))
        ],
        backbone=config.BackboneConfig(
            name="roformer",
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1,
            config_overrides={"intermediate_size": 32, "max_position_embeddings": 16},
        ),
        projection=config.ProjectionConfig(name="simclr", enabled=True, hidden_dim=16, out_dim=8),
        cls=config.ClsConfig(downstream="tokens", embedding_type=None),
        head=config.HeadConfig(
            name="classification",
            channel_agg=config.ChannelAggConfig(name="mean"),
            temporal_agg=config.TemporalAggConfig(name="mean"),
            hidden_dim=8,
            dropout=0.0,
        ),
    )
    tuning = config.FinetuneTuningConfig(
        preset=preset,
        groups={
            name: config.FinetuneGroupConfig(train=train, lr_scale=scale)
            for name, (train, scale) in config._FINETUNE_TUNING_PRESETS[preset].items()
        },
    )
    finetune = config.FinetuneConfig(
        tuning=tuning,
        layer_mix=config.LayerMixConfig(enabled=True, layer_indices=[1, 2]),
        task=config.TaskConfig(
            type="classification", output_dim=2, is_seq=False, monitor="val_sex_acc", monitor_mod="max"
        ),
    )
    args = SimpleNamespace(
        device="cpu",
        label_name="sex",
        output_dim=2,
        is_classification=True,
        is_seq=False,
        pretrained_backbone_path=None,
        init_checkpoint_path=checkpoint,
        ckpt_path=None,
        lr=lr,
        weight_decay=0.01,
        warmup_steps=0,
    )
    return Sleep2vecFinetuning(args, model, finetune_config=finetune)


def _checkpoint(module):
    return {
        "state_dict": module.state_dict(),
        "model_config": asdict(module.model_config),
        "finetune_config": asdict(module.finetune_config),
        "epoch": 52,
        "global_step": 12345,
        "optimizer_states": [{"state": {0: {"step": 12345}}}],
        "lr_schedulers": [{"last_epoch": 12345}],
    }


def test_full_init_preserves_backbone_head_and_layer_mix_with_fresh_optimizer(tmp_path):
    source = _module()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.fill_((index + 1) / 100)
    path = tmp_path / "selected.ckpt"
    torch.save(_checkpoint(source), path)

    target = _module(checkpoint=str(path), preset="head_only", lr=0.004)

    assert target.current_epoch == 0
    assert target.global_step == 0
    assert any("layer_mix" in name for name in source.state_dict())
    assert any("head" in name for name in source.state_dict())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in target.backbone.encoder.parameters())
    assert any(parameter.requires_grad for parameter in target.model.head.parameters())
    target._trainer = SimpleNamespace(estimated_stepping_batches=100)
    optimizers, schedulers = target.configure_optimizers()
    assert optimizers[0].state == {}
    assert {group["initial_lr"] for group in optimizers[0].param_groups} == {0.004}
    assert schedulers[0]["scheduler"].last_epoch == 0


@pytest.mark.parametrize("field", ["model_config", "tokenizer", "task", "layer_mix", "loss"])
def test_full_init_rejects_same_shape_semantic_mismatch(tmp_path, field):
    source = _module()
    payload = _checkpoint(source)
    if field == "model_config":
        payload[field]["backbone"]["config_overrides"]["hidden_dropout_prob"] = 0.25
    elif field == "tokenizer":
        payload["model_config"]["channels"][0]["tokenizer"]["kwargs"]["residual_mixing"] = "ffn_scaled"
    else:
        payload["finetune_config"][field] = {"different": True}
    path = tmp_path / "mismatch.ckpt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model_config" if field == "tokenizer" else field):
        _module(checkpoint=str(path))


@pytest.mark.parametrize("saved_init_from", ["absent", "eeg_original"])
def test_full_init_ignores_channel_init_from_in_model_config(tmp_path, saved_init_from):
    # init_from only directs --pretrained-backbone-path loading, so a checkpoint saved before the field
    # existed, or one recording a different init_from, still matches the current model semantics.
    source = _module()
    payload = _checkpoint(source)
    for channel in payload["model_config"]["channels"]:
        if saved_init_from == "absent":
            del channel["init_from"]
        else:
            channel["init_from"] = saved_init_from
    path = tmp_path / "selected.ckpt"
    torch.save(payload, path)

    target = _module(checkpoint=str(path))

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("subtree", ["head", "layer_mix"])
def test_full_init_does_not_tolerate_missing_learned_state(tmp_path, subtree):
    payload = _checkpoint(_module())
    key = next(name for name in payload["state_dict"] if subtree in name)
    del payload["state_dict"][key]
    path = tmp_path / "partial.ckpt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        _module(checkpoint=str(path))


@pytest.mark.parametrize("field,value", [("alpha", 32), ("dropout", 0.1)])
def test_full_init_rejects_same_shape_lora_semantic_mismatch(tmp_path, field, value):
    payload = _checkpoint(_module(preset="lora"))
    payload["finetune_config"]["tuning"]["lora"][field] = value
    path = tmp_path / "lora-mismatch.ckpt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="LoRA configuration"):
        _module(checkpoint=str(path), preset="lora")


def test_full_init_preserves_matching_lora_weights(tmp_path):
    source = _module(preset="lora")
    path = tmp_path / "lora.ckpt"
    torch.save(_checkpoint(source), path)

    target = _module(checkpoint=str(path), preset="lora", lr=0.004)

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
