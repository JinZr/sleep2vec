from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import torch.nn as nn


class _BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)


class _FakePeftEncoder(nn.Module):
    def __init__(self, base_encoder: nn.Module, cfg):
        super().__init__()
        self.base_encoder = base_encoder
        self.peft_config = {"default": cfg}
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.active_adapter = None

    def add_adapter(self, name: str, cfg):
        self.peft_config[name] = cfg
        self.lora_A[name] = nn.Linear(1, 1, bias=False)
        self.lora_B[name] = nn.Linear(1, 1, bias=False)

    def set_adapter(self, name: str):
        self.active_adapter = name


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _BaseEncoder()

    def get_encoder(self):
        return self.encoder

    def replace_encoder(self, encoder: nn.Module):
        self.encoder = encoder


def _downstream_with_backbone(model_cls, channel_names):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.backbone = _Backbone()
    model.channel_names = list(channel_names)
    model._adapter_warning_logged = False
    return model


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_separate_adapters_only_train_channel_lora_weights(monkeypatch, module_name: str):
    downstream_module = importlib.import_module(module_name)

    def fake_get_peft_model(encoder, cfg):
        return _FakePeftEncoder(encoder, cfg)

    monkeypatch.setattr(downstream_module, "get_peft_model", fake_get_peft_model)

    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat", "breath"])
    model.freeze_backbone_and_insert_lora(
        insert_lora=True,
        r=4,
        lora_alpha=12,
        lora_dropout=0.15,
        target_modules=("query", "dense"),
        use_dora=True,
        separate_adapters=True,
    )

    encoder = model._backbone_encoder()
    assert model.separate_adapters is True
    assert set(encoder.peft_config) == {"default", "ch_heartbeat", "ch_breath"}
    cfg = encoder.peft_config["default"]
    assert cfg.r == 4
    assert cfg.lora_alpha == 12
    assert cfg.lora_dropout == 0.15
    assert set(cfg.target_modules) == {"query", "dense"}
    assert cfg.use_dora is True

    lora_params = dict(encoder.named_parameters())
    assert lora_params["lora_A.default.weight"].requires_grad is False
    assert lora_params["lora_B.default.weight"].requires_grad is False
    assert lora_params["lora_A.ch_heartbeat.weight"].requires_grad is True
    assert lora_params["lora_B.ch_heartbeat.weight"].requires_grad is True
    assert lora_params["lora_A.ch_breath.weight"].requires_grad is True
    assert lora_params["lora_B.ch_breath.weight"].requires_grad is True
    assert lora_params["base_encoder.proj.weight"].requires_grad is False

    model._set_active_adapter("ch_breath")
    assert encoder.active_adapter == "ch_breath"


def test_sleep2vec2_real_peft_lora_forward_backward_smoke():
    from sleep2vec2.config import (
        BackboneConfig,
        ChannelAggConfig,
        ChannelConfig,
        ClsConfig,
        HeadConfig,
        ModelConfig,
        ProjectionConfig,
        TemporalAggConfig,
        TokenizerConfig,
    )
    from sleep2vec2.downstream_model import Sleep2vecDownstreamModel
    import sleep2vec2.downstreams.heads  # noqa: F401
    from sleep2vec2.pretrain_model import Sleep2vecPretrainModel

    model_config = ModelConfig(
        channels=[
            ChannelConfig(name="heartbeat", input_dim=8, tokenizer=TokenizerConfig(name="linear", out_dim=16)),
            ChannelConfig(name="breath", input_dim=8, tokenizer=TokenizerConfig(name="linear", out_dim=16)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=1,
            config_overrides={
                "intermediate_size": 32,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "max_position_embeddings": 16,
            },
        ),
        projection=ProjectionConfig(name="simclr", enabled=False, hidden_dim=16, out_dim=8),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            name="classification",
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            hidden_dim=8,
            dropout=0.0,
        ),
    )
    backbone = Sleep2vecPretrainModel(model_config=model_config, device="cpu")
    downstream = Sleep2vecDownstreamModel(
        target="stage",
        backbone=backbone,
        channel_names=["heartbeat", "breath"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
    ).train()
    downstream.freeze_backbone_and_insert_lora(
        insert_lora=True,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
    )

    batch = {
        "tokens": {
            "heartbeat": torch.randn(2, 4, 8),
            "breath": torch.randn(2, 4, 8),
        },
        "mlm_mask": {
            "heartbeat": torch.zeros(2, 4, dtype=torch.long),
            "breath": torch.zeros(2, 4, dtype=torch.long),
        },
        "length": torch.tensor([4, 3], dtype=torch.long),
    }

    output = downstream(batch)
    loss = output.square().mean()
    loss.backward()

    lora_params = [(name, param) for name, param in downstream.named_parameters() if "lora_" in name]
    trainable_names = [name for name, param in downstream.named_parameters() if param.requires_grad]
    assert output.shape == (2, 2)
    assert lora_params
    assert all(param.requires_grad for _, param in lora_params)
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for _, param in lora_params)
    assert any(name.startswith("head.") for name in trainable_names)
    assert all("lora_" in name or not name.startswith("backbone.") for name in trainable_names)
