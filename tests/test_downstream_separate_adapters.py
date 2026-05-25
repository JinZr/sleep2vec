from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import torch.nn as nn

from sleep2vec import downstream_model as downstream_module
from sleep2vec.downstream_model import Sleep2vecDownstreamModel


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


def _downstream_with_backbone(channel_names):
    model = Sleep2vecDownstreamModel.__new__(Sleep2vecDownstreamModel)
    nn.Module.__init__(model)
    model.backbone = _Backbone()
    model.channel_names = list(channel_names)
    model._adapter_warning_logged = False
    return model


def test_separate_adapters_only_train_channel_lora_weights(monkeypatch):
    def fake_get_peft_model(encoder, cfg):
        return _FakePeftEncoder(encoder, cfg)

    monkeypatch.setattr(downstream_module, "get_peft_model", fake_get_peft_model)

    model = _downstream_with_backbone(["heartbeat", "breath"])
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
