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
        self.tokenizer_mapping = nn.ModuleDict(
            {"heartbeat": nn.Sequential(nn.Linear(1, 1, bias=False), nn.Dropout(p=0.5))}
        )

    def get_encoder(self):
        return self.encoder

    def replace_encoder(self, encoder: nn.Module):
        self.encoder = encoder

    def set_tokenizers_trainable(self, trainable: bool):
        for parameter in self.tokenizer_mapping.parameters():
            parameter.requires_grad = trainable


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

    model.eval()
    model.train()

    assert model.backbone.training is True
    assert encoder.training is True


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_frozen_backbone_without_lora_stays_in_eval_mode(module_name: str):
    downstream_module = importlib.import_module(module_name)
    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat"])

    model.train()
    model.freeze_backbone_and_insert_lora(insert_lora=False)

    assert model.training is True
    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())

    model.eval()
    model.train()

    assert model.training is True
    assert model.backbone.training is False


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_trainable_tokenizer_remains_in_train_mode_with_frozen_backbone(module_name: str):
    downstream_module = importlib.import_module(module_name)
    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat"])

    model.freeze_backbone_and_insert_lora(insert_lora=False)
    model.backbone.set_tokenizers_trainable(True)
    model.train()

    assert model.backbone.training is True
    assert model.backbone.encoder.training is False
    assert model.backbone.tokenizer_mapping.training is True
    assert model.backbone.tokenizer_mapping["heartbeat"].training is True
    assert model.backbone.tokenizer_mapping["heartbeat"][1].training is True
    assert all(parameter.requires_grad for parameter in model.backbone.tokenizer_mapping.parameters())
    assert all(not parameter.requires_grad for parameter in model.backbone.encoder.parameters())

    model.eval()
    model.train()

    assert model.training is True
    assert model.backbone.training is True
    assert model.backbone.encoder.training is False
    assert model.backbone.tokenizer_mapping["heartbeat"][1].training is True


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_backbone_frozen_without_lora_helper_stays_in_eval_mode(module_name: str):
    """Head-only recipes freeze the backbone without calling freeze_backbone_and_insert_lora."""
    downstream_module = importlib.import_module(module_name)
    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat"])

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.sync_backbone_mode_policy()

    assert model.training is True
    assert model.backbone.training is False

    model.eval()
    model.train()

    assert model.training is True
    assert model.backbone.training is False
    assert model.backbone.tokenizer_mapping["heartbeat"][1].training is False


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_partially_frozen_backbone_without_lora_helper_keeps_trainable_groups(module_name: str):
    downstream_module = importlib.import_module(module_name)
    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat"])

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.encoder.parameters():
        parameter.requires_grad = True
    model.train()

    assert model.backbone.training is True
    assert model.backbone.encoder.training is True
    assert model.backbone.tokenizer_mapping.training is False


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.downstream_model",
        "sleep2vec2.downstream_model",
        "sleep2expert.downstream_model",
    ],
)
def test_later_trainable_backbone_parameters_preserve_train_mode(module_name: str):
    downstream_module = importlib.import_module(module_name)
    model = _downstream_with_backbone(downstream_module.Sleep2vecDownstreamModel, ["heartbeat"])

    model.freeze_backbone_and_insert_lora(insert_lora=False)
    for parameter in model.backbone.encoder.parameters():
        parameter.requires_grad = True
    model.train()

    assert model.backbone.training is True
    assert model.backbone.encoder.training is True
    assert model.backbone.tokenizer_mapping.training is False


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("layer_mix_enabled", [False, True])
def test_real_peft_lora_forward_backward_smoke(variant: str, layer_mix_enabled: bool):
    from peft import PeftModelForFeatureExtraction

    config = importlib.import_module(f"{variant}.config")
    Sleep2vecDownstreamModel = importlib.import_module(f"{variant}.downstream_model").Sleep2vecDownstreamModel
    importlib.import_module(f"{variant}.downstreams.heads")
    Sleep2vecPretrainModel = importlib.import_module(f"{variant}.pretrain_model").Sleep2vecPretrainModel
    torch.manual_seed(0)

    model_config = config.ModelConfig(
        channels=[
            config.ChannelConfig(
                name="heartbeat", input_dim=8, tokenizer=config.TokenizerConfig(name="linear", out_dim=16)
            ),
            config.ChannelConfig(
                name="breath", input_dim=8, tokenizer=config.TokenizerConfig(name="linear", out_dim=16)
            ),
        ],
        backbone=config.BackboneConfig(
            name="roformer",
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1,
            config_overrides={
                "intermediate_size": 32,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "max_position_embeddings": 16,
            },
        ),
        projection=config.ProjectionConfig(name="simclr", enabled=False, hidden_dim=16, out_dim=8),
        cls=config.ClsConfig(downstream="tokens", embedding_type=None),
        head=config.HeadConfig(
            name="classification",
            channel_agg=config.ChannelAggConfig(name="mean"),
            temporal_agg=config.TemporalAggConfig(name="mean"),
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
        layer_mix_cfg=(
            config.LayerMixConfig(enabled=True, layer_indices=[1, 2], shared_across_modalities=False)
            if layer_mix_enabled
            else None
        ),
    ).train()
    downstream.freeze_backbone_and_insert_lora(
        insert_lora=True,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
    )
    assert isinstance(backbone.get_encoder(), PeftModelForFeatureExtraction)

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
    assert torch.isfinite(output).all()
    assert lora_params
    assert all(param.requires_grad for _, param in lora_params)
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for _, param in lora_params)
    assert all(torch.isfinite(param.grad).all() for _, param in lora_params if param.grad is not None)
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in downstream.head.parameters())
    assert any(name.startswith("head.") for name in trainable_names)
    assert all("lora_" in name or not name.startswith("backbone.") for name in trainable_names)
    if layer_mix_enabled:
        assert downstream.layer_mix.weight.grad is not None
        assert torch.isfinite(downstream.layer_mix.weight.grad).all()
        assert downstream.layer_mix.weight.grad.abs().sum() > 0
    else:
        assert downstream.layer_mix is None
