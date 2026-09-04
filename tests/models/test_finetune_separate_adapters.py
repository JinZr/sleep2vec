"""Full finetuning initialization under `finetune.tuning.lora.separate_adapters`.

`tests/models/test_downstream_separate_adapters.py` checks the downstream model's
adapter helper in isolation. That is not enough: the finetuning module runs a policy
pass over `named_parameters()` *after* the helper, and a pass that treats the whole
`lora` group as one unit silently un-freezes PEFT's `default` adapter. These tests pin
the end state of a real module build — the parameters and the optimizer groups.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("peft")

import torch.nn as nn

VARIANTS = ["sleep2vec", "sleep2vec2", "sleep2expert"]
CHANNELS = ["eeg", "ppg"]


class _FakePeftEncoder(nn.Module):
    """Just enough PEFT surface to hold several named adapters."""

    def __init__(self, base_encoder: nn.Module, cfg):
        super().__init__()
        self.base_encoder = base_encoder
        self.peft_config = {"default": cfg}
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.active_adapter = "default"

    def add_adapter(self, name: str, cfg):
        self.peft_config[name] = cfg
        self.lora_A[name] = nn.Linear(1, 1, bias=False)
        self.lora_B[name] = nn.Linear(1, 1, bias=False)

    def set_adapter(self, name: str):
        self.active_adapter = name


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        label_name="stage5",
        output_dim=5,
        is_classification=True,
        is_seq=True,
        pretrained_backbone_path=None,
        print_diagnostics=False,
        diagnostics_steps=5,
        is_multilabel=False,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=0,
    )


def _model_config(config):
    return config.ModelConfig(
        channels=[
            config.ChannelConfig(name=ch, input_dim=8, tokenizer=config.TokenizerConfig(name="linear", out_dim=16))
            for ch in CHANNELS
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


def _lora_module(monkeypatch, variant: str, *, separate_adapters: bool):
    """Build a finetuning module under the shipped `lora` preset."""
    config = importlib.import_module(f"{variant}.config")
    downstream_module = importlib.import_module(f"{variant}.downstream_model")
    importlib.import_module(f"{variant}.downstreams.heads")
    finetuning = importlib.import_module(f"{variant}.sleep2vec_finetuning")

    monkeypatch.setattr(downstream_module, "get_peft_model", lambda encoder, cfg: _FakePeftEncoder(encoder, cfg))

    # The preset table is the same one the config loader materializes, so the test
    # exercises the runtime pass rather than a hand-written group table.
    table = config._FINETUNE_TUNING_PRESETS["lora"]
    tuning = config.FinetuneTuningConfig(
        preset="lora",
        groups={
            group: config.FinetuneGroupConfig(train=train, lr_scale=lr_scale)
            for group, (train, lr_scale) in table.items()
        },
        lora=config.FinetuneTuningLoraConfig(
            r=4,
            alpha=12,
            dropout=0.15,
            target_modules=["query", "key", "value"],
            use_dora=False,
            separate_adapters=separate_adapters,
        ),
    )
    module = finetuning.Sleep2vecFinetuning(
        _args(), _model_config(config), finetune_config=config.FinetuneConfig(tuning=tuning)
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=100)
    return module


def _lora_params(module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in module.model.named_parameters() if "lora_" in name}


@pytest.mark.parametrize("variant", VARIANTS)
def test_separate_adapters_leave_the_default_adapter_frozen(monkeypatch, variant: str):
    module = _lora_module(monkeypatch, variant, separate_adapters=True)

    lora_params = _lora_params(module)
    channel_names = [name for name in lora_params if ".ch_eeg." in name or ".ch_ppg." in name]
    default_names = [name for name in lora_params if ".default." in name]
    assert channel_names
    assert default_names

    assert all(lora_params[name].requires_grad for name in channel_names)
    assert not any(lora_params[name].requires_grad for name in default_names)
    assert module.model.channel_adapters == [f"ch_{ch}" for ch in CHANNELS]


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_default_adapter_reaches_no_optimizer_group(monkeypatch, variant: str):
    module = _lora_module(monkeypatch, variant, separate_adapters=True)
    default_params = {id(param) for name, param in _lora_params(module).items() if ".default." in name}

    optimizers, _ = module.configure_optimizers()
    optimized = {id(param) for group in optimizers[0].param_groups for param in group["params"]}

    assert default_params
    assert not (default_params & optimized)
    assert module._finetune_group_summary["lora"]["trainable_params"] > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_single_adapter_still_trains_every_lora_weight(monkeypatch, variant: str):
    """The freeze is specific to the multi-adapter layout, not to LoRA generally."""
    module = _lora_module(monkeypatch, variant, separate_adapters=False)

    lora_params = _lora_params(module)
    assert lora_params
    assert all(param.requires_grad for param in lora_params.values())
    assert getattr(module.model, "separate_adapters", False) is False
