import typing as t

import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel

try:
    from switch_roformers import SwitchRoFormerConfig, SwitchRoFormerEncoderModel
except ImportError as exc:  # pragma: no cover - dependency is external
    SwitchRoFormerConfig = None
    SwitchRoFormerEncoderModel = None
    _SWITCH_ROFORMER_IMPORT_ERROR = exc
else:
    _SWITCH_ROFORMER_IMPORT_ERROR = None

from .config import BackboneConfig
from .registry import register_backbone


class TransformerEncoderFactory:
    """Utility that builds transformer-style encoders on demand."""

    def __init__(
        self,
        *,
        name: str,
        build_fn: t.Callable[[], nn.Module],
        hidden_size: int | None = None,
    ):
        self.name = name
        self._build_fn = build_fn
        self.hidden_size = hidden_size

    def build(self) -> tuple[nn.Module, int]:
        encoder = self._build_fn()
        hidden_size = self.hidden_size
        if hidden_size is None:
            config = getattr(encoder, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"EncoderFactory '{self.name}' cannot infer hidden size. "
                "Pass hidden_size explicitly when creating the factory."
            )
        return encoder, hidden_size

    @classmethod
    def from_hf_config(
        cls,
        *,
        name: str,
        model_cls: t.Type[nn.Module],
        config,
    ) -> "TransformerEncoderFactory":
        def build() -> nn.Module:
            return model_cls(config)

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        return cls(name=name, build_fn=build, hidden_size=hidden_size)

    @classmethod
    def roformer(
        cls,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        vocab_size: int = 1,
        **config_overrides: t.Any,
    ) -> "TransformerEncoderFactory":
        config = RoFormerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            **config_overrides,
        )
        return cls.from_hf_config(name="roformer", model_cls=RoFormerModel, config=config)

    @classmethod
    def switch_roformer(
        cls,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        vocab_size: int = 1,
        **config_overrides: t.Any,
    ) -> "TransformerEncoderFactory":
        if SwitchRoFormerConfig is None or SwitchRoFormerEncoderModel is None:
            raise ImportError(
                "switch_roformers is unavailable. Install the local transformers branch `switch-roformer-rope`."
            ) from _SWITCH_ROFORMER_IMPORT_ERROR

        config = SwitchRoFormerConfig(
            vocab_size=vocab_size,
            d_model=hidden_size,
            num_layers=num_hidden_layers,
            num_heads=num_attention_heads,
            is_encoder_decoder=False,
            **config_overrides,
        )
        return cls.from_hf_config(name="switch_roformer", model_cls=SwitchRoFormerEncoderModel, config=config)


@register_backbone("roformer")
def build_roformer(cfg: BackboneConfig) -> TransformerEncoderFactory:
    overrides = dict(cfg.config_overrides or {})
    return TransformerEncoderFactory.roformer(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        **overrides,
    )


@register_backbone("switch_roformer")
def build_switch_roformer(cfg: BackboneConfig) -> TransformerEncoderFactory:
    overrides = dict(cfg.config_overrides or {})
    return TransformerEncoderFactory.switch_roformer(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        **overrides,
    )


@register_backbone("hf_bert")
def build_hf_bert(cfg: BackboneConfig) -> TransformerEncoderFactory:
    # Local import to avoid forcing Bert dependency if unused.
    from transformers import BertConfig, BertModel

    bert_config = BertConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.hidden_size * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        vocab_size=cfg.vocab_size,
        **(cfg.config_overrides or {}),
    )
    return TransformerEncoderFactory.from_hf_config(
        name="bert",
        model_cls=BertModel,
        config=bert_config,
    )
