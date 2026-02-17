from __future__ import annotations

"""Configuration for standalone RoFormer encoder."""

from dataclasses import dataclass, field


@dataclass
class RoFormerConfig:
    """Lightweight configuration for RoFormer encoder-only usage."""

    vocab_size: int = 50000
    embedding_size: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1536
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    rotary_value: bool = False
    moe_enabled: bool = False
    moe_num_experts: int = 0
    moe_top_k: int = 1
    moe_layers: str | list[int] = "none"
    moe_capacity_factor_train: float = 1.25
    moe_capacity_factor_eval: float = 2.0
    moe_router_noise_std: float = 0.0
    moe_aux_loss_coef: float = 1e-2
    moe_z_loss_coef: float = 0.0
    moe_group_balance_coef: float = 0.0
    moe_router_use_context: bool = False
    moe_router_ctx_dim: int = 0
    moe_ctx_use_age: bool = True
    moe_ctx_use_sex: bool = True
    moe_ctx_use_source: bool = True
    moe_ctx_use_modality: bool = True
    moe_source_vocab_size: int = 64
    moe_age_bins: int = 10
    moe_group_keys: list[str] = field(default_factory=lambda: ["source"])

    def __post_init__(self) -> None:
        if self.embedding_size is None:
            self.embedding_size = self.hidden_size
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")
        if self.moe_capacity_factor_train <= 0:
            raise ValueError(
                f"moe_capacity_factor_train must be positive, got {self.moe_capacity_factor_train}"
            )
        if self.moe_capacity_factor_eval <= 0:
            raise ValueError(f"moe_capacity_factor_eval must be positive, got {self.moe_capacity_factor_eval}")
        if self.moe_top_k not in {1, 2}:
            raise ValueError(f"moe_top_k must be 1 or 2, got {self.moe_top_k}")
        if self.moe_source_vocab_size <= 0:
            raise ValueError(f"moe_source_vocab_size must be positive, got {self.moe_source_vocab_size}")
        if self.moe_age_bins <= 0:
            raise ValueError(f"moe_age_bins must be positive, got {self.moe_age_bins}")
        if not isinstance(self.moe_group_keys, list):
            raise ValueError(f"moe_group_keys must be a list, got {type(self.moe_group_keys)}")
        supported_group_keys = {"source", "sex", "age_bin", "modality"}
        unsupported_keys = [key for key in self.moe_group_keys if key not in supported_group_keys]
        if unsupported_keys:
            raise ValueError(
                f"Unsupported moe_group_keys={unsupported_keys}. "
                f"Supported keys: {sorted(supported_group_keys)}"
            )

        if self.moe_enabled:
            if self.moe_num_experts <= 0:
                raise ValueError(f"moe_num_experts must be > 0 when moe_enabled=True, got {self.moe_num_experts}")
            if self.moe_router_use_context and self.moe_router_ctx_dim <= 0:
                raise ValueError(
                    "moe_router_ctx_dim must be > 0 when moe_router_use_context=True, "
                    f"got {self.moe_router_ctx_dim}"
                )
            if not self.resolve_moe_layer_indices():
                raise ValueError("moe_enabled=True but moe_layers resolved to an empty set.")

    def resolve_moe_layer_indices(self) -> set[int]:
        """Resolve the configured MoE layer selector to zero-based layer indices."""

        layers_cfg = self.moe_layers
        if isinstance(layers_cfg, str):
            name = layers_cfg.lower().strip()
            if name == "none":
                return set()
            if name == "all":
                return set(range(self.num_hidden_layers))
            if name in {"every_2", "every_4"}:
                every_n = int(name.split("_")[1])
                return {idx for idx in range(self.num_hidden_layers) if (idx + 1) % every_n == 0}
            if name == "last_4":
                start = max(0, self.num_hidden_layers - 4)
                return set(range(start, self.num_hidden_layers))
            raise ValueError(
                "Unsupported moe_layers string. Supported values: "
                "'none', 'all', 'every_2', 'every_4', 'last_4'. "
                f"Got: {self.moe_layers}"
            )

        if isinstance(layers_cfg, (list, tuple)):
            if not all(isinstance(layer_id, int) for layer_id in layers_cfg):
                raise ValueError("moe_layers list must contain only integers.")
            layer_ids = {int(layer_id) for layer_id in layers_cfg}
            invalid = [layer_id for layer_id in layer_ids if layer_id < 0 or layer_id >= self.num_hidden_layers]
            if invalid:
                raise ValueError(
                    "moe_layers contains invalid layer indices "
                    f"(expected 0 <= idx < {self.num_hidden_layers}): {sorted(invalid)}"
                )
            return layer_ids

        raise ValueError(f"moe_layers must be a string or list[int], got {type(layers_cfg)}")
