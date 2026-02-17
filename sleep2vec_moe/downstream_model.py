import logging
import typing as t
from dataclasses import asdict

from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
import yaml

from sleep2vec.config import HeadConfig, LayerMixConfig, ModelConfig, MoEFineTuneConfig
from sleep2vec.modules.layer_mix import LayerMix

from .downstreams.head_registry import create_head
from .downstreams.temporal_aggregation import build_temporal_aggregator
from .pretrain_model import Sleep2vecPretrainModel


def _resolve_act(name: str | None):
    if not name:
        return None
    mapping = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported activation '{name}'.")
    return mapping[key]


class Sleep2vecDownstreamModel(nn.Module):
    def __init__(
        self,
        target: str,
        backbone: Sleep2vecPretrainModel,
        channel_names: t.List[str],
        output_dim: int,
        is_classification: bool,
        is_seq: bool,
        device: str = "cuda",
        head_name: str | None = None,
        head_kwargs: t.Optional[dict] = None,
        model_config: ModelConfig | None = None,
        layer_mix_cfg: LayerMixConfig | None = None,
        head_config: HeadConfig | None = None,
        moe_finetune_cfg: MoEFineTuneConfig | None = None,
    ):
        super().__init__()
        # core attributes
        self.model_config = model_config
        self.backbone = backbone
        self.channel_names = [c.name for c in model_config.channels] if model_config else channel_names
        self.device = device
        self.output_dim = output_dim
        self.is_classification = is_classification
        self.is_seq = is_seq
        self.target = target
        self.moe_finetune_cfg = moe_finetune_cfg
        self.last_moe_stats: dict[str, t.Any] | None = None

        self.n_channels = len(self.channel_names)
        self.cls_embedding = getattr(self.backbone, "cls_embedding", None)
        cls_cfg = model_config.cls if model_config else None
        self.cls_usage = cls_cfg.downstream if cls_cfg else None

        if self.cls_usage == "cls" and self.cls_embedding is None:
            raise ValueError(
                "Backbone provides no CLS embedding; set model.cls.embedding_type to 'bert' when "
                "model.cls.downstream is 'cls'."
            )

        head_kwargs = head_kwargs or {}
        if head_config is None:
            raise ValueError("head_config must be provided for downstream model construction.")

        inferred_head = head_config.name
        if head_config.act:
            head_kwargs.setdefault("act", _resolve_act(head_config.act))
        channel_cfg = head_config.channel_agg
        head_kwargs.setdefault("agg", channel_cfg.name)
        head_kwargs.setdefault("hidden_dim", head_config.hidden_dim)
        head_kwargs.setdefault("dropout", head_config.dropout)
        head_kwargs.update(head_config.kwargs or {})
        temporal_cfg = head_config.temporal_agg

        self.head = create_head(
            inferred_head,
            target=target,
            feature_dim=self.backbone.transformer_hidden_size,
            n_mods=self.n_channels,
            output_dim=self.output_dim,
            is_classification=self.is_classification,
            is_seq=self.is_seq,
            **head_kwargs,
        )

        self.separate_adapters = False  # default
        self._adapter_warning_logged = False
        self._seq_cls_warning_logged = False
        self._head_accepts_token_mask: bool | None = None

        # configure temporal aggregation (required)
        self.temporal_agg = build_temporal_aggregator(
            temporal_cfg.name,
            hidden_size=self.backbone.transformer_hidden_size,
            **dict(temporal_cfg.kwargs or {}),
        )

        self.layer_mix_cfg = layer_mix_cfg
        self.layer_indices: t.List[int] | None = None
        self.layer_mix: LayerMix | None = None
        if self.layer_mix_cfg and getattr(self.layer_mix_cfg, "enabled", False):
            if self.layer_mix_cfg.layer_indices:
                self.layer_indices = list(self.layer_mix_cfg.layer_indices)
                num_layers = len(self.layer_indices)
            else:
                num_layers = self.model_config.backbone.num_hidden_layers
            self.layer_mix = LayerMix(
                num_layers=num_layers,
                n_mods=self.n_channels,
                shared_across_modalities=self.layer_mix_cfg.shared_across_modalities,
            )

        self._router_param_prefixes: set[str] = set()
        self._expert_param_prefixes: set[str] = set()
        self._refresh_moe_param_matchers()

    def _backbone_encoder(self) -> nn.Module:
        """Returns the encoder module inside the backbone."""
        if hasattr(self.backbone, "get_encoder"):
            return self.backbone.get_encoder()

        encoder = getattr(self.backbone, "encoder", None)
        if encoder is None:
            raise AttributeError(
                "Provided backbone does not expose an encoder. If you pass a custom "
                "backbone ensure it implements get_encoder()."
            )
        return encoder

    def _replace_backbone_encoder(self, encoder: nn.Module):
        """Swap the current encoder inside the backbone."""
        if hasattr(self.backbone, "replace_encoder"):
            self.backbone.replace_encoder(encoder)
        else:
            self.backbone.encoder = encoder

    def _set_active_adapter(self, adapter_name: str):
        """Switch adapters if the encoder exposes adapter APIs."""
        encoder = self._backbone_encoder()
        if hasattr(encoder, "set_adapter"):
            encoder.set_adapter(adapter_name)
        elif not self._adapter_warning_logged:
            logging.warning("Encoder lacks 'set_adapter'; separate adapters are ignored.")
            self._adapter_warning_logged = True

    def _layer_mix_enabled(self) -> bool:
        return self.layer_mix is not None

    def _refresh_moe_param_matchers(self) -> None:
        self._router_param_prefixes = set()
        self._expert_param_prefixes = set()
        for module_name, module in self.backbone.named_modules():
            cls_name = module.__class__.__name__.lower()
            if "router" in cls_name or "gate" in cls_name:
                self._router_param_prefixes.add(module_name)
            if "expert" in cls_name:
                self._expert_param_prefixes.add(module_name)

    @staticmethod
    def _prefix_match(param_name: str, prefixes: set[str]) -> bool:
        for prefix in prefixes:
            if not prefix:
                continue
            if param_name == prefix or param_name.startswith(prefix + "."):
                return True
        return False

    def is_router_parameter_name(self, name: str) -> bool:
        if self._prefix_match(name, self._router_param_prefixes):
            return True
        lowered = name.lower()
        return ".router." in lowered or ".gate." in lowered or lowered.startswith("router.")

    def is_expert_parameter_name(self, name: str) -> bool:
        if self._prefix_match(name, self._expert_param_prefixes):
            return True
        lowered = name.lower()
        return ".experts." in lowered or ".expert." in lowered or lowered.startswith("experts.")

    def layer_mix_snapshot(self) -> dict[str, t.Any] | None:
        """Returns raw and normalized layer-mix weights in a serialization-friendly format."""
        if not self._layer_mix_enabled():
            return None

        if self.layer_mix is None:
            return None

        layer_ids = list(self.layer_indices) if self.layer_indices else list(range(1, self.layer_mix.num_layers + 1))
        raw = self.layer_mix.weight.detach().cpu()
        normalized = self.layer_mix.normalized_weight_matrix().detach().cpu()
        shared = bool(self.layer_mix.shared_across_modalities)

        if shared:
            row_names = ["shared"]
        else:
            row_names = list(self.channel_names)

        if len(row_names) != int(raw.size(0)):
            row_names = [f"row_{idx}" for idx in range(int(raw.size(0)))]

        rows: dict[str, dict[str, t.Any]] = {}
        for row_idx, row_name in enumerate(row_names):
            rows[row_name] = {
                "row_index": int(row_idx),
                "raw_logits": [float(v) for v in raw[row_idx].tolist()],
                "layer_weights": [float(v) for v in normalized[row_idx].tolist()],
            }

        effective_by_modality: dict[str, dict[str, t.Any]] = {}
        for mod_idx, mod_name in enumerate(self.channel_names):
            row_idx = 0 if shared else mod_idx
            row_name = row_names[row_idx]
            effective_by_modality[mod_name] = {
                "row_name": row_name,
                "row_index": int(row_idx),
                "layer_weights": [float(v) for v in normalized[row_idx].tolist()],
            }

        return {
            "shared_across_modalities": shared,
            "layer_indices": layer_ids,
            "rows": rows,
            "effective_by_modality": effective_by_modality,
        }

    def _select_layer_states(self, hidden_states: t.Any) -> t.List[torch.Tensor]:
        if hidden_states is None:
            raise ValueError("Layer mix requested but encoder returned no hidden states.")
        if not isinstance(hidden_states, (list, tuple)):
            raise ValueError(f"Hidden states must be a list/tuple, got {type(hidden_states)}.")

        expected_layers = self.model_config.backbone.num_hidden_layers if self.model_config else len(hidden_states)
        if len(hidden_states) == expected_layers + 1:
            layer_states = list(hidden_states[1:])  # drop embedding output
        elif len(hidden_states) == expected_layers:
            layer_states = list(hidden_states)
        else:
            raise ValueError(
                f"Unexpected hidden_states length {len(hidden_states)} "
                f"(expected {expected_layers} or {expected_layers + 1})."
            )

        if self.layer_indices:
            max_idx = max(self.layer_indices)
            if max_idx > len(layer_states):
                raise ValueError(f"layer_indices {self.layer_indices} exceed available layers ({len(layer_states)}).")
            layer_states = [layer_states[idx - 1] for idx in self.layer_indices]
        return layer_states

    def _split_layer_states(
        self,
        layer_states: t.List[torch.Tensor],
        attn_mask: torch.Tensor | None,
    ) -> tuple[t.List[torch.Tensor], t.List[torch.Tensor] | None, torch.Tensor | None]:
        if self.cls_embedding is None:
            return list(layer_states), None, attn_mask

        token_layers: t.List[torch.Tensor] = []
        cls_layers: t.List[torch.Tensor] = []
        token_mask = None
        for layer_hidden in layer_states:
            token_hidden, cls_hidden, layer_mask = self.cls_embedding.split_hidden(layer_hidden, attn_mask)
            token_layers.append(token_hidden)
            cls_layers.append(cls_hidden)
            if token_mask is None:
                token_mask = layer_mask
        return token_layers, cls_layers, token_mask

    def forward(self, batch):
        tokens = batch["tokens"]

        token_embeddings = self.backbone._tokenize_all(tokens)
        token_names, token_embeddings = list(token_embeddings.keys()), list(token_embeddings.values())

        metadata = batch.get("metadata")
        moe_cfg = self.moe_finetune_cfg
        collect_moe_stats = False
        if moe_cfg is not None:
            collect_moe_stats = bool(
                moe_cfg.enable_aux_losses
                or moe_cfg.train_router
                or moe_cfg.train_experts
                or (moe_cfg.group_balance is not None and moe_cfg.group_balance.enabled)
            )

        feature_of_different_mods = []
        token_masks: list[torch.Tensor | None] = []
        modality_moe_losses: list[torch.Tensor] = []
        modality_moe_stats: dict[str, dict[str, t.Any]] = {}
        layer_mix_enabled = self._layer_mix_enabled()
        for mod_idx, (token_name, single_mod_token_embeddings) in enumerate(zip(token_names, token_embeddings)):

            if getattr(self, "separate_adapters", False):
                self._set_active_adapter(f"ch_{token_name}")

            batch_size = int(single_mod_token_embeddings.shape[0])
            mod_device = single_mod_token_embeddings.device
            router_ctx = self.backbone._build_router_ctx(
                metadata,
                token_name,
                batch_size=batch_size,
                device=mod_device,
            )
            router_group_ids = self.backbone._build_router_group_ids(
                metadata,
                batch_size=batch_size,
                device=mod_device,
                channel_name=token_name,
            )

            if layer_mix_enabled:
                _, attn_mask, hidden_states, aux = self.backbone._token_embeddings_to_hidden(
                    single_mod_token_embeddings,
                    batch,
                    return_hidden_states=True,
                    router_ctx=router_ctx,
                    router_group_ids=router_group_ids,
                    return_aux=True,
                    collect_moe_stats=collect_moe_stats,
                )
                layer_states = self._select_layer_states(hidden_states)
                token_layers, cls_layers, token_mask = self._split_layer_states(layer_states, attn_mask)

                if self.is_seq:
                    token_stack = torch.stack(token_layers, dim=0)
                    mixed_tokens = self.layer_mix.mix(token_stack, mod_idx=mod_idx)
                    feature = self._forward_seq(mixed_tokens, None)
                else:
                    if self.cls_usage == "cls":
                        if not cls_layers or cls_layers[0] is None:
                            raise RuntimeError("cls_usage='cls' requested but backbone provides no CLS embedding.")
                        cls_stack = torch.stack(cls_layers, dim=0)
                        feature = self.layer_mix.mix(cls_stack, mod_idx=mod_idx)
                    else:
                        if token_mask is None:
                            token_mask = self._build_token_mask(
                                batch["length"], token_layers[0].size(1), token_layers[0].device
                            )
                        elif token_mask.dim() == 3 and token_mask.size(1) == 1:
                            token_mask = token_mask.squeeze(1)
                        if token_mask is not None:
                            token_mask = token_mask.to(torch.bool)
                        pooled_layers = [self.temporal_agg(layer_tokens, token_mask) for layer_tokens in token_layers]
                        pooled_stack = torch.stack(pooled_layers, dim=0)
                        feature = self.layer_mix.mix(pooled_stack, mod_idx=mod_idx)

                feature_of_different_mods.append(feature)
                if self.is_seq:
                    if token_mask is None:
                        token_mask = self._build_token_mask(
                            batch["length"], token_layers[0].size(1), token_layers[0].device
                        )
                    elif token_mask.dim() == 3 and token_mask.size(1) == 1:
                        token_mask = token_mask.squeeze(1)
                    if token_mask is not None:
                        token_mask = token_mask.to(torch.bool)
                    token_masks.append(token_mask)
                if isinstance(aux, dict):
                    moe_loss = aux.get("moe_loss")
                    if torch.is_tensor(moe_loss):
                        modality_moe_losses.append(moe_loss)
                    moe_metrics = aux.get("moe_metrics")
                    if isinstance(moe_metrics, dict):
                        modality_moe_stats[token_name] = moe_metrics
                continue

            hidden, attn_mask, _, aux = self.backbone._token_embeddings_to_hidden(
                single_mod_token_embeddings,
                batch,
                router_ctx=router_ctx,
                router_group_ids=router_group_ids,
                return_aux=True,
                collect_moe_stats=collect_moe_stats,
            )

            strategy = self.cls_embedding
            if strategy is None:
                token_hidden, cls_hidden, token_mask = hidden, None, attn_mask
            else:
                token_hidden, cls_hidden, token_mask = strategy.split_hidden(hidden, attn_mask)

            if self.is_seq:
                feature = self._forward_seq(token_hidden, cls_hidden)
            else:
                feature = self._forward_nonseq(token_hidden, cls_hidden, token_mask, batch)
            feature_of_different_mods.append(feature)
            if self.is_seq:
                if token_mask is None:
                    token_mask = self._build_token_mask(batch["length"], token_hidden.size(1), token_hidden.device)
                elif token_mask.dim() == 3 and token_mask.size(1) == 1:
                    token_mask = token_mask.squeeze(1)
                if token_mask is not None:
                    token_mask = token_mask.to(torch.bool)
                token_masks.append(token_mask)
            if isinstance(aux, dict):
                moe_loss = aux.get("moe_loss")
                if torch.is_tensor(moe_loss):
                    modality_moe_losses.append(moe_loss)
                moe_metrics = aux.get("moe_metrics")
                if isinstance(moe_metrics, dict):
                    modality_moe_stats[token_name] = moe_metrics

        if self.is_seq and token_masks:
            merged_mask = token_masks[0]
            for mask in token_masks[1:]:
                if mask is None:
                    continue
                if mask.shape != merged_mask.shape:
                    raise ValueError("Token masks must share the same shape for sequence heads.")
                merged_mask = merged_mask & mask
            output = self._call_head(feature_of_different_mods, merged_mask)
        else:
            output = self._call_head(feature_of_different_mods, None)

        if modality_moe_stats:
            merged: dict[str, t.Any] = {}
            routed_lists: dict[str, list[t.Any]] = {
                "router_logits": [],
                "router_probs": [],
                "expert_indices": [],
                "dispatch_mask": [],
                "dropped_mask": [],
                "capacity": [],
            }
            for modality_name, stats in modality_moe_stats.items():
                for key, value in stats.items():
                    merged[f"modality/{modality_name}/{key}"] = value
                for key in routed_lists:
                    value = stats.get(key)
                    if isinstance(value, (list, tuple)):
                        routed_lists[key].extend(list(value))

            if modality_moe_losses:
                merged["moe_loss_backbone"] = torch.stack(
                    [loss.to(dtype=torch.float32) for loss in modality_moe_losses], dim=0
                ).mean()

            candidate_avg_keys = sorted(
                {
                    key
                    for stats in modality_moe_stats.values()
                    for key in stats.keys()
                    if key.startswith("mean/") or key.startswith("last/")
                }
            )
            for key in candidate_avg_keys:
                values = [stats[key] for stats in modality_moe_stats.values() if key in stats]
                if len(values) != len(modality_moe_stats):
                    continue
                if not all(torch.is_tensor(value) for value in values):
                    continue
                if not all(values[0].shape == value.shape for value in values[1:]):
                    continue
                merged[key] = torch.stack([value.to(dtype=torch.float32) for value in values], dim=0).mean(dim=0)

            self.last_moe_stats = {
                "merged": merged,
                "by_modality": modality_moe_stats,
                **{key: value for key, value in routed_lists.items() if value},
            }
        else:
            self.last_moe_stats = None
        return output

    def _forward_seq(self, token_hidden, cls_hidden):
        if self.cls_usage == "cls":
            # Sequence labeling expects per-token logits; using a single CLS embedding would
            # collapse the time dimension and later break loss shapes (targets are [B,L]).
            if not self._seq_cls_warning_logged:
                logging.warning(
                    "model.cls.downstream='cls' is incompatible with is_seq=True; "
                    "ignoring CLS and using token embeddings for sequence prediction."
                )
                self._seq_cls_warning_logged = True
        return token_hidden

    def _forward_nonseq(self, token_hidden, cls_hidden, token_mask, batch):
        if self.cls_usage == "cls":
            if cls_hidden is None:
                raise RuntimeError("cls_usage='cls' requested but backbone provides no CLS embedding.")
            return cls_hidden

        if token_mask is None:
            B, L, _ = token_hidden.shape
            token_mask = torch.zeros(B, L, dtype=torch.bool, device=token_hidden.device)
            for i in range(B):
                token_mask[i, : batch["length"][i].item()] = True

        return self.temporal_agg(token_hidden, token_mask)

    @staticmethod
    def _build_token_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
        mask = torch.zeros(int(lengths.shape[0]), max_len, dtype=torch.bool, device=device)
        for i in range(mask.size(0)):
            valid_len = int(lengths[i].item())
            mask[i, :valid_len] = True
        return mask

    def _head_supports_token_mask(self) -> bool:
        if self._head_accepts_token_mask is None:
            self._head_accepts_token_mask = bool(getattr(self.head, "supports_token_mask", False))
        return self._head_accepts_token_mask

    def _call_head(self, features: t.List[torch.Tensor], token_mask: torch.Tensor | None):
        if token_mask is not None and self._head_supports_token_mask():
            return self.head(features, token_mask=token_mask)
        return self.head(features)

    def load_pretrained_backbone(self, ckpt_path, use_ema: bool | str | None = True):
        logging.info(f"Loading backbone from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        averaging_name = None
        if isinstance(use_ema, str):
            averaging_name = use_ema
        elif use_ema:
            averaging_name = "ema"

        prefix = f"{averaging_name}_model." if averaging_name else "model."
        filtered_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}

        if averaging_name and not filtered_state_dict:
            logging.warning(f"{averaging_name} weights not found in checkpoint; falling back to student weights.")
            prefix = "model."
            filtered_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}

        # Sanity check CLS settings against serialized config in checkpoint (assumes YAML is present)
        self._warn_on_cls_mismatch(ckpt)
        self._warn_on_moe_mismatch(ckpt)

        # 加载到 self.backbone
        load_info = self.backbone.load_state_dict(filtered_state_dict, strict=False)

        # 打印加载结果
        total_keys = len(filtered_state_dict)
        missing_keys = load_info.missing_keys
        unexpected_keys = load_info.unexpected_keys

        logging.info(f"✅ Loaded {total_keys - len(missing_keys)} / {total_keys} keys into backbone.")
        if missing_keys:
            logging.warning(f"Missing keys ({len(missing_keys)}):")
            for k in missing_keys:
                logging.warning(f"    {k}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys ({len(unexpected_keys)}):")
            for k in unexpected_keys:
                logging.warning(f"    {k}")
            self._warn_on_dropped_cls_weights(unexpected_keys)

    def freeze_backbone_and_insert_lora(
        self,
        insert_lora: bool = True,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules=("query", "key", "value"),  # 只打注意力；若也想打FFN，加 "dense"
        separate_adapters: bool = False,
    ):
        # 0) 先冻结 backbone 全部参数
        self.separate_adapters = False
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False

        if insert_lora:

            # 1) 注入 LoRA 到 encoder
            cfg = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                # layers_to_transform=[6, 7],      # 想只打高层就改成 [4,5,6,7] 之类
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=list(target_modules),
            )
            encoder_with_lora = get_peft_model(self._backbone_encoder(), cfg)
            self._replace_backbone_encoder(encoder_with_lora)
            self._refresh_moe_param_matchers()

            # 3) 为每个通道创建独立 adapter
            self.separate_adapters = separate_adapters
            if separate_adapters:
                self.channel_adapters = []
                for ch in self.channel_names:
                    name = f"ch_{ch}"
                    encoder = self._backbone_encoder()
                    if name not in encoder.peft_config:  # 避免重复添加
                        encoder.add_adapter(name, cfg)
                    self.channel_adapters.append(name)
                self._enable_all_adapters_trainable()

        train_router = bool(getattr(self.moe_finetune_cfg, "train_router", False))
        train_experts = bool(getattr(self.moe_finetune_cfg, "train_experts", False))
        if train_router or train_experts:
            self._refresh_moe_param_matchers()
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    continue
                if train_router and self.is_router_parameter_name(name):
                    param.requires_grad = True
                elif train_experts and self.is_expert_parameter_name(name):
                    param.requires_grad = True

        # —— 全模型统计 ——
        total_all = sum(p.numel() for _, p in self.named_parameters())
        train_all = sum(p.numel() for _, p in self.named_parameters() if p.requires_grad)
        logging.info(
            "[insert_lora] model trainable params: %s/%s (%s)",
            train_all,
            total_all,
            f"{train_all/total_all:.4%}",
        )

        # —— 只看 backbone 统计 ——
        b_total = sum(p.numel() for _, p in self.backbone.named_parameters())
        b_train = sum(p.numel() for _, p in self.backbone.named_parameters() if p.requires_grad)
        b_ratio = b_train / b_total if b_total > 0 else 0.0
        lora_train = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad and "lora_" in n)
        router_train = sum(
            p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad and self.is_router_parameter_name(n)
        )
        experts_train = sum(
            p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad and self.is_expert_parameter_name(n)
        )
        head_train = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and n.startswith("head."))
        logging.info(
            "[insert_lora] backbone trainable params: %s/%s (%s); LoRA-only trainable: %s",
            b_train,
            b_total,
            f"{b_ratio:.4%}",
            lora_train,
        )
        logging.info(
            "[insert_lora] trainable breakdown (params): head=%s lora=%s router=%s experts=%s",
            head_train,
            lora_train,
            router_train,
            experts_train,
        )

    # 在所有 adapter 都 add 完之后调用
    def _enable_all_adapters_trainable(self):
        encoder = self._backbone_encoder()
        for n, p in encoder.named_parameters():
            # 只放开 LoRA 权重；底座仍然冻结
            if "lora_" in n:
                # 仅放开这些 adapter 的参数（避免误放开 default）
                if any(
                    (f".{adp}." in n or f"_{adp}." in n or n.endswith(f".{adp}.weight"))
                    for adp in self.channel_adapters
                ):
                    p.requires_grad = True

    # ---- helpers ----
    @staticmethod
    def _normalize_cls_cfg(cfg_obj):
        if cfg_obj is None:
            return None, None
        if isinstance(cfg_obj, dict):
            emb = cfg_obj.get("embedding_type")
            down = cfg_obj.get("downstream")
        else:  # dataclass instance
            emb = getattr(cfg_obj, "embedding_type", None)
            down = getattr(cfg_obj, "downstream", None)
        emb_norm = None if emb is None or str(emb).lower() in {"none", "null"} else str(emb).lower()
        return emb_norm, down

    @staticmethod
    def _load_ckpt_model_cfg(ckpt: dict) -> dict[str, t.Any]:
        model_cfg_yaml = ckpt.get("model_config_yaml")
        if model_cfg_yaml is None:
            return {}
        cfg = yaml.safe_load(model_cfg_yaml)
        if not isinstance(cfg, dict):
            return {}
        return cfg

    def _warn_on_moe_mismatch(self, ckpt: dict):
        ckpt_model_cfg = self._load_ckpt_model_cfg(ckpt)
        if not ckpt_model_cfg or self.model_config is None:
            return

        ckpt_backbone = ckpt_model_cfg.get("backbone")
        if not isinstance(ckpt_backbone, dict):
            return

        curr_backbone = asdict(self.model_config.backbone)

        def _is_moe_key(key: str) -> bool:
            lowered = key.lower()
            return any(token in lowered for token in ("moe", "expert", "router", "capacity"))

        def _collect(backbone_cfg: dict[str, t.Any]) -> dict[str, t.Any]:
            moe_cfg: dict[str, t.Any] = {}
            for key, value in backbone_cfg.items():
                if key == "config_overrides":
                    continue
                if _is_moe_key(key):
                    moe_cfg[key] = value
            overrides = backbone_cfg.get("config_overrides")
            if isinstance(overrides, dict):
                for key, value in overrides.items():
                    if _is_moe_key(key):
                        moe_cfg[key] = value
            return moe_cfg

        ckpt_moe = _collect(ckpt_backbone)
        curr_moe = _collect(curr_backbone)
        all_keys = sorted(set(ckpt_moe.keys()) | set(curr_moe.keys()))
        if not all_keys:
            return

        mismatches: list[tuple[str, t.Any, t.Any]] = []
        for key in all_keys:
            ckpt_value = ckpt_moe.get(key, "<missing>")
            curr_value = curr_moe.get(key, "<missing>")
            if ckpt_value != curr_value:
                mismatches.append((key, ckpt_value, curr_value))

        if not mismatches:
            return

        logging.warning("MoE backbone config mismatch between checkpoint and current recipe:")
        for key, ckpt_value, curr_value in mismatches:
            logging.warning("  %s: checkpoint=%r current=%r", key, ckpt_value, curr_value)

    def _warn_on_cls_mismatch(self, ckpt: dict):
        """Assumes serialized YAML config exists in checkpoint; warn if CLS settings differ."""
        ckpt_model_cfg = self._load_ckpt_model_cfg(ckpt)
        if not ckpt_model_cfg:
            return
        ckpt_cls_block = ckpt_model_cfg.get("cls")
        ckpt_emb, ckpt_down = self._normalize_cls_cfg(ckpt_cls_block)
        curr_emb, curr_down = self._normalize_cls_cfg(self.model_config.cls if self.model_config else None)

        if ckpt_emb != curr_emb:
            logging.warning(
                "CLS embedding_type mismatch: checkpoint=%s, current=%s. CLS weights may be ignored or missing.",
                ckpt_emb or "none",
                curr_emb or "none",
            )
        if ckpt_down != curr_down:
            logging.warning(
                "CLS downstream mismatch: checkpoint=%s, current=%s. Feature pooling behavior may differ.",
                ckpt_down or "tokens",
                curr_down or "tokens",
            )

    def _warn_on_dropped_cls_weights(self, unexpected_keys: list[str]):
        cls_unexpected = [k for k in unexpected_keys if k.startswith("cls_embedding")]
        if cls_unexpected and (not self.model_config or not self.model_config.cls):
            logging.warning(
                "Checkpoint contained CLS embedding weights (e.g., %s) but current finetune config disables CLS "
                "(model.cls is null/none). The CLS token has been dropped. If this is unintended, set "
                "model.cls.embedding_type='bert' (and optionally downstream='cls').",
                cls_unexpected[0],
            )

    def print_backbone_param_names(self):
        logging.info("== All parameters (name, shape, dtype, device) ==")
        for n, p in self._backbone_encoder().named_parameters():
            logging.info(f"{n:80s} {tuple(p.shape)} {p.dtype} {p.device}")
