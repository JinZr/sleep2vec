import logging
import random
import typing as t

import torch
import torch.nn as nn

from sleep2vec.backbones.encoder_factory import TransformerEncoderFactory
from sleep2vec.builders import build_encoder_factory, build_projection, build_tokenizers_and_dim
from sleep2vec.cls import build_cls_embedding
from sleep2vec.config import ModelConfig, ProjectionConfig
from sleep2vec.modules.router_context import RouterContextEncoder
from sleep2vec.modules.tokenizers import LinearTokenizer, SundialTokenizer


class Sleep2vecPretrainModel(nn.Module):
    def __init__(
        self,
        channel_feature_dim: int | None = None,
        transformer_hidden_size: int | None = None,
        channel_names: t.List[str] | None = None,
        projection: bool | None = None,
        transformer_num_hidden_layers: int = 12,
        transformer_num_attention_heads: int = 16,
        encoder_factory: TransformerEncoderFactory | None = None,
        encoder_config_overrides: t.Optional[t.Dict[t.Dict, t.Any]] = None,
        encoder_forward: t.Optional[t.Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        specified_two_mods: t.List[str] | None = None,
        two_layer_embedding: bool = False,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        projection_config: ProjectionConfig | None = None,
    ):
        super().__init__()
        self.specified_two_mods = specified_two_mods
        self.device = device
        self.high_sr, self.low_sr = 3840, 120
        self._custom_encoder_forward = encoder_forward
        overrides = dict(encoder_config_overrides or {})
        self.cls_cfg = None

        if model_config is not None:
            self.channel_names = [c.name for c in model_config.channels]
            tokenizer_mapping, channel_feature_dim = build_tokenizers_and_dim(model_config, device=self.device)
            self.tokenizer_mapping = nn.ModuleDict(tokenizer_mapping)
            projection_config = projection_config or model_config.projection
            projection = projection_config.enabled
            encoder_factory = encoder_factory or build_encoder_factory(model_config.backbone)
            transformer_hidden_size = transformer_hidden_size or model_config.backbone.hidden_size
            self.cls_cfg = model_config.cls
        else:
            if channel_feature_dim is None or channel_names is None:
                raise ValueError("channel_feature_dim and channel_names are required when model_config is absent.")

            self.channel_names = channel_names
            tokenizer_type = SundialTokenizer if two_layer_embedding else LinearTokenizer

            self.high_tokenizer_1 = tokenizer_type(
                in_feature_dim=self.high_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.high_tokenizer_2 = tokenizer_type(
                in_feature_dim=self.high_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.high_tokenizer_3 = tokenizer_type(
                in_feature_dim=self.high_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.high_tokenizer_4 = tokenizer_type(
                in_feature_dim=self.high_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.low_tokenizer_1 = tokenizer_type(
                in_feature_dim=self.low_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.low_tokenizer_2 = tokenizer_type(
                in_feature_dim=self.low_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.low_tokenizer_3 = tokenizer_type(
                in_feature_dim=self.low_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.low_tokenizer_4 = tokenizer_type(
                in_feature_dim=self.low_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )
            self.low_tokenizer_5 = tokenizer_type(
                in_feature_dim=self.low_sr,
                out_feature_dim=channel_feature_dim,
                device=self.device,
            )

            self.tokenizer_mapping = {
                "eeg_original": self.high_tokenizer_1,
                "eog_original": self.high_tokenizer_2,
                "emg_original": self.high_tokenizer_3,
                "ecg_original": self.high_tokenizer_4,
                "heartbeat": self.low_tokenizer_1,
                "spo2": self.low_tokenizer_2,
                "breath": self.low_tokenizer_3,
                "resp_original": self.low_tokenizer_4,
                "resp_nasal_original": self.low_tokenizer_5,
            }
            self.tokenizer_mapping = nn.ModuleDict(self.tokenizer_mapping)

            if projection_config is None:
                projection_config = ProjectionConfig(
                    name="simclr",
                    enabled=bool(projection) if projection is not None else True,
                    hidden_dim=transformer_hidden_size,
                    out_dim=128,
                )

            if encoder_factory is None:
                vocab_size = overrides.pop("vocab_size", 1)
                encoder_factory = TransformerEncoderFactory.roformer(
                    hidden_size=transformer_hidden_size,
                    num_hidden_layers=transformer_num_hidden_layers,
                    num_attention_heads=transformer_num_attention_heads,
                    vocab_size=vocab_size,
                    **overrides,
                )

        if transformer_hidden_size is None:
            raise ValueError("transformer_hidden_size must be provided or inferred.")

        if encoder_factory is None:
            raise ValueError("encoder_factory could not be built.")

        self.encoder_factory = encoder_factory
        self.encoder, inferred_hidden_size = encoder_factory.build()
        self.transformer_hidden_size = inferred_hidden_size
        self.encoder_name = encoder_factory.name
        self.cls_embedding = build_cls_embedding(
            strategy=self.cls_cfg.embedding_type if self.cls_cfg else None,
            hidden_size=self.transformer_hidden_size,
        )
        # downstream preference (cls/tokens/None) stored for heads to read
        self.cls_downstream = self.cls_cfg.downstream if self.cls_cfg else None
        self.channels = list(self.channel_names)
        self.channel_to_id = {channel: idx for idx, channel in enumerate(self.channels)}

        self.router_ctx_encoder: RouterContextEncoder | None = None
        encoder_cfg = getattr(self.encoder, "config", None)
        if encoder_cfg is not None and getattr(encoder_cfg, "moe_router_use_context", False):
            self.router_ctx_encoder = RouterContextEncoder(
                ctx_dim=encoder_cfg.moe_router_ctx_dim,
                source_vocab_size=encoder_cfg.moe_source_vocab_size,
                num_modalities=max(1, len(self.channels)),
                use_age=bool(getattr(encoder_cfg, "moe_ctx_use_age", True)),
                use_sex=bool(getattr(encoder_cfg, "moe_ctx_use_sex", True)),
                use_source=bool(getattr(encoder_cfg, "moe_ctx_use_source", True)),
                use_modality=bool(getattr(encoder_cfg, "moe_ctx_use_modality", True)),
            )

        self.mask_embed = nn.ParameterDict(
            {channel_name: nn.Parameter(torch.ones(channel_feature_dim)) for channel_name in self.channel_names}
        )

        self.embedding_projection = nn.Linear(channel_feature_dim, self.transformer_hidden_size)

        self.proj_head = build_projection(
            (projection_config if projection_config is not None else ProjectionConfig(enabled=projection or False)),
            in_dim=self.transformer_hidden_size,
        )
        self.projection = bool(self.proj_head)

        self.total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Total parameters: {self.total_params}")
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {self.trainable_params}")

    def _tokenize_two_random_channels(self, tokens):
        """
        随机选择两个不同的模态并做tokenization
        """

        if self.specified_two_mods:
            chosen_channels = [ch for ch in self.specified_two_mods if ch in tokens]
            if len(chosen_channels) < 2:
                available = [ch for ch in self.channel_names if ch in tokens]
                if len(available) < 2:
                    raise ValueError(f"可用通道不足 2 个：{available}")
                chosen_channels = random.sample(available, 2)
        else:
            # 交集：只保留 tokens 里确实存在的通道（并保持原有顺序）
            available = [ch for ch in self.channel_names if ch in tokens]
            if len(available) < 2:
                raise ValueError(f"可用通道不足 2 个：{available}")
            chosen_channels = random.sample(available, 2)  # 随机选2个不同的模态
            # logging.info(f"chosen_channels: {chosen_channels}")
        return {
            channel_name: self.tokenizer_mapping[channel_name](tokens[channel_name]) for channel_name in chosen_channels
        }

    def _tokenize_one_channel(self, tokens):

        return {
            channel_name: self.tokenizer_mapping[channel_name](tokens[channel_name])
            for channel_name in list(tokens.keys())
        }

    def _tokenize_all(self, tokens):
        """
        对每个模态做tokenization
        """
        return {
            channel_name: self.tokenizer_mapping[channel_name](tokens[channel_name])
            for channel_name in self.channel_names
        }

    def _mask_modalities(self, tokens, mlm_mask):
        """
        对每个模态做mask替换（使用self.mask_embed），保留shape不变。
        自动扩展 mask 和 embed 的维度，使其与原始 token 对齐。
        mlm_mask: Tensor, shape = [N]，值为 0（不mask）或 1（mask）
        """

        def mask_one(name):
            x = tokens[name]  # [B, L, D] or [B, L, H, W]
            m = mlm_mask[name]  # [B, L]
            m = m.unsqueeze(-1)  # [B, L, 1]
            while m.dim() < x.dim():
                m = m.unsqueeze(-1)  # → 对齐 x 的维度

            embed = self.mask_embed[name]  # 可能是 [1, D] or [1, 1, D] 等

            # reshape embed 为 [1, 1, ...] + [dim of x[2:]]
            embed_shape = [1] * 2 + list(x.shape[2:])
            embed = embed.view(*embed_shape)  # 与 x 对齐

            # ✅ mask==1 → 替换为 embed，mask==0 → 保留原值
            return torch.where(m.bool(), embed, x)

        return {channel_name: mask_one(channel_name) for channel_name in tokens.keys()}

    def _fuse_modalities(self, token_embeddings):
        modalities = [token_embeddings[key] for key in token_embeddings]  # 取出 N 个 [4, 120, 512] 张量
        fused_token_embeddings = torch.cat(modalities, dim=-1)  # 拼接最后一维
        return fused_token_embeddings

    def apply_padding_mask(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """
        Legacy wrapper: delegates to the configured CLS strategy to (optionally) prepend CLS
        and build an attention mask.
        """
        return self.cls_embedding.add_cls_and_mask(tokens, lengths)

    def _build_router_ctx(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        channel_name: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.router_ctx_encoder is None:
            return None
        modality_id = self.channel_to_id.get(channel_name, 0)
        modality_ids = torch.full((batch_size,), modality_id, dtype=torch.long, device=device)
        return self.router_ctx_encoder(
            metadata=metadata,
            modality_ids=modality_ids,
            batch_size=batch_size,
            device=device,
        )

    def _build_router_group_ids(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        *,
        batch_size: int,
        device: torch.device,
        channel_name: str | None = None,
    ) -> dict[str, torch.Tensor] | None:
        encoder_cfg = getattr(self.encoder, "config", None)
        if encoder_cfg is None or not getattr(encoder_cfg, "moe_enabled", False):
            return None

        group_keys = list(getattr(encoder_cfg, "moe_group_keys", ["source"]))
        out: dict[str, torch.Tensor] = {}
        if metadata is None:
            return None

        def _align(vec: torch.Tensor, name: str) -> torch.Tensor:
            if vec.dim() == 0:
                vec = vec.reshape(1)
            if vec.shape[0] == 1 and batch_size > 1:
                vec = vec.expand(batch_size)
            if vec.shape[0] != batch_size:
                raise ValueError(f"{name} has batch size {vec.shape[0]}, expected {batch_size}")
            return vec

        for key in group_keys:
            if key == "source":
                if self.router_ctx_encoder is not None:
                    source_ids = self.router_ctx_encoder.encode_source_ids(
                        metadata.get("source"),
                        device=device,
                        batch_size=batch_size,
                    )
                else:
                    source_values = metadata.get("source")
                    if isinstance(source_values, (list, tuple)):
                        source_to_id: dict[str, int] = {}
                        encoded: list[int] = []
                        for value in source_values:
                            source_key = str(value)
                            if source_key not in source_to_id:
                                source_to_id[source_key] = len(source_to_id)
                            encoded.append(source_to_id[source_key])
                        source_ids = torch.tensor(encoded, dtype=torch.long, device=device)
                    else:
                        source_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
                out["source"] = _align(source_ids, "metadata.source")
            elif key == "sex":
                sex_raw = metadata.get("sex")
                if sex_raw is None:
                    continue
                sex = torch.as_tensor(sex_raw, dtype=torch.long, device=device)
                out["sex"] = _align(sex, "metadata.sex")
            elif key == "age_bin":
                age_bins = int(getattr(encoder_cfg, "moe_age_bins", 10))
                age_raw = metadata.get("age")
                if age_raw is None:
                    continue
                age = torch.as_tensor(age_raw, dtype=torch.float32, device=device)
                age = _align(age, "metadata.age")
                valid = age >= 0
                bins = torch.full((batch_size,), -1, dtype=torch.long, device=device)
                if valid.any():
                    bin_ids = torch.clamp((age[valid] / 100.0 * age_bins).long(), min=0, max=age_bins - 1)
                    bins[valid] = bin_ids
                out["age_bin"] = bins
            elif key == "modality":
                modality_id = self.channel_to_id.get(channel_name, 0) if channel_name is not None else 0
                out["modality"] = torch.full((batch_size,), modality_id, dtype=torch.long, device=device)

        return out or None

    def _extract_moe_from_encoder_output(
        self,
        encoder_output,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        moe_loss = getattr(encoder_output, "moe_loss", None)
        moe_metrics = getattr(encoder_output, "moe_metrics", None)

        if moe_loss is None:
            moe_loss = torch.zeros((), dtype=torch.float32, device=device)
        if moe_metrics is None:
            moe_metrics = {}
        return moe_loss, moe_metrics

    @staticmethod
    def _merge_moe_metrics(
        first_metrics: dict[str, torch.Tensor],
        second_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for key, value in first_metrics.items():
            merged[f"moe/first/{key}"] = value
        for key, value in second_metrics.items():
            merged[f"moe/second/{key}"] = value

        shared_keys = set(first_metrics.keys()) & set(second_metrics.keys())
        for key in shared_keys:
            first_value = first_metrics[key]
            second_value = second_metrics[key]
            if torch.is_tensor(first_value) and torch.is_tensor(second_value):
                if first_value.shape == second_value.shape:
                    avg_value = 0.5 * (first_value + second_value)
                    merged[f"moe/avg/{key}"] = avg_value
                    if key.startswith("mean/"):
                        merged[f"moe/avg/{key[len('mean/') :]}"] = avg_value
                else:
                    merged[f"moe/avg/{key}"] = first_value
            else:
                merged[f"moe/avg/{key}"] = first_value
        return merged

    def _token_embeddings_to_hidden(
        self,
        token_embeddings,
        batch,
        *,
        return_hidden_states: bool = False,
        router_ctx: torch.Tensor | None = None,
        router_group_ids: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None] | tuple[
        torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, t.Any]
    ]:

        # 3. 调整 token_embedding 维度为 transformer hidden_size
        token_embeddings = self.embedding_projection(token_embeddings)

        # 4. 通过策略添加 CLS & 构造 padding mask
        token_embeddings, attention_mask = self.apply_padding_mask(token_embeddings, batch["length"])

        # 5. Transformer encoder
        if return_hidden_states:
            hidden, hidden_states, aux = self._run_encoder(
                token_embeddings,
                attention_mask,
                return_hidden_states=True,
                router_ctx=router_ctx,
                router_group_ids=router_group_ids,
                return_aux=return_aux,
            )
        else:
            hidden, aux = self._run_encoder(
                token_embeddings,
                attention_mask,
                router_ctx=router_ctx,
                router_group_ids=router_group_ids,
                return_aux=return_aux,
            )
            hidden_states = None

        if return_aux:
            return hidden, attention_mask, hidden_states, aux
        return hidden, attention_mask, hidden_states

    def _run_encoder(
        self,
        token_embeddings,
        attention_mask,
        *,
        return_hidden_states: bool = False,
        router_ctx: torch.Tensor | None = None,
        router_group_ids: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ):
        """Routes embeddings through the selected encoder."""
        if self._custom_encoder_forward is not None:
            if return_hidden_states:
                raise ValueError("Custom encoder forward does not support hidden states.")
            hidden = self._custom_encoder_forward(self.encoder, token_embeddings, attention_mask)
            aux = {"moe_loss": torch.zeros((), dtype=torch.float32, device=hidden.device), "moe_metrics": {}}
            return (hidden, aux) if return_aux else (hidden, None)

        call_kwargs = dict(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
            router_ctx=router_ctx,
            router_group_ids=router_group_ids,
        )
        try:
            encoder_output = self.encoder(**call_kwargs)
        except TypeError:
            call_kwargs.pop("router_ctx", None)
            call_kwargs.pop("router_group_ids", None)
            try:
                encoder_output = self.encoder(**call_kwargs)
            except TypeError as exc:
                if return_hidden_states:
                    raise ValueError(f"Encoder '{self.encoder_name}' does not support output_hidden_states.") from exc
                encoder_output = self.encoder(inputs_embeds=token_embeddings, attention_mask=attention_mask)

        def _extract_last_hidden(output):
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            if isinstance(output, (list, tuple)):
                return output[0]
            raise ValueError(
                f"Encoder '{self.encoder_name}' returned unsupported output type "
                f"{type(output)}. Provide encoder_forward to customize the "
                "forward pass."
            )

        last_hidden = _extract_last_hidden(encoder_output)
        if return_aux:
            moe_loss, moe_metrics = self._extract_moe_from_encoder_output(encoder_output, device=last_hidden.device)
            aux = {"moe_loss": moe_loss, "moe_metrics": moe_metrics}
        else:
            aux = None

        if not return_hidden_states:
            if return_aux:
                return last_hidden, aux
            return last_hidden, None

        hidden_states = getattr(encoder_output, "hidden_states", None)
        if hidden_states is None and isinstance(encoder_output, (list, tuple)):
            for candidate_idx in (1, 2, 3):
                if len(encoder_output) <= candidate_idx:
                    continue
                candidate = encoder_output[candidate_idx]
                if isinstance(candidate, (list, tuple)):
                    hidden_states = candidate
                    break
        if hidden_states is None:
            raise ValueError(
                f"Encoder '{self.encoder_name}' did not return hidden states. "
                "Ensure output_hidden_states is supported by the backbone."
            )
        if return_aux:
            return last_hidden, hidden_states, aux
        return last_hidden, hidden_states, None

    def get_encoder(self) -> nn.Module:
        """Returns the active encoder module."""
        return self.encoder

    def replace_encoder(self, encoder: nn.Module):
        """Swap the underlying encoder (e.g., to wrap with PEFT)."""
        self.encoder = encoder

    def forward(self, batch, apply_mask):
        tokens = batch["tokens"]
        if self.specified_two_mods:
            chosen_channels = [channel for channel in self.specified_two_mods if channel in tokens]
        else:
            chosen_channels = list(tokens.keys())
        if len(chosen_channels) < 2:
            raise ValueError(f"Expected at least two channels in batch['tokens'], got {chosen_channels}")
        first_channel, second_channel = chosen_channels[0], chosen_channels[1]

        token_embeddings = {
            first_channel: self.tokenizer_mapping[first_channel](tokens[first_channel]),
            second_channel: self.tokenizer_mapping[second_channel](tokens[second_channel]),
        }
        if apply_mask:
            token_embeddings = self._mask_modalities(token_embeddings, batch["mlm_mask"])

        first_mod_token_embeddings = token_embeddings[first_channel]
        second_mod_token_embeddings = token_embeddings[second_channel]

        metadata = batch.get("metadata")
        batch_size = int(first_mod_token_embeddings.shape[0])
        device = first_mod_token_embeddings.device
        first_group_ids = self._build_router_group_ids(
            metadata,
            batch_size=batch_size,
            device=device,
            channel_name=first_channel,
        )
        second_group_ids = self._build_router_group_ids(
            metadata,
            batch_size=batch_size,
            device=device,
            channel_name=second_channel,
        )
        first_ctx = self._build_router_ctx(metadata, first_channel, batch_size=batch_size, device=device)
        second_ctx = self._build_router_ctx(metadata, second_channel, batch_size=batch_size, device=device)

        first_hidden, _, _, first_aux = self._token_embeddings_to_hidden(
            first_mod_token_embeddings,
            batch,
            router_ctx=first_ctx,
            router_group_ids=first_group_ids,
            return_aux=True,
        )
        second_hidden, _, _, second_aux = self._token_embeddings_to_hidden(
            second_mod_token_embeddings,
            batch,
            router_ctx=second_ctx,
            router_group_ids=second_group_ids,
            return_aux=True,
        )

        if self.projection:
            first_hidden = self.proj_head(first_hidden)
            second_hidden = self.proj_head(second_hidden)

        first_moe_loss = first_aux["moe_loss"]
        second_moe_loss = second_aux["moe_loss"]
        moe_loss = 0.5 * (first_moe_loss + second_moe_loss)

        first_metrics = first_aux.get("moe_metrics", {})
        second_metrics = second_aux.get("moe_metrics", {})
        merged_metrics = self._merge_moe_metrics(first_metrics, second_metrics)
        aux = {"moe_loss": moe_loss, **merged_metrics}
        return first_hidden, second_hidden, aux

    def encode(self, batch, channel_name):
        tokens = batch["tokens"]

        # 1. tokenize
        token_embeddings = self.tokenizer_mapping[channel_name](tokens[channel_name])

        # 3/4/5
        hidden, _, _ = self._token_embeddings_to_hidden(token_embeddings, batch)

        # 对所有 token 逐个投影：得到 [B, L, 128]
        if self.projection:
            hidden = self.proj_head(hidden)

        return hidden

    def set_tokenizers_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze tokenizer parameters without altering module train/eval mode."""
        if hasattr(self, "tokenizer_mapping"):
            for param in self.tokenizer_mapping.parameters():
                param.requires_grad = trainable
        else:
            for name, module in self.named_children():
                if "tokenizer" in name:
                    for param in module.parameters():
                        param.requires_grad = trainable

    def freeze_backbone_groups(
        self,
        train_projection: bool = False,
        train_mask_embed: bool = False,  # 基本保持 False
        train_tokenizers: bool = False,  # 基本保持 False
        train_ln_in_roformer: bool = False,  # Historical name; still controls encoder LN
    ):
        """Freezes most of the backbone while keeping optional groups trainable."""
        for _, p in self.named_parameters():
            p.requires_grad = False

        # 2) 只开 LoRA 分支
        encoder = self.get_encoder()
        if hasattr(encoder, "enable_adapter_layers"):
            encoder.enable_adapter_layers()
        for n, p in encoder.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        def _select_parameters(predicate: t.Callable[[str], bool]):
            for name, param in self.named_parameters():
                if predicate(name):
                    param.requires_grad = True

        # 3) 可选：解冻 projection / mask_embed / tokenizer
        if train_projection:
            _select_parameters(lambda n: "embedding_projection" in n)
        if train_mask_embed:
            _select_parameters(lambda n: n.startswith("mask_embed."))
        if train_tokenizers:
            _select_parameters(lambda n: any(key in n for key in ["high_tokenizer", "low_tokenizer"]))

        # 4) 冻结了 tokenizer 时，把其中的 BN/Dropout 置 eval，避免统计量漂移
        if not train_tokenizers:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                    m.eval()

        # 5) 可选：解冻 encoder 的 LayerNorm（保持旧参数名以兼容脚本）
        if train_ln_in_roformer:
            for n, p in encoder.named_parameters():
                if any(tag in n.lower() for tag in ["layernorm", "ln"]):
                    p.requires_grad = True

        # 打印统计
        total = sum(p.numel() for _, p in self.named_parameters())
        trainable = sum(p.numel() for _, p in self.named_parameters() if p.requires_grad)
        logging.info(f"[freeze_backbone_groups] backbone trainable: {trainable}/{total} ({trainable/total:.4%})")
