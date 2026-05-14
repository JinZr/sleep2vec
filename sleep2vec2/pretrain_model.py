import logging
import random
import typing as t

import torch
import torch.nn as nn

from sleep2vec2.backbones.encoder_factory import TransformerEncoderFactory
from sleep2vec2.builders import build_encoder_factory, build_projection, build_tokenizers_and_dim
from sleep2vec2.cls import build_cls_embedding
from sleep2vec2.config import ModelConfig, ProjectionConfig
from sleep2vec2.modules.tokenizers import LinearTokenizer, SundialTokenizer


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
        self._forced_eval_modules: list[nn.Module] = []
        self._forced_train_modules: list[nn.Module] = []

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

    def _token_embeddings_to_hidden(
        self,
        token_embeddings,
        batch,
        *,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:

        # 3. 调整 token_embedding 维度为 transformer hidden_size
        token_embeddings = self.embedding_projection(token_embeddings)

        # 4. 通过策略添加 CLS & 构造 padding mask
        token_embeddings, attention_mask = self.apply_padding_mask(token_embeddings, batch["length"])

        # 5. Transformer encoder
        if return_hidden_states:
            hidden, hidden_states = self._run_encoder(token_embeddings, attention_mask, return_hidden_states=True)
        else:
            hidden = self._run_encoder(token_embeddings, attention_mask)
            hidden_states = None

        return hidden, attention_mask, hidden_states

    def _run_encoder(self, token_embeddings, attention_mask, *, return_hidden_states: bool = False):
        """Routes embeddings through the selected encoder."""
        if self._custom_encoder_forward is not None:
            if return_hidden_states:
                raise ValueError("Custom encoder forward does not support hidden states.")
            return self._custom_encoder_forward(self.encoder, token_embeddings, attention_mask)

        try:
            encoder_output = self.encoder(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
            )
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
        if not return_hidden_states:
            return last_hidden

        hidden_states = getattr(encoder_output, "hidden_states", None)
        if hidden_states is None and isinstance(encoder_output, (list, tuple)) and len(encoder_output) > 2:
            hidden_states = encoder_output[2]
        if hidden_states is None:
            raise ValueError(
                f"Encoder '{self.encoder_name}' did not return hidden states. "
                "Ensure output_hidden_states is supported by the backbone."
            )
        return last_hidden, hidden_states

    def get_encoder(self) -> nn.Module:
        """Returns the active encoder module."""
        return self.encoder

    def replace_encoder(self, encoder: nn.Module):
        """Swap the underlying encoder (e.g., to wrap with PEFT)."""
        self.encoder = encoder

    def forward(self, batch, apply_mask):

        # 随机选择两个 channel 做 mask 对比学习
        tokens = batch["tokens"]

        # 1. 随机选择两个通道并 tokenize
        token_embeddings = self._tokenize_two_random_channels(tokens)

        # 2. mask 选择的两个通道
        if apply_mask:
            token_embeddings = self._mask_modalities(token_embeddings, batch["mlm_mask"])

        # modality_names = list(token_embeddings.keys())
        token_embeddings = list(token_embeddings.values())
        first_mod_token_embeddings, second_mod_token_embeddings = (
            token_embeddings[0],
            token_embeddings[1],
        )

        # 3/4/5
        first_hidden, _, _ = self._token_embeddings_to_hidden(first_mod_token_embeddings, batch)
        second_hidden, _, _ = self._token_embeddings_to_hidden(second_mod_token_embeddings, batch)

        # ★ 对所有 token 逐个投影：得到 [B, L, 128]
        if self.projection:
            first_hidden = self.proj_head(first_hidden)
            second_hidden = self.proj_head(second_hidden)

        return first_hidden, second_hidden

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

    def _resolve_adaptation_channels(self, new_channels: t.Sequence[str]) -> tuple[set[str], set[str]]:
        new_set = {str(name) for name in new_channels}
        unknown = sorted(new_set - set(self.channel_names))
        if unknown:
            raise ValueError(f"Unknown adaptation channels {unknown}. Available channels: {self.channel_names}")
        legacy_set = {name for name in self.channel_names if name not in new_set}
        return new_set, legacy_set

    def _adaptation_group_for_param(self, name: str, new_channels: t.Set[str]) -> str | None:
        if name.startswith("encoder.") or name.startswith("cls_embedding."):
            return "encoder_cls"
        if name.startswith("embedding_projection.") or name.startswith("proj_head."):
            return "shared_projection"
        if name.startswith("tokenizer_mapping."):
            channel_name = name.split(".", 2)[1]
            return "new_modalities" if channel_name in new_channels else "legacy_modalities"
        if name.startswith("mask_embed."):
            channel_name = name.split(".", 2)[1]
            return "new_modalities" if channel_name in new_channels else "legacy_modalities"
        return None

    def get_adaptation_param_groups(self, new_channels: t.Sequence[str]) -> dict[str, list[tuple[str, nn.Parameter]]]:
        new_set, _ = self._resolve_adaptation_channels(new_channels)
        groups = {
            "encoder_cls": [],
            "shared_projection": [],
            "legacy_modalities": [],
            "new_modalities": [],
        }
        for name, param in self.named_parameters():
            group_name = self._adaptation_group_for_param(name, new_set)
            if group_name is not None:
                groups[group_name].append((name, param))
        return groups

    def _set_adaptation_group_trainable(
        self,
        groups: dict[str, list[tuple[str, nn.Parameter]]],
        trainable_groups: set[str],
    ) -> None:
        for group_name, params in groups.items():
            flag = group_name in trainable_groups
            for _, param in params:
                param.requires_grad = flag

    def _tokenizer_modules(self, channel_names: t.Iterable[str]) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for channel_name in channel_names:
            if channel_name in self.tokenizer_mapping:
                modules.append(self.tokenizer_mapping[channel_name])
        return modules

    def _set_mode_policy(self, *, forced_eval: list[nn.Module], forced_train: list[nn.Module]) -> None:
        self._forced_eval_modules = list(dict.fromkeys(forced_eval))
        self._forced_train_modules = list(dict.fromkeys(forced_train))
        self.apply_forced_module_modes()

    def apply_forced_module_modes(self) -> None:
        for module in self._forced_eval_modules:
            module.eval()
        for module in self._forced_train_modules:
            module.train()

    def apply_adaptation_freeze_policy(
        self,
        *,
        phase: str,
        new_channels: t.Sequence[str],
        train_shared_projection: bool = False,
    ) -> None:
        new_set, legacy_set = self._resolve_adaptation_channels(new_channels)
        groups = self.get_adaptation_param_groups(new_set)

        for _, param in self.named_parameters():
            param.requires_grad = False

        if phase == "stage1":
            trainable_groups = {"new_modalities"}
            if train_shared_projection:
                trainable_groups.add("shared_projection")
            self._set_adaptation_group_trainable(groups, trainable_groups)

            forced_eval = [
                self.encoder,
                self.cls_embedding,
                *self._tokenizer_modules(legacy_set),
            ]
            if not train_shared_projection and self.proj_head is not None:
                forced_eval.append(self.proj_head)
            forced_train = self._tokenizer_modules(new_set)
            if train_shared_projection and self.proj_head is not None:
                forced_train.append(self.proj_head)
            self._set_mode_policy(
                forced_eval=[module for module in forced_eval if module is not None],
                forced_train=[module for module in forced_train if module is not None],
            )
        elif phase == "stage2":
            trainable_groups = {"encoder_cls", "shared_projection", "legacy_modalities", "new_modalities"}
            self._set_adaptation_group_trainable(groups, trainable_groups)
            forced_train = [self.encoder, self.cls_embedding, *self._tokenizer_modules(self.channel_names)]
            if self.proj_head is not None:
                forced_train.append(self.proj_head)
            self._set_mode_policy(
                forced_eval=[],
                forced_train=[module for module in forced_train if module is not None],
            )
        else:
            raise ValueError(f"Unsupported adaptation phase '{phase}'. Expected 'stage1' or 'stage2'.")

        total = sum(param.numel() for _, param in self.named_parameters())
        trainable = sum(param.numel() for _, param in self.named_parameters() if param.requires_grad)
        logging.info(
            "[adaptation_freeze_policy] phase=%s trainable=%s/%s (%s)",
            phase,
            trainable,
            total,
            f"{trainable / total:.4%}" if total else "0.0000%",
        )
