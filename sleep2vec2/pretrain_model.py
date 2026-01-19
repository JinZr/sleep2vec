import logging
import random
import typing as t

import torch
import torch.nn as nn

from sleep2vec2.builders import build_encoder, build_projection, build_tokenizers_and_dim
from sleep2vec2.cls import build_cls_embedding
from sleep2vec2.config import BackboneConfig, ModelConfig, ProjectionConfig
from sleep2vec2.modules.metadata_context import MetadataContextEncoder, stable_hash_to_bucket
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
        encoder: nn.Module | None = None,
        encoder_config_overrides: t.Optional[dict[str, t.Any]] = None,
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
            backbone_cfg = model_config.backbone
            # In sleep2vec2 the YAML config is authoritative. These legacy arguments are
            # accepted for backwards compatibility but are ignored when model_config is given.
            if transformer_hidden_size is not None and transformer_hidden_size != backbone_cfg.hidden_size:
                logging.warning(
                    "Ignoring transformer_hidden_size=%s; using model.backbone.hidden_size=%s from YAML.",
                    transformer_hidden_size,
                    backbone_cfg.hidden_size,
                )

            transformer_hidden_size = backbone_cfg.hidden_size
            encoder = encoder or build_encoder(backbone_cfg)
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

            if transformer_hidden_size is None:
                raise ValueError("transformer_hidden_size must be provided when model_config is absent.")

            # Legacy CLI / programmatic initialization (no YAML): build a local
            # RoFormer encoder directly.
            if encoder is None:
                cfg = BackboneConfig(
                    name="roformer",
                    hidden_size=transformer_hidden_size,
                    num_hidden_layers=transformer_num_hidden_layers,
                    num_attention_heads=transformer_num_attention_heads,
                    # keep legacy vocab_size so old configs don't break
                    vocab_size=int(overrides.pop("vocab_size", 1)),
                    config_overrides=overrides,
                )
                encoder = build_encoder(cfg)

        if transformer_hidden_size is None:
            raise ValueError("transformer_hidden_size must be provided or inferred.")

        if encoder is None:
            raise ValueError("encoder could not be built.")

        if channel_feature_dim is None:
            raise ValueError("channel_feature_dim must be provided or inferred.")

        enc_cfg = getattr(encoder, "config", None)
        inferred_hidden_size = getattr(enc_cfg, "hidden_size", None) or transformer_hidden_size
        inferred_embedding_size = (
            getattr(enc_cfg, "embedding_size", None) or getattr(enc_cfg, "hidden_size", None) or inferred_hidden_size
        )

        self.encoder = encoder
        # Output (encoder) hidden dim used by downstream heads.
        self.transformer_hidden_size = int(inferred_hidden_size)
        # Input embedding dim expected by encoder(inputs_embeds=...).
        self.encoder_embedding_size = int(inferred_embedding_size)
        self.modality_to_id = {name: idx for idx, name in enumerate(self.channel_names)}
        self.encoder_name = (
            (getattr(model_config.backbone, "name", None) if model_config is not None else None)
            or getattr(enc_cfg, "model_type", None)
            or "roformer"
        )
        self.cls_embedding = build_cls_embedding(
            strategy=self.cls_cfg.embedding_type if self.cls_cfg else None,
            hidden_size=self.encoder_embedding_size,
        )
        # downstream preference (cls/tokens/None) stored for heads to read
        self.cls_downstream = self.cls_cfg.downstream if self.cls_cfg else None

        self.mask_embed = nn.ParameterDict(
            {channel_name: nn.Parameter(torch.ones(channel_feature_dim)) for channel_name in self.channel_names}
        )

        self.embedding_projection = nn.Linear(channel_feature_dim, self.encoder_embedding_size)

        self.proj_head = build_projection(
            (projection_config if projection_config is not None else ProjectionConfig(enabled=projection or False)),
            in_dim=self.transformer_hidden_size,
        )
        self.projection = bool(self.proj_head)

        self._token_type_count = int(getattr(enc_cfg, "num_token_types", 0) or 0)
        self._token_type_enabled = self._token_type_count > 0
        self._token_type_warned = False

        self._router_context_dim = int(getattr(enc_cfg, "moe_router_context_dim", 0) or 0)
        self._use_router_context = self._router_context_dim > 0
        if self._use_router_context:
            self.modality_embed = nn.Embedding(len(self.channel_names), self._router_context_dim)
            self.meta_encoder = MetadataContextEncoder(meta_dim=self._router_context_dim)
            self.context_proj = nn.Sequential(
                nn.Linear(self._router_context_dim * 3, self._router_context_dim),
                nn.GELU(),
                nn.Linear(self._router_context_dim, self._router_context_dim),
            )
        else:
            self.modality_embed = None
            self.meta_encoder = None
            self.context_proj = None

        self.total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Total parameters: {self.total_params}")
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {self.trainable_params}")

    def _tokenize_two_random_channels(self, tokens):
        """
        随机选择两个不同的模态并做tokenization
        """

        if self.specified_two_mods:
            chosen_channels = self.specified_two_mods
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

    def _stable_hash_to_bucket(self, values: t.Sequence[str], num_buckets: int, device: torch.device) -> torch.Tensor:
        return stable_hash_to_bucket(values, num_buckets).to(device=device)

    def _build_meta_context(self, batch: dict, device: torch.device) -> torch.Tensor | None:
        if self.meta_encoder is None:
            return None
        metadata = batch.get("metadata")
        if metadata is None:
            return None

        age = metadata.get("age")
        sex = metadata.get("sex")
        if age is None or sex is None:
            return None

        source = metadata.get("source") or ["nan"] * len(age)
        path = metadata.get("path") or ["nan"] * len(age)

        source_ids = self._stable_hash_to_bucket(source, self.meta_encoder.num_source_buckets, device)
        subject_ids = self._stable_hash_to_bucket(path, self.meta_encoder.num_subject_buckets, device)

        return self.meta_encoder(
            age=age,
            sex=sex,
            source_ids=source_ids,
            subject_ids=subject_ids,
        )

    def _build_set_embedding(self, available_modalities: t.Sequence[str], device: torch.device) -> torch.Tensor | None:
        if self.modality_embed is None:
            return None
        ids = [self.modality_to_id[m] for m in available_modalities if m in self.modality_to_id]
        if not ids:
            return None
        id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        return self.modality_embed(id_tensor).mean(dim=0)

    def _build_router_context(
        self,
        batch: dict,
        *,
        modality_name: str | None,
        available_modalities: t.Sequence[str] | None,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor | None:
        if not self._use_router_context or self.context_proj is None:
            return None

        meta = self._build_meta_context(batch, device)
        if meta is None:
            meta = torch.zeros(batch_size, self._router_context_dim, device=device)

        if modality_name is None or self.modality_embed is None:
            return None
        mod_id = self.modality_to_id.get(modality_name)
        if mod_id is None:
            return None
        mod_emb = self.modality_embed(torch.tensor([mod_id], dtype=torch.long, device=device)).expand(batch_size, -1)

        set_emb = None
        if available_modalities is not None:
            set_emb = self._build_set_embedding(available_modalities, device)
        if set_emb is None:
            set_emb = mod_emb[0]
        set_emb = set_emb.expand(batch_size, -1)

        context = torch.cat([meta, mod_emb, set_emb], dim=-1)
        return self.context_proj(context)

    def _token_embeddings_to_hidden(
        self,
        token_embeddings,
        batch,
        *,
        modality_name: str | None = None,
        available_modalities: t.Sequence[str] | None = None,
        return_mask: bool = False,
        return_extras: bool = False,
    ):

        # 3. 调整 token_embedding 维度为 transformer hidden_size
        token_embeddings = self.embedding_projection(token_embeddings)

        # 4. 通过策略添加 CLS & 构造 padding mask
        token_embeddings, attention_mask = self.apply_padding_mask(token_embeddings, batch["length"])

        # 5. Transformer encoder
        token_type_ids = None
        if self._token_type_enabled and modality_name is not None:
            mod_id = self.modality_to_id.get(modality_name)
            if mod_id is not None and mod_id < self._token_type_count:
                token_type_ids = torch.full(
                    attention_mask.shape,
                    int(mod_id),
                    dtype=torch.long,
                    device=token_embeddings.device,
                )
            elif mod_id is not None and not self._token_type_warned:
                logging.warning(
                    "modality id %s exceeds num_token_types=%s; "
                    "set model.backbone.num_token_types to >= %s to enable token type ids.",
                    mod_id,
                    self._token_type_count,
                    max(self.modality_to_id.values()) + 1,
                )
                self._token_type_warned = True

        router_context = self._build_router_context(
            batch,
            modality_name=modality_name,
            available_modalities=available_modalities,
            device=token_embeddings.device,
            batch_size=token_embeddings.shape[0],
        )

        extras = None
        if return_extras:
            hidden, extras = self._run_encoder(
                token_embeddings,
                attention_mask,
                token_type_ids=token_type_ids,
                router_context=router_context,
                token_mask=attention_mask,
                return_extras=True,
            )
        else:
            hidden = self._run_encoder(
                token_embeddings,
                attention_mask,
                token_type_ids=token_type_ids,
                router_context=router_context,
                token_mask=attention_mask,
                return_extras=False,
            )

        if return_mask and return_extras:
            return hidden, attention_mask, extras
        if return_mask:
            return hidden, attention_mask
        if return_extras:
            return hidden, extras
        return hidden

    def _run_encoder(
        self,
        token_embeddings,
        attention_mask,
        *,
        token_type_ids: torch.Tensor | None = None,
        router_context: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        return_extras: bool = False,
    ):
        """Routes embeddings through the selected encoder."""
        if self._custom_encoder_forward is not None:
            hidden = self._custom_encoder_forward(self.encoder, token_embeddings, attention_mask)
            if return_extras:
                return hidden, {}
            return hidden

        try:
            encoder_output = self.encoder(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                router_context=router_context,
                token_mask=token_mask,
            )
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            encoder_output = self.encoder(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        if isinstance(encoder_output, torch.Tensor):
            if return_extras:
                return encoder_output, {}
            return encoder_output
        if hasattr(encoder_output, "last_hidden_state"):
            hidden = encoder_output.last_hidden_state
            if return_extras:
                extras = {
                    "moe_aux_loss": getattr(encoder_output, "moe_aux_loss", None),
                    "moe_z_loss": getattr(encoder_output, "moe_z_loss", None),
                    "moe_route_mean": getattr(encoder_output, "moe_route_mean", None),
                    "moe_importance": getattr(encoder_output, "moe_importance", None),
                    "moe_load": getattr(encoder_output, "moe_load", None),
                    "moe_entropy": getattr(encoder_output, "moe_entropy", None),
                }
                return hidden, extras
            return hidden
        if isinstance(encoder_output, (list, tuple)):
            hidden = encoder_output[0]
            if return_extras:
                return hidden, {}
            return hidden
        raise ValueError(
            f"Encoder '{self.encoder_name}' returned unsupported output type "
            f"{type(encoder_output)}. Provide encoder_forward to customize the "
            "forward pass."
        )

    def get_encoder(self) -> nn.Module:
        """Returns the active encoder module."""
        return self.encoder

    def replace_encoder(self, encoder: nn.Module):
        """Swap the underlying encoder (e.g., to wrap with PEFT)."""
        self.encoder = encoder

    def forward(self, batch, apply_mask):

        # 随机选择两个 channel 做 mask 对比学习
        assert len(self.channel_names) >= 2, "At two channels are required for this method!"
        tokens = batch["tokens"]

        # 1. 随机选择两个通道并 tokenize
        token_embeddings = self._tokenize_two_random_channels(tokens)
        modality_names = list(token_embeddings.keys())

        # 2. mask 选择的两个通道
        if apply_mask:
            token_embeddings = self._mask_modalities(token_embeddings, batch["mlm_mask"])

        token_embeddings_list = list(token_embeddings.values())
        first_mod_token_embeddings, second_mod_token_embeddings = token_embeddings_list[0], token_embeddings_list[1]
        first_name, second_name = modality_names[0], modality_names[1]

        # 3/4/5
        first_hidden, first_extras = self._token_embeddings_to_hidden(
            first_mod_token_embeddings,
            batch,
            modality_name=first_name,
            available_modalities=modality_names,
            return_extras=True,
        )
        second_hidden, second_extras = self._token_embeddings_to_hidden(
            second_mod_token_embeddings,
            batch,
            modality_name=second_name,
            available_modalities=modality_names,
            return_extras=True,
        )

        # ★ 对所有 token 逐个投影：得到 [B, L, 128]
        if self.projection:
            first_hidden = self.proj_head(first_hidden)
            second_hidden = self.proj_head(second_hidden)

        extras = {}
        if first_extras:
            extras["moe_aux_loss_first"] = first_extras.get("moe_aux_loss")
            extras["moe_z_loss_first"] = first_extras.get("moe_z_loss")
            extras["moe_route_mean_first"] = first_extras.get("moe_route_mean")
            extras["moe_importance_first"] = first_extras.get("moe_importance")
            extras["moe_load_first"] = first_extras.get("moe_load")
            extras["moe_entropy_first"] = first_extras.get("moe_entropy")
        if second_extras:
            extras["moe_aux_loss_second"] = second_extras.get("moe_aux_loss")
            extras["moe_z_loss_second"] = second_extras.get("moe_z_loss")
            extras["moe_route_mean_second"] = second_extras.get("moe_route_mean")
            extras["moe_importance_second"] = second_extras.get("moe_importance")
            extras["moe_load_second"] = second_extras.get("moe_load")
            extras["moe_entropy_second"] = second_extras.get("moe_entropy")

        if any(v is not None for v in extras.values()):
            return first_hidden, second_hidden, extras
        return first_hidden, second_hidden

    def encode(self, batch, channel_name):
        tokens = batch["tokens"]

        # 1. tokenize
        token_embeddings = self.tokenizer_mapping[channel_name](tokens[channel_name])

        # 3/4/5
        hidden = self._token_embeddings_to_hidden(
            token_embeddings,
            batch,
            modality_name=channel_name,
            available_modalities=[channel_name],
        )

        # 对所有 token 逐个投影：得到 [B, L, 128]
        if self.projection:
            hidden = self.proj_head(hidden)

        return hidden

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
