import logging
import typing as t

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from .downstream.head_registry import create_head
from .downstream.heads import AttnPooling
from .pretrain_model import Sleep2vecPretrainModel


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
    ):
        super().__init__()
        self.backbone = backbone
        self.channel_names = channel_names
        self.device = device
        self.output_dim = output_dim
        self.is_classification = is_classification
        self.is_seq = is_seq
        self.target = target

        self.n_channels = len(self.channel_names)

        head_kwargs = head_kwargs or {}
        inferred_head = head_name or (
            "classification" if self.is_classification else "regression"
        )
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

        self.use_temporal_attn = False  # control flag
        self.separate_adapters = False  # default
        self._adapter_warning_logged = False

        if (not self.is_seq) and self.use_temporal_attn:
            self.temporal_agg = AttnPooling(
                self.backbone.transformer_hidden_size, heads=1, temp=1.0, dropout=0.0
            )

    def _backbone_encoder(self) -> nn.Module:
        """Returns the encoder module inside the backbone."""
        if hasattr(self.backbone, "get_encoder"):
            return self.backbone.get_encoder()

        encoder = getattr(self.backbone, "encoder", None)
        if encoder is None:
            encoder = getattr(self.backbone, "roformer", None)
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
            if hasattr(self.backbone, "roformer"):
                self.backbone.roformer = encoder

    def _set_active_adapter(self, adapter_name: str):
        """Switch adapters if the encoder exposes adapter APIs."""
        encoder = self._backbone_encoder()
        if hasattr(encoder, "set_adapter"):
            encoder.set_adapter(adapter_name)
        elif not self._adapter_warning_logged:
            logging.warning(
                "Encoder lacks 'set_adapter'; separate adapters are ignored."
            )
            self._adapter_warning_logged = True

    def forward(self, batch):
        tokens = batch["tokens"]

        token_embeddings = self.backbone._tokenize_all(tokens)
        token_names, token_embeddings = list(token_embeddings.keys()), list(
            token_embeddings.values()
        )

        feature_of_different_mods = []
        for token_name, single_mod_token_embeddings in zip(
            token_names, token_embeddings
        ):

            if getattr(self, "separate_adapters", False):
                self._set_active_adapter(f"ch_{token_name}")

            hidden = self.backbone._token_embeddings_to_hidden(
                single_mod_token_embeddings, batch
            )

            if self.is_seq:
                seq_hidden = hidden
                feature_of_different_mods.append(seq_hidden)
            else:
                seq_hidden = hidden
                B, L, _ = seq_hidden.shape
                mask = torch.zeros(B, L, dtype=torch.bool, device=seq_hidden.device)
                for i in range(B):
                    mask[i, : batch["length"][i].item()] = True

                if self.use_temporal_attn:
                    pooled, _ = self.temporal_agg(seq_hidden, mask)
                else:
                    seq_hidden_masked = seq_hidden * mask.unsqueeze(-1)
                    denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
                    pooled = seq_hidden_masked.sum(dim=1) / denom
                feature_of_different_mods.append(pooled)

        output = self.head(feature_of_different_mods)
        return output

    def load_pretrained_backbone(self, ckpt_path):
        logging.info(f"Loading backbone from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # 仅保留属于 model.backbone 的权重
        filtered_state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        # 加载到 self.backbone
        load_info = self.backbone.load_state_dict(filtered_state_dict, strict=False)

        # 打印加载结果
        total_keys = len(filtered_state_dict)
        missing_keys = load_info.missing_keys
        unexpected_keys = load_info.unexpected_keys

        logging.info(
            f"✅ Loaded {total_keys - len(missing_keys)} / {total_keys} keys into backbone."
        )
        if missing_keys:
            logging.warning(f"Missing keys ({len(missing_keys)}):")
            for k in missing_keys:
                logging.warning(f"    {k}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys ({len(unexpected_keys)}):")
            for k in unexpected_keys:
                logging.warning(f"    {k}")

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

        # —— 全模型统计 ——
        total_all = sum(p.numel() for _, p in self.named_parameters())
        train_all = sum(
            p.numel() for _, p in self.named_parameters() if p.requires_grad
        )
        logging.info(
            f"[insert_lora] model trainable params: {train_all}/{total_all} ({train_all/total_all:.4%})"
        )

        # —— 只看 backbone 统计 ——
        b_total = sum(p.numel() for _, p in self.backbone.named_parameters())
        b_train = sum(
            p.numel() for _, p in self.backbone.named_parameters() if p.requires_grad
        )
        b_ratio = b_train / b_total if b_total > 0 else 0.0
        lora_train = sum(
            p.numel()
            for n, p in self.backbone.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        logging.info(
            f"[insert_lora] backbone trainable params: {b_train}/{b_total} ({b_ratio:.4%}); LoRA-only trainable: {lora_train}"
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

    def print_backbone_param_names(self):
        logging.info("== All parameters (name, shape, dtype, device) ==")
        for n, p in self._backbone_encoder().named_parameters():
            logging.info(f"{n:80s} {tuple(p.shape)} {p.dtype} {p.device}")
