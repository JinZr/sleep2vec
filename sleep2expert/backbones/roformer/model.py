from __future__ import annotations

"""Public standalone RoFormer encoder model."""

import torch
from torch import nn

from .configuration import RoFormerConfig
from .modeling_roformer import RoFormerEmbeddings, RoFormerEncoder


class RoFormerEncoderModel(nn.Module):
    """Standalone RoFormer encoder with an embeddings-first forward API."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = RoFormerEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = RoFormerEncoder(config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    @staticmethod
    def _extend_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        else:
            raise ValueError(
                "attention_mask should have 2, 3, or 4 dimensions; " f"received shape {tuple(attention_mask.shape)}"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        # Match HF behavior exactly for masked positions.
        return (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        modality_name: str | None = None,
        collect_moe_aux: bool = False,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
    ):
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided for RoFormerEncoderModel.forward")

        if attention_mask is None:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            attention_mask = torch.ones(input_shape, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)

        extended_attention_mask = self._extend_attention_mask(attention_mask, dtype=embedding_output.dtype)

        return self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modality_name=modality_name,
            collect_moe_aux=collect_moe_aux,
        )
