from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from sleep2vec.downstreams.heads.classification import ClassificationHead
from sleep2vec.modules.covariates import build_covariate_embedding, embed_covariates, ordered_covariates

from .config import BaselineConfig, covariate_task_config


class SexAgeMLP(nn.Module):
    """The signal models' covariate pathway without a backbone: shared covariate embeddings into a dense head."""

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        task_cfg = covariate_task_config(cfg)
        self.covariate_normalization = dict(task_cfg.covariate_normalization)
        self.embeddings = nn.ModuleDict(
            {
                name: build_covariate_embedding(name, task_cfg.covariate_embedding_dim)
                for name in ordered_covariates(task_cfg.covariates)
            }
        )
        builders = {
            1: ClassificationHead._build_single_layer_mlp,
            2: ClassificationHead._build_two_layer_mlp,
            3: ClassificationHead._build_three_layer_mlp,
        }
        self.head = builders[cfg.model.head.kwargs["num_layers"]](
            len(self.embeddings) * task_cfg.covariate_embedding_dim,
            cfg.model.head.hidden_dim,
            cfg.finetune.task.output_dim,
            cfg.model.head.dropout,
            type(_activation(cfg.model.head.act)),
        )

    def forward(self, metadata: Mapping[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.head(embed_covariates(self.embeddings, metadata, self.covariate_normalization, device))


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
