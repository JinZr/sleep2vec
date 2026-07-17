from __future__ import annotations

import torch
from torch import nn

from .config import BaselineConfig


class SexAgeMLP(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.age_scale = float(cfg.model.age.scale)
        self.age_projection = nn.Linear(1, cfg.model.age.embedding_dim)
        self.sex_embedding = nn.Embedding(2, cfg.model.sex.embedding_dim)
        in_dim = cfg.model.age.embedding_dim + cfg.model.sex.embedding_dim
        self.head = nn.Sequential(
            nn.Dropout(cfg.model.head.dropout),
            nn.Linear(in_dim, cfg.model.head.hidden_dim),
            _activation(cfg.model.head.activation),
            nn.Dropout(cfg.model.head.dropout),
            nn.Linear(cfg.model.head.hidden_dim, cfg.finetune.task.output_dim),
        )

    def forward(self, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        age_input = (age.float() / self.age_scale).view(-1, 1)
        age_features = self.age_projection(age_input)
        sex_features = self.sex_embedding(sex.long())
        return self.head(torch.cat([age_features, sex_features], dim=-1))


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
