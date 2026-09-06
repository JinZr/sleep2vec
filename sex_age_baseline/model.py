from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from sleep2vec.downstreams.heads.classification import ClassificationHead

from .config import BaselineConfig


class SexAgeMLP(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.features = tuple(cfg.model.features)
        self.scales = {}
        self.encoders = nn.ModuleDict()
        in_dim = 0
        for feature in self.features:
            encoding = getattr(cfg.model, feature)
            module = (
                nn.Embedding(2, encoding.embedding_dim) if feature == "sex" else nn.Linear(1, encoding.embedding_dim)
            )
            if feature != "sex":
                self.scales[feature] = encoding.scale
            if encoding.initialization == "zeros":
                for parameter in module.parameters():
                    nn.init.zeros_(parameter)
            self.encoders[feature] = module
            in_dim += encoding.embedding_dim
        builders = {
            1: ClassificationHead._build_single_layer_mlp,
            2: ClassificationHead._build_two_layer_mlp,
            3: ClassificationHead._build_three_layer_mlp,
        }
        self.head = builders[cfg.model.head.kwargs["num_layers"]](
            in_dim,
            cfg.model.head.hidden_dim,
            cfg.finetune.task.output_dim,
            cfg.model.head.dropout,
            type(_activation(cfg.model.head.act)),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        for feature in self.features:
            value = features[feature]
            value = (
                value.long().reshape(-1) if feature == "sex" else (value.float() / self.scales[feature]).reshape(-1, 1)
            )
            encoded.append(self.encoders[feature](value))
        return self.head(torch.cat(encoded, dim=-1))


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
