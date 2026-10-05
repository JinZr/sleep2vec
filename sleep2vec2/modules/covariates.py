"""Covariate encoding shared by the survival and multilabel downstream heads.

Covariates are consumed in the fixed ``SUPPORTED_COVARIATES`` order, whatever order a recipe lists them in, so the
concatenated feature layout does not depend on the spelling of the list. ``age`` and ``bmi`` are continuous and
standardized with the frozen training statistics in ``covariate_normalization``; without an ``age`` entry, age keeps
its historical ``age / 100`` scaling. ``sex`` and ``bmi_missing`` are 0/1 indicators.
"""

import typing as t

import torch
import torch.nn as nn

from sleep2vec2.config import SUPPORTED_COVARIATES

CONTINUOUS_COVARIATES = ("age", "bmi")


def ordered_covariates(names: t.Iterable[str]) -> t.Tuple[str, ...]:
    selected = set(names)
    return tuple(name for name in SUPPORTED_COVARIATES if name in selected)


def build_covariate_embedding(name: str, embedding_dim: int) -> nn.Module:
    """Zero-initialized so a newly attached covariate starts as a no-op on the head input."""
    if name in CONTINUOUS_COVARIATES:
        layer: nn.Module = nn.Linear(1, embedding_dim)
        nn.init.zeros_(layer.bias)
    else:
        layer = nn.Embedding(2, embedding_dim)
    nn.init.zeros_(layer.weight)
    return layer


def _covariate_column(
    name: str,
    metadata: t.Mapping[str, torch.Tensor],
    normalization: t.Mapping[str, t.Mapping[str, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in metadata:
        raise ValueError(f"Survival covariate '{name}' requires batch metadata {name}.")
    value = metadata[name].to(device=device)
    if name not in CONTINUOUS_COVARIATES:
        if ((value != 0) & (value != 1)).any():
            raise ValueError(f"Survival covariate '{name}' must be 0 or 1 for every sample.")
        return value.to(dtype=dtype)
    value = value.to(dtype=dtype)
    if name == "age" and (value < 0).any():
        raise ValueError("Survival covariate 'age' is missing for at least one sample.")
    if name == "bmi" and not torch.isfinite(value).all():
        raise ValueError("Survival covariate 'bmi' requires finite imputed values.")
    if name == "age" and "age" not in normalization:
        return value / 100.0
    stats = normalization[name]
    return (value - stats["mean"]) / stats["std"]


def covariate_values(
    names: t.Iterable[str],
    metadata: t.Mapping[str, torch.Tensor],
    normalization: t.Mapping[str, t.Mapping[str, float]],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Scaled covariate values as a ``[B, k]`` tensor on the reference device and dtype."""
    columns = [
        _covariate_column(name, metadata, normalization, reference.device, reference.dtype).view(-1, 1)
        for name in ordered_covariates(names)
    ]
    return torch.cat(columns, dim=-1)


def embed_covariates(
    embeddings: t.Mapping[str, nn.Module],
    metadata: t.Mapping[str, torch.Tensor],
    normalization: t.Mapping[str, t.Mapping[str, float]],
    device: torch.device,
) -> torch.Tensor:
    """Concatenated covariate embeddings as a ``[B, k * embedding_dim]`` tensor."""
    features = []
    for name in ordered_covariates(embeddings):
        layer = embeddings[name]
        column = _covariate_column(name, metadata, normalization, device, layer.weight.dtype)
        features.append(layer(column.view(-1, 1)) if name in CONTINUOUS_COVARIATES else layer(column.long()))
    return torch.cat(features, dim=-1)
