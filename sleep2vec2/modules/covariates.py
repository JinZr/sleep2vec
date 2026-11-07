"""Covariate encoding shared by the survival and multilabel downstream heads.

Covariates are consumed in the fixed ``SUPPORTED_COVARIATES`` order, whatever order a recipe lists them in, so the
concatenated feature layout does not depend on the recipe order. Continuous inputs are already imputed in their
original units and standardized with the supplied training-fold statistics. Without age statistics, age keeps
its historical ``age / 100`` scaling. Sex and missingness flags use two rows; multiclass inputs use one row per ID.
"""

import typing as t

import torch
import torch.nn as nn

from sleep2vec2.config import CONTINUOUS_COVARIATES, SUPPORTED_COVARIATES


def ordered_covariates(names: t.Iterable[str]) -> t.Tuple[str, ...]:
    selected = set(names)
    return tuple(name for name in SUPPORTED_COVARIATES if name in selected)


def build_covariate_embedding(name: str, embedding_dim: int, num_categories: int = 2) -> nn.Module:
    """Zero-initialized so a newly attached covariate starts as a no-op on the head input."""
    if name in CONTINUOUS_COVARIATES:
        layer: nn.Module = nn.Linear(1, embedding_dim)
        nn.init.zeros_(layer.bias)
    else:
        layer = nn.Embedding(num_categories, embedding_dim)
    nn.init.zeros_(layer.weight)
    return layer


def _covariate_column(
    name: str,
    metadata: t.Mapping[str, torch.Tensor],
    normalization: t.Mapping[str, t.Mapping[str, float]],
    device: torch.device,
    dtype: torch.dtype,
    num_categories: int = 2,
) -> torch.Tensor:
    if name not in metadata:
        raise ValueError(f"Survival covariate '{name}' requires batch metadata {name}.")
    value = metadata[name].to(device=device)
    if name not in CONTINUOUS_COVARIATES:
        ids = value.long()
        if ((value != ids) | (ids < 0) | (ids >= num_categories)).any():
            domain = "0 or 1" if num_categories == 2 else f"an integer ID in [0, {num_categories})"
            raise ValueError(f"Survival covariate '{name}' must be {domain} for every sample.")
        return ids
    value = value.to(dtype=dtype)
    if name == "age" and (value < 0).any():
        raise ValueError("Survival covariate 'age' is missing for at least one sample.")
    if not torch.isfinite(value).all():
        raise ValueError(f"Survival covariate '{name}' requires finite imputed values.")
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
    # Standardize in float32: a reduced-precision reference would round raw bmi before the frozen mean is removed.
    columns = [
        _covariate_column(name, metadata, normalization, reference.device, torch.float32).view(-1, 1)
        for name in ordered_covariates(names)
    ]
    return torch.cat(columns, dim=-1).to(dtype=reference.dtype)


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
        num_categories = layer.num_embeddings if isinstance(layer, nn.Embedding) else 2
        column = _covariate_column(name, metadata, normalization, device, layer.weight.dtype, num_categories)
        features.append(layer(column.view(-1, 1)) if name in CONTINUOUS_COVARIATES else layer(column))
    return torch.cat(features, dim=-1)
