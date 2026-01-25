from __future__ import annotations

import hashlib
import typing as t

import torch
from torch import Tensor, nn


def stable_hash_to_bucket(values: t.Sequence[str], num_buckets: int) -> Tensor:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be > 0")
    buckets = []
    for v in values:
        s = "nan" if v is None else str(v)
        digest = hashlib.md5(s.encode("utf-8")).hexdigest()
        buckets.append(int(digest, 16) % num_buckets)
    return torch.tensor(buckets, dtype=torch.long)


class MetadataContextEncoder(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        *,
        num_source_buckets: int = 128,
        num_subject_buckets: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.num_source_buckets = int(num_source_buckets)
        self.num_subject_buckets = int(num_subject_buckets)

        self.age_proj = nn.Linear(1, self.meta_dim)
        self.sex_embed = nn.Embedding(3, self.meta_dim)
        self.source_embed = nn.Embedding(self.num_source_buckets, self.meta_dim)
        self.subject_embed = nn.Embedding(self.num_subject_buckets, self.meta_dim)
        self.age_missing = nn.Parameter(torch.zeros(self.meta_dim))

        self.fuse = nn.Sequential(
            nn.Linear(self.meta_dim * 4, self.meta_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        *,
        age: Tensor,
        sex: Tensor,
        source_ids: Tensor,
        subject_ids: Tensor,
    ) -> Tensor:
        device = self.age_missing.device

        age = age.to(device=device, dtype=torch.float32)
        sex = sex.to(device=device, dtype=torch.long)
        source_ids = source_ids.to(device=device, dtype=torch.long)
        subject_ids = subject_ids.to(device=device, dtype=torch.long)

        age_finite = torch.isfinite(age)
        age_missing = (~age_finite) | (age < 0)
        if not age_finite.all():
            age = age.clone()
            age[~age_finite] = 0.0
        age_norm = age / 100.0
        age_feat = self.age_proj(age_norm.unsqueeze(-1))
        if age_missing.any():
            age_feat = age_feat.clone()
            age_feat[age_missing] = self.age_missing.to(dtype=age_feat.dtype, device=age_feat.device)

        sex_idx = sex.clone()
        sex_idx[sex_idx < 0] = 2
        sex_idx[sex_idx > 1] = 2
        sex_feat = self.sex_embed(sex_idx)

        source_feat = self.source_embed(source_ids)
        subject_feat = self.subject_embed(subject_ids)

        fused = torch.cat([age_feat, sex_feat, source_feat, subject_feat], dim=-1)
        return self.fuse(fused)


__all__ = ["MetadataContextEncoder", "stable_hash_to_bucket"]
