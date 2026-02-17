from __future__ import annotations

"""Router context encoder for metadata-aware MoE routing."""

import typing as t
import zlib

import torch
from torch import nn


class RouterContextEncoder(nn.Module):
    """Encodes metadata + modality id into a dense router context vector."""

    def __init__(
        self,
        *,
        ctx_dim: int,
        source_vocab_size: int,
        num_modalities: int,
        use_age: bool = True,
        use_sex: bool = True,
        use_source: bool = True,
        use_modality: bool = True,
    ) -> None:
        super().__init__()
        if ctx_dim <= 0:
            raise ValueError(f"ctx_dim must be positive, got {ctx_dim}")
        if source_vocab_size <= 0:
            raise ValueError(f"source_vocab_size must be positive, got {source_vocab_size}")
        if num_modalities <= 0:
            raise ValueError(f"num_modalities must be positive, got {num_modalities}")

        self.ctx_dim = ctx_dim
        self.source_vocab_size = source_vocab_size
        self.num_modalities = num_modalities
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_source = use_source
        self.use_modality = use_modality

        self.modality_emb = nn.Embedding(num_modalities, ctx_dim) if use_modality else None
        self.sex_emb = nn.Embedding(3, ctx_dim) if use_sex else None
        self.source_emb = nn.Embedding(source_vocab_size, ctx_dim) if use_source else None
        self.age_mlp = (
            nn.Sequential(
                nn.Linear(1, ctx_dim),
                nn.GELU(),
                nn.Linear(ctx_dim, ctx_dim),
            )
            if use_age
            else None
        )
        self.layer_norm = nn.LayerNorm(ctx_dim)

    def _infer_batch_size(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        modality_ids: int | torch.Tensor | None,
        batch_size: int | None,
    ) -> int:
        if batch_size is not None:
            return int(batch_size)
        if torch.is_tensor(modality_ids):
            if modality_ids.dim() == 0:
                return 1
            return int(modality_ids.shape[0])
        if metadata:
            for value in metadata.values():
                if torch.is_tensor(value):
                    if value.dim() == 0:
                        continue
                    return int(value.shape[0])
                if isinstance(value, (list, tuple)):
                    return len(value)
        raise ValueError("Unable to infer batch_size for RouterContextEncoder.")

    @staticmethod
    def _align_vector(
        value: torch.Tensor,
        batch_size: int,
        *,
        name: str,
    ) -> torch.Tensor:
        if value.dim() == 0:
            return value.reshape(1).expand(batch_size)
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size)
        raise ValueError(f"{name} has incompatible first dimension: {tuple(value.shape)} vs batch_size={batch_size}")

    def encode_source_ids(
        self,
        source_values: t.Sequence[t.Any] | torch.Tensor | None,
        *,
        device: torch.device,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if source_values is None:
            if batch_size is None:
                raise ValueError("batch_size is required when source_values is None.")
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if torch.is_tensor(source_values):
            ids = source_values.to(device=device, dtype=torch.long)
            if batch_size is not None:
                ids = self._align_vector(ids, batch_size, name="source_values")
            return ids.clamp(min=0, max=self.source_vocab_size - 1)

        encoded: list[int] = []
        for value in source_values:
            if value is None:
                encoded.append(0)
                continue
            if isinstance(value, str):
                stripped = value.strip().lower()
                if stripped in {"", "nan", "none"}:
                    encoded.append(0)
                    continue
            try:
                numeric = int(float(value))
            except (TypeError, ValueError):
                if self.source_vocab_size == 1:
                    encoded.append(0)
                else:
                    hashed = zlib.crc32(str(value).encode("utf-8"))
                    encoded.append(int(hashed % (self.source_vocab_size - 1)) + 1)
                continue

            if 0 <= numeric < self.source_vocab_size:
                encoded.append(numeric)
            else:
                encoded.append(0)

        ids = torch.tensor(encoded, dtype=torch.long, device=device)
        if batch_size is not None:
            ids = self._align_vector(ids, batch_size, name="source_values")
        return ids

    def _encode_modality(
        self,
        modality_ids: int | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.modality_emb is None:
            return None
        if modality_ids is None:
            modality_ids = 0
        if torch.is_tensor(modality_ids):
            ids = modality_ids.to(device=device, dtype=torch.long)
            ids = self._align_vector(ids, batch_size, name="modality_ids")
        else:
            ids = torch.full((batch_size,), int(modality_ids), dtype=torch.long, device=device)
        ids = ids.clamp(min=0, max=self.num_modalities - 1)
        return self.modality_emb(ids)

    def _encode_age(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.age_mlp is None:
            return None
        age_raw = None if metadata is None else metadata.get("age")
        if age_raw is None:
            return None
        age = torch.as_tensor(age_raw, dtype=torch.float32, device=device)
        age = self._align_vector(age, batch_size, name="metadata.age")
        valid = age >= 0
        age_norm = torch.where(valid, age / 100.0, torch.zeros_like(age))
        encoded = self.age_mlp(age_norm.unsqueeze(-1))
        return encoded * valid.to(dtype=encoded.dtype).unsqueeze(-1)

    def _encode_sex(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.sex_emb is None:
            return None
        sex_raw = None if metadata is None else metadata.get("sex")
        if sex_raw is None:
            return None
        sex_all = torch.as_tensor(sex_raw, dtype=torch.long, device=device)
        sex_all = self._align_vector(sex_all, batch_size, name="metadata.sex")
        unknown = sex_all < 0
        sex = sex_all.clamp(min=0, max=1)
        sex = torch.where(unknown, torch.full_like(sex, 2), sex)
        return self.sex_emb(sex)

    def _encode_source(
        self,
        metadata: t.Mapping[str, t.Any] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.source_emb is None:
            return None
        source_raw = None if metadata is None else metadata.get("source")
        ids = self.encode_source_ids(source_raw, device=device, batch_size=batch_size)
        return self.source_emb(ids)

    def forward(
        self,
        *,
        metadata: t.Mapping[str, t.Any] | None,
        modality_ids: int | torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        batch_size = self._infer_batch_size(metadata, modality_ids, batch_size)
        if device is None:
            device = self.layer_norm.weight.device
            if metadata:
                for value in metadata.values():
                    if torch.is_tensor(value):
                        device = value.device
                        break
            if torch.is_tensor(modality_ids):
                device = modality_ids.device

        context = torch.zeros((batch_size, self.ctx_dim), dtype=torch.float32, device=device)
        used_any = False

        part = self._encode_modality(modality_ids, batch_size, device)
        if part is not None:
            context = context + part
            used_any = True

        part = self._encode_age(metadata, batch_size, device)
        if part is not None:
            context = context + part
            used_any = True

        part = self._encode_sex(metadata, batch_size, device)
        if part is not None:
            context = context + part
            used_any = True

        part = self._encode_source(metadata, batch_size, device)
        if part is not None:
            context = context + part
            used_any = True

        if not used_any:
            return context
        return self.layer_norm(context)
