from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch
import torch.nn as nn

from sleep2wave.data.modalities import CANONICAL_MODALITIES, validate_modality_sequence
from sleep2wave.diffusion.task_masks import TaskAttentionMask, TokenLayout, build_directional_task_attention_mask
from sleep2wave.diffusion.tasks import AUX_MODALITY, GenerationTask, is_restoration_task, validate_generation_task
from sleep2wave.generative.config import DiffusionConfig


@dataclass
class Sleep2WaveDiffusionOutput:
    predicted_noise: dict[str, torch.Tensor]
    task_mask: TaskAttentionMask


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        attn_mask = blocked.repeat_interleave(self.num_heads, dim=0)
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class Sleep2WaveDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        diffusion_steps: int,
        context_epochs: int,
        modalities: t.Sequence[str] = CANONICAL_MODALITIES,
        use_diffusion_step_embedding: bool = True,
        use_modality_embedding: bool = True,
        use_epoch_position_embedding: bool = True,
        use_sleep_night_position_embedding: bool = True,
        use_availability_embedding: bool = True,
        use_quality_embedding: bool = True,
        include_aux: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")

        self.modalities = tuple(validate_modality_sequence(list(modalities), allow_aliases=False))
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.diffusion_steps = int(diffusion_steps)
        self.layout = TokenLayout(
            modalities=self.modalities,
            context_epochs=context_epochs,
            include_aux=include_aux,
        )
        self.use_diffusion_step_embedding = bool(use_diffusion_step_embedding)
        self.use_modality_embedding = bool(use_modality_embedding)
        self.use_epoch_position_embedding = bool(use_epoch_position_embedding)
        self.use_sleep_night_position_embedding = bool(use_sleep_night_position_embedding)
        self.use_availability_embedding = bool(use_availability_embedding)
        self.use_quality_embedding = bool(use_quality_embedding)

        self.input_projection = nn.Linear(self.latent_dim, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.latent_dim)
        self.diffusion_step_embedding = (
            nn.Embedding(self.diffusion_steps, self.hidden_size) if self.use_diffusion_step_embedding else None
        )
        self.modality_embedding = (
            nn.Embedding(len(self.layout.token_modalities), self.hidden_size) if self.use_modality_embedding else None
        )
        self.epoch_position_embedding = (
            nn.Embedding(self.layout.context_epochs, self.hidden_size) if self.use_epoch_position_embedding else None
        )
        self.sleep_night_position_projection = (
            nn.Linear(1, self.hidden_size) if self.use_sleep_night_position_embedding else None
        )
        self.availability_embedding = nn.Embedding(2, self.hidden_size) if self.use_availability_embedding else None
        self.quality_projection = nn.Linear(1, self.hidden_size) if self.use_quality_embedding else None
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)

        token_modalities = []
        epoch_positions = []
        for modality_index, _modality in enumerate(self.layout.token_modalities):
            token_modalities.extend([modality_index] * self.layout.context_epochs)
            epoch_positions.extend(range(self.layout.context_epochs))
        self.register_buffer("token_modality_ids", torch.tensor(token_modalities, dtype=torch.long), persistent=False)
        self.register_buffer("token_epoch_ids", torch.tensor(epoch_positions, dtype=torch.long), persistent=False)

    @classmethod
    def from_config(
        cls,
        config: DiffusionConfig,
        *,
        modalities: t.Sequence[str] = CANONICAL_MODALITIES,
    ) -> Sleep2WaveDiffusionTransformer:
        return cls(
            latent_dim=config.latent_dim,
            hidden_size=config.transformer.hidden_size,
            num_layers=config.transformer.num_layers,
            num_heads=config.transformer.num_heads,
            mlp_ratio=config.transformer.mlp_ratio,
            diffusion_steps=config.diffusion_steps,
            context_epochs=config.context_epochs,
            modalities=modalities,
            use_diffusion_step_embedding=config.embeddings.diffusion_step,
            use_modality_embedding=config.embeddings.modality,
            use_epoch_position_embedding=config.embeddings.epoch_position,
            use_sleep_night_position_embedding=config.embeddings.sleep_night_position,
            use_availability_embedding=config.embeddings.availability,
            use_quality_embedding=config.embeddings.quality,
            include_aux=config.auxiliary_restoration_token,
        )

    def _validate_latent(self, latent: torch.Tensor, modality: str) -> tuple[int, int]:
        if latent.dim() != 3:
            raise ValueError(f"Latent for '{modality}' must have shape [B, E, D], got {tuple(latent.shape)}.")
        batch_size, context_epochs, latent_dim = latent.shape
        if context_epochs != self.layout.context_epochs:
            raise ValueError(
                f"Latent for '{modality}' has {context_epochs} epochs; " f"expected {self.layout.context_epochs}."
            )
        if latent_dim != self.latent_dim:
            raise ValueError(f"Latent for '{modality}' has dim {latent_dim}; expected {self.latent_dim}.")
        return batch_size, context_epochs

    def _infer_batch_size(
        self,
        condition_latents: dict[str, torch.Tensor],
        noisy_target_latents: dict[str, torch.Tensor],
    ) -> int:
        for mapping in (condition_latents, noisy_target_latents):
            for modality, latent in mapping.items():
                batch_size, _context = self._validate_latent(latent, modality)
                return batch_size
        raise ValueError("At least one condition or target latent is required.")

    def _validate_timesteps(self, timesteps: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        timesteps = torch.as_tensor(timesteps, device=device)
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(batch_size)
        if timesteps.shape != (batch_size,):
            raise ValueError(f"timesteps must have shape ({batch_size},), got {tuple(timesteps.shape)}.")
        if timesteps.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            timesteps = timesteps.to(dtype=torch.long)
        if ((timesteps < 0) | (timesteps >= self.diffusion_steps)).any():
            raise ValueError(f"timesteps must be in [0, {self.diffusion_steps}).")
        return timesteps.to(dtype=torch.long)

    def _empty_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.layout.token_count, self.latent_dim),
            dtype=self.input_projection.weight.dtype,
            device=device,
        )

    def _write_tokens(self, tokens: torch.Tensor, modality: str, values: torch.Tensor) -> None:
        batch_size, _context_epochs = self._validate_latent(values, modality)
        if batch_size != tokens.shape[0]:
            raise ValueError(f"Latent for '{modality}' has batch size {batch_size}; expected {tokens.shape[0]}.")
        for epoch in range(self.layout.context_epochs):
            tokens[:, self.layout.token_index(modality, epoch), :] = values[:, epoch, :]

    def _build_token_latents(
        self,
        task: GenerationTask,
        condition_latents: dict[str, torch.Tensor],
        noisy_target_latents: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens = self._empty_tokens(batch_size, device)
        for modality in task.condition_modalities:
            if modality not in condition_latents:
                raise ValueError(f"Missing condition latent for modality '{modality}'.")
            self._write_tokens(tokens, modality, condition_latents[modality].to(device=device))

        if is_restoration_task(task):
            target = task.target_modalities[0]
            if target not in noisy_target_latents:
                raise ValueError(f"Missing noisy target latent for modality '{target}'.")
            self._write_tokens(tokens, AUX_MODALITY, noisy_target_latents[target].to(device=device))
        else:
            for modality in task.target_modalities:
                if modality not in noisy_target_latents:
                    raise ValueError(f"Missing noisy target latent for modality '{modality}'.")
                self._write_tokens(tokens, modality, noisy_target_latents[modality].to(device=device))
        return tokens

    def _require_epoch_values(
        self,
        mapping: dict[str, torch.Tensor] | None,
        modality: str,
        batch_size: int,
        device: torch.device,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mapping is None or modality not in mapping:
            raise ValueError(f"{name} is required for modality '{modality}'.")
        values = torch.as_tensor(mapping[modality], dtype=dtype, device=device)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.shape != (batch_size, self.layout.context_epochs):
            raise ValueError(
                f"{name}['{modality}'] must have shape "
                f"({batch_size}, {self.layout.context_epochs}), got {tuple(values.shape)}."
            )
        return values

    def _token_epoch_values(
        self,
        task: GenerationTask,
        mapping: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.zeros((batch_size, self.layout.token_count), dtype=dtype, device=device)
        for modality in self.modalities:
            modality_values = self._require_epoch_values(
                mapping,
                modality,
                batch_size,
                device,
                name=name,
                dtype=dtype,
            )
            for epoch in range(self.layout.context_epochs):
                values[:, self.layout.token_index(modality, epoch)] = modality_values[:, epoch]
        if self.layout.include_aux:
            aux_source = task.target_modalities[0] if is_restoration_task(task) else self.modalities[0]
            aux_values = self._require_epoch_values(
                mapping,
                aux_source,
                batch_size,
                device,
                name=name,
                dtype=dtype,
            )
            for epoch in range(self.layout.context_epochs):
                values[:, self.layout.token_index(AUX_MODALITY, epoch)] = aux_values[:, epoch]
        return values

    def _add_embeddings(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        task: GenerationTask,
        *,
        availability_mask: dict[str, torch.Tensor] | None,
        quality_mask: dict[str, torch.Tensor] | None,
        night_position: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        device = tokens.device
        hidden = self.input_projection(tokens)

        if self.diffusion_step_embedding is not None:
            hidden = hidden + self.diffusion_step_embedding(timesteps)[:, None, :]
        if self.modality_embedding is not None:
            hidden = hidden + self.modality_embedding(self.token_modality_ids.to(device))[None, :, :]
        if self.epoch_position_embedding is not None:
            hidden = hidden + self.epoch_position_embedding(self.token_epoch_ids.to(device))[None, :, :]
        if self.sleep_night_position_projection is not None:
            if night_position is None:
                raise ValueError("night_position is required when sleep_night_position embedding is enabled.")
            night_position = torch.as_tensor(night_position, dtype=hidden.dtype, device=device)
            if night_position.shape != (batch_size, self.layout.context_epochs):
                raise ValueError(
                    f"night_position must have shape ({batch_size}, {self.layout.context_epochs}), "
                    f"got {tuple(night_position.shape)}."
                )
            token_positions = night_position[:, self.token_epoch_ids.to(device)]
            hidden = hidden + self.sleep_night_position_projection(token_positions.unsqueeze(-1))
        if self.availability_embedding is not None:
            availability = self._token_epoch_values(
                task,
                availability_mask,
                batch_size,
                device,
                name="availability_mask",
                dtype=torch.bool,
            )
            hidden = hidden + self.availability_embedding(availability.to(dtype=torch.long))
        if self.quality_projection is not None:
            quality = self._token_epoch_values(
                task,
                quality_mask,
                batch_size,
                device,
                name="quality_mask",
                dtype=hidden.dtype,
            )
            hidden = hidden + self.quality_projection(quality.unsqueeze(-1))
        return hidden

    def _mask_for_attention(self, task_mask: TaskAttentionMask) -> torch.Tensor:
        blocked = task_mask.blocked.clone()
        inactive = ~task_mask.active_tokens
        token_count = blocked.shape[-1]
        diagonal = torch.eye(token_count, dtype=torch.bool, device=blocked.device).unsqueeze(0)
        blocked[inactive] = True
        blocked = torch.where(diagonal, torch.zeros_like(blocked), blocked)
        return blocked

    def _collect_predictions(
        self,
        hidden: torch.Tensor,
        task: GenerationTask,
    ) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        if is_restoration_task(task):
            target = task.target_modalities[0]
            indices = self.layout.modality_indices(AUX_MODALITY)
            predictions[target] = self.output_projection(hidden[:, indices, :])
            return predictions

        for modality in task.target_modalities:
            indices = self.layout.modality_indices(modality)
            predictions[modality] = self.output_projection(hidden[:, indices, :])
        return predictions

    def forward(
        self,
        *,
        noisy_target_latents: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        task: GenerationTask,
        condition_latents: dict[str, torch.Tensor],
        availability_mask: dict[str, torch.Tensor] | None = None,
        quality_mask: dict[str, torch.Tensor] | None = None,
        night_position: torch.Tensor | None = None,
    ) -> Sleep2WaveDiffusionOutput:
        task = validate_generation_task(task)
        batch_size = self._infer_batch_size(condition_latents, noisy_target_latents)
        if condition_latents:
            device = next(iter(condition_latents.values())).device
        else:
            device = next(iter(noisy_target_latents.values())).device
        timesteps = self._validate_timesteps(timesteps, batch_size, device)
        tokens = self._build_token_latents(
            task,
            condition_latents,
            noisy_target_latents,
            batch_size,
            device,
        )
        task_mask = build_directional_task_attention_mask(
            task,
            self.layout,
            availability_mask=availability_mask,
            batch_size=batch_size,
        )
        hidden = self._add_embeddings(
            tokens,
            timesteps,
            task,
            availability_mask=availability_mask,
            quality_mask=quality_mask,
            night_position=night_position,
        )
        hidden = hidden * task_mask.active_tokens.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        attention_blocked = self._mask_for_attention(task_mask).to(device=hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, attention_blocked)
            hidden = hidden * task_mask.active_tokens.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        hidden = self.final_norm(hidden)
        return Sleep2WaveDiffusionOutput(
            predicted_noise=self._collect_predictions(hidden, task),
            task_mask=task_mask,
        )


__all__ = [
    "Sleep2WaveDiffusionOutput",
    "Sleep2WaveDiffusionTransformer",
]
