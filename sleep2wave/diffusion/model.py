from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch
import torch.nn as nn

from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS, validate_modality_sequence
from sleep2wave.diffusion.task_masks import TaskAttentionMask, TokenLayout, build_directional_task_attention_mask
from sleep2wave.diffusion.tasks import AUX_MODALITY, GenerationTask, is_restoration_task, validate_generation_task
from sleep2wave.generative.config import DiffusionConfig

DEFAULT_LATENT_FRAMES_PER_EPOCH = {
    "high_frequency": 60,
    "low_frequency": 30,
}


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
        latent_frames_per_epoch: t.Mapping[str, int] | None = None,
        patches_per_epoch: int = 6,
        modalities: t.Sequence[str] = CANONICAL_MODALITIES,
        use_diffusion_step_embedding: bool = True,
        use_modality_embedding: bool = True,
        use_epoch_position_embedding: bool = True,
        use_channel_position_embedding: bool = True,
        use_patch_position_embedding: bool = True,
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
        if patches_per_epoch <= 0:
            raise ValueError("patches_per_epoch must be positive.")

        self.modalities = tuple(validate_modality_sequence(list(modalities), allow_aliases=False))
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.diffusion_steps = int(diffusion_steps)
        self.latent_frames_per_epoch = dict(latent_frames_per_epoch or DEFAULT_LATENT_FRAMES_PER_EPOCH)
        self.layout = TokenLayout(
            modalities=self.modalities,
            context_epochs=context_epochs,
            patches_per_epoch=patches_per_epoch,
            include_aux=include_aux,
        )
        self.use_diffusion_step_embedding = bool(use_diffusion_step_embedding)
        self.use_modality_embedding = bool(use_modality_embedding)
        self.use_epoch_position_embedding = bool(use_epoch_position_embedding)
        self.use_channel_position_embedding = bool(use_channel_position_embedding)
        self.use_patch_position_embedding = bool(use_patch_position_embedding)
        self.use_sleep_night_position_embedding = bool(use_sleep_night_position_embedding)
        self.use_availability_embedding = bool(use_availability_embedding)
        self.use_quality_embedding = bool(use_quality_embedding)

        self.input_projections = nn.ModuleDict(
            {modality: nn.Linear(self._patch_payload_dim(modality), self.hidden_size) for modality in self.modalities}
        )
        self.output_projections = nn.ModuleDict(
            {modality: nn.Linear(self.hidden_size, self._patch_payload_dim(modality)) for modality in self.modalities}
        )
        self.diffusion_step_embedding = (
            nn.Embedding(self.diffusion_steps, self.hidden_size) if self.use_diffusion_step_embedding else None
        )
        self.modality_embedding = (
            nn.Embedding(len(self.layout.token_modalities), self.hidden_size) if self.use_modality_embedding else None
        )
        self.epoch_position_embedding = (
            nn.Embedding(self.layout.context_epochs, self.hidden_size) if self.use_epoch_position_embedding else None
        )
        self.channel_position_projection = (
            nn.Linear(1, self.hidden_size) if self.use_channel_position_embedding else None
        )
        self.patch_position_embedding = (
            nn.Embedding(self.layout.patches_per_epoch, self.hidden_size) if self.use_patch_position_embedding else None
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
            latent_frames_per_epoch=config.latent_frames_per_epoch,
            patches_per_epoch=config.patches_per_epoch,
            modalities=modalities,
            use_diffusion_step_embedding=config.embeddings.diffusion_step,
            use_modality_embedding=config.embeddings.modality,
            use_epoch_position_embedding=config.embeddings.epoch_position,
            use_channel_position_embedding=config.embeddings.channel_position,
            use_patch_position_embedding=config.embeddings.patch_position,
            use_sleep_night_position_embedding=config.embeddings.sleep_night_position,
            use_availability_embedding=config.embeddings.availability,
            use_quality_embedding=config.embeddings.quality,
            include_aux=config.auxiliary_restoration_token,
        )

    def _latent_frames_for_modality(self, modality: str) -> int:
        frequency_group = MODALITY_SPECS[modality].frequency_group
        if frequency_group not in self.latent_frames_per_epoch:
            raise ValueError(f"Missing latent frame count for frequency group '{frequency_group}'.")
        frames = self.latent_frames_per_epoch[frequency_group]
        if frames % self.layout.patches_per_epoch != 0:
            raise ValueError(f"latent_frames_per_epoch.{frequency_group} must be divisible by patches_per_epoch.")
        return int(frames)

    def _latent_frames_per_patch(self, modality: str) -> int:
        return self._latent_frames_for_modality(modality) // self.layout.patches_per_epoch

    def _patch_payload_dim(self, modality: str) -> int:
        return self._latent_frames_per_patch(modality) * self.latent_dim

    def _layout_for_channel_count(self, channel_count: int) -> TokenLayout:
        return TokenLayout(
            modalities=self.modalities,
            context_epochs=self.layout.context_epochs,
            channel_count=channel_count,
            patches_per_epoch=self.layout.patches_per_epoch,
            include_aux=self.layout.include_aux,
        )

    def _validate_latent(self, latent: torch.Tensor, modality: str) -> tuple[int, int, int]:
        if modality not in self.modalities:
            raise ValueError(f"Unknown diffusion modality '{modality}'.")
        if latent.dim() != 5:
            raise ValueError(f"Latent for '{modality}' must have shape [B, E, C, L, D], got {tuple(latent.shape)}.")
        batch_size, context_epochs, channels, latent_frames, latent_dim = latent.shape
        if context_epochs != self.layout.context_epochs:
            raise ValueError(
                f"Latent for '{modality}' has {context_epochs} epochs; expected {self.layout.context_epochs}."
            )
        if channels <= 0:
            raise ValueError(f"Latent for '{modality}' must include at least one channel.")
        expected_frames = self._latent_frames_for_modality(modality)
        if latent_frames != expected_frames:
            raise ValueError(f"Latent for '{modality}' has {latent_frames} frames; expected {expected_frames}.")
        if latent_dim != self.latent_dim:
            raise ValueError(f"Latent for '{modality}' has dim {latent_dim}; expected {self.latent_dim}.")
        return batch_size, context_epochs, channels

    def _validate_channel_mask(
        self,
        channel_mask: dict[str, torch.Tensor] | None,
        modality: str,
        *,
        batch_size: int,
        channels: int,
        device: torch.device,
    ) -> None:
        if channels == 1 and (channel_mask is None or modality not in channel_mask):
            return
        if channel_mask is None or modality not in channel_mask:
            raise ValueError(f"channel_mask['{modality}'] is required for multi-channel latents.")
        mask = torch.as_tensor(channel_mask[modality], dtype=torch.bool, device=device)
        expected = (batch_size, self.layout.context_epochs, channels)
        if mask.shape != expected:
            raise ValueError(f"channel_mask['{modality}'] must have shape {expected}, got {tuple(mask.shape)}.")

    def _infer_batch_shape(
        self,
        condition_latents: dict[str, torch.Tensor],
        noisy_target_latents: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None,
        device: torch.device,
    ) -> tuple[int, int, dict[str, int]]:
        batch_size: int | None = None
        max_channels = 1
        channels_by_modality: dict[str, int] = {}
        for mapping in (condition_latents, noisy_target_latents):
            for modality, latent in mapping.items():
                current_batch_size, _context, channels = self._validate_latent(latent, modality)
                if batch_size is None:
                    batch_size = current_batch_size
                elif current_batch_size != batch_size:
                    raise ValueError(
                        f"Latent for '{modality}' has batch size {current_batch_size}; expected {batch_size}."
                    )
                max_channels = max(max_channels, channels)
                channels_by_modality[modality] = max(channels_by_modality.get(modality, 0), channels)
        if batch_size is None:
            raise ValueError("At least one condition or target latent is required.")
        for modality, channels in channels_by_modality.items():
            self._validate_channel_mask(
                channel_mask,
                modality,
                batch_size=batch_size,
                channels=channels,
                device=device,
            )
        return batch_size, max_channels, channels_by_modality

    def _effective_channel_mask(
        self,
        channel_mask: dict[str, torch.Tensor] | None,
        channels_by_modality: dict[str, int],
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        effective: dict[str, torch.Tensor] = {}
        for modality, channels in channels_by_modality.items():
            if channel_mask is not None and modality in channel_mask:
                effective[modality] = torch.as_tensor(channel_mask[modality], dtype=torch.bool, device=device)
            else:
                effective[modality] = torch.ones(
                    batch_size,
                    self.layout.context_epochs,
                    channels,
                    dtype=torch.bool,
                    device=device,
                )
        return effective

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

    def _first_projection_weight(self) -> torch.Tensor:
        return next(iter(self.input_projections.values())).weight

    def _empty_hidden(self, batch_size: int, layout: TokenLayout, device: torch.device) -> torch.Tensor:
        weight = self._first_projection_weight()
        return torch.zeros(
            (batch_size, layout.token_count, self.hidden_size),
            dtype=weight.dtype,
            device=device,
        )

    def _patchify(self, latent: torch.Tensor, modality: str, layout: TokenLayout) -> torch.Tensor:
        self._validate_latent(latent, modality)
        batch_size, context_epochs, channels, latent_frames, latent_dim = latent.shape
        frames_per_patch = self._latent_frames_per_patch(modality)
        return latent.reshape(
            batch_size,
            context_epochs,
            channels,
            layout.patches_per_epoch,
            frames_per_patch,
            latent_dim,
        ).reshape(batch_size, context_epochs, channels, layout.patches_per_epoch, frames_per_patch * latent_dim)

    def _unpatchify(
        self,
        patches: torch.Tensor,
        modality: str,
        layout: TokenLayout,
        *,
        channel_count: int,
    ) -> torch.Tensor:
        batch_size, context_epochs, channels, patch_count, payload_dim = patches.shape
        if context_epochs != layout.context_epochs:
            raise ValueError(f"Patch output for '{modality}' has {context_epochs} epochs.")
        if channels != channel_count:
            raise ValueError(f"Patch output for '{modality}' has {channels} channels; expected {channel_count}.")
        if patch_count != layout.patches_per_epoch:
            raise ValueError(f"Patch output for '{modality}' has {patch_count} patches.")
        expected_payload_dim = self._patch_payload_dim(modality)
        if payload_dim != expected_payload_dim:
            raise ValueError(f"Patch output for '{modality}' has dim {payload_dim}; expected {expected_payload_dim}.")
        frames_per_patch = self._latent_frames_per_patch(modality)
        return patches.reshape(
            batch_size,
            context_epochs,
            channel_count,
            layout.patches_per_epoch * frames_per_patch,
            self.latent_dim,
        )

    def _write_hidden_tokens(
        self,
        hidden: torch.Tensor,
        *,
        layout: TokenLayout,
        token_modality: str,
        source_modality: str,
        values: torch.Tensor,
    ) -> None:
        batch_size, _context_epochs, channels = self._validate_latent(values, source_modality)
        if batch_size != hidden.shape[0]:
            raise ValueError(f"Latent for '{source_modality}' has batch size {batch_size}; expected {hidden.shape[0]}.")
        if channels > layout.channel_count:
            raise ValueError(f"Latent for '{source_modality}' has too many channels for the token layout.")
        projected = self.input_projections[source_modality](self._patchify(values, source_modality, layout))
        for epoch in range(layout.context_epochs):
            for channel in range(channels):
                for patch in range(layout.patches_per_epoch):
                    hidden[:, layout.token_index(token_modality, epoch, channel, patch), :] = projected[
                        :, epoch, channel, patch, :
                    ]

    def _build_token_hidden(
        self,
        task: GenerationTask,
        condition_latents: dict[str, torch.Tensor],
        noisy_target_latents: dict[str, torch.Tensor],
        layout: TokenLayout,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        hidden = self._empty_hidden(batch_size, layout, device)
        for modality in task.condition_modalities:
            if modality not in condition_latents:
                raise ValueError(f"Missing condition latent for modality '{modality}'.")
            self._write_hidden_tokens(
                hidden,
                layout=layout,
                token_modality=modality,
                source_modality=modality,
                values=condition_latents[modality].to(device=device),
            )

        if is_restoration_task(task):
            target = task.target_modalities[0]
            if target not in noisy_target_latents:
                raise ValueError(f"Missing noisy target latent for modality '{target}'.")
            self._write_hidden_tokens(
                hidden,
                layout=layout,
                token_modality=AUX_MODALITY,
                source_modality=target,
                values=noisy_target_latents[target].to(device=device),
            )
        else:
            for modality in task.target_modalities:
                if modality not in noisy_target_latents:
                    raise ValueError(f"Missing noisy target latent for modality '{modality}'.")
                self._write_hidden_tokens(
                    hidden,
                    layout=layout,
                    token_modality=modality,
                    source_modality=modality,
                    values=noisy_target_latents[modality].to(device=device),
                )
        return hidden

    def _require_token_values(
        self,
        mapping: dict[str, torch.Tensor] | None,
        modality: str,
        layout: TokenLayout,
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
        if values.shape == (batch_size, layout.context_epochs):
            return values[:, :, None, None].expand(
                batch_size,
                layout.context_epochs,
                layout.channel_count,
                layout.patches_per_epoch,
            )
        if values.shape == (batch_size, layout.context_epochs, layout.patches_per_epoch):
            return values[:, :, None, :].expand(
                batch_size,
                layout.context_epochs,
                layout.channel_count,
                layout.patches_per_epoch,
            )
        if values.dim() == 4 and values.shape[:2] == (batch_size, layout.context_epochs):
            if values.shape[2] > layout.channel_count or values.shape[3] != layout.patches_per_epoch:
                raise ValueError(
                    f"{name}['{modality}'] must have at most {layout.channel_count} channels and "
                    f"{layout.patches_per_epoch} patches."
                )
            if values.shape[2] < layout.channel_count:
                pad_shape = list(values.shape)
                pad_shape[2] = layout.channel_count - values.shape[2]
                pad = torch.zeros(pad_shape, dtype=values.dtype, device=values.device)
                values = torch.cat([values, pad], dim=2)
            return values
        raise ValueError(
            f"{name}['{modality}'] must have shape "
            f"({batch_size}, {layout.context_epochs}), "
            f"({batch_size}, {layout.context_epochs}, {layout.patches_per_epoch}), or "
            f"({batch_size}, {layout.context_epochs}, C, {layout.patches_per_epoch}), "
            f"got {tuple(values.shape)}."
        )

    def _token_values(
        self,
        task: GenerationTask,
        mapping: dict[str, torch.Tensor] | None,
        layout: TokenLayout,
        batch_size: int,
        device: torch.device,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.zeros((batch_size, layout.token_count), dtype=dtype, device=device)
        for modality in self.modalities:
            modality_values = self._require_token_values(
                mapping,
                modality,
                layout,
                batch_size,
                device,
                name=name,
                dtype=dtype,
            )
            for epoch in range(layout.context_epochs):
                for channel in range(layout.channel_count):
                    for patch in range(layout.patches_per_epoch):
                        values[:, layout.token_index(modality, epoch, channel, patch)] = modality_values[
                            :, epoch, channel, patch
                        ]
        if layout.include_aux:
            aux_source = task.target_modalities[0] if is_restoration_task(task) else self.modalities[0]
            aux_values = self._require_token_values(
                mapping,
                aux_source,
                layout,
                batch_size,
                device,
                name=name,
                dtype=dtype,
            )
            for epoch in range(layout.context_epochs):
                for channel in range(layout.channel_count):
                    for patch in range(layout.patches_per_epoch):
                        values[:, layout.token_index(AUX_MODALITY, epoch, channel, patch)] = aux_values[
                            :, epoch, channel, patch
                        ]
        return values

    def _token_id_tensors(
        self,
        layout: TokenLayout,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_modalities = []
        epoch_positions = []
        channel_positions = []
        patch_positions = []
        for modality_index, _modality in enumerate(layout.token_modalities):
            for epoch in range(layout.context_epochs):
                for channel in range(layout.channel_count):
                    for patch in range(layout.patches_per_epoch):
                        token_modalities.append(modality_index)
                        epoch_positions.append(epoch)
                        channel_positions.append(channel)
                        patch_positions.append(patch)
        return (
            torch.tensor(token_modalities, dtype=torch.long, device=device),
            torch.tensor(epoch_positions, dtype=torch.long, device=device),
            torch.tensor(channel_positions, dtype=torch.long, device=device),
            torch.tensor(patch_positions, dtype=torch.long, device=device),
        )

    def _add_embeddings(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        task: GenerationTask,
        layout: TokenLayout,
        *,
        availability_mask: dict[str, torch.Tensor] | None,
        quality_mask: dict[str, torch.Tensor] | None,
        night_position: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        device = hidden.device
        token_modality_ids, token_epoch_ids, token_channel_ids, token_patch_ids = self._token_id_tensors(layout, device)

        if self.diffusion_step_embedding is not None:
            hidden = hidden + self.diffusion_step_embedding(timesteps)[:, None, :]
        if self.modality_embedding is not None:
            hidden = hidden + self.modality_embedding(token_modality_ids)[None, :, :]
        if self.epoch_position_embedding is not None:
            hidden = hidden + self.epoch_position_embedding(token_epoch_ids)[None, :, :]
        if self.channel_position_projection is not None:
            channel_positions = token_channel_ids.to(dtype=hidden.dtype).unsqueeze(-1)
            hidden = hidden + self.channel_position_projection(channel_positions)[None, :, :]
        if self.patch_position_embedding is not None:
            hidden = hidden + self.patch_position_embedding(token_patch_ids)[None, :, :]
        if self.sleep_night_position_projection is not None:
            if night_position is None:
                raise ValueError("night_position is required when sleep_night_position embedding is enabled.")
            night_position = torch.as_tensor(night_position, dtype=hidden.dtype, device=device)
            if night_position.shape != (batch_size, layout.context_epochs):
                raise ValueError(
                    f"night_position must have shape ({batch_size}, {layout.context_epochs}), "
                    f"got {tuple(night_position.shape)}."
                )
            token_positions = night_position[:, token_epoch_ids]
            hidden = hidden + self.sleep_night_position_projection(token_positions.unsqueeze(-1))
        if self.availability_embedding is not None:
            availability = self._token_values(
                task,
                availability_mask,
                layout,
                batch_size,
                device,
                name="availability_mask",
                dtype=torch.bool,
            )
            hidden = hidden + self.availability_embedding(availability.to(dtype=torch.long))
        if self.quality_projection is not None:
            quality = self._token_values(
                task,
                quality_mask,
                layout,
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

    def _collect_modality_prediction(
        self,
        hidden: torch.Tensor,
        *,
        layout: TokenLayout,
        token_modality: str,
        source_modality: str,
        channel_count: int,
    ) -> torch.Tensor:
        indices = layout.modality_indices(token_modality)
        projected = self.output_projections[source_modality](hidden[:, indices, :])
        batch_size = hidden.shape[0]
        patches = projected.reshape(
            batch_size,
            layout.context_epochs,
            layout.channel_count,
            layout.patches_per_epoch,
            self._patch_payload_dim(source_modality),
        )[:, :, :channel_count, :, :]
        return self._unpatchify(patches, source_modality, layout, channel_count=channel_count)

    def _collect_predictions(
        self,
        hidden: torch.Tensor,
        task: GenerationTask,
        layout: TokenLayout,
        channels_by_modality: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        if is_restoration_task(task):
            target = task.target_modalities[0]
            predictions[target] = self._collect_modality_prediction(
                hidden,
                layout=layout,
                token_modality=AUX_MODALITY,
                source_modality=target,
                channel_count=channels_by_modality[target],
            )
            return predictions

        for modality in task.target_modalities:
            predictions[modality] = self._collect_modality_prediction(
                hidden,
                layout=layout,
                token_modality=modality,
                source_modality=modality,
                channel_count=channels_by_modality[modality],
            )
        return predictions

    def forward(
        self,
        *,
        noisy_target_latents: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        task: GenerationTask,
        condition_latents: dict[str, torch.Tensor],
        availability_mask: dict[str, torch.Tensor] | None = None,
        condition_availability_mask: dict[str, torch.Tensor] | None = None,
        channel_mask: dict[str, torch.Tensor] | None = None,
        quality_mask: dict[str, torch.Tensor] | None = None,
        night_position: torch.Tensor | None = None,
    ) -> Sleep2WaveDiffusionOutput:
        task = validate_generation_task(task)
        if condition_latents:
            device = next(iter(condition_latents.values())).device
        else:
            device = next(iter(noisy_target_latents.values())).device
        batch_size, channel_count, channels_by_modality = self._infer_batch_shape(
            condition_latents,
            noisy_target_latents,
            channel_mask,
            device,
        )
        effective_channel_mask = self._effective_channel_mask(
            channel_mask,
            channels_by_modality,
            batch_size=batch_size,
            device=device,
        )
        layout = self._layout_for_channel_count(channel_count)
        timesteps = self._validate_timesteps(timesteps, batch_size, device)
        hidden = self._build_token_hidden(
            task,
            condition_latents,
            noisy_target_latents,
            layout,
            batch_size,
            device,
        )
        task_mask = build_directional_task_attention_mask(
            task,
            layout,
            availability_mask=availability_mask,
            condition_availability_mask=condition_availability_mask,
            channel_mask=effective_channel_mask,
            batch_size=batch_size,
        )
        hidden = self._add_embeddings(
            hidden,
            timesteps,
            task,
            layout,
            availability_mask=availability_mask,
            quality_mask=quality_mask,
            night_position=night_position,
        )
        active_tokens = task_mask.active_tokens.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        hidden = hidden * active_tokens
        attention_blocked = self._mask_for_attention(task_mask).to(device=hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, attention_blocked)
            hidden = hidden * active_tokens
        hidden = self.final_norm(hidden)
        return Sleep2WaveDiffusionOutput(
            predicted_noise=self._collect_predictions(hidden, task, layout, channels_by_modality),
            task_mask=task_mask,
        )


__all__ = [
    "Sleep2WaveDiffusionOutput",
    "Sleep2WaveDiffusionTransformer",
]
