from __future__ import annotations

from dataclasses import dataclass

import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES, validate_modality_sequence
from sleep2wave.diffusion.tasks import AUX_MODALITY, GenerationTask, is_restoration_task, validate_generation_task


@dataclass(frozen=True)
class TokenLayout:
    modalities: tuple[str, ...] = CANONICAL_MODALITIES
    context_epochs: int = 15
    channel_count: int = 1
    patches_per_epoch: int = 6
    include_aux: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modalities",
            tuple(validate_modality_sequence(list(self.modalities), allow_aliases=False)),
        )
        invalid_context = (
            not isinstance(self.context_epochs, int)
            or isinstance(self.context_epochs, bool)
            or self.context_epochs <= 0
        )
        if invalid_context:
            raise ValueError("context_epochs must be a positive integer.")
        invalid_channels = (
            not isinstance(self.channel_count, int) or isinstance(self.channel_count, bool) or self.channel_count <= 0
        )
        if invalid_channels:
            raise ValueError("channel_count must be a positive integer.")
        invalid_patches = (
            not isinstance(self.patches_per_epoch, int)
            or isinstance(self.patches_per_epoch, bool)
            or self.patches_per_epoch <= 0
        )
        if invalid_patches:
            raise ValueError("patches_per_epoch must be a positive integer.")
        if not isinstance(self.include_aux, bool):
            raise ValueError("include_aux must be a boolean.")

    @property
    def token_modalities(self) -> tuple[str, ...]:
        if self.include_aux:
            return (*self.modalities, AUX_MODALITY)
        return self.modalities

    @property
    def token_count(self) -> int:
        return len(self.token_modalities) * self.context_epochs * self.channel_count * self.patches_per_epoch

    @property
    def token_names(self) -> list[str]:
        return [
            f"{modality}_{epoch}_{channel}_{patch}"
            for modality in self.token_modalities
            for epoch in range(self.context_epochs)
            for channel in range(self.channel_count)
            for patch in range(self.patches_per_epoch)
        ]

    def token_index(self, modality: str, epoch: int, channel: int, patch: int) -> int:
        if modality == AUX_MODALITY:
            if not self.include_aux:
                raise ValueError("Auxiliary token is not included in this layout.")
        elif modality not in self.modalities:
            raise ValueError(f"Unknown layout modality: {modality}")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < self.context_epochs:
            raise ValueError(f"epoch must be in [0, {self.context_epochs}).")
        if not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel < self.channel_count:
            raise ValueError(f"channel must be in [0, {self.channel_count}).")
        if not isinstance(patch, int) or isinstance(patch, bool) or not 0 <= patch < self.patches_per_epoch:
            raise ValueError(f"patch must be in [0, {self.patches_per_epoch}).")
        modality_offset = (
            self.token_modalities.index(modality) * self.context_epochs * self.channel_count * self.patches_per_epoch
        )
        epoch_offset = epoch * self.channel_count * self.patches_per_epoch
        channel_offset = channel * self.patches_per_epoch
        return modality_offset + epoch_offset + channel_offset + patch

    def modality_indices(self, modality: str) -> list[int]:
        return [
            self.token_index(modality, epoch, channel, patch)
            for epoch in range(self.context_epochs)
            for channel in range(self.channel_count)
            for patch in range(self.patches_per_epoch)
        ]


@dataclass(frozen=True)
class TaskAttentionMask:
    blocked: torch.Tensor
    active_tokens: torch.Tensor
    condition_tokens: torch.Tensor
    target_tokens: torch.Tensor


def _resolve_device(*masks: dict[str, torch.Tensor] | None) -> torch.device:
    for mask in masks:
        if mask is None:
            continue
        for value in mask.values():
            if isinstance(value, torch.Tensor):
                return value.device
    return torch.device("cpu")


def _pad_channels(mask: torch.Tensor, channel_count: int, *, modality: str, name: str) -> torch.Tensor:
    if mask.shape[2] > channel_count:
        raise ValueError(f"{name}['{modality}'] has {mask.shape[2]} channels; expected at most {channel_count}.")
    if mask.shape[2] == channel_count:
        return mask
    pad_shape = list(mask.shape)
    pad_shape[2] = channel_count - mask.shape[2]
    pad = torch.zeros(pad_shape, dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, pad], dim=2)


def _availability_for_modality(
    availability_mask: dict[str, torch.Tensor] | None,
    modality: str,
    *,
    batch_size: int,
    context_epochs: int,
    channel_count: int,
    patches_per_epoch: int,
    device: torch.device,
) -> torch.Tensor:
    if availability_mask is None or modality not in availability_mask:
        shape = (batch_size, context_epochs, channel_count, patches_per_epoch)
        return torch.ones(shape, dtype=torch.bool, device=device)

    mask = torch.as_tensor(availability_mask[modality], dtype=torch.bool, device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape == (batch_size, context_epochs):
        return mask[:, :, None, None].expand(batch_size, context_epochs, channel_count, patches_per_epoch)
    if mask.shape == (batch_size, context_epochs, patches_per_epoch):
        return mask[:, :, None, :].expand(batch_size, context_epochs, channel_count, patches_per_epoch)
    if mask.dim() == 4 and mask.shape[:2] == (batch_size, context_epochs) and mask.shape[3] == patches_per_epoch:
        return _pad_channels(mask, channel_count, modality=modality, name="availability_mask")
    raise ValueError(
        f"availability_mask['{modality}'] must have shape "
        f"({batch_size}, {context_epochs}), "
        f"({batch_size}, {context_epochs}, {patches_per_epoch}), or "
        f"({batch_size}, {context_epochs}, C, {patches_per_epoch}), got {tuple(mask.shape)}."
    )


def _channel_mask_for_modality(
    channel_mask: dict[str, torch.Tensor] | None,
    modality: str,
    *,
    batch_size: int,
    context_epochs: int,
    channel_count: int,
    device: torch.device,
) -> torch.Tensor:
    if channel_mask is None or modality not in channel_mask:
        return torch.ones((batch_size, context_epochs, channel_count), dtype=torch.bool, device=device)
    mask = torch.as_tensor(channel_mask[modality], dtype=torch.bool, device=device)
    if mask.dim() != 3 or mask.shape[:2] != (batch_size, context_epochs):
        raise ValueError(
            f"channel_mask['{modality}'] must have shape "
            f"({batch_size}, {context_epochs}, C), got {tuple(mask.shape)}."
        )
    return _pad_channels(mask, channel_count, modality=modality, name="channel_mask")


def _set_modality_tokens(token_mask: torch.Tensor, layout: TokenLayout, modality: str, values: torch.Tensor) -> None:
    if values.shape != (token_mask.shape[0], layout.context_epochs, layout.channel_count, layout.patches_per_epoch):
        raise ValueError(
            f"Token values for '{modality}' must have shape "
            f"({token_mask.shape[0]}, {layout.context_epochs}, {layout.channel_count}, {layout.patches_per_epoch})."
        )
    for epoch in range(layout.context_epochs):
        for channel in range(layout.channel_count):
            for patch in range(layout.patches_per_epoch):
                token_mask[:, layout.token_index(modality, epoch, channel, patch)] = values[:, epoch, channel, patch]


def frame_mask_to_patch_mask(mask: torch.Tensor, patches_per_epoch: int) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.dim() == 3:
        mask = mask.unsqueeze(2)
    if mask.dim() != 4:
        raise ValueError(f"corruption mask must have shape [B, E, C, S], got {tuple(mask.shape)}.")
    batch_size, context_epochs, channels, frames = mask.shape
    if frames % patches_per_epoch != 0:
        raise ValueError("corruption mask frame count must be divisible by diffusion.patches_per_epoch.")
    frames_per_patch = frames // patches_per_epoch
    return mask.reshape(batch_size, context_epochs, channels, patches_per_epoch, frames_per_patch).any(dim=-1)


def build_patch_condition_availability(
    availability_mask: dict[str, torch.Tensor],
    corruption_mask: dict[str, torch.Tensor],
    task: GenerationTask,
    *,
    patches_per_epoch: int,
) -> dict[str, torch.Tensor]:
    if not corruption_mask:
        return availability_mask
    updated = dict(availability_mask)
    for modality in task.condition_modalities:
        if modality not in corruption_mask:
            continue
        corrupted = frame_mask_to_patch_mask(corruption_mask[modality], patches_per_epoch)
        base = torch.as_tensor(updated[modality], dtype=torch.bool, device=corrupted.device)
        if base.dim() == 1:
            base = base.unsqueeze(0)
        if base.shape == corrupted.shape[:2]:
            base = base[:, :, None, None].expand_as(corrupted)
        elif base.shape == (corrupted.shape[0], corrupted.shape[1], corrupted.shape[3]):
            base = base[:, :, None, :].expand_as(corrupted)
        if base.shape != corrupted.shape:
            raise ValueError(
                f"availability_mask['{modality}'] must broadcast to patch mask shape {tuple(corrupted.shape)}."
            )
        updated[modality] = base & ~corrupted
    return updated


def build_directional_task_attention_mask(
    task: GenerationTask,
    layout: TokenLayout,
    *,
    availability_mask: dict[str, torch.Tensor] | None = None,
    condition_availability_mask: dict[str, torch.Tensor] | None = None,
    channel_mask: dict[str, torch.Tensor] | None = None,
    batch_size: int = 1,
) -> TaskAttentionMask:
    task = validate_generation_task(task)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if task.use_auxiliary_token and not layout.include_aux:
        raise ValueError("Task requires an auxiliary token, but layout.include_aux=False.")

    device = _resolve_device(channel_mask, condition_availability_mask, availability_mask)
    token_count = layout.token_count
    condition_tokens = torch.zeros((batch_size, token_count), dtype=torch.bool, device=device)
    target_tokens = torch.zeros((batch_size, token_count), dtype=torch.bool, device=device)

    for modality in task.condition_modalities:
        condition_source = (
            condition_availability_mask
            if condition_availability_mask is not None and modality in condition_availability_mask
            else availability_mask
        )
        values = _availability_for_modality(
            condition_source,
            modality,
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            channel_count=layout.channel_count,
            patches_per_epoch=layout.patches_per_epoch,
            device=device,
        )
        values = values & _channel_mask_for_modality(
            channel_mask,
            modality,
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            channel_count=layout.channel_count,
            device=device,
        ).unsqueeze(-1)
        _set_modality_tokens(condition_tokens, layout, modality, values)

    if is_restoration_task(task):
        target_modality = task.target_modalities[0]
        aux_values = _availability_for_modality(
            availability_mask,
            target_modality,
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            channel_count=layout.channel_count,
            patches_per_epoch=layout.patches_per_epoch,
            device=device,
        )
        aux_values = aux_values & _channel_mask_for_modality(
            channel_mask,
            target_modality,
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            channel_count=layout.channel_count,
            device=device,
        ).unsqueeze(-1)
        _set_modality_tokens(target_tokens, layout, AUX_MODALITY, aux_values)
    else:
        for modality in task.target_modalities:
            target_values = _availability_for_modality(
                availability_mask,
                modality,
                batch_size=batch_size,
                context_epochs=layout.context_epochs,
                channel_count=layout.channel_count,
                patches_per_epoch=layout.patches_per_epoch,
                device=device,
            )
            target_values = target_values & _channel_mask_for_modality(
                channel_mask,
                modality,
                batch_size=batch_size,
                context_epochs=layout.context_epochs,
                channel_count=layout.channel_count,
                device=device,
            ).unsqueeze(-1)
            _set_modality_tokens(target_tokens, layout, modality, target_values)

    active_tokens = condition_tokens | target_tokens
    allowed = torch.zeros((batch_size, token_count, token_count), dtype=torch.bool, device=device)

    for batch_idx in range(batch_size):
        condition = condition_tokens[batch_idx]
        target = target_tokens[batch_idx]

        allowed[batch_idx] |= condition[:, None] & condition[None, :]
        allowed[batch_idx] |= target[:, None] & condition[None, :]
        if task.allow_target_target_attention:
            allowed[batch_idx] |= target[:, None] & target[None, :]
        else:
            target_indices = torch.nonzero(target, as_tuple=False).flatten()
            allowed[batch_idx, target_indices, target_indices] = True

        active = active_tokens[batch_idx]
        allowed[batch_idx] &= active[:, None] & active[None, :]

    return TaskAttentionMask(
        blocked=~allowed,
        active_tokens=active_tokens,
        condition_tokens=condition_tokens,
        target_tokens=target_tokens,
    )


__all__ = [
    "TaskAttentionMask",
    "TokenLayout",
    "build_directional_task_attention_mask",
    "build_patch_condition_availability",
    "frame_mask_to_patch_mask",
]
