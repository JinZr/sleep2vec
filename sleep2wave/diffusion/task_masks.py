from __future__ import annotations

from dataclasses import dataclass

import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES, validate_modality_sequence
from sleep2wave.diffusion.tasks import AUX_MODALITY, GenerationTask, is_restoration_task, validate_generation_task


@dataclass(frozen=True)
class TokenLayout:
    modalities: tuple[str, ...] = CANONICAL_MODALITIES
    context_epochs: int = 15
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
        if not isinstance(self.include_aux, bool):
            raise ValueError("include_aux must be a boolean.")

    @property
    def token_modalities(self) -> tuple[str, ...]:
        if self.include_aux:
            return (*self.modalities, AUX_MODALITY)
        return self.modalities

    @property
    def token_count(self) -> int:
        return len(self.token_modalities) * self.context_epochs

    @property
    def token_names(self) -> list[str]:
        return [f"{modality}_{epoch}" for modality in self.token_modalities for epoch in range(self.context_epochs)]

    def token_index(self, modality: str, epoch: int) -> int:
        if modality == AUX_MODALITY:
            if not self.include_aux:
                raise ValueError("Auxiliary token is not included in this layout.")
        elif modality not in self.modalities:
            raise ValueError(f"Unknown layout modality: {modality}")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < self.context_epochs:
            raise ValueError(f"epoch must be in [0, {self.context_epochs}).")
        return self.token_modalities.index(modality) * self.context_epochs + epoch

    def modality_indices(self, modality: str) -> list[int]:
        return [self.token_index(modality, epoch) for epoch in range(self.context_epochs)]


@dataclass(frozen=True)
class TaskAttentionMask:
    blocked: torch.Tensor
    active_tokens: torch.Tensor
    condition_tokens: torch.Tensor
    target_tokens: torch.Tensor


def _resolve_device(availability_mask: dict[str, torch.Tensor] | None) -> torch.device:
    if availability_mask is None:
        return torch.device("cpu")
    for value in availability_mask.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _availability_for_modality(
    availability_mask: dict[str, torch.Tensor] | None,
    modality: str,
    *,
    batch_size: int,
    context_epochs: int,
    device: torch.device,
) -> torch.Tensor:
    if availability_mask is None or modality not in availability_mask:
        return torch.ones((batch_size, context_epochs), dtype=torch.bool, device=device)

    mask = torch.as_tensor(availability_mask[modality], dtype=torch.bool, device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != (batch_size, context_epochs):
        raise ValueError(
            f"availability_mask['{modality}'] must have shape "
            f"({batch_size}, {context_epochs}), got {tuple(mask.shape)}."
        )
    return mask


def _set_modality_tokens(token_mask: torch.Tensor, layout: TokenLayout, modality: str, values: torch.Tensor) -> None:
    if values.shape != (token_mask.shape[0], layout.context_epochs):
        raise ValueError(
            f"Token values for '{modality}' must have shape " f"({token_mask.shape[0]}, {layout.context_epochs})."
        )
    for epoch in range(layout.context_epochs):
        token_mask[:, layout.token_index(modality, epoch)] = values[:, epoch]


def build_directional_task_attention_mask(
    task: GenerationTask,
    layout: TokenLayout,
    *,
    availability_mask: dict[str, torch.Tensor] | None = None,
    batch_size: int = 1,
) -> TaskAttentionMask:
    task = validate_generation_task(task)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if task.use_auxiliary_token and not layout.include_aux:
        raise ValueError("Task requires an auxiliary token, but layout.include_aux=False.")

    device = _resolve_device(availability_mask)
    token_count = layout.token_count
    condition_tokens = torch.zeros((batch_size, token_count), dtype=torch.bool, device=device)
    target_tokens = torch.zeros((batch_size, token_count), dtype=torch.bool, device=device)

    for modality in task.condition_modalities:
        values = _availability_for_modality(
            availability_mask,
            modality,
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            device=device,
        )
        _set_modality_tokens(condition_tokens, layout, modality, values)

    if is_restoration_task(task):
        aux_values = _availability_for_modality(
            availability_mask,
            task.target_modalities[0],
            batch_size=batch_size,
            context_epochs=layout.context_epochs,
            device=device,
        )
        _set_modality_tokens(target_tokens, layout, AUX_MODALITY, aux_values)
    else:
        for modality in task.target_modalities:
            target_values = _availability_for_modality(
                availability_mask,
                modality,
                batch_size=batch_size,
                context_epochs=layout.context_epochs,
                device=device,
            )
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
]
