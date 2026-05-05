from __future__ import annotations

import random
import typing as t

import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES, validate_modality_sequence
from sleep2wave.diffusion.tasks import GenerationTask, build_generation_task
from sleep2wave.training.phase_schedule import build_phase_schedule


class Sleep2WaveTaskSampler:
    def __init__(
        self,
        *,
        modalities: t.Sequence[str] = CANONICAL_MODALITIES,
        phase: int,
        task_mix: dict[str, float] | None = None,
        condition_counts: t.Sequence[int] | None = None,
        auxiliary_restoration_token: bool = True,
        seed: int = 0,
    ) -> None:
        self.modalities = tuple(validate_modality_sequence(list(modalities), allow_aliases=False))
        self.schedule = build_phase_schedule(phase, task_mix)
        self.condition_counts = tuple(condition_counts or (1,))
        if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in self.condition_counts):
            raise ValueError("condition_counts must contain positive integers.")
        self.auxiliary_restoration_token = bool(auxiliary_restoration_token)
        self.rng = random.Random(seed)

    def _available_modality_sets(self, availability_mask: dict[str, torch.Tensor] | None) -> list[list[str]]:
        if availability_mask is None:
            return [list(self.modalities)]

        batch_size: int | None = None
        modality_values: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if modality not in availability_mask:
                continue
            values = torch.as_tensor(availability_mask[modality], dtype=torch.bool)
            if values.dim() == 1:
                values = values.unsqueeze(0)
            if values.dim() != 2:
                raise ValueError(f"availability_mask['{modality}'] must have shape [B, E] or [E].")
            if batch_size is None:
                batch_size = values.shape[0]
            elif values.shape[0] != batch_size:
                raise ValueError("All availability masks must share the same batch size.")
            modality_values[modality] = values

        if batch_size is None:
            return []

        available_sets: list[list[str]] = []
        for batch_idx in range(batch_size):
            available_sets.append(
                [
                    modality
                    for modality in self.modalities
                    if modality in modality_values and bool(modality_values[modality][batch_idx].any())
                ]
            )
        return available_sets

    def _choose_available_set(self, availability_mask: dict[str, torch.Tensor] | None, *, min_size: int) -> list[str]:
        available_sets = self._available_modality_sets(availability_mask)
        if not available_sets:
            common_available: list[str] = []
        else:
            common_available = [
                modality for modality in self.modalities if all(modality in available for available in available_sets)
            ]
        if len(common_available) < min_size:
            if min_size <= 1:
                raise ValueError("No modalities are available for sleep2wave task sampling.")
            if min_size > 2:
                raise ValueError("Not enough available modalities to sample a disjoint condition set.")
            raise ValueError("Translation and partial_full tasks require at least two available modalities.")
        return common_available

    def _weighted_family(self) -> str:
        normalized = self.schedule.normalized()
        threshold = self.rng.random()
        cumulative = 0.0
        for name, weight in normalized.items():
            cumulative += weight
            if threshold <= cumulative:
                return name
        return next(reversed(normalized))

    def _choice(self, values: t.Sequence[str]) -> str:
        if not values:
            raise ValueError("Cannot sample from an empty modality set.")
        return values[self.rng.randrange(len(values))]

    def _sample_condition_set(self, available: list[str], target: str, count: int) -> list[str]:
        candidates = [modality for modality in available if modality != target]
        if len(candidates) < count:
            raise ValueError("Not enough available modalities to sample a disjoint condition set.")
        return self.rng.sample(candidates, count)

    def _sample_target_set(self, available: list[str], condition: list[str]) -> list[str]:
        condition_set = set(condition)
        candidates = [modality for modality in available if modality not in condition_set]
        if not candidates:
            raise ValueError("No available target modalities remain after condition sampling.")
        return candidates

    def sample(self, availability_mask: dict[str, torch.Tensor] | None = None) -> GenerationTask:
        family = self._weighted_family()
        if family == "two_condition":
            available = self._choose_available_set(availability_mask, min_size=3)
        elif family in {"restoration", "imputation"}:
            available = self._choose_available_set(availability_mask, min_size=1)
        else:
            available = self._choose_available_set(availability_mask, min_size=2)

        if family in {"restoration", "imputation"}:
            target = self._choice(available)
            return build_generation_task(
                family,
                condition_modalities=[target],
                target_modalities=[target],
                auxiliary_restoration_token=self.auxiliary_restoration_token,
            )

        target = self._choice(available)
        if family == "two_condition":
            condition_count = 2
            task_type = "translation"
        elif family == "partial_full":
            condition_count = min(max(self.condition_counts), len(available) - 1)
            task_type = "partial_full"
        else:
            condition_count = min(self.rng.choice(self.condition_counts), len(available) - 1)
            task_type = "translation"
        condition = self._sample_condition_set(available, target, condition_count)
        if task_type == "partial_full":
            target_modalities = self._sample_target_set(available, condition)
        else:
            target_modalities = [target]
        return build_generation_task(
            task_type,
            condition_modalities=condition,
            target_modalities=target_modalities,
        )


__all__ = ["Sleep2WaveTaskSampler"]
