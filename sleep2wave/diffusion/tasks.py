from __future__ import annotations

from dataclasses import dataclass
import typing as t

from sleep2wave.data.modalities import validate_modality_sequence

TASK_TYPES = {"restoration", "imputation", "translation", "partial_full"}
AUX_MODALITY = "aux"
_RESTORATION_TASK_TYPES = {"restoration", "imputation"}


@dataclass(frozen=True)
class GenerationTask:
    task_type: str
    condition_modalities: tuple[str, ...]
    target_modalities: tuple[str, ...]
    use_auxiliary_token: bool
    allow_target_target_attention: bool = True


def _normalize_required_modalities(raw: t.Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{field_name} must be a non-empty sequence.")
    return tuple(validate_modality_sequence(list(raw), allow_aliases=False))


def validate_generation_task(task: GenerationTask) -> GenerationTask:
    if task.task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {sorted(TASK_TYPES)}.")
    if not isinstance(task.use_auxiliary_token, bool):
        raise ValueError("use_auxiliary_token must be a boolean.")
    if not isinstance(task.allow_target_target_attention, bool):
        raise ValueError("allow_target_target_attention must be a boolean.")

    condition_modalities = _normalize_required_modalities(task.condition_modalities, "condition_modalities")
    target_modalities = _normalize_required_modalities(task.target_modalities, "target_modalities")
    condition_set = set(condition_modalities)
    target_set = set(target_modalities)

    if task.task_type in {"translation", "partial_full"}:
        overlap = sorted(condition_set & target_set)
        if overlap:
            raise ValueError(f"{task.task_type} requires disjoint condition and target modalities: {overlap}")
        if task.use_auxiliary_token:
            raise ValueError(f"{task.task_type} does not use the auxiliary restoration token.")
    else:
        if len(target_modalities) != 1:
            raise ValueError(f"{task.task_type} requires exactly one target modality.")
        target = target_modalities[0]
        if target not in condition_set:
            raise ValueError(f"{task.task_type} requires the target modality '{target}' as a condition.")
        if not task.use_auxiliary_token:
            raise ValueError(f"{task.task_type} requires auxiliary_restoration_token=True.")

    return GenerationTask(
        task_type=task.task_type,
        condition_modalities=condition_modalities,
        target_modalities=target_modalities,
        use_auxiliary_token=task.use_auxiliary_token,
        allow_target_target_attention=task.allow_target_target_attention,
    )


def build_generation_task(
    task_type: str,
    *,
    condition_modalities: t.Sequence[str],
    target_modalities: t.Sequence[str],
    auxiliary_restoration_token: bool = False,
    allow_target_target_attention: bool = True,
) -> GenerationTask:
    return validate_generation_task(
        GenerationTask(
            task_type=task_type,
            condition_modalities=tuple(condition_modalities),
            target_modalities=tuple(target_modalities),
            use_auxiliary_token=auxiliary_restoration_token,
            allow_target_target_attention=allow_target_target_attention,
        )
    )


def is_restoration_task(task: GenerationTask) -> bool:
    return task.task_type in _RESTORATION_TASK_TYPES


__all__ = [
    "AUX_MODALITY",
    "TASK_TYPES",
    "GenerationTask",
    "build_generation_task",
    "is_restoration_task",
    "validate_generation_task",
]
