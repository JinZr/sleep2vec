from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PHASE_TASK_MIX: dict[int, dict[str, float]] = {
    1: {"restoration": 0.5, "imputation": 0.5},
    2: {"restoration": 0.25, "imputation": 0.25, "translation": 0.5},
    3: {"restoration": 0.125, "imputation": 0.125, "translation": 0.25, "two_condition": 0.5},
    4: {"restoration": 0.1, "imputation": 0.1, "translation": 0.2, "two_condition": 0.3, "partial_full": 0.3},
    5: {"restoration": 0.2, "imputation": 0.2, "translation": 0.2, "two_condition": 0.2, "partial_full": 0.2},
}
CURRENT_PHASE_TASK_MIX: dict[int, dict[str, float]] = {
    1: {"restoration": 1.0},
    2: {"translation": 1.0},
    3: {"two_condition": 1.0},
    4: {"partial_full": 1.0},
    5: {"partial_full": 1.0},
}
TASK_FAMILIES = {"restoration", "imputation", "translation", "two_condition", "partial_full"}


@dataclass(frozen=True)
class PhaseSchedule:
    phase: int
    task_mix: dict[str, float]

    def normalized(self) -> dict[str, float]:
        total = sum(self.task_mix.values())
        if total <= 0:
            raise ValueError("task_mix must have positive total weight.")
        return {name: value / total for name, value in self.task_mix.items()}


def _validate_task_mix(task_mix: dict[str, float]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for name, value in task_mix.items():
        if name not in TASK_FAMILIES:
            raise ValueError(f"Unsupported sleep2wave task family: {name}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Task weight for '{name}' must be a non-negative number.")
        if value > 0:
            parsed[name] = float(value)
    if not parsed:
        raise ValueError("task_mix must include at least one positive task weight.")
    return parsed


def build_phase_schedule(
    phase: int,
    task_mix: dict[str, float] | None = None,
    *,
    replay_enabled: bool = True,
) -> PhaseSchedule:
    if not isinstance(phase, int) or isinstance(phase, bool) or phase < 1 or phase > 5:
        raise ValueError("phase must be an integer between 1 and 5 for diffusion training.")
    default_mix = DEFAULT_PHASE_TASK_MIX[phase] if replay_enabled else CURRENT_PHASE_TASK_MIX[phase]
    raw_mix = default_mix if not task_mix else task_mix
    return PhaseSchedule(phase=phase, task_mix=_validate_task_mix(raw_mix))


__all__ = ["CURRENT_PHASE_TASK_MIX", "DEFAULT_PHASE_TASK_MIX", "PhaseSchedule", "build_phase_schedule"]
