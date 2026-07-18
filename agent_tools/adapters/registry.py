from __future__ import annotations

from typing import Any

from ..models import VARIANTLESS_TASKS
from .base import TaskAdapter

# Registration order is the probing order for config-shape claims
# (index_summary_inputs_override / matches_config_data): config-probing
# adapters must precede task-keyed ones.
from .finetune import FINETUNE_ADAPTER
from .hparam_tune import HPARAM_TUNE_ADAPTER
from .infer_evaluate import EVALUATE_ADAPTER, INFER_ADAPTER
from .preset_prepare import PRESET_PREPARE_ADAPTER
from .sleep2stat import SLEEP2STAT_ADAPTER

_REGISTERED_ADAPTERS = (
    SLEEP2STAT_ADAPTER,
    PRESET_PREPARE_ADAPTER,
    INFER_ADAPTER,
    EVALUATE_ADAPTER,
    FINETUNE_ADAPTER,
    HPARAM_TUNE_ADAPTER,
)
if len({adapter.task for adapter in _REGISTERED_ADAPTERS}) != len(_REGISTERED_ADAPTERS):
    raise RuntimeError("TaskAdapter.task values must be unique")
TASK_ADAPTERS: dict[str, TaskAdapter] = {adapter.task: adapter for adapter in _REGISTERED_ADAPTERS}
SUPPORTED_TASKS = frozenset(TASK_ADAPTERS)


def get_adapter(task: Any) -> TaskAdapter | None:
    if task in (None, ""):
        return None
    return TASK_ADAPTERS.get(str(task))


def all_adapters() -> tuple[TaskAdapter, ...]:
    return tuple(TASK_ADAPTERS.values())


def composite_adapter() -> TaskAdapter:
    """The single composite (layered-recipe) adapter. Recipes carrying
    _base_recipe/_local_recipe layers close under its contract."""
    composites = [adapter for adapter in TASK_ADAPTERS.values() if adapter.base_task is not None]
    assert len(composites) == 1, "exactly one composite task adapter is expected"
    return composites[0]


assert all(
    (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant) for adapter in TASK_ADAPTERS.values()
), "TaskAdapter.requires_variant must stay in sync with models.VARIANTLESS_TASKS"
assert all(
    adapter.base_task is None or adapter.base_task in TASK_ADAPTERS for adapter in TASK_ADAPTERS.values()
), "TaskAdapter.base_task must reference a registered adapter"
