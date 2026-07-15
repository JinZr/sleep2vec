from __future__ import annotations

from typing import Any

from ..models import VARIANTLESS_TASKS
from .base import TaskAdapter

# Registration order is the probing order for config-shape claims
# (index_summary_inputs_override / matches_config_data): config-probing
# adapters must precede task-keyed ones.
TASK_ADAPTERS: dict[str, TaskAdapter] = {}


def get_adapter(task: Any) -> TaskAdapter | None:
    if task in (None, ""):
        return None
    return TASK_ADAPTERS.get(str(task))


def all_adapters() -> tuple[TaskAdapter, ...]:
    return tuple(TASK_ADAPTERS.values())


assert all(
    (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant) for adapter in TASK_ADAPTERS.values()
), "TaskAdapter.requires_variant must stay in sync with models.VARIANTLESS_TASKS"
