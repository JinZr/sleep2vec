"""Layer 1: the task-adapter protocol and registry.

Import ``TaskAdapter`` and the registry helpers from here. Per-task adapters
are imported by ``registry`` in a deliberate order and should not be reached
directly.
"""

from .base import TaskAdapter
from .registry import SUPPORTED_TASKS, all_adapters, composite_adapter, get_adapter

__all__ = ["SUPPORTED_TASKS", "TaskAdapter", "all_adapters", "composite_adapter", "get_adapter"]
