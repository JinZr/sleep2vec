"""Layer 1: the task-adapter protocol and registry.

Import ``TaskAdapter`` and the registry helpers from here. Per-task adapters
are imported by ``registry`` in a deliberate order; callers should use the
registry helpers rather than importing adapter implementations directly. The
direct ``sleep2stat`` import in ``configs`` is an intentional compatibility
exception for the frozen ``configs.sleep2stat_config_summary`` path.
"""

from .base import TaskAdapter
from .registry import SUPPORTED_TASKS, all_adapters, composite_adapter, get_adapter

__all__ = ["SUPPORTED_TASKS", "TaskAdapter", "all_adapters", "composite_adapter", "get_adapter"]
