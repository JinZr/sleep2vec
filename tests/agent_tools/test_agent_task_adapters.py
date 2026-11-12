"""Guard tests for the TaskAdapter boundary.

Registered adapters stay in sync with the variant table and the composite
adapter's base task is registered.
"""

from agent_tools.adapters.registry import SUPPORTED_TASKS, all_adapters, composite_adapter, get_adapter
from agent_tools.models import VARIANTLESS_TASKS

ALL_TASKS = (
    "sleep2stat",
    "preset_prepare",
    "infer",
    "finetune",
    "hparam_tune",
)


def test_registry_resolves_migrated_tasks():
    for task in ALL_TASKS:
        adapter = get_adapter(task)
        assert adapter is not None
        assert adapter.task == task
    assert get_adapter(None) is None
    assert get_adapter("") is None
    assert SUPPORTED_TASKS == frozenset(ALL_TASKS)


def test_requires_variant_matches_variantless_tasks():
    for adapter in all_adapters():
        assert (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant)


def test_composite_adapter_base_task_is_registered():
    composite = composite_adapter()
    assert composite.task == "hparam_tune"
    assert get_adapter(composite.base_task) is not None
