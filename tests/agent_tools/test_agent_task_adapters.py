"""Guard tests for the TaskAdapter boundary.

These freeze the migration's acceptance criteria: registered adapters stay
in sync with the kernel's variant table, the composite adapter's base task
is registered, and kernel modules contain no task-name string constants at
all -- every per-task behaviour lives in an adapter. Matching is AST-based
on string constants, so identifiers like inference_preset_path or
evaluate_recipe can never false-positive.
"""

import ast
from pathlib import Path

from agent_tools.adapters import all_adapters, composite_adapter, get_adapter
from agent_tools.models import VARIANTLESS_TASKS

KERNEL_MODULES = (
    "plans.py",
    "decisions.py",
    "plan_context.py",
    "decision_rules.py",
    "decision_paths.py",
    "plan_rendering.py",
    "configs.py",
)

ALL_TASKS = ("sleep2stat", "preset_prepare", "infer", "evaluate", "finetune", "hparam_tune")


def _kernel_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


def test_registry_resolves_migrated_tasks():
    for task in ALL_TASKS:
        adapter = get_adapter(task)
        assert adapter is not None
        assert adapter.task == task
    assert get_adapter(None) is None
    assert get_adapter("") is None


def test_requires_variant_matches_variantless_tasks():
    for adapter in all_adapters():
        assert (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant)


def test_composite_adapter_base_task_is_registered():
    composite = composite_adapter()
    assert composite.task == "hparam_tune"
    assert get_adapter(composite.base_task) is not None


def test_kernel_modules_do_not_name_adapter_tasks():
    kernel_dir = _kernel_dir()
    task_names = {adapter.task for adapter in all_adapters()}
    offending: list[tuple[str, int, str]] = []
    for module_name in KERNEL_MODULES:
        tree = ast.parse((kernel_dir / module_name).read_text(), module_name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in task_names:
                offending.append((module_name, node.lineno, node.value))
    assert offending == [], f"adapter task names leaked into kernel modules: {offending}"
