"""Guard tests for the TaskAdapter boundary.

These freeze the migration's acceptance criteria: registered adapters stay
in sync with the kernel's variant table, and kernel modules stay free of
adapter-owned task names. Matching is AST-based on string constants (so
identifiers like inference_preset_path or evaluate_recipe never false-
positive), with one structural exemption: a task-name constant inside a
set/list/tuple literal that also names a not-yet-migrated kernel task is
transitional cross-task dispatch. That exemption evaporates automatically
the moment the remaining task registers an adapter.
"""

import ast
from pathlib import Path

from agent_tools.adapters import all_adapters, get_adapter
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

# Kernel tasks not yet migrated to adapters.
KERNEL_LEGACY_TASKS = {"finetune", "hparam_tune"}


def _kernel_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


def _exempt_constant_ids(tree: ast.Module, unmigrated: set[str]) -> set[int]:
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            continue
        values = {elt.value for elt in node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
        if values & unmigrated:
            exempt.update(id(elt) for elt in node.elts if isinstance(elt, ast.Constant))
    return exempt


def test_registry_resolves_migrated_tasks():
    for task in ("sleep2stat", "preset_prepare"):
        adapter = get_adapter(task)
        assert adapter is not None
        assert adapter.task == task
    assert get_adapter(None) is None
    assert get_adapter("") is None


def test_requires_variant_matches_variantless_tasks():
    for adapter in all_adapters():
        assert (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant)


def test_kernel_modules_do_not_name_adapter_tasks():
    kernel_dir = _kernel_dir()
    task_names = {adapter.task for adapter in all_adapters()}
    unmigrated = KERNEL_LEGACY_TASKS - task_names
    offending: list[tuple[str, int, str]] = []
    for module_name in KERNEL_MODULES:
        tree = ast.parse((kernel_dir / module_name).read_text(), module_name)
        exempt = _exempt_constant_ids(tree, unmigrated)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in task_names
                and id(node) not in exempt
            ):
                offending.append((module_name, node.lineno, node.value))
    assert offending == [], f"adapter task names leaked into kernel modules: {offending}"
