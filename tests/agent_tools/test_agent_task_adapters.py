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


def _is_field_target_map(node: ast.AST) -> bool:
    """A {"field": (section, key), ...} literal -- the shape of a decision-field
    to recipe-target mapping. schema_map owns the one canonical copy; kernel
    modules must not grow their own."""
    if not isinstance(node, ast.Dict) or not node.keys:
        return False
    for key, value in zip(node.keys, node.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return False
        if not (
            isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in value.elts)
        ):
            return False
    return True


def test_kernel_modules_do_not_hardcode_field_target_maps():
    """decisions/plans must source decision-field -> recipe (section, key)
    mappings from schema_map, never a local literal."""
    kernel_dir = _kernel_dir()
    offending: list[tuple[str, int]] = []
    for module_name in ("decisions.py", "plans.py"):
        tree = ast.parse((kernel_dir / module_name).read_text(), module_name)
        for node in ast.walk(tree):
            if _is_field_target_map(node):
                offending.append((module_name, node.lineno))
    assert offending == [], f"field->target maps leaked into kernel modules: {offending}"


def test_write_targets_resolve_from_schema_map():
    """The write side stays in lockstep with schema_map + adapter overrides."""
    from agent_tools import plans, schema_map

    for adapter in all_adapters():
        expected = schema_map.merged_write_targets(dict(adapter.decision_recipe_targets))
        assert plans._resolve_write_targets(adapter.task) == expected
