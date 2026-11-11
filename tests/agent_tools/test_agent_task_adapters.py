"""Guard tests for the TaskAdapter boundary.

Registered adapters stay in sync with the variant table, the composite
adapter's base task is registered, and decision-field to recipe-target
mappings come only from schema_map.
"""

import ast
from pathlib import Path

from agent_tools.adapters.registry import SUPPORTED_TASKS, all_adapters, composite_adapter, get_adapter
from agent_tools.models import VARIANTLESS_TASKS

ALL_TASKS = (
    "sleep2stat",
    "embedding_extraction",
    "preset_prepare",
    "infer",
    "finetune",
    "hparam_tune",
)


def _package_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


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


def _is_field_target_map(node: ast.AST) -> bool:
    """A {"field": (section, key), ...} literal -- the shape of a decision-field
    to recipe-target mapping. schema_map owns the one canonical copy; decisions
    and plans must not grow their own."""
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


def test_decisions_and_plans_do_not_hardcode_field_target_maps():
    """decisions/plans must source decision-field -> recipe (section, key)
    mappings from schema_map, never a local literal."""
    package_dir = _package_dir()
    offending: list[tuple[str, int]] = []
    for module_name in ("decisions.py", "plans.py"):
        tree = ast.parse((package_dir / module_name).read_text(), module_name)
        for node in ast.walk(tree):
            if _is_field_target_map(node):
                offending.append((module_name, node.lineno))
    assert offending == [], f"field->target maps outside schema_map: {offending}"


def test_write_targets_resolve_from_schema_map():
    """The write side stays in lockstep with schema_map + adapter overrides."""
    from agent_tools import plans, schema_map

    for adapter in all_adapters():
        expected = schema_map.merged_write_targets(dict(adapter.decision_recipe_targets))
        assert plans._resolve_write_targets(adapter.task) == expected
