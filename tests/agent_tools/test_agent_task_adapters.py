"""Guard tests for the TaskAdapter boundary.

These freeze the pilot's acceptance criteria: registered adapters stay in
sync with the kernel's variant table, and kernel modules stay free of
adapter-owned task names outside of imports from the adapters package
(layer 2 importing layer 1 is the allowed direction, and those import
statements necessarily contain the task name in the module path).
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


def _kernel_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


def _adapter_import_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("adapters"):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def test_registry_resolves_sleep2stat():
    adapter = get_adapter("sleep2stat")
    assert adapter is not None
    assert adapter.task == "sleep2stat"
    assert get_adapter(None) is None
    assert get_adapter("") is None


def test_requires_variant_matches_variantless_tasks():
    for adapter in all_adapters():
        assert (adapter.task in VARIANTLESS_TASKS) == (not adapter.requires_variant)


def test_kernel_modules_do_not_name_adapter_tasks():
    kernel_dir = _kernel_dir()
    task_names = {adapter.task for adapter in all_adapters()}
    offending: list[tuple[str, int, str]] = []
    for module_name in KERNEL_MODULES:
        source = (kernel_dir / module_name).read_text()
        import_lines = _adapter_import_lines(ast.parse(source, module_name))
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line_number in import_lines:
                continue
            if any(task_name in line for task_name in task_names):
                offending.append((module_name, line_number, line.strip()))
    assert offending == [], f"adapter task names leaked into kernel modules: {offending}"
