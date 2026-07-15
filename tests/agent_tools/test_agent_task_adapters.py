"""Guard tests for the TaskAdapter boundary.

These freeze the pilot's acceptance criteria: registered adapters stay in
sync with the kernel's variant table, and kernel modules stay free of
adapter-owned task names (one whitelisted re-export in configs.py).
"""

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
# configs.py keeps a re-export because tests import
# agent_tools.configs.sleep2stat_config_summary; it goes away once every
# task is an adapter.
WHITELISTED_LINES = {
    ("configs.py", "from .adapters.sleep2stat import sleep2stat_config_summary"),
}


def _kernel_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


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
    offending: list[tuple[str, int, str]] = []
    task_names = {adapter.task for adapter in all_adapters()}
    for module_name in KERNEL_MODULES:
        for line_number, line in enumerate((kernel_dir / module_name).read_text().splitlines(), start=1):
            for task_name in task_names:
                if task_name in line and (module_name, line.strip().split("  #")[0].strip()) not in {
                    (module, text) for module, text in WHITELISTED_LINES
                }:
                    offending.append((module_name, line_number, line.strip()))
    assert offending == [], f"adapter task names leaked into kernel modules: {offending}"
