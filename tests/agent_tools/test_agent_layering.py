"""Layering guard: kernel and mixed modules must not grow new domain imports.

Complements test_agent_task_adapters.py (which forbids task-name string
constants in kernel modules). Here we scan import statements: any module in
KERNEL_MODULES or MIXED_MODULES that imports a DOMAIN_MODULES module is a
reverse edge, allowed only if it is in KNOWN_DOMAIN_IMPORT_EXEMPTIONS.

Pure-kernel modules carry no exemptions, so they must stay domain-free. The
mixed bridges carry only the grandfathered edges in
KNOWN_DOMAIN_IMPORT_EXEMPTIONS; a new one fails.

It also checks the package import graph itself. Counting the implicit package
``__init__`` edges, the graph must be acyclic, and no module may import another
package module inside a function. Imports under ``if TYPE_CHECKING:`` never run,
so they do not count.

Reads only ast + agent_tools.layering (a zero-dependency data module), so it
runs in the domain-free CI environment.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

from agent_tools import layering


def _package_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


def _module_source(module: str) -> str:
    return (_package_dir() / (module.replace(".", "/") + ".py")).read_text()


_PACKAGE = "agent_tools"


def _strip_package(dotted: str) -> str | None:
    """Package-local remainder of an absolute ``agent_tools[.x.y]`` name.

    ``agent_tools`` -> None (the package itself, no submodule target)
    ``agent_tools.domain.presets`` -> "domain.presets"
    non-package names -> None
    """
    if dotted == _PACKAGE:
        return None
    prefix = _PACKAGE + "."
    return dotted[len(prefix) :] if dotted.startswith(prefix) else None


def _normalize_import(node: ast.AST, source_module: str) -> list[str]:
    """Package-local dotted targets an import node reaches.

    Both relative and absolute intra-package forms normalize to the same names,
    so the guard can't be bypassed by spelling the import absolutely:

    ``from .domain.presets import x``            -> ["domain.presets.x", "domain.presets"]
    ``from agent_tools.domain.presets import x`` -> ["domain.presets.x", "domain.presets"]
    ``import agent_tools.domain.presets``        -> ["domain.presets"]
    ``from . import configs`` / ``from agent_tools import configs`` -> ["configs"]

    Out-of-package imports are ignored -- only intra-package targets matter.
    """
    if isinstance(node, ast.Import):
        # ``import agent_tools.domain.presets [as p]``
        return [local for alias in node.names if (local := _strip_package(alias.name)) is not None]
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            package_parts = source_module.split(".")[:-1]
            parents_up = node.level - 1
            if parents_up > len(package_parts):
                return []
            base_parts = package_parts[: len(package_parts) - parents_up] if parents_up else package_parts
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        elif node.module is not None:  # ``from agent_tools.X import ...``
            base = _strip_package(node.module)
            if node.module != _PACKAGE and base is None:
                return []  # unrelated absolute import
        else:
            return []
        if base:
            # names may be submodules (``from .domain import presets``) or symbols.
            return [f"{base}.{alias.name}" for alias in node.names] + [base]
        # ``from . import configs`` / ``from agent_tools import configs`` -> siblings.
        return [alias.name for alias in node.names]
    return []


def _reverse_import_offenders(module: str, source: str) -> list[tuple[str, str, int]]:
    """Non-exempt (module, domain_target, lineno) reverse edges in ``source``."""
    offenders: list[tuple[str, str, int]] = []
    tree = ast.parse(source, module)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for target in _normalize_import(node, module):
            if target in layering.DOMAIN_MODULES and (module, target) not in layering.KNOWN_DOMAIN_IMPORT_EXEMPTIONS:
                offenders.append((module, target, node.lineno))
    return offenders


def _scanned_modules() -> set[str]:
    # Kernel + mixed, plus any module that is the source of an exemption (so the
    # exemption is actually exercised).
    return (
        set(layering.KERNEL_MODULES)
        | set(layering.MIXED_MODULES)
        | {source for source, _ in layering.KNOWN_DOMAIN_IMPORT_EXEMPTIONS}
    )


def _adapter_modules() -> set[str]:
    root = _package_dir()
    return {str(path.relative_to(root).with_suffix("")).replace("/", ".") for path in (root / "adapters").rglob("*.py")}


def _adapter_l2_offenders(module: str, source: str) -> list[tuple[str, str, int]]:
    offenders = []
    tree = ast.parse(source, module)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for target in _normalize_import(node, module):
            if target in layering.L2_MODULES:
                offenders.append((module, target, node.lineno))
    return offenders


def test_no_new_reverse_domain_imports():
    offenders: list[tuple[str, str, int]] = []
    for module in sorted(_scanned_modules()):
        offenders.extend(_reverse_import_offenders(module, _module_source(module)))
    assert offenders == [], f"new kernel/mixed -> domain imports (not in exemptions): {offenders}"


def test_guard_catches_synthetic_violation():
    # A pure-kernel module (decisions) reaching into a domain module is not
    # exempt and must be flagged -- in relative and both absolute spellings, so
    # the guard can't be bypassed by writing the import differently.
    relative = _reverse_import_offenders("decisions", "from .domain.presets import preset_summary\n")
    assert ("decisions", "domain.presets", 1) in relative

    absolute_from = _reverse_import_offenders("decisions", "from agent_tools.domain.presets import preset_summary\n")
    assert ("decisions", "domain.presets", 1) in absolute_from

    absolute_import = _reverse_import_offenders("decisions", "import agent_tools.domain.presets\n")
    assert ("decisions", "domain.presets", 1) in absolute_import

    nested_relative = _reverse_import_offenders("nested.worker", "from ..domain import presets\n")
    assert ("nested.worker", "domain.presets", 1) in nested_relative


def test_adapters_do_not_import_l2_orchestration():
    offenders = []
    for module in sorted(_adapter_modules()):
        offenders.extend(_adapter_l2_offenders(module, _module_source(module)))
    assert offenders == [], f"adapter -> L2 imports: {offenders}"


def test_adapter_guard_catches_relative_and_absolute_l2_imports():
    relative = _adapter_l2_offenders("adapters.finetune", "from ..plans import build_plan\n")
    absolute = _adapter_l2_offenders("adapters.finetune", "from agent_tools.plans import build_plan\n")
    assert ("adapters.finetune", "plans", 1) in relative
    assert ("adapters.finetune", "plans", 1) in absolute

    pipeline_relative = _adapter_l2_offenders(
        "adapters.finetune", "from ..experiment_pipeline import run_experiment_pipeline\n"
    )
    pipeline_absolute = _adapter_l2_offenders(
        "adapters.finetune", "from agent_tools.experiment_pipeline import run_experiment_pipeline\n"
    )
    assert ("adapters.finetune", "experiment_pipeline", 1) in pipeline_relative
    assert ("adapters.finetune", "experiment_pipeline", 1) in pipeline_absolute


def test_every_exemption_is_live():
    # No stale exemptions: each grandfathered edge must still exist in source.
    for source, target in layering.KNOWN_DOMAIN_IMPORT_EXEMPTIONS:
        tree = ast.parse(_module_source(source), source)
        edges = {
            t
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for t in _normalize_import(node, source)
        }
        assert target in edges, f"stale exemption: {source} no longer imports {target}"


def test_layering_modules_exist():
    root = _package_dir()
    for module in layering.KERNEL_MODULES | layering.DOMAIN_MODULES | layering.MIXED_MODULES:
        assert (root / (module.replace(".", "/") + ".py")).exists(), f"declared module missing: {module}"


#: Modules deliberately outside the three-way partition: the declaration module
#: itself and the adapter protocol skeleton (generic plumbing, documented in
#: ARCHITECTURE.md but not kernel/domain/mixed). Everything else must be
#: classified -- an unregistered module bypasses the reverse-import guard.
_PARTITION_EXEMPT = frozenset(
    {
        "layering",
        "adapters.base",
        "adapters.registry",
        "adapters.config_providers",
    }
)


def test_every_module_is_classified():
    root = _package_dir()
    declared = (
        set(layering.KERNEL_MODULES) | set(layering.DOMAIN_MODULES) | set(layering.MIXED_MODULES) | _PARTITION_EXEMPT
    )
    unclassified = sorted(
        str(path.relative_to(root).with_suffix("")).replace("/", ".")
        for path in root.rglob("*.py")
        if path.name not in ("__init__.py", "__main__.py")
        and str(path.relative_to(root).with_suffix("")).replace("/", ".") not in declared
    )
    assert unclassified == [], f"modules missing from the layering partition (add to layering.py): {unclassified}"


def test_layering_partitions_disjoint():
    assert layering.KERNEL_MODULES.isdisjoint(layering.DOMAIN_MODULES)
    assert layering.KERNEL_MODULES.isdisjoint(layering.MIXED_MODULES)
    assert layering.DOMAIN_MODULES.isdisjoint(layering.MIXED_MODULES)
    assert _PARTITION_EXEMPT.isdisjoint(layering.KERNEL_MODULES | layering.DOMAIN_MODULES | layering.MIXED_MODULES)
    assert layering.L2_MODULES <= layering.KERNEL_MODULES | layering.MIXED_MODULES


def test_mixed_modules_acknowledged():
    # Freeze the mixed set so a module can't silently slide into "mixed".
    assert layering.MIXED_MODULES == frozenset(
        {
            "models",
            "configs",
            "plan_rendering",
            "decision_paths",
            "decision_hparam",
            "plan_hparam",
            "plan_context",
            "hparam_postprocess",
            "cli",
        }
    )


def _module_summary(source: str, module: str) -> str | None:
    """First line of the module docstring, or None when there is no docstring."""
    docstring = ast.get_docstring(ast.parse(source, module))
    return docstring.strip().splitlines()[0].strip() if docstring and docstring.strip() else None


def test_every_module_has_nonempty_docstring():
    # This checks docstring presence only; ownership accuracy needs code review.
    root = _package_dir()
    undocumented = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if _module_summary(path.read_text(), path.stem) is None
    )
    assert undocumented == [], f"modules without a non-empty docstring: {undocumented}"


def test_docstring_guard_catches_missing_and_empty():
    assert _module_summary("from __future__ import annotations\n", "x") is None
    assert _module_summary('"""   """\n', "x") is None
    assert _module_summary('"""Owns the thing.\n\nDetail.\n"""\n', "x") == "Owns the thing."


def test_retired_compatibility_paths_stay_removed():
    # Each name is imported from its defining module; the old spellings must not return.
    import agent_tools.adapters
    import agent_tools.adapters.base
    import agent_tools.adapters.registry
    import agent_tools.configs
    import agent_tools.experiment_io
    import agent_tools.hparam_runtime
    import agent_tools.managed_scheduler
    import agent_tools.plan_context
    import agent_tools.plan_hparam
    import agent_tools.recipes
    import agent_tools.run_artifacts

    assert not (_package_dir() / "index_csv.py").exists()
    assert not hasattr(agent_tools.configs, "sleep2stat_config_summary")
    assert not hasattr(agent_tools.plan_context, "load_config_summary_for_recipe")
    for name in ("get_adapter", "all_adapters", "SUPPORTED_TASKS", "composite_adapter", "TaskAdapter"):
        assert not hasattr(agent_tools.adapters, name), name
    assert not hasattr(agent_tools.recipes, "recipe_name")
    assert not hasattr(agent_tools.experiment_io, "SSH_TIMEOUT_SECONDS")
    assert not hasattr(agent_tools.managed_scheduler, "managed_run_lock")
    for name in (
        "_EXECUTION_PREFLIGHT_TIMEOUT_SECONDS",
        "EXECUTION_SNAPSHOT_NAME",
        "PlannedArgv",
        "ExecutionSnapshot",
        "ExecutionSnapshotResult",
        "validated_execution_snapshot",
        "write_execution_snapshot_file",
        "inspect_execution_target",
        "run_execution_command",
    ):
        assert not hasattr(agent_tools.managed_scheduler, name), name
    assert not hasattr(agent_tools.hparam_runtime, "EXECUTION_SNAPSHOT_NAME")
    assert not hasattr(agent_tools.run_artifacts, "find_run_manifest")
    assert not hasattr(agent_tools.run_artifacts, "checkpoint_names")
    for name in ("HparamRegistrationPreflightError", "preflight_hparam_plan", "commit_hparam_plan"):
        assert not hasattr(agent_tools.plan_hparam, name), name
    assert not hasattr(agent_tools.adapters.base, "PlanRegistrationPreflightError")
    for adapter in agent_tools.adapters.registry.all_adapters():
        for hook in ("precommit_plan", "commit_plan", "matches_config_data", "config_summary"):
            assert not hasattr(adapter, hook), (adapter.task, hook)


def _executed_imports(node: ast.AST, local: bool = False) -> Iterator[tuple[ast.Import | ast.ImportFrom, bool]]:
    """(import, is function-local) for each import under ``node`` that runs, i.e. outside ``if TYPE_CHECKING:``."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child, local
        elif isinstance(child, ast.If) and ast.unparse(child.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            # Only the guarded body never runs; an ``else:`` branch does.
            yield from _executed_imports(ast.Module(body=child.orelse, type_ignores=[]), local)
        else:
            yield from _executed_imports(child, local or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _import_edges(sources: dict[str, str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(all, function-local) intra-package import edges among the modules in ``sources``.

    ``sources`` maps each module's ``_normalize_import`` source name to its text;
    a package is spelled ``adapters.__init__`` there and is the node ``adapters``.
    Targets that are not modules in ``sources`` (imported symbols) drop out.
    """
    modules = {name.removesuffix(".__init__") for name in sources}
    edges: set[tuple[str, str]] = set()
    lazy: set[tuple[str, str]] = set()
    for name, text in sources.items():
        source = name.removesuffix(".__init__")
        for node, local in _executed_imports(ast.parse(text, name)):
            targets = {target for target in _normalize_import(node, name) if target in modules}
            # Importing ``adapters.x`` first runs ``adapters/__init__.py``. The package
            # itself and the modules inside it are already running it: no edge.
            targets |= {target.rpartition(".")[0] for target in targets}
            for target in targets & modules:
                if not f"{source}.".startswith(f"{target}."):
                    edges.add((source, target))
                    if local:
                        lazy.add((source, target))
    return edges, lazy


def _cycle_members(edges: set[tuple[str, str]]) -> set[str]:
    """Modules that reach themselves: the members of every strongly connected component larger than one.

    ``_import_edges`` emits no self-edge, so a module can only reach itself through another module.
    """
    successors: dict[str, set[str]] = {}
    for source, target in edges:
        successors.setdefault(source, set()).add(target)
    members: set[str] = set()
    for start in successors:
        reached: set[str] = set()
        frontier = [start]
        while frontier:
            for module in successors.get(frontier.pop(), set()) - reached:
                reached.add(module)
                frontier.append(module)
        if start in reached:
            members.add(start)
    return members


def _package_sources() -> dict[str, str]:
    root = _package_dir()
    return {
        str(path.relative_to(root).with_suffix("")).replace("/", "."): path.read_text() for path in root.rglob("*.py")
    }


def test_import_graph_is_acyclic():
    edges, _ = _import_edges(_package_sources())
    members = _cycle_members(edges)
    inside = sorted((source, target) for source, target in edges if source in members and target in members)
    assert not members, (
        f"import cycle among {sorted(members)} through the edges {inside}; "
        "point the import down the layers instead of closing the cycle"
    )


def test_no_function_local_intra_package_imports():
    _, lazy = _import_edges(_package_sources())
    assert not lazy, f"function-local intra-package import edges (hoist them to module level): {sorted(lazy)}"


def test_import_graph_guard_catches_synthetic_cycle():
    # ``a`` imports ``pkg.b`` at top level, which first runs ``pkg/__init__``; ``pkg.b``
    # imports ``a`` back only inside a function: a lazy 2-cycle. The TYPE_CHECKING
    # import of ``c`` never runs but the ``else:`` import of ``e`` does, and ``pkg.b``
    # importing its sibling ``pkg.d`` adds no edge to the ``pkg`` __init__ that is
    # already running.
    edges, lazy = _import_edges(
        {
            "a": (
                "from typing import TYPE_CHECKING\nfrom .pkg.b import f\n"
                "if TYPE_CHECKING:\n    from .c import C\nelse:\n    from .e import E\n"
            ),
            "c": "",
            "e": "",
            "pkg.__init__": "",
            "pkg.b": "from .d import g\n\n\ndef f():\n    from ..a import h\n",
            "pkg.d": "",
        }
    )
    assert edges == {("a", "e"), ("a", "pkg"), ("a", "pkg.b"), ("pkg.b", "a"), ("pkg.b", "pkg.d")}
    assert lazy == {("pkg.b", "a")}
    assert _cycle_members(edges) == {"a", "pkg.b"}
