"""Package structure guards for agent_tools.

The intra-package import graph, counting the implicit package ``__init__``
edges, must be acyclic, and no module may import another package module inside
a function. Imports under ``if TYPE_CHECKING:`` never run, so they do not count.
Every module carries an ownership docstring, and retired compatibility paths
stay removed.

Reads only ast and the package sources, so it runs in the domain-free CI
environment.
"""

import ast
from collections.abc import Iterator
from pathlib import Path


def _package_dir() -> Path:
    import agent_tools

    return Path(agent_tools.__file__).parent


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
    so an edge cannot be hidden by spelling the import absolutely:

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
        "move the shared code to the lower module instead of closing the cycle"
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
