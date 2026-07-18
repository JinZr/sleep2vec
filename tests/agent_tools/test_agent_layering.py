"""Layering guard: kernel and mixed modules must not grow new domain imports.

Complements test_agent_task_adapters.py (which forbids task-name string
constants in kernel modules). Here we scan import statements: any module in
KERNEL_MODULES or MIXED_MODULES that imports a DOMAIN_MODULES module is a
reverse edge, allowed only if it is in KNOWN_DOMAIN_IMPORT_EXEMPTIONS.

Pure-kernel modules carry no exemptions, so they must stay domain-free. The
mixed bridges carry exactly the eight grandfathered edges; a ninth fails.

Reads only ast + agent_tools.layering (a zero-dependency data module), so it
runs in the domain-free CI environment.
"""

import ast
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
    # exemption is actually exercised, e.g. the index_csv shim).
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
