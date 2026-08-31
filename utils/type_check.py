#!/usr/bin/env python3
"""Type-check agent_tools and hold its mypy debt ledger to a shrink-only contract.

Scope and settings live in ``[tool.mypy]`` in ``pyproject.toml``. The ledger is
the ``ignore_errors`` block under ``[[tool.mypy.overrides]]``: modules
grandfathered out of checking while their pre-existing errors are worked off.
Running mypy alone would not keep that list honest, so this runs three checks:

1. **mypy** over the configured scope.
2. **No ledger additions.** A contributor could introduce a type error and
   silence it by appending its module to the ledger -- mypy then passes, and a
   staleness check alone sees a perfectly live entry. Caught by diffing the
   ledger against a base revision's.
3. **No stale ledger entries.** A module whose errors have been fixed must
   leave the ledger, or the list decays into a permanent, meaningless block.
   Caught by re-running mypy with the ledger neutralized.

Both ``utils/style_check.sh`` and the ``style_check`` workflow call this, so a
clean local run means a clean CI run. Check 2 needs a revision to diff against:
CI passes the pull request's base sha via ``LEDGER_BASE_SHA``, and locally it is
skipped unless you pass ``--base <rev>``.

This mirrors ``test_agent_layering.test_every_exemption_is_live``, which keeps
the domain-import exemption set honest the same way.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import tomli  # a mypy dependency on Python < 3.11

PYPROJECT = Path("pyproject.toml")
ERROR_LINE = re.compile(r"^(agent_tools/[\w/]+)\.py:\d+: error:")
IGNORE_ERRORS_TOGGLE = re.compile(r"^ignore_errors = true$", re.M)


def run_mypy(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "mypy", *arguments], capture_output=True, text=True)


def ledger_of(document: str) -> set[str] | None:
    """Grandfathered modules, or None if the revision has no mypy config."""
    mypy = tomli.loads(document).get("tool", {}).get("mypy")
    if mypy is None:
        return None
    return {
        module
        for override in mypy.get("overrides", [])
        if override.get("ignore_errors")
        for module in override["module"]
    }


def base_document(revision: str) -> str | None:
    """``pyproject.toml`` as of ``revision``, or None if it cannot be read."""
    if not revision or set(revision) == {"0"}:
        print("No base revision given; skipping the additions check.")
        return None
    show = subprocess.run(["git", "show", f"{revision}:{PYPROJECT}"], capture_output=True, text=True)
    if show.returncode != 0:
        print(f"Could not read {PYPROJECT} at {revision}; skipping the additions check.")
        return None
    return show.stdout


def added_entries(ledger: set[str], revision: str) -> list[str]:
    document = base_document(revision)
    if document is None:
        return []
    previous = ledger_of(document)
    if previous is None:
        print("Base revision has no [tool.mypy]; this change introduces the ledger.")
        return []
    added = sorted(ledger - previous)
    if not added:
        print(f"mypy ledger: no additions against {revision[:12]}.")
    return added


def modules_with_errors(document: str) -> set[str]:
    """Modules mypy still reports errors for once the ledger is neutralized."""
    neutralized, count = IGNORE_ERRORS_TOGGLE.subn("ignore_errors = false", document)
    if count != 1:
        sys.exit(f"Expected exactly one ignore_errors toggle in {PYPROJECT}, found {count}.")

    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "pyproject.toml"
        config.write_text(neutralized, encoding="utf-8")
        result = run_mypy("--config-file", str(config), "--no-incremental", "agent_tools")

    failing = set()
    for line in result.stdout.splitlines():
        match = ERROR_LINE.match(line)
        if match:
            parts = match[1].split("/")
            failing.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    if not failing:
        # A neutralized run that reports nothing means the check broke, not that
        # the debt is gone -- fail loudly rather than passing vacuously.
        print(result.stdout or result.stderr, file=sys.stderr)
        sys.exit("Neutralized mypy run reported no errors at all; the ledger check is not working.")
    return failing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=os.environ.get("LEDGER_BASE_SHA", ""),
        help="Revision to diff the ledger against; without it the additions check is skipped.",
    )
    args = parser.parse_args(argv)

    print("== mypy ==")
    result = run_mypy()
    print(result.stdout or result.stderr, end="")
    if result.returncode != 0:
        return result.returncode

    document = PYPROJECT.read_text(encoding="utf-8")
    ledger = ledger_of(document)
    if ledger is None:
        sys.exit(f"{PYPROJECT} has no [tool.mypy] section.")
    if not ledger:
        print("\n== ledger ==\nmypy ledger is empty; delete this check along with the last entry.")
        return 0

    print("\n== ledger ==")
    added = added_entries(ledger, args.base.strip())
    if added:
        print("The mypy ledger may only shrink, and these entries are new:")
        for module in added:
            print(f'    "{module}",')
        print()
        print("Fix the module's type errors instead of grandfathering it.")
        return 1

    # Bind the result: inlining the call into the comprehension's condition
    # would re-run mypy once per ledger entry.
    failing = modules_with_errors(document)
    stale = sorted(module for module in ledger if module not in failing)
    if stale:
        print("Stale mypy ledger entries -- these modules type-check cleanly now.")
        print("Delete their lines from the [[tool.mypy.overrides]] block in pyproject.toml:")
        for module in stale:
            print(f'    "{module}",')
        return 1

    print(f"mypy ledger: {len(ledger)} grandfathered modules, none stale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
