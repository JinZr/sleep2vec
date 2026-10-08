#!/usr/bin/env python3
"""Type-check agent_tools once, without global or per-module error suppression.

Scope and settings live in pyproject.toml. Both utils/style_check.sh and CI use
this entrypoint; neither requires Git history or a base revision.

mypy accepts ``Any`` wherever it is written, so before mypy runs this also holds
the explicit ``Any`` spellings in the same scope to ``ANY_LEDGER``. The counts
must match exactly: growth fails, and so does an unrecorded shrink, whose slack
would let a later change grow back unseen. Only review stops someone raising a
number, as with the ledgers in ``utils/complexity_check.py``. The check counts
spellings, not precision: an alias counts once however often it is used, and
bare generics, ``object`` and casts do not count at all, so AGENTS.md rules
those out as ways to lower it.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    # mypy supplies tomli on Python versions without stdlib tomllib.
    import tomli as tomllib

PYPROJECT = Path("pyproject.toml")
#: Every ``Any`` or ``typing.Any`` reference in the ``[tool.mypy]`` scope
#: (imports are not references), and the ``dict[str, Any]`` subscripts among
#: them. Lower a number in the commit that removes the spellings; do not
#: duplicate these counts in documentation.
ANY_LEDGER = {"Any": 920, "dict[str, Any]": 559}


def is_any(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "Any") or (isinstance(node, ast.Attribute) and node.attr == "Any")


def any_counts(scope: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(ANY_LEDGER, 0)
    for path in sorted(module for entry in scope for module in Path(entry).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if is_any(node):
                counts["Any"] += 1
            elif (
                isinstance(node, ast.Subscript)
                and ast.unparse(node.value) == "dict"
                and isinstance(node.slice, ast.Tuple)
                and ast.unparse(node.slice.elts[0]) == "str"
                and is_any(node.slice.elts[-1])
            ):
                counts["dict[str, Any]"] += 1
    return counts


def check_any_ledger(scope: list[str]) -> bool:
    counts = any_counts(scope)
    for spelling, recorded in ANY_LEDGER.items():
        found = counts[spelling]
        if found > recorded:
            advice = "The ledger only shrinks; type the new code instead."
        elif found < recorded:
            advice = f"Lower ANY_LEDGER[{spelling!r}] to {found} in this commit."
        else:
            continue
        print(f"{spelling}: found {found}, ledger {recorded}. {advice}")
    if counts == ANY_LEDGER:
        print(f"Any ledger: {counts['Any']} Any, {counts['dict[str, Any]']} dict[str, Any], as recorded.")
    return counts == ANY_LEDGER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    mypy = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]
    if mypy.get("ignore_errors") or any(override.get("ignore_errors") for override in mypy.get("overrides", [])):
        print("ignore_errors is not allowed in the mypy configuration; fix type errors instead of suppressing them.")
        return 1

    print("== Any ledger ==")
    # mypy runs either way, so one pass reports both.
    ledger_ok = check_any_ledger(mypy["files"])
    print("== mypy ==", flush=True)
    returncode = subprocess.run([sys.executable, "-m", "mypy", "--config-file", str(PYPROJECT)]).returncode
    return returncode if ledger_ok else returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
